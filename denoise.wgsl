// Ignis SVGF — Dual-Signal Variance-Guided Denoiser
//
// Hybrid denoiser combining SVGF (Schied 2017) with NRD/ReBLUR/ReLAX techniques:
// - À-trous wavelet filter (5×5 B3-spline, σ_n=128, σ_l=4.0×√var)
// - Variance filtering propagated through passes (SVGF §4.2)
// - Separated diffuse irradiance + specular radiance (ReLAX, NRD)
// - Roughness-dependent specular sigma (tight for glossy, wide for rough)
// - Per-pixel history length for adaptive temporal/spatial balance (ReBLUR)
// - Hit distance-modulated blur radius (ReBLUR hitT)
// - Pre-blur pass with anti-firefly 3σ percentile clamp (HPG 2025)
// - Depth-relative gradient floor for distance-invariant filtering

struct Params {
  resolution: vec2f,
  step_size: f32,
  frames_still: f32,
  tonemap_mode: u32,
  exposure: f32,
  saturation: f32,
  contrast: f32,
  // Legacy camera data kept to preserve the composite uniform layout.
  cam_pos: vec3f,
  _pad_cam0: f32,
  cam_forward: vec3f,
  fov_factor: f32,
  cam_right: vec3f,
  aspect: f32,
  cam_up: vec3f,
  _pad_cam1: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var in_color: texture_2d<f32>;       // diffuse input
@group(0) @binding(2) var out_color: texture_storage_2d<rgba16float, write>; // diffuse output
@group(0) @binding(3) var gbuf_nd: texture_2d<f32>;        // normal.xyz + depth
@group(0) @binding(4) var in_spec: texture_2d<f32>;        // specular input
@group(0) @binding(5) var out_spec: texture_storage_2d<rgba16float, write>; // specular output
@group(0) @binding(6) var albedo_tex: texture_2d<f32>;     // albedo.rgb + roughness.a

// Material struct (same layout as pathtracer.wgsl)
struct Material {
  d0: vec4f,
  d1: vec4f,
  d2: vec4f,
  d3: vec4f,
  d4: vec4f,
  d5: vec4f,
  d6: vec4f,
  d7: vec4f,
  d8: vec4f,
  d9: vec4f,
};

fn luma(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }

// 5x5 B3-spline kernel weights (used by atrous_sm first pass)
const KW = array<f32, 25>(
  1.0/256.0,  4.0/256.0,  6.0/256.0,  4.0/256.0, 1.0/256.0,
  4.0/256.0, 16.0/256.0, 24.0/256.0, 16.0/256.0, 4.0/256.0,
  6.0/256.0, 24.0/256.0, 36.0/256.0, 24.0/256.0, 6.0/256.0,
  4.0/256.0, 16.0/256.0, 24.0/256.0, 16.0/256.0, 4.0/256.0,
  1.0/256.0,  4.0/256.0,  6.0/256.0,  4.0/256.0, 1.0/256.0,
);

// ReBLUR-style Poisson disk (8 taps, z = radius for Gaussian weight)
const POISSON8 = array<vec3f, 8>(
  vec3f(-1.00,  0.20, 0.15),
  vec3f( 0.38, -0.85, 0.25),
  vec3f( 0.94,  0.34, 0.35),
  vec3f(-0.47,  0.81, 0.45),
  vec3f(-0.74, -0.64, 0.55),
  vec3f( 0.72, -0.58, 0.65),
  vec3f( 0.15,  0.98, 0.75),
  vec3f(-0.28, -0.36, 0.85),
);

// ============================================================
// ReBLUR-style adaptive Poisson blur (replaces multi-pass à-trous)
// Single pass with per-pixel adaptive radius based on accumulation history.
// 8 Poisson taps with per-frame rotation — no multi-pass bleeding.
// Geometry-only weights (normal + plane distance) — cannot cross shadow boundaries.
// ============================================================
@compute @workgroup_size(16, 16)
fn atrous(@builtin(global_invocation_id) gid: vec3u) {
  let px = vec2i(gid.xy);
  let sz = vec2i(params.resolution);
  if px.x >= sz.x || px.y >= sz.y { return; }

  // ReBLUR: 3 spatial passes with different radius scales
  // Pass 1 (step=2): Blur — main spatial filter, radiusScale=1.0
  // Pass 2 (step=4): PostBlur — aggressive cleanup, radiusScale=2.0, fractionScale=0.5
  // Pass 3+ (step>=8): passthrough
  if params.step_size >= 7.5 {
    textureStore(out_color, px, textureLoad(in_color, px, 0));
    textureStore(out_spec, px, textureLoad(in_spec, px, 0));
    return;
  }
  let radiusScale = select(1.0, 2.0, params.step_size > 3.5);   // Blur=1x, PostBlur=2x
  let fractionScale = select(1.0, 0.5, params.step_size > 3.5); // PostBlur tightens weights

  let cnd = textureLoad(gbuf_nd, px, 0);
  let cn = cnd.xyz;
  let cz = cnd.w;
  let diff_sample = textureLoad(in_color, px, 0);
  let cc = diff_sample.rgb;
  let cs = textureLoad(in_spec, px, 0).rgb;
  let hit_dist = textureLoad(in_spec, px, 0).a;
  let roughness = textureLoad(albedo_tex, px, 0).a;

  // === Adaptive blur radius (ReBLUR core idea) ===
  let history_len = diff_sample.a;
  let accumSpeed = min(history_len, 64.0);
  let nonLinearAccumSpeed = 1.0 / (1.0 + accumSpeed);

  // Blur radius: large when new, small when converged
  // ReBLUR: blurRadius = radiusScale * sqrt(hitDistFactor * nonLinearAccumSpeed) * maxBlurRadius
  let maxBlurRadius = 20.0;
  let hitFactor = clamp(hit_dist, 0.1, 1.0);
  let smc = sqrt(max(roughness, 0.01)); // specMagicCurve approximation
  let diffRadius = radiusScale * maxBlurRadius * sqrt(hitFactor * nonLinearAccumSpeed);
  let specRadius = radiusScale * maxBlurRadius * sqrt(hitFactor * nonLinearAccumSpeed) * smc;

  // Minimum radius
  let diff_blur_r = max(diffRadius, 0.5);
  let spec_blur_r = max(specRadius, 0.5);

  // === Per-frame Poisson rotation (reduces banding) ===
  let frame_angle = f32(u32(params.frames_still * 100.0) % 256u) * (6.2832 / 256.0);
  let rot_cos = cos(frame_angle);
  let rot_sin = sin(frame_angle);

  // Depth weight normalization: relative to depth gradient
  let zr = textureLoad(gbuf_nd, clamp(px + vec2i(1, 0), vec2i(0), sz - 1), 0).w;
  let zu = textureLoad(gbuf_nd, clamp(px + vec2i(0, 1), vec2i(0), sz - 1), 0).w;
  let gz = max(max(abs(zr - cz), abs(zu - cz)), cz * 0.002);

  // === 8-tap Poisson blur ===
  var d_sum = cc;
  var s_sum = cs;
  var d_wsum = 1.0;
  var s_wsum = 1.0;

  for (var i = 0u; i < 8u; i++) {
    let tap = POISSON8[i];

    // Rotate and scale offset by blur radius
    let ox = tap.x * rot_cos - tap.y * rot_sin;
    let oy = tap.x * rot_sin + tap.y * rot_cos;
    let d_offset = vec2i(vec2f(ox, oy) * diff_blur_r + 0.5);
    let s_offset = vec2i(vec2f(ox, oy) * spec_blur_r + 0.5);

    // Gaussian weight from Poisson radius
    let gauss_w = exp(-0.66 * tap.z * tap.z);

    // --- Diffuse tap ---
    let dp = clamp(px + d_offset, vec2i(0), sz - 1);
    let dnd = textureLoad(gbuf_nd, dp, 0);
    let d_col = textureLoad(in_color, dp, 0).rgb;

    // Normal weight: tighter for PostBlur (fractionScale=0.5 → exponent doubles)
    let normal_exp = 32.0 / fractionScale;
    let d_wn = pow(max(dot(cn, dnd.xyz), 0.0), normal_exp);
    // Depth weight: exponential falloff based on depth gradient
    let d_dz = abs(cz - dnd.w);
    let d_wz = exp(-d_dz / (gz * diff_blur_r + 1e-3));
    let d_w = gauss_w * d_wn * d_wz;

    d_sum += d_col * d_w;
    d_wsum += d_w;

    // --- Specular tap (separate radius) ---
    let sp = clamp(px + s_offset, vec2i(0), sz - 1);
    let snd = textureLoad(gbuf_nd, sp, 0);
    let s_col = textureLoad(in_spec, sp, 0).rgb;

    let s_wn = pow(max(dot(cn, snd.xyz), 0.0), 64.0 / fractionScale);
    let s_dz = abs(cz - snd.w);
    let s_wz = exp(-s_dz / (gz * spec_blur_r + 1e-3));
    let s_w = gauss_w * s_wn * s_wz;

    s_sum += s_col * s_w;
    s_wsum += s_w;
  }

  textureStore(out_color, px, vec4f(d_sum / d_wsum, history_len));
  textureStore(out_spec, px, vec4f(s_sum / s_wsum, hit_dist));
}

// === TILED ATROUS (step=1 only): shared memory reduces 99 textureLoad → ~5 per thread ===
// 16x16 workgroup + ±2 halo = 20x20 tile cached in workgroup memory
const TILE: i32 = 16;
const HALO: i32 = 2;
const PADDED: i32 = 20; // TILE + 2*HALO

var<workgroup> sm_nd: array<vec4f, 400>;   // normal.xyz + depth
var<workgroup> sm_diff: array<vec4f, 400>; // diffuse.rgb + variance.a
var<workgroup> sm_spec: array<vec4f, 400>; // specular.rgb + hit_dist.a

@compute @workgroup_size(16, 16)
fn atrous_sm(@builtin(global_invocation_id) gid: vec3u,
             @builtin(local_invocation_id) lid: vec3u,
             @builtin(workgroup_id) wid: vec3u) {
  let sz = vec2i(params.resolution);
  let tile_origin = vec2i(wid.xy) * TILE - HALO;

  // Cooperative tile load: 400 texels / 256 threads ≈ 1.6 loads per thread
  let flat_id = lid.y * 16u + lid.x;
  for (var t = flat_id; t < 400u; t += 256u) {
    let ty = i32(t / 20u);
    let tx = i32(t % 20u);
    let coord = clamp(tile_origin + vec2i(tx, ty), vec2i(0), sz - 1);
    sm_nd[t] = textureLoad(gbuf_nd, coord, 0);
    sm_diff[t] = textureLoad(in_color, coord, 0);
    sm_spec[t] = textureLoad(in_spec, coord, 0);
  }
  workgroupBarrier();

  let px = vec2i(gid.xy);
  if px.x >= sz.x || px.y >= sz.y { return; }

  let lx = i32(lid.x);
  let ly = i32(lid.y);
  let ci = u32((ly + HALO) * PADDED + (lx + HALO));

  // === ReBLUR-style HistoryFix + Bilateral (combined single pass) ===
  // For pixels with short history: borrow from neighbors with LONGER history (ReBLUR HistoryFix)
  // For all pixels: 5x5 bilateral with geometry weights (cleanup pass)
  let cnd = sm_nd[ci];
  let cn = cnd.xyz;
  let cz = cnd.w;
  let cc = sm_diff[ci].rgb;
  let cs = sm_spec[ci].rgb;
  let center_history = sm_diff[ci].a; // history_len from temporal

  // Depth gradient from tile
  let zr_i = u32((ly + HALO) * PADDED + min(lx + HALO + 1, PADDED - 1));
  let zu_i = u32(min(ly + HALO + 1, PADDED - 1) * PADDED + (lx + HALO));
  let gz = max(max(abs(sm_nd[zr_i].w - cz), abs(sm_nd[zu_i].w - cz)), cz * 0.002);

  // HistoryFix: for pixels with short history, weight neighbors by THEIR history length
  // This fills disoccluded pixels with converged neighbor data (ReBLUR HistoryFix concept)
  let historyFixThreshold = 8.0; // frames before HistoryFix kicks in
  let needsFix = center_history < historyFixThreshold;

  // 5x5 bilateral with optional HistoryFix weighting
  var d_sum = cc * (1.0 + center_history); // center weighted by its own history
  var s_sum = cs * (1.0 + center_history);
  var w_sum = 1.0 + center_history;

  for (var dy = -2; dy <= 2; dy++) {
    for (var dx = -2; dx <= 2; dx++) {
      if dx == 0 && dy == 0 { continue; }
      let si = u32((ly + HALO + dy) * PADDED + (lx + HALO + dx));
      let snd = sm_nd[si];
      let sd = sm_diff[si].rgb;
      let ss = sm_spec[si].rgb;
      let s_history = sm_diff[si].a;

      // Geometry weights (normal + depth plane)
      let wn = pow(max(dot(cn, snd.xyz), 0.0), 64.0);
      let dz = abs(cz - snd.w);
      let wz = exp(-dz / (gz + 1e-3));
      let geom = KW[u32((dy + 2) * 5 + (dx + 2))] * wn * wz;

      if geom < 0.001 { continue; }

      // HistoryFix: if this pixel needs fixing, boost weight of neighbors with more history
      // Neighbor with history=30 contributes 30x more than neighbor with history=1
      var w = geom;
      if needsFix {
        w *= (1.0 + s_history); // ReBLUR: w *= (1.0 + frameNum_sample)
      }

      d_sum += sd * w;
      s_sum += ss * w;
      w_sum += w;
    }
  }

  textureStore(out_color, px, vec4f(d_sum / max(w_sum, 1e-6), center_history));
  textureStore(out_spec, px, vec4f(s_sum / max(w_sum, 1e-6), sm_spec[ci].a));
}

// === PRE-BLUR: anti-firefly percentile clamp + lightweight 3x3 bilateral ===
// Stabilizes temporal AABB and eliminates bright speckles adaptively.
@compute @workgroup_size(16, 16)
fn preblur(@builtin(global_invocation_id) gid: vec3u) {
  let px = vec2i(gid.xy);
  let sz = vec2i(params.resolution);
  if px.x >= sz.x || px.y >= sz.y { return; }

  let cnd = textureLoad(gbuf_nd, px, 0);
  let cn = cnd.xyz;
  let cz = cnd.w;
  let cc_raw = textureLoad(in_color, px, 0).rgb;
  let cs_raw = textureLoad(in_spec, px, 0).rgb;

  // === Anti-firefly: 3σ percentile clamp over 3×3 neighborhood ===
  // Only clips extreme outliers; preserves legitimate bright samples (sun highlights, etc.)
  var d_m1 = 0.0; var d_m2 = 0.0;
  var s_m1 = 0.0; var s_m2 = 0.0;
  for (var fy = -1; fy <= 1; fy++) {
    for (var fx = -1; fx <= 1; fx++) {
      let fp = clamp(px + vec2i(fx, fy), vec2i(0), sz - 1);
      let fl = luma(textureLoad(in_color, fp, 0).rgb);
      d_m1 += fl; d_m2 += fl * fl;
      let fsl = luma(textureLoad(in_spec, fp, 0).rgb);
      s_m1 += fsl; s_m2 += fsl * fsl;
    }
  }
  let d_mean = d_m1 / 9.0;
  let d_std = sqrt(max(d_m2 / 9.0 - d_mean * d_mean, 0.0));
  let d_max_lum = d_mean + 1.5 * d_std + 0.05;

  let s_mean = s_m1 / 9.0;
  let s_std = sqrt(max(s_m2 / 9.0 - s_mean * s_mean, 0.0));
  let s_max_lum = s_mean + 1.5 * s_std + 0.05;

  // Anti-firefly clamp
  let cl_raw = luma(cc_raw);
  var cc = cc_raw;
  if cl_raw > d_max_lum && cl_raw > 0.01 { cc = cc_raw * (d_max_lum / cl_raw); }
  let csl_raw = luma(cs_raw);
  var cs = cs_raw;
  if csl_raw > s_max_lum && csl_raw > 0.01 { cs = cs_raw * (s_max_lum / csl_raw); }

  // === ReBLUR PrePass: 8-tap Poisson blur (radius=15px) ===
  // Large spatial filter BEFORE temporal → gives temporal a cleaner base on frame 1
  // Uses geometry-only weights (normal + depth plane) — no shadow bleeding
  let frustumSize = max(cz * 0.01, 0.01);
  let prepassRadius = 15.0;
  let pre_angle = f32(u32(params.frames_still * 73.0) % 256u) * (6.2832 / 256.0);
  let pre_cos = cos(pre_angle);
  let pre_sin = sin(pre_angle);

  var d_sum = cc;
  var s_sum = cs;
  var w_sum = 1.0;

  for (var i = 0u; i < 8u; i++) {
    let tap = POISSON8[i];
    let ox = tap.x * pre_cos - tap.y * pre_sin;
    let oy = tap.x * pre_sin + tap.y * pre_cos;
    let offset = vec2i(vec2f(ox, oy) * prepassRadius + 0.5);
    let sp = clamp(px + offset, vec2i(0), sz - 1);

    let snd = textureLoad(gbuf_nd, sp, 0);
    let wn = pow(max(dot(cn, snd.xyz), 0.0), 16.0);
    let planeDist = abs(cz - snd.w) / max(frustumSize, 0.01);
    let wz = exp(-planeDist * planeDist);
    let gauss = exp(-0.66 * tap.z * tap.z);
    let w = gauss * wn * wz;

    if w > 0.01 {
      // Also clamp each neighbor (anti-firefly propagation)
      var sd = textureLoad(in_color, sp, 0).rgb;
      let sdl = luma(sd);
      if sdl > d_max_lum && sdl > 0.01 { sd *= d_max_lum / sdl; }
      var ss = textureLoad(in_spec, sp, 0).rgb;
      let ssl = luma(ss);
      if ssl > s_max_lum && ssl > 0.01 { ss *= s_max_lum / ssl; }

      d_sum += sd * w;
      s_sum += ss * w;
      w_sum += w;
    }
  }

  let center_hit_dist = textureLoad(in_spec, px, 0).a;
  textureStore(out_color, px, vec4f(d_sum / w_sum, 1.0));
  textureStore(out_spec, px, vec4f(s_sum / w_sum, center_hit_dist));
}

// === TILED PRE-BLUR: shared memory version (18x18 tile, ±1 halo) ===
// Reuses sm_nd/sm_diff/sm_spec from atrous_sm (400 entries > 324 needed)
const PB_HALO: i32 = 1;
const PB_PADDED: i32 = 18; // TILE + 2*PB_HALO
const PB_TILES: u32 = 324u; // 18*18

@compute @workgroup_size(16, 16)
fn preblur_sm(@builtin(global_invocation_id) gid: vec3u,
              @builtin(local_invocation_id) lid: vec3u,
              @builtin(workgroup_id) wid: vec3u) {
  let sz = vec2i(params.resolution);
  let tile_origin = vec2i(wid.xy) * TILE - PB_HALO;

  // Cooperative tile load: 324 texels / 256 threads ≈ 1.27 loads per thread
  let flat_id = lid.y * 16u + lid.x;
  for (var t = flat_id; t < PB_TILES; t += 256u) {
    let ty = i32(t / 18u);
    let tx = i32(t % 18u);
    let coord = clamp(tile_origin + vec2i(tx, ty), vec2i(0), sz - 1);
    sm_nd[t] = textureLoad(gbuf_nd, coord, 0);
    sm_diff[t] = textureLoad(in_color, coord, 0);
    sm_spec[t] = textureLoad(in_spec, coord, 0);
  }
  workgroupBarrier();

  let px = vec2i(gid.xy);
  if px.x >= sz.x || px.y >= sz.y { return; }

  let lx = i32(lid.x);
  let ly = i32(lid.y);
  let ci = u32((ly + PB_HALO) * PB_PADDED + (lx + PB_HALO));

  let cn = sm_nd[ci].xyz;
  let cz = sm_nd[ci].w;
  let cc_raw = sm_diff[ci].rgb;
  let cs_raw = sm_spec[ci].rgb;

  // Anti-firefly: 3σ percentile clamp from 3×3 tile (preserves legitimate bright samples)
  var d_m1 = 0.0; var d_m2 = 0.0;
  var s_m1 = 0.0; var s_m2 = 0.0;
  for (var fy = -1; fy <= 1; fy++) {
    for (var fx = -1; fx <= 1; fx++) {
      let fi = u32((ly + PB_HALO + fy) * PB_PADDED + (lx + PB_HALO + fx));
      let fl = luma(sm_diff[fi].rgb);
      d_m1 += fl; d_m2 += fl * fl;
      let fsl = luma(sm_spec[fi].rgb);
      s_m1 += fsl; s_m2 += fsl * fsl;
    }
  }
  let d_mean = d_m1 / 9.0;
  let d_std = sqrt(max(d_m2 / 9.0 - d_mean * d_mean, 0.0));
  let d_max_lum = d_mean + 1.5 * d_std + 0.05;
  let s_mean = s_m1 / 9.0;
  let s_std = sqrt(max(s_m2 / 9.0 - s_mean * s_mean, 0.0));
  let s_max_lum = s_mean + 1.5 * s_std + 0.05;

  // Anti-firefly clamp
  let cl_raw = luma(cc_raw);
  var cc = cc_raw;
  if cl_raw > d_max_lum && cl_raw > 0.01 { cc = cc_raw * (d_max_lum / cl_raw); }
  let csl_raw = luma(cs_raw);
  var cs = cs_raw;
  if csl_raw > s_max_lum && csl_raw > 0.01 { cs = cs_raw * (s_max_lum / csl_raw); }

  // === ReBLUR PrePass: 8-tap Poisson blur ===
  // Can't use shared memory here (radius=15 exceeds ±1 halo), so textureLoad directly
  let frustumSize_pb = max(cz * 0.01, 0.01);
  let pre_angle_sm = f32(u32(params.frames_still * 73.0) % 256u) * (6.2832 / 256.0);
  let pre_cos_sm = cos(pre_angle_sm);
  let pre_sin_sm = sin(pre_angle_sm);

  var d_sum_pb = cc;
  var s_sum_pb = cs;
  var w_sum_pb = 1.0;

  for (var i = 0u; i < 8u; i++) {
    let tap = POISSON8[i];
    let ox = tap.x * pre_cos_sm - tap.y * pre_sin_sm;
    let oy = tap.x * pre_sin_sm + tap.y * pre_cos_sm;
    let offset = vec2i(vec2f(ox, oy) * 15.0 + 0.5);
    let sp = clamp(px + offset, vec2i(0), sz - 1);

    let snd = textureLoad(gbuf_nd, sp, 0);
    let wn = pow(max(dot(cn, snd.xyz), 0.0), 16.0);
    let pd = abs(cz - snd.w) / max(frustumSize_pb, 0.01);
    let wz = exp(-pd * pd);
    let gauss = exp(-0.66 * tap.z * tap.z);
    let w = gauss * wn * wz;

    if w > 0.01 {
      var sd = textureLoad(in_color, sp, 0).rgb;
      let sdl = luma(sd);
      if sdl > d_max_lum && sdl > 0.01 { sd *= d_max_lum / sdl; }
      var ss = textureLoad(in_spec, sp, 0).rgb;
      let ssl = luma(ss);
      if ssl > s_max_lum && ssl > 0.01 { ss *= s_max_lum / ssl; }

      d_sum_pb += sd * w;
      s_sum_pb += ss * w;
      w_sum_pb += w;
    }
  }

  textureStore(out_color, px, vec4f(d_sum_pb / w_sum_pb, 1.0));
  textureStore(out_spec, px, vec4f(s_sum_pb / w_sum_pb, sm_spec[ci].a));
}

// ============================================================
// COMPOSITE: remodulate + tonemap + color controls
// Pipeline: HDR → Exposure → Tonemap → Saturation → Gamma → Contrast → Dither
// ============================================================
@group(0) @binding(7) var composite_out: texture_storage_2d<rgba8unorm, write>;
// Legacy extra bindings kept for layout stability after moving glass into the path tracer.
@group(0) @binding(8) var gbuf_mat_uv: texture_2d<f32>;
@group(0) @binding(9) var gbuf_nd_comp: texture_2d<f32>;
@group(0) @binding(10) var prev_frame: texture_2d<f32>;
@group(0) @binding(11) var prev_sampler: sampler;
@group(0) @binding(12) var<storage, read> material_buf_comp: array<Material>;
@group(0) @binding(13) var<storage, read_write> exposure_buf: array<atomic<u32>>; // [0]=logLumSum, [1]=pixelCount

// --- Tonemap 0: AgX Punchy (Blender 4 / Troy Sobotka) ---
fn tonemap_agx(color_in: vec3f) -> vec3f {
  // sRGB → Rec.2020
  var c = mat3x3f(
    vec3f(0.6274, 0.0691, 0.0164), vec3f(0.3293, 0.9195, 0.0880), vec3f(0.0433, 0.0113, 0.8956)
  ) * color_in;
  // AgX inset
  c = mat3x3f(
    vec3f(0.856627, 0.137319, 0.111898), vec3f(0.095121, 0.761242, 0.076799), vec3f(0.048252, 0.101439, 0.811302)
  ) * c;
  c = max(c, vec3f(1e-10));
  c = clamp(log2(c), vec3f(-12.47393), vec3f(4.026069));
  c = (c + 12.47393) / (4.026069 + 12.47393);
  let x2 = c * c; let x4 = x2 * x2;
  c = 15.5*x4*x2 - 40.14*x4*c + 31.96*x4 - 6.868*x2*c + 0.4298*x2 + 0.1191*c - 0.00232;
  // Punchy: contrast + saturation boost
  c = pow(max(vec3f(0.0), c), vec3f(1.35));
  let l = dot(c, vec3f(0.2126, 0.7152, 0.0722));
  c = l + 1.4 * (c - l);
  // AgX outset + linearize
  c = mat3x3f(
    vec3f(1.1271, -0.1413, -0.1413), vec3f(-0.1106, 1.1578, -0.1106), vec3f(-0.0165, -0.0165, 1.2519)
  ) * c;
  c = pow(max(vec3f(0.0), c), vec3f(2.2));
  // Rec.2020 → sRGB
  c = mat3x3f(
    vec3f(1.6605, -0.1246, -0.0182), vec3f(-0.5876, 1.1329, -0.1006), vec3f(-0.0728, -0.0083, 1.1187)
  ) * c;
  return c; // returns linear
}

// --- Tonemap 1: ACES Narkowicz 2015 fit ---
fn tonemap_aces(v: vec3f) -> vec3f {
  let x = v * 0.6; // exposure bias
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
}

// --- Tonemap 2: Reinhard (luminance-preserving) ---
fn tonemap_reinhard(v: vec3f) -> vec3f {
  let l = dot(v, vec3f(0.2126, 0.7152, 0.0722));
  if l < 1e-6 { return v; }
  let l_new = l / (1.0 + l);
  return v * (l_new / l);
}

// --- Tonemap 3: Uncharted 2 / Hable filmic ---
fn uc2_partial(x: vec3f) -> vec3f {
  let A = 0.15; let B = 0.50; let C = 0.10; let D = 0.20; let E = 0.02; let F = 0.30;
  return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}
fn tonemap_uncharted2(v: vec3f) -> vec3f {
  let curr = uc2_partial(v * 2.0); // exposure bias = 2.0
  let white_scale = vec3f(1.0) / uc2_partial(vec3f(11.2));
  return curr * white_scale;
}

// --- Tonemap 4: Khronos PBR Neutral (May 2024, true-to-life color) ---
fn tonemap_pbr_neutral(color_in: vec3f) -> vec3f {
  let startCompression = 0.8 - 0.04;
  let desaturation = 0.15;
  let x = min(color_in.r, min(color_in.g, color_in.b));
  let offset = select(0.04, x - 6.25 * x * x, x < 0.08);
  var color = color_in - offset;
  let peak = max(color.r, max(color.g, color.b));
  if peak < startCompression { return color; }
  let d = 1.0 - startCompression;
  let newPeak = 1.0 - d * d / (peak + d - startCompression);
  color *= newPeak / peak;
  let g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
  return mix(color, vec3f(newPeak), g);
}

// --- Tonemap 5: Standard (clamp) ---
fn tonemap_standard(v: vec3f) -> vec3f {
  return clamp(v, vec3f(0.0), vec3f(1.0));
}

@compute @workgroup_size(16, 16)
fn composite(@builtin(global_invocation_id) gid: vec3u) {
  let px = vec2i(gid.xy);
  let sz = vec2i(params.resolution);
  if px.x >= sz.x || px.y >= sz.y { return; }

  let denoised_diff = textureLoad(in_color, px, 0).rgb;
  let denoised_spec = textureLoad(in_spec, px, 0).rgb;
  let albedo = textureLoad(albedo_tex, px, 0).rgb;

  // Remodulate: albedo × diffuse_irradiance + specular_radiance
  // When step_size < 0 (OIDN mode), diffuse IS the full denoised beauty pass — skip remodulation
  var hdr: vec3f;
  if params.step_size < 0.0 {
    hdr = denoised_diff; // OIDN: already combined beauty pass
  } else {
    hdr = albedo * denoised_diff + denoised_spec;
  }

  // Auto-exposure: accumulate log2-luminance, excluding extreme outliers (sun disc)
  let lum = dot(hdr, vec3f(0.2126, 0.7152, 0.0722));
  if lum > 0.001 && lum < 100.0 { // exclude sun disc (lum >> 100) from exposure calc
    let logLum = log2(lum) + 20.0;
    atomicAdd(&exposure_buf[0], u32(logLum * 16.0));
    atomicAdd(&exposure_buf[1], 1u);
  }

  // 1. Exposure (pre-tonemap)
  hdr *= params.exposure;

  // 2. Tonemap curve
  var ldr: vec3f;
  switch params.tonemap_mode {
    case 0u: { ldr = tonemap_agx(hdr); }       // AgX Punchy
    case 1u: { ldr = tonemap_aces(hdr); }       // ACES (Narkowicz)
    case 2u: { ldr = tonemap_reinhard(hdr); }   // Reinhard
    case 3u: { ldr = tonemap_uncharted2(hdr); } // Uncharted 2
    case 4u: { ldr = tonemap_pbr_neutral(hdr); } // Khronos PBR Neutral
    case 5u: { ldr = tonemap_standard(hdr); }   // Standard (clamp)
    default: { ldr = hdr; }                     // None (linear, debug)
  }
  ldr = clamp(ldr, vec3f(0.0), vec3f(1.0));

  // 3. Saturation (post-tonemap)
  let sat_luma = dot(ldr, vec3f(0.2126, 0.7152, 0.0722));
  ldr = clamp(mix(vec3f(sat_luma), ldr, params.saturation), vec3f(0.0), vec3f(1.0));

  // 4. Gamma encode (sRGB 2.2) → f16 for post-tonemap chain (output is 8-bit)
  var c = vec3h(pow(max(ldr, vec3f(0.0)), vec3f(1.0 / 2.2)));

  // 5. Contrast (post-gamma Hermite smoothstep, ignis-rt)
  if params.contrast > 0.01 {
    let curved = c * c * (3.0h - 2.0h * c);
    c = mix(c, curved, f16(params.contrast));
  }

  // 6. Dither (triangular ±0.5/255, prevents banding)
  let dither_hash = f16(fract(sin(dot(vec2f(f32(px.x), f32(px.y)), vec2f(12.9898, 78.233))) * 43758.5453));
  c += (dither_hash - 0.5h) / 255.0h;

  textureStore(composite_out, px, vec4f(vec3f(c), 1.0));
}

// ============================================================
// TEMPORAL STABILIZATION (ReBLUR-style anti-flicker)
// Operates on LUMINANCE only — clamps to neighborhood statistics.
// This prevents frame-to-frame flickering without color artifacts.
// Uses same layout as à-trous (bindings 0-6):
//   in_color(1) = denoised diffuse, out_color(2) = history diffuse
//   gbuf_nd(3) = temporal hdrTex (reused: .a = history_len)
//   in_spec(4) = denoised specular, out_spec(5) = history specular
//   albedo(6) = temporal specHdrTex (reused: .a = cam_z)
// ============================================================
@compute @workgroup_size(16, 16)
fn copy_to_history(@builtin(global_invocation_id) gid: vec3u) {
  let px = vec2i(gid.xy);
  let sz = vec2i(params.resolution);
  if px.x >= sz.x || px.y >= sz.y { return; }

  let history_len = textureLoad(gbuf_nd, px, 0).a;
  let cam_z = textureLoad(albedo_tex, px, 0).a;

  // === Temporal Stabilization: neighborhood luminance clamp ===
  // Compute luma statistics from 5x5 neighborhood of current denoised frame
  var diff_luma_m1 = 0.0;
  var diff_luma_m2 = 0.0;
  var spec_luma_m1 = 0.0;
  var spec_luma_m2 = 0.0;
  var count = 0.0;
  for (var dy = -2; dy <= 2; dy++) {
    for (var dx = -2; dx <= 2; dx++) {
      let sp = clamp(px + vec2i(dx, dy), vec2i(0), sz - 1);
      let dl = luma(textureLoad(in_color, sp, 0).rgb);
      let sl = luma(textureLoad(in_spec, sp, 0).rgb);
      diff_luma_m1 += dl;
      diff_luma_m2 += dl * dl;
      spec_luma_m1 += sl;
      spec_luma_m2 += sl * sl;
      count += 1.0;
    }
  }
  let d_mean = diff_luma_m1 / count;
  let d_sigma = sqrt(max(diff_luma_m2 / count - d_mean * d_mean, 0.0));
  let s_mean = spec_luma_m1 / count;
  let s_sigma = sqrt(max(spec_luma_m2 / count - s_mean * s_mean, 0.0));

  // Stabilization strength based on history (more history → more stabilization)
  let stabilization = clamp(history_len / 8.0, 0.0, 1.0);

  // Read current denoised
  var diff = textureLoad(in_color, px, 0).rgb;
  var spec = textureLoad(in_spec, px, 0).rgb;
  let diff_l = luma(diff);
  let spec_l = luma(spec);

  // Clamp luminance to neighborhood range (± sigma * scale)
  // ReBLUR uses sigma * (1 + 3*framerateScale*historyWeight) — we simplify
  let sigma_scale = 1.0 + 2.0 * stabilization;
  let clamped_dl = clamp(diff_l, d_mean - d_sigma * sigma_scale, d_mean + d_sigma * sigma_scale);
  let clamped_sl = clamp(spec_l, s_mean - s_sigma * sigma_scale, s_mean + s_sigma * sigma_scale);

  // Apply clamped luminance (preserves hue, only changes brightness)
  if diff_l > 0.001 { diff *= clamped_dl / diff_l; }
  if spec_l > 0.001 { spec *= clamped_sl / spec_l; }

  textureStore(out_color, px, vec4f(diff, history_len));
  textureStore(out_spec, px, vec4f(spec, cam_z));
}
