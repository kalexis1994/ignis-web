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

// 5x5 B3-spline kernel weights
const KW = array<f32, 25>(
  1.0/256.0,  4.0/256.0,  6.0/256.0,  4.0/256.0, 1.0/256.0,
  4.0/256.0, 16.0/256.0, 24.0/256.0, 16.0/256.0, 4.0/256.0,
  6.0/256.0, 24.0/256.0, 36.0/256.0, 24.0/256.0, 6.0/256.0,
  4.0/256.0, 16.0/256.0, 24.0/256.0, 16.0/256.0, 4.0/256.0,
  1.0/256.0,  4.0/256.0,  6.0/256.0,  4.0/256.0, 1.0/256.0,
);

@compute @workgroup_size(16, 16)
fn atrous(@builtin(global_invocation_id) gid: vec3u) {
  let px = vec2i(gid.xy);
  let sz = vec2i(params.resolution);
  if px.x >= sz.x || px.y >= sz.y { return; }

  let cnd = textureLoad(gbuf_nd, px, 0);
  let cn = cnd.xyz;
  let cz = cnd.w;
  let diff_sample = textureLoad(in_color, px, 0);
  let cc = diff_sample.rgb;
  let cl = luma(cc);
  let spec_sample = textureLoad(in_spec, px, 0);
  let cs = spec_sample.rgb;
  let csl = luma(cs);
  let hit_dist = spec_sample.a;

  // Roughness for specular filter width
  let roughness = textureLoad(albedo_tex, px, 0).a;

  // Hit distance modulates blur: contact shadows stay sharp, far GI gets smoothed
  let hit_factor = mix(0.5, 1.0, hit_dist);

  // Depth gradient with distance-relative floor (prevents filter rejection at far distances)
  let zr = textureLoad(gbuf_nd, min(px + vec2i(1,0), sz-1), 0).w;
  let zu = textureLoad(gbuf_nd, min(px + vec2i(0,1), sz-1), 0).w;
  let gz = max(max(abs(zr - cz), abs(zu - cz)), cz * 0.002);

  let step = i32(params.step_size);

  // === VARIANCE (SVGF §4.2): propagated through à-trous passes ===
  // Pass 0 (step=1): compute spatial variance, write to alpha
  // Pass 1+ (step>1): read propagated variance from alpha (filtered by previous passes)
  // This ensures sigma reflects ORIGINAL noise, not the already-filtered signal.
  var d_var: f32;
  var s_m1 = 0.0; var s_m2 = 0.0; // specular always uses spatial (no alpha channel)

  if step <= 1 {
    // First pass: compute spatial 3×3 variance, initialize alpha chain
    var d_m1 = 0.0; var d_m2 = 0.0;
    for (var vy = -1; vy <= 1; vy++) {
      for (var vx = -1; vx <= 1; vx++) {
        let vp = clamp(px + vec2i(vx, vy), vec2i(0), sz - 1);
        let vl = luma(textureLoad(in_color, vp, 0).rgb);
        d_m1 += vl; d_m2 += vl * vl;
        let vsl = luma(textureLoad(in_spec, vp, 0).rgb);
        s_m1 += vsl; s_m2 += vsl * vsl;
      }
    }
    d_m1 /= 9.0; d_m2 /= 9.0;
    d_var = max(d_m2 - d_m1 * d_m1, 0.0);
  } else {
    // Subsequent passes: read propagated variance from diffuse alpha
    // Also Gaussian-filter the variance from 3×3 neighbors for stability
    var var_sum = 0.0;
    for (var vy = -1; vy <= 1; vy++) {
      for (var vx = -1; vx <= 1; vx++) {
        let vp = clamp(px + vec2i(vx, vy), vec2i(0), sz - 1);
        var_sum += textureLoad(in_color, vp, 0).a;
        let vsl = luma(textureLoad(in_spec, vp, 0).rgb);
        s_m1 += vsl; s_m2 += vsl * vsl;
      }
    }
    d_var = max(var_sum / 9.0, 0.0);
  }
  let d_std = sqrt(d_var);
  s_m1 /= 9.0; s_m2 /= 9.0;
  let s_std = sqrt(max(s_m2 - s_m1 * s_m1, 0.0));

  // σ_l = 4.0 per SVGF (Schied 2017, eq.5)
  let diff_sigma_scale = 4.0;
  // Specular: roughness modulates (ReLAX, NRD)
  let spec_sigma_scale = mix(4.0 * 0.3, 4.0, roughness);

  // === FILTER (both signals + variance filtering in one pass) ===
  // f16 weight sums: reduces register pressure, 2x faster weight multiply on NVIDIA
  var d_sum = vec3f(0.0); var d_wsum: f16 = 0.0h;
  var s_sum = vec3f(0.0); var s_wsum: f16 = 0.0h;
  var d_var_filtered = 0.0; var d_var_wsum: f16 = 0.0h;

  for (var dy = -2; dy <= 2; dy++) {
    for (var dx = -2; dx <= 2; dx++) {
      let ki = u32((dy + 2) * 5 + (dx + 2));
      let sp = clamp(px + vec2i(dx, dy) * step, vec2i(0), sz - 1);
      let snd = textureLoad(gbuf_nd, sp, 0);
      let s_diff = textureLoad(in_color, sp, 0);
      let sd = s_diff.rgb;
      let ss = textureLoad(in_spec, sp, 0).rgb;

      // Normal edge-stopping (SVGF, Schied 2017, eq.4, σ_n=128)
      let wn = f16(pow(max(dot(cn, snd.xyz), 0.0), 128.0));
      let dz = abs(cz - snd.w);
      let wz = f16(exp(-dz / (gz * f32(step) + 1e-3)));

      // Diffuse luminance edge-stopping (hit distance modulated)
      let d_dl = abs(cl - luma(sd));
      let d_sigma = diff_sigma_scale * d_std * hit_factor + 0.01;
      let d_wl = f16(exp(-d_dl / max(d_sigma, 0.01)));
      let d_w = f16(KW[ki]) * wn * wz * d_wl;
      d_sum += sd * f32(d_w);
      d_wsum += d_w;

      // Filter variance through à-trous kernel (SVGF §4.2)
      let geom_w = f16(KW[ki]) * wn * wz;
      d_var_filtered += s_diff.a * f32(geom_w);
      d_var_wsum += geom_w;

      // Specular luminance edge-stopping (roughness + hit distance modulated)
      let s_dl = abs(csl - luma(ss));
      let s_sigma = spec_sigma_scale * s_std * hit_factor + 0.01;
      let s_wl = f16(exp(-s_dl / max(s_sigma, 0.01)));
      let s_w = f16(KW[ki]) * wn * wz * s_wl;
      s_sum += ss * f32(s_w);
      s_wsum += s_w;
    }
  }

  // Output: color + filtered variance in alpha (propagated to next pass)
  let out_var = d_var_filtered / max(f32(d_var_wsum), 1e-6);
  textureStore(out_color, px, vec4f(d_sum / max(f32(d_wsum), 1e-6), out_var));
  textureStore(out_spec, px, vec4f(s_sum / max(f32(s_wsum), 1e-6), textureLoad(in_spec, px, 0).a));
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

  // === Anti-firefly: 3σ percentile clamp (HPG 2025, Lalber) ===
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
  let d_max_lum = d_mean + 3.0 * d_std + 0.1;

  let s_mean = s_m1 / 9.0;
  let s_std = sqrt(max(s_m2 / 9.0 - s_mean * s_mean, 0.0));
  let s_max_lum = s_mean + 3.0 * s_std + 0.1;

  let cl_raw = luma(cc_raw);
  var cc = cc_raw;
  if cl_raw > d_max_lum && cl_raw > 0.01 { cc = cc_raw * (d_max_lum / cl_raw); }
  let csl_raw = luma(cs_raw);
  var cs = cs_raw;
  if csl_raw > s_max_lum && csl_raw > 0.01 { cs = cs_raw * (s_max_lum / csl_raw); }

  let cl = luma(cc);
  let csl = luma(cs);

  // === 3x3 bilateral filter (f16 weights) ===
  var d_sum = cc;  var d_wsum: f16 = 1.0h;
  var s_sum = cs;  var s_wsum: f16 = 1.0h;

  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      if dx == 0 && dy == 0 { continue; }
      let sp = clamp(px + vec2i(dx, dy), vec2i(0), sz - 1);
      let snd = textureLoad(gbuf_nd, sp, 0);
      let sd = textureLoad(in_color, sp, 0).rgb;
      let ss = textureLoad(in_spec, sp, 0).rgb;

      let sd_l = luma(sd);
      var sd_c = sd;
      if sd_l > d_max_lum && sd_l > 0.01 { sd_c = sd * (d_max_lum / sd_l); }
      let ss_l = luma(ss);
      var ss_c = ss;
      if ss_l > s_max_lum && ss_l > 0.01 { ss_c = ss * (s_max_lum / ss_l); }

      let wn = f16(pow(max(dot(cn, snd.xyz), 0.0), 32.0));
      let wz = f16(exp(-abs(cz - snd.w) / max(cz * 0.05, 1e-2)));

      let d_wl = f16(exp(-abs(cl - luma(sd_c)) / max(cl * 0.5 + 0.1, 0.01)));
      let d_w = wn * wz * d_wl;
      d_sum += sd_c * f32(d_w);
      d_wsum += d_w;

      let s_wl = f16(exp(-abs(csl - luma(ss_c)) / max(csl * 0.5 + 0.1, 0.01)));
      let s_w = wn * wz * s_wl;
      s_sum += ss_c * f32(s_w);
      s_wsum += s_w;
    }
  }

  let center_hit_dist = textureLoad(in_spec, px, 0).a;
  textureStore(out_color, px, vec4f(d_sum / f32(d_wsum), 1.0));
  textureStore(out_spec, px, vec4f(s_sum / f32(s_wsum), center_hit_dist));
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
  var hdr = max(albedo, vec3f(0.02)) * denoised_diff + denoised_spec;

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
