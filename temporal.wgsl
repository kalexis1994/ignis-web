// Ignis SVGF — Temporal Reprojection (dual-signal diffuse + specular)
//
// Per-pixel history length tracking (NRD/ReBLUR-style):
// - Diffuse history alpha = accumulated frame count
// - Specular history alpha = previous camera-space Z (depth disocclusion)
// - AABB clip (Salvi) with adaptive expansion
// - Catmull-Rom sampling function available (Jimenez 2014)
// - ReSTIR GI temporal radiance reuse (Talbot et al.)

struct TemporalParams {
  resolution: vec2f,
  alpha: f32,        // blend factor: 0.02 = keep 98% history (converged)
  frames_still: f32, // global counter (kept for path tracer, not used here)
  // Current camera: view matrix columns (right, up, forward, pos)
  cam_right: vec4f,
  cam_up: vec4f,
  cam_fwd: vec4f,
  cam_pos: vec4f,
  // Previous camera
  prev_right: vec4f,
  prev_up: vec4f,
  prev_fwd: vec4f,
  prev_pos: vec4f,
  fov_factor: f32,
  aspect: f32,
  depth_reject_scale: f32, // relative depth threshold for disocclusion (default 0.1 = 10%)
  max_history: f32,        // cap on history length (default 512)
};

@group(0) @binding(0) var<uniform> params: TemporalParams;
// Diffuse signal — alpha stores history length
@group(0) @binding(1) var current_hdr: texture_2d<f32>;        // new 1spp noisy diffuse
@group(0) @binding(2) var prev_denoised: texture_2d<f32>;      // previous denoised diffuse (.a = history count)
@group(0) @binding(3) var depth_tex: texture_2d<f32>;          // current depth (gbuf_nd.w)
@group(0) @binding(4) var accum_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var history_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var tex_sampler: sampler;
// Specular signal — alpha stores camera-space Z for depth rejection
@group(0) @binding(7) var current_spec: texture_2d<f32>;       // new 1spp noisy specular
@group(0) @binding(8) var prev_spec: texture_2d<f32>;          // previous denoised specular (.a = prev cam Z)
@group(0) @binding(9) var spec_accum_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(10) var spec_history_out: texture_storage_2d<rgba16float, write>;

fn luma(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }

// Catmull-Rom (bicubic) sampling using 5 bilinear taps (Jimenez 2014).
// Sharper than bilinear — reduces accumulated blur from temporal reprojection.
fn sample_catmull_rom(tex: texture_2d<f32>, samp: sampler, uv: vec2f, tex_size: vec2f) -> vec4f {
  let pixel = uv * tex_size - 0.5;
  let tc = floor(pixel) + 0.5;
  let f = pixel - tc;
  let f2 = f * f;
  let f3 = f2 * f;

  // Catmull-Rom weights
  let w0 = f2 - 0.5 * (f3 + f);
  let w1 = 1.5 * f3 - 2.5 * f2 + vec2f(1.0);
  let w3 = 0.5 * (f3 - f2);
  let w2 = vec2f(1.0) - w0 - w1 - w3;

  // Collapse to 5 bilinear taps
  let w12 = w1 + w2;
  let tc0 = (tc - 1.0) / tex_size;
  let tc12 = (tc + w2 / w12) / tex_size;
  let tc3 = (tc + 2.0) / tex_size;

  var color = vec4f(0.0);
  // Center cross (5 taps)
  color += textureSampleLevel(tex, samp, vec2f(tc12.x, tc0.y), 0.0) * (w12.x * w0.y);
  color += textureSampleLevel(tex, samp, vec2f(tc0.x, tc12.y), 0.0) * (w0.x * w12.y);
  color += textureSampleLevel(tex, samp, vec2f(tc12.x, tc12.y), 0.0) * (w12.x * w12.y);
  color += textureSampleLevel(tex, samp, vec2f(tc3.x, tc12.y), 0.0) * (w3.x * w12.y);
  color += textureSampleLevel(tex, samp, vec2f(tc12.x, tc3.y), 0.0) * (w12.x * w3.y);

  // Clamp to prevent ringing (negative lobes can cause undershoot)
  return max(color, vec4f(0.0));
}

// Reconstruct world position from pixel + depth + camera
fn reconstruct_world(px: vec2f, depth: f32, right: vec3f, up: vec3f, fwd: vec3f, pos: vec3f, fov: f32, aspect: f32, res: vec2f) -> vec3f {
  let uv = (px + 0.5) / res;
  let ndc = uv * 2.0 - 1.0;
  let rd = normalize(fwd + ndc.x * aspect * fov * right + ndc.y * fov * up);
  return pos + rd * depth;
}

// Project world position to screen UV with a camera
fn project_to_uv(world_pos: vec3f, right: vec3f, up: vec3f, fwd: vec3f, pos: vec3f, fov: f32, aspect: f32) -> vec3f {
  let local = world_pos - pos;
  let z = dot(local, fwd);
  if z <= 0.0 { return vec3f(-1.0); } // behind camera
  let x = dot(local, right);
  let y = dot(local, up);
  let ndcx = x / (z * aspect * fov);
  let ndcy = y / (z * fov);
  return vec3f(ndcx * 0.5 + 0.5, ndcy * 0.5 + 0.5, z);
}

// RGB ↔ YCoCg (lossless, tighter AABB bounds than RGB)
fn rgb_to_ycocg(c: vec3f) -> vec3f {
  return vec3f(
    0.25 * c.r + 0.5 * c.g + 0.25 * c.b,
    0.5 * c.r - 0.5 * c.b,
    -0.25 * c.r + 0.5 * c.g - 0.25 * c.b
  );
}
fn ycocg_to_rgb(c: vec3f) -> vec3f {
  return vec3f(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

// AABB clip (Salvi) — clip history color to neighborhood AABB
fn clip_aabb(mn: vec3f, mx: vec3f, history: vec3f) -> vec3f {
  let center = 0.5 * (mn + mx);
  let half_ext = 0.5 * (mx - mn) + vec3f(0.001);
  let offset = history - center;
  let unit = abs(offset / half_ext);
  let max_unit = max(unit.x, max(unit.y, unit.z));
  if max_unit > 1.0 { return center + offset / max_unit; }
  return history;
}

// Temporal accumulation for a single signal — returns blended result
fn accumulate_signal(
  current: vec3f, history: vec3f, motion: f32, base_alpha: f32,
  px: vec2i, sz: vec2i, src: texture_2d<f32>
) -> vec3f {
  // Neighborhood statistics for AABB clamping (RGB space — wider bounds preserve energy)
  var mn = current;
  var mx = current;
  var m1 = vec3f(0.0);
  var m2 = vec3f(0.0);
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let sp = clamp(px + vec2i(dx, dy), vec2i(0), sz - 1);
      let sc = textureLoad(src, sp, 0).rgb;
      mn = min(mn, sc); mx = max(mx, sc);
      m1 += sc;
      m2 += sc * sc;
    }
  }
  m1 /= 9.0;
  m2 /= 9.0;
  let variance = max(m2 - m1 * m1, vec3f(0.0));
  let stddev = sqrt(variance);

  // Adaptive AABB: tight for disoccluded, wider for stable (accumulate more)
  let aabb_scale = mix(3.0, 6.0, motion);
  let aabb_expand = max(stddev * aabb_scale, (mx - mn) * mix(0.2, 0.4, motion));
  mn -= aabb_expand;
  mx += aabb_expand;

  let clipped = clip_aabb(mn, mx, history);

  // Variance-dependent alpha: high variance → more current (faster update)
  let var_lum = dot(stddev, vec3f(0.2126, 0.7152, 0.0722));
  let motion_alpha = mix(0.08, base_alpha, motion);
  let alpha = max(motion_alpha * 0.1, motion_alpha / (1.0 + var_lum * 10.0));

  return mix(clipped, current, alpha);
}

@compute @workgroup_size(16, 16)
fn temporal(@builtin(global_invocation_id) gid: vec3u) {
  let px = vec2i(gid.xy);
  let sz = vec2i(params.resolution);
  if px.x >= sz.x || px.y >= sz.y { return; }

  let nd_center = textureLoad(depth_tex, px, 0);
  let cn = nd_center.xyz;
  let cz = nd_center.w;
  let diff_raw = textureLoad(current_hdr, px, 0).rgb;
  let spec_raw_s = textureLoad(current_spec, px, 0);
  let spec_raw = spec_raw_s.rgb;
  let cur_hit_dist = spec_raw_s.a;

  // === Spatial pre-filter: 3x3 geometry-weighted average of noisy input ===
  // Done HERE instead of a separate pass — no multi-pass bleeding.
  // Only blends neighbors on the same surface (normal+depth test).
  var d_sum = diff_raw;
  var s_sum = spec_raw;
  var w_sum = 1.0;
  let gz = max(abs(cz) * 0.01, 0.1);
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      if dx == 0 && dy == 0 { continue; }
      let np = clamp(px + vec2i(dx, dy), vec2i(0), sz - 1);
      let nnd = textureLoad(depth_tex, np, 0);
      let wn = pow(max(dot(cn, nnd.xyz), 0.0), 64.0);
      let wz = exp(-abs(cz - nnd.w) / gz);
      let w = wn * wz;
      if w > 0.01 {
        d_sum += textureLoad(current_hdr, np, 0).rgb * w;
        s_sum += textureLoad(current_spec, np, 0).rgb * w;
        w_sum += w;
      }
    }
  }
  let diff_cur = d_sum / w_sum;
  let spec_cur = s_sum / w_sum;

  // === Temporal accumulation with reprojection ===
  var diff_blend = diff_cur;
  var spec_blend = spec_cur;
  var history_len = 0.0;

  // Sky: no history
  if cz > 1e5 {
    textureStore(accum_out, px, vec4f(diff_cur, 0.0));
    textureStore(history_out, px, vec4f(diff_cur, 0.0));
    textureStore(spec_accum_out, px, vec4f(spec_cur, 1.0));
    textureStore(spec_history_out, px, vec4f(spec_cur, 0.0));
    return;
  }

  // Reconstruct world position and reproject to previous frame
  // depth_tex.w = ray hit distance (not camera-space Z)
  let world_pos = reconstruct_world(
    vec2f(f32(px.x), f32(px.y)), nd_center.w,
    params.cam_right.xyz, params.cam_up.xyz, params.cam_fwd.xyz, params.cam_pos.xyz,
    params.fov_factor, params.aspect, params.resolution
  );
  let cam_z = dot(world_pos - params.cam_pos.xyz, params.cam_fwd.xyz);
  let prev_uv_z = project_to_uv(
    world_pos,
    params.prev_right.xyz, params.prev_up.xyz, params.prev_fwd.xyz, params.prev_pos.xyz,
    params.fov_factor, params.aspect
  );

  // Sample history if reprojection is valid
  if prev_uv_z.x >= 0.0 && prev_uv_z.x <= 1.0 && prev_uv_z.y >= 0.0 && prev_uv_z.y <= 1.0 {
    // CatRom (bicubic) history sampling — sharper than bilinear, less temporal blur
    let tex_size = params.resolution;
    let diff_hist_s = sample_catmull_rom(prev_denoised, tex_sampler, prev_uv_z.xy, tex_size);
    let spec_hist_s = sample_catmull_rom(prev_spec, tex_sampler, prev_uv_z.xy, tex_size);
    let prev_z = spec_hist_s.a;
    let z_threshold = max(abs(prev_z) * params.depth_reject_scale, 0.5);
    let depth_valid = abs(prev_uv_z.z - prev_z) < z_threshold;

    if depth_valid {
      // When camera moves, cap history to respond faster to illumination changes
      // frames_still=0 → max_hist=8 (alpha~12%), still=32+ → max_hist=512 (alpha~0.2%)
      let effective_max = mix(8.0, params.max_history, clamp(params.frames_still / 32.0, 0.0, 1.0));
      history_len = min(diff_hist_s.a + 1.0, effective_max);

      // AABB ghosting rejection: only during motion (history < 32 frames)
      // When still, skip AABB to preserve converged illumination
      var diff_hist = diff_hist_s.rgb;
      var spec_hist = spec_hist_s.rgb;
      if history_len < 32.0 {
        // Compute 3x3 AABB from current noisy input
        var d_mn = diff_cur; var d_mx = diff_cur;
        var s_mn = spec_cur; var s_mx = spec_cur;
        for (var dy = -1; dy <= 1; dy++) {
          for (var dx = -1; dx <= 1; dx++) {
            let sp = clamp(px + vec2i(dx, dy), vec2i(0), sz - 1);
            let sd = textureLoad(current_hdr, sp, 0).rgb;
            let ss = textureLoad(current_spec, sp, 0).rgb;
            d_mn = min(d_mn, sd); d_mx = max(d_mx, sd);
            s_mn = min(s_mn, ss); s_mx = max(s_mx, ss);
          }
        }
        // Expand AABB by 50% to tolerate noise while still rejecting ghosts
        let d_expand = (d_mx - d_mn) * 0.5;
        let s_expand = (s_mx - s_mn) * 0.5;
        diff_hist = clip_aabb(d_mn - d_expand, d_mx + d_expand, diff_hist);
        spec_hist = clip_aabb(s_mn - s_expand, s_mx + s_expand, spec_hist);
      }

      // AntiLag (ReBLUR): detect lighting change → reduce accumSpeed
      // Compare history luminance with current neighborhood mean
      let hist_dl = luma(diff_hist);
      let cur_dl = luma(diff_cur);
      let d_change = abs(hist_dl - cur_dl) / (max(hist_dl, cur_dl) + 0.001);
      // If change > threshold relative to accumSpeed, reduce history confidence
      let d_antilag = 1.0 / (1.0 + d_change * history_len * 0.5);
      let effective_history = history_len * d_antilag;

      let alpha = max(1.0 / (effective_history + 1.0), 0.02);
      diff_blend = mix(diff_hist, diff_cur, alpha);

      // === Virtual motion for specular (ReBLUR) ===
      // Specular reflections move differently than surfaces. For smooth surfaces,
      // reproject from a VIRTUAL position behind the surface along the reflection.
      let roughness = textureLoad(depth_tex, px, 0).x; // roughness stored in nd.x? No...
      // We don't have roughness in temporal. Use surface motion for now, but with
      // reduced alpha for specular (accumulate slower = more stable reflections)
      let V = normalize(params.cam_pos.xyz - world_pos);
      let N = cn;
      let spec_hit = cur_hit_dist;
      let dominantFactor = 1.0 - textureLoad(current_spec, px, 0).a * textureLoad(current_spec, px, 0).a; // hitdist as proxy

      // Virtual position: X + reflect(-V, N) * hitDist * dominantFactor
      // Only use virtual motion for pixels with short hit distance (close reflections)
      var spec_hist_final = spec_hist;
      if spec_hit > 0.01 && spec_hit < 0.9 {
        let R = reflect(-V, N);
        // Decode hit distance from normalized log space
        let real_hit = exp2(spec_hit * 8.0) - 1.0;
        let Xvirtual = world_pos + R * real_hit;
        let vmb_uv = project_to_uv(
          Xvirtual,
          params.prev_right.xyz, params.prev_up.xyz, params.prev_fwd.xyz, params.prev_pos.xyz,
          params.fov_factor, params.aspect
        );
        // If virtual reprojection is valid, blend between surface and virtual history
        if vmb_uv.x >= 0.0 && vmb_uv.x <= 1.0 && vmb_uv.y >= 0.0 && vmb_uv.y <= 1.0 {
          let vmb_hist = textureSampleLevel(prev_spec, tex_sampler, vmb_uv.xy, 0.0).rgb;
          // Blend: smooth surfaces use virtual, rough use surface
          // hit_dist < 0.3 = close reflection = strong virtual motion
          let vmb_weight = clamp(1.0 - spec_hit * 3.0, 0.0, 0.8);
          spec_hist_final = mix(spec_hist, vmb_hist, vmb_weight);
        }
      }
      spec_blend = mix(spec_hist_final, spec_cur, alpha);

      // Store the antilag-adjusted history for downstream passes
      history_len = effective_history;
    }
  }


  textureStore(accum_out, px, vec4f(diff_blend, history_len));
  textureStore(history_out, px, vec4f(diff_blend, history_len));
  textureStore(spec_accum_out, px, vec4f(spec_blend, cur_hit_dist));
  textureStore(spec_history_out, px, vec4f(spec_blend, cam_z));
}
