// Temporal reprojection — dual-signal (diffuse + specular)
// Per-pixel history length tracking (NRD-style):
// - diffuse history alpha = accumulated frame count
// - specular history alpha = previous camera-space Z (for disocclusion detection)
// Disoccluded pixels reset to 0 → aggressive denoising only where needed

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
  // Neighborhood statistics for AABB clamping
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

  // Adaptive AABB
  let aabb_scale = mix(4.0, 8.0, motion);
  let aabb_expand = max(stddev * aabb_scale, (mx - mn) * mix(0.25, 0.5, motion));
  mn -= aabb_expand;
  mx += aabb_expand;

  let clipped = clip_aabb(mn, mx, history);

  // Adaptive alpha
  let var_lum = dot(stddev, vec3f(0.2126, 0.7152, 0.0722));
  let motion_alpha = mix(0.08, base_alpha, motion);
  let alpha = max(motion_alpha * 0.1, motion_alpha / (1.0 + var_lum * 10.0));

  return mix(clipped, current, alpha);
}

@compute @workgroup_size(8, 8)
fn temporal(@builtin(global_invocation_id) gid: vec3u) {
  let px = vec2i(gid.xy);
  let sz = vec2i(params.resolution);
  if px.x >= sz.x || px.y >= sz.y { return; }

  let diff_cur = textureLoad(current_hdr, px, 0).rgb;
  let spec_cur_sample = textureLoad(current_spec, px, 0);
  let spec_cur = spec_cur_sample.rgb;
  let cur_hit_dist = spec_cur_sample.a; // hit distance from path tracer (via preblur)
  let nd = textureLoad(depth_tex, px, 0);
  let depth = nd.w;

  // Sky (depth very large): no meaningful history
  if depth > 1e5 {
    textureStore(accum_out, px, vec4f(diff_cur, 0.0));
    textureStore(history_out, px, vec4f(diff_cur, 0.0));
    textureStore(spec_accum_out, px, vec4f(spec_cur, 1.0));  // hit_dist=1.0 (far/sky)
    textureStore(spec_history_out, px, vec4f(spec_cur, 0.0));
    return;
  }

  // Reconstruct world position from current pixel + depth
  let world_pos = reconstruct_world(
    vec2f(f32(px.x), f32(px.y)), depth,
    params.cam_right.xyz, params.cam_up.xyz, params.cam_fwd.xyz, params.cam_pos.xyz,
    params.fov_factor, params.aspect, params.resolution
  );

  // Camera-space Z for current pixel (stored in specular alpha for next frame)
  let cam_z = dot(world_pos - params.cam_pos.xyz, params.cam_fwd.xyz);

  // Reproject to previous frame UV
  let prev_uv_z = project_to_uv(
    world_pos,
    params.prev_right.xyz, params.prev_up.xyz, params.prev_fwd.xyz, params.prev_pos.xyz,
    params.fov_factor, params.aspect
  );

  var diff_blend = diff_cur;
  var spec_blend = spec_cur;
  var history_len = 0.0; // disoccluded / new pixel

  // Valid reprojection? (UV in bounds)
  if prev_uv_z.x >= 0.0 && prev_uv_z.x <= 1.0 && prev_uv_z.y >= 0.0 && prev_uv_z.y <= 1.0 {
    let diff_hist_sample = textureSampleLevel(prev_denoised, tex_sampler, prev_uv_z.xy, 0.0);
    let spec_hist_sample = textureSampleLevel(prev_spec, tex_sampler, prev_uv_z.xy, 0.0);
    let diff_hist = diff_hist_sample.rgb;
    let spec_hist = spec_hist_sample.rgb;
    let prev_history = diff_hist_sample.a;
    let prev_z = spec_hist_sample.a;

    let z_threshold = max(abs(prev_z) * params.depth_reject_scale, 0.5);
    let depth_valid = abs(prev_uv_z.z - prev_z) < z_threshold;

    if depth_valid {
      history_len = min(prev_history + 1.0, params.max_history);
      let motion = clamp(history_len / 32.0, 0.0, 1.0);

      diff_blend = accumulate_signal(diff_cur, diff_hist, motion, params.alpha, px, sz, current_hdr);
      spec_blend = accumulate_signal(spec_cur, spec_hist, motion, params.alpha, px, sz, current_spec);
    }
  }

  textureStore(accum_out, px, vec4f(diff_blend, history_len));
  textureStore(history_out, px, vec4f(diff_blend, history_len));
  textureStore(spec_accum_out, px, vec4f(spec_blend, cur_hit_dist));
  textureStore(spec_history_out, px, vec4f(spec_blend, cam_z));
}
