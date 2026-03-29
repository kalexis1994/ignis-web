// Temporal reprojection — reproject previous denoised frame into current accumulation
// Uses depth buffer + previous/current camera matrices to compute motion vectors per-pixel
// Includes neighborhood clamping to prevent ghosting (Salvi's AABB clip)

struct TemporalParams {
  resolution: vec2f,
  alpha: f32,        // blend factor: 0.05 = keep 95% history
  _pad: f32,
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
  _pad2: f32,
  _pad3: f32,
};

@group(0) @binding(0) var<uniform> params: TemporalParams;
@group(0) @binding(1) var current_hdr: texture_2d<f32>;        // new 1spp noisy frame
@group(0) @binding(2) var prev_denoised: texture_2d<f32>;      // previous denoised HDR
@group(0) @binding(3) var depth_tex: texture_2d<f32>;          // current depth (gbuf_nd.w)
@group(0) @binding(4) var accum_out: texture_storage_2d<rgba16float, write>; // blended output
@group(0) @binding(5) var history_out: texture_storage_2d<rgba16float, write>; // save for next frame
@group(0) @binding(6) var tex_sampler: sampler;

fn luma(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }

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

@compute @workgroup_size(8, 8)
fn temporal(@builtin(global_invocation_id) gid: vec3u) {
  let px = vec2i(gid.xy);
  let sz = vec2i(params.resolution);
  if px.x >= sz.x || px.y >= sz.y { return; }

  let current = textureLoad(current_hdr, px, 0).rgb;
  let nd = textureLoad(depth_tex, px, 0);
  let depth = nd.w;

  // If sky (depth very large), just use current frame
  if depth > 1e5 {
    textureStore(accum_out, px, vec4f(current, 1.0));
    textureStore(history_out, px, vec4f(current, 1.0));
    return;
  }

  // Reconstruct world position from current pixel + depth
  let world_pos = reconstruct_world(
    vec2f(f32(px.x), f32(px.y)), depth,
    params.cam_right.xyz, params.cam_up.xyz, params.cam_fwd.xyz, params.cam_pos.xyz,
    params.fov_factor, params.aspect, params.resolution
  );

  // Reproject to previous frame UV
  let prev_uv_z = project_to_uv(
    world_pos,
    params.prev_right.xyz, params.prev_up.xyz, params.prev_fwd.xyz, params.prev_pos.xyz,
    params.fov_factor, params.aspect
  );

  var blended = current;

  // Valid reprojection?
  if prev_uv_z.x >= 0.0 && prev_uv_z.x <= 1.0 && prev_uv_z.y >= 0.0 && prev_uv_z.y <= 1.0 {
    // Sample previous denoised frame with bilinear
    let history = textureSampleLevel(prev_denoised, tex_sampler, prev_uv_z.xy, 0.0).rgb;

    // Neighborhood statistics: min/max + variance for adaptive clamping
    var mn = current;
    var mx = current;
    var m1 = vec3f(0.0);
    var m2 = vec3f(0.0);
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let sp = clamp(px + vec2i(dx, dy), vec2i(0), sz - 1);
        let sc = textureLoad(current_hdr, sp, 0).rgb;
        mn = min(mn, sc); mx = max(mx, sc);
        m1 += sc;
        m2 += sc * sc;
      }
    }
    m1 /= 9.0;
    m2 /= 9.0;
    let variance = max(m2 - m1 * m1, vec3f(0.0));
    let stddev = sqrt(variance);

    // Very wide AABB for 1SPP: Monte Carlo noise is extreme, don't reject history
    let aabb_expand = max(stddev * 8.0, (mx - mn) * 0.5);
    mn -= aabb_expand;
    mx += aabb_expand;

    let clipped = clip_aabb(mn, mx, history);

    // Progressive accumulation via temporal: alpha = 1/(frame_count+1)
    // frames_still=0 → alpha=params.alpha (fast blend for camera motion)
    // frames_still=100 → alpha≈0.01 (slow convergence, stable image)
    let var_lum = dot(stddev, vec3f(0.2126, 0.7152, 0.0722));
    let base_alpha = params.alpha;
    let alpha = max(base_alpha * 0.1, base_alpha / (1.0 + var_lum * 10.0));

    blended = mix(clipped, current, alpha);
  }

  textureStore(accum_out, px, vec4f(blended, 1.0));
  textureStore(history_out, px, vec4f(blended, 1.0));
}
