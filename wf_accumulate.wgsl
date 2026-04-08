// ============================================================
// Wavefront Stage 5: Accumulate shadow ray results + write output
// Two entry points:
//   accumulate_shadows — add unoccluded shadow contributions to pixel accum
//   write_output — write accumulated result to output textures
// ============================================================

struct AccumParams {
  shadow_count: u32,
  resolution: vec2f,
  _pad: f32,
};

@group(0) @binding(0) var<uniform> params: AccumParams;
@group(0) @binding(1) var<storage, read> shadow_rays: array<vec4f>;   // 3 vec4f per shadow ray
@group(0) @binding(2) var<storage, read> shadow_hits: array<vec4f>;   // occlusion result
@group(0) @binding(3) var<storage, read_write> accum: array<vec4f>;   // 2 vec4f per pixel
@group(0) @binding(4) var noisy_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var spec_out: texture_storage_2d<rgba16float, write>;

fn decode_pixel_x(id: u32) -> u32 { return id & 0xFFFFu; }
fn decode_pixel_y(id: u32) -> u32 { return id >> 16u; }

@compute @workgroup_size(256)
fn accumulate_shadows(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if idx >= params.shadow_count { return; }

  // Check if shadow ray was occluded
  let occ = shadow_hits[idx];
  if occ.x > 0.5 { return; } // occluded, skip

  // Read shadow ray data
  let base = idx * 3u;
  let pixel_id = bitcast<u32>(shadow_rays[base + 1u].w);
  let radiance = shadow_rays[base + 2u].xyz;
  let is_diffuse = shadow_rays[base + 2u].w > 0.5;

  let px = decode_pixel_x(pixel_id);
  let py = decode_pixel_y(pixel_id);
  let pixel_idx = py * u32(params.resolution.x) + px;
  let acc_base = pixel_idx * 2u;

  // Accumulate (note: not atomic, may have races — acceptable for 1SPP noise)
  if is_diffuse {
    let prev = accum[acc_base];
    accum[acc_base] = vec4f(prev.xyz + radiance, prev.w);
  } else {
    let prev = accum[acc_base + 1u];
    accum[acc_base + 1u] = vec4f(prev.xyz + radiance, prev.w);
  }
}

@compute @workgroup_size(16, 16)
fn write_output(@builtin(global_invocation_id) gid: vec3u) {
  let px = vec2i(gid.xy);
  let sz = vec2i(params.resolution);
  if px.x >= sz.x || px.y >= sz.y { return; }

  let pixel_idx = u32(px.y) * u32(sz.x) + u32(px.x);
  let acc_base = pixel_idx * 2u;

  let diff = accum[acc_base];
  let spec = accum[acc_base + 1u];

  // Firefly clamp
  let MAX_LUM = 10.0;
  var d = diff.xyz;
  let dl = max(d.x, max(d.y, d.z));
  if dl > MAX_LUM { d *= MAX_LUM / dl; }
  var s = spec.xyz;
  let sl = max(s.x, max(s.y, s.z));
  if sl > MAX_LUM { s *= MAX_LUM / sl; }

  textureStore(noisy_out, px, vec4f(d, 1.0));
  textureStore(spec_out, px, vec4f(s, diff.w)); // depth in spec.w for denoiser

  // Clear accumulator for next frame
  accum[acc_base] = vec4f(0.0);
  accum[acc_base + 1u] = vec4f(0.0);
}
