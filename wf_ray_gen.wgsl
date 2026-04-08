// ============================================================
// Wavefront Stage 1: Primary Ray Generation
// Generates one ray per pixel, writes to ray buffer
// ============================================================

struct GenParams {
  resolution: vec2f,
  frame_seed: u32,
  _pad: u32,
  camera_pos: vec3f,
  _pad1: f32,
  camera_forward: vec3f,
  _pad2: f32,
  camera_right: vec3f,
  _pad3: f32,
  camera_up: vec3f,
  fov_factor: f32,
};

@group(0) @binding(0) var<uniform> params: GenParams;
@group(0) @binding(1) var<storage, read_write> rays: array<vec4f>;       // 3 vec4f per ray
@group(0) @binding(2) var<storage, read_write> counters: array<atomic<u32>>; // [0]=active_rays
@group(0) @binding(3) var gbuf_nd: texture_2d<f32>;  // depth hint from rasterizer

// PCG random
var<private> rng_state: u32;
fn pcg(state: ptr<private, u32>) -> u32 {
  let s = *state;
  *state = s * 747796405u + 2891336453u;
  let word = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (word >> 22u) ^ word;
}
fn rand() -> f32 { return f32(pcg(&rng_state)) / 4294967295.0; }

const R2_A1: f32 = 0.7548776662466927;
const R2_A2: f32 = 0.5698402909980532;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let pixel = vec2u(gid.xy);
  let res = vec2u(params.resolution);
  if pixel.x >= res.x || pixel.y >= res.y { return; }

  // RNG init with spatial decorrelation
  rng_state = (pixel.x * 1973u + pixel.y * 9277u + params.frame_seed * 26699u) | 1u;
  _ = pcg(&rng_state);
  let spatial_bn = fract(R2_A1 * f32(pixel.x) + R2_A2 * f32(pixel.y));
  rng_state += u32(spatial_bn * 4294967295.0);
  _ = pcg(&rng_state);

  // Sub-pixel jitter (Halton per-pixel)
  let halton_hash = (pixel.x * 12979u + pixel.y * 48271u) & 0xFFu;
  let hidx = ((params.frame_seed + halton_hash) % 256u) + 1u;
  var hx = 0.0; var hb2 = 0.5;
  var hi2 = hidx;
  for (var _h = 0u; _h < 10u; _h++) { if hi2 == 0u { break; } hx += hb2 * f32(hi2 % 2u); hi2 /= 2u; hb2 *= 0.5; }
  var hy = 0.0; var hb3 = 1.0 / 3.0;
  var hi3 = hidx;
  for (var _h = 0u; _h < 10u; _h++) { if hi3 == 0u { break; } hy += hb3 * f32(hi3 % 3u); hi3 /= 3u; hb3 /= 3.0; }

  let jitter = vec2f(hx, hy);
  let uv = (vec2f(f32(pixel.x), f32(pixel.y)) + jitter) / params.resolution;
  let ndc = uv * 2.0 - 1.0;
  let aspect = params.resolution.x / params.resolution.y;

  let ray_dir = normalize(
    params.camera_forward +
    ndc.x * aspect * params.fov_factor * params.camera_right +
    ndc.y * params.fov_factor * params.camera_up
  );

  // Write ray to buffer (3 vec4f per ray)
  let ray_idx = pixel.y * res.x + pixel.x;
  let base = ray_idx * 3u;
  rays[base]     = vec4f(params.camera_pos, 1e6);              // origin + t_max
  rays[base + 1u] = vec4f(ray_dir, bitcast<f32>(encode_pixel(pixel.x, pixel.y))); // dir + pixel_id
  rays[base + 2u] = vec4f(1.0, 1.0, 1.0, bitcast<f32>(encode_bounce(0u, true, false))); // throughput + bounce

  // Increment active ray counter
  atomicAdd(&counters[0], 1u);
}

fn encode_pixel(x: u32, y: u32) -> u32 { return x | (y << 16u); }
fn encode_bounce(count: u32, is_diffuse: bool, is_specular: bool) -> u32 {
  return count | (select(0u, 256u, is_diffuse)) | (select(0u, 512u, is_specular));
}
