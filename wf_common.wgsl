// ============================================================
// Wavefront Path Tracer — Shared structures and buffers
// ============================================================

// Ray state: 48 bytes per ray (3 × vec4f)
struct Ray {
  origin: vec3f,
  t_max: f32,
  dir: vec3f,
  pixel_id: u32,  // encodes x | (y << 16)
  throughput: vec3f,
  bounce: u32,     // bits 0-7: bounce count, bit 8: is_diffuse_path, bit 9: specular_bounce
};

// Hit result: 16 bytes
struct Hit {
  tri_idx: u32,
  u: f32,
  v: f32,
  t: f32,
};

// Shadow ray: 48 bytes (3 × vec4f)
struct ShadowRay {
  origin: vec3f,
  _pad0: f32,
  dir: vec3f,
  pixel_id: u32,
  radiance: vec3f,  // light contribution if unoccluded
  is_diffuse: f32,  // 1.0 = add to diffuse, 0.0 = add to specular
};

// Per-pixel accumulator: 32 bytes (2 × vec4f)
struct PixelAccum {
  diffuse: vec3f,
  sample_count: f32,
  specular: vec3f,
  depth: f32,
};

// Wavefront counters
struct WfCounters {
  active_rays: atomic<u32>,
  shadow_rays: atomic<u32>,
  bounce_rays: atomic<u32>,
  _pad: u32,
};

// Encode/decode pixel ID
fn encode_pixel(x: u32, y: u32) -> u32 { return x | (y << 16u); }
fn decode_pixel_x(id: u32) -> u32 { return id & 0xFFFFu; }
fn decode_pixel_y(id: u32) -> u32 { return id >> 16u; }

// Encode bounce info
fn encode_bounce(count: u32, is_diffuse: bool, is_specular: bool) -> u32 {
  return count | (select(0u, 256u, is_diffuse)) | (select(0u, 512u, is_specular));
}
fn decode_bounce_count(b: u32) -> u32 { return b & 0xFFu; }
fn decode_is_diffuse(b: u32) -> bool { return (b & 256u) != 0u; }
fn decode_is_specular(b: u32) -> bool { return (b & 512u) != 0u; }
