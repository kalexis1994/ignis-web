// Fullscreen blit with exposure + Reinhard tonemap + sRGB gamma
// so the raw HDR noisy output is actually visible on a non-sRGB canvas.

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
  var pos = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f( 3.0, -1.0),
    vec2f(-1.0,  3.0),
  );
  var out: VertexOutput;
  out.position = vec4f(pos[idx], 0.0, 1.0);
  out.uv = (pos[idx] + 1.0) * 0.5;
  return out;
}

@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;

const EXPOSURE: f32 = 2.0;

fn linear_to_srgb(c: vec3f) -> vec3f {
  let lo = c * 12.92;
  let hi = 1.055 * pow(max(c, vec3f(0.0)), vec3f(1.0/2.4)) - 0.055;
  return select(hi, lo, c < vec3f(0.0031308));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
  let hdr = textureSample(tex, tex_sampler, in.uv).rgb * EXPOSURE;
  // Reinhard extended: x / (1 + x), per-channel
  let tm = hdr / (vec3f(1.0) + hdr);
  return vec4f(linear_to_srgb(tm), 1.0);
}
