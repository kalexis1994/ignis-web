// Shadow Map — depth-only rasterization from sun's perspective
// Replaces expensive BVH shadow ray traces with a single texture lookup

struct ShadowUniforms {
  lightViewProj: mat4x4f,
};

@group(0) @binding(0) var<uniform> u: ShadowUniforms;
@group(0) @binding(1) var<storage, read> materials: array<vec4f>;
@group(0) @binding(2) var tex_arr: texture_2d_array<f32>;
@group(0) @binding(3) var tex_samp: sampler;

struct VsIn {
  @location(0) pos: vec4f,     // xyz + uv.x in w
  @location(1) normal: vec4f,  // xyz + uv.y in w
  @location(2) matId: f32,
  @location(3) uv1: vec2f,
  @location(4) uv2: vec2f,
  @location(5) uv3: vec2f,
};

struct VsOut {
  @builtin(position) clipPos: vec4f,
  @location(0) @interpolate(flat) matId: f32,
  @location(1) uv0: vec2f,
  @location(2) uv1: vec2f,
  @location(3) uv2: vec2f,
  @location(4) uv3: vec2f,
};

@vertex fn vs(in: VsIn) -> VsOut {
  var out: VsOut;
  out.clipPos = u.lightViewProj * vec4f(in.pos.xyz, 1.0);
  out.matId = in.matId;
  out.uv0 = vec2f(in.pos.w, in.normal.w);
  out.uv1 = in.uv1;
  out.uv2 = in.uv2;
  out.uv3 = in.uv3;
  return out;
}

fn select_uv(uv0: vec2f, uv1: vec2f, uv2: vec2f, uv3: vec2f, texcoord: u32) -> vec2f {
  if texcoord == 1u { return uv1; }
  if texcoord == 2u { return uv2; }
  if texcoord == 3u { return uv3; }
  return uv0;
}

fn decode_tex_index(encoded: f32) -> i32 {
  if encoded < 0.0 { return -1; }
  return i32(encoded + 0.5);
}

// Depth-only: no color output, just depth buffer
// Alpha-tested geometry still needs discard
@fragment fn fs(in: VsOut) {
  let mi = u32(in.matId + 0.5);
  let base = mi * 20u;
  let m2 = materials[base + 2u];
  let m3 = materials[base + 3u];
  let m5 = materials[base + 5u];

  let alpha_mode = u32(m3.x + 0.5);
  if alpha_mode >= 1u {
    let base_tex = decode_tex_index(m2.y);
    let uv = select_uv(in.uv0, in.uv1, in.uv2, in.uv3, u32(m5.y + 0.5));
    var alpha = m5.x;
    if base_tex >= 0 {
      alpha *= textureSampleLevel(tex_arr, tex_samp, uv, base_tex, 0.0).a;
    }
    if alpha_mode == 1u && alpha < m3.y { discard; }
    if alpha_mode == 2u && alpha <= 0.001 { discard; }
  }
}
