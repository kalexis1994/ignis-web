// G-Buffer Rasterization — replaces primary ray BVH traversal
// Adreno rasterizes 3.7M tris in ~1ms vs ~15ms for BVH primary rays
// Outputs: normal + materialId + linear depth

struct GBufUniforms {
  viewProj: mat4x4f,
  camPos: vec3f,
  _pad: f32,
};

@group(0) @binding(0) var<uniform> u: GBufUniforms;
@group(0) @binding(1) var<storage, read> materials: array<vec4f>; // 20 vec4f per material
@group(0) @binding(2) var tex_arr: texture_2d_array<f32>;
@group(0) @binding(3) var tex_samp: sampler;
const MAT_FLAG_THIN_TRANSMISSION: u32 = 1u;
const MAT_FLAG_DOUBLE_SIDED: u32 = 2u;

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
  @location(0) worldNormal: vec3f,
  @location(1) worldPos: vec3f,
  @location(2) @interpolate(flat) matId: f32,
  @location(3) uv0: vec2f,
  @location(4) uv1: vec2f,
  @location(5) uv2: vec2f,
  @location(6) uv3: vec2f,
};

@vertex fn vs(in: VsIn) -> VsOut {
  var out: VsOut;
  var cp = u.viewProj * vec4f(in.pos.xyz, 1.0);
  out.clipPos = cp;
  out.worldNormal = in.normal.xyz;
  out.worldPos = in.pos.xyz;
  out.matId = in.matId;
  out.uv0 = vec2f(in.pos.w, in.normal.w); // UV0 packed in .w channels
  out.uv1 = in.uv1;
  out.uv2 = in.uv2;
  out.uv3 = in.uv3;
  return out;
}

struct FsOut {
  @location(0) normalDepth: vec4f,  // normal.xyz + depth
  @location(1) matIdUV: vec4f,      // materialId + UV.xy + 0
};

fn decode_tex_index(encoded: f32) -> i32 {
  if encoded < 0.0 { return -1; }
  return i32(encoded + 0.5);
}

fn select_uv(uv0: vec2f, uv1: vec2f, uv2: vec2f, uv3: vec2f, texcoord: u32) -> vec2f {
  if texcoord == 1u { return uv1; }
  if texcoord == 2u { return uv2; }
  if texcoord == 3u { return uv3; }
  return uv0;
}

@fragment fn fs(in: VsOut, @builtin(front_facing) front: bool) -> FsOut {
  let mi = u32(in.matId + 0.5);
  // Material data layout: 20 vec4f per mat
  // vec4f[0] = albedo.xyz + mat_type
  // vec4f[1] = emission.xyz + roughness
  // vec4f[2] = metallic + base_tex + mr_tex + normal_tex
  // vec4f[3] = alpha_mode + alpha_cutoff + ior + emission_strength
  // vec4f[4] = transmission + transmission_tex + thickness + flags
  // vec4f[5] = base_alpha + base_texcoord + mr_texcoord + normal_texcoord
  // vec4f[6] = normal_scale + emissive_tex + occlusion_tex + thickness_tex
  // vec4f[7] = transmission_texcoord + emissive_texcoord + occlusion_texcoord + thickness_texcoord
  // vec4f[8] = occlusion_strength + attenuation_distance + attenuation_color.rg
  // vec4f[9] = attenuation_color.b + pad
  let base = mi * 20u;
  let m2 = materials[base + 2u];
  let m3 = materials[base + 3u];
  let m4 = materials[base + 4u];
  let m5 = materials[base + 5u];

  let flags = u32(m4.w + 0.5);
  let double_sided = (flags & MAT_FLAG_DOUBLE_SIDED) != 0u || (m4.x > 0.001 && (flags & MAT_FLAG_THIN_TRANSMISSION) == 0u);
  if !double_sided && !front { discard; }

  // Alpha coverage using baseColorFactor.a * baseColorTexture.a
  let base_tex = decode_tex_index(m2.y);
  let alpha_mode = u32(m3.x + 0.5);
  if alpha_mode >= 1u {
    let uv = select_uv(in.uv0, in.uv1, in.uv2, in.uv3, u32(m5.y + 0.5));
    var alpha = m5.x;
    if base_tex >= 0 {
      alpha *= textureSampleLevel(tex_arr, tex_samp, uv, base_tex, 0.0).a;
    }
    if alpha_mode == 1u && alpha < m3.y { discard; }
    if alpha_mode == 2u && alpha <= 0.001 { discard; }
  }

  var out: FsOut;
  let depth = length(in.worldPos - u.camPos);
  var n = normalize(in.worldNormal);
  if !front { n = -n; }
  out.normalDepth = vec4f(n, depth);
  let debug_uv = select_uv(in.uv0, in.uv1, in.uv2, in.uv3, u32(m5.y + 0.5));
  out.matIdUV = vec4f(in.matId, fract(debug_uv), 0.0); // kept for debug / legacy consumers
  return out;
}
