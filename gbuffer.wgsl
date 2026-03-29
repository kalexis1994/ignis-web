// G-Buffer Rasterization — replaces primary ray BVH traversal
// Adreno rasterizes 3.7M tris in ~1ms vs ~15ms for BVH primary rays
// Outputs: normal + materialId + linear depth

struct GBufUniforms {
  viewProj: mat4x4f,
  camPos: vec3f,
  _pad: f32,
};

@group(0) @binding(0) var<uniform> u: GBufUniforms;
@group(0) @binding(1) var<storage, read> materials: array<vec4f>; // 4 vec4f per material
@group(0) @binding(2) var tex_arr: texture_2d_array<f32>;
@group(0) @binding(3) var tex_samp: sampler;

struct VsIn {
  @location(0) pos: vec4f,     // xyz + uv.x in w
  @location(1) normal: vec4f,  // xyz + uv.y in w
  @location(2) matId: f32,
};

struct VsOut {
  @builtin(position) clipPos: vec4f,
  @location(0) worldNormal: vec3f,
  @location(1) worldPos: vec3f,
  @location(2) @interpolate(flat) matId: f32,
  @location(3) uv: vec2f,
};

@vertex fn vs(in: VsIn) -> VsOut {
  var out: VsOut;
  var cp = u.viewProj * vec4f(in.pos.xyz, 1.0);
  out.clipPos = cp;
  out.worldNormal = in.normal.xyz;
  out.worldPos = in.pos.xyz;
  out.matId = in.matId;
  out.uv = vec2f(in.pos.w, in.normal.w); // UV packed in .w channels
  return out;
}

struct FsOut {
  @location(0) normalDepth: vec4f,  // normal.xyz + depth
  @location(1) matIdUV: vec4f,      // materialId + UV.xy + 0
};

@fragment fn fs(in: VsOut, @builtin(front_facing) front: bool) -> FsOut {
  let mi = u32(in.matId + 0.5);
  // Material data layout: 4 vec4f per mat
  // vec4f[0] = albedo.xyz + mat_type
  // vec4f[1] = emission.xyz + roughness
  // vec4f[2] = metallic + base_tex + mr_tex + normal_tex
  // vec4f[3] = alpha_mode + alpha_cutoff + ior + pad
  let m2 = materials[mi * 4u + 2u]; // metallic, base_tex, mr_tex, normal_tex
  let m3 = materials[mi * 4u + 3u]; // alpha_mode, alpha_cutoff, ior

  // Alpha test using base color texture
  let base_tex = i32(m2.y + 0.5);
  let alpha_mode = u32(m3.x + 0.5);
  if base_tex >= 0 && alpha_mode >= 1u {
    let ta = textureSampleLevel(tex_arr, tex_samp, in.uv, base_tex, 0.0).a;
    if ta < 0.5 { discard; }
  }

  var out: FsOut;
  let depth = length(in.worldPos - u.camPos);
  var n = normalize(in.worldNormal);
  if !front { n = -n; }
  out.normalDepth = vec4f(n, depth);
  out.matIdUV = vec4f(in.matId, fract(in.uv), 0.0); // fract: fits fp16 precision
  return out;
}
