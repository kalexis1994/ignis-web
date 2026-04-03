// Monte Carlo Path Tracer — WebGPU Compute Shader
// Techniques ported from ignis-rt (NVIDIA):
// - R2 quasi-random jitter (better convergence)
// - GGX Cook-Torrance BRDF + VNDF sampling (Heitz 2018)
// - Real dielectric Fresnel (not Schlick)
// - Smith height-correlated geometry
// - AgX Punchy tone mapping (Blender 4 / Troy Sobotka)
// - Firefly luminance clamping

struct PunctualLight {
  pos_range: vec4f,
  dir_inner: vec4f,
  color_intensity: vec4f,
  params: vec4f,
};

struct Uniforms {
  resolution: vec2f,
  sample_count: u32,
  frame_seed: u32,
  camera_pos: vec3f,
  _pad0: f32,
  camera_forward: vec3f,
  _pad1: f32,
  camera_right: vec3f,
  _pad2: f32,
  camera_up: vec3f,
  fov_factor: f32,
  sun_dir: vec3f,
  emissive_tri_count: u32,
  max_bounces: u32,
  frames_still: u32,
  aspect: f32,
  restir_enabled: u32,
  // Previous camera for ReSTIR reprojection
  prev_pos: vec3f,
  _pad6: f32,
  prev_forward: vec3f,
  _pad7: f32,
  prev_right: vec3f,
  _pad8: f32,
  prev_up: vec3f,
  _pad9: f32,
  light_count: u32,
  sun_enabled: u32,
  _pad10: vec2u,
  lights: array<PunctualLight, 16>,
};

struct BVHNode { aabb_min: vec3f, left_first: u32, aabb_max: vec3f, tri_count: u32, };
struct Material {
  d0: vec4f, // albedo.rgb + mat_type
  d1: vec4f, // emission.rgb + roughness
  d2: vec4f, // metallic + base_tex + mr_tex + normal_tex
  d3: vec4f, // alpha_mode + alpha_cutoff + ior + emission_strength
  d4: vec4f, // transmission + transmission_tex + thickness + flags
  d5: vec4f, // base_alpha + base_texcoord + mr_texcoord + normal_texcoord
  d6: vec4f, // normal_scale + emissive_tex + occlusion_tex + thickness_tex
  d7: vec4f, // transmission_texcoord + emissive_texcoord + occlusion_texcoord + thickness_texcoord
  d8: vec4f, // occlusion_strength + attenuation_distance + attenuation_color.rg
  d9: vec4f, // attenuation_color.b + pad
  d10: vec4f, // specular_factor + specular_tex + specular_color_tex + specular_color.r
  d11: vec4f, // specular_color.gb + specular_texcoord + specular_color_texcoord
  d12: vec4f, // clearcoat_factor + clearcoat_tex + clearcoat_roughness + clearcoat_rough_tex
  d13: vec4f, // clearcoat_texcoord + clearcoat_rough_texcoord + clearcoat_normal_tex + clearcoat_normal_texcoord
  d14: vec4f, // clearcoat_normal_scale + sheen_color.rgb
  d15: vec4f, // sheen_roughness + sheen_color_tex + sheen_rough_tex + sheen_color_texcoord
  d16: vec4f, // sheen_rough_texcoord + anisotropy_strength + anisotropy_rotation + anisotropy_tex
  d17: vec4f, // anisotropy_texcoord + iridescence_factor + iridescence_tex + iridescence_ior
  d18: vec4f, // iridescence_thickness_min + iridescence_thickness_max + iridescence_thickness_tex + iridescence_texcoord
  d19: vec4f, // iridescence_thickness_texcoord + dispersion + pad
};
struct HitInfo { t: f32, u: f32, v: f32, tri_idx: u32, hit: bool, };

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var noisy_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var gbuf_nd: texture_2d<f32>;
@group(0) @binding(3) var gbuf_mat_uv: texture_2d<f32>;
@group(0) @binding(4) var albedo_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(5) var denoise_nd_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var specular_out: texture_storage_2d<rgba16float, write>;
// accumulation handled by temporal pass (no extra buffer needed)

@group(1) @binding(0) var<storage, read> vertices: array<vec4f>;
@group(1) @binding(1) var<storage, read> vert_normals: array<vec4f>;
@group(1) @binding(2) var<storage, read> tri_data: array<vec4u>;
@group(1) @binding(3) var<storage, read> bvh_nodes: array<BVHNode>;
@group(1) @binding(4) var<storage, read> material_buf: array<Material>;
// Emissive tris: 1 vec4f each [tri_idx(bitcast u32), area, CDF, 0]
@group(1) @binding(5) var<storage, read> emissive_tris: array<vec4f>;
@group(1) @binding(6) var<storage, read> vert_uv_extra: array<f32>;
// SHaRC radiance cache — packed into 2 buffers to fit 8 storage buffer limit
// keys_accum: [0..cap) = hash keys (atomic), [cap..cap*5) = accum RGBS (atomic)
// resolved: [0..cap*4) = resolved RGBS (read-only from PT)
struct SharcParams { capacity: u32, frame_index: u32, scene_scale: f32, stale_max: u32, camera_pos: vec3f, _pad: f32, };
@group(2) @binding(0) var<uniform> sharc_params: SharcParams;
@group(2) @binding(1) var<storage, read_write> sharc_keys_accum: array<atomic<u32>>;
@group(2) @binding(2) var<storage, read> sharc_resolved: array<u32>;

@group(3) @binding(0) var tex_array: texture_2d_array<f32>;
@group(3) @binding(1) var tex_sampler_pt: sampler;

// ReSTIR GI — Weighted Reservoir Sampling for indirect lighting
// Packed as 3 vec4f per pixel: [pos+wSum, rad+M, octNorm+hitDist+age]
// Merged into SHaRC bind group (group 2) to stay within 4 bind group limit
@group(2) @binding(3) var<storage, read_write> restir_curr: array<vec4f>;
@group(2) @binding(4) var<storage, read> restir_prev: array<vec4f>;

const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;
const INV_PI: f32 = 0.31830988618;
const INF: f32 = 1e30;
const T_MIN: f32 = 0.00001;    // minimum ray t (intersection test)
const BIAS: f32 = 0.0002;     // shadow/bounce ray offset
// MAX_BOUNCES now comes from uniforms.max_bounces

const SUN_COLOR: vec3f = vec3f(8.0, 7.2, 5.5);
const SKY_COLOR: vec3f = vec3f(3.0, 3.5, 5.5);
const SUN_ANGLE: f32 = 0.03;
const COS_SUN_ANGLE: f32 = 0.99955;
const SUN_SOLID_ANGLE: f32 = 0.002827;
const SUN_MULT: f32 = 100.0;
const MAX_FIREFLY_LUM: f32 = 32.0;
const MAT_FLAG_THIN_TRANSMISSION: u32 = 1u;
const MAT_FLAG_DOUBLE_SIDED: u32 = 2u;
const MAT_FLAG_UNLIT: u32 = 4u;
const MAX_THIN_GLASS_PASSES: u32 = 4u;
const MAX_VOLUME_STACK: u32 = 4u;
const LIGHT_TYPE_DIRECTIONAL: u32 = 0u;
const LIGHT_TYPE_POINT: u32 = 1u;
const LIGHT_TYPE_SPOT: u32 = 2u;

// ============================================================
// RNG: PCG + spatio-temporal R2 blue noise stratification
// R2 quasi-random sequence provides low-discrepancy sampling that:
// - Spatially: nearby pixels get maximally different offsets (blue-noise-like)
// - Temporally: each frame shifts the pattern (Cranley-Patterson rotation)
// Result: noise is spatially uniform → denoiser works much better
// ============================================================
var<private> rng_state: u32;
var<private> g_sun_dir: vec3f;
var<private> g_r2_offset: vec2f;
var<private> g_sample_idx: u32;
var<private> g_alpha_dither: f32; // spatiotemporal R2 dither for alpha testing

// R2 quasi-random constants (plastic constant)
const R2_A1: f32 = 0.7548776662466927;
const R2_A2: f32 = 0.5698402909980532;

fn pcg(state: ptr<private, u32>) -> u32 {
  let s = *state;
  *state = s * 747796405u + 2891336453u;
  let word = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (word >> 22u) ^ word;
}
fn rand() -> f32 { return f32(pcg(&rng_state)) / 4294967295.0; }

// Blue-noise-like 2D sample: R2 spatial stratification + temporal rotation
// Each call returns a different R2 point (indexed by g_sample_idx)
fn rand2() -> vec2f {
  let idx = f32(g_sample_idx);
  g_sample_idx += 1u;
  return fract(vec2f(R2_A1 * idx, R2_A2 * idx) + g_r2_offset);
}

// ============================================================
// Sampling utilities
// ============================================================
fn build_onb(n: vec3f) -> mat3x3f {
  let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.999);
  let t = normalize(cross(up, n));
  let b = cross(n, t);
  return mat3x3f(t, b, n);
}

fn cosine_sample_hemisphere(normal: vec3f) -> vec3f {
  let u = rand2();
  let phi = TWO_PI * u.x;
  let sr2 = sqrt(u.y);
  let local = vec3f(cos(phi)*sr2, sin(phi)*sr2, sqrt(1.0-u.y));
  return normalize(build_onb(normal) * local);
}

fn sample_cone(axis: vec3f, cos_theta_max: f32) -> vec3f {
  let u = rand2();
  let cos_theta = 1.0 + u.x * (cos_theta_max - 1.0);
  let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
  let phi = TWO_PI * u.y;
  let local = vec3f(cos(phi)*sin_theta, sin(phi)*sin_theta, cos_theta);
  return normalize(build_onb(axis) * local);
}

fn random_in_unit_sphere() -> vec3f {
  let z = rand() * 2.0 - 1.0;
  let r = sqrt(max(0.0, 1.0 - z * z));
  let phi = TWO_PI * rand();
  let radius = pow(rand(), 1.0 / 3.0);
  return vec3f(r * cos(phi), r * sin(phi), z) * radius;
}

// ============================================================
// GGX VNDF Sampling (Heitz 2018) — ported from ignis-rt
// ============================================================
fn sample_ggx_vndf(u: vec2f, V: vec3f, N: vec3f, alpha: f32) -> vec3f {
  let onb = build_onb(N);
  let T1 = onb[0]; let T2 = onb[1];
  // Transform V to local space, stretch by alpha
  let Vh = normalize(vec3f(dot(V, T1) * alpha, dot(V, T2) * alpha, dot(V, N)));
  // ONB in hemisphere space
  let lensq = Vh.x * Vh.x + Vh.y * Vh.y;
  let T1h = select(vec3f(1.0, 0.0, 0.0), vec3f(-Vh.y, Vh.x, 0.0) / sqrt(lensq), lensq > 0.0);
  let T2h = cross(Vh, T1h);
  // Uniform disk sample
  let r = sqrt(u.x);
  let phi = TWO_PI * u.y;
  var t1 = r * cos(phi);
  var t2 = r * sin(phi);
  let s = 0.5 * (1.0 + Vh.z);
  t2 = (1.0 - s) * sqrt(max(0.0, 1.0 - t1 * t1)) + s * t2;
  // Microfacet normal in hemisphere, transform back
  let Nh = t1 * T1h + t2 * T2h + sqrt(max(0.0, 1.0 - t1*t1 - t2*t2)) * Vh;
  return normalize(T1 * (alpha * Nh.x) + T2 * (alpha * Nh.y) + N * max(0.0, Nh.z));
}

// ============================================================
// PBR BRDF — ported from ignis-rt (GGX Cook-Torrance)
// ============================================================
// SHaRC helpers (inline for path tracer access)
// ============================================================
fn sharc_jenkins(a_in: u32) -> u32 {
  var a = a_in;
  a=(a+0x7ed55d16u)+(a<<12u); a=(a^0xc761c23cu)^(a>>19u);
  a=(a+0x165667b1u)+(a<<5u);  a=(a+0xd3a2646cu)^(a<<9u);
  a=(a+0xfd7046c5u)+(a<<3u);  a=(a^0xb55a4f09u)^(a>>16u);
  return a;
}

fn sharc_make_key(wp: vec3f, n: vec3f) -> u32 {
  let dist = length(wp - sharc_params.camera_pos);
  let level = u32(max(log2(max(dist * sharc_params.scene_scale, 1.0)), 0.0));
  let vs = pow(2.0, f32(level)) / max(sharc_params.scene_scale, 1e-6);
  let gp = vec3i(floor(wp / vs));
  let an = abs(n);
  var nb = 0u;
  if an.x > an.y && an.x > an.z { nb = select(1u, 0u, n.x > 0.0); }
  else if an.y > an.z { nb = select(3u, 2u, n.y > 0.0); }
  else { nb = select(5u, 4u, n.z > 0.0); }
  let key = (u32(gp.x) & 0x1FFu) | ((u32(gp.y) & 0x1FFu) << 9u)
          | ((u32(gp.z) & 0x1FFu) << 18u) | ((nb & 0x7u) << 27u) | ((level & 0x3u) << 30u);
  return select(key, 1u, key == 0u);
}

// keys_accum layout: [0..cap) = keys, [cap..cap*7) = accum R,G,B,count,Dx,Dy,Dz per slot
// resolved layout: [0..cap*7) = resolved R,G,B,samples|stale,Dx,Dy,Dz per slot

fn sharc_find_slot(key: u32) -> u32 {
  let base = sharc_jenkins(key) % sharc_params.capacity;
  // Short probe (4 instead of 8) + check resolved sample count to skip empty
  for (var i = 0u; i < 4u; i++) {
    let slot = (base + i) % sharc_params.capacity;
    // Read from resolved (non-atomic, fast) to check if slot has data
    let rSamples = sharc_resolved[slot * 7u + 3u] & 0xFFFFu;
    if rSamples == 0u { return 0xFFFFFFFFu; } // empty chain, stop
    // Verify key match via atomic (only if slot has data)
    let stored = atomicLoad(&sharc_keys_accum[slot]);
    if stored == key { return slot; }
  }
  return 0xFFFFFFFFu;
}

fn sharc_insert_slot(key: u32) -> u32 {
  let base = sharc_jenkins(key) % sharc_params.capacity;
  for (var i = 0u; i < 4u; i++) {
    let slot = (base + i) % sharc_params.capacity;
    let ex = atomicCompareExchangeWeak(&sharc_keys_accum[slot], 0u, key);
    if ex.exchanged || ex.old_value == key { return slot; }
  }
  return 0xFFFFFFFFu;
}

fn sharc_read_cached(wp: vec3f, n: vec3f) -> vec3f {
  let key = sharc_make_key(wp, n);
  let slot = sharc_find_slot(key);
  if slot == 0xFFFFFFFFu { return vec3f(0.0); }
  let rBase = slot * 7u;
  let samples = f32(sharc_resolved[rBase + 3u] & 0xFFFFu);
  if samples < 1.0 { return vec3f(0.0); }
  return vec3f(
    bitcast<f32>(sharc_resolved[rBase]),
    bitcast<f32>(sharc_resolved[rBase + 1u]),
    bitcast<f32>(sharc_resolved[rBase + 2u])
  );
}

// Store radiance + incoming light direction for path guiding (L1 SH)
// Direction encoded as offset u32 for safe atomic accumulation of signed values:
// val = (dir_component * lum + lum) * GUIDE_SCALE → always positive
const GUIDE_SCALE: f32 = 10000.0;

// Atomic RGB+count accumulation helper (shared by both store functions)
fn sharc_accum_rgbs(aBase: u32, s: vec3f) {
  if u32(s.x) > 0u { atomicAdd(&sharc_keys_accum[aBase], u32(s.x)); }
  if u32(s.y) > 0u { atomicAdd(&sharc_keys_accum[aBase + 1u], u32(s.y)); }
  if u32(s.z) > 0u { atomicAdd(&sharc_keys_accum[aBase + 2u], u32(s.z)); }
  atomicAdd(&sharc_keys_accum[aBase + 3u], 1u);
}

fn sharc_store_radiance(wp: vec3f, n: vec3f, rad: vec3f) {
  let key = sharc_make_key(wp, n);
  let slot = sharc_insert_slot(key);
  if slot == 0xFFFFFFFFu { return; }
  let s = max(rad * 1000.0, vec3f(0.0));
  let aBase = sharc_params.capacity + slot * 7u;
  sharc_accum_rgbs(aBase, s);
}

// Store with direction (for indirect bounces — records where light came from)
fn sharc_store_radiance_dir(wp: vec3f, n: vec3f, rad: vec3f, incoming_dir: vec3f) {
  let key = sharc_make_key(wp, n);
  let slot = sharc_insert_slot(key);
  if slot == 0xFFFFFFFFu { return; }
  let s = max(rad * 1000.0, vec3f(0.0));
  let aBase = sharc_params.capacity + slot * 7u;
  sharc_accum_rgbs(aBase, s);
  // Accumulate luminance-weighted direction (offset encoding for signed→unsigned)
  let lum = dot(rad, vec3f(0.2126, 0.7152, 0.0722));
  if lum > 0.001 {
    let d = incoming_dir; // direction FROM which light arrives
    atomicAdd(&sharc_keys_accum[aBase + 4u], u32((d.x * lum + lum) * GUIDE_SCALE));
    atomicAdd(&sharc_keys_accum[aBase + 5u], u32((d.y * lum + lum) * GUIDE_SCALE));
    atomicAdd(&sharc_keys_accum[aBase + 6u], u32((d.z * lum + lum) * GUIDE_SCALE));
  }
}

// Read guide direction from resolved cache
fn sharc_read_guide(wp: vec3f, n: vec3f) -> vec4f {
  // Returns xyz = dominant direction, w = concentration (0=no info, 1=strong)
  let key = sharc_make_key(wp, n);
  let slot = sharc_find_slot(key);
  if slot == 0xFFFFFFFFu { return vec4f(0.0); }
  let rBase = slot * 7u;
  let samples = f32(sharc_resolved[rBase + 3u] & 0xFFFFu);
  if samples < 4.0 { return vec4f(0.0); } // need minimum samples for reliable direction
  // Reconstruct direction from resolved L1 SH
  let dx = bitcast<f32>(sharc_resolved[rBase + 4u]);
  let dy = bitcast<f32>(sharc_resolved[rBase + 5u]);
  let dz = bitcast<f32>(sharc_resolved[rBase + 6u]);
  let dir_len = length(vec3f(dx, dy, dz));
  if dir_len < 0.01 { return vec4f(0.0); } // isotropic, no dominant direction
  let dir = vec3f(dx, dy, dz) / dir_len;
  let concentration = min(dir_len, 1.0); // 0 = diffuse light, 1 = directional
  return vec4f(dir, concentration);
}

// ============================================================
// ReSTIR GI — Temporal Radiance Reuse (Talbot et al.)
// ============================================================
struct GIReservoir {
  position: vec3f, normal: vec3f, radiance: vec3f,
  weight_sum: f32, M: f32, hit_dist: f32, age: f32,
};

fn empty_reservoir() -> GIReservoir {
  return GIReservoir(vec3f(0.0), vec3f(0.0), vec3f(0.0), 0.0, 0.0, 0.0, 0.0);
}

// Octahedral normal encoding (3D → 2 floats)
fn oct_encode(n: vec3f) -> vec2f {
  let s = abs(n.x) + abs(n.y) + abs(n.z);
  var o = n.xy / s;
  if n.z < 0.0 { o = (1.0 - abs(o.yx)) * select(vec2f(-1.0), vec2f(1.0), o >= vec2f(0.0)); }
  return o * 0.5 + 0.5;
}

fn oct_decode(e: vec2f) -> vec3f {
  let f = e * 2.0 - 1.0;
  var n = vec3f(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
  if n.z < 0.0 { n = vec3f((1.0 - abs(n.yx)) * select(vec2f(-1.0), vec2f(1.0), n.xy >= vec2f(0.0)), n.z); }
  return normalize(n);
}

fn write_reservoir(idx: u32, r: GIReservoir) {
  let base = idx * 3u;
  restir_curr[base]     = vec4f(r.position, r.weight_sum);
  restir_curr[base + 1u] = vec4f(r.radiance, r.M);
  let on = oct_encode(r.normal);
  restir_curr[base + 2u] = vec4f(on.x, on.y, r.hit_dist, r.age);
}

fn read_reservoir_prev(idx: u32) -> GIReservoir {
  let base = idx * 3u;
  let d0 = restir_prev[base];
  let d1 = restir_prev[base + 1u];
  let d2 = restir_prev[base + 2u];
  return GIReservoir(d0.xyz, oct_decode(d2.xy), d1.xyz, d0.w, d1.w, d2.z, d2.w);
}

// Target PDF: luminance × cosine-weighted importance
fn gi_target_pdf(primary_normal: vec3f, primary_pos: vec3f, s: GIReservoir) -> f32 {
  let dir = s.position - primary_pos;
  let dist = length(dir);
  if dist < 0.001 { return 0.0; }
  let cos_theta = max(dot(primary_normal, dir / dist), 0.0);
  let lum = dot(s.radiance, vec3f(0.2126, 0.7152, 0.0722));
  return lum * cos_theta;
}

// WRS update: accept sample with probability weight/weightSum
fn reservoir_update(r: ptr<function, GIReservoir>, s: GIReservoir, weight: f32) {
  (*r).weight_sum += weight;
  (*r).M += 1.0;
  if rand() * (*r).weight_sum < weight {
    (*r).position = s.position;
    (*r).normal = s.normal;
    (*r).radiance = s.radiance;
    (*r).hit_dist = s.hit_dist;
    (*r).age = s.age;
  }
}

// Merge source reservoir into destination
fn reservoir_merge(dest: ptr<function, GIReservoir>, src: GIReservoir, target_pdf: f32) {
  let weight = target_pdf * src.M;
  let old_M = (*dest).M;
  reservoir_update(dest, src, weight);
  (*dest).M = old_M + src.M;
}

// Reproject current pixel to previous frame UV
fn restir_reproject(world_pos: vec3f) -> vec2f {
  let local = world_pos - uniforms.prev_pos;
  let z = dot(local, uniforms.prev_forward);
  if z <= 0.0 { return vec2f(-1.0); }
  let x = dot(local, uniforms.prev_right);
  let y = dot(local, uniforms.prev_up);
  let ndcx = x / (z * uniforms.aspect * uniforms.fov_factor);
  let ndcy = y / (z * uniforms.fov_factor);
  return vec2f(ndcx * 0.5 + 0.5, ndcy * 0.5 + 0.5);
}

// ============================================================
fn fresnel_dielectric(cosi: f32, eta: f32) -> f32 {
  let c = abs(cosi);
  let g = eta * eta - 1.0 + c * c;
  if g > 0.0 {
    let sg = sqrt(g);
    let A = (sg - c) / (sg + c);
    let B = (c * (sg + c) - 1.0) / (c * (sg - c) + 1.0);
    return 0.5 * A * A * (1.0 + B * B);
  }
  return 1.0; // TIR
}

fn fresnel_real(cosTheta: f32, F0: vec3f) -> vec3f {
  let avgF0 = max((F0.r + F0.g + F0.b) / 3.0, 1e-6);
  let sqrtF0 = sqrt(clamp(avgF0, 0.0, 0.999));
  let eta = (1.0 + sqrtF0) / max(1.0 - sqrtF0, 1e-6);
  let F0_real = pow((eta - 1.0) / (eta + 1.0), 2.0);
  let F_real = fresnel_dielectric(cosTheta, eta);
  let s = clamp((F_real - F0_real) / max(1.0 - F0_real, 1e-6), 0.0, 1.0);
  return mix(F0, vec3f(1.0), s);
}

fn mat_type(mat: Material) -> u32 { return u32(mat.d0.w + 0.5); }
fn mat_base_color_factor(mat: Material) -> vec3f { return mat.d0.xyz; }
fn mat_emissive_factor(mat: Material) -> vec3f { return mat.d1.xyz; }
fn mat_roughness_factor(mat: Material) -> f32 { return mat.d1.w; }
fn mat_metallic_factor(mat: Material) -> f32 { return mat.d2.x; }
fn decode_tex_index(encoded: f32) -> i32 {
  if encoded < 0.0 { return -1; }
  return i32(encoded + 0.5);
}
fn mat_base_tex(mat: Material) -> i32 { return decode_tex_index(mat.d2.y); }
fn mat_mr_tex(mat: Material) -> i32 { return decode_tex_index(mat.d2.z); }
fn mat_normal_tex(mat: Material) -> i32 { return decode_tex_index(mat.d2.w); }
fn mat_alpha_mode(mat: Material) -> u32 { return u32(mat.d3.x + 0.5); }
fn mat_alpha_cutoff(mat: Material) -> f32 { return mat.d3.y; }
fn mat_ior(mat: Material) -> f32 { return max(mat.d3.z, 1.0); }
fn mat_emission_strength(mat: Material) -> f32 { return max(mat.d3.w, 0.0); }
fn mat_transmission_factor(mat: Material) -> f32 { return clamp(mat.d4.x, 0.0, 1.0); }
fn mat_transmission_tex(mat: Material) -> i32 { return decode_tex_index(mat.d4.y); }
fn mat_thickness_factor(mat: Material) -> f32 { return max(mat.d4.z, 0.0); }
fn mat_flags(mat: Material) -> u32 { return u32(mat.d4.w + 0.5); }
fn mat_base_alpha(mat: Material) -> f32 { return clamp(mat.d5.x, 0.0, 1.0); }
fn mat_base_texcoord(mat: Material) -> u32 { return u32(mat.d5.y + 0.5); }
fn mat_mr_texcoord(mat: Material) -> u32 { return u32(mat.d5.z + 0.5); }
fn mat_normal_texcoord(mat: Material) -> u32 { return u32(mat.d5.w + 0.5); }
fn mat_normal_scale(mat: Material) -> f32 { return mat.d6.x; }
fn mat_emissive_tex(mat: Material) -> i32 { return decode_tex_index(mat.d6.y); }
fn mat_occlusion_tex(mat: Material) -> i32 { return decode_tex_index(mat.d6.z); }
fn mat_thickness_tex(mat: Material) -> i32 { return decode_tex_index(mat.d6.w); }
fn mat_transmission_texcoord(mat: Material) -> u32 { return u32(mat.d7.x + 0.5); }
fn mat_emissive_texcoord(mat: Material) -> u32 { return u32(mat.d7.y + 0.5); }
fn mat_occlusion_texcoord(mat: Material) -> u32 { return u32(mat.d7.z + 0.5); }
fn mat_thickness_texcoord(mat: Material) -> u32 { return u32(mat.d7.w + 0.5); }
fn mat_occlusion_strength(mat: Material) -> f32 { return clamp(mat.d8.x, 0.0, 1.0); }
fn mat_attenuation_distance(mat: Material) -> f32 { return max(mat.d8.y, 1e-6); }
fn mat_attenuation_color(mat: Material) -> vec3f { return vec3f(mat.d8.z, mat.d8.w, mat.d9.x); }
fn mat_specular_factor(mat: Material) -> f32 { return clamp(mat.d10.x, 0.0, 1.0); }
fn mat_specular_tex(mat: Material) -> i32 { return decode_tex_index(mat.d10.y); }
fn mat_specular_color_tex(mat: Material) -> i32 { return decode_tex_index(mat.d10.z); }
fn mat_specular_color_factor(mat: Material) -> vec3f { return vec3f(mat.d10.w, mat.d11.x, mat.d11.y); }
fn mat_specular_texcoord(mat: Material) -> u32 { return u32(mat.d11.z + 0.5); }
fn mat_specular_color_texcoord(mat: Material) -> u32 { return u32(mat.d11.w + 0.5); }
fn mat_clearcoat_factor(mat: Material) -> f32 { return clamp(mat.d12.x, 0.0, 1.0); }
fn mat_clearcoat_tex(mat: Material) -> i32 { return decode_tex_index(mat.d12.y); }
fn mat_clearcoat_roughness_factor(mat: Material) -> f32 { return clamp(mat.d12.z, 0.0, 1.0); }
fn mat_clearcoat_rough_tex(mat: Material) -> i32 { return decode_tex_index(mat.d12.w); }
fn mat_clearcoat_texcoord(mat: Material) -> u32 { return u32(mat.d13.x + 0.5); }
fn mat_clearcoat_rough_texcoord(mat: Material) -> u32 { return u32(mat.d13.y + 0.5); }
fn mat_clearcoat_normal_tex(mat: Material) -> i32 { return decode_tex_index(mat.d13.z); }
fn mat_clearcoat_normal_texcoord(mat: Material) -> u32 { return u32(mat.d13.w + 0.5); }
fn mat_clearcoat_normal_scale(mat: Material) -> f32 { return mat.d14.x; }
fn mat_sheen_color_factor(mat: Material) -> vec3f { return vec3f(mat.d14.y, mat.d14.z, mat.d14.w); }
fn mat_sheen_roughness_factor(mat: Material) -> f32 { return clamp(mat.d15.x, 0.0, 1.0); }
fn mat_sheen_color_tex(mat: Material) -> i32 { return decode_tex_index(mat.d15.y); }
fn mat_sheen_rough_tex(mat: Material) -> i32 { return decode_tex_index(mat.d15.z); }
fn mat_sheen_color_texcoord(mat: Material) -> u32 { return u32(mat.d15.w + 0.5); }
fn mat_sheen_rough_texcoord(mat: Material) -> u32 { return u32(mat.d16.x + 0.5); }
fn mat_anisotropy_strength(mat: Material) -> f32 { return clamp(mat.d16.y, 0.0, 1.0); }
fn mat_anisotropy_rotation(mat: Material) -> f32 { return mat.d16.z; }
fn mat_anisotropy_tex(mat: Material) -> i32 { return decode_tex_index(mat.d16.w); }
fn mat_anisotropy_texcoord(mat: Material) -> u32 { return u32(mat.d17.x + 0.5); }
fn mat_iridescence_factor(mat: Material) -> f32 { return clamp(mat.d17.y, 0.0, 1.0); }
fn mat_iridescence_tex(mat: Material) -> i32 { return decode_tex_index(mat.d17.z); }
fn mat_iridescence_ior(mat: Material) -> f32 { return max(mat.d17.w, 1.0); }
fn mat_iridescence_thickness_min(mat: Material) -> f32 { return max(mat.d18.x, 0.0); }
fn mat_iridescence_thickness_max(mat: Material) -> f32 { return max(mat.d18.y, mat.d18.x); }
fn mat_iridescence_thickness_tex(mat: Material) -> i32 { return decode_tex_index(mat.d18.z); }
fn mat_iridescence_texcoord(mat: Material) -> u32 { return u32(mat.d18.w + 0.5); }
fn mat_iridescence_thickness_texcoord(mat: Material) -> u32 { return u32(mat.d19.x + 0.5); }
fn mat_dispersion(mat: Material) -> f32 { return max(mat.d19.y, 0.0); }

fn material_is_unlit(mat: Material) -> bool {
  return (mat_flags(mat) & MAT_FLAG_UNLIT) != 0u || mat_type(mat) == 1u;
}

fn material_has_transmission(mat: Material) -> bool {
  return mat_type(mat) == 3u && (mat_transmission_factor(mat) > 0.001 || mat_transmission_tex(mat) >= 0);
}

fn material_is_thin(mat: Material) -> bool {
  return (mat_flags(mat) & MAT_FLAG_THIN_TRANSMISSION) != 0u;
}

fn material_has_volume(mat: Material) -> bool {
  return material_has_transmission(mat) && !material_is_thin(mat);
}

fn material_is_double_sided(mat: Material) -> bool {
  return (mat_flags(mat) & MAT_FLAG_DOUBLE_SIDED) != 0u || material_has_volume(mat);
}

fn material_needs_attenuation(mat: Material) -> bool {
  if !material_has_volume(mat) { return false; }
  let att_dist = mat_attenuation_distance(mat);
  if att_dist >= 1e29 { return false; }
  let att = clamp(mat_attenuation_color(mat), vec3f(0.0), vec3f(1.0));
  return any(att < vec3f(0.9999));
}

fn ggx_d(NdotH: f32, alpha: f32) -> f32 {
  let a2 = alpha * alpha;
  let d = NdotH * NdotH * (a2 - 1.0) + 1.0;
  return a2 / (PI * d * d);
}

fn smith_g1(NdotV: f32, alpha: f32) -> f32 {
  let a2 = alpha * alpha;
  return 2.0 * NdotV / (NdotV + sqrt(a2 + (1.0 - a2) * NdotV * NdotV));
}

// Split BRDF evaluation: returns demodulated diffuse irradiance + specular radiance separately
struct BRDFSplit { diffuse: vec3f, specular: vec3f, };

fn ggx_d_anisotropic(NdotH: f32, TdotH: f32, BdotH: f32, at: f32, ab: f32) -> f32 {
  let a2 = at * ab;
  let f = vec3f(ab * TdotH, at * BdotH, a2 * NdotH);
  let denom = dot(f, f);
  if denom <= 1e-8 { return 0.0; }
  let w2 = a2 / denom;
  return a2 * w2 * w2 * INV_PI;
}

fn ggx_v_anisotropic(NdotL: f32, NdotV: f32, BdotV: f32, TdotV: f32, TdotL: f32, BdotL: f32, at: f32, ab: f32) -> f32 {
  let ggxv = NdotL * length(vec3f(at * TdotV, ab * BdotV, NdotV));
  let ggxl = NdotV * length(vec3f(at * TdotL, ab * BdotL, NdotL));
  let denom = ggxv + ggxl;
  if denom <= 1e-6 { return 0.0; }
  return clamp(0.5 / denom, 0.0, 1.0);
}

fn sheen_charlie_distribution(NdotH: f32, sheen_roughness: f32) -> f32 {
  let alpha_g = max(sheen_roughness * sheen_roughness, 1e-4);
  let inv_r = 1.0 / alpha_g;
  let sin2h = max(1.0 - NdotH * NdotH, 0.0);
  return (2.0 + inv_r) * pow(sin2h, 0.5 * inv_r) / (2.0 * PI);
}

fn sheen_visibility_ashikhmin(NdotL: f32, NdotV: f32) -> f32 {
  return 1.0 / max(4.0 * (NdotL + NdotV - NdotL * NdotV), 1e-4);
}

struct SurfaceEval {
  normal: vec3f,
  roughness: f32,
  base_color: vec3f,
  metallic: f32,
  transmission: f32,
  dielectric_f0: vec3f,
  dielectric_f90: f32,
  clearcoat_normal: vec3f,
  clearcoat: f32,
  clearcoat_roughness: f32,
  sheen_color: vec3f,
  sheen_roughness: f32,
  anisotropic_t: vec3f,
  anisotropy: f32,
  anisotropic_b: vec3f,
  _pad: f32,
};

fn build_surface_eval(td: vec4u, mat: Material, normal: vec3f, base_color: vec3f, roughness: f32, metallic: f32, transmission: f32, uv0: vec2f, uv1: vec2f, uv2: vec2f, uv3: vec2f) -> SurfaceEval {
  let specular_params = sample_specular_params(mat, uv0, uv1, uv2, uv3);
  let ior = max(mat_ior(mat), 1.0);
  let base_f0 = pow((ior - 1.0) / max(ior + 1.0, 1e-6), 2.0);
  let dielectric_f0 = min(vec3f(base_f0) * specular_params.rgb, vec3f(1.0)) * specular_params.a;
  let dielectric_f90 = specular_params.a;

  let clearcoat_params = sample_clearcoat_params(mat, uv0, uv1, uv2, uv3);
  var clearcoat_normal = normal;
  if clearcoat_params.x > 0.001 {
    clearcoat_normal = apply_detail_normal(
      mat_clearcoat_normal_tex(mat),
      mat_clearcoat_normal_texcoord(mat),
      mat_clearcoat_normal_scale(mat),
      td, normal, uv0, uv1, uv2, uv3
    );
  }

  let sheen_params = sample_sheen_params(mat, uv0, uv1, uv2, uv3);
  let anisotropy_params = sample_anisotropy_params(mat, uv0, uv1, uv2, uv3);
  let anisotropy_uv = select(mat_anisotropy_texcoord(mat), mat_normal_texcoord(mat), mat_anisotropy_tex(mat) < 0);
  let tbn = tangent_basis_from_texcoord(td, normal, anisotropy_uv);
  let anisotropic_t = normalize(tbn * vec3f(anisotropy_params.xy, 0.0));
  let anisotropic_b = normalize(cross(normal, anisotropic_t));

  return SurfaceEval(
    normal,
    roughness,
    base_color,
    metallic,
    transmission,
    dielectric_f0,
    dielectric_f90,
    clearcoat_normal,
    clearcoat_params.x,
    max(clearcoat_params.y, 0.001),
    sheen_params.rgb,
    sheen_params.a,
    anisotropic_t,
    anisotropy_params.z,
    anisotropic_b,
    0.0
  );
}

fn build_surface_eval_basic(mat: Material, normal: vec3f, base_color: vec3f, roughness: f32, metallic: f32, transmission: f32) -> SurfaceEval {
  let ior = max(mat_ior(mat), 1.0);
  let base_f0 = pow((ior - 1.0) / max(ior + 1.0, 1e-6), 2.0);
  let onb = build_onb(normal);
  return SurfaceEval(
    normal,
    roughness,
    base_color,
    metallic,
    transmission,
    vec3f(base_f0),
    1.0,
    normal,
    0.0,
    0.0,
    vec3f(0.0),
    0.0,
    onb[0],
    0.0,
    onb[1],
    0.0
  );
}

fn eval_surface_split(surface: SurfaceEval, V: vec3f, L: vec3f) -> BRDFSplit {
  let N = surface.normal;
  let NdotL = max(dot(N, L), 0.0);
  let NdotV = max(dot(N, V), 0.001);
  if NdotL <= 0.0 { return BRDFSplit(vec3f(0.0), vec3f(0.0)); }

  let H = normalize(V + L);
  let NdotH = max(dot(N, H), 0.0);
  let VdotH = max(dot(V, H), 0.0);
  let alpha = max(surface.roughness * surface.roughness, 0.001);
  let dielectric_f90 = vec3f(surface.dielectric_f90);
  let F0 = mix(surface.dielectric_f0, surface.base_color, surface.metallic);
  let F90 = mix(dielectric_f90, vec3f(1.0), surface.metallic);
  let F = fresnel_schlick_vec(F0, F90, VdotH);

  var D = ggx_d(NdotH, alpha);
  var Vterm = smith_g1(NdotL, alpha) * smith_g1(NdotV, alpha) / max(4.0 * NdotL * NdotV, 1e-4);
  if surface.anisotropy > 0.001 {
    let at = mix(alpha, 1.0, surface.anisotropy * surface.anisotropy);
    let ab = alpha;
    let TdotV = dot(surface.anisotropic_t, V);
    let BdotV = dot(surface.anisotropic_b, V);
    let TdotL = dot(surface.anisotropic_t, L);
    let BdotL = dot(surface.anisotropic_b, L);
    let TdotH = dot(surface.anisotropic_t, H);
    let BdotH = dot(surface.anisotropic_b, H);
    D = ggx_d_anisotropic(NdotH, TdotH, BdotH, at, ab);
    Vterm = ggx_v_anisotropic(NdotL, NdotV, BdotV, TdotV, TdotL, BdotL, at, ab);
  }

  let specular = F * D * Vterm * NdotL;
  let diffuse_weight = (1.0 - surface.metallic) * (1.0 - surface.transmission) * (1.0 - max_component(F));
  var diffuse = vec3f(diffuse_weight * INV_PI) * NdotL;
  var spec_total = specular;

  if max_component(surface.sheen_color) > 0.001 {
    let sheen_d = sheen_charlie_distribution(NdotH, max(surface.sheen_roughness, 0.001));
    let sheen_v = sheen_visibility_ashikhmin(NdotL, NdotV);
    spec_total += surface.sheen_color * sheen_d * sheen_v * NdotL;
  }

  if surface.clearcoat > 0.001 {
    let Nc = surface.clearcoat_normal;
    let NcdotL = max(dot(Nc, L), 0.0);
    let NcdotV = max(dot(Nc, V), 0.001);
    if NcdotL > 0.0 {
      let Hc = normalize(V + L);
      let NcdotH = max(dot(Nc, Hc), 0.0);
      let VdotHc = max(dot(V, Hc), 0.0);
      let alpha_c = max(surface.clearcoat_roughness * surface.clearcoat_roughness, 0.001);
      let Dc = ggx_d(NcdotH, alpha_c);
      let Vc = smith_g1(NcdotL, alpha_c) * smith_g1(NcdotV, alpha_c) / max(4.0 * NcdotL * NcdotV, 1e-4);
      let layer_f = surface.clearcoat * fresnel_schlick_scalar(0.04, 1.0, abs(dot(Nc, V)));
      let Fc = fresnel_schlick_vec(vec3f(0.04), vec3f(1.0), VdotHc);
      diffuse *= max(1.0 - layer_f, 0.0);
      spec_total *= max(1.0 - layer_f, 0.0);
      spec_total += layer_f * Fc * Dc * Vc * NcdotL;
    }
  }

  return BRDFSplit(diffuse, spec_total);
}

fn eval_surface_total(surface: SurfaceEval, V: vec3f, L: vec3f) -> vec3f {
  let split = eval_surface_split(surface, V, L);
  return split.diffuse * surface.base_color + split.specular;
}

fn punctual_distance_attenuation(dist: f32, range: f32) -> f32 {
  let dist2 = max(dist * dist, 1e-4);
  if range <= 0.0 { return 1.0 / dist2; }
  let falloff = clamp(1.0 - pow(dist / range, 4.0), 0.0, 1.0);
  return falloff / dist2;
}

fn sample_punctual_nee_split(pos: vec3f, origin: vec3f, V: vec3f, surface: SurfaceEval) -> BRDFSplit {
  var result = BRDFSplit(vec3f(0.0), vec3f(0.0));
  for (var li = 0u; li < uniforms.light_count; li++) {
    let light = uniforms.lights[li];
    let light_type = u32(light.params.x + 0.5);
    var L = vec3f(0.0);
    var max_t = 1e30;
    var light_radiance = vec3f(0.0);

    if light_type == LIGHT_TYPE_DIRECTIONAL {
      L = normalize(-light.dir_inner.xyz);
      if dot(surface.normal, L) <= 0.0 { continue; }
      if trace_shadow(origin, L, 1e6) { continue; }
      light_radiance = light.color_intensity.rgb * light.color_intensity.w;
    } else {
      let to_light = light.pos_range.xyz - pos;
      let dist = length(to_light);
      if dist <= 1e-4 { continue; }
      L = to_light / dist;
      if dot(surface.normal, L) <= 0.0 { continue; }
      max_t = dist - 0.01;
      if max_t <= 0.0 { continue; }
      var attenuation = punctual_distance_attenuation(dist, light.pos_range.w);
      if light_type == LIGHT_TYPE_SPOT {
        let cd = dot(normalize(light.dir_inner.xyz), -L);
        let angle_scale = 1.0 / max(0.001, light.dir_inner.w - light.params.y);
        let angle_offset = -light.params.y * angle_scale;
        let angular = clamp(cd * angle_scale + angle_offset, 0.0, 1.0);
        attenuation *= angular * angular;
      }
      if attenuation <= 0.0 || trace_shadow(origin, L, max_t) { continue; }
      light_radiance = light.color_intensity.rgb * light.color_intensity.w * attenuation;
    }

    let brdf = eval_surface_split(surface, V, L);
    result.diffuse += brdf.diffuse * light_radiance;
    result.specular += brdf.specular * light_radiance;
  }
  return result;
}

fn sample_sun_nee_split(pos: vec3f, normal: vec3f, V: vec3f, td: vec4u, mat: Material, uv0: vec2f, uv1: vec2f, uv2: vec2f, uv3: vec2f, baseColor: vec3f, roughness: f32, metallic: f32, transmission: f32) -> BRDFSplit {
  let origin = pos + normal * BIAS;
  let surface = build_surface_eval(td, mat, normal, baseColor, roughness, metallic, transmission, uv0, uv1, uv2, uv3);
  var result_split = BRDFSplit(vec3f(0.0), vec3f(0.0));

  // Sun NEE
  if uniforms.sun_enabled != 0u {
    var shadow_val = 0.0;
    let L1 = sample_cone(g_sun_dir, COS_SUN_ANGLE);
    let L2 = sample_cone(g_sun_dir, COS_SUN_ANGLE);
    if !trace_shadow(origin, L1, 50.0) { shadow_val += 0.5; }
    if !trace_shadow(origin, L2, 50.0) { shadow_val += 0.5; }
    if shadow_val > 0.0 {
      let L = normalize(L1 + L2);
      if dot(normal, L) > 0.0 {
        let brdf = eval_surface_split(surface, V, L);
        let light = SUN_COLOR * SUN_MULT * SUN_SOLID_ANGLE * shadow_val;
        result_split.diffuse += brdf.diffuse * light;
        result_split.specular += brdf.specular * light;
      }
    }
  }

  let punctual_split = sample_punctual_nee_split(pos, origin, V, surface);
  result_split.diffuse += punctual_split.diffuse;
  result_split.specular += punctual_split.specular;

  // NEE to emissive triangles (CDF importance sampling + MIS, split)
  if uniforms.emissive_tri_count > 0u {
    let rnd = rand();
    var lo = 0u; var hi = uniforms.emissive_tri_count - 1u;
    while lo < hi {
      let mid = (lo + hi) / 2u;
      if rnd <= emissive_tris[mid].z { hi = mid; } else { lo = mid + 1u; }
    }
    let etri = emissive_tris[lo];
    let tri_idx = bitcast<u32>(etri.x);
    let etd = tri_data[tri_idx];
    let emat = material_buf[etd.w];
    let ev0 = vertices[etd.x].xyz;
    let ev1 = vertices[etd.y].xyz;
    let ev2 = vertices[etd.z].xyz;
    let earea = etri.y;
    var prevCdf = 0.0;
    if lo > 0u { prevCdf = emissive_tris[lo - 1u].z; }
    let triProb = etri.z - prevCdf;
    var eu = rand(); var ev_r = rand();
    if eu + ev_r > 1.0 { eu = 1.0 - eu; ev_r = 1.0 - ev_r; }
    let ew = 1.0 - eu - ev_r;
    let epos = ev0 * (1.0 - eu - ev_r) + ev1 * eu + ev2 * ev_r;
    let enormal = normalize(cross(ev1 - ev0, ev2 - ev0));
    let euv0 = get_uv0(etd, ew, eu, ev_r);
    let euv1 = get_uv1(etd, ew, eu, ev_r);
    let euv2 = get_uv2(etd, ew, eu, ev_r);
    let euv3 = get_uv3(etd, ew, eu, ev_r);
    let eemission = sample_emissive(emat, euv0, euv1, euv2, euv3) * alpha_coverage_factor(emat, euv0, euv1, euv2, euv3);
    let eto = epos - pos;
    let edist = length(eto);
    let edir = eto / max(edist, 1e-6);
    let endotl = dot(normal, edir);
    let ecos_theta = dot(-edir, enormal);
    if endotl > 0.0 && ecos_theta > 0.0 && edist > 0.01 && earea > 1e-6 && dot(eemission, vec3f(1.0)) > 1e-6 {
      if !trace_shadow_skip_mat(origin, edir, edist - 0.01, etd.w) {
        let ebrdf = eval_surface_split(surface, V, edir);
        let eradiance = eemission * ecos_theta / (edist * edist);
        let light_pdf = triProb / earea;
        let sa_pdf = light_pdf * edist * edist / ecos_theta;
        let bsdf_pdf = endotl * INV_PI;
        let mis_w = (sa_pdf * sa_pdf) / (sa_pdf * sa_pdf + bsdf_pdf * bsdf_pdf + 1e-8);
        let escale = eradiance * mis_w / max(light_pdf, 1e-6);
        result_split.diffuse += ebrdf.diffuse * escale;
        result_split.specular += ebrdf.specular * escale;
      }
    }
  }

  return result_split;
}

fn sample_sun_nee(pos: vec3f, normal: vec3f, V: vec3f, td: vec4u, mat: Material, uv0: vec2f, uv1: vec2f, uv2: vec2f, uv3: vec2f, baseColor: vec3f, roughness: f32, metallic: f32, transmission: f32) -> vec3f {
  let split = sample_sun_nee_split(pos, normal, V, td, mat, uv0, uv1, uv2, uv3, baseColor, roughness, metallic, transmission);
  return split.diffuse * baseColor + split.specular;
}

fn sample_scene_nee_basic(pos: vec3f, normal: vec3f, V: vec3f, surface: SurfaceEval) -> vec3f {
  let origin = pos + normal * BIAS;
  var result_split = BRDFSplit(vec3f(0.0), vec3f(0.0));

  if uniforms.sun_enabled != 0u {
    var shadow_val = 0.0;
    let L1 = sample_cone(g_sun_dir, COS_SUN_ANGLE);
    let L2 = sample_cone(g_sun_dir, COS_SUN_ANGLE);
    if !trace_shadow(origin, L1, 50.0) { shadow_val += 0.5; }
    if !trace_shadow(origin, L2, 50.0) { shadow_val += 0.5; }
    if shadow_val > 0.0 {
      let L = normalize(L1 + L2);
      if dot(normal, L) > 0.0 {
        let brdf = eval_surface_split(surface, V, L);
        let light = SUN_COLOR * SUN_MULT * SUN_SOLID_ANGLE * shadow_val;
        result_split.diffuse += brdf.diffuse * light;
        result_split.specular += brdf.specular * light;
      }
    }
  }

  let punctual_split = sample_punctual_nee_split(pos, origin, V, surface);
  result_split.diffuse += punctual_split.diffuse;
  result_split.specular += punctual_split.specular;

  if uniforms.emissive_tri_count > 0u {
    let rnd = rand();
    var lo = 0u; var hi = uniforms.emissive_tri_count - 1u;
    while lo < hi {
      let mid = (lo + hi) / 2u;
      if rnd <= emissive_tris[mid].z { hi = mid; } else { lo = mid + 1u; }
    }
    let etri = emissive_tris[lo];
    let tri_idx = bitcast<u32>(etri.x);
    let etd = tri_data[tri_idx];
    let emat = material_buf[etd.w];
    let ev0 = vertices[etd.x].xyz;
    let ev1 = vertices[etd.y].xyz;
    let ev2 = vertices[etd.z].xyz;
    let earea = etri.y;
    var prevCdf = 0.0;
    if lo > 0u { prevCdf = emissive_tris[lo - 1u].z; }
    let triProb = etri.z - prevCdf;
    var eu = rand(); var ev_r = rand();
    if eu + ev_r > 1.0 { eu = 1.0 - eu; ev_r = 1.0 - ev_r; }
    let ew = 1.0 - eu - ev_r;
    let epos = ev0 * ew + ev1 * eu + ev2 * ev_r;
    let enormal = normalize(cross(ev1 - ev0, ev2 - ev0));
    let euv0 = get_uv0(etd, ew, eu, ev_r);
    let euv1 = get_uv1(etd, ew, eu, ev_r);
    let euv2 = get_uv2(etd, ew, eu, ev_r);
    let euv3 = get_uv3(etd, ew, eu, ev_r);
    let eemission = sample_emissive(emat, euv0, euv1, euv2, euv3) * alpha_coverage_factor(emat, euv0, euv1, euv2, euv3);
    let eto = epos - pos;
    let edist = length(eto);
    let edir = eto / max(edist, 1e-6);
    let endotl = dot(normal, edir);
    let ecos_theta = dot(-edir, enormal);
    if endotl > 0.0 && ecos_theta > 0.0 && edist > 0.01 && earea > 1e-6 && dot(eemission, vec3f(1.0)) > 1e-6 {
      if !trace_shadow_skip_mat(origin, edir, edist - 0.01, etd.w) {
        let ebrdf = eval_surface_split(surface, V, edir);
        let eradiance = eemission * ecos_theta / (edist * edist);
        let light_pdf = triProb / earea;
        let sa_pdf = light_pdf * edist * edist / ecos_theta;
        let bsdf_pdf = endotl * INV_PI;
        let mis_w = (sa_pdf * sa_pdf) / (sa_pdf * sa_pdf + bsdf_pdf * bsdf_pdf + 1e-8);
        let escale = eradiance * mis_w / max(light_pdf, 1e-6);
        result_split.diffuse += ebrdf.diffuse * escale;
        result_split.specular += ebrdf.specular * escale;
      }
    }
  }

  return result_split.diffuse * surface.base_color + result_split.specular;
}

// ============================================================
// Ray-AABB + Triangle intersection
// ============================================================
fn intersect_aabb(origin: vec3f, inv_dir: vec3f, bmin: vec3f, bmax: vec3f, t_max: f32) -> f32 {
  let t1 = (bmin - origin) * inv_dir;
  let t2 = (bmax - origin) * inv_dir;
  let tmin = max(max(min(t1.x,t2.x), min(t1.y,t2.y)), min(t1.z,t2.z));
  let tmax = min(min(max(t1.x,t2.x), max(t1.y,t2.y)), max(t1.z,t2.z));
  if tmax >= max(tmin, 0.0) && tmin < t_max { return max(tmin, 0.0); }
  return INF;
}

fn intersect_tri(origin: vec3f, dir: vec3f, v0: vec3f, v1: vec3f, v2: vec3f, t_max: f32) -> vec3f {
  let e1 = v1 - v0; let e2 = v2 - v0;
  let h = cross(dir, e2); let a = dot(e1, h);
  if abs(a) < 1e-8 { return vec3f(INF, 0.0, 0.0); }
  let f = 1.0 / a; let s = origin - v0;
  let u = f * dot(s, h);
  if u < 0.0 || u > 1.0 { return vec3f(INF, 0.0, 0.0); }
  let q = cross(s, e1); let v = f * dot(dir, q);
  if v < 0.0 || u + v > 1.0 { return vec3f(INF, 0.0, 0.0); }
  let t = f * dot(e2, q);
  if t < T_MIN || t > t_max { return vec3f(INF, 0.0, 0.0); }
  return vec3f(t, u, v);
}

// ============================================================
// BVH traversal (short stack for Adreno register pressure)
// ============================================================
fn trace_bvh(origin: vec3f, dir: vec3f) -> HitInfo {
  return trace_bvh_hint(origin, dir, INF);
}
// BVH with depth hint: start with t_max = known depth + margin
// Prunes all nodes beyond the known hit → 60-80% less traversal
fn trace_bvh_hint(origin: vec3f, dir: vec3f, t_hint: f32) -> HitInfo {
  var hit: HitInfo; hit.t = t_hint; hit.hit = false;
  let inv_dir = 1.0 / dir;
  var stk: array<u32, 16>; var sp = 0u; var cur = 0u;
  let root = bvh_nodes[0u];
  if intersect_aabb(origin, inv_dir, root.aabb_min, root.aabb_max, INF) >= INF { return hit; }
  loop {
    let nd = bvh_nodes[cur];
    if nd.tri_count > 0u {
      for (var i = 0u; i < nd.tri_count; i++) {
        let ti = nd.left_first + i; let td = tri_data[ti];
        let r = intersect_tri(origin, dir, vertices[td.x].xyz, vertices[td.y].xyz, vertices[td.z].xyz, hit.t);
        if r.x < hit.t {
          let mat = material_buf[td.w];
          let geo_normal = triangle_geo_normal(td);
          let front_face = dot(dir, geo_normal) < 0.0;
          if !material_is_double_sided(mat) && !front_face { continue; }
          let bw = 1.0 - r.y - r.z;
          let uv0 = get_uv0(td, bw, r.y, r.z);
          let uv1 = get_uv1(td, bw, r.y, r.z);
          let uv2 = get_uv2(td, bw, r.y, r.z);
          let uv3 = get_uv3(td, bw, r.y, r.z);
          if !passes_alpha_surface_hit(mat, uv0, uv1, uv2, uv3) { continue; }
          hit.t=r.x; hit.u=r.y; hit.v=r.z; hit.tri_idx=ti; hit.hit=true;
        }
      }
      if sp==0u{break;} sp--; cur=stk[sp]; continue;
    }
    let l = nd.left_first; let r = l+1u;
    let tl = intersect_aabb(origin, inv_dir, bvh_nodes[l].aabb_min, bvh_nodes[l].aabb_max, hit.t);
    let tr = intersect_aabb(origin, inv_dir, bvh_nodes[r].aabb_min, bvh_nodes[r].aabb_max, hit.t);
    if tl < tr {
      if tr < hit.t && sp < 16u { stk[sp]=r; sp++; }
      if tl < hit.t { cur=l; } else { if sp==0u{break;} sp--; cur=stk[sp]; }
    } else {
      if tl < hit.t && sp < 16u { stk[sp]=l; sp++; }
      if tr < hit.t { cur=r; } else { if sp==0u{break;} sp--; cur=stk[sp]; }
    }
  }
  return hit;
}

// Shadow with material skip (for emissive NEE — skip the emissive's own geometry)
fn trace_shadow_skip_mat(origin: vec3f, dir: vec3f, max_t: f32, skip_mat: u32) -> bool {
  let inv_dir = 1.0 / dir;
  var stk: array<u32, 6>; var sp = 0u; var cur = 0u;
  let root = bvh_nodes[0u];
  if intersect_aabb(origin, inv_dir, root.aabb_min, root.aabb_max, max_t) >= max_t { return false; }
  loop {
    let nd = bvh_nodes[cur];
    if nd.tri_count > 0u {
      for (var i = 0u; i < nd.tri_count; i++) {
        let ti = nd.left_first+i; let td = tri_data[ti];
        if td.w == skip_mat { continue; } // skip target emissive material
        let r = intersect_tri(origin, dir, vertices[td.x].xyz, vertices[td.y].xyz, vertices[td.z].xyz, max_t);
        if r.x < max_t {
          let mat = material_buf[td.w];
          let geo_normal = triangle_geo_normal(td);
          let front_face = dot(dir, geo_normal) < 0.0;
          if !material_is_double_sided(mat) && !front_face { continue; }
          let bw = 1.0 - r.y - r.z;
          let uv0 = get_uv0(td, bw, r.y, r.z);
          let uv1 = get_uv1(td, bw, r.y, r.z);
          let uv2 = get_uv2(td, bw, r.y, r.z);
          let uv3 = get_uv3(td, bw, r.y, r.z);
          if !passes_alpha_shadow(td, bw, r.y, r.z, mat, uv0, uv1, uv2, uv3) { continue; }
          if material_has_transmission(mat) {
            let thickness = sample_thickness(mat, uv0, uv1, uv2, uv3);
            if material_is_thin(mat) || thickness <= 1e-5 { continue; }
          }
          return true;
        }
      }
      if sp==0u{break;} sp--; cur=stk[sp]; continue;
    }
    let l=nd.left_first; let r=l+1u;
    let tl=intersect_aabb(origin,inv_dir,bvh_nodes[l].aabb_min,bvh_nodes[l].aabb_max,max_t);
    let tr=intersect_aabb(origin,inv_dir,bvh_nodes[r].aabb_min,bvh_nodes[r].aabb_max,max_t);
    if tl<tr {
      if tr<max_t && sp<6u { stk[sp]=r; sp++; }
      if tl<max_t { cur=l; } else { if sp==0u{break;} sp--; cur=stk[sp]; }
    } else {
      if tl<max_t && sp<6u { stk[sp]=l; sp++; }
      if tr<max_t { cur=r; } else { if sp==0u{break;} sp--; cur=stk[sp]; }
    }
  }
  return false;
}

// Shadow (any hit, stack of 6)
fn trace_shadow(origin: vec3f, dir: vec3f, max_t: f32) -> bool {
  let inv_dir = 1.0 / dir;
  var stk: array<u32, 6>; var sp = 0u; var cur = 0u;
  let root = bvh_nodes[0u];
  if intersect_aabb(origin, inv_dir, root.aabb_min, root.aabb_max, max_t) >= max_t { return false; }
  loop {
    let nd = bvh_nodes[cur];
    if nd.tri_count > 0u {
      for (var i = 0u; i < nd.tri_count; i++) {
        let ti = nd.left_first+i; let td = tri_data[ti];
        let r = intersect_tri(origin, dir, vertices[td.x].xyz, vertices[td.y].xyz, vertices[td.z].xyz, max_t);
        if r.x < max_t {
          // Inline alpha test — transparent surfaces don't block light
          let mat = material_buf[td.w];
          let geo_normal = triangle_geo_normal(td);
          let front_face = dot(dir, geo_normal) < 0.0;
          if !material_is_double_sided(mat) && !front_face { continue; }
          let bw = 1.0 - r.y - r.z;
          let uv0 = get_uv0(td, bw, r.y, r.z);
          let uv1 = get_uv1(td, bw, r.y, r.z);
          let uv2 = get_uv2(td, bw, r.y, r.z);
          let uv3 = get_uv3(td, bw, r.y, r.z);
          if !passes_alpha_shadow(td, bw, r.y, r.z, mat, uv0, uv1, uv2, uv3) { continue; }
          if material_has_transmission(mat) {
            let thickness = sample_thickness(mat, uv0, uv1, uv2, uv3);
            if material_is_thin(mat) || thickness <= 1e-5 { continue; }
          }
          return true;
        }
      }
      if sp==0u{break;} sp--; cur=stk[sp]; continue;
    }
    let l=nd.left_first; let r=l+1u;
    let tl=intersect_aabb(origin,inv_dir,bvh_nodes[l].aabb_min,bvh_nodes[l].aabb_max,max_t);
    let tr=intersect_aabb(origin,inv_dir,bvh_nodes[r].aabb_min,bvh_nodes[r].aabb_max,max_t);
    if tl<tr {
      if tr<max_t && sp<6u { stk[sp]=r; sp++; }
      if tl<max_t { cur=l; } else { if sp==0u{break;} sp--; cur=stk[sp]; }
    } else {
      if tl<max_t && sp<6u { stk[sp]=l; sp++; }
      if tr<max_t { cur=r; } else { if sp==0u{break;} sp--; cur=stk[sp]; }
    }
  }
  return false;
}

// ============================================================
// Sky + Sun NEE
// ============================================================
fn sky_color(dir: vec3f) -> vec3f {
  let t = max(dir.y * 0.5 + 0.5, 0.0);
  var sky = mix(vec3f(0.3, 0.3, 0.35), SKY_COLOR, t);
  if dot(dir, g_sun_dir) > COS_SUN_ANGLE { sky += SUN_COLOR * SUN_MULT; }
  return sky;
}

// ============================================================
// AgX Punchy tone mapping — ported from ignis-rt (Blender 4)
// ============================================================
fn agx_tonemap(color_in: vec3f) -> vec3f {
  // sRGB -> REC.2020
  var c = mat3x3f(
    vec3f(0.6274, 0.0691, 0.0164),
    vec3f(0.3293, 0.9195, 0.0880),
    vec3f(0.0433, 0.0113, 0.8956)
  ) * color_in;
  // AgX Inset
  c = mat3x3f(
    vec3f(0.856627, 0.137319, 0.111898),
    vec3f(0.095121, 0.761242, 0.076799),
    vec3f(0.048252, 0.101439, 0.811302)
  ) * c;
  c = max(c, vec3f(1e-10));
  c = clamp(log2(c), vec3f(-12.47393), vec3f(4.026069));
  c = (c - (-12.47393)) / (4.026069 - (-12.47393));
  // Polynomial fit
  let x2 = c * c; let x4 = x2 * x2;
  c = 15.5*x4*x2 - 40.14*x4*c + 31.96*x4 - 6.868*x2*c + 0.4298*x2 + 0.1191*c - 0.00232;
  // Punchy: contrast + saturation boost
  c = pow(max(vec3f(0.0), c), vec3f(1.35));
  let luma = dot(c, vec3f(0.2126, 0.7152, 0.0722));
  c = luma + 1.4 * (c - luma);
  // AgX Outset + gamma encode
  c = mat3x3f(
    vec3f(1.1271, -0.1413, -0.1413),
    vec3f(-0.1106, 1.1578, -0.1106),
    vec3f(-0.0165, -0.0165, 1.2519)
  ) * c;
  c = pow(max(vec3f(0.0), c), vec3f(2.2));
  // REC.2020 -> sRGB
  c = mat3x3f(
    vec3f(1.6605, -0.1246, -0.0182),
    vec3f(-0.5876, 1.1329, -0.1006),
    vec3f(-0.0728, -0.0083, 1.1187)
  ) * c;
  return clamp(c, vec3f(0.0), vec3f(1.0));
}

// ============================================================
// Texture helpers
// ============================================================
fn srgb_to_linear(c: vec3f) -> vec3f {
  return pow(max(c, vec3f(0.0)), vec3f(2.2));
}

fn get_uv0(td: vec4u, bw: f32, u: f32, v: f32) -> vec2f {
  return bw * vec2f(vertices[td.x].w, vert_normals[td.x].w)
       + u * vec2f(vertices[td.y].w, vert_normals[td.y].w)
       + v * vec2f(vertices[td.z].w, vert_normals[td.z].w);
}

fn load_uv_extra(vertex_idx: u32, set_idx: u32) -> vec2f {
  let base = vertex_idx * 6u + set_idx * 2u;
  return vec2f(vert_uv_extra[base], vert_uv_extra[base + 1u]);
}

fn get_uv1(td: vec4u, bw: f32, u: f32, v: f32) -> vec2f {
  return bw * load_uv_extra(td.x, 0u) + u * load_uv_extra(td.y, 0u) + v * load_uv_extra(td.z, 0u);
}

fn get_uv2(td: vec4u, bw: f32, u: f32, v: f32) -> vec2f {
  return bw * load_uv_extra(td.x, 1u) + u * load_uv_extra(td.y, 1u) + v * load_uv_extra(td.z, 1u);
}

fn get_uv3(td: vec4u, bw: f32, u: f32, v: f32) -> vec2f {
  return bw * load_uv_extra(td.x, 2u) + u * load_uv_extra(td.y, 2u) + v * load_uv_extra(td.z, 2u);
}

fn select_uv(uv0: vec2f, uv1: vec2f, uv2: vec2f, uv3: vec2f, texcoord: u32) -> vec2f {
  if texcoord == 1u { return uv1; }
  if texcoord == 2u { return uv2; }
  if texcoord == 3u { return uv3; }
  return uv0;
}

fn triangle_geo_normal(td: vec4u) -> vec3f {
  let e1 = vertices[td.y].xyz - vertices[td.x].xyz;
  let e2 = vertices[td.z].xyz - vertices[td.x].xyz;
  let n = cross(e1, e2);
  let len2 = dot(n, n);
  if len2 <= 1e-20 { return vec3f(0.0, 1.0, 0.0); }
  return n / sqrt(len2);
}

fn tangent_basis_from_texcoord(td: vec4u, N: vec3f, texcoord: u32) -> mat3x3f {
  let v0 = vertices[td.x].xyz;
  let v1 = vertices[td.y].xyz;
  let v2 = vertices[td.z].xyz;
  let tuv0 = select_uv(vec2f(vertices[td.x].w, vert_normals[td.x].w), load_uv_extra(td.x, 0u), load_uv_extra(td.x, 1u), load_uv_extra(td.x, 2u), texcoord);
  let tuv1 = select_uv(vec2f(vertices[td.y].w, vert_normals[td.y].w), load_uv_extra(td.y, 0u), load_uv_extra(td.y, 1u), load_uv_extra(td.y, 2u), texcoord);
  let tuv2 = select_uv(vec2f(vertices[td.z].w, vert_normals[td.z].w), load_uv_extra(td.z, 0u), load_uv_extra(td.z, 1u), load_uv_extra(td.z, 2u), texcoord);
  let dp1 = v1 - v0;
  let dp2 = v2 - v0;
  let duv1 = tuv1 - tuv0;
  let duv2 = tuv2 - tuv0;
  let det = duv1.x * duv2.y - duv2.x * duv1.y;
  if abs(det) < 1e-8 {
    return build_onb(N);
  }
  let inv_det = 1.0 / det;
  var T = normalize((dp1 * duv2.y - dp2 * duv1.y) * inv_det);
  T = normalize(T - N * dot(N, T));
  let B = normalize(cross(N, T));
  return mat3x3f(T, B, N);
}

fn apply_detail_normal(tex_idx: i32, texcoord: u32, scale: f32, td: vec4u, N: vec3f, uv0: vec2f, uv1: vec2f, uv2: vec2f, uv3: vec2f) -> vec3f {
  if tex_idx < 0 { return N; }
  let uv = select_uv(uv0, uv1, uv2, uv3, texcoord);
  let ns = textureSampleLevel(tex_array, tex_sampler_pt, uv, tex_idx, 0.0);
  let tn = vec3f((ns.rg * 2.0 - vec2f(1.0)) * scale, ns.b * 2.0 - 1.0);
  let tbn = tangent_basis_from_texcoord(td, N, texcoord);
  return normalize(tbn[0] * tn.x + tbn[1] * tn.y + tbn[2] * tn.z);
}

fn sample_base_rgba(mat: Material, uv0: vec2f, uv1: vec2f, uv2: vec2f, uv3: vec2f) -> vec4f {
  var rgba = vec4f(mat_base_color_factor(mat), mat_base_alpha(mat));
  let base_tex_idx = mat_base_tex(mat);
  if base_tex_idx >= 0 {
    let uv = select_uv(uv0, uv1, uv2, uv3, mat_base_texcoord(mat));
    let tc = textureSampleLevel(tex_array, tex_sampler_pt, uv, base_tex_idx, 0.0);
    rgba *= vec4f(srgb_to_linear(tc.rgb), tc.a);
  }
  return vec4f(clamp(rgba.rgb, vec3f(0.0), vec3f(1e6)), clamp(rgba.a, 0.0, 1.0));
}

fn sample_metallic_roughness(mat: Material, uv0: vec2f, uv1: vec2f, uv2: vec2f, uv3: vec2f) -> vec2f {
  var roughness = clamp(mat_roughness_factor(mat), 0.0, 1.0);
  var metallic = clamp(mat_metallic_factor(mat), 0.0, 1.0);
  let mr_tex_idx = mat_mr_tex(mat);
  if mr_tex_idx >= 0 {
    let uv = select_uv(uv0, uv1, uv2, uv3, mat_mr_texcoord(mat));
    let mr = textureSampleLevel(tex_array, tex_sampler_pt, uv, mr_tex_idx, 0.0);
    roughness *= mr.g;
    metallic *= mr.b;
  }
  return vec2f(clamp(metallic, 0.0, 1.0), clamp(roughness, 0.0, 1.0));
}

fn sample_emissive(mat: Material, uv0: vec2f, uv1: vec2f, uv2: vec2f, uv3: vec2f) -> vec3f {
  var emission = max(mat_emissive_factor(mat), vec3f(0.0));
  let emissive_tex_idx = mat_emissive_tex(mat);
  if emissive_tex_idx >= 0 {
    let uv = select_uv(uv0, uv1, uv2, uv3, mat_emissive_texcoord(mat));
    emission *= srgb_to_linear(textureSampleLevel(tex_array, tex_sampler_pt, uv, emissive_tex_idx, 0.0).rgb);
  }
  return emission * mat_emission_strength(mat);
}

fn sample_occlusion(mat: Material, uv0: vec2f, uv1: vec2f, uv2: vec2f, uv3: vec2f) -> f32 {
  let occlusion_tex_idx = mat_occlusion_tex(mat);
  if occlusion_tex_idx < 0 { return 1.0; }
  let uv = select_uv(uv0, uv1, uv2, uv3, mat_occlusion_texcoord(mat));
  let ao = textureSampleLevel(tex_array, tex_sampler_pt, uv, occlusion_tex_idx, 0.0).r;
  return mix(1.0, ao, mat_occlusion_strength(mat));
}

fn sample_transmission(mat: Material, uv0: vec2f, uv1: vec2f, uv2: vec2f, uv3: vec2f, metallic: f32) -> f32 {
  var transmission = mat_transmission_factor(mat);
  let tx = mat_transmission_tex(mat);
  if tx >= 0 {
    let uv = select_uv(uv0, uv1, uv2, uv3, mat_transmission_texcoord(mat));
    transmission *= textureSampleLevel(tex_array, tex_sampler_pt, uv, tx, 0.0).r;
  }
  return clamp(transmission * (1.0 - metallic), 0.0, 1.0);
}

fn sample_thickness(mat: Material, uv0: vec2f, uv1: vec2f, uv2: vec2f, uv3: vec2f) -> f32 {
  var thickness = mat_thickness_factor(mat);
  let tx = mat_thickness_tex(mat);
  if tx >= 0 {
    let uv = select_uv(uv0, uv1, uv2, uv3, mat_thickness_texcoord(mat));
    thickness *= textureSampleLevel(tex_array, tex_sampler_pt, uv, tx, 0.0).g;
  }
  return max(thickness, 0.0);
}

fn max_component(v: vec3f) -> f32 {
  return max(v.x, max(v.y, v.z));
}

fn fresnel_schlick_vec(F0: vec3f, F90: vec3f, cos_theta: f32) -> vec3f {
  let f = pow(1.0 - clamp(cos_theta, 0.0, 1.0), 5.0);
  return F0 + (F90 - F0) * f;
}

fn fresnel_schlick_scalar(f0: f32, f90: f32, cos_theta: f32) -> f32 {
  let f = pow(1.0 - clamp(cos_theta, 0.0, 1.0), 5.0);
  return f0 + (f90 - f0) * f;
}

fn sample_specular_params(mat: Material, uv0: vec2f, uv1: vec2f, uv2: vec2f, uv3: vec2f) -> vec4f {
  var specular_weight = mat_specular_factor(mat);
  let spec_tex_idx = mat_specular_tex(mat);
  if spec_tex_idx >= 0 {
    let uv = select_uv(uv0, uv1, uv2, uv3, mat_specular_texcoord(mat));
    specular_weight *= textureSampleLevel(tex_array, tex_sampler_pt, uv, spec_tex_idx, 0.0).a;
  }

  var specular_color = clamp(mat_specular_color_factor(mat), vec3f(0.0), vec3f(1.0));
  let spec_color_tex_idx = mat_specular_color_tex(mat);
  if spec_color_tex_idx >= 0 {
    let uv = select_uv(uv0, uv1, uv2, uv3, mat_specular_color_texcoord(mat));
    specular_color *= srgb_to_linear(textureSampleLevel(tex_array, tex_sampler_pt, uv, spec_color_tex_idx, 0.0).rgb);
  }
  return vec4f(specular_color, clamp(specular_weight, 0.0, 1.0));
}

fn sample_clearcoat_params(mat: Material, uv0: vec2f, uv1: vec2f, uv2: vec2f, uv3: vec2f) -> vec2f {
  var clearcoat = mat_clearcoat_factor(mat);
  let clearcoat_tex_idx = mat_clearcoat_tex(mat);
  if clearcoat_tex_idx >= 0 {
    let uv = select_uv(uv0, uv1, uv2, uv3, mat_clearcoat_texcoord(mat));
    clearcoat *= textureSampleLevel(tex_array, tex_sampler_pt, uv, clearcoat_tex_idx, 0.0).r;
  }

  var clearcoat_roughness = mat_clearcoat_roughness_factor(mat);
  let clearcoat_rough_tex_idx = mat_clearcoat_rough_tex(mat);
  if clearcoat_rough_tex_idx >= 0 {
    let uv = select_uv(uv0, uv1, uv2, uv3, mat_clearcoat_rough_texcoord(mat));
    clearcoat_roughness *= textureSampleLevel(tex_array, tex_sampler_pt, uv, clearcoat_rough_tex_idx, 0.0).g;
  }
  return vec2f(clamp(clearcoat, 0.0, 1.0), clamp(clearcoat_roughness, 0.0, 1.0));
}

fn sample_sheen_params(mat: Material, uv0: vec2f, uv1: vec2f, uv2: vec2f, uv3: vec2f) -> vec4f {
  var sheen_color = max(mat_sheen_color_factor(mat), vec3f(0.0));
  let sheen_color_tex_idx = mat_sheen_color_tex(mat);
  if sheen_color_tex_idx >= 0 {
    let uv = select_uv(uv0, uv1, uv2, uv3, mat_sheen_color_texcoord(mat));
    sheen_color *= srgb_to_linear(textureSampleLevel(tex_array, tex_sampler_pt, uv, sheen_color_tex_idx, 0.0).rgb);
  }

  var sheen_roughness = mat_sheen_roughness_factor(mat);
  let sheen_rough_tex_idx = mat_sheen_rough_tex(mat);
  if sheen_rough_tex_idx >= 0 {
    let uv = select_uv(uv0, uv1, uv2, uv3, mat_sheen_rough_texcoord(mat));
    sheen_roughness *= textureSampleLevel(tex_array, tex_sampler_pt, uv, sheen_rough_tex_idx, 0.0).a;
  }
  return vec4f(sheen_color, clamp(sheen_roughness, 0.0, 1.0));
}

fn sample_anisotropy_params(mat: Material, uv0: vec2f, uv1: vec2f, uv2: vec2f, uv3: vec2f) -> vec3f {
  var direction = vec2f(cos(mat_anisotropy_rotation(mat)), sin(mat_anisotropy_rotation(mat)));
  var strength = mat_anisotropy_strength(mat);
  let anisotropy_tex_idx = mat_anisotropy_tex(mat);
  if anisotropy_tex_idx >= 0 {
    let uv = select_uv(uv0, uv1, uv2, uv3, mat_anisotropy_texcoord(mat));
    let anisotropy_tex = textureSampleLevel(tex_array, tex_sampler_pt, uv, anisotropy_tex_idx, 0.0).rgb;
    let map_dir_raw = anisotropy_tex.rg * 2.0 - vec2f(1.0);
    if dot(map_dir_raw, map_dir_raw) > 1e-6 {
      let map_dir = normalize(map_dir_raw);
      let c = cos(mat_anisotropy_rotation(mat));
      let s = sin(mat_anisotropy_rotation(mat));
      direction = vec2f(
        c * map_dir.x + s * map_dir.y,
        -s * map_dir.x + c * map_dir.y
      );
    }
    strength *= anisotropy_tex.b;
  }
  return vec3f(direction, clamp(strength, 0.0, 1.0));
}

fn apply_volume_attenuation(distance: f32, attenuation_color: vec3f, attenuation_distance: f32) -> vec3f {
  if distance <= 0.0 || attenuation_distance >= 1e29 { return vec3f(1.0); }
  let safe_color = clamp(attenuation_color, vec3f(1e-3), vec3f(1.0));
  return exp(log(safe_color) * (distance / max(attenuation_distance, 1e-6)));
}

fn sample_alpha_coverage(mat: Material, uv0: vec2f, uv1: vec2f, uv2: vec2f, uv3: vec2f) -> f32 {
  return sample_base_rgba(mat, uv0, uv1, uv2, uv3).a;
}

fn alpha_coverage_factor(mat: Material, uv0: vec2f, uv1: vec2f, uv2: vec2f, uv3: vec2f) -> f32 {
  let alpha_mode = mat_alpha_mode(mat);
  if alpha_mode == 0u { return 1.0; }
  let alpha = sample_alpha_coverage(mat, uv0, uv1, uv2, uv3);
  if alpha_mode == 1u {
    return select(0.0, 1.0, alpha >= mat_alpha_cutoff(mat));
  }
  return alpha;
}

fn hash_u32(x_in: u32) -> u32 {
  var x = x_in;
  x ^= x >> 16u;
  x *= 0x7feb352du;
  x ^= x >> 15u;
  x *= 0x846ca68bu;
  x ^= x >> 16u;
  return x;
}

fn blend_alpha_threshold(td: vec4u, bw: f32, u: f32, v: f32, mat: Material, uv0: vec2f, uv1: vec2f, uv2: vec2f, uv3: vec2f) -> f32 {
  let base_uv = select_uv(uv0, uv1, uv2, uv3, mat_base_texcoord(mat));
  let pos = bw * vertices[td.x].xyz + u * vertices[td.y].xyz + v * vertices[td.z].xyz;
  let uv_q = vec2u(floor(fract(base_uv) * 4096.0));
  let cell = vec3i(floor(pos * 64.0));

  var h = hash_u32(td.w * 747796405u + 2891336453u);
  h ^= hash_u32(uv_q.x * 1597334677u + uv_q.y * 3812015801u);
  h ^= hash_u32(bitcast<u32>(cell.x) * 958689281u);
  h ^= hash_u32(bitcast<u32>(cell.y) * 3266489917u);
  h ^= hash_u32(bitcast<u32>(cell.z) * 668265263u);
  return f32(h & 0x00ffffffu) * (1.0 / 16777216.0);
}

fn passes_alpha_surface_hit(mat: Material, uv0: vec2f, uv1: vec2f, uv2: vec2f, uv3: vec2f) -> bool {
  let alpha_mode = mat_alpha_mode(mat);
  if alpha_mode == 0u { return true; }
  let alpha = sample_alpha_coverage(mat, uv0, uv1, uv2, uv3);
  if alpha_mode == 1u { return alpha >= mat_alpha_cutoff(mat); }
  return alpha > 0.001;
}

fn passes_alpha_shadow(td: vec4u, bw: f32, u: f32, v: f32, mat: Material, uv0: vec2f, uv1: vec2f, uv2: vec2f, uv3: vec2f) -> bool {
  let alpha_mode = mat_alpha_mode(mat);
  if alpha_mode == 0u { return true; }
  let alpha = sample_alpha_coverage(mat, uv0, uv1, uv2, uv3);
  if alpha_mode == 1u { return alpha >= mat_alpha_cutoff(mat); }
  return alpha > blend_alpha_threshold(td, bw, u, v, mat, uv0, uv1, uv2, uv3);
}

fn apply_normal_map(td: vec4u, N: vec3f, uv0: vec2f, uv1: vec2f, uv2: vec2f, uv3: vec2f, mat: Material) -> vec3f {
  return apply_detail_normal(mat_normal_tex(mat), mat_normal_texcoord(mat), mat_normal_scale(mat), td, N, uv0, uv1, uv2, uv3);
}

// ============================================================
// Path tracing with PBR BRDF + texture sampling
// Returns radiance; writes first-hit normal+depth to out params
// ============================================================
struct PathResult {
  diffuse: vec3f, specular: vec3f, normal: vec3f, depth: f32,
  hit_pos: vec3f, direct: vec3f, tri_idx: u32, bary: vec2f,
  albedo: vec3f, roughness: f32, hit_dist: f32,
  // SHaRC backpropagation + path guiding data
  sharc_pos: array<vec3f, 4>, sharc_nrm: array<vec3f, 4>,
  sharc_rad: array<vec3f, 4>, sharc_dir: array<vec3f, 4>,
  sharc_count: u32,
  through_glass: bool,
};

var<private> g_depth_hint: f32; // G-buffer depth hint for first bounce acceleration

fn path_trace(primary_origin: vec3f, primary_dir: vec3f) -> PathResult {
  var result: PathResult;
  result.normal = vec3f(0.0, 1.0, 0.0);
  result.depth = 1e6;
  result.hit_pos = vec3f(0.0);
  result.direct = vec3f(0.0);
  result.albedo = vec3f(1.0);  // default white (sky/miss pixels)
  result.roughness = 1.0;
  result.hit_dist = 1e4;      // default far (sky/no indirect)

  var diff_rad = vec3f(0.0);   // demodulated diffuse irradiance
  var spec_rad = vec3f(0.0);   // specular radiance
  var throughput = vec3f(1.0);
  var origin = primary_origin;
  var dir = primary_dir;
  var specular_bounce = true;
  var is_diffuse_path = true;
  var glass_bounces = 0u;
  var thin_glass_passes = 0u;
  var went_through_glass = false;
  // SHaRC backpropagation + path guiding: store up to 4 bounce points
  var sharc_pos: array<vec3f, 4>;
  var sharc_nrm: array<vec3f, 4>;
  var sharc_rad: array<vec3f, 4>;
  var sharc_dir: array<vec3f, 4>; // incoming light direction per bounce
  var sharc_count = 0u;
  var medium_mat: array<u32, MAX_VOLUME_STACK>;
  var medium_att_dist: array<f32, MAX_VOLUME_STACK>;
  var medium_att_color: array<vec3f, MAX_VOLUME_STACK>;
  var medium_depth = 0u;

  var bounce = 0u;
  loop {
    if bounce >= uniforms.max_bounces { break; }
    let hit = trace_bvh(origin, dir);
    if !hit.hit {
      let sky = throughput * sky_color(dir);
      // Sky at bounce 0: no surface → specular (not multiplied by albedo in composite)
      if bounce == 0u || !is_diffuse_path { spec_rad += sky; }
      else { diff_rad += sky; }
      break;
    }

    // Capture first indirect bounce hit distance (for denoiser blur radius)
    if bounce == 1u { result.hit_dist = hit.t; }

    let td = tri_data[hit.tri_idx];
    let mat = material_buf[td.w];
    if medium_depth > 0u {
      let top = medium_depth - 1u;
      throughput *= apply_volume_attenuation(hit.t, medium_att_color[top], medium_att_dist[top]);
    }

    let bw = 1.0 - hit.u - hit.v;
    let geo_normal_raw = triangle_geo_normal(td);
    let front_face = dot(dir, geo_normal_raw) < 0.0;
    let geo_normal = select(-geo_normal_raw, geo_normal_raw, front_face);
    var smooth_normal = normalize(bw * vert_normals[td.x].xyz + hit.u * vert_normals[td.y].xyz + hit.v * vert_normals[td.z].xyz);
    if !front_face { smooth_normal = -smooth_normal; }
    let hit_pos = origin + dir * hit.t;
    let V = -dir;
    let uv0 = get_uv0(td, bw, hit.u, hit.v);
    let uv1 = get_uv1(td, bw, hit.u, hit.v);
    let uv2 = get_uv2(td, bw, hit.u, hit.v);
    let uv3 = get_uv3(td, bw, hit.u, hit.v);
    let base_rgba = sample_base_rgba(mat, uv0, uv1, uv2, uv3);
    let base_color = base_rgba.rgb;
    let surface_alpha = base_rgba.a;
    let alpha_mode = mat_alpha_mode(mat);
    let mr = sample_metallic_roughness(mat, uv0, uv1, uv2, uv3);
    let metallic = mr.x;
    let roughness = max(mr.y, 0.04);
    let emissive = sample_emissive(mat, uv0, uv1, uv2, uv3);
    let specular_params = sample_specular_params(mat, uv0, uv1, uv2, uv3);
    let clearcoat_params = sample_clearcoat_params(mat, uv0, uv1, uv2, uv3);
    var normal = apply_normal_map(td, smooth_normal, uv0, uv1, uv2, uv3, mat);
    let material_ior = max(mat_ior(mat), 1.0);
    let dielectric_base_f0 = pow((material_ior - 1.0) / max(material_ior + 1.0, 1e-6), 2.0);
    let dielectric_f0 = min(vec3f(dielectric_base_f0) * specular_params.rgb, vec3f(1.0)) * specular_params.a;

    // Fix normal facing
    if dot(smooth_normal, V) < 0.01 {
      smooth_normal = normalize(smooth_normal + V * 0.2);
    }
    if dot(normal, V) < 0.01 {
      normal = normalize(normal + V * 0.2);
    }

    let is_alpha_blend = alpha_mode == 2u && surface_alpha > 0.001 && surface_alpha < 0.999;

    var glass_transmission = 0.0;
    var glass_thickness = 0.0;
    if material_has_transmission(mat) {
      glass_transmission = sample_transmission(mat, uv0, uv1, uv2, uv3, metallic);
      glass_thickness = sample_thickness(mat, uv0, uv1, uv2, uv3);
    }
    let is_glass = glass_transmission > 0.01;
    let thin_glass = is_glass && (material_is_thin(mat) || glass_thickness <= 1e-5);

    // PSR: capture the first non-interface surface for denoiser G-buffer.
    if result.depth > 1e5 && !is_glass && !is_alpha_blend {
      result.normal = normal; result.depth = hit.t; result.hit_pos = hit_pos;
      result.tri_idx = hit.tri_idx; result.bary = vec2f(hit.u, hit.v);
      result.albedo = base_color; result.roughness = roughness;
    }

    if is_glass {
      let glass_ior = max(material_ior, 1.01);
      let transmission_tint = clamp(base_color, vec3f(0.0), vec3f(1.0));

      if thin_glass {
        thin_glass_passes += 1u;
        if thin_glass_passes > MAX_THIN_GLASS_PASSES { break; }

        let cos_theta = abs(dot(normal, V));
        let fresnel_thin = fresnel_dielectric(cos_theta, 1.0 / glass_ior);
        let thin_pass_prob = glass_transmission * (1.0 - fresnel_thin);
        let thin_reflect_prob = glass_transmission * fresnel_thin;
        let thin_total_prob = thin_pass_prob + thin_reflect_prob;
        let thin_rnd = rand();

        if thin_pass_prob > 0.01 && thin_rnd < thin_pass_prob {
          let alpha_thin = roughness * roughness;
          var H_thin = normal;
          if alpha_thin > 0.0005 {
            H_thin = sample_ggx_vndf(rand2(), V, normal, max(alpha_thin, 0.001));
          }
          let inside_thin = refract(-V, H_thin, 1.0 / glass_ior);
          var thin_dir = dir;
          if dot(inside_thin, inside_thin) > 0.0001 {
            let out_thin = refract(inside_thin, -H_thin, glass_ior);
            if dot(out_thin, out_thin) > 0.0001 {
              thin_dir = normalize(out_thin);
            }
          }
          throughput *= transmission_tint / max(thin_pass_prob, 0.01);
          dir = thin_dir;
          origin = hit_pos + dir * 0.002;
          went_through_glass = true;
          specular_bounce = true;
          is_diffuse_path = false;
          continue;
        } else if thin_reflect_prob > 0.001 && thin_rnd < thin_total_prob {
          let alpha_thin = roughness * roughness;
          var H_thin = normal;
          if alpha_thin > 0.0005 {
            H_thin = sample_ggx_vndf(rand2(), V, normal, max(alpha_thin, 0.001));
          }
          dir = reflect(-V, H_thin);
          if dot(normal, dir) <= 0.0 { break; }
          throughput *= vec3f(1.0) / max(thin_reflect_prob, 0.01);
          origin = hit_pos + geo_normal * BIAS;
          went_through_glass = true;
          specular_bounce = true;
          is_diffuse_path = false;
          continue;
        }
      } else {
        let glass_select = rand();
        if glass_select < glass_transmission {
          glass_bounces += 1u;
          if glass_bounces > 16u { break; }

          let eta = select(glass_ior, 1.0 / glass_ior, front_face);
          let alpha_glass = roughness * roughness;
          var H_glass = normal;
          if alpha_glass > 0.0005 {
            H_glass = sample_ggx_vndf(rand2(), V, normal, max(alpha_glass, 0.001));
          }

          let cos_i = abs(dot(V, H_glass));
          let fresnel_glass = fresnel_dielectric(cos_i, eta);
          let refracted = refract(-V, H_glass, eta);
          let can_refract = dot(refracted, refracted) > 0.0001;
          let reflect_prob = glass_transmission * fresnel_glass;
          let refract_prob = glass_transmission * (1.0 - fresnel_glass);

          if !can_refract || rand() < fresnel_glass {
            dir = reflect(-V, H_glass);
            origin = hit_pos + geo_normal * BIAS;
            throughput *= vec3f(1.0) / max(max(reflect_prob, glass_transmission * 0.01), 0.01);
          } else {
            dir = normalize(refracted);
            origin = hit_pos - geo_normal * BIAS;
            throughput *= transmission_tint / max(max(refract_prob, glass_transmission * 0.01), 0.01);
            if front_face {
              if medium_depth < MAX_VOLUME_STACK && material_needs_attenuation(mat) {
                medium_mat[medium_depth] = td.w;
                medium_att_dist[medium_depth] = mat_attenuation_distance(mat);
                medium_att_color[medium_depth] = mat_attenuation_color(mat);
                medium_depth += 1u;
              }
            } else if medium_depth > 0u {
              let top = medium_depth - 1u;
              if medium_mat[top] == td.w {
                medium_depth = top;
              }
            }
          }

          went_through_glass = true;
          specular_bounce = true;
          is_diffuse_path = false;
          continue;
        }
      }
    }

    if is_alpha_blend && !is_glass {
      let blend_weight = clamp(surface_alpha, 0.0, 1.0);
      let use_primary_split = bounce == 0u && !went_through_glass;
      var blend_diff = vec3f(0.0);
      var blend_spec = vec3f(0.0);

      if material_is_unlit(mat) {
        blend_spec += base_color;
      } else {
        if use_primary_split {
          let nee = sample_sun_nee_split(hit_pos, normal, V, td, mat, uv0, uv1, uv2, uv3, base_color, roughness, metallic, 0.0);
          blend_diff += nee.diffuse * base_color;
          blend_spec += nee.specular;
          result.direct += blend_weight * (nee.diffuse * base_color + nee.specular);
        } else {
          let direct = sample_sun_nee(hit_pos, normal, V, td, mat, uv0, uv1, uv2, uv3, base_color, roughness, metallic, 0.0);
          if is_diffuse_path { blend_diff += direct; }
          else { blend_spec += direct; }
        }
      }

      let blend_emission = emissive;
      let blend_throughput = throughput * blend_weight;
      diff_rad += blend_throughput * blend_diff;
      spec_rad += blend_throughput * (blend_spec + blend_emission);

      throughput *= max(1.0 - blend_weight, 0.0);
      origin = hit_pos + dir * 0.002;
      specular_bounce = true;
      continue;
    }

    if material_is_unlit(mat) {
      let unlit = throughput * (base_color + emissive);
      if bounce == 0u || !is_diffuse_path { spec_rad += unlit; }
      else { diff_rad += unlit; }
      break;
    }

    // Emission
    if dot(emissive, vec3f(1.0)) > 1e-6 {
      if specular_bounce {
        let e = throughput * emissive;
        if bounce == 0u { spec_rad += e; }
        else if is_diffuse_path { diff_rad += e; }
        else { spec_rad += e; }
      }
      break;
    }

    // NEE: use the demodulated primary split only before crossing any glass interface.
    let use_primary_split = bounce == 0u && !went_through_glass;
    if use_primary_split {
      let nee = sample_sun_nee_split(hit_pos, normal, V, td, mat, uv0, uv1, uv2, uv3, base_color, roughness, metallic, glass_transmission);
      diff_rad += throughput * nee.diffuse;
      spec_rad += throughput * nee.specular;
      result.direct = nee.diffuse * base_color + nee.specular;
    } else {
      let direct = sample_sun_nee(hit_pos, normal, V, td, mat, uv0, uv1, uv2, uv3, base_color, roughness, metallic, glass_transmission);
      if is_diffuse_path { diff_rad += throughput * direct; }
      else { spec_rad += throughput * direct; }
      if bounce == 0u { result.direct = direct; }
      // SHaRC backpropagation with direction (for path guiding)
      if bounce > 0u && sharc_count < 4u {
        sharc_pos[sharc_count] = hit_pos;
        sharc_nrm[sharc_count] = normal;
        sharc_rad[sharc_count] = direct;
        sharc_dir[sharc_count] = -dir; // incoming light direction = reverse of bounce dir
        sharc_count += 1u;
      }
    }

    // Bounce 2+: try SHaRC cache for late indirect GI.
    // The cache currently stores direct illumination at cached bounce points,
    // so using it at the first indirect hit truncates higher-order transport too early.
    if bounce >= 2u && !is_glass {
      let cached_gi = sharc_read_cached(hit_pos, normal);
      let has_cache = dot(cached_gi, vec3f(1.0)) > 0.001;
      if has_cache {
        // cached_gi already includes local BRDF/albedo from the cached point.
        // Do not darken it with glTF AO; AO is not a transport term.
        let gi = throughput * cached_gi;
        if is_diffuse_path { diff_rad += gi; } else { spec_rad += gi; }
        break;
      }
      let indirect_direct = sample_sun_nee(hit_pos, normal, V, td, mat, uv0, uv1, uv2, uv3, base_color, roughness, metallic, glass_transmission);
      let ind = throughput * indirect_direct;
      if is_diffuse_path { diff_rad += ind; } else { spec_rad += ind; }

      // Last bounce: energy terminates. No fake sky — SHaRC + extra bounces provide real GI.
      if bounce >= uniforms.max_bounces - 1u { break; }
    }

    // BRDF sampling — at bounce 0, classify path and demodulate diffuse throughput
    let clearcoat_prob = clamp(clearcoat_params.x * 0.25, 0.0, 0.5);
    let base_specular_prob = clamp(mix(specular_params.a, 1.0, metallic), 0.05, 0.95);
    if clearcoat_prob > 0.001 && rand() < clearcoat_prob {
      let clearcoat_normal = apply_detail_normal(
        mat_clearcoat_normal_tex(mat),
        mat_clearcoat_normal_texcoord(mat),
        mat_clearcoat_normal_scale(mat),
        td, normal, uv0, uv1, uv2, uv3
      );
      let alpha_c = max(clearcoat_params.y * clearcoat_params.y, 0.001);
      let Hc = sample_ggx_vndf(rand2(), V, clearcoat_normal, alpha_c);
      dir = reflect(-V, Hc);
      if dot(clearcoat_normal, dir) <= 0.0 { break; }
      let VdotHc = max(dot(V, Hc), 0.0);
      let Fc = fresnel_schlick_vec(vec3f(0.04), vec3f(1.0), VdotHc);
      let G1c = smith_g1(max(dot(clearcoat_normal, dir), 0.0), alpha_c);
      throughput *= Fc * G1c / max(clearcoat_prob, 0.01);
      specular_bounce = true;
      if bounce == 0u { is_diffuse_path = false; }
    } else if metallic > 0.5 || base_specular_prob > 0.35 || roughness < 0.18 {
      // Metal / specular-dominant: GGX VNDF sampling → specular path
      let alpha = max(roughness * roughness, 0.001);
      let H = sample_ggx_vndf(rand2(), V, normal, alpha);
      dir = reflect(-V, H);
      if dot(normal, dir) <= 0.0 { break; }
      let VdotH = max(dot(V, H), 0.0);
      let F0 = mix(dielectric_f0, base_color, metallic);
      let F90 = mix(vec3f(specular_params.a), vec3f(1.0), metallic);
      let F = fresnel_schlick_vec(F0, F90, VdotH);
      let G1_L = smith_g1(max(dot(normal, dir), 0.0), alpha);
      throughput *= F * G1_L;
      specular_bounce = roughness < 0.1;
      if bounce == 0u { is_diffuse_path = false; }
    } else {
      // Dielectric: guided cosine hemisphere → diffuse path
      // Path guiding: read dominant light direction from SHaRC cache
      let guide = sharc_read_guide(hit_pos, normal);
      var bent = normal;
      if guide.w > 0.1 && dot(guide.xyz, normal) > 0.1 {
        // Bend normal toward guide direction (L1 SH path guiding)
        // Strength proportional to concentration: 0 = pure cosine, 0.5 = max bend
        bent = normalize(mix(normal, guide.xyz, guide.w * 0.5));
        if dot(bent, normal) < 0.2 { bent = normal; } // safety: don't bend too far
      }
      dir = cosine_sample_hemisphere(bent);
      if dot(normal, dir) <= 0.0 { break; }
      // PDF correction for bent normal: compensate bias for unbiased result
      // cosine hemisphere PDF relative to bent: cos(dir,bent)/π
      // actual BRDF uses cos(dir,normal)/π → correction = cos(dir,normal)/cos(dir,bent)
      let pdf_correction = max(dot(dir, normal), 0.001) / max(dot(dir, bent), 0.001);
      let diffuse_scale = max((1.0 - metallic) * (1.0 - glass_transmission) * (1.0 - max_component(dielectric_f0)), 0.0);
      if bounce == 0u && !went_through_glass {
        // True primary diffuse surface: store demodulated irradiance for the denoiser.
        throughput *= vec3f(diffuse_scale * pdf_correction);
      } else {
        throughput *= base_color * diffuse_scale * pdf_correction;
      }
      if bounce == 0u { is_diffuse_path = true; }
      specular_bounce = false;
    }

    origin = hit_pos + normal * BIAS;

    // Perceptual √ Russian roulette (ignis-rt / NRD guideline)
    // sqrt prevents dim paths from surviving with huge weight → fewer fireflies
    // Min 0.05 = max 20× boost (NRD recommendation)
    if bounce > 0u {
      let p = clamp(sqrt(max(throughput.x, max(throughput.y, throughput.z))), 0.05, 0.9);
      if rand() > p { break; }
      throughput /= p;
    }
    bounce += 1u;
  }
  result.diffuse = diff_rad;
  result.specular = spec_rad;
  result.sharc_pos = sharc_pos;
  result.sharc_nrm = sharc_nrm;
  result.sharc_rad = sharc_rad;
  result.sharc_dir = sharc_dir;
  result.sharc_count = sharc_count;
  result.through_glass = went_through_glass;
  return result;
}

// ============================================================
// Shade from rasterized G-buffer (no primary BVH traversal!)
// Only traces shadow rays + indirect bounces
// ============================================================
fn path_trace_from_gbuffer(hit_pos: vec3f, normal_in: vec3f, view_dir: vec3f, mat_id: u32, uv: vec2f) -> vec3f {
  let mat = material_buf[mat_id];
  let V = -view_dir;
  var normal = normal_in;
  if dot(normal, V) < 0.01 { normal = normalize(normal + V * 0.2); }

  // Sample textures using rasterized UVs
  let base_color = sample_base_rgba(mat, uv, uv, uv, uv).rgb;
  let mr = sample_metallic_roughness(mat, uv, uv, uv, uv);
  let metallic = mr.x;
  let roughness = max(mr.y, 0.04);
  let transmission = sample_transmission(mat, uv, uv, uv, uv, metallic);
  let surface = build_surface_eval_basic(mat, normal, base_color, roughness, metallic, transmission);

  // Direct lighting (NEE to sun)
  let direct = sample_scene_nee_basic(hit_pos, normal, V, surface);
  var radiance = direct;

  // SHaRC store (sparse)
  // Handled in main() after this returns

  // Indirect: trace one bounce for GI
  if uniforms.max_bounces >= 2u {
    let bounce_dir = cosine_sample_hemisphere(normal);
    if dot(normal, bounce_dir) > 0.0 {
      let bounce_hit = trace_bvh(hit_pos + normal * BIAS, bounce_dir);
      if !bounce_hit.hit {
        // Sky contribution
        radiance += base_color * (1.0 - metallic) * sky_color(bounce_dir);
      } else {
        // Indirect: check SHaRC first, fallback to NEE
        let btd = tri_data[bounce_hit.tri_idx];
        let bmat = material_buf[btd.w];
        let bhit_pos = hit_pos + normal * BIAS + bounce_dir * bounce_hit.t;
        let bbw = 1.0 - bounce_hit.u - bounce_hit.v;
        var bnormal = normalize(bbw*vert_normals[btd.x].xyz + bounce_hit.u*vert_normals[btd.y].xyz + bounce_hit.v*vert_normals[btd.z].xyz);
        if dot(bounce_dir, bnormal) > 0.0 { bnormal = -bnormal; }

          let cached = sharc_read_cached(bhit_pos, bnormal);
          if dot(cached, vec3f(1.0)) > 0.001 {
            radiance += base_color * (1.0 - metallic) * cached * mat_base_color_factor(bmat);
          } else {
            let bV = -bounce_dir;
            let bbase = mat_base_color_factor(bmat);
            let bsurface = build_surface_eval_basic(bmat, bnormal, bbase, max(mat_roughness_factor(bmat), 0.04), mat_metallic_factor(bmat), 0.0);
            let bind = sample_scene_nee_basic(bhit_pos, bnormal, bV, bsurface);
            radiance += base_color * (1.0 - metallic) * bind;

          // Sky irradiance on last bounce
          if uniforms.max_bounces <= 3u {
            let sky_up = sky_color(bnormal);
            let sky_side = sky_color(vec3f(bnormal.x, 0.0, bnormal.z));
            let sky_irr = mix(sky_side, sky_up, max(bnormal.y, 0.0)) * INV_PI;
            radiance += base_color * (1.0 - metallic) * sky_irr * bbase;
          }
        }
      }
    }
  }

  return radiance;
}

// ============================================================
// Main compute entry
// ============================================================
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let pixel = vec2u(gid.xy);
  let res = vec2u(uniforms.resolution);
  if pixel.x >= res.x || pixel.y >= res.y { return; }

  // All pixels traced every frame — temporal denoiser handles accumulation
  // Checkerboard/adaptive disabled: causes visible jitter with temporal blend

  let idx = pixel.y * res.x + pixel.x;

  // RNG init: PCG state + spatio-temporal R2 offset
  rng_state = (pixel.x * 1973u + pixel.y * 9277u + uniforms.frame_seed * 26699u) | 1u;
  _ = pcg(&rng_state);
  g_sun_dir = normalize(uniforms.sun_dir);
  g_sample_idx = 0u;
  // R2 offset: spatial (per-pixel stratification) + temporal (per-frame rotation)
  // Spatial R2 distributes samples across pixels with low discrepancy (blue-noise property)
  // Temporal R2 rotates pattern each frame (Cranley-Patterson)
  let frame_f = f32(uniforms.frame_seed);
  let spatial_offset = fract(vec2f(
    R2_A1 * f32(pixel.x) + R2_A2 * f32(pixel.y),
    R2_A2 * f32(pixel.x) + R2_A1 * f32(pixel.y)
  ));
  let temporal_offset = fract(vec2f(R2_A1 * frame_f, R2_A2 * frame_f));
  g_r2_offset = fract(spatial_offset + temporal_offset);
  // Alpha dither: IGN spatial (Jimenez 2014) + per-pixel phased golden ratio temporal
  // IGN provides blue-noise-like spatial distribution
  // Per-pixel phase offset prevents coherent movement across the screen
  let ign = fract(52.9829189 * fract(dot(vec2f(f32(pixel.x), f32(pixel.y)), vec2f(0.06711056, 0.00583715))));
  let pixel_phase = f32((pixel.x * 1973u + pixel.y * 9277u) % 256u);
  g_alpha_dither = fract(ign + (frame_f + pixel_phase) * 0.3819660113);

  // Read G-buffer depth hint (rasterized — much faster than BVH for primary visibility)
  let nd = textureLoad(gbuf_nd, vec2i(pixel), 0);
  g_depth_hint = nd.w; // set global hint for trace_bvh_hint in bounce 0

  // Full quality path trace with all features:
  // - Normal maps, texture sampling, alpha testing, full BRDF
  // - G-buffer depth hint accelerates BVH traversal ~60-80% (prunes far nodes)
  let jitter = rand2();
  let uv_px = (vec2f(f32(pixel.x), f32(pixel.y)) + jitter) / uniforms.resolution;
  let ndc = uv_px * 2.0 - 1.0;
  let aspect = uniforms.resolution.x / uniforms.resolution.y;
  let ray_dir = normalize(
    uniforms.camera_forward +
    ndc.x * aspect * uniforms.fov_factor * uniforms.camera_right +
    ndc.y * uniforms.fov_factor * uniforms.camera_up
  );

  let pt = path_trace(uniforms.camera_pos, ray_dir);

  var diff_color = pt.diffuse;
  var spec_color = pt.specular;

  // Firefly clamp (per-signal)
  let dl = dot(diff_color, vec3f(0.2126, 0.7152, 0.0722));
  if dl > MAX_FIREFLY_LUM { diff_color *= MAX_FIREFLY_LUM / dl; }
  let sl = dot(spec_color, vec3f(0.2126, 0.7152, 0.0722));
  if sl > MAX_FIREFLY_LUM { spec_color *= MAX_FIREFLY_LUM / sl; }

  // SHaRC sparse update with backpropagation (ignis-rt style):
  // Store direct lighting at up to 4 bounce points per path (not just first hit).
  // Cache fills 4× faster and covers more of the scene.
  if pt.depth < 1e5 {
    let bx = pixel.x / 5u; let by = pixel.y / 5u;
    let bh = ((bx * 73856093u) ^ (by * 19349663u) ^ (uniforms.frame_seed * 83492791u));
    let sx = bx * 5u + (bh % 5u); let sy = by * 5u + ((bh / 5u) % 5u);
    if pixel.x == sx && pixel.y == sy {
      sharc_store_radiance(pt.hit_pos, pt.normal, pt.direct); // first hit (no direction)
      for (var si = 0u; si < pt.sharc_count; si++) {
        sharc_store_radiance_dir(pt.sharc_pos[si], pt.sharc_nrm[si], pt.sharc_rad[si], pt.sharc_dir[si]);
      }
    }
  }

  // === ReSTIR GI: temporal radiance reuse ===
  if uniforms.restir_enabled > 0u && pt.depth < 1e5 {
    // Build current reservoir from bounce-1 sample
    var reservoir = empty_reservoir();
    if pt.hit_dist < 1e3 {
      var gi_sample = empty_reservoir();
      gi_sample.radiance = diff_color;
      gi_sample.position = pt.hit_pos;
      gi_sample.normal = pt.normal;
      gi_sample.hit_dist = pt.hit_dist;
      gi_sample.age = 0.0;
      let p_hat = gi_target_pdf(pt.normal, pt.hit_pos, gi_sample);
      if p_hat > 0.0 { reservoir_update(&reservoir, gi_sample, p_hat); }
    }

    // Temporal reuse: reproject to previous frame, merge reservoir
    if uniforms.frame_seed > 0u {
      let prev_uv = restir_reproject(pt.hit_pos);
      if prev_uv.x >= 0.0 && prev_uv.x < 1.0 && prev_uv.y >= 0.0 && prev_uv.y < 1.0 {
        let prev_px = vec2i(vec2f(prev_uv.x * uniforms.resolution.x, prev_uv.y * uniforms.resolution.y));
        let prev_idx = u32(prev_px.y) * res.x + u32(prev_px.x);
        var prev = read_reservoir_prev(prev_idx);

        // Validate: age < 20, normal similarity > 0.5
        if prev.age < 20.0 && prev.M >= 1.0 && dot(pt.normal, prev.normal) > 0.5 {
          prev.age += 1.0;
          prev.M = min(prev.M, 20.0); // clamp M to prevent weight explosion
          let prev_pdf = gi_target_pdf(pt.normal, pt.hit_pos, prev);
          reservoir_merge(&reservoir, prev, prev_pdf);
        }
      }
    }

    // Apply reused GI to diffuse radiance
    if reservoir.weight_sum > 0.0 && reservoir.M > 0.0 {
      let final_pdf = gi_target_pdf(pt.normal, pt.hit_pos, reservoir);
      if final_pdf > 0.0 {
        let W = reservoir.weight_sum / (reservoir.M * final_pdf);
        let reused_gi = reservoir.radiance * final_pdf * W;
        let blend = clamp(reservoir.M / 10.0, 0.0, 0.5);
        diff_color = mix(diff_color, diff_color + reused_gi * 0.3, blend);
      }
    }

    write_reservoir(idx, reservoir);
  } else if uniforms.restir_enabled > 0u {
    // Sky pixel: write empty reservoir
    write_reservoir(idx, empty_reservoir());
  }

  // Adreno: clamp to fp16 range (values >65504 produce artifacts)
  diff_color = min(diff_color, vec3f(65000.0));
  spec_color = min(spec_color, vec3f(65000.0));

  let norm_hit_dist = clamp(log2(pt.hit_dist + 1.0) / 8.0, 0.0, 1.0);

  textureStore(noisy_out, vec2i(pixel), vec4f(diff_color, 1.0));
  textureStore(specular_out, vec2i(pixel), vec4f(spec_color, norm_hit_dist));
  textureStore(albedo_out, vec2i(pixel), vec4f(pt.albedo, pt.roughness));
  textureStore(denoise_nd_out, vec2i(pixel), vec4f(pt.normal, pt.depth));
}
