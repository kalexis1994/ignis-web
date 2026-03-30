// Monte Carlo Path Tracer — WebGPU Compute Shader
// Techniques ported from ignis-rt (NVIDIA):
// - R2 quasi-random jitter (better convergence)
// - GGX Cook-Torrance BRDF + VNDF sampling (Heitz 2018)
// - Real dielectric Fresnel (not Schlick)
// - Smith height-correlated geometry
// - AgX Punchy tone mapping (Blender 4 / Troy Sobotka)
// - Firefly luminance clamping

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
};

struct BVHNode { aabb_min: vec3f, left_first: u32, aabb_max: vec3f, tri_count: u32, };
struct Material {
  albedo: vec3f, mat_type: f32,
  emission: vec3f, roughness: f32,
  metallic: f32, base_tex: f32, mr_tex: f32, normal_tex: f32,
  alpha_mode: f32, alpha_cutoff: f32, ior: f32, emission_strength: f32,
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
// Emissive tris: 4 vec4f each [v0.xyz+area, v1.xyz+CDF, v2.xyz+totalPower, emission.rgb+0]
@group(1) @binding(5) var<storage, read> emissive_tris: array<vec4f>;
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
const SKY_COLOR: vec3f = vec3f(0.8, 0.9, 1.2);
const SUN_ANGLE: f32 = 0.03;
const COS_SUN_ANGLE: f32 = 0.99955;
const SUN_SOLID_ANGLE: f32 = 0.002827;
const SUN_MULT: f32 = 100.0;
const MAX_FIREFLY_LUM: f32 = 8.0;

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

fn sharc_store_radiance(wp: vec3f, n: vec3f, rad: vec3f) {
  let key = sharc_make_key(wp, n);
  let slot = sharc_insert_slot(key);
  if slot == 0xFFFFFFFFu { return; }
  let s = max(rad * 1000.0, vec3f(0.0));
  let aBase = sharc_params.capacity + slot * 7u; // accum: RGBS + DxDyDz
  if u32(s.x) > 0u { atomicAdd(&sharc_keys_accum[aBase], u32(s.x)); }
  if u32(s.y) > 0u { atomicAdd(&sharc_keys_accum[aBase + 1u], u32(s.y)); }
  if u32(s.z) > 0u { atomicAdd(&sharc_keys_accum[aBase + 2u], u32(s.z)); }
  atomicAdd(&sharc_keys_accum[aBase + 3u], 1u);
}

// Store with direction (for indirect bounces — records where light came from)
fn sharc_store_radiance_dir(wp: vec3f, n: vec3f, rad: vec3f, incoming_dir: vec3f) {
  let key = sharc_make_key(wp, n);
  let slot = sharc_insert_slot(key);
  if slot == 0xFFFFFFFFu { return; }
  let s = max(rad * 1000.0, vec3f(0.0));
  let aBase = sharc_params.capacity + slot * 7u;
  if u32(s.x) > 0u { atomicAdd(&sharc_keys_accum[aBase], u32(s.x)); }
  if u32(s.y) > 0u { atomicAdd(&sharc_keys_accum[aBase + 1u], u32(s.y)); }
  if u32(s.z) > 0u { atomicAdd(&sharc_keys_accum[aBase + 2u], u32(s.z)); }
  atomicAdd(&sharc_keys_accum[aBase + 3u], 1u);
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

fn ggx_d(NdotH: f32, alpha: f32) -> f32 {
  let a2 = alpha * alpha;
  let d = NdotH * NdotH * (a2 - 1.0) + 1.0;
  return a2 / (PI * d * d);
}

fn smith_g1(NdotV: f32, alpha: f32) -> f32 {
  let a2 = alpha * alpha;
  return 2.0 * NdotV / (NdotV + sqrt(a2 + (1.0 - a2) * NdotV * NdotV));
}

fn eval_cook_torrance(N: vec3f, V: vec3f, L: vec3f, baseColor: vec3f, roughness: f32, metallic: f32) -> vec3f {
  let NdotL = max(dot(N, L), 0.0);
  let NdotV = max(dot(N, V), 0.001);
  if NdotL <= 0.0 { return vec3f(0.0); }
  let H = normalize(V + L);
  let NdotH = max(dot(N, H), 0.0);
  let VdotH = max(dot(V, H), 0.0);
  let alpha = max(roughness * roughness, 0.001);
  let F0 = mix(vec3f(0.04), baseColor, metallic);
  let F = fresnel_real(VdotH, F0);
  let D = ggx_d(NdotH, alpha);
  let G = smith_g1(NdotL, alpha) * smith_g1(NdotV, alpha);
  let spec = (D * G * F) / max(4.0 * NdotV * NdotL, 0.001);
  let kd = (1.0 - F) * (1.0 - metallic);
  return (kd * baseColor * INV_PI + spec) * NdotL;
}

// Split BRDF evaluation: returns demodulated diffuse irradiance + specular radiance separately
struct BRDFSplit { diffuse: vec3f, specular: vec3f, };

fn eval_ct_split(N: vec3f, V: vec3f, L: vec3f, baseColor: vec3f, roughness: f32, metallic: f32) -> BRDFSplit {
  let NdotL = max(dot(N, L), 0.0);
  let NdotV = max(dot(N, V), 0.001);
  if NdotL <= 0.0 { return BRDFSplit(vec3f(0.0), vec3f(0.0)); }
  let H = normalize(V + L);
  let NdotH = max(dot(N, H), 0.0);
  let VdotH = max(dot(V, H), 0.0);
  let alpha = max(roughness * roughness, 0.001);
  let F0 = mix(vec3f(0.04), baseColor, metallic);
  let F = fresnel_real(VdotH, F0);
  let D = ggx_d(NdotH, alpha);
  let G = smith_g1(NdotL, alpha) * smith_g1(NdotV, alpha);
  let spec = (D * G * F) / max(4.0 * NdotV * NdotL, 0.001);
  let kd = (1.0 - F) * (1.0 - metallic);
  return BRDFSplit(
    vec3f(kd * INV_PI) * NdotL,   // diffuse irradiance (demodulated: no baseColor)
    spec * NdotL                   // specular radiance (includes F0 color)
  );
}

fn sample_sun_nee_split(pos: vec3f, normal: vec3f, V: vec3f, baseColor: vec3f, roughness: f32, metallic: f32) -> BRDFSplit {
  let origin = pos + normal * BIAS;
  var result_split = BRDFSplit(vec3f(0.0), vec3f(0.0));

  // Sun NEE
  var shadow_val = 0.0;
  let L1 = sample_cone(g_sun_dir, COS_SUN_ANGLE);
  let L2 = sample_cone(g_sun_dir, COS_SUN_ANGLE);
  if !trace_shadow(origin, L1, 50.0) { shadow_val += 0.5; }
  if !trace_shadow(origin, L2, 50.0) { shadow_val += 0.5; }
  if shadow_val > 0.0 {
    let L = normalize(L1 + L2);
    let cos_theta = dot(normal, L);
    if cos_theta > 0.0 {
      let brdf = eval_ct_split(normal, V, L, baseColor, roughness, metallic);
      let light = SUN_COLOR * SUN_MULT * SUN_SOLID_ANGLE * shadow_val;
      result_split = BRDFSplit(brdf.diffuse * light, brdf.specular * light);
    }
  }

  // NEE to emissive triangles (CDF importance sampling + MIS, split)
  if uniforms.emissive_tri_count > 0u {
    let rnd = rand();
    var lo = 0u; var hi = uniforms.emissive_tri_count - 1u;
    while lo < hi {
      let mid = (lo + hi) / 2u;
      if rnd <= emissive_tris[mid * 4u + 1u].w { hi = mid; } else { lo = mid + 1u; }
    }
    let base = lo * 4u;
    let d0 = emissive_tris[base]; let d1 = emissive_tris[base+1u];
    let d2 = emissive_tris[base+2u]; let d3 = emissive_tris[base+3u];
    let ev0 = d0.xyz; let ev1 = d1.xyz; let ev2 = d2.xyz;
    let earea = d0.w;
    // Read live emission strength from material buffer (editable at runtime)
    let ematIdx = u32(d3.w + 0.5);
    let estrength = material_buf[ematIdx].emission_strength;
    let eemission = d3.xyz * estrength;
    let prevCdf = select(emissive_tris[(lo - 1u) * 4u + 1u].w, 0.0, lo == 0u);
    let triProb = d1.w - prevCdf;
    var eu = rand(); var ev_r = rand();
    if eu + ev_r > 1.0 { eu = 1.0 - eu; ev_r = 1.0 - ev_r; }
    let epos = ev0 * (1.0 - eu - ev_r) + ev1 * eu + ev2 * ev_r;
    let enormal = normalize(cross(ev1 - ev0, ev2 - ev0));
    let eto = epos - pos;
    let edist = length(eto);
    let edir = eto / max(edist, 1e-6);
    let endotl = dot(normal, edir);
    let ecos_theta = dot(-edir, enormal);
    if endotl > 0.0 && ecos_theta > 0.0 && edist > 0.01 && earea > 1e-6 {
      if !trace_shadow_skip_mat(origin, edir, edist - 0.01, ematIdx) {
        let ebrdf = eval_ct_split(normal, V, edir, baseColor, roughness, metallic);
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
          // Inline alpha test: reject transparent hits without updating hit.t
          // BVH naturally continues to find the next opaque hit behind
          let mat = material_buf[td.w];
          let am = u32(mat.alpha_mode + 0.5);
          if am >= 1u {
            let bw = 1.0 - r.y - r.z;
            let uv = get_uv(td, bw, r.y, r.z);
            let btx = i32(mat.base_tex + 0.5);
            var ta = 1.0;
            if btx >= 0 { ta = textureSampleLevel(tex_array, tex_sampler_pt, uv, btx, 0.0).a; }
            if am == 1u && ta < max(mat.alpha_cutoff, 0.5) { continue; }
            if am == 2u && ta < g_alpha_dither { continue; } // BLEND: R2 spatiotemporal dither
          }
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
          let am = u32(mat.alpha_mode + 0.5);
          if am >= 1u {
            let bw = 1.0 - r.y - r.z;
            let uv = get_uv(td, bw, r.y, r.z);
            let btx = i32(mat.base_tex + 0.5);
            var ta = 1.0;
            if btx >= 0 { ta = textureSampleLevel(tex_array, tex_sampler_pt, uv, btx, 0.0).a; }
            if am == 1u && ta < mat.alpha_cutoff { continue; }
            if am == 2u && ta < g_alpha_dither { continue; }
          }
          let shadow_mat_type = u32(mat.mat_type + 0.5);
          if shadow_mat_type == 3u { continue; }
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
          let am = u32(mat.alpha_mode + 0.5);
          if am >= 1u {
            let bw = 1.0 - r.y - r.z;
            let uv = get_uv(td, bw, r.y, r.z);
            let btx = i32(mat.base_tex + 0.5);
            var ta = 1.0;
            if btx >= 0 { ta = textureSampleLevel(tex_array, tex_sampler_pt, uv, btx, 0.0).a; }
            if am == 1u && ta < mat.alpha_cutoff { continue; }
            if am == 2u && ta < g_alpha_dither { continue; }
          }
          // Glass is transparent to shadow rays
          let shadow_mat_type = u32(mat.mat_type + 0.5);
          if shadow_mat_type == 3u { continue; }
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

fn sample_sun_nee(pos: vec3f, normal: vec3f, V: vec3f, baseColor: vec3f, roughness: f32, metallic: f32) -> vec3f {
  let origin = pos + normal * BIAS;
  var result = vec3f(0.0);

  // Sun NEE: 2 jittered shadow rays averaged
  var shadow_val = 0.0;
  let L1 = sample_cone(g_sun_dir, COS_SUN_ANGLE);
  let L2 = sample_cone(g_sun_dir, COS_SUN_ANGLE);
  if !trace_shadow(origin, L1, 50.0) { shadow_val += 0.5; }
  if !trace_shadow(origin, L2, 50.0) { shadow_val += 0.5; }
  if shadow_val > 0.0 {
    let L = normalize(L1 + L2);
    let cos_theta = dot(normal, L);
    if cos_theta > 0.0 {
      result = SUN_COLOR * SUN_MULT * eval_cook_torrance(normal, V, L, baseColor, roughness, metallic) * SUN_SOLID_ANGLE * shadow_val;
    }
  }

  // NEE to emissive triangles (CDF importance sampling + MIS, ignis-rt format)
  if uniforms.emissive_tri_count > 0u {
    // CDF binary search: select triangle proportional to power (area × luminance)
    let rnd = rand();
    var lo = 0u; var hi = uniforms.emissive_tri_count - 1u;
    while lo < hi {
      let mid = (lo + hi) / 2u;
      let cdf = emissive_tris[mid * 4u + 1u].w; // d1.w = CDF
      if rnd <= cdf { hi = mid; } else { lo = mid + 1u; }
    }
    let eti = lo;
    let base = eti * 4u;
    let d0 = emissive_tris[base]; let d1 = emissive_tris[base+1u];
    let d2 = emissive_tris[base+2u]; let d3 = emissive_tris[base+3u];
    let ev0 = d0.xyz; let ev1 = d1.xyz; let ev2 = d2.xyz;
    let earea = d0.w;
    let ematIdx2 = u32(d3.w + 0.5);
    let estrength2 = material_buf[ematIdx2].emission_strength;
    let eemission = d3.xyz * estrength2;
    let prevCdf = select(emissive_tris[(eti - 1u) * 4u + 1u].w, 0.0, eti == 0u);
    let triProb = d1.w - prevCdf;

    var eu = rand(); var ev_r = rand();
    if eu + ev_r > 1.0 { eu = 1.0 - eu; ev_r = 1.0 - ev_r; }
    let epos = ev0 * (1.0 - eu - ev_r) + ev1 * eu + ev2 * ev_r;
    let enormal = normalize(cross(ev1 - ev0, ev2 - ev0));

    let eto = epos - pos;
    let edist = length(eto);
    let edir = eto / max(edist, 1e-6);
    let endotl = dot(normal, edir);
    let ecos_theta = dot(-edir, enormal);

    if endotl > 0.0 && ecos_theta > 0.0 && edist > 0.01 && earea > 1e-6 {
      if !trace_shadow_skip_mat(origin, edir, edist - 0.01, ematIdx2) {
        let ebrdf = eval_cook_torrance(normal, V, edir, baseColor, roughness, metallic);
        let eradiance = eemission * ecos_theta / (edist * edist);
        let light_pdf = triProb / earea;
        let solid_angle_pdf = light_pdf * edist * edist / ecos_theta;
        let bsdf_pdf = endotl * INV_PI;
        let mis_w = (solid_angle_pdf * solid_angle_pdf) / (solid_angle_pdf * solid_angle_pdf + bsdf_pdf * bsdf_pdf + 1e-8);
        result += ebrdf * eradiance * mis_w / max(light_pdf, 1e-6);
      }
    }
  }

  return result;
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

fn get_uv(td: vec4u, bw: f32, u: f32, v: f32) -> vec2f {
  return bw * vec2f(vertices[td.x].w, vert_normals[td.x].w)
       + u * vec2f(vertices[td.y].w, vert_normals[td.y].w)
       + v * vec2f(vertices[td.z].w, vert_normals[td.z].w);
}

fn apply_normal_map(td: vec4u, N: vec3f, uv: vec2f, tex_idx: i32) -> vec3f {
  if tex_idx < 0 { return N; }
  let ns = textureSampleLevel(tex_array, tex_sampler_pt, uv, tex_idx, 0.0);
  let tn = ns.rgb * 2.0 - 1.0;
  // Compute tangent frame from UV deltas (no stored tangents needed)
  let v0 = vertices[td.x].xyz; let v1 = vertices[td.y].xyz; let v2 = vertices[td.z].xyz;
  let uv0 = vec2f(vertices[td.x].w, vert_normals[td.x].w);
  let uv1 = vec2f(vertices[td.y].w, vert_normals[td.y].w);
  let uv2 = vec2f(vertices[td.z].w, vert_normals[td.z].w);
  let dp1 = v1 - v0; let dp2 = v2 - v0;
  let duv1 = uv1 - uv0; let duv2 = uv2 - uv0;
  let det = duv1.x * duv2.y - duv2.x * duv1.y;
  if abs(det) < 1e-8 { return N; }
  let inv_det = 1.0 / det;
  var T = normalize((dp1 * duv2.y - dp2 * duv1.y) * inv_det);
  T = normalize(T - N * dot(N, T)); // Gram-Schmidt orthogonalization
  let B = cross(N, T);
  return normalize(T * tn.x + B * tn.y + N * tn.z);
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
  // SHaRC backpropagation + path guiding: store up to 4 bounce points
  var sharc_pos: array<vec3f, 4>;
  var sharc_nrm: array<vec3f, 4>;
  var sharc_rad: array<vec3f, 4>;
  var sharc_dir: array<vec3f, 4>; // incoming light direction per bounce
  var sharc_count = 0u;

  for (var bounce = 0u; bounce < uniforms.max_bounces; bounce++) {
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
    let mat_type = u32(mat.mat_type + 0.5);
    let bw = 1.0 - hit.u - hit.v;
    var normal = normalize(bw*vert_normals[td.x].xyz + hit.u*vert_normals[td.y].xyz + hit.v*vert_normals[td.z].xyz);
    let front_face = dot(dir, normal) < 0.0;
    if !front_face { normal = -normal; }
    let hit_pos = origin + dir * hit.t;
    let V = -dir;

    // Interpolate UV
    let uv = get_uv(td, bw, hit.u, hit.v);

    // Sample base color texture
    var base_color = mat.albedo;
    var tex_alpha = 1.0;
    let base_tex_idx = i32(mat.base_tex + 0.5);
    let alpha_mode = u32(mat.alpha_mode + 0.5);
    if base_tex_idx >= 0 {
      let tc = textureSampleLevel(tex_array, tex_sampler_pt, uv, base_tex_idx, 0.0);
      var tex_rgb = srgb_to_linear(tc.rgb);
      tex_alpha = tc.a;
      // BLEND decals: unpremultiply RGB (many decal textures have pre-multiplied alpha)
      if alpha_mode == 2u && tex_alpha > 0.01 {
        tex_rgb /= tex_alpha;
      }
      base_color *= tex_rgb;
    }

    // Sample metallic-roughness texture (G=roughness, B=metallic per GLTF spec)
    var metallic = mat.metallic;
    var roughness = mat.roughness;
    let mr_tex_idx = i32(mat.mr_tex + 0.5);
    if mr_tex_idx >= 0 {
      let mr = textureSampleLevel(tex_array, tex_sampler_pt, uv, mr_tex_idx, 0.0);
      roughness *= mr.g;
      metallic *= mr.b;
    }
    roughness = max(roughness, 0.04);

    // Apply normal map
    let normal_tex_idx = i32(mat.normal_tex + 0.5);
    normal = apply_normal_map(td, normal, uv, normal_tex_idx);

    // Fix normal facing
    if dot(normal, V) < 0.01 {
      normal = normalize(normal + V * 0.2);
    }

    // Capture first-hit G-buffer data
    if bounce == 0u {
      result.normal = normal; result.depth = hit.t; result.hit_pos = hit_pos;
      result.tri_idx = hit.tri_idx; result.bary = vec2f(hit.u, hit.v);
      result.albedo = base_color; result.roughness = roughness;
    }

    // Emission
    if mat_type == 2u {
      if specular_bounce {
        let e = throughput * mat.emission * mat.emission_strength;
        if bounce == 0u { spec_rad += e; }
        else if is_diffuse_path { diff_rad += e; }
        else { spec_rad += e; }
      }
      break;
    }

    // NEE: sun direct lighting — split at bounce 0
    if bounce == 0u {
      let nee = sample_sun_nee_split(hit_pos, normal, V, base_color, roughness, metallic);
      diff_rad += nee.diffuse;
      spec_rad += nee.specular;
      result.direct = nee.diffuse * base_color + nee.specular; // combined for SHaRC
    } else {
      let direct = sample_sun_nee(hit_pos, normal, V, base_color, roughness, metallic);
      if is_diffuse_path { diff_rad += throughput * direct; }
      else { spec_rad += throughput * direct; }
      // SHaRC backpropagation with direction (for path guiding)
      if sharc_count < 4u {
        sharc_pos[sharc_count] = hit_pos;
        sharc_nrm[sharc_count] = normal;
        sharc_rad[sharc_count] = direct;
        sharc_dir[sharc_count] = -dir; // incoming light direction = reverse of bounce dir
        sharc_count += 1u;
      }
    }

    // Bounce 1+: try SHaRC cache for indirect GI
    if bounce >= 1u && mat_type != 3u {
      let cached_gi = sharc_read_cached(hit_pos, normal);
      let has_cache = dot(cached_gi, vec3f(1.0)) > 0.001;
      if has_cache {
        // cached_gi already includes surface albedo (stored as pt.direct with BRDF)
        // Don't multiply by base_color again — that would square the albedo
        let gi = throughput * cached_gi;
        if is_diffuse_path { diff_rad += gi; } else { spec_rad += gi; }
        break;
      }
      let indirect_direct = sample_sun_nee(hit_pos, normal, V, base_color, roughness, metallic);
      let ind = throughput * indirect_direct;
      if is_diffuse_path { diff_rad += ind; } else { spec_rad += ind; }

      // Last bounce: energy terminates. No fake sky — SHaRC + extra bounces provide real GI.
      if bounce >= uniforms.max_bounces - 1u { break; }
    }

    // BRDF sampling — at bounce 0, classify path and demodulate diffuse throughput
    if mat_type == 3u {
      // Glass: solid refraction/reflection (ported from ignis-rt Vulkan)
      // Separate bounce budget — glass doesn't consume main bounces
      glass_bounces += 1u;
      if glass_bounces > 16u { break; }

      let glass_ior = max(mat.ior, 1.01);
      let entering = front_face; // dot(dir, original_normal) < 0 = entering glass
      let refract_n = select(-normal, normal, entering);
      let eta = select(glass_ior, 1.0 / glass_ior, entering);

      // GGX roughness for microfacet refraction
      var alpha_glass = roughness * roughness;
      alpha_glass = max(alpha_glass, 0.001);

      // Microfacet half-vector: smooth glass (< 0.0005) uses flat normal
      var H_glass: vec3f;
      if alpha_glass >= 0.0005 {
        H_glass = sample_ggx_vndf(rand2(), V, refract_n, alpha_glass);
      } else {
        H_glass = refract_n;
      }

      let cos_i = abs(dot(V, H_glass));
      let fresnel = fresnel_dielectric(cos_i, eta);
      let refracted = refract(-V, H_glass, eta);
      let can_refract = dot(refracted, refracted) > 0.0001;

      if !can_refract || rand() < fresnel {
        // Reflection (TIR or Fresnel)
        dir = reflect(-dir, H_glass);
        origin = hit_pos + refract_n * BIAS;
      } else {
        // Refraction
        dir = refracted;
        origin = hit_pos - refract_n * BIAS;
      }

      // Cycles Principled transmission tint: sqrt(baseColor)
      // Simpler than Beer's law, matches Blender exactly
      throughput *= sqrt(clamp(base_color, vec3f(0.0), vec3f(1.0)));

      specular_bounce = roughness < 0.1;
      if bounce == 0u { is_diffuse_path = false; }
      bounce -= 1u;
      continue;
    } else if metallic > 0.5 {
      // Metal / specular-dominant: GGX VNDF sampling → specular path
      let alpha = max(roughness * roughness, 0.001);
      let H = sample_ggx_vndf(rand2(), V, normal, alpha);
      dir = reflect(-V, H);
      if dot(normal, dir) <= 0.0 { break; }
      let VdotH = max(dot(V, H), 0.0);
      let F0 = mix(vec3f(0.04), base_color, metallic);
      let F = fresnel_real(VdotH, F0);
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
      if bounce == 0u {
        is_diffuse_path = true;
        throughput *= vec3f((1.0 - metallic) * pdf_correction);
      } else {
        throughput *= base_color * (1.0 - metallic) * pdf_correction;
      }
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
  }
  result.diffuse = diff_rad;
  result.specular = spec_rad;
  result.sharc_pos = sharc_pos;
  result.sharc_nrm = sharc_nrm;
  result.sharc_rad = sharc_rad;
  result.sharc_dir = sharc_dir;
  result.sharc_count = sharc_count;
  return result;
}

// ============================================================
// Shade from rasterized G-buffer (no primary BVH traversal!)
// Only traces shadow rays + indirect bounces
// ============================================================
fn path_trace_from_gbuffer(hit_pos: vec3f, normal_in: vec3f, view_dir: vec3f, mat_id: u32, uv: vec2f) -> vec3f {
  let mat = material_buf[mat_id];
  let mat_type = u32(mat.mat_type + 0.5);
  let V = -view_dir;
  var normal = normal_in;
  if dot(normal, V) < 0.01 { normal = normalize(normal + V * 0.2); }

  // Sample textures using rasterized UVs
  var base_color = mat.albedo;
  let base_tex_idx = i32(mat.base_tex + 0.5);
  if base_tex_idx >= 0 {
    let tc = textureSampleLevel(tex_array, tex_sampler_pt, uv, base_tex_idx, 0.0);
    base_color *= srgb_to_linear(tc.rgb);
  }
  var roughness = mat.roughness;
  var metallic = mat.metallic;
  let mr_tex_idx = i32(mat.mr_tex + 0.5);
  if mr_tex_idx >= 0 {
    let mr = textureSampleLevel(tex_array, tex_sampler_pt, uv, mr_tex_idx, 0.0);
    roughness *= mr.g;
    metallic *= mr.b;
  }
  roughness = max(roughness, 0.04);

  // Direct lighting (NEE to sun)
  let direct = sample_sun_nee(hit_pos, normal, V, base_color, roughness, metallic);
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
          radiance += base_color * (1.0 - metallic) * cached * bmat.albedo;
        } else {
          let bV = -bounce_dir;
          let ind = sample_sun_nee(bhit_pos, bnormal, bV, bmat.albedo, max(bmat.roughness, 0.04), bmat.metallic);
          radiance += base_color * (1.0 - metallic) * ind;

          // Sky irradiance on last bounce
          if uniforms.max_bounces <= 3u {
            let sky_up = sky_color(bnormal);
            let sky_side = sky_color(vec3f(bnormal.x, 0.0, bnormal.z));
            let sky_irr = mix(sky_side, sky_up, max(bnormal.y, 0.0)) * INV_PI;
            radiance += base_color * (1.0 - metallic) * sky_irr * bmat.albedo;
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
@compute @workgroup_size(8, 8)
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
  // Albedo.rgb for diffuse remodulation, .a = roughness for specular filter
  textureStore(albedo_out, vec2i(pixel), vec4f(pt.albedo, pt.roughness));
  textureStore(denoise_nd_out, vec2i(pixel), vec4f(pt.normal, pt.depth));
}
