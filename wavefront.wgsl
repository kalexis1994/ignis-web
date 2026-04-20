// Wavefront Path Tracer — minimal v1
// Architecture: per-pixel ray state in storage buffer, two kernels re-dispatched
// per bounce. Much smaller per-kernel footprint than the mega-kernel in
// pathtracer.wgsl → better occupancy on Adreno and mobile in general.
//
// v1 scope: BVH closest-hit + any-hit, cosine-diffuse BSDF, NEE sun with
// shadow ray + MIS against simple gradient sky. No textures, no normal maps,
// no GGX spec, no env CDF, no SHaRC/ReSTIR/glass. Those come back in later
// commits once the wavefront architecture is proven stable.
//
// FP16: Adreno has 2× ALU throughput for FP16. Color/radiance/throughput
// operations run in f16 (vec3h); positions/directions/t/PDFs stay f32
// because their magnitudes or precision needs exceed f16 range/precision.
enable f16;

// subgroups: wave-level primitives. Used in the bounce kernel to compact
// survivors with ONE atomicAdd per wave instead of one per surviving
// lane, and to early-out NEE when no lane in the wave needs a shadow.
enable subgroups;

struct Uniforms {
  resolution: vec2f,
  frame_seed: u32,
  max_bounces: u32,
  camera_pos: vec3f,   _pad0: f32,
  camera_forward: vec3f, _pad1: f32,
  camera_right: vec3f, _pad2: f32,
  camera_up: vec3f,    fov_factor: f32,
  sun_dir: vec3f,      frames_still: u32,
  scene_origin: vec3f, emissive_count: u32, // BVH AABB dequantization offset + number of emissive tris in CDF
  scene_scale: vec3f,  _pad4: f32, // BVH AABB dequantization scale (extent / 65535)
  // Previous-frame camera pose for ReSTIR GI temporal motion reprojection.
  // Used by restir_temporal to transform this frame's primary hit world
  // position into the previous frame's NDC, locating the pixel whose
  // reservoir we should consider reusing.
  prev_cam_pos: vec3f,     _pad5: f32,
  prev_cam_forward: vec3f, _pad6: f32,
  prev_cam_right: vec3f,   _pad7: f32,
  prev_cam_up: vec3f,      prev_fov_factor: f32,
};

// BVH node with uint16-quantized AABB relative to scene bounds (20 B vs
// previous 32 B). 6 uint16 packed into 3 u32 for the AABB, plus 2 u32
// for left_first + tri_count. Dequantization uses uniforms.scene_origin
// and uniforms.scene_scale (precomputed = extent / 65535).
struct BVHNode {
  aabb0: u32, // min.x (u16) | min.y (u16)
  aabb1: u32, // min.z (u16) | max.x (u16)
  aabb2: u32, // max.y (u16) | max.z (u16)
  left_first: u32,
  tri_count: u32,
};
struct AABB { mn: vec3f, mx: vec3f, };
fn node_aabb(n: BVHNode) -> AABB {
  let mnx = f32(n.aabb0 & 0xFFFFu);
  let mny = f32(n.aabb0 >> 16u);
  let mnz = f32(n.aabb1 & 0xFFFFu);
  let mxx = f32(n.aabb1 >> 16u);
  let mxy = f32(n.aabb2 & 0xFFFFu);
  let mxz = f32(n.aabb2 >> 16u);
  var out: AABB;
  out.mn = uniforms.scene_origin + vec3f(mnx, mny, mnz) * uniforms.scene_scale;
  out.mx = uniforms.scene_origin + vec3f(mxx, mxy, mxz) * uniforms.scene_scale;
  return out;
}
// Material v2: 2 vec4f = 32 bytes (vs 320 before). 10× less per-hit bandwidth.
//   d0 = (albedo.r, albedo.g, albedo.b, unlit_flag)
//   d1 = (emission.r * strength, emission.g * strength, emission.b * strength, 0)
struct Material { d0: vec4f, d1: vec4f, };

// Per-ray state, 64 bytes, stored as 4 vec4f per pixel in ray_state_buf.
// Packing:
//   ray_state[4i+0] = vec4f(origin.xyz,       bitcast(sampler_dim))
//   ray_state[4i+1] = vec4f(dir.xyz,          bitcast(flags))
//                        flags: bit 0 = alive, bit 1 = diffuse_path
//                               bits 8..15 = bounce count
//   ray_state[4i+2] = vec4f(throughput.xyz,   last_bsdf_pdf)
//   ray_state[4i+3] = vec4f(radiance.xyz,     hit_t)  // hit_t from bounce 1 (denoiser)

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var noisy_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var denoise_nd_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var albedo_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var<storage, read_write> ray_state: array<vec4f>;
// Shadow request buffer: 3 vec4f per pixel, overwritten each bounce.
//   [0] = (origin.xyz, valid_flag: 1.0 = trace it, 0.0 = skip)
//   [1] = (dir.xyz, _pad)
//   [2] = (contribution.xyz, _pad) — added to radiance if shadow ray unblocked
@group(0) @binding(5) var<storage, read_write> shadow_req: array<vec4f>;

@group(1) @binding(0) var<storage, read> vertices: array<vec4f>;
@group(1) @binding(1) var<storage, read> vert_normals: array<vec4f>;
@group(1) @binding(2) var<storage, read> tri_data: array<vec4u>;
@group(1) @binding(3) var<storage, read> bvh_nodes: array<BVHNode>;
@group(1) @binding(4) var<storage, read> material_buf: array<Material>;
// Emissive triangle light list, power-weighted CDF. 4 floats per entry:
//   [0] = sorted-triangle index (u32 bitcast to f32)
//   [1] = triangle area
//   [2] = cumulative CDF (monotonic in [0,1])
//   [3] = unused
// Built by scene-loader.js::loadScene and uploaded by renderer.js.
// Sorted by power descending, capped at MAX_EMISSIVE (see scene-loader).
@group(1) @binding(5) var<storage, read> emissive_tris: array<vec4f>;

// Ray compaction queues — ping-pong between bounces. Each bounce reads
// from one and pushes survivors to the other via atomicAdd on counts.
// dispatch_args is updated by prep_dispatch between bounces to size the
// indirect dispatch to just the active ray count.
@group(2) @binding(0) var<storage, read_write> queue_a: array<u32>;
@group(2) @binding(1) var<storage, read_write> queue_b: array<u32>;
@group(2) @binding(2) var<storage, read_write> counts: array<atomic<u32>, 2>;
@group(2) @binding(3) var<storage, read_write> dispatch_args: array<u32, 3>;

// Pipeline specialization constants — set at pipeline creation so the
// same WGSL source compiles to an a→b or b→a bounce (and a prep that
// reads either queue_a's count or queue_b's count) without a uniform.
override SRC_QUEUE: u32 = 0u;   // bounce: 0 reads queue_a, 1 reads queue_b
override READ_IDX: u32 = 0u;    // prep: which count feeds the next bounce
override FIRST_BOUNCE: u32 = 0u; // bounce: 1 = skip queue read, use gid.x directly

// Temporal accumulation textures (group 3, ping-pong per frame).
// Separate from group 0 so only the composite kernel needs this group;
// gen/bounce/etc. use a leaner pipeline layout without it.
@group(3) @binding(0) var noisy_read: texture_2d<f32>;    // curr frame's finalize output
@group(3) @binding(1) var accum_prev: texture_2d<f32>;    // previous frame's accumulator
@group(3) @binding(2) var accum_new:  texture_storage_2d<rgba16float, write>;
// RELAX intermediate: relax_temporal writes here, anti_firefly reads it.
// Same physical texture, two views — keeps the chain single-buffered.
@group(3) @binding(3) var temp_write: texture_storage_2d<rgba16float, write>;
@group(3) @binding(4) var temp_read:  texture_2d<f32>;

const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;
const INV_PI: f32 = 0.31830988618;
const INF: f32 = 1e30;
const T_MIN: f32 = 1e-4;
const SUN_COS_HALF: f32 = 0.9999619; // ~0.5° angular radius
const SUN_RADIANCE: vec3f = vec3f(6.5, 6.0, 5.0); // punchy sun, tonemap trims
// Firefly filter threshold. Caps max RGB channel post-scale by
// firefly_k (path tracker) or nee_k (NEE-specific). 6.0 is a balance:
// allows direct sun-lit surfaces (~6.5 RGB) and close-view emitters
// near full emission intensity, while clamping amplification spikes
// (rare CDF picks × 1/d² × 1/p_pick). The d² floor and NEE-derived k
// do the heavy lifting for NEE-specific firefly sources; the global
// threshold is a secondary safety net.
const FIREFLY_THRESHOLD: f32 = 6.0;

const FLAG_ALIVE: u32 = 1u;
const FLAG_DIFFUSE_PATH: u32 = 2u;

// ============================================================
// Sampler — Owen-scrambled low-discrepancy sequences.
// Based on open academic publications:
//   Burley 2020 — "Practical Hash-Based Owen Scrambling" (JCGT 9(4))
//   Heitz 2021 — extension used by NVIDIA RTXPT
// Algorithm: for sample i / dim d / per-pixel seed, apply Owen scramble
// to a bit-reversed counter (Van der Corput for dim 0). Produces near-
// blue-noise spatial distribution → fine-grained noise instead of the
// "clumpy" white noise of unstratified PCG.
// ============================================================
fn hash_u32(x_in: u32) -> u32 {
  // PCG32 single-step hash (Jarzynski-Olano 2020, good avalanche)
  var h = x_in * 747796405u + 2891336453u;
  let w = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
  return (w >> 22u) ^ w;
}

fn reverse_bits_u32(x_in: u32) -> u32 {
  var x = x_in;
  x = ((x >> 1u) & 0x55555555u) | ((x & 0x55555555u) << 1u);
  x = ((x >> 2u) & 0x33333333u) | ((x & 0x33333333u) << 2u);
  x = ((x >> 4u) & 0x0F0F0F0Fu) | ((x & 0x0F0F0F0Fu) << 4u);
  x = ((x >> 8u) & 0x00FF00FFu) | ((x & 0x00FF00FFu) << 8u);
  x = (x >> 16u) | (x << 16u);
  return x;
}

// Laine-Karras hash-based Owen scramble, 4-round variant (Burley 2020 JCGT).
// The magic multipliers are the ones published in that paper, tuned so that
// each nested-uniform scrambling tree node gets an independently random flip.
fn owen_scramble(x_in: u32, seed: u32) -> u32 {
  var x = reverse_bits_u32(x_in);
  x = x + seed;
  x = x ^ (x * 0x6c50b47cu);
  x = x ^ (x * 0xb82f1e52u);
  x = x ^ (x * 0xc7afe638u);
  x = x ^ (x * 0x8d22f6e6u);
  return reverse_bits_u32(x);
}

// Sampler state: per-pixel base seed + temporal sample index + dimension
// counter. Lives in private/function memory per lane; persisted across
// bounces via ray_state (only `dim` needs storing; pixel_hash and sample
// are recomputed from pixel_idx and uniforms.frame_seed each kernel).
struct Sampler {
  pixel_hash: u32,
  sample: u32,
  dim: u32,
};

fn sampler_init(idx: u32, frame: u32, start_dim: u32) -> Sampler {
  var s: Sampler;
  // Mezclar frame dentro del pixel_hash. Las multiplicadores LK de
  // Burley 2020 son todos pares → 0x80000000*K ≡ 0 mod 2^32, con lo
  // que la diferencia entre samples adyacentes (bit 31 tras reverse)
  // se propaga como XOR puro y termina en bit 0 del output, que luego
  // cae en el `>> 8u`. Mezclando frame acá, shuffle_seed/dim_seed
  // varían per-frame y no dependemos del Owen para la dimensión temporal.
  s.pixel_hash = hash_u32(idx * 0x9E3779B1u + frame * 0x85EBCA77u + 0x165667B1u);
  s.sample = frame;                // sigue usándose para Owen (Sobol vdc)
  s.dim = start_dim;
  return s;
}

// Sample next 1D value in [0,1). Advances dim counter.
fn sampler_1d(s: ptr<function, Sampler>) -> f32 {
  let dim = (*s).dim;
  let shuffle_seed = hash_u32((*s).pixel_hash + (dim * 0x9E3779B9u));
  let dim_seed     = hash_u32((*s).pixel_hash + (dim * 0x85EBCA77u) + 0x1u);
  let shuffled = owen_scramble((*s).sample, shuffle_seed);
  // For dim 0 just bit-reverse (Van der Corput / Sobol d=0); higher dims
  // would need true Sobol direction matrices — we approximate via Owen-
  // scrambling the shuffled index again with a per-dim seed, which gives
  // a full set of blue-noise-distributed 1D samples (the approach used in
  // Andrew Helmer's shadertoy demo referenced by RTXPT).
  let x = owen_scramble(shuffled, dim_seed);
  (*s).dim = dim + 1u;
  // Top 24 bits have best distribution; convert to float in [0,1).
  return f32(x >> 8u) / 16777216.0;
}

fn sampler_2d(s: ptr<function, Sampler>) -> vec2f {
  let x = sampler_1d(s);
  let y = sampler_1d(s);
  return vec2f(x, y);
}

// ============================================================
// Sampling helpers
// ============================================================
fn build_onb(n: vec3f) -> mat3x3f {
  let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.999);
  let t = normalize(cross(up, n));
  let b = cross(n, t);
  return mat3x3f(t, b, n);
}

fn cosine_sample_hemisphere(normal: vec3f, u: vec2f) -> vec3f {
  let phi = TWO_PI * u.x;
  let sr2 = sqrt(u.y);
  let local = vec3f(cos(phi)*sr2, sin(phi)*sr2, sqrt(max(0.0, 1.0 - u.y)));
  return normalize(build_onb(normal) * local);
}

fn sample_cone(axis: vec3f, cos_theta_max: f32, u: vec2f) -> vec3f {
  let cos_theta = 1.0 + u.x * (cos_theta_max - 1.0);
  let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
  let phi = TWO_PI * u.y;
  let local = vec3f(cos(phi)*sin_theta, sin(phi)*sin_theta, cos_theta);
  return normalize(build_onb(axis) * local);
}

// ============================================================
// Ray offset (Wächter & Binder 2019) — robust against self-intersection
// ============================================================
fn ray_offset(P: vec3f, Ng: vec3f) -> vec3f {
  let int_scale = 256.0;
  let oi = vec3i(vec3f(int_scale) * Ng);
  let pi = vec3f(
    bitcast<f32>(bitcast<i32>(P.x) + select(oi.x, -oi.x, P.x < 0.0)),
    bitcast<f32>(bitcast<i32>(P.y) + select(oi.y, -oi.y, P.y < 0.0)),
    bitcast<f32>(bitcast<i32>(P.z) + select(oi.z, -oi.z, P.z < 0.0)),
  );
  let origin_threshold = 1.0 / 32.0;
  let float_scale = 1.0 / 65536.0;
  return vec3f(
    select(pi.x, P.x + float_scale * Ng.x, abs(P.x) < origin_threshold),
    select(pi.y, P.y + float_scale * Ng.y, abs(P.y) < origin_threshold),
    select(pi.z, P.z + float_scale * Ng.z, abs(P.z) < origin_threshold),
  );
}

// ============================================================
// Ray/primitive intersection
// ============================================================
struct HitInfo { t: f32, u: f32, v: f32, tri_idx: u32, hit: bool, };

fn intersect_aabb(origin: vec3f, inv_dir: vec3f, mn: vec3f, mx: vec3f, max_t: f32) -> f32 {
  let t0 = (mn - origin) * inv_dir;
  let t1 = (mx - origin) * inv_dir;
  let tsmall = min(t0, t1);
  let tlarge = max(t0, t1);
  let tmin = max(max(tsmall.x, tsmall.y), max(tsmall.z, T_MIN));
  let tmax = min(min(tlarge.x, tlarge.y), min(tlarge.z, max_t));
  return select(max_t, tmin, tmax >= tmin);
}

fn intersect_tri(origin: vec3f, dir: vec3f, a: vec3f, b: vec3f, c: vec3f, max_t: f32) -> vec3f {
  let e1 = b - a; let e2 = c - a;
  let p = cross(dir, e2);
  let det = dot(e1, p);
  if abs(det) < 1e-8 { return vec3f(max_t, 0.0, 0.0); }
  let inv_det = 1.0 / det;
  let tvec = origin - a;
  let u = dot(tvec, p) * inv_det;
  if u < 0.0 || u > 1.0 { return vec3f(max_t, 0.0, 0.0); }
  let q = cross(tvec, e1);
  let v = dot(dir, q) * inv_det;
  if v < 0.0 || u + v > 1.0 { return vec3f(max_t, 0.0, 0.0); }
  let t = dot(e2, q) * inv_det;
  if t < T_MIN { return vec3f(max_t, 0.0, 0.0); }
  return vec3f(t, u, v);
}

// ============================================================
// BVH traversal — closest hit (v1: opaque-only, no alpha)
// ============================================================
fn trace_bvh(origin: vec3f, dir: vec3f) -> HitInfo {
  var hit: HitInfo; hit.t = INF; hit.hit = false;
  let inv_dir = 1.0 / dir;
  var stk: array<u32, 12>; var sp = 0u; var cur = 0u;
  let root_ab = node_aabb(bvh_nodes[0u]);
  if intersect_aabb(origin, inv_dir, root_ab.mn, root_ab.mx, INF) >= INF { return hit; }
  loop {
    let nd = bvh_nodes[cur];
    if nd.tri_count > 0u {
      for (var i = 0u; i < nd.tri_count; i++) {
        let ti = nd.left_first + i;
        let td = tri_data[ti];
        let r = intersect_tri(origin, dir,
          vertices[td.x].xyz, vertices[td.y].xyz, vertices[td.z].xyz, hit.t);
        if r.x < hit.t {
          hit.t = r.x; hit.u = r.y; hit.v = r.z;
          hit.tri_idx = ti; hit.hit = true;
        }
      }
      if sp == 0u { break; } sp--; cur = stk[sp]; continue;
    }
    let l = nd.left_first; let r = l + 1u;
    let lab = node_aabb(bvh_nodes[l]);
    let rab = node_aabb(bvh_nodes[r]);
    let tl = intersect_aabb(origin, inv_dir, lab.mn, lab.mx, hit.t);
    let tr = intersect_aabb(origin, inv_dir, rab.mn, rab.mx, hit.t);
    if tl < tr {
      if tr < hit.t && sp < 12u { stk[sp] = r; sp++; }
      if tl < hit.t { cur = l; } else { if sp == 0u { break; } sp--; cur = stk[sp]; }
    } else {
      if tl < hit.t && sp < 12u { stk[sp] = l; sp++; }
      if tr < hit.t { cur = r; } else { if sp == 0u { break; } sp--; cur = stk[sp]; }
    }
  }
  return hit;
}

// BVH any-hit for shadows (returns true if blocked before max_t)
fn trace_shadow(origin: vec3f, dir: vec3f, max_t: f32) -> bool {
  let inv_dir = 1.0 / dir;
  var stk: array<u32, 12>; var sp = 0u; var cur = 0u;
  let root_ab = node_aabb(bvh_nodes[0u]);
  if intersect_aabb(origin, inv_dir, root_ab.mn, root_ab.mx, max_t) >= max_t { return false; }
  loop {
    let nd = bvh_nodes[cur];
    if nd.tri_count > 0u {
      for (var i = 0u; i < nd.tri_count; i++) {
        let ti = nd.left_first + i;
        let td = tri_data[ti];
        let r = intersect_tri(origin, dir,
          vertices[td.x].xyz, vertices[td.y].xyz, vertices[td.z].xyz, max_t);
        if r.x < max_t { return true; }
      }
      if sp == 0u { break; } sp--; cur = stk[sp]; continue;
    }
    let l = nd.left_first; let r = l + 1u;
    let lab = node_aabb(bvh_nodes[l]);
    let rab = node_aabb(bvh_nodes[r]);
    let tl = intersect_aabb(origin, inv_dir, lab.mn, lab.mx, max_t);
    let tr = intersect_aabb(origin, inv_dir, rab.mn, rab.mx, max_t);
    if tl < tr {
      if tr < max_t && sp < 12u { stk[sp] = r; sp++; }
      if tl < max_t { cur = l; } else { if sp == 0u { break; } sp--; cur = stk[sp]; }
    } else {
      if tl < max_t && sp < 12u { stk[sp] = l; sp++; }
      if tr < max_t { cur = r; } else { if sp == 0u { break; } sp--; cur = stk[sp]; }
    }
  }
  return false;
}

// ============================================================
// Sky — simple analytic gradient (v1). Sun is NOT drawn here because
// it's accounted for via NEE as a delta light — drawing the disk too
// would double-count. Brightness tuned high enough for indoor scenes
// (Sponza) to pick up visible indirect via skylight. Env CDF later.
// ============================================================
fn sky_color(dir: vec3f) -> vec3f {
  let t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
  let horizon = vec3f(0.9, 0.85, 0.75);
  let zenith = vec3f(0.4, 0.6, 1.0);
  return mix(horizon, zenith, t) * 3.0;
}

// ============================================================
// Firefly filter (ported from RTXPT v1.8.1)
//   PathTracer/PathTracerHelpers.hlsli:180-219
//
// Per-path scalar `firefly_k` (stored in ray_state) starts at 1.0 and
// shrinks after every BSDF scatter by a factor derived from the ray-
// cone spread angle implied by that scatter's pdf. Low-pdf (wide, near-
// diffuse) lobes collapse k quickly; high-pdf (near-specular) lobes
// preserve k. Downstream emission contributions are then clamped to
// FIREFLY_THRESHOLD * k so rare high-variance paths (e.g. a wide
// diffuse bounce that lands on a bright emitter) don't spike single
// pixels. Uses RGB average instead of luminance per RTXPT comment —
// luminance causes a hue shift toward blue under clamp.
// ============================================================
fn luminance(c: vec3f) -> f32 {
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

// Handbook polynomial acos, max error ~7e-3 rad. Adequate for the
// ray-cone heuristic which is already empirical.
fn fast_acos(x: f32) -> f32 {
  let a = abs(x);
  var r = -0.0187293 * a + 0.0742610;
  r = r * a - 0.2121144;
  r = r * a + 1.5707288;
  r = r * sqrt(max(0.0, 1.0 - a));
  return select(PI - r, r, x >= 0.0);
}

// Ray-cone spread angle (plane angle) derived from sphere-cap solid
// angle omega = 1/pdf. alpha = 2·acos(1 − omega/2π). RTXPT uses a
// growth factor 0.3 for MIP LOD but 1.0 for firefly-K so diffuse
// lobes collapse k faster.
fn ray_cone_spread_by_pdf(pdf: f32) -> f32 {
  if pdf <= 0.0 { return 0.0; }
  return 2.0 * fast_acos(max(-1.0, 1.0 - (1.0 / pdf) / TWO_PI));
}

fn update_firefly_k(current_k: f32, bounce_pdf: f32, lobe_p: f32) -> f32 {
  let angle = ray_cone_spread_by_pdf(bounce_pdf);
  let k_emp: f32 = 32.0;                       // RTXPT empirical
  var p = k_emp / (k_emp + angle * angle);
  p *= sqrt(max(lobe_p, 0.0));                 // single-lobe: lobe_p = 1
  return max(0.00001, current_k * p);
}

fn firefly_filter(signal: vec3f, threshold: f32, k: f32) -> vec3f {
  let t = threshold * k;
  // RTXPT uses average-RGB to avoid hue shift toward blue under
  // luminance-based clamping, but chromatic emitters (pure-red
  // candles, cyan neon) let single-channel spikes slip through the
  // average. Max-channel catches every spike regardless of hue and
  // preserves color ratios by scaling uniformly — same property as
  // RTXPT's rationale, without the hue bias.
  let max_c = max(max(signal.x, signal.y), signal.z);
  if max_c > t {
    return signal * (t / max_c);
  }
  return signal;
}

// ============================================================
// Emissive-triangle NEE sampling helpers
//
// The emissive_tris buffer is a power-weighted CDF built host-side:
// entry i carries the sorted-tri index, the triangle area, and the
// cumulative probability mass (ascending, last entry = 1.0). We
// importance-sample it with a binary search on a uniform u ∈ [0,1),
// returning the slot whose CDF range [cdf[i-1], cdf[i]) contains u.
// The pick probability is cdf[i] - cdf[i-1] (or cdf[0] for i=0).
//
// Uniform point on a triangle uses the standard fold:
//   if (u1+u2) > 1: u1,u2 ← (1-u1, 1-u2)
// which maps the [0,1]² square onto the triangle (α=u1, β=u2, γ=1-α-β).
// ============================================================
fn sample_emissive_cdf(u: f32) -> u32 {
  let n = uniforms.emissive_count;
  if n == 0u { return 0u; }
  var lo: u32 = 0u;
  var hi: u32 = n;
  while lo < hi {
    let mid = (lo + hi) >> 1u;
    if u < emissive_tris[mid].z { hi = mid; } else { lo = mid + 1u; }
  }
  return min(lo, n - 1u);
}

fn emissive_pick_pdf(i: u32) -> f32 {
  let cdf_curr = emissive_tris[i].z;
  // Guard: `select` evaluates both branches, so conditioning the index
  // expression inside select would wrap i=0 to 0xFFFFFFFF. Compute the
  // guarded index explicitly before the load.
  var prev_idx: u32 = 0u;
  if i > 0u { prev_idx = i - 1u; }
  let cdf_prev = select(0.0, emissive_tris[prev_idx].z, i > 0u);
  return max(cdf_curr - cdf_prev, 1e-20);
}

fn sample_triangle_uniform(u_in: vec2f, a: vec3f, b: vec3f, c: vec3f) -> vec3f {
  var uu = u_in;
  if uu.x + uu.y > 1.0 { uu = vec2f(1.0 - uu.x, 1.0 - uu.y); }
  return a + uu.x * (b - a) + uu.y * (c - a);
}

// ============================================================
// Ray state load / store (packed into array<vec4f>)
// ============================================================
struct RayState {
  origin: vec3f,
  dir: vec3f,
  throughput: vec3f,
  radiance: vec3f,        // kept so legacy `finalize` still compiles
  sampler_dim: u32,
  flags: u32,
  last_bsdf_pdf: f32,
  firefly_k: f32,         // RTXPT firefly-filter tracker (repurposed hit_t)
};

fn load_ray_state(idx: u32) -> RayState {
  let base = idx * 4u;
  let r0 = ray_state[base];
  let r1 = ray_state[base + 1u];
  let r2 = ray_state[base + 2u];
  let r3 = ray_state[base + 3u];
  var rs: RayState;
  rs.origin = r0.xyz;
  rs.sampler_dim = bitcast<u32>(r0.w);
  rs.dir = r1.xyz;
  rs.flags = bitcast<u32>(r1.w);
  rs.throughput = r2.xyz;
  rs.last_bsdf_pdf = r2.w;
  rs.radiance = r3.xyz;
  rs.firefly_k = r3.w;
  return rs;
}

fn store_ray_state(idx: u32, rs: RayState) {
  let base = idx * 4u;
  ray_state[base]       = vec4f(rs.origin,    bitcast<f32>(rs.sampler_dim));
  ray_state[base + 1u]  = vec4f(rs.dir,       bitcast<f32>(rs.flags));
  ray_state[base + 2u]  = vec4f(rs.throughput, rs.last_bsdf_pdf);
  ray_state[base + 3u]  = vec4f(rs.radiance,  rs.firefly_k);
}

fn bounce_of(flags: u32) -> u32 { return (flags >> 8u) & 0xFFu; }
fn set_bounce(flags: u32, b: u32) -> u32 { return (flags & 0xFFFF00FFu) | ((b & 0xFFu) << 8u); }

// ============================================================
// Material access helpers (v1: core fields only)
// ============================================================
// Material v2 accessors — only the fields v1 actually uses.
fn mat_base_color(m: Material) -> vec3f { return m.d0.xyz; }
fn mat_emission(m: Material) -> vec3f { return m.d1.xyz; } // already premultiplied with strength by scene-loader
fn mat_is_unlit(m: Material) -> bool { return m.d0.w > 0.5; }

fn triangle_geo_normal(td: vec4u) -> vec3f {
  let a = vertices[td.x].xyz; let b = vertices[td.y].xyz; let c = vertices[td.z].xyz;
  return normalize(cross(b - a, c - a));
}

// ============================================================
// Kernel: GENERATE — one dispatch per frame. Writes primary rays.
// ============================================================
@compute @workgroup_size(8, 8)
fn generate(@builtin(global_invocation_id) gid: vec3u) {
  let pixel = vec2u(gid.xy);
  let res = vec2u(uniforms.resolution);
  if pixel.x >= res.x || pixel.y >= res.y { return; }
  let idx = pixel.y * res.x + pixel.x;

  // Owen-scrambled Sobol sampler: per-pixel seed + temporal sample index.
  // Dim 0/1 = sub-pixel jitter; subsequent bounces continue the dim
  // sequence (stored as s.dim across bounces via ray_state).
  var s = sampler_init(idx, uniforms.frame_seed, 0u);

  // Sub-pixel jitter (consumes sampler dims 0-1)
  let j = sampler_2d(&s);
  let uv_px = (vec2f(f32(pixel.x), f32(pixel.y)) + j) / uniforms.resolution;
  let ndc = uv_px * 2.0 - 1.0;
  let aspect = uniforms.resolution.x / uniforms.resolution.y;
  let dir = normalize(
    uniforms.camera_forward
    + ndc.x * aspect * uniforms.fov_factor * uniforms.camera_right
    + ndc.y * uniforms.fov_factor * uniforms.camera_up
  );

  var rs: RayState;
  rs.origin = uniforms.camera_pos;
  rs.dir = dir;
  rs.throughput = vec3f(1.0);
  rs.radiance = vec3f(0.0);
  rs.sampler_dim = s.dim; // = 2 after jitter consumed dims 0,1
  rs.flags = FLAG_ALIVE | FLAG_DIFFUSE_PATH;
  rs.last_bsdf_pdf = 0.0;
  rs.firefly_k = 1.0;     // identity; shrinks on each BSDF scatter
  store_ray_state(idx, rs);

  // Reset ReSTIR GI candidate for this pixel at frame start. All fields
  // zero → valid=0 → shade treats indirect as 0 unless bounce captures.
  cand_reset(idx);
  // Invalidate current-frame G-buffer slot. Bounce overwrites it at
  // b==0 on a primary hit in the Lambertian path; miss / unlit / out-
  // of-frustum pixels keep valid=0 so temporal reprojection skips them.
  // Without this, the ping-pong buffer would hold data from TWO frames
  // ago and falsely pass validation.
  gbuf_curr_invalidate(idx);

  // NO queue_a[idx] = idx init — the first bounce pipeline is compiled with
  // FIRST_BOUNCE=1 and reads gid.x directly. Subsequent bounces write
  // their own survivors, so queue_a only gets populated from bounce 1 on.
}

// ============================================================
// Kernel: BOUNCE — one dispatch per bounce iteration.
// Reads ray_state, traces, shades with NEE sun, samples cosine
// hemisphere for next ray, updates state.
// ============================================================
@compute @workgroup_size(64)
fn bounce(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(subgroup_invocation_id) sgid: u32,
) {
  // No early `return`s in this kernel — the wave-level compaction at the
  // bottom uses subgroup ops that require subgroup-uniform control flow
  // (all lanes must reach them). Instead, each lane carries a `processing`
  // flag; lanes that bail out keep running but skip work blocks, and
  // divergent "bail" inside the processing block uses `loop{break}` as a
  // goto-forward so all lanes still rejoin at the function's single exit.

  let qi = gid.x;
  let res = vec2u(uniforms.resolution);

  var idx: u32 = 0u;
  var processing = true;

  // First bounce: every primary ray is alive, so pixel_idx == gid.x and
  // we skip the queue-read indirection entirely. Subsequent bounces: read
  // pixel_idx from the source queue.
  if FIRST_BOUNCE == 1u {
    if qi >= res.x * res.y { processing = false; }
    idx = qi;
  } else {
    let src_count = atomicLoad(&counts[SRC_QUEUE]);
    if qi >= src_count {
      processing = false;
    } else {
      if SRC_QUEUE == 0u { idx = queue_a[qi]; } else { idx = queue_b[qi]; }
    }
  }

  var rs: RayState;
  if processing {
    // Clear shadow request upfront — any "bail" inside the work loop below
    // leaves it zero so shadow_trace skips this pixel this bounce.
    shadow_req[idx * 3u] = vec4f(0.0);
    rs = load_ray_state(idx);
    if (rs.flags & FLAG_ALIVE) == 0u { processing = false; }
  }

  if processing {
    // Resume the sampler from where the previous bounce left off.
    var s = sampler_init(idx, uniforms.frame_seed, rs.sampler_dim);
    let b = bounce_of(rs.flags);
    // Single-iter loop = forward-goto. `break` = "done with this lane's
    // work for this bounce, skip to the store + compaction at the end".
    // ReSTIR GI note: this kernel splits radiance into "direct" (primary
    // hit emission + sun NEE at primary) and "Lo" (indirect from the
    // captured sample point x_s onwards). rs.throughput is re-purposed
    // as the SAMPLE-FRAME throughput: stays 1.0 across bounce-0's BSDF
    // sample (we enter bounce 1 with throughput=1 at x_s) and compounds
    // with base_color only at b >= 1. The BRDF*cos/pdf cancel at the
    // primary hit is applied in restir_shade as "base_primary * Lo", so
    // direct + base_primary * Lo == old rs.radiance for M=1 no-reuse.
    loop {
      let hit = trace_bvh(rs.origin, rs.dir);
      if !hit.hit {
        // Miss. Apply firefly filter at all depths (at primary k=1 the
        // cap is effectively baseline threshold; at deeper bounces it
        // tightens as k has collapsed through prior scatters).
        let sky = firefly_filter(sky_color(rs.dir), FIREFLY_THRESHOLD, rs.firefly_k);
        if b == 0u {
          // Primary miss → sky is direct (no visible point, valid stays 0)
          cand_add_direct(idx, sky);
        } else {
          // Indirect miss → sky into Lo via sample-frame throughput, plus
          // a synthetic "far" sample so shade uses Lo (valid=1). Lo lives
          // in candidate_buf (not ray_state) so restir_shade can read it.
          let sky_h = vec3h(sky);
          let tp_h  = vec3h(rs.throughput);
          cand_add_Lo(idx, vec3f(tp_h * sky_h));
          let far_pos = rs.origin + rs.dir * 1e5;
          cand_set_sample(idx, far_pos, -rs.dir);
        }
        rs.flags &= ~FLAG_ALIVE;
        break;
      }

      let td = tri_data[hit.tri_idx];
      let mat = material_buf[td.w];
      let hit_pos = rs.origin + rs.dir * hit.t;
      let geo_normal_raw = triangle_geo_normal(td);
      let front_face = dot(rs.dir, geo_normal_raw) < 0.0;
      let geo_normal = select(-geo_normal_raw, geo_normal_raw, front_face);
      let bw = 1.0 - hit.u - hit.v;
      var normal = normalize(
        bw * vert_normals[td.x].xyz
        + hit.u * vert_normals[td.y].xyz
        + hit.v * vert_normals[td.z].xyz
      );
      if !front_face { normal = -normal; }
      if dot(normal, -rs.dir) < 0.01 { normal = geo_normal; } // fallback

      // Capture sample point at bounce 1 hit (visible-from-primary surface).
      if b == 1u {
        cand_set_sample(idx, hit_pos, normal);
      }

      // Emission at hit.
      //   b == 0: always added (direct view of emitter — independent of NEE).
      //   b >= 1: only when emissive NEE is NOT available. If it is, the
      //           same integrand is covered by NEE at the previous vertex;
      //           summing both without MIS would double-count. When MIS
      //           between BSDF and NEE lands (future phase with GGX), this
      //           becomes `em * w_bsdf` at all depths.
      // Firefly filter applied at source so the ReSTIR reservoir never
      // locks in an unfiltered spike for future temporal reuse.
      let em = mat_emission(mat);
      if dot(em, vec3f(1.0)) > 1e-6 {
        let em_ff = firefly_filter(em, FIREFLY_THRESHOLD, rs.firefly_k);
        if b == 0u {
          cand_add_direct(idx, em_ff);
        } else if uniforms.emissive_count == 0u {
          cand_add_Lo(idx, rs.throughput * em_ff);
        }
      }

      if mat_is_unlit(mat) {
        let unlit_ff = firefly_filter(mat_base_color(mat), FIREFLY_THRESHOLD, rs.firefly_k);
        if b == 0u {
          cand_add_direct(idx, unlit_ff);
        } else {
          cand_add_Lo(idx, rs.throughput * unlit_ff);
        }
        rs.flags &= ~FLAG_ALIVE;
        break;
      }

      // Lambertian path — scene-loader defaults metallic to 1.0 per glTF
      // spec when the material expects a texture, so we ignore metallic.
      let base_h   = vec3h(mat_base_color(mat));
      let albedo_h = base_h * f16(INV_PI);
      let throughput_h = vec3h(rs.throughput);

      // NEE sun (delta). Flag 1.0 = direct (b==0), 2.0 = indirect (b>=1),
      // decoded by shadow_trace to route contribution to candidate.direct
      // vs. rs.radiance (Lo). At indirect depth, firefly-filter the
      // contribution with the path's current k (delta light has no
      // derivative k update — pdf → ∞ keeps k unchanged).
      let nsl = dot(normal, uniforms.sun_dir);
      if nsl > 0.0 {
        var contribution = vec3f(throughput_h * albedo_h * vec3h(SUN_RADIANCE) * f16(nsl));
        if b > 0u {
          contribution = firefly_filter(contribution, FIREFLY_THRESHOLD, rs.firefly_k);
        }
        let sun_origin = ray_offset(hit_pos, geo_normal);
        let sbase = idx * 3u;
        let nee_flag = select(2.0, 1.0, b == 0u);
        shadow_req[sbase]      = vec4f(sun_origin,     nee_flag);
        shadow_req[sbase + 1u] = vec4f(uniforms.sun_dir, 0.0);
        shadow_req[sbase + 2u] = vec4f(contribution,   0.0);
      }

      // NEE to an emissive triangle. Picks one tri via power-weighted CDF,
      // samples a uniform point on it, and traces a shadow ray inline
      // (the dedicated shadow_trace kernel has only one request slot per
      // pixel and it's already used by the sun above). Contribution:
      //   BRDF * Le * cos_surf / p_sa
      // where the solid-angle pdf converting from area sampling is
      //   p_sa = p_pick * d² / (area * cos_light)
      // with p_pick = cdf[i]-cdf[i-1], cos_light = dot(-w, n_l).
      //
      // At b >= 1 this is the ONLY channel for emissive GI — the BSDF-hit
      // emission accumulation below is gated off when emissives exist to
      // avoid double-counting (NEE at vertex N-1 and BSDF hit of emissive
      // at vertex N sample the same integrand). MIS between the two is
      // the proper fix and a future step; for v1 pure-NEE is unbiased.
      if uniforms.emissive_count > 0u {
        let u_pick = sampler_1d(&s);
        let u_tri  = sampler_2d(&s);
        let ei = sample_emissive_cdf(u_pick);
        let et = emissive_tris[ei];
        let e_tri_idx = bitcast<u32>(et.x);
        let e_area    = et.y;
        let p_pick    = emissive_pick_pdf(ei);

        let etd = tri_data[e_tri_idx];
        let v0 = vertices[etd.x].xyz;
        let v1 = vertices[etd.y].xyz;
        let v2 = vertices[etd.z].xyz;
        let x_l = sample_triangle_uniform(u_tri, v0, v1, v2);
        let n_l = normalize(cross(v1 - v0, v2 - v0));

        let w         = x_l - hit_pos;
        // Distance² floor: the solid-angle pdf has a 1/d² singularity
        // that spikes hard at corners/edges where an emissive surface
        // sits within a few cm of the shading point. Flooring d² at
        // 0.01 (= d = 0.1 m) bounds 1/d² ≤ 100 regardless of geometry
        // pathology. Biases very-near emitter illumination downward by
        // at most the ratio (true d² / 0.01), which for scene-scale
        // geometries (m units) is a small underestimation in a narrow
        // corner band and catches the firefly tail elsewhere.
        let d_sq_raw  = dot(w, w);
        let d_sq      = max(d_sq_raw, 0.01);
        let d         = sqrt(d_sq);
        let w_hat     = w / sqrt(max(d_sq_raw, 1e-20));
        let cos_surf  = dot(normal, w_hat);
        let cos_light = -dot(n_l, w_hat);

        if cos_surf > 1e-4 && cos_light > 1e-4 && d_sq_raw > 1e-8 {
          let e_mat = material_buf[etd.w];
          let Le = mat_emission(e_mat);
          // Solid-angle pdf: p_pick * d² / (area * cos_light). Uses the
          // floored d_sq so the downstream 1/pdf_sa factor is capped.
          let pdf_sa = p_pick * d_sq / max(e_area * cos_light, 1e-20);
          // BRDF × Li × cos / pdf_sa. albedo_h = base/π already, so the
          // full estimator is (albedo/π) * Le * cos_surf / pdf_sa ×
          // path throughput.
          var contrib = vec3f(throughput_h * albedo_h) * Le * (cos_surf / pdf_sa);
          // Firefly defense: NEE-derived k shrinks the threshold for
          // wide-lobe (low-pdf) samples which are the amplification-
          // prone ones. Combined with the d² floor above this is
          // enough for most cases. A harder hard luminance cap was
          // tested (max_nee_lum = 2.0) but over-dimmed legitimate
          // close-emitter illumination on nearby walls; relying on
          // the pdf-floor + filter pair keeps physically plausible
          // close contributions visible while bounding spikes.
          let nee_k = update_firefly_k(rs.firefly_k, pdf_sa, 1.0);
          contrib = firefly_filter(contrib, FIREFLY_THRESHOLD, nee_k);

          let e_origin = ray_offset(hit_pos, geo_normal);
          // max_t = d * 0.999 so we stop just short of x_l; the shadow
          // test only needs to confirm free-space between the two points.
          if !trace_shadow(e_origin, w_hat, d * 0.999) {
            if b == 0u {
              cand_add_direct(idx, contrib);
            } else {
              cand_add_Lo(idx, contrib);
            }
          }
        }
      }

      // BSDF sample — cosine hemisphere
      let u_bsdf = sampler_2d(&s);
      let new_dir = cosine_sample_hemisphere(normal, u_bsdf);
      if dot(normal, new_dir) <= 0.0 {
        rs.flags &= ~FLAG_ALIVE;
        break;
      }

      // Sample-frame throughput: multiply by base only at b >= 1. At b==0
      // the BRDF*cos/pdf cancel is deferred to shade time (base_primary
      // * Lo), so throughput stays 1.0 entering bounce 1 = the x_s frame.
      if b == 0u {
        // Persist source pdf for ReSTIR merge in later phases
        cand_set_source_pdf(idx, max(dot(normal, new_dir), 0.0) * INV_PI);
        // G-buffer: primary albedo is the Lambertian base color.
        // restir_shade multiplies this by Lo to form the indirect term.
        let px = vec2u(idx % res.x, idx / res.x);
        textureStore(albedo_out, vec2i(px), vec4f(mat_base_color(mat), 1.0));
        // Persistent G-buffer for ReSTIR temporal reprojection: world-
        // space primary hit position, linear view Z, and shading normal.
        // restir_temporal reads the ping-pong-prev copy to validate
        // reprojected history.
        let depth_view = dot(hit_pos - uniforms.camera_pos, uniforms.camera_forward);
        gbuf_curr_write(idx, hit_pos, depth_view, normal);
      } else {
        rs.throughput = vec3f(vec3h(rs.throughput) * base_h);
      }
      rs.origin = ray_offset(hit_pos, geo_normal);
      rs.dir = new_dir;
      rs.last_bsdf_pdf = max(dot(normal, new_dir), 0.0) * INV_PI;
      rs.flags = set_bounce(rs.flags, b + 1u);

      // Update firefly-K from the scatter's pdf. Diffuse (wide lobe,
      // low pdf) collapses k; specular (narrow, high pdf) preserves it.
      // Single-lobe Lambertian → lobe_p = 1.
      rs.firefly_k = update_firefly_k(rs.firefly_k, rs.last_bsdf_pdf, 1.0);

      // Russian Roulette — RTXPT Falcor "milder" variant, port from
      // PathTracer/PathTracer.hlsli:182-207. Gated at b >= 1 so the
      // primary hit always contributes (rare low-throughput primaries
      // still get processed even if RR formula alone would tolerate
      // termination). Termination probability:
      //   rr  = sqrt(luminance(throughput))                // perceptual
      //   p   = saturate(0.85 - rr)²                        // squared Falcor
      //   p  += max(0, vertex_ratio - 0.4)                  // ramp to max
      //   if u < p: terminate
      //   throughput /= (1 - p)                             // unbiased boost
      // vertex_ratio = (b+1) / max_bounces so the additive ramp reaches
      // 0.6 as the path nears the depth budget, guaranteeing more
      // aggressive termination near max depth while staying light early.
      if b >= 1u {
        let rr_val = sqrt(luminance(rs.throughput));
        var prob = clamp(0.85 - rr_val, 0.0, 1.0);
        prob = prob * prob;
        let vertex_ratio = f32(b + 1u) / max(f32(uniforms.max_bounces), 1.0);
        prob = clamp(prob + max(0.0, vertex_ratio - 0.4), 0.0, 1.0);
        if sampler_1d(&s) < prob {
          rs.flags &= ~FLAG_ALIVE;
        } else {
          rs.throughput = rs.throughput / max(1.0 - prob, 1e-5);
        }
      }

      // Terminate if over max bounces
      if (b + 1u) >= uniforms.max_bounces {
        rs.flags &= ~FLAG_ALIVE;
      }
      break;
    }
    rs.sampler_dim = s.dim;
    store_ray_state(idx, rs);
  }

  // ---- Uniform control flow reaches here ----
  // Wave-level compaction: ONE atomicAdd per wave (instead of one per
  // surviving lane), each alive lane writes to wave_base + exclusive_rank.
  let alive = processing && ((rs.flags & FLAG_ALIVE) != 0u);
  let alive_i: u32 = select(0u, 1u, alive);
  let my_offset  = subgroupExclusiveAdd(alive_i);
  let wave_total = subgroupAdd(alive_i);

  var wave_base: u32 = 0u;
  if sgid == 0u {
    wave_base = atomicAdd(&counts[1u - SRC_QUEUE], wave_total);
  }
  wave_base = subgroupBroadcastFirst(wave_base);

  if alive {
    let slot = wave_base + my_offset;
    if SRC_QUEUE == 0u { queue_b[slot] = idx; } else { queue_a[slot] = idx; }
  }
}

// ============================================================
// Kernel: PREP_DISPATCH — 1-thread kernel that runs between bounces.
// Reads the count that the next bounce will consume, writes it into
// dispatch_args (rounded up to workgroups), and zeroes the count that
// the next bounce will *fill*. READ_IDX is a pipeline specialization
// constant: 0 = next bounce reads queue_a, 1 = next reads queue_b.
// ============================================================
@compute @workgroup_size(1)
fn prep_dispatch() {
  let count = atomicLoad(&counts[READ_IDX]);
  let wg = (count + 63u) / 64u;
  dispatch_args[0] = wg;
  dispatch_args[1] = 1u;
  dispatch_args[2] = 1u;
  // Zero the OTHER count so the upcoming bounce can fill it from zero
  atomicStore(&counts[1u - READ_IDX], 0u);
}

// ============================================================
// Kernel: SHADOW_TRACE — dedicated shadow ray kernel. Reads each
// pixel's queued NEE request (origin + dir + contribution) and traces
// an any-hit BVH. Unblocked → accumulate contribution into radiance.
// One thread per pixel; each pixel has at most one request per bounce
// so the non-atomic read-modify-write on ray_state radiance is safe.
// Work is uniform (pure BVH, no material eval) → good cache behavior.
// ============================================================
@compute @workgroup_size(8, 8)
fn shadow_trace(@builtin(global_invocation_id) gid: vec3u) {
  let pixel = vec2u(gid.xy);
  let res = vec2u(uniforms.resolution);
  if pixel.x >= res.x || pixel.y >= res.y { return; }
  let idx = pixel.y * res.x + pixel.x;

  let sbase = idx * 3u;
  let r0 = shadow_req[sbase];
  let flag = r0.w;
  if flag < 0.5 { return; } // no request this bounce

  let origin = r0.xyz;
  let dir    = shadow_req[sbase + 1u].xyz;
  let contrib = shadow_req[sbase + 2u].xyz;

  // Consume the request — mark invalid so a later bounce that skips this
  // pixel (because it was compacted out of the queue) doesn't re-process
  // stale data. Compaction means the bounce kernel no longer touches
  // every pixel each iteration, so clearing upfront there isn't enough.
  shadow_req[sbase] = vec4f(0.0);

  if trace_shadow(origin, dir, INF) { return; } // sun occluded

  // Route the contribution: flag 1.0 = direct (primary NEE, bounce 0)
  // → candidate.direct; flag 2.0 = indirect (bounce >= 1 NEE) → Lo in
  // candidate_buf (so restir_shade can read it alongside the sample
  // point). Set by the bounce kernel at the bounce NEE was queued.
  if flag < 1.5 {
    cand_add_direct(idx, contrib);
  } else {
    cand_add_Lo(idx, contrib);
  }
}

// ============================================================
// Kernel: COMPOSITE — temporal accumulation. Reads the current frame's
// noisy output and the previous accumulator; writes the blend to the
// new accumulator. When the camera is still, frames_still grows and
// alpha shrinks, so the history dominates and noise averages out.
// On camera move, JS resets frames_still to 0, alpha=1, and the new
// accumulator fully replaces history in one frame.
// ============================================================
@compute @workgroup_size(8, 8)
fn composite(@builtin(global_invocation_id) gid: vec3u) {
  let pixel = vec2u(gid.xy);
  let res = vec2u(uniforms.resolution);
  if pixel.x >= res.x || pixel.y >= res.y { return; }

  // Running-mean en f32. alpha = 1/(n+1) converge a la media insesgada;
  // reseteado a 1 cuando JS detecta movimiento de cámara (frames_still=0).
  let curr = textureLoad(noisy_read, vec2i(pixel), 0).rgb;
  let prev = textureLoad(accum_prev, vec2i(pixel), 0).rgb;
  let alpha = 1.0 / f32(uniforms.frames_still + 1u);
  let mixed = mix(prev, curr, alpha);
  textureStore(accum_new, vec2i(pixel), vec4f(mixed, 1.0));
}

// ============================================================
// Kernel: FINALIZE — write accumulated radiance to noisy output.
// ============================================================
@compute @workgroup_size(8, 8)
fn finalize(@builtin(global_invocation_id) gid: vec3u) {
  let pixel = vec2u(gid.xy);
  let res = vec2u(uniforms.resolution);
  if pixel.x >= res.x || pixel.y >= res.y { return; }
  let idx = pixel.y * res.x + pixel.x;

  let rs = load_ray_state(idx);
  // Firefly luminance clamp
  var rad = rs.radiance;
  let lum = dot(rad, vec3f(0.2126, 0.7152, 0.0722));
  let max_lum = 32.0;
  if lum > max_lum { rad *= max_lum / lum; }
  textureStore(noisy_out, vec2i(pixel), vec4f(rad, 1.0));
}
