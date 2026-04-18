// Wavefront Path Tracer — minimal v1
// Architecture: per-pixel ray state in storage buffer, two kernels re-dispatched
// per bounce. Much smaller per-kernel footprint than the mega-kernel in
// pathtracer.wgsl → better occupancy on Adreno and mobile in general.
//
// v1 scope: BVH closest-hit + any-hit, cosine-diffuse BSDF, NEE sun with
// shadow ray + MIS against simple gradient sky. No textures, no normal maps,
// no GGX spec, no env CDF, no SHaRC/ReSTIR/glass. Those come back in later
// commits once the wavefront architecture is proven stable.

struct Uniforms {
  resolution: vec2f,
  frame_seed: u32,
  max_bounces: u32,
  camera_pos: vec3f,   _pad0: f32,
  camera_forward: vec3f, _pad1: f32,
  camera_right: vec3f, _pad2: f32,
  camera_up: vec3f,    fov_factor: f32,
  sun_dir: vec3f,      _pad3: f32,
};

struct BVHNode { aabb_min: vec3f, left_first: u32, aabb_max: vec3f, tri_count: u32, };
// Material: keep same 20-vec4f layout as legacy PT so scene-loader.js is untouched.
// v1 only reads d0 (albedo + mat_type), d1 (emission + roughness), d2.x (metallic).
struct Material {
  d0: vec4f, d1: vec4f, d2: vec4f, d3: vec4f, d4: vec4f,
  d5: vec4f, d6: vec4f, d7: vec4f, d8: vec4f, d9: vec4f,
  d10: vec4f, d11: vec4f, d12: vec4f, d13: vec4f, d14: vec4f,
  d15: vec4f, d16: vec4f, d17: vec4f, d18: vec4f, d19: vec4f,
};

// Per-ray state, 64 bytes, stored as 4 vec4f per pixel in ray_state_buf.
// Packing:
//   ray_state[4i+0] = vec4f(origin.xyz,       bitcast(rng_state))
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

@group(1) @binding(0) var<storage, read> vertices: array<vec4f>;
@group(1) @binding(1) var<storage, read> vert_normals: array<vec4f>;
@group(1) @binding(2) var<storage, read> tri_data: array<vec4u>;
@group(1) @binding(3) var<storage, read> bvh_nodes: array<BVHNode>;
@group(1) @binding(4) var<storage, read> material_buf: array<Material>;

const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;
const INV_PI: f32 = 0.31830988618;
const INF: f32 = 1e30;
const T_MIN: f32 = 1e-4;
const SUN_COS_HALF: f32 = 0.9999619; // ~0.5° angular radius
const SUN_RADIANCE: vec3f = vec3f(6.5, 6.0, 5.0); // punchy sun, tonemap trims
const CLAMP_INDIRECT: f32 = 10.0;

const FLAG_ALIVE: u32 = 1u;
const FLAG_DIFFUSE_PATH: u32 = 2u;

// ============================================================
// RNG — PCG + PCG3D for decorrelated vec2 sampling
// ============================================================
fn pcg(state: ptr<function, u32>) -> u32 {
  let s = *state;
  *state = s * 747796405u + 2891336453u;
  let word = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (word >> 22u) ^ word;
}

fn pcg3d(v_in: vec3u) -> vec3u {
  var v = v_in * 1664525u + 1013904223u;
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  v ^= v >> vec3u(16u);
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  return v;
}

fn rand1(s: ptr<function, u32>) -> f32 {
  return f32(pcg(s)) / 4294967295.0;
}
fn rand2(s: ptr<function, u32>) -> vec2f {
  let h = pcg3d(vec3u(*s, pcg(s), pcg(s)));
  *s = h.z;
  return vec2f(f32(h.x), f32(h.y)) / 4294967295.0;
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
  var stk: array<u32, 16>; var sp = 0u; var cur = 0u;
  let root = bvh_nodes[0u];
  if intersect_aabb(origin, inv_dir, root.aabb_min, root.aabb_max, INF) >= INF { return hit; }
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
    let tl = intersect_aabb(origin, inv_dir, bvh_nodes[l].aabb_min, bvh_nodes[l].aabb_max, hit.t);
    let tr = intersect_aabb(origin, inv_dir, bvh_nodes[r].aabb_min, bvh_nodes[r].aabb_max, hit.t);
    if tl < tr {
      if tr < hit.t && sp < 16u { stk[sp] = r; sp++; }
      if tl < hit.t { cur = l; } else { if sp == 0u { break; } sp--; cur = stk[sp]; }
    } else {
      if tl < hit.t && sp < 16u { stk[sp] = l; sp++; }
      if tr < hit.t { cur = r; } else { if sp == 0u { break; } sp--; cur = stk[sp]; }
    }
  }
  return hit;
}

// BVH any-hit for shadows (returns true if blocked before max_t)
fn trace_shadow(origin: vec3f, dir: vec3f, max_t: f32) -> bool {
  let inv_dir = 1.0 / dir;
  var stk: array<u32, 16>; var sp = 0u; var cur = 0u;
  let root = bvh_nodes[0u];
  if intersect_aabb(origin, inv_dir, root.aabb_min, root.aabb_max, max_t) >= max_t { return false; }
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
    let tl = intersect_aabb(origin, inv_dir, bvh_nodes[l].aabb_min, bvh_nodes[l].aabb_max, max_t);
    let tr = intersect_aabb(origin, inv_dir, bvh_nodes[r].aabb_min, bvh_nodes[r].aabb_max, max_t);
    if tl < tr {
      if tr < max_t && sp < 16u { stk[sp] = r; sp++; }
      if tl < max_t { cur = l; } else { if sp == 0u { break; } sp--; cur = stk[sp]; }
    } else {
      if tl < max_t && sp < 16u { stk[sp] = l; sp++; }
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
// Ray state load / store (packed into array<vec4f>)
// ============================================================
struct RayState {
  origin: vec3f,
  dir: vec3f,
  throughput: vec3f,
  radiance: vec3f,
  rng_state: u32,
  flags: u32,
  last_bsdf_pdf: f32,
  hit_t: f32,
};

fn load_ray_state(idx: u32) -> RayState {
  let base = idx * 4u;
  let r0 = ray_state[base];
  let r1 = ray_state[base + 1u];
  let r2 = ray_state[base + 2u];
  let r3 = ray_state[base + 3u];
  var rs: RayState;
  rs.origin = r0.xyz;
  rs.rng_state = bitcast<u32>(r0.w);
  rs.dir = r1.xyz;
  rs.flags = bitcast<u32>(r1.w);
  rs.throughput = r2.xyz;
  rs.last_bsdf_pdf = r2.w;
  rs.radiance = r3.xyz;
  rs.hit_t = r3.w;
  return rs;
}

fn store_ray_state(idx: u32, rs: RayState) {
  let base = idx * 4u;
  ray_state[base]       = vec4f(rs.origin,    bitcast<f32>(rs.rng_state));
  ray_state[base + 1u]  = vec4f(rs.dir,       bitcast<f32>(rs.flags));
  ray_state[base + 2u]  = vec4f(rs.throughput, rs.last_bsdf_pdf);
  ray_state[base + 3u]  = vec4f(rs.radiance,  rs.hit_t);
}

fn bounce_of(flags: u32) -> u32 { return (flags >> 8u) & 0xFFu; }
fn set_bounce(flags: u32, b: u32) -> u32 { return (flags & 0xFFFF00FFu) | ((b & 0xFFu) << 8u); }

// ============================================================
// Material access helpers (v1: core fields only)
// ============================================================
fn mat_base_color(m: Material) -> vec3f { return m.d0.xyz; }
fn mat_mat_type(m: Material) -> u32 { return u32(m.d0.w + 0.5); }
fn mat_emission(m: Material) -> vec3f { return m.d1.xyz * max(m.d3.w, 0.0); }
fn mat_roughness(m: Material) -> f32 { return max(m.d1.w, 0.04); }
fn mat_metallic(m: Material) -> f32 { return clamp(m.d2.x, 0.0, 1.0); }
fn mat_is_unlit(m: Material) -> bool { return (u32(m.d4.w + 0.5) & 4u) != 0u; }

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

  // PCG3D-seeded RNG per-pixel per-frame
  let seed = pcg3d(vec3u(pixel.x, pixel.y, uniforms.frame_seed | 1u));
  var rng = seed.x | 1u;

  // Sub-pixel jitter
  let j = rand2(&rng);
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
  rs.rng_state = rng;
  rs.flags = FLAG_ALIVE | FLAG_DIFFUSE_PATH;
  rs.last_bsdf_pdf = 0.0;
  rs.hit_t = 1e4;
  store_ray_state(idx, rs);

  // Default albedo (white) — overwritten by bounce kernel on first hit
  textureStore(albedo_out, vec2i(pixel), vec4f(1.0, 1.0, 1.0, 1.0));
  textureStore(denoise_nd_out, vec2i(pixel), vec4f(0.0, 1.0, 0.0, 1e4));
}

// ============================================================
// Kernel: BOUNCE — one dispatch per bounce iteration.
// Reads ray_state, traces, shades with NEE sun, samples cosine
// hemisphere for next ray, updates state.
// ============================================================
@compute @workgroup_size(8, 8)
fn bounce(@builtin(global_invocation_id) gid: vec3u) {
  let pixel = vec2u(gid.xy);
  let res = vec2u(uniforms.resolution);
  if pixel.x >= res.x || pixel.y >= res.y { return; }
  let idx = pixel.y * res.x + pixel.x;

  var rs = load_ray_state(idx);
  if (rs.flags & FLAG_ALIVE) == 0u { return; }

  var rng = rs.rng_state;
  let b = bounce_of(rs.flags);

  let hit = trace_bvh(rs.origin, rs.dir);
  if !hit.hit {
    // Miss: add sky weighted by throughput, kill ray.
    rs.radiance += rs.throughput * sky_color(rs.dir);
    rs.flags &= ~FLAG_ALIVE;
    rs.rng_state = rng;
    store_ray_state(idx, rs);
    return;
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

  // Emission — add only on first hit or after specular bounce (v1 all diffuse)
  let em = mat_emission(mat);
  if b == 0u && dot(em, vec3f(1.0)) > 1e-6 {
    rs.radiance += rs.throughput * em;
  }

  if b == 0u {
    // Capture first-hit gbuffer for denoiser (future use)
    textureStore(albedo_out, vec2i(pixel), vec4f(mat_base_color(mat), 1.0));
    textureStore(denoise_nd_out, vec2i(pixel), vec4f(normal, hit.t));
  }
  if b == 1u { rs.hit_t = hit.t; } // second-bounce distance for denoiser

  // Unlit: just emit base color and terminate
  if mat_is_unlit(mat) {
    rs.radiance += rs.throughput * mat_base_color(mat);
    rs.flags &= ~FLAG_ALIVE;
    rs.rng_state = rng;
    store_ray_state(idx, rs);
    return;
  }

  // v1 treats every non-unlit material as pure Lambertian diffuse with
  // the glTF baseColorFactor as albedo. Metallic is *ignored*: scene-loader
  // defaults metallicFactor to 1.0 per glTF spec when the material expects
  // a metallic-roughness texture to drive it; with no texture sampling
  // that value is meaningless and taking (1 - metallic) would zero the
  // albedo on every Sponza material (→ indirect black). Textures + real
  // metallic handling come back in the next commit.
  let base = mat_base_color(mat);
  let albedo = base * INV_PI;

  // NEE sun — delta light (directional). No cone sampling, no MIS.
  let nsl = dot(normal, uniforms.sun_dir);
  if nsl > 0.0 {
    let sun_origin = ray_offset(hit_pos, geo_normal);
    if !trace_shadow(sun_origin, uniforms.sun_dir, INF) {
      var direct = rs.throughput * albedo * SUN_RADIANCE * nsl;
      if b > 0u {
        let s = abs(direct.x) + abs(direct.y) + abs(direct.z);
        if s > CLAMP_INDIRECT { direct *= CLAMP_INDIRECT / s; }
      }
      rs.radiance += direct;
    }
  }

  // Sample next direction: cosine hemisphere
  let u_bsdf = rand2(&rng);
  let new_dir = cosine_sample_hemisphere(normal, u_bsdf);
  if dot(normal, new_dir) <= 0.0 {
    rs.flags &= ~FLAG_ALIVE;
    rs.rng_state = rng;
    store_ray_state(idx, rs);
    return;
  }

  // Lambertian throughput update (cosine-weighted sampling → pdf cancels)
  rs.throughput *= base;
  rs.origin = ray_offset(hit_pos, geo_normal);
  rs.dir = new_dir;
  rs.last_bsdf_pdf = max(dot(normal, new_dir), 0.0) * INV_PI;
  rs.flags = set_bounce(rs.flags, b + 1u);

  // Russian roulette after bounce 1
  if b >= 1u {
    let p = clamp(sqrt(max(max(rs.throughput.x, rs.throughput.y), rs.throughput.z)), 0.05, 0.9);
    if rand1(&rng) > p {
      rs.flags &= ~FLAG_ALIVE;
    } else {
      rs.throughput /= p;
    }
  }

  // Terminate if over max bounces
  if (b + 1u) >= uniforms.max_bounces {
    rs.flags &= ~FLAG_ALIVE;
  }

  rs.rng_state = rng;
  store_ray_state(idx, rs);
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
