// ============================================================
// Wavefront Stage 2: BVH Trace
// Traces all active rays through BVH, writes hit results
// Also used for shadow ray tracing (stage 4)
// ============================================================

struct TraceParams {
  ray_count: u32,
  is_shadow: u32,  // 0 = closest hit, 1 = any hit (shadow)
  _pad: vec2u,
};

struct BVHNode {
  aabb_min: vec3f,
  left_first: u32,
  aabb_max: vec3f,
  tri_count: u32,
};

@group(0) @binding(0) var<uniform> params: TraceParams;
@group(0) @binding(1) var<storage, read> rays: array<vec4f>;        // 3 vec4f per ray
@group(0) @binding(2) var<storage, read_write> hits: array<vec4f>;  // 1 vec4f per ray (tri_idx, u, v, t)

@group(1) @binding(0) var<storage, read> vertices: array<vec4f>;    // pos.xyz + uv.x
@group(1) @binding(1) var<storage, read> vert_normals: array<vec4f>;
@group(1) @binding(2) var<storage, read> tri_data: array<vec4u>;    // v0, v1, v2, mat_id
@group(1) @binding(3) var<storage, read> bvh_nodes: array<BVHNode>;

const INF: f32 = 1e30;

fn intersect_aabb(origin: vec3f, inv_dir: vec3f, bmin: vec3f, bmax: vec3f, t_max: f32) -> f32 {
  let t1 = (bmin - origin) * inv_dir;
  let t2 = (bmax - origin) * inv_dir;
  let tmin = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
  let tmax = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));
  if tmax < 0.0 || tmin > tmax || tmin > t_max { return INF; }
  return tmin;
}

fn intersect_tri(origin: vec3f, dir: vec3f, v0: vec3f, v1: vec3f, v2: vec3f, t_max: f32) -> vec3f {
  let e1 = v1 - v0;
  let e2 = v2 - v0;
  let h = cross(dir, e2);
  let a = dot(e1, h);
  if abs(a) < 1e-8 { return vec3f(INF, 0.0, 0.0); }
  let f = 1.0 / a;
  let s = origin - v0;
  let u = f * dot(s, h);
  if u < 0.0 || u > 1.0 { return vec3f(INF, 0.0, 0.0); }
  let q = cross(s, e1);
  let v = f * dot(dir, q);
  if v < 0.0 || u + v > 1.0 { return vec3f(INF, 0.0, 0.0); }
  let t = f * dot(e2, q);
  if t < 1e-5 || t > t_max { return vec3f(INF, 0.0, 0.0); }
  return vec3f(t, u, v);
}

@compute @workgroup_size(256)
fn trace_closest(@builtin(global_invocation_id) gid: vec3u) {
  let ray_idx = gid.x;
  if ray_idx >= params.ray_count { return; }

  let base = ray_idx * 3u;
  let r0 = rays[base];
  let r1 = rays[base + 1u];

  let origin = r0.xyz;
  var t_max = r0.w;
  let dir = r1.xyz;
  let inv_dir = 1.0 / dir;

  var best_t = t_max;
  var best_u = 0.0;
  var best_v = 0.0;
  var best_tri = 0xFFFFFFFFu;

  // BVH traversal (stack-based)
  var stk: array<u32, 24>;  // deeper stack for native
  var sp = 0u;
  var cur = 0u;

  let root = bvh_nodes[0u];
  if intersect_aabb(origin, inv_dir, root.aabb_min, root.aabb_max, best_t) >= best_t {
    hits[ray_idx] = vec4f(bitcast<f32>(0xFFFFFFFFu), 0.0, 0.0, INF);
    return;
  }

  loop {
    let nd = bvh_nodes[cur];
    if nd.tri_count > 0u {
      // Leaf: test triangles
      for (var i = 0u; i < nd.tri_count; i++) {
        let ti = nd.left_first + i;
        let td = tri_data[ti];
        let r = intersect_tri(
          origin, dir,
          vertices[td.x].xyz, vertices[td.y].xyz, vertices[td.z].xyz,
          best_t
        );
        if r.x < best_t {
          best_t = r.x;
          best_u = r.y;
          best_v = r.z;
          best_tri = ti;
        }
      }
      if sp == 0u { break; }
      sp--; cur = stk[sp]; continue;
    }

    // Internal node: traverse children
    let l = nd.left_first;
    let r = l + 1u;
    let tl = intersect_aabb(origin, inv_dir, bvh_nodes[l].aabb_min, bvh_nodes[l].aabb_max, best_t);
    let tr = intersect_aabb(origin, inv_dir, bvh_nodes[r].aabb_min, bvh_nodes[r].aabb_max, best_t);

    if tl < tr {
      if tr < best_t && sp < 24u { stk[sp] = r; sp++; }
      if tl < best_t { cur = l; } else { if sp == 0u { break; } sp--; cur = stk[sp]; }
    } else {
      if tl < best_t && sp < 24u { stk[sp] = l; sp++; }
      if tr < best_t { cur = r; } else { if sp == 0u { break; } sp--; cur = stk[sp]; }
    }
  }

  hits[ray_idx] = vec4f(bitcast<f32>(best_tri), best_u, best_v, best_t);
}

// Shadow ray version: any-hit (early exit on first intersection)
@compute @workgroup_size(256)
fn trace_shadow(@builtin(global_invocation_id) gid: vec3u) {
  let ray_idx = gid.x;
  if ray_idx >= params.ray_count { return; }

  let base = ray_idx * 3u;
  let r0 = rays[base];
  let r1 = rays[base + 1u];

  let origin = r0.xyz;
  let max_t = r0.w;
  let dir = r1.xyz;
  let inv_dir = 1.0 / dir;

  var stk: array<u32, 24>;
  var sp = 0u;
  var cur = 0u;

  let root = bvh_nodes[0u];
  if intersect_aabb(origin, inv_dir, root.aabb_min, root.aabb_max, max_t) >= max_t {
    hits[ray_idx] = vec4f(0.0); // no hit = unoccluded
    return;
  }

  loop {
    let nd = bvh_nodes[cur];
    if nd.tri_count > 0u {
      for (var i = 0u; i < nd.tri_count; i++) {
        let ti = nd.left_first + i;
        let td = tri_data[ti];
        let r = intersect_tri(
          origin, dir,
          vertices[td.x].xyz, vertices[td.y].xyz, vertices[td.z].xyz,
          max_t
        );
        if r.x < max_t {
          hits[ray_idx] = vec4f(1.0, 0.0, 0.0, r.x); // occluded
          return;
        }
      }
      if sp == 0u { break; }
      sp--; cur = stk[sp]; continue;
    }

    let l = nd.left_first;
    let r = l + 1u;
    let tl = intersect_aabb(origin, inv_dir, bvh_nodes[l].aabb_min, bvh_nodes[l].aabb_max, max_t);
    let tr = intersect_aabb(origin, inv_dir, bvh_nodes[r].aabb_min, bvh_nodes[r].aabb_max, max_t);

    if tl < tr {
      if tr < max_t && sp < 24u { stk[sp] = r; sp++; }
      if tl < max_t { cur = l; } else { if sp == 0u { break; } sp--; cur = stk[sp]; }
    } else {
      if tl < max_t && sp < 24u { stk[sp] = l; sp++; }
      if tr < max_t { cur = r; } else { if sp == 0u { break; } sp--; cur = stk[sp]; }
    }
  }

  hits[ray_idx] = vec4f(0.0); // unoccluded
}
