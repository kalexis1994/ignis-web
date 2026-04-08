// ============================================================
// Wavefront Stage 3: Shade
// For each hit ray: evaluate material, generate shadow ray (NEE) + bounce ray
// Writes to shadow_ray buffer + bounce_ray buffer with compaction
// ============================================================

struct ShadeParams {
  ray_count: u32,
  frame_seed: u32,
  max_bounces: u32,
  sun_enabled: u32,
  sun_dir: vec3f,
  emissive_tri_count: u32,
  resolution: vec2f,
  _pad: vec2f,
};

struct Material {
  d0: vec4f, d1: vec4f, d2: vec4f, d3: vec4f, d4: vec4f,
  d5: vec4f, d6: vec4f, d7: vec4f, d8: vec4f, d9: vec4f,
};

@group(0) @binding(0) var<uniform> params: ShadeParams;
@group(0) @binding(1) var<storage, read> rays: array<vec4f>;          // input rays (3 vec4f each)
@group(0) @binding(2) var<storage, read> hits: array<vec4f>;          // hit results
@group(0) @binding(3) var<storage, read_write> shadow_rays: array<vec4f>; // output shadow rays (3 vec4f each)
@group(0) @binding(4) var<storage, read_write> bounce_rays: array<vec4f>; // output bounce rays (3 vec4f each)
@group(0) @binding(5) var<storage, read_write> accum: array<vec4f>;   // pixel accumulator (2 vec4f per pixel)
@group(0) @binding(6) var<storage, read_write> counters: array<atomic<u32>>; // [0]=shadow, [1]=bounce

@group(1) @binding(0) var<storage, read> vertices: array<vec4f>;
@group(1) @binding(1) var<storage, read> vert_normals: array<vec4f>;
@group(1) @binding(2) var<storage, read> tri_data: array<vec4u>;
@group(1) @binding(3) var<storage, read> material_buf: array<vec4f>;  // 20 vec4f per material

const PI: f32 = 3.14159265;
const INV_PI: f32 = 0.31830989;

// RNG
var<private> rng_state: u32;
fn pcg(state: ptr<private, u32>) -> u32 {
  let s = *state;
  *state = s * 747796405u + 2891336453u;
  let word = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (word >> 22u) ^ word;
}
fn rand() -> f32 { return f32(pcg(&rng_state)) / 4294967295.0; }

fn cosine_sample_hemisphere(n: vec3f) -> vec3f {
  let r1 = rand();
  let r2 = rand();
  let phi = 2.0 * PI * r1;
  let cos_theta = sqrt(r2);
  let sin_theta = sqrt(1.0 - r2);
  var up = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(n.y) < 0.999);
  let t = normalize(cross(up, n));
  let b = cross(n, t);
  return normalize(t * cos(phi) * sin_theta + b * sin(phi) * sin_theta + n * cos_theta);
}

// Wächter & Binder robust ray offset
fn ray_offset(P: vec3f, Ng: vec3f) -> vec3f {
  let int_scale = 256.0;
  let oi = vec3i(vec3f(int_scale) * Ng);
  let pi = vec3f(
    bitcast<f32>(bitcast<i32>(P.x) + select(oi.x, -oi.x, P.x < 0.0)),
    bitcast<f32>(bitcast<i32>(P.y) + select(oi.y, -oi.y, P.y < 0.0)),
    bitcast<f32>(bitcast<i32>(P.z) + select(oi.z, -oi.z, P.z < 0.0)),
  );
  let float_scale = 1.0 / 65536.0;
  let origin_offset = 1.0 / 32.0;
  return vec3f(
    select(pi.x, P.x + float_scale * Ng.x, abs(P.x) < origin_offset),
    select(pi.y, P.y + float_scale * Ng.y, abs(P.y) < origin_offset),
    select(pi.z, P.z + float_scale * Ng.z, abs(P.z) < origin_offset),
  );
}

fn decode_pixel_x(id: u32) -> u32 { return id & 0xFFFFu; }
fn decode_pixel_y(id: u32) -> u32 { return id >> 16u; }
fn decode_bounce_count(b: u32) -> u32 { return b & 0xFFu; }
fn decode_is_diffuse(b: u32) -> bool { return (b & 256u) != 0u; }
fn encode_pixel(x: u32, y: u32) -> u32 { return x | (y << 16u); }
fn encode_bounce(count: u32, is_diffuse: bool, is_specular: bool) -> u32 {
  return count | (select(0u, 256u, is_diffuse)) | (select(0u, 512u, is_specular));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let ray_idx = gid.x;
  if ray_idx >= params.ray_count { return; }

  // Read ray
  let base = ray_idx * 3u;
  let r0 = rays[base];
  let r1 = rays[base + 1u];
  let r2 = rays[base + 2u];

  let origin = r0.xyz;
  let dir = r1.xyz;
  let pixel_id = bitcast<u32>(r1.w);
  let throughput = r2.xyz;
  let bounce_info = bitcast<u32>(r2.w);
  let bounce = decode_bounce_count(bounce_info);
  let is_diffuse_path = decode_is_diffuse(bounce_info);

  // Read hit
  let hit = hits[ray_idx];
  let tri_idx = bitcast<u32>(hit.x);
  let hit_t = hit.w;

  let px = decode_pixel_x(pixel_id);
  let py = decode_pixel_y(pixel_id);
  let pixel_idx = py * u32(params.resolution.x) + px;

  // RNG init
  rng_state = (px * 1973u + py * 9277u + params.frame_seed * 26699u + bounce * 12347u) | 1u;
  _ = pcg(&rng_state);

  // Miss: accumulate sky color
  if tri_idx == 0xFFFFFFFFu || hit_t > 1e5 {
    // Simple sky gradient
    let sky_up = dir.y * 0.5 + 0.5;
    let sky = mix(vec3f(0.8, 0.9, 1.0), vec3f(0.3, 0.5, 0.9), sky_up) * 2.0;
    let contrib = throughput * sky;

    // Accumulate
    let acc_base = pixel_idx * 2u;
    if is_diffuse_path {
      let prev = accum[acc_base];
      accum[acc_base] = vec4f(prev.xyz + contrib, prev.w);
    } else {
      let prev = accum[acc_base + 1u];
      accum[acc_base + 1u] = vec4f(prev.xyz + contrib, prev.w);
    }
    return;
  }

  // Hit: evaluate surface
  let td = tri_data[tri_idx];
  let bw = 1.0 - hit.y - hit.z;
  let hit_pos = origin + dir * hit_t;
  let V = -dir;

  // Interpolate normal
  let n0 = vert_normals[td.x].xyz;
  let n1 = vert_normals[td.y].xyz;
  let n2 = vert_normals[td.z].xyz;
  var normal = normalize(bw * n0 + hit.y * n1 + hit.z * n2);

  // Geometric normal
  let v0 = vertices[td.x].xyz;
  let v1 = vertices[td.y].xyz;
  let v2 = vertices[td.z].xyz;
  let geo_normal_raw = normalize(cross(v1 - v0, v2 - v0));
  let front_face = dot(dir, geo_normal_raw) < 0.0;
  let geo_normal = select(-geo_normal_raw, geo_normal_raw, front_face);
  if !front_face { normal = -normal; }

  // Material (simplified: read base color and roughness)
  let mat_idx = td.w;
  let mat_base = mat_idx * 20u;
  let base_color = vec3f(
    material_buf[mat_base].x,
    material_buf[mat_base].y,
    material_buf[mat_base].z
  );
  let roughness = max(material_buf[mat_base + 1u].y, 0.04);
  let metallic = material_buf[mat_base + 2u].x;

  // Store first-hit info
  if bounce == 0u {
    let acc_base2 = pixel_idx * 2u;
    accum[acc_base2] = vec4f(accum[acc_base2].xyz, hit_t); // depth in .w
  }

  // === Generate shadow ray (NEE to sun) ===
  if params.sun_enabled > 0u {
    let L = normalize(params.sun_dir);
    let NdotL = dot(normal, L);
    if NdotL > 0.0 {
      let shadow_origin = ray_offset(hit_pos, geo_normal);
      // Lambertian BRDF weight × sun radiance
      let sun_radiance = vec3f(3.0, 2.8, 2.5) * 5.0; // warm sun
      let brdf = base_color * INV_PI * NdotL * (1.0 - metallic);
      let contrib = throughput * brdf * sun_radiance;

      // Write shadow ray with compaction
      let shadow_idx = atomicAdd(&counters[0], 1u);
      let sb = shadow_idx * 3u;
      shadow_rays[sb]     = vec4f(shadow_origin, 200.0);     // origin + max_t
      shadow_rays[sb + 1u] = vec4f(L, bitcast<f32>(pixel_id)); // dir + pixel_id
      shadow_rays[sb + 2u] = vec4f(contrib, select(0.0, 1.0, is_diffuse_path)); // radiance + is_diffuse
    }
  }

  // === Generate bounce ray ===
  if bounce < params.max_bounces - 1u {
    // Russian roulette after bounce 2
    var continue_prob = 1.0;
    if bounce >= 2u {
      continue_prob = clamp(max(throughput.x, max(throughput.y, throughput.z)), 0.05, 0.9);
      if rand() > continue_prob { return; }
    }

    // Cosine hemisphere sampling (diffuse)
    let bounce_dir = cosine_sample_hemisphere(normal);
    if dot(normal, bounce_dir) <= 0.0 { return; }

    let new_throughput = throughput * base_color * (1.0 - metallic) / continue_prob;

    // Clamp throughput to prevent fireflies
    let max_throughput = 10.0;
    let t_lum = max(new_throughput.x, max(new_throughput.y, new_throughput.z));
    let clamped_throughput = select(new_throughput, new_throughput * (max_throughput / t_lum), t_lum > max_throughput);

    // Write bounce ray with compaction
    let bounce_idx = atomicAdd(&counters[1], 1u);
    let bb = bounce_idx * 3u;
    let new_origin = ray_offset(hit_pos, geo_normal);
    bounce_rays[bb]     = vec4f(new_origin, 1e6);
    bounce_rays[bb + 1u] = vec4f(bounce_dir, bitcast<f32>(pixel_id));
    bounce_rays[bb + 2u] = vec4f(clamped_throughput, bitcast<f32>(encode_bounce(bounce + 1u, true, false)));
  }
}
