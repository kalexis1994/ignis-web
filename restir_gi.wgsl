// ReSTIR GI — Phase 1a (scaffolding, no reuse yet)
//
// Concatenated after wavefront.wgsl by renderer.js before createShaderModule,
// so it shares Uniforms, Sampler, sampler_*, ray_offset, vec3h, bg0/bg1/bg2
// in the merged compilation unit.
//
// Phase 1a scope:
//   - candidate_buf: one captured path sample per pixel (re-written each frame)
//   - Reservoir struct + helpers (unused until Phase 1b/1c)
//   - restir_shade kernel: final = direct_rad + albedo_primary * Lo
//
// With a single candidate and no reuse, the output equals plain 1-SPP PT
// exactly (Lambertian + cosine-weighted sampling makes BRDF*cos/pdf cancel
// to base_color, so "base * Lo" reproduces the old rs.radiance).
//
// Later phases:
//   1b — temporal reuse: reproject prev reservoir, WRS merge with current
//        candidate, clamp M, reject on normal/depth mismatch.
//   1c — spatial reuse: k random neighbors with geometry jacobians for
//        valid transform between visible points.

// ============================================================
// Candidate sample buffer — bg2 binding 4, shared with bounce/shadow.
// 4 vec4f = 64 B per pixel:
//   [0] = (x_s.xyz,       source_pdf)          BSDF pdf at primary for v→s
//   [1] = (n_s.xyz,       sample_valid_flag)    1.0 = sample captured
//   [2] = (Lo.xyz,        _pad)                 outgoing radiance at x_s
//   [3] = (direct_rad.xyz,_pad)                 direct lighting at primary
// ============================================================
@group(2) @binding(4) var<storage, read_write> candidate_buf: array<vec4f>;

// Primary visible-point albedo, read by restir_shade. Written as
// storage-texture in bounce at b==0; read here as sampled texture.
// Placed at group 3 binding 3 (composite uses 0/1/2 of group 3) so
// we stay within the maxBindGroups=4 limit enforced by Adreno 7xx.
@group(3) @binding(3) var albedo_read: texture_2d<f32>;

// ============================================================
// Candidate helpers
// ============================================================
fn cand_reset(idx: u32) {
  let base = idx * 4u;
  candidate_buf[base]      = vec4f(0.0);
  candidate_buf[base + 1u] = vec4f(0.0);
  candidate_buf[base + 2u] = vec4f(0.0);
  candidate_buf[base + 3u] = vec4f(0.0);
}

fn cand_add_direct(idx: u32, v: vec3f) {
  let base = idx * 4u + 3u;
  let cur = candidate_buf[base];
  candidate_buf[base] = vec4f(cur.xyz + v, cur.w);
}

fn cand_add_Lo(idx: u32, v: vec3f) {
  let base = idx * 4u + 2u;
  let cur = candidate_buf[base];
  candidate_buf[base] = vec4f(cur.xyz + v, cur.w);
}

fn cand_set_sample(idx: u32, x_s: vec3f, n_s: vec3f) {
  let base = idx * 4u;
  // Keep source_pdf already set at bounce-0 BSDF sample
  let v0 = candidate_buf[base];
  candidate_buf[base]      = vec4f(x_s, v0.w);
  candidate_buf[base + 1u] = vec4f(n_s, 1.0);
}

fn cand_set_source_pdf(idx: u32, pdf: f32) {
  let base = idx * 4u;
  let v0 = candidate_buf[base];
  candidate_buf[base] = vec4f(v0.xyz, pdf);
}

// ============================================================
// Reservoir — defined for Phase 1b/1c readiness; unused in 1a.
// Standard WRS reservoir: running weighted sample with sum of weights,
// total sample count M, normalized weight W applied at shade time.
// ============================================================
struct Reservoir {
  x_s: vec3f,
  n_s: vec3f,
  Lo: vec3f,
  source_pdf: f32,
  w_sum: f32,   // running sum of target_pdf × source_W (WRS)
  M: f32,       // effective sample count
  W: f32,       // normalized contribution weight at shade time
};

fn reservoir_empty() -> Reservoir {
  var r: Reservoir;
  r.x_s = vec3f(0.0);
  r.n_s = vec3f(0.0);
  r.Lo = vec3f(0.0);
  r.source_pdf = 0.0;
  r.w_sum = 0.0;
  r.M = 0.0;
  r.W = 0.0;
  return r;
}

fn luminance(c: vec3f) -> f32 {
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

// Target pdf (at visible point) for a given sample. For Lambertian with
// albedo a and outgoing radiance Lo: p̂ = luminance(a * Lo). This matches
// the integrand proportional to the final shaded contribution.
fn target_pdf_at(albedo: vec3f, Lo: vec3f) -> f32 {
  return luminance(albedo * Lo);
}

// WRS update: returns true if the new sample is selected by the reservoir.
fn reservoir_update(r: ptr<function, Reservoir>,
                    x_s: vec3f, n_s: vec3f, Lo: vec3f, source_pdf: f32,
                    w_new: f32, xi: f32) -> bool {
  (*r).w_sum = (*r).w_sum + w_new;
  (*r).M = (*r).M + 1.0;
  let select_new = xi * (*r).w_sum < w_new;
  if select_new {
    (*r).x_s = x_s;
    (*r).n_s = n_s;
    (*r).Lo = Lo;
    (*r).source_pdf = source_pdf;
  }
  return select_new;
}

// Finalize W from w_sum and target pdf. Call after all merges done.
fn reservoir_finalize(r: ptr<function, Reservoir>, target_pdf: f32) {
  if target_pdf > 0.0 && (*r).M > 0.0 {
    (*r).W = (*r).w_sum / ((*r).M * target_pdf);
  } else {
    (*r).W = 0.0;
  }
}

// ============================================================
// restir_shade — final combine, replaces old `finalize` kernel.
// final = direct_rad + albedo_primary * Lo
// ============================================================
@compute @workgroup_size(8, 8)
fn restir_shade(@builtin(global_invocation_id) gid: vec3u) {
  let pixel = vec2u(gid.xy);
  let res = vec2u(uniforms.resolution);
  if pixel.x >= res.x || pixel.y >= res.y { return; }
  let idx = pixel.y * res.x + pixel.x;

  let base = idx * 4u;
  let valid = candidate_buf[base + 1u].w;
  let Lo = candidate_buf[base + 2u].xyz;
  let direct = candidate_buf[base + 3u].xyz;
  let albedo = textureLoad(albedo_read, vec2i(pixel), 0).rgb;

  var indirect = vec3f(0.0);
  if valid > 0.5 {
    indirect = albedo * Lo;
  }

  var rad = direct + indirect;
  // Firefly luminance clamp — same threshold as the old finalize path.
  let lum = luminance(rad);
  let max_lum = 32.0;
  if lum > max_lum { rad *= max_lum / lum; }
  textureStore(noisy_out, vec2i(pixel), vec4f(rad, 1.0));
}
