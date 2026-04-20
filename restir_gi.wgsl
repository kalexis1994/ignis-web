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

// Ping-pong reservoir — 3 vec4f = 48 B per pixel.
// Phase 1b.1 requires ping-pong because motion reprojection reads from
// a DIFFERENT pixel than the one being written (thread at pixel r reads
// reservoir_prev[r_prev] and writes reservoir_curr[r]; with a single
// buffer r_prev might alias another thread's target pixel, producing
// undefined ordering between reads and writes).
//   [0] = (x_s.xyz,     source_pdf)
//   [1] = (n_s.xyz,     M_float)
//   [2] = (Lo.xyz,      W_normalized)
@group(2) @binding(5) var<storage, read_write> reservoir_curr: array<vec4f>;
@group(2) @binding(6) var<storage, read_write> reservoir_prev: array<vec4f>;

// Ping-pong G-buffer — 2 vec4f = 32 B per pixel. Written by `bounce` at
// b==0 to `gbuf_curr`; previous-frame copy lives in `gbuf_prev` for the
// temporal kernel's disocclusion test + jacobian.
//   [0] = (x_v_world.xyz, depth_view_linear)   primary hit world position + linear view Z
//   [1] = (n_v.xyz,       valid_flag)          1.0 = primary hit wrote this slot
@group(2) @binding(7) var<storage, read_write> gbuf_curr: array<vec4f>;
@group(2) @binding(8) var<storage, read_write> gbuf_prev: array<vec4f>;

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

// `luminance` is defined in wavefront.wgsl (prepended at load time).

// Target pdf at a visible point for a given sample.
// Ouyang 2021: p̂(X) = || BRDF(v,X) × L(X) × cos(n_v, v→s) ||.
// For Lambertian BRDF=albedo/π, luminance of |BRDF·L·cos| ∝
// luminance(albedo·Lo)·cos_v (the /π is constant and drops out under
// RIS normalization). Including cos_v makes p̂ consistent across
// different visible points (needed for unbiased MIS weights under
// reuse); the old target_pdf_at omitted it and was only correct up
// to the shade-time BRDF×cos factor for same-pixel reuse.
fn target_pdf_gi(albedo: vec3f, Lo: vec3f, n_v: vec3f, x_v: vec3f, x_s: vec3f) -> f32 {
  let d = x_s - x_v;
  let d_sq = dot(d, d);
  if d_sq < 1e-10 { return 0.0; }
  let cos_v = max(dot(n_v, d * inverseSqrt(d_sq)), 0.0);
  return luminance(albedo * Lo) * cos_v;
}

// Generalized WRS combine — merges either a fresh candidate (M_add=1,
// w = p̂/p_src) or a previously-resampled reservoir (M_add=prev.M,
// w = p̂_at_this_pixel × prev.M × prev.W). Returns true if the new
// sample was selected.
fn reservoir_combine(
  r: ptr<function, Reservoir>,
  x_s: vec3f, n_s: vec3f, Lo: vec3f, source_pdf: f32,
  w_contrib: f32, M_add: f32, xi: f32
) -> bool {
  (*r).w_sum = (*r).w_sum + w_contrib;
  (*r).M = (*r).M + M_add;
  let select_new = xi * (*r).w_sum < w_contrib;
  if select_new {
    (*r).x_s = x_s;
    (*r).n_s = n_s;
    (*r).Lo = Lo;
    (*r).source_pdf = source_pdf;
  }
  return select_new;
}

// Finalize W = w_sum / p̂(selected). With pairwise MIS the M weighting
// is already inside each m_i (balance-heuristic has M_i in numerator
// and Σ M_j terms in denominator), so per-sample contributions are
// w_i = m_i × p̂_r(X_i) × W_i — no extra M_i factor — and W divides by
// p̂ only, not by M_total. Dividing by M_total here would shrink W by
// that factor and produce systematic darkening (most noticeable in
// shadow regions where history M accumulates). M stays tracked on the
// reservoir for clamping / next-frame merging.
fn reservoir_finalize(r: ptr<function, Reservoir>, target_pdf: f32) {
  if target_pdf > 0.0 {
    (*r).W = (*r).w_sum / target_pdf;
  } else {
    (*r).W = 0.0;
  }
}

// Persistent I/O. w_sum is NOT stored (only meaningful transiently inside
// a combine pass). M and W are enough to re-merge next frame.
// Separate curr/prev helpers — WGSL can't easily take storage-buffer
// pointers as function args, so we duplicate the tiny accessors.
fn load_reservoir_prev(idx: u32) -> Reservoir {
  let base = idx * 3u;
  let v0 = reservoir_prev[base];
  let v1 = reservoir_prev[base + 1u];
  let v2 = reservoir_prev[base + 2u];
  var r: Reservoir;
  r.x_s = v0.xyz;
  r.source_pdf = v0.w;
  r.n_s = v1.xyz;
  r.M = v1.w;
  r.Lo = v2.xyz;
  r.W = v2.w;
  r.w_sum = 0.0;
  return r;
}

fn load_reservoir_curr(idx: u32) -> Reservoir {
  let base = idx * 3u;
  let v0 = reservoir_curr[base];
  let v1 = reservoir_curr[base + 1u];
  let v2 = reservoir_curr[base + 2u];
  var r: Reservoir;
  r.x_s = v0.xyz;
  r.source_pdf = v0.w;
  r.n_s = v1.xyz;
  r.M = v1.w;
  r.Lo = v2.xyz;
  r.W = v2.w;
  r.w_sum = 0.0;
  return r;
}

fn store_reservoir_curr(idx: u32, r: Reservoir) {
  let base = idx * 3u;
  reservoir_curr[base]      = vec4f(r.x_s, r.source_pdf);
  reservoir_curr[base + 1u] = vec4f(r.n_s, r.M);
  reservoir_curr[base + 2u] = vec4f(r.Lo, r.W);
}

// ============================================================
// G-buffer: primary-hit world position, linear view depth, and normal.
// Written once per frame by bounce at b==0; ping-ponged so restir_temporal
// can read the previous-frame copy at the reprojected pixel for its
// disocclusion / plane-distance validation.
// ============================================================
struct GBufEntry {
  x_v: vec3f,
  depth_view: f32,
  n_v: vec3f,
  valid: f32,
};

fn gbuf_curr_write(idx: u32, x_v: vec3f, depth_view: f32, n_v: vec3f) {
  let base = idx * 2u;
  gbuf_curr[base]      = vec4f(x_v, depth_view);
  gbuf_curr[base + 1u] = vec4f(n_v, 1.0);
}

fn gbuf_curr_invalidate(idx: u32) {
  let base = idx * 2u;
  gbuf_curr[base]      = vec4f(0.0);
  gbuf_curr[base + 1u] = vec4f(0.0);  // valid=0
}

fn gbuf_curr_load(idx: u32) -> GBufEntry {
  let base = idx * 2u;
  let v0 = gbuf_curr[base];
  let v1 = gbuf_curr[base + 1u];
  var g: GBufEntry;
  g.x_v = v0.xyz;
  g.depth_view = v0.w;
  g.n_v = v1.xyz;
  g.valid = v1.w;
  return g;
}

fn gbuf_prev_load(idx: u32) -> GBufEntry {
  let base = idx * 2u;
  let v0 = gbuf_prev[base];
  let v1 = gbuf_prev[base + 1u];
  var g: GBufEntry;
  g.x_v = v0.xyz;
  g.depth_view = v0.w;
  g.n_v = v1.xyz;
  g.valid = v1.w;
  return g;
}

// ============================================================
// Reproject a world-space point into the previous frame's pixel grid.
// Uses the inverse of the camera-decomposed ray generation in `generate`.
// Returns (uv_prev, depth_view_prev). uv_prev outside [0,1] → off-screen.
// ============================================================
struct ReprojectResult {
  uv: vec2f,
  depth_view: f32,
  in_front: bool,
};

fn reproject_prev(x_world: vec3f) -> ReprojectResult {
  let rel = x_world - uniforms.prev_cam_pos;
  let z_view = dot(rel, uniforms.prev_cam_forward);
  var r: ReprojectResult;
  r.depth_view = z_view;
  r.in_front = z_view > 1e-4;
  if !r.in_front {
    r.uv = vec2f(-1.0);
    return r;
  }
  let x_cam = dot(rel, uniforms.prev_cam_right);
  let y_cam = dot(rel, uniforms.prev_cam_up);
  let aspect = uniforms.resolution.x / uniforms.resolution.y;
  let ndc_x = x_cam / (z_view * aspect * uniforms.prev_fov_factor);
  let ndc_y = y_cam / (z_view * uniforms.prev_fov_factor);
  r.uv = vec2f((ndc_x + 1.0) * 0.5, (ndc_y + 1.0) * 0.5);
  return r;
}

// ============================================================
// restir_temporal — Phase 1b.1 temporal reuse with motion reprojection.
//
// Pipeline per pixel:
//   1. Load current-frame candidate and current-frame g-buffer entry.
//   2. WRS-add current candidate (M=1, w = p̂/p_src).
//   3. Reproject x_v_curr into previous-frame NDC using prev cam pose.
//   4. If reprojection lands on-screen, fetch g_prev + reservoir_prev
//      at that pixel.
//   5. Validate: normal dot > 0.9 AND relative depth error < 10 %.
//   6. Compute reconnection-shift jacobian
//         J = (cos(θ_curr)/cos(θ_prev)) × (d_prev² / d_curr²)
//      where d = ||x_v - x_s|| and θ is the angle between n_s and
//      (x_v - x_s) normalized.
//   7. WRS-add prev reservoir with w = p̂_curr(Lo_prev) × M_prev × W_prev × J,
//      M_add = min(M_prev, M_CLAMP).
//   8. Finalize W, store into reservoir_curr.
// ============================================================
const M_CLAMP: f32 = 20.0;          // ≈ 0.33 s at 60 fps — caps stale influence
const NORMAL_VALID_COS: f32 = 0.9;  // reject if normals diverge by >~26°
const DEPTH_VALID_RELERR: f32 = 0.1;// reject if view-z relative error >10%

@compute @workgroup_size(8, 8)
fn restir_temporal(@builtin(global_invocation_id) gid: vec3u) {
  let pixel = vec2u(gid.xy);
  let res = vec2u(uniforms.resolution);
  if pixel.x >= res.x || pixel.y >= res.y { return; }
  let idx = pixel.y * res.x + pixel.x;

  let g_curr = gbuf_curr_load(idx);

  // Current candidate
  let cbase = idx * 4u;
  let x_s_c        = candidate_buf[cbase].xyz;
  let source_pdf_c = candidate_buf[cbase].w;
  let n_s_c        = candidate_buf[cbase + 1u].xyz;
  let valid_c      = candidate_buf[cbase + 1u].w > 0.5;
  let Lo_c         = candidate_buf[cbase + 2u].xyz;

  let albedo = textureLoad(albedo_read, vec2i(pixel), 0).rgb;

  var s = sampler_init(idx, uniforms.frame_seed ^ 0x9E3779B1u, 0u);
  var r = reservoir_empty();

  // First locate + validate the prev reservoir (if any). We need to
  // know whether it's usable before we set MIS weights on the canonical
  // candidate, since m_r depends on whether R_q contributes at all.
  var has_prev = false;
  var prev: Reservoir;
  var g_prev: GBufEntry;
  var jacobian_prev: f32 = 1.0;

  if g_curr.valid > 0.5 {
    let rp = reproject_prev(g_curr.x_v);
    if rp.in_front && rp.uv.x >= 0.0 && rp.uv.x < 1.0
                   && rp.uv.y >= 0.0 && rp.uv.y < 1.0 {
      let pixel_prev = vec2u(rp.uv * uniforms.resolution);
      let idx_prev = pixel_prev.y * res.x + pixel_prev.x;
      let g_p = gbuf_prev_load(idx_prev);

      let depth_rel_err = abs(rp.depth_view - g_p.depth_view)
                          / max(rp.depth_view, 0.1);
      let normal_dot = dot(g_curr.n_v, g_p.n_v);
      let valid_hist = g_p.valid > 0.5
                     && depth_rel_err < DEPTH_VALID_RELERR
                     && normal_dot > NORMAL_VALID_COS;

      if valid_hist {
        let pr = load_reservoir_prev(idx_prev);
        if pr.M > 0.0 {
          // Visibility: trace v_curr → pr.x_s, reject if occluded.
          let seg = pr.x_s - g_curr.x_v;
          let seg_len = length(seg);
          let seg_dir = seg / max(seg_len, 1e-10);
          let ray_org = ray_offset(g_curr.x_v, g_curr.n_v);
          if !trace_shadow(ray_org, seg_dir, seg_len * 0.999) {
            // Reconnection-shift jacobian
            let s_to_v_curr = g_curr.x_v - pr.x_s;
            let s_to_v_prev = g_p.x_v - pr.x_s;
            let d_curr_sq = max(dot(s_to_v_curr, s_to_v_curr), 1e-10);
            let d_prev_sq = max(dot(s_to_v_prev, s_to_v_prev), 1e-10);
            let cos_curr = max(dot(pr.n_s, s_to_v_curr * inverseSqrt(d_curr_sq)), 0.0);
            let cos_prev = max(dot(pr.n_s, s_to_v_prev * inverseSqrt(d_prev_sq)), 0.0);
            if cos_curr > 1e-4 && cos_prev > 1e-4 {
              has_prev = true;
              prev = pr;
              g_prev = g_p;
              jacobian_prev = clamp((cos_curr / cos_prev) * (d_prev_sq / d_curr_sq),
                                    0.0, 10.0);
            }
          }
        }
      }
    }
  }

  // Pairwise MIS for k=1: exact balance heuristic between canonical
  // candidate (M_r=1) and the reprojected prev reservoir (M_q = clamped
  // prev.M). Sums to 1 per sample.
  //   m_r(X) = M_r·p̂_r(X) / (M_r·p̂_r(X) + M_q·p̂_q(X))
  //   m_q(X) = M_q·p̂_q(X) / (M_r·p̂_r(X) + M_q·p̂_q(X))
  // p̂_q evaluated at prev pixel's visible point. Prev-frame albedo is
  // not persisted, so we approximate it as current-frame albedo — fine
  // for static scenes, slight MIS inaccuracy under moving recolored
  // surfaces (not a correctness-critical approximation; it just means
  // MIS weights are slightly suboptimal in variance-reduction terms).
  let M_r: f32 = 1.0;
  let M_q: f32 = select(0.0, min(prev.M, M_CLAMP), has_prev);

  // (1) Canonical candidate (X_c)
  if valid_c && source_pdf_c > 1e-8 {
    let p_hat_r_Xc = target_pdf_gi(albedo, Lo_c, g_curr.n_v, g_curr.x_v, x_s_c);
    var m_r: f32 = 1.0;
    if has_prev {
      let p_hat_q_Xc = target_pdf_gi(albedo, Lo_c, g_prev.n_v, g_prev.x_v, x_s_c);
      let denom = M_r * p_hat_r_Xc + M_q * p_hat_q_Xc;
      if denom > 1e-10 {
        m_r = (M_r * p_hat_r_Xc) / denom;
      }
    }
    if p_hat_r_Xc > 0.0 {
      // Pairwise MIS weight formula: w_i = m_i · p̂_r(X_i) · W_i.
      // For the canonical fresh sample W_i = 1/source_pdf_c (the
      // standard RIS initial weight). No M factor here — M is carried
      // inside m_i's balance-heuristic numerator.
      let w = m_r * p_hat_r_Xc * (1.0 / source_pdf_c);
      let xi = sampler_1d(&s);
      reservoir_combine(&r, x_s_c, n_s_c, Lo_c, source_pdf_c, w, M_r, xi);
    }
  }

  // (2) Reused prev reservoir (X_q)
  if has_prev {
    let p_hat_r_Xq = target_pdf_gi(albedo, prev.Lo, g_curr.n_v, g_curr.x_v, prev.x_s);
    let p_hat_q_Xq = target_pdf_gi(albedo, prev.Lo, g_prev.n_v, g_prev.x_v, prev.x_s);
    let denom = M_r * p_hat_r_Xq + M_q * p_hat_q_Xq;
    if denom > 1e-10 && p_hat_r_Xq > 0.0 {
      let m_q = (M_q * p_hat_q_Xq) / denom;
      let w = m_q * p_hat_r_Xq * prev.W * jacobian_prev;
      let xi = sampler_1d(&s);
      reservoir_combine(&r, prev.x_s, prev.n_s, prev.Lo, prev.source_pdf,
                        w, M_q, xi);
    }
  }

  let p_hat_sel = target_pdf_gi(albedo, r.Lo, g_curr.n_v, g_curr.x_v, r.x_s);
  reservoir_finalize(&r, p_hat_sel);

  store_reservoir_curr(idx, r);
}

// ============================================================
// restir_spatial — Phase 1c spatial reuse.
// Runs after restir_temporal. Reads from reservoir_prev binding (which
// under the bg2_spatial mapping is the temporal-out slot), merges the
// pixel's own temporal reservoir plus k random neighbors' temporal
// reservoirs, writes to reservoir_curr (spatial-out slot).
//
// For each neighbor q:
//   1. Validate surface (normal dot > 0.9, depth rel err < 10 %).
//   2. Load neighbor's temporal reservoir.
//   3. Shadow ray v_r → prev_q.x_s for visibility.
//   4. Reconnection-shift jacobian from q's visible point to r's.
//   5. WRS merge with weight w = p̂_r(Lo_q) × M_q × W_q × J.
// ============================================================
const SPATIAL_K: u32 = 5u;          // neighbors per pixel
const INV_SPATIAL_K: f32 = 0.2;     // 1 / SPATIAL_K — pairwise MIS normalizer
const SPATIAL_RADIUS: f32 = 16.0;   // disk radius in pixels

@compute @workgroup_size(8, 8)
fn restir_spatial(@builtin(global_invocation_id) gid: vec3u) {
  let pixel = vec2u(gid.xy);
  let res = vec2u(uniforms.resolution);
  if pixel.x >= res.x || pixel.y >= res.y { return; }
  let idx = pixel.y * res.x + pixel.x;

  let g_r = gbuf_curr_load(idx);
  let albedo_r = textureLoad(albedo_read, vec2i(pixel), 0).rgb;

  var s = sampler_init(idx, uniforms.frame_seed ^ 0xC6BC279Fu, 0u);

  // Load the canonical (own temporal-stage) reservoir up front. Its MIS
  // weight depends on which neighbors end up validating, so we merge it
  // AFTER the neighbors while tracking the running sum of m_q(X_r).
  let own = load_reservoir_prev(idx);
  let M_r = own.M;
  let p_hat_r_Xr = target_pdf_gi(albedo_r, own.Lo, g_r.n_v, g_r.x_v, own.x_s);

  var r = reservoir_empty();
  // Running sum of m_i(X_r) over contributing neighbors. m_r = 1 - this.
  var canonical_mis_minus: f32 = 0.0;

  if g_r.valid > 0.5 && own.M > 0.0 {
    for (var k = 0u; k < SPATIAL_K; k = k + 1u) {
      let u = sampler_2d(&s);
      let theta = u.x * TWO_PI;
      let radius = sqrt(u.y) * SPATIAL_RADIUS;
      let off_x = i32(round(radius * cos(theta)));
      let off_y = i32(round(radius * sin(theta)));
      let px_q = vec2i(pixel) + vec2i(off_x, off_y);
      if px_q.x < 0 || px_q.y < 0
         || px_q.x >= i32(res.x) || px_q.y >= i32(res.y) { continue; }
      let pq = vec2u(px_q);
      let idx_q = pq.y * res.x + pq.x;
      if idx_q == idx { continue; }

      let g_q = gbuf_curr_load(idx_q);
      if g_q.valid < 0.5 { continue; }

      let depth_rel_err = abs(g_r.depth_view - g_q.depth_view)
                          / max(g_r.depth_view, 0.1);
      let normal_dot = dot(g_r.n_v, g_q.n_v);
      if depth_rel_err > DEPTH_VALID_RELERR || normal_dot < NORMAL_VALID_COS {
        continue;
      }

      let prev_q = load_reservoir_prev(idx_q);
      if prev_q.M <= 0.0 { continue; }

      // Visibility shadow ray
      let seg = prev_q.x_s - g_r.x_v;
      let seg_len = length(seg);
      let seg_dir = seg / max(seg_len, 1e-10);
      let ray_org = ray_offset(g_r.x_v, g_r.n_v);
      if trace_shadow(ray_org, seg_dir, seg_len * 0.999) { continue; }

      // Reconnection-shift jacobian from q's visible point to ours.
      let s_to_v_curr = g_r.x_v - prev_q.x_s;
      let s_to_v_nb   = g_q.x_v - prev_q.x_s;
      let d_curr_sq = max(dot(s_to_v_curr, s_to_v_curr), 1e-10);
      let d_nb_sq   = max(dot(s_to_v_nb,   s_to_v_nb),   1e-10);
      let cos_curr = max(dot(prev_q.n_s, s_to_v_curr * inverseSqrt(d_curr_sq)), 0.0);
      let cos_nb   = max(dot(prev_q.n_s, s_to_v_nb   * inverseSqrt(d_nb_sq)),   0.0);
      if cos_curr < 1e-4 || cos_nb < 1e-4 { continue; }
      let jacobian = clamp((cos_curr / cos_nb) * (d_nb_sq / d_curr_sq),
                           0.0, 10.0);

      let M_q = min(prev_q.M, M_CLAMP);
      let albedo_q = textureLoad(albedo_read, vec2i(pq), 0).rgb;

      // Defensive pairwise MIS (Bitterli 2020). Each per-pair balance
      // heuristic is scaled by 1/k so the k neighbor weights sum to at
      // most 1, leaving m_r = 1 − Σ m_q_i non-negative without clamping
      // and the total MIS weight equal to exactly 1 (energy-conserving).
      // Without the 1/k, Σ m_q_i could exceed 1 while m_r clamps to 0 →
      // MIS weights sum > 1 → up to k× over-brightening in regions
      // where neighbors dominate the canonical.
      //   m_q(X_q) = (1/k) · M_q·p̂_q(X_q) / (M_r·p̂_r(X_q) + M_q·p̂_q(X_q))
      //   m_q(X_r) = (1/k) · M_q·p̂_q(X_r) / (M_r·p̂_r(X_r) + M_q·p̂_q(X_r))
      // canonical_mis gets 1 − Σ m_q(X_r). Exact energy conservation
      // for any k; residual MIS-suboptimality from ignoring inter-
      // neighbor q_i vs q_j pairs is a variance-only effect.
      let p_hat_r_Xq = target_pdf_gi(albedo_r, prev_q.Lo, g_r.n_v, g_r.x_v, prev_q.x_s);
      let p_hat_q_Xq = target_pdf_gi(albedo_q, prev_q.Lo, g_q.n_v, g_q.x_v, prev_q.x_s);

      if p_hat_r_Xq <= 0.0 { continue; }

      let denom_q = M_r * p_hat_r_Xq + M_q * p_hat_q_Xq;
      if denom_q <= 1e-10 { continue; }
      let m_q = INV_SPATIAL_K * (M_q * p_hat_q_Xq) / denom_q;

      // Accumulate canonical MIS fraction at X_r from this neighbor
      let p_hat_q_Xr = target_pdf_gi(albedo_q, own.Lo, g_q.n_v, g_q.x_v, own.x_s);
      let denom_r = M_r * p_hat_r_Xr + M_q * p_hat_q_Xr;
      if denom_r > 1e-10 {
        canonical_mis_minus = canonical_mis_minus
                            + INV_SPATIAL_K * (M_q * p_hat_q_Xr) / denom_r;
      }

      // Pairwise MIS weight formula (same as temporal):
      // w_i = m_i · p̂_r(X_i) · W_i · J_i. No M factor.
      let w = m_q * p_hat_r_Xq * prev_q.W * jacobian;
      let xi = sampler_1d(&s);
      reservoir_combine(&r, prev_q.x_s, prev_q.n_s, prev_q.Lo, prev_q.source_pdf,
                        w, M_q, xi);
    }
  }

  // Merge canonical last with complement MIS weight.
  if own.M > 0.0 && p_hat_r_Xr > 0.0 {
    let m_r = clamp(1.0 - canonical_mis_minus, 0.0, 1.0);
    let w = m_r * p_hat_r_Xr * own.W;
    let xi = sampler_1d(&s);
    reservoir_combine(&r, own.x_s, own.n_s, own.Lo, own.source_pdf,
                      w, M_r, xi);
  }

  let p_hat_sel = target_pdf_gi(albedo_r, r.Lo, g_r.n_v, g_r.x_v, r.x_s);
  reservoir_finalize(&r, p_hat_sel);
  store_reservoir_curr(idx, r);
}

// ============================================================
// restir_shade — combine direct + resampled indirect.
// Proper ReSTIR estimator: indirect = f_r × Lo × cos_v × W
// For Lambertian f_r = albedo / π, and cos_v is the angle between the
// current pixel's n_v and the direction to the selected sample point
// x_s. source_pdf stored in the reservoir came from the ORIGINAL
// sample's pixel (via cosine BSDF) and doesn't apply here when the
// sample was reused from a neighbor — we must re-evaluate cos_v at
// our own visible point.
// ============================================================
@compute @workgroup_size(8, 8)
fn restir_shade(@builtin(global_invocation_id) gid: vec3u) {
  let pixel = vec2u(gid.xy);
  let res = vec2u(uniforms.resolution);
  if pixel.x >= res.x || pixel.y >= res.y { return; }
  let idx = pixel.y * res.x + pixel.x;

  // Direct component stays in candidate_buf (per-frame, not resampled).
  let cbase = idx * 4u;
  let direct = candidate_buf[cbase + 3u].xyz;

  let r = load_reservoir_curr(idx);
  let g_curr = gbuf_curr_load(idx);
  let albedo = textureLoad(albedo_read, vec2i(pixel), 0).rgb;

  var indirect = vec3f(0.0);
  if r.M > 0.0 && r.W > 0.0 && g_curr.valid > 0.5 {
    let v_to_s = r.x_s - g_curr.x_v;
    let d_sq = dot(v_to_s, v_to_s);
    if d_sq > 1e-10 {
      let cos_v = max(dot(g_curr.n_v, v_to_s * inverseSqrt(d_sq)), 0.0);
      indirect = (albedo * INV_PI) * r.Lo * cos_v * r.W;
    }
  }

  var rad = direct + indirect;
  let lum = luminance(rad);
  let max_lum = 32.0;
  if lum > max_lum { rad *= max_lum / lum; }
  textureStore(noisy_out, vec2i(pixel), vec4f(rad, 1.0));
}

// ============================================================
// relax_temporal — ReLAX-style per-pixel temporal accumulation.
//
// Port of the core logic in NVIDIA NRD's
// RELAX_TemporalAccumulation.cs.hlsl (v4.x), simplified for a single
// diffuse stream and Lambertian geometry:
//   • Single (non-fast/non-SH) history, packed into accum.rgb.
//   • Per-pixel history length packed into accum.a (range 0..63, so
//     rgba16float has ample precision and no extra texture is needed).
//   • No checkerboard, no material IDs, no hit distance reconstruction.
//
// Per-pixel pipeline:
//   1. Load the current primary hit's g-buffer entry. If invalid
//      (primary miss, unlit, out of frustum), pass through with
//      history = 1.
//   2. Reproject the current world-space visible point into the
//      previous frame's NDC using the stored prev camera pose.
//   3. Bilinear custom-weights sample over the 4 surrounding prev
//      pixels. Each tap is validated:
//        • prev g-buffer entry is valid (primary hit existed there)
//        • plane distance from current surface < disocclusion threshold
//        • normal agreement dot ≥ NORMAL_DOT_MIN (backface / cone)
//      Invalid taps get weight 0; valid weights are re-normalized by
//      their sum. If every tap rejects (weight_sum ≈ 0) the pixel is
//      treated as disoccluded → alpha = 1.
//   4. alpha = max(1 / (MAX_FRAMES + 1), 1 / history_length) once a
//      valid reprojection is found. During warmup (history < MAX),
//      1/history_length dominates so the first few frames effectively
//      do a running mean; once history saturates, alpha floors at
//      1/(MAX+1) and behaves like an exponential blend with that time
//      constant. RELAX 4.x uses exactly this formula — see line 604
//      of RELAX_TemporalAccumulation.cs.hlsl.
//   5. history_length is incremented by 1 per successful reproject,
//      clamped to MAX_ACCUM_FRAMES. A disocclusion resets it to 1.
//
// The disocclusion threshold is derived from the world-space size of a
// pixel at the current view depth, scaled by a slack factor. This
// matches RELAX's PixelRadiusToWorld-based computation but with a
// constant NoV factor (1.0) — proper NoV-adaptive slack is a Phase 1.5
// refinement and mostly matters at grazing angles.
// ============================================================
const RELAX_MAX_ACCUM_FRAMES: f32 = 63.0;
const RELAX_DISOCC_PIXEL_SLACK: f32 = 10.0;
const RELAX_NORMAL_DOT_MIN: f32 = 0.9;

@compute @workgroup_size(8, 8)
fn relax_temporal(@builtin(global_invocation_id) gid: vec3u) {
  let pixel = vec2u(gid.xy);
  let res = vec2u(uniforms.resolution);
  if pixel.x >= res.x || pixel.y >= res.y { return; }
  let idx = pixel.y * res.x + pixel.x;

  let curr = textureLoad(noisy_read, vec2i(pixel), 0).rgb;
  let g_curr = gbuf_curr_load(idx);

  // Primary miss / unlit / out-of-frustum: no geometry to anchor
  // reprojection on. Just emit the current sample with a fresh
  // history. Downstream HistoryFix (Phase 2) fills these from
  // neighbors if present; at this stage they show the raw frame.
  if g_curr.valid < 0.5 {
    textureStore(accum_new, vec2i(pixel), vec4f(curr, 1.0 / RELAX_MAX_ACCUM_FRAMES));
    return;
  }

  // Reproject. `reproject_prev` uses the stored prev camera pose and
  // returns uv in [0,1] when on-screen.
  let rp = reproject_prev(g_curr.x_v);

  var prev_rgb = vec3f(0.0);
  var prev_hist: f32 = 0.0;
  var weight_sum: f32 = 0.0;

  if rp.in_front && rp.uv.x > 0.0 && rp.uv.x < 1.0 && rp.uv.y > 0.0 && rp.uv.y < 1.0 {
    // Bilinear footprint: pixel-space (uv - 0.5 shift puts the sample
    // centers at integer coords), fractional component gives weights.
    let prev_pf = rp.uv * uniforms.resolution - vec2f(0.5);
    let p0 = floor(prev_pf);
    let fr = prev_pf - p0;
    let w00 = (1.0 - fr.x) * (1.0 - fr.y);
    let w10 = fr.x * (1.0 - fr.y);
    let w01 = (1.0 - fr.x) * fr.y;
    let w11 = fr.x * fr.y;

    // World-size of a pixel at the current depth → disocclusion slack.
    //   tan(fov/2) = fov_factor, so half-height at depth d = d*fov_factor
    //   pixel_world_size = 2 * d * fov_factor / res.y
    let pixel_world = g_curr.depth_view * uniforms.fov_factor * 2.0
                      / max(uniforms.resolution.y, 1.0);
    let disocc_th = pixel_world * RELAX_DISOCC_PIXEL_SLACK;

    // 4 bilinear taps, each validated against current surface.
    let base_p = vec2i(p0);

    // Tap (0,0)
    {
      let tp = base_p + vec2i(0, 0);
      let w = w00;
      if tp.x >= 0 && tp.x < i32(res.x) && tp.y >= 0 && tp.y < i32(res.y) && w > 0.0 {
        let tidx = u32(tp.y) * res.x + u32(tp.x);
        let g_tap = gbuf_prev_load(tidx);
        // View-space Z distance in PREV camera frame. rp.depth_view is
        // the current world point projected into prev view space;
        // g_tap.depth_view is the prev frame's stored view Z at the
        // reprojected pixel. Matches the approach used by NVIDIA NRD's
        // RELAX_TemporalAccumulation.cs.hlsl:121 — world-space plane
        // distance along the shading normal (what we had before) is
        // zero for lateral displacements perpendicular to the normal,
        // letting stale history from a different surface at the same
        // pixel pass as valid whenever the two surfaces happened to be
        // on the same normal-plane (common on floors/ceilings seen from
        // rotated viewpoints).
        let plane_d = abs(rp.depth_view - g_tap.depth_view);
        if g_tap.valid > 0.5 && plane_d < disocc_th
           && dot(g_curr.n_v, g_tap.n_v) >= RELAX_NORMAL_DOT_MIN {
          let a = textureLoad(accum_prev, tp, 0);
          prev_rgb += w * a.rgb;
          prev_hist += w * a.a * RELAX_MAX_ACCUM_FRAMES;
          weight_sum += w;
        }
      }
    }
    // Tap (1,0)
    {
      let tp = base_p + vec2i(1, 0);
      let w = w10;
      if tp.x >= 0 && tp.x < i32(res.x) && tp.y >= 0 && tp.y < i32(res.y) && w > 0.0 {
        let tidx = u32(tp.y) * res.x + u32(tp.x);
        let g_tap = gbuf_prev_load(tidx);
        // View-space Z distance in PREV camera frame. rp.depth_view is
        // the current world point projected into prev view space;
        // g_tap.depth_view is the prev frame's stored view Z at the
        // reprojected pixel. Matches the approach used by NVIDIA NRD's
        // RELAX_TemporalAccumulation.cs.hlsl:121 — world-space plane
        // distance along the shading normal (what we had before) is
        // zero for lateral displacements perpendicular to the normal,
        // letting stale history from a different surface at the same
        // pixel pass as valid whenever the two surfaces happened to be
        // on the same normal-plane (common on floors/ceilings seen from
        // rotated viewpoints).
        let plane_d = abs(rp.depth_view - g_tap.depth_view);
        if g_tap.valid > 0.5 && plane_d < disocc_th
           && dot(g_curr.n_v, g_tap.n_v) >= RELAX_NORMAL_DOT_MIN {
          let a = textureLoad(accum_prev, tp, 0);
          prev_rgb += w * a.rgb;
          prev_hist += w * a.a * RELAX_MAX_ACCUM_FRAMES;
          weight_sum += w;
        }
      }
    }
    // Tap (0,1)
    {
      let tp = base_p + vec2i(0, 1);
      let w = w01;
      if tp.x >= 0 && tp.x < i32(res.x) && tp.y >= 0 && tp.y < i32(res.y) && w > 0.0 {
        let tidx = u32(tp.y) * res.x + u32(tp.x);
        let g_tap = gbuf_prev_load(tidx);
        // View-space Z distance in PREV camera frame. rp.depth_view is
        // the current world point projected into prev view space;
        // g_tap.depth_view is the prev frame's stored view Z at the
        // reprojected pixel. Matches the approach used by NVIDIA NRD's
        // RELAX_TemporalAccumulation.cs.hlsl:121 — world-space plane
        // distance along the shading normal (what we had before) is
        // zero for lateral displacements perpendicular to the normal,
        // letting stale history from a different surface at the same
        // pixel pass as valid whenever the two surfaces happened to be
        // on the same normal-plane (common on floors/ceilings seen from
        // rotated viewpoints).
        let plane_d = abs(rp.depth_view - g_tap.depth_view);
        if g_tap.valid > 0.5 && plane_d < disocc_th
           && dot(g_curr.n_v, g_tap.n_v) >= RELAX_NORMAL_DOT_MIN {
          let a = textureLoad(accum_prev, tp, 0);
          prev_rgb += w * a.rgb;
          prev_hist += w * a.a * RELAX_MAX_ACCUM_FRAMES;
          weight_sum += w;
        }
      }
    }
    // Tap (1,1)
    {
      let tp = base_p + vec2i(1, 1);
      let w = w11;
      if tp.x >= 0 && tp.x < i32(res.x) && tp.y >= 0 && tp.y < i32(res.y) && w > 0.0 {
        let tidx = u32(tp.y) * res.x + u32(tp.x);
        let g_tap = gbuf_prev_load(tidx);
        // View-space Z distance in PREV camera frame. rp.depth_view is
        // the current world point projected into prev view space;
        // g_tap.depth_view is the prev frame's stored view Z at the
        // reprojected pixel. Matches the approach used by NVIDIA NRD's
        // RELAX_TemporalAccumulation.cs.hlsl:121 — world-space plane
        // distance along the shading normal (what we had before) is
        // zero for lateral displacements perpendicular to the normal,
        // letting stale history from a different surface at the same
        // pixel pass as valid whenever the two surfaces happened to be
        // on the same normal-plane (common on floors/ceilings seen from
        // rotated viewpoints).
        let plane_d = abs(rp.depth_view - g_tap.depth_view);
        if g_tap.valid > 0.5 && plane_d < disocc_th
           && dot(g_curr.n_v, g_tap.n_v) >= RELAX_NORMAL_DOT_MIN {
          let a = textureLoad(accum_prev, tp, 0);
          prev_rgb += w * a.rgb;
          prev_hist += w * a.a * RELAX_MAX_ACCUM_FRAMES;
          weight_sum += w;
        }
      }
    }
  }

  var history_length: f32;
  var mixed: vec3f;
  if weight_sum > 0.01 {
    // Renormalize over the valid taps that contributed.
    let inv_ws = 1.0 / weight_sum;
    prev_rgb *= inv_ws;
    prev_hist *= inv_ws;
    history_length = min(prev_hist + 1.0, RELAX_MAX_ACCUM_FRAMES);
    // RELAX alpha (line 604): warmup 1/history_length dominates;
    // once history saturates, floors at 1/(MAX+1). This gives a
    // running mean up to MAX then transitions to exponential blend.
    let alpha = max(1.0 / (RELAX_MAX_ACCUM_FRAMES + 1.0), 1.0 / history_length);
    mixed = mix(prev_rgb, curr, alpha);
  } else {
    // Disocclusion: every tap rejected. Fresh sample, history = 1.
    history_length = 1.0;
    mixed = curr;
  }

  textureStore(accum_new, vec2i(pixel), vec4f(mixed, history_length / RELAX_MAX_ACCUM_FRAMES));
}
