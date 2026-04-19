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

fn luminance(c: vec3f) -> f32 {
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

// Target pdf (at visible point) for a given sample. For Lambertian with
// albedo a and outgoing radiance Lo: p̂ = luminance(a * Lo). This matches
// the integrand proportional to the final shaded contribution.
fn target_pdf_at(albedo: vec3f, Lo: vec3f) -> f32 {
  return luminance(albedo * Lo);
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

// Finalize W = w_sum / (M × p̂(selected)). Call after all combines done.
fn reservoir_finalize(r: ptr<function, Reservoir>, target_pdf: f32) {
  if target_pdf > 0.0 && (*r).M > 0.0 {
    (*r).W = (*r).w_sum / ((*r).M * target_pdf);
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

  // (1) Current candidate
  if valid_c && source_pdf_c > 1e-8 {
    let p_hat = target_pdf_at(albedo, Lo_c);
    if p_hat > 0.0 {
      let w = p_hat / source_pdf_c;
      let xi = sampler_1d(&s);
      reservoir_combine(&r, x_s_c, n_s_c, Lo_c, source_pdf_c, w, 1.0, xi);
    }
  }

  // (2) Temporal reuse via motion reprojection
  if g_curr.valid > 0.5 {
    let rp = reproject_prev(g_curr.x_v);
    if rp.in_front && rp.uv.x >= 0.0 && rp.uv.x < 1.0
                   && rp.uv.y >= 0.0 && rp.uv.y < 1.0 {
      let pixel_prev = vec2u(rp.uv * uniforms.resolution);
      let idx_prev = pixel_prev.y * res.x + pixel_prev.x;
      let g_prev = gbuf_prev_load(idx_prev);

      // Validate: same surface under the reprojected pixel?
      let depth_rel_err = abs(rp.depth_view - g_prev.depth_view)
                          / max(rp.depth_view, 0.1);
      let normal_dot = dot(g_curr.n_v, g_prev.n_v);
      let valid_hist = g_prev.valid > 0.5
                     && depth_rel_err < DEPTH_VALID_RELERR
                     && normal_dot > NORMAL_VALID_COS;

      if valid_hist {
        let prev = load_reservoir_prev(idx_prev);
        if prev.M > 0.0 {
          // Visibility validation: trace a shadow ray from the current
          // visible point to the previous sample point. If occluded, the
          // path the prev reservoir represents is no longer realisable
          // from this pixel — reuse would introduce energy that doesn't
          // exist in the current scene (a dynamic occluder moved into
          // the connection). max_t just short of x_s so we don't count
          // x_s's own surface as blocker.
          let seg = prev.x_s - g_curr.x_v;
          let seg_len = length(seg);
          let seg_dir = seg / max(seg_len, 1e-10);
          let ray_org = ray_offset(g_curr.x_v, g_curr.n_v);
          let max_t = seg_len * 0.999;
          let occluded = trace_shadow(ray_org, seg_dir, max_t);

          if !occluded {
            // Reconnection-shift jacobian: transforms prev's reservoir from
            // g_prev.x_v's hemisphere to g_curr.x_v's hemisphere around the
            // same sample point prev.x_s. Clamped for numerical safety.
            let v_to_s_curr = g_curr.x_v - prev.x_s;
            let v_to_s_prev = g_prev.x_v - prev.x_s;
            let d_curr_sq = max(dot(v_to_s_curr, v_to_s_curr), 1e-10);
            let d_prev_sq = max(dot(v_to_s_prev, v_to_s_prev), 1e-10);
            let d_curr_inv = inverseSqrt(d_curr_sq);
            let d_prev_inv = inverseSqrt(d_prev_sq);
            let cos_curr = max(dot(prev.n_s, -v_to_s_curr * d_curr_inv), 0.0);
            let cos_prev = max(dot(prev.n_s, -v_to_s_prev * d_prev_inv), 0.0);

            if cos_curr > 1e-4 && cos_prev > 1e-4 {
              let jacobian = clamp((cos_curr / cos_prev) * (d_prev_sq / d_curr_sq),
                                   0.0, 10.0);
              let p_hat = target_pdf_at(albedo, prev.Lo);
              if p_hat > 0.0 {
                let M_p = min(prev.M, M_CLAMP);
                let w = p_hat * M_p * prev.W * jacobian;
                let xi = sampler_1d(&s);
                reservoir_combine(&r, prev.x_s, prev.n_s, prev.Lo, prev.source_pdf,
                                  w, M_p, xi);
              }
            }
          }
        }
      }
    }
  }

  let p_hat_sel = target_pdf_at(albedo, r.Lo);
  reservoir_finalize(&r, p_hat_sel);

  store_reservoir_curr(idx, r);
}

// ============================================================
// restir_shade — combine direct + resampled indirect.
// For Lambertian + cosine BSDF at primary, f_r × cos = albedo × source_pdf
// (since source_pdf = cos/π and f_r = albedo/π). The full ReSTIR estimator
// is f_r × cos × Lo × W, which collapses to albedo × source_pdf × Lo × W.
// For M=1 (Phase 1a case) W = 1/source_pdf → reduces to albedo × Lo.
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
  let albedo = textureLoad(albedo_read, vec2i(pixel), 0).rgb;

  var indirect = vec3f(0.0);
  if r.M > 0.0 && r.W > 0.0 {
    indirect = albedo * r.source_pdf * r.Lo * r.W;
  }

  var rad = direct + indirect;
  let lum = luminance(rad);
  let max_lum = 32.0;
  if lum > max_lum { rad *= max_lum / lum; }
  textureStore(noisy_out, vec2i(pixel), vec4f(rad, 1.0));
}
