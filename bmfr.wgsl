// BMFR — Blockwise Multi-Order Feature Regression (Koskela et al., TOG 2019)
// =========================================================================
// Reconstructs the DEMODULATED DIFFUSE irradiance from a 1-spp path tracer by
// fitting it, per 32x32 block, to noise-free geometric features:
//     f = [1, n.x, n.y, n.z, p.x, p.y, p.z]                       (7 features)
//
// Two passes (the proven Koskela structure — per-frame block offset + temporal
// accumulation removes the block grid; NO cross-block bilinear, which extrapolates):
//   fit        — per block: least-squares fit (normal equations + relative ridge +
//                Cholesky), each pixel reconstructed in ITS OWN block. Writes bmfrFitTex.
//   accumulate — per pixel: temporal EMA (reprojected) of the fit result.
//
// Solver math is f32 for conditioning; I/O f16.

enable subgroups;

struct BmfrParams {
  resolution   : vec2f,
  block_offset : vec2i,   // per-frame grid jitter in [0,31] (de-grid via temporal)
  cam_right    : vec4f,
  cam_up       : vec4f,
  cam_fwd      : vec4f,
  cam_pos      : vec4f,
  prev_right   : vec4f,
  prev_up      : vec4f,
  prev_fwd     : vec4f,
  prev_pos     : vec4f,
  fov_factor   : f32,
  aspect       : f32,
  ridge        : f32,     // RELATIVE ridge factor (scale-invariant): A[i,i] *= (1+ridge)
  min_alpha    : f32,
  frames_still : f32,
  max_history  : f32,
  depth_reject : f32,     // reserved
  _pad         : f32,
};

@group(0) @binding(0) var<uniform> bp: BmfrParams;
@group(0) @binding(1) var in_diff : texture_2d<f32>;                        // fit: noisy demodulated diffuse
@group(0) @binding(2) var in_nd   : texture_2d<f32>;                        // both: normal(xyz)+hit-distance(w)
@group(0) @binding(3) var out_diff: texture_storage_2d<rgba16float, write>; // fit: reconstructed diffuse
@group(0) @binding(4) var in_fit  : texture_2d<f32>;                        // accumulate: this frame's fit
@group(0) @binding(5) var in_hist : texture_2d<f32>;                        // accumulate: previous history
@group(0) @binding(6) var hdr_out : texture_storage_2d<rgba16float, write>; // accumulate: final (composite reads)
@group(0) @binding(7) var hist_out: texture_storage_2d<rgba16float, write>; // accumulate: new history

const BLOCK   : i32 = 32;
const WG      : u32 = 256u;
const PPT     : u32 = 4u;       // pixels per thread (1024/256)
const NF      : u32 = 7u;
const NA      : u32 = 28u;      // lower-tri of 7x7
const NB      : u32 = 21u;      // 7x3
const MAXSG   : u32 = 16u;
const SKY_Z   : f32 = 1.0e5;

var<workgroup> sh_A : array<array<f32, 28>, 16>;
var<workgroup> sh_b : array<array<f32, 21>, 16>;
var<workgroup> sh_alpha : array<f32, 21>;
var<workgroup> sh_valid : u32;
var<workgroup> sh_ref_pos : vec3f;     // FIX #1: valid (non-sky) reference for normalization
var<workgroup> sh_inv_scale : f32;

fn tri(i: u32, j: u32) -> u32 { return (i * (i + 1u)) / 2u + j; }

fn reconstruct_world(px: vec2f, dist: f32) -> vec3f {
  let uv  = (px + 0.5) / bp.resolution;
  let ndc = uv * 2.0 - 1.0;
  let rd  = normalize(bp.cam_fwd.xyz
                      + ndc.x * bp.aspect * bp.fov_factor * bp.cam_right.xyz
                      + ndc.y * bp.fov_factor * bp.cam_up.xyz);
  return bp.cam_pos.xyz + rd * dist;
}

// ============================================================
// FIT — one workgroup per 32x32 block, per-pixel own-block reconstruction
// ============================================================
@compute @workgroup_size(256)
fn fit(@builtin(workgroup_id) wid: vec3u,
       @builtin(local_invocation_index) li: u32,
       @builtin(subgroup_size) sg_size: u32,
       @builtin(subgroup_invocation_id) lane: u32) {

  let sz = vec2i(bp.resolution);
  let block_org = vec2i(i32(wid.x) * BLOCK, i32(wid.y) * BLOCK) - bp.block_offset;

  // --- FIX #1: thread 0 scans a 5x5 grid for a VALID (non-sky) reference pixel ---
  // Using the (possibly sky) centre pixel would give inv_scale ~ 1/1e6 and a ref_pos
  // ~1e6 units away, collapsing all positions -> degenerate fit -> bright glitches.
  if li == 0u {
    sh_valid = 0u;
    var rp = vec3f(0.0);
    var iscale = 1.0;
    var found = false;
    for (var j = 0; j < 5; j++) {
      for (var i = 0; i < 5; i++) {
        let sp = clamp(block_org + vec2i(i * 6 + 4, j * 6 + 4), vec2i(0), sz - 1);
        let w = textureLoad(in_nd, sp, 0).w;
        if w < SKY_Z {
          rp = reconstruct_world(vec2f(sp), w);
          iscale = 1.0 / max(w * bp.fov_factor * (f32(BLOCK) / bp.resolution.y), 1.0e-3);
          found = true;
          break;
        }
      }
      if found { break; }
    }
    sh_ref_pos = rp;
    sh_inv_scale = iscale;
  }
  workgroupBarrier();
  let ref_pos = sh_ref_pos;
  let inv_scale = sh_inv_scale;

  // --- Phase A: accumulate A = sum f fᵀ, b = sum f c over valid pixels ---
  var accA: array<f32, 28>;
  var accB: array<f32, 21>;
  for (var k = 0u; k < NA; k++) { accA[k] = 0.0; }
  for (var k = 0u; k < NB; k++) { accB[k] = 0.0; }
  var local_valid = 0u;

  for (var s = 0u; s < PPT; s++) {
    let pidx = li + s * WG;
    let lpx  = vec2i(i32(pidx % 32u), i32(pidx / 32u));
    let px   = block_org + lpx;
    if px.x < 0 || px.y < 0 || px.x >= sz.x || px.y >= sz.y { continue; }

    let nd = textureLoad(in_nd, px, 0);
    if nd.w >= SKY_Z { continue; }
    local_valid = 1u;

    let n = nd.xyz;
    let p = (reconstruct_world(vec2f(px), nd.w) - ref_pos) * inv_scale;
    let c = textureLoad(in_diff, px, 0).rgb;

    var f: array<f32, 7>;
    f[0] = 1.0; f[1] = n.x; f[2] = n.y; f[3] = n.z; f[4] = p.x; f[5] = p.y; f[6] = p.z;

    for (var i = 0u; i < NF; i++) {
      for (var jj = 0u; jj <= i; jj++) { accA[tri(i, jj)] += f[i] * f[jj]; }
      accB[i * 3u + 0u] += f[i] * c.r;
      accB[i * 3u + 1u] += f[i] * c.g;
      accB[i * 3u + 2u] += f[i] * c.b;
    }
  }

  // --- Reduce across workgroup ---
  let sg = min(li / sg_size, MAXSG - 1u);
  for (var k = 0u; k < NA; k++) {
    let v = subgroupAdd(accA[k]);
    if lane == 0u { sh_A[sg][k] = v; }
  }
  for (var k = 0u; k < NB; k++) {
    let v = subgroupAdd(accB[k]);
    if lane == 0u { sh_b[sg][k] = v; }
  }
  let any_valid = subgroupAdd(local_valid);
  if lane == 0u && any_valid > 0u { sh_valid = 1u; }
  workgroupBarrier();

  // --- Phase B: thread 0 sums, relative ridge, relative-guard Cholesky, solve ---
  if li == 0u {
    let n_sub = min((WG + sg_size - 1u) / sg_size, MAXSG);
    var A: array<f32, 28>;
    var b: array<f32, 21>;
    for (var k = 0u; k < NA; k++) { var acc = 0.0; for (var s2 = 0u; s2 < n_sub; s2++) { acc += sh_A[s2][k]; } A[k] = acc; }
    for (var k = 0u; k < NB; k++) { var acc = 0.0; for (var s2 = 0u; s2 < n_sub; s2++) { acc += sh_b[s2][k]; } b[k] = acc; }

    let n_pix = A[tri(0u, 0u)];   // pixel count (before ridge) for the fallback mean

    // FIX #3: scale-invariant ridge — regularize relative to each diagonal magnitude.
    for (var i = 0u; i < NF; i++) { A[tri(i, i)] += bp.ridge * A[tri(i, i)] + 1.0e-4; }

    var L: array<f32, 28>;
    var ok = sh_valid == 1u;
    for (var i = 0u; i < NF && ok; i++) {
      for (var jj = 0u; jj <= i; jj++) {
        var sum = A[tri(i, jj)];
        for (var k = 0u; k < jj; k++) { sum -= L[tri(i, k)] * L[tri(jj, k)]; }
        if i == jj {
          // FIX #7: RELATIVE near-singularity guard (catches ill-conditioned blocks
          // that an absolute 1e-8 threshold misses -> avoids huge gradients / glitches).
          if sum <= 1.0e-5 * A[tri(i, i)] { ok = false; break; }
          L[tri(i, i)] = sqrt(sum);
        } else {
          L[tri(i, jj)] = sum / L[tri(jj, jj)];
        }
      }
    }

    if ok {
      for (var ch = 0u; ch < 3u; ch++) {
        var y: array<f32, 7>;
        for (var i = 0u; i < NF; i++) {
          var sum = b[i * 3u + ch];
          for (var k = 0u; k < i; k++) { sum -= L[tri(i, k)] * y[k]; }
          y[i] = sum / L[tri(i, i)];
        }
        var a: array<f32, 7>;
        var ii = NF;
        loop {
          ii -= 1u;
          var sum = y[ii];
          for (var k = ii + 1u; k < NF; k++) { sum -= L[tri(k, ii)] * a[k]; }
          a[ii] = sum / L[tri(ii, ii)];
          if ii == 0u { break; }
        }
        for (var i = 0u; i < NF; i++) { sh_alpha[i * 3u + ch] = a[i]; }
      }
    } else {
      // Degenerate / ill-conditioned -> constant term = block mean, gradients 0
      let cnt = max(n_pix, 1.0);
      for (var ch = 0u; ch < 3u; ch++) {
        sh_alpha[0u * 3u + ch] = b[0u * 3u + ch] / cnt;
        for (var i = 1u; i < NF; i++) { sh_alpha[i * 3u + ch] = 0.0; }
      }
    }
  }
  workgroupBarrier();

  // --- Phase C: reconstruct each pixel in its own block ---
  for (var s = 0u; s < PPT; s++) {
    let pidx = li + s * WG;
    let lpx  = vec2i(i32(pidx % 32u), i32(pidx / 32u));
    let px   = block_org + lpx;
    if px.x < 0 || px.y < 0 || px.x >= sz.x || px.y >= sz.y { continue; }

    let nd = textureLoad(in_nd, px, 0);
    if nd.w >= SKY_Z {
      textureStore(out_diff, px, textureLoad(in_diff, px, 0));   // sky passthrough
      continue;
    }

    let n = nd.xyz;
    let p = (reconstruct_world(vec2f(px), nd.w) - ref_pos) * inv_scale;
    var f: array<f32, 7>;
    f[0] = 1.0; f[1] = n.x; f[2] = n.y; f[3] = n.z; f[4] = p.x; f[5] = p.y; f[6] = p.z;

    var col = vec3f(0.0);
    for (var i = 0u; i < NF; i++) {
      col.r += f[i] * sh_alpha[i * 3u + 0u];
      col.g += f[i] * sh_alpha[i * 3u + 1u];
      col.b += f[i] * sh_alpha[i * 3u + 2u];
    }
    // FIX #4: sanitize — clamp bounds Inf, the self-compare maps NaN -> 0.
    let safe = clamp(col, vec3f(0.0), vec3f(64.0));
    col = select(vec3f(0.0), safe, safe == safe);
    textureStore(out_diff, px, vec4f(col, 1.0));
  }
}

// Project a world position to screen UV with a given camera basis.
fn project_to_uv(world: vec3f, right: vec3f, up: vec3f, fwd: vec3f, pos: vec3f) -> vec3f {
  let local = world - pos;
  let z = dot(local, fwd);
  if z <= 0.0 { return vec3f(-1.0); }
  let x = dot(local, right);
  let y = dot(local, up);
  return vec3f((x / (z * bp.aspect * bp.fov_factor)) * 0.5 + 0.5,
               (y / (z * bp.fov_factor)) * 0.5 + 0.5, z);
}

// ============================================================
// ACCUMULATE — temporal EMA (reprojected) of the fit result
// ============================================================
@compute @workgroup_size(8, 8)
fn accumulate(@builtin(global_invocation_id) gid: vec3u) {
  let px = vec2i(gid.xy);
  let sz = vec2i(bp.resolution);
  if px.x >= sz.x || px.y >= sz.y { return; }

  let nd  = textureLoad(in_nd, px, 0);
  let cz  = nd.w;
  let cur = textureLoad(in_fit, px, 0).rgb;

  if cz >= SKY_Z {
    textureStore(hdr_out, px, vec4f(cur, 1.0));
    textureStore(hist_out, px, vec4f(cur, 0.0));
    return;
  }

  let world = reconstruct_world(vec2f(px), cz);
  let puv = project_to_uv(world, bp.prev_right.xyz, bp.prev_up.xyz, bp.prev_fwd.xyz, bp.prev_pos.xyz);

  var blended = cur;
  var hist_len = 0.0;
  if puv.x >= 0.0 && puv.x <= 1.0 && puv.y >= 0.0 && puv.y <= 1.0 {
    let ppx = clamp(vec2i(puv.xy * bp.resolution), vec2i(0), sz - 1);
    let h = textureLoad(in_hist, ppx, 0);

    let eff_max = mix(8.0, bp.max_history, clamp(bp.frames_still / 32.0, 0.0, 1.0));
    hist_len = min(max(h.a, 0.0) + 1.0, eff_max);

    // AABB clamp history to local neighbourhood — ONLY while accumulating (hist<32).
    // Once converged, trust history fully so the per-frame offset grid averages out.
    // (Re-clamping to the current frame's grid here would re-inject it -> never converges.)
    var hc = h.rgb;
    if hist_len < 16.0 {
      var mn = cur; var mx = cur;
      for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
          let sp = clamp(px + vec2i(dx, dy), vec2i(0), sz - 1);
          let c = textureLoad(in_fit, sp, 0).rgb;
          mn = min(mn, c); mx = max(mx, c);
        }
      }
      let ext = (mx - mn) * 0.5 + vec3f(0.002);
      hc = clamp(h.rgb, mn - ext, mx + ext);
    }

    let alpha = max(1.0 / (hist_len + 1.0), bp.min_alpha);
    blended = mix(hc, cur, alpha);
  }

  textureStore(hdr_out, px, vec4f(blended, 1.0));
  textureStore(hist_out, px, vec4f(blended, hist_len));
}
