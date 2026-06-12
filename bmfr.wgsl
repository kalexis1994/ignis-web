// Fast Local Linear Regression denoiser (FLNR / guided-filter family)
// =========================================================================
// Reconstructs DEMODULATED DIFFUSE irradiance by fitting a local linear model
//     c ~= f . alpha,   f = [1, n.x, n.y, n.z, rp.x, rp.y, rp.z]   (7 features)
// where rp = camera-relative world position. The model is solved at 1/8 res
// (cheap) and bilinearly upsampled, with the moments BLURRED at coarse res so
// neighbouring blocks share information -> smooth coefficients -> NO grid.
//
//   moments    — 1 workgroup / 8x8 block: reduce per-pixel outer products
//                (A = sum f fᵀ, b = sum f c) into a coarse moment buffer.
//   solve      — 1 thread / coarse cell: gaussian-blur the 5x5 neighbourhood of
//                moments (overlap -> smooth), relative ridge + Cholesky -> coef.
//   apply      — 1 thread / pixel: bilinear-interpolate the coarse coefficients
//                and evaluate f . alpha at the pixel.
//   accumulate — temporal EMA (reprojected).
//
// All math f32; I/O f16. No subgroups (each thread independent).

struct BmfrParams {
  resolution   : vec2f,
  block_offset : vec2i,   // unused
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
  ridge        : f32,     // relative ridge factor
  min_alpha    : f32,
  frames_still : f32,
  max_history  : f32,
  depth_reject : f32,
  _pad         : f32,
};

@group(0) @binding(0) var<uniform> bp: BmfrParams;
@group(0) @binding(1) var in_diff : texture_2d<f32>;                        // noisy demodulated diffuse
@group(0) @binding(2) var in_nd   : texture_2d<f32>;                        // normal(xyz)+hit-distance(w)
@group(0) @binding(3) var out_diff: texture_storage_2d<rgba16float, write>; // apply: reconstructed diffuse
@group(0) @binding(4) var in_fit  : texture_2d<f32>;                        // accumulate: this frame's fit
@group(0) @binding(5) var in_hist : texture_2d<f32>;                        // accumulate: previous history
@group(0) @binding(6) var hdr_out : texture_storage_2d<rgba16float, write>; // accumulate: final
@group(0) @binding(7) var hist_out: texture_storage_2d<rgba16float, write>; // accumulate: new history
@group(0) @binding(8) var<storage, read_write> momBuf : array<f32>;         // coarse: 49 / cell (28 A + 21 b)
@group(0) @binding(9) var<storage, read_write> coefs   : array<f32>;        // coarse: 22 / cell (21 alpha + 1 wsum)
// --- Specular denoiser bindings ---
@group(0) @binding(10) var in_spec       : texture_2d<f32>;                        // noisy specular (rgb + normHitDist .a)
@group(0) @binding(11) var in_rough      : texture_2d<f32>;                        // albedo (.a = roughness)
@group(0) @binding(12) var in_spec_hist  : texture_2d<f32>;                        // previous specular history
@group(0) @binding(13) var spec_out      : texture_storage_2d<rgba16float, write>; // denoised specular (composite reads)
@group(0) @binding(14) var spec_hist_out : texture_storage_2d<rgba16float, write>; // new specular history

const B       : i32 = 8;        // block size
const NF      : u32 = 7u;
const NA      : u32 = 28u;      // lower-tri of 7x7
const NB      : u32 = 21u;      // 7x3
const MACC    : u32 = 49u;      // accumulated moments per cell (28 A + 21 b)
const MSTRIDE : u32 = 52u;      // + 3 cell reference position
const CSTRIDE : u32 = 25u;      // 18 a + 3 intercept + 1 wsum + 3 cell ref
const SKY_Z   : f32 = 1.0e5;

fn tri(i: u32, j: u32) -> u32 { return (i * (i + 1u)) / 2u + j; }
fn gw5(d: i32) -> f32 { let a = abs(d); if a == 0 { return 6.0; } if a == 1 { return 4.0; } return 1.0; }
fn nbx() -> u32 { return (u32(bp.resolution.x) + 7u) / 8u; }
fn nby() -> u32 { return (u32(bp.resolution.y) + 7u) / 8u; }

fn reconstruct_world(px: vec2f, dist: f32) -> vec3f {
  let uv  = (px + 0.5) / bp.resolution;
  let ndc = uv * 2.0 - 1.0;
  let rd  = normalize(bp.cam_fwd.xyz
                      + ndc.x * bp.aspect * bp.fov_factor * bp.cam_right.xyz
                      + ndc.y * bp.fov_factor * bp.cam_up.xyz);
  return bp.cam_pos.xyz + rd * dist;
}

fn luma(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }

// Roughness->blur curve (ported from NRD/ReBLUR get_spec_magic_curve): ~0 for mirror, ~1 for rough.
fn get_spec_magic_curve(roughness: f32) -> f32 {
  let f = 1.0 - exp2(-200.0 * roughness * roughness);
  return f * pow(clamp(roughness, 0.0, 1.0), 0.25);
}

// Anti-firefly read of the noisy specular (cap luminance to ~3x the brightest cross-neighbour).
fn read_spec_clamped(px: vec2i, sz: vec2i) -> vec3f {
  var c = textureLoad(in_spec, px, 0).rgb;
  let lc = luma(c);
  let lmax = max(
    max(luma(textureLoad(in_spec, clamp(px + vec2i(-1, 0), vec2i(0), sz - 1), 0).rgb),
        luma(textureLoad(in_spec, clamp(px + vec2i( 1, 0), vec2i(0), sz - 1), 0).rgb)),
    max(luma(textureLoad(in_spec, clamp(px + vec2i(0, -1), vec2i(0), sz - 1), 0).rgb),
        luma(textureLoad(in_spec, clamp(px + vec2i(0,  1), vec2i(0), sz - 1), 0).rgb)));
  let cap = lmax * 3.0 + 0.02;
  if lc > cap { c *= cap / lc; }
  return c;
}

// Read noisy diffuse with an anti-firefly clamp: cap a pixel's luminance to ~2x the
// brightest cross-neighbour, killing isolated bright outliers before they skew the
// local regression. Preserves hue; leaves genuinely bright (clustered) regions intact.
fn read_diff_clamped(px: vec2i, sz: vec2i) -> vec3f {
  var c = textureLoad(in_diff, px, 0).rgb;
  let lc = luma(c);
  let lmax = max(
    max(luma(textureLoad(in_diff, clamp(px + vec2i(-1, 0), vec2i(0), sz - 1), 0).rgb),
        luma(textureLoad(in_diff, clamp(px + vec2i( 1, 0), vec2i(0), sz - 1), 0).rgb)),
    max(luma(textureLoad(in_diff, clamp(px + vec2i(0, -1), vec2i(0), sz - 1), 0).rgb),
        luma(textureLoad(in_diff, clamp(px + vec2i(0,  1), vec2i(0), sz - 1), 0).rgb)));
  let cap = lmax * 2.0 + 0.02;
  if lc > cap { c *= cap / lc; }
  return c;
}

// ============================================================
// KERNEL 0 — pre_accumulate: reprojected EMA of the RAW noisy diffuse.
// The BMFR-paper structure is accumulate -> fit -> accumulate; fitting the
// raw 1-spp signal makes the per-cell model coefficients flicker frame to
// frame (coarse, cell-sized noise while the camera moves). A short
// moving-average -> 80/20 EMA (paper's rule) cuts input variance ~5x.
// History is always AABB-clamped to the current 3x3 envelope: ghosts here
// would poison the fit itself.
// ============================================================
@compute @workgroup_size(8, 8)
fn pre_accumulate(@builtin(global_invocation_id) gid: vec3u) {
  let px = vec2i(gid.xy);
  let sz = vec2i(bp.resolution);
  if px.x >= sz.x || px.y >= sz.y { return; }

  let nd = textureLoad(in_nd, px, 0);
  let cz = nd.w;
  // Firefly-clamped read: with the 0.2 alpha floor a raw GI firefly would inject
  // 20% of its energy into the accumulation instantly (even when still) and then
  // decay slowly under the loose ghost gate -> wandering bright blobs.
  let cur = read_diff_clamped(px, sz);
  if cz >= SKY_Z {
    textureStore(hist_out, px, vec4f(cur, 0.0));
    return;
  }

  let world = reconstruct_world(vec2f(px), cz);
  let puv = project_to_uv(world, bp.prev_right.xyz, bp.prev_up.xyz, bp.prev_fwd.xyz, bp.prev_pos.xyz);

  var blended = cur;
  var cnt = 1.0;
  if puv.x >= 0.0 && puv.x <= 1.0 && puv.y >= 0.0 && puv.y <= 1.0 {
    let ppx = clamp(vec2i(puv.xy * bp.resolution), vec2i(0), sz - 1);
    let h = textureLoad(in_hist, ppx, 0);

    // LOOSE ghost gate only. A tight 1-spp colour clamp zeroes the history
    // wherever the current 3x3 happens to be all-dark (sparse GI samples) —
    // exactly the regions accumulation must help. Bright-residue ghosts are
    // the visible kind, so cap history at a multiple of the local max; the
    // short window (<=16) bounds any surviving ghost's lifetime.
    var mx = cur;
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let sp = clamp(px + vec2i(dx, dy), vec2i(0), sz - 1);
        mx = max(mx, textureLoad(in_diff, sp, 0).rgb);
      }
    }
    let hc = clamp(h.rgb, vec3f(0.0), mx * 4.0 + vec3f(0.25));

    cnt = min(max(h.a, 0.0) + 1.0, 16.0);
    let alpha = max(1.0 / cnt, 0.2);     // moving average for the first 5, then 80/20 EMA
    blended = mix(hc, cur, alpha);
  }
  textureStore(hist_out, px, vec4f(blended, cnt));
}

// ============================================================
// KERNEL 1 — moments: one 8x8 workgroup per coarse cell
// Positions are accumulated RELATIVE TO A PER-CELL REFERENCE (first valid
// pixel's world pos). Raw camera-relative positions (magnitude ~ scene size)
// made the later centered-covariance step a catastrophic f32 cancellation
// (E[p^2] - mu^2 with mu >> sigma) -> garbage gradients -> bright patches.
// With p ~ O(cell size) the solve is well-conditioned.
// ============================================================
var<workgroup> sh_part : array<f32, 3136>;   // 64 threads * 49 moments
var<workgroup> sh_ref  : array<vec4f, 64>;   // per-thread (world, valid)
var<workgroup> sh_cellref : vec3f;

@compute @workgroup_size(8, 8)
fn moments(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
  let sz = vec2i(bp.resolution);
  let px = vec2i(i32(wid.x) * B + i32(li % 8u), i32(wid.y) * B + i32(li / 8u));

  let inb = px.x < sz.x && px.y < sz.y;
  var nd = vec4f(0.0, 0.0, 0.0, 1.0e9);
  if inb { nd = textureLoad(in_nd, px, 0); }
  let valid = inb && nd.w < SKY_Z;
  var world = vec3f(0.0);
  if valid { world = reconstruct_world(vec2f(px), nd.w); }

  sh_ref[li] = vec4f(world, select(0.0, 1.0, valid));
  workgroupBarrier();
  if li == 0u {
    var r = vec3f(0.0);
    for (var t = 0u; t < 64u; t++) {
      if sh_ref[t].w > 0.5 { r = sh_ref[t].xyz; break; }
    }
    sh_cellref = r;
  }
  workgroupBarrier();
  let cellref = sh_cellref;

  var m: array<f32, 49>;
  for (var k = 0u; k < MACC; k++) { m[k] = 0.0; }

  if valid {
    let n = nd.xyz;
    let p = world - cellref;
    let c = read_diff_clamped(px, sz);
    var f: array<f32, 7>;
    f[0] = 1.0; f[1] = n.x; f[2] = n.y; f[3] = n.z; f[4] = p.x; f[5] = p.y; f[6] = p.z;
    for (var i = 0u; i < NF; i++) {
      for (var j = 0u; j <= i; j++) { m[tri(i, j)] += f[i] * f[j]; }
      m[28u + i * 3u + 0u] += f[i] * c.r;
      m[28u + i * 3u + 1u] += f[i] * c.g;
      m[28u + i * 3u + 2u] += f[i] * c.b;
    }
  }

  for (var k = 0u; k < MACC; k++) { sh_part[li * MACC + k] = m[k]; }
  workgroupBarrier();

  if li == 0u {
    let base = (wid.y * nbx() + wid.x) * MSTRIDE;
    for (var k = 0u; k < MACC; k++) {
      var acc = 0.0;
      for (var t = 0u; t < 64u; t++) { acc += sh_part[t * MACC + k]; }
      momBuf[base + k] = acc;
    }
    momBuf[base + 49u] = cellref.x;
    momBuf[base + 50u] = cellref.y;
    momBuf[base + 51u] = cellref.z;
  }
}

// ============================================================
// KERNEL 2 — solve: blur 5x5 neighbour moments, then Cholesky
// ============================================================
@compute @workgroup_size(8, 8)
fn solve(@builtin(global_invocation_id) gid: vec3u) {
  let mx = nbx();
  let my = nby();
  if gid.x >= mx || gid.y >= my { return; }

  // Gaussian-blur the 5x5 neighbourhood of coarse moments (overlap -> smooth coefs).
  // Each cell's moments use ITS OWN position reference, so translate every
  // neighbour's moments into the CENTER cell's frame before summing:
  //   p_c = p_n + t,  t = ref_n - ref_c   (exact moment translation, no cancellation)
  let cbase = (gid.y * mx + gid.x) * MSTRIDE;
  let ref_c = vec3f(momBuf[cbase + 49u], momBuf[cbase + 50u], momBuf[cbase + 51u]);

  // Bilateral moment blur: weight neighbour cells by mean-luminance similarity.
  // A lighting edge inside one surface (light pool / shadow) has NO feature
  // difference, so the linear model alone smears it across the whole window
  // ("bloom" around lit areas). Cell means (64-px averages) are stable enough
  // at 1 spp to gate the mixing: similar lighting -> full sharing, different
  // lighting -> cells keep their own statistics and the edge stays sharp.
  let S_c = momBuf[cbase + 0u];
  let valid_c = S_c > 1.0e-3;
  let lum_c = luma(vec3f(momBuf[cbase + 28u], momBuf[cbase + 29u], momBuf[cbase + 30u])) / max(S_c, 1.0e-3);

  var A: array<f32, 28>;
  var b: array<f32, 21>;
  for (var k = 0u; k < NA; k++) { A[k] = 0.0; }
  for (var k = 0u; k < NB; k++) { b[k] = 0.0; }

  for (var dj = -2; dj <= 2; dj++) {
    for (var di = -2; di <= 2; di++) {
      let cx = u32(clamp(i32(gid.x) + di, 0, i32(mx) - 1));
      let cy = u32(clamp(i32(gid.y) + dj, 0, i32(my) - 1));
      let base = (cy * mx + cx) * MSTRIDE;

      let S_n = momBuf[base + 0u];
      if S_n <= 1.0e-3 { continue; }     // empty/sky cell: nothing to contribute

      var w = gw5(di) * gw5(dj);
      if valid_c {
        let lum_n = luma(vec3f(momBuf[base + 28u], momBuf[base + 29u], momBuf[base + 30u])) / max(S_n, 1.0e-3);
        let rel = abs(lum_c - lum_n) / (max(lum_c, lum_n) + 1.0e-3);
        w *= exp(-rel * rel * 8.0);      // ~1 for similar lighting, ~0 across light/shadow
      }

      var mA: array<f32, 28>;
      var mb: array<f32, 21>;
      for (var k = 0u; k < NA; k++) { mA[k] = momBuf[base + k]; }
      for (var k = 0u; k < NB; k++) { mb[k] = momBuf[base + 28u + k]; }

      let ref_n = vec3f(momBuf[base + 49u], momBuf[base + 50u], momBuf[base + 51u]);
      let t = ref_n - ref_c;
      if dot(t, t) > 0.0 {
        let S = mA[0];                                      // sum of weights (tri(0,0))
        var sp: vec3f;                                       // original Sp_a (save before overwrite)
        sp.x = mA[tri(4u, 0u)]; sp.y = mA[tri(5u, 0u)]; sp.z = mA[tri(6u, 0u)];
        var tv: array<f32, 3>; tv[0] = t.x; tv[1] = t.y; tv[2] = t.z;
        var spv: array<f32, 3>; spv[0] = sp.x; spv[1] = sp.y; spv[2] = sp.z;
        for (var a = 0u; a < 3u; a++) {
          mA[tri(4u + a, 0u)] += tv[a] * S;                 // Sp
          for (var i = 1u; i <= 3u; i++) {                  // Snp (normals don't shift)
            mA[tri(4u + a, i)] += tv[a] * mA[tri(i, 0u)];
          }
          for (var bb = 0u; bb <= a; bb++) {                // Spp lower-tri
            mA[tri(4u + a, 4u + bb)] += tv[a] * spv[bb] + tv[bb] * spv[a] + tv[a] * tv[bb] * S;
          }
          for (var ch = 0u; ch < 3u; ch++) {                // b_p
            mb[(4u + a) * 3u + ch] += tv[a] * mb[ch];
          }
        }
      }

      for (var k = 0u; k < NA; k++) { A[k] += w * mA[k]; }
      for (var k = 0u; k < NB; k++) { b[k] += w * mb[k]; }
    }
  }

  let n = A[tri(0u, 0u)];                       // sum of weights
  let inv_n = 1.0 / max(n, 1.0e-6);

  // Means of the 6 guides (features 1..6) and of the colour.
  var muf: array<f32, 6>;
  for (var i = 0u; i < 6u; i++) { muf[i] = A[tri(i + 1u, 0u)] * inv_n; }
  let muc = vec3f(b[0], b[1], b[2]) * inv_n;

  // Centred 6x6 covariance C and 6x3 cross-cov D (guided-filter form). Centring
  // makes the system well-conditioned regardless of position magnitude, killing
  // the gradient-extrapolation lighting glitches of the raw normal-equations form.
  var C: array<f32, 21>;
  for (var i = 0u; i < 6u; i++) {
    for (var j = 0u; j <= i; j++) { C[tri(i, j)] = A[tri(i + 1u, j + 1u)] * inv_n - muf[i] * muf[j]; }
  }
  var D: array<f32, 18>;
  for (var i = 0u; i < 6u; i++) {
    for (var ch = 0u; ch < 3u; ch++) { D[i * 3u + ch] = b[(i + 1u) * 3u + ch] * inv_n - muf[i] * muc[ch]; }
  }
  for (var i = 0u; i < 6u; i++) { C[tri(i, i)] += bp.ridge * C[tri(i, i)] + 1.0e-7; }

  // Cholesky C = LLᵀ (6x6); solve C a = D -> a (6x3); intercept = muc - aᵀ muf.
  var L: array<f32, 21>;
  var a: array<f32, 18>;
  var bint = muc;
  var ok = n > 1.0e-6;
  for (var i = 0u; i < 6u && ok; i++) {
    for (var j = 0u; j <= i; j++) {
      var sum = C[tri(i, j)];
      for (var k = 0u; k < j; k++) { sum -= L[tri(i, k)] * L[tri(j, k)]; }
      if i == j {
        if sum <= 1.0e-5 * C[tri(i, i)] { ok = false; break; }
        L[tri(i, i)] = sqrt(sum);
      } else {
        L[tri(i, j)] = sum / L[tri(j, j)];
      }
    }
  }

  if ok {
    for (var ch = 0u; ch < 3u; ch++) {
      var y: array<f32, 6>;
      for (var i = 0u; i < 6u; i++) {
        var sum = D[i * 3u + ch];
        for (var k = 0u; k < i; k++) { sum -= L[tri(i, k)] * y[k]; }
        y[i] = sum / L[tri(i, i)];
      }
      var ii = 6u;
      loop {
        ii -= 1u;
        var sum = y[ii];
        for (var k = ii + 1u; k < 6u; k++) { sum -= L[tri(k, ii)] * a[k * 3u + ch]; }
        a[ii * 3u + ch] = sum / L[tri(ii, ii)];
        if ii == 0u { break; }
      }
    }
    for (var i = 0u; i < 6u; i++) {
      bint.x -= a[i * 3u + 0u] * muf[i];
      bint.y -= a[i * 3u + 1u] * muf[i];
      bint.z -= a[i * 3u + 2u] * muf[i];
    }
  } else {
    for (var k = 0u; k < 18u; k++) { a[k] = 0.0; }
    bint = muc;
  }

  let obase = (gid.y * mx + gid.x) * CSTRIDE;
  for (var k = 0u; k < 18u; k++) { coefs[obase + k] = a[k]; }
  coefs[obase + 18u] = bint.x;
  coefs[obase + 19u] = bint.y;
  coefs[obase + 20u] = bint.z;
  coefs[obase + 21u] = n;        // validity weight
  coefs[obase + 22u] = ref_c.x;  // this cell's position reference (eval must match)
  coefs[obase + 23u] = ref_c.y;
  coefs[obase + 24u] = ref_c.z;
}

// Evaluate one coarse cell's model at (normal, world). Positions are taken
// relative to THAT cell's stored reference — must match how solve built it.
fn eval_cell(cx: u32, cy: u32, n: vec3f, world: vec3f) -> vec4f {
  let base = (cy * nbx() + cx) * CSTRIDE;
  let cref = vec3f(coefs[base + 22u], coefs[base + 23u], coefs[base + 24u]);
  let p = world - cref;
  var g: array<f32, 6>;
  g[0] = n.x; g[1] = n.y; g[2] = n.z; g[3] = p.x; g[4] = p.y; g[5] = p.z;
  // I = a . guides + intercept
  var c = vec3f(coefs[base + 18u], coefs[base + 19u], coefs[base + 20u]);
  for (var i = 0u; i < 6u; i++) {
    c.r += g[i] * coefs[base + i * 3u + 0u];
    c.g += g[i] * coefs[base + i * 3u + 1u];
    c.b += g[i] * coefs[base + i * 3u + 2u];
  }
  return vec4f(c, coefs[base + 21u]);
}

// ============================================================
// KERNEL 3 — apply: bilinear-interpolate coarse coefficients, evaluate
// ============================================================
@compute @workgroup_size(8, 8)
fn apply(@builtin(global_invocation_id) gid: vec3u) {
  let px = vec2i(gid.xy);
  let sz = vec2i(bp.resolution);
  if px.x >= sz.x || px.y >= sz.y { return; }

  let nd = textureLoad(in_nd, px, 0);
  if nd.w >= SKY_Z {
    textureStore(out_diff, px, textureLoad(in_diff, px, 0));   // sky passthrough
    return;
  }
  let nrm = nd.xyz;
  let world = reconstruct_world(vec2f(px), nd.w);

  // Bilinear over the coarse grid (cell centres at 8*b + 4).
  let mxn = nbx();
  let myn = nby();
  let gx = (f32(px.x) - 4.0) / 8.0;
  let gy = (f32(px.y) - 4.0) / 8.0;
  let x0 = i32(floor(gx)); let y0 = i32(floor(gy));
  let fx = gx - f32(x0);   let fy = gy - f32(y0);
  let cx0 = u32(clamp(x0,     0, i32(mxn) - 1));
  let cx1 = u32(clamp(x0 + 1, 0, i32(mxn) - 1));
  let cy0 = u32(clamp(y0,     0, i32(myn) - 1));
  let cy1 = u32(clamp(y0 + 1, 0, i32(myn) - 1));

  let v00 = eval_cell(cx0, cy0, nrm, world);
  let v10 = eval_cell(cx1, cy0, nrm, world);
  let v01 = eval_cell(cx0, cy1, nrm, world);
  let v11 = eval_cell(cx1, cy1, nrm, world);
  // Validity-weighted bilinear (sky/empty cells excluded).
  let w00 = (1.0 - fx) * (1.0 - fy) * step(1.0e-6, v00.a);
  let w10 = fx * (1.0 - fy) * step(1.0e-6, v10.a);
  let w01 = (1.0 - fx) * fy * step(1.0e-6, v01.a);
  let w11 = fx * fy * step(1.0e-6, v11.a);
  var col = (v00.rgb * w00 + v10.rgb * w10 + v01.rgb * w01 + v11.rgb * w11)
            / max(w00 + w10 + w01 + w11, 1.0e-4);

  // Anti-bleed clamp: a model fitted/blended across a lighting edge (cell straddling
  // light/shadow, or bilinear pulling a lit cell's model into shadow pixels) can emit
  // values the local pixels never contained -> "bloom". Bound the result to the 3x3
  // range of the noisy input (wide under noise, so it rarely bites elsewhere).
  let c0 = textureLoad(in_diff, px, 0).rgb;   // centre pixel is non-sky here
  var bmn = c0; var bmx = c0;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      if dx == 0 && dy == 0 { continue; }
      let sp = clamp(px + vec2i(dx, dy), vec2i(0), sz - 1);
      if textureLoad(in_nd, sp, 0).w < SKY_Z {
        let c = textureLoad(in_diff, sp, 0).rgb;
        bmn = min(bmn, c); bmx = max(bmx, c);
      }
    }
  }
  let bext = (bmx - bmn) * 0.5 + vec3f(0.01);
  col = clamp(col, max(bmn - bext, vec3f(0.0)), bmx + bext);

  let safe = clamp(col, vec3f(0.0), vec3f(64.0));
  col = select(vec3f(0.0), safe, safe == safe);
  textureStore(out_diff, px, vec4f(col, 1.0));
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
// KERNEL 4 — accumulate: temporal EMA (reprojected)
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

    var hc = h.rgb;
    if hist_len < 16.0 {
      var mn = cur; var mx = cur;
      for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
          let sp = clamp(px + vec2i(dx, dy), vec2i(0), sz - 1);
          let cc = textureLoad(in_fit, sp, 0).rgb;
          mn = min(mn, cc); mx = max(mx, cc);
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

// ============================================================
// SPEC_TEMPORAL — robust specular temporal accumulation (NRD-style)
//   firefly clamp + surface/virtual-motion reprojection (roughness-blended) +
//   roughness-scaled history + disocclusion + neighbourhood clamp.
// ============================================================
@compute @workgroup_size(8, 8)
fn spec_temporal(@builtin(global_invocation_id) gid: vec3u) {
  let px = vec2i(gid.xy);
  let sz = vec2i(bp.resolution);
  if px.x >= sz.x || px.y >= sz.y { return; }

  let nd = textureLoad(in_nd, px, 0);
  let cz = nd.w;
  let s_in = textureLoad(in_spec, px, 0);
  if cz >= SKY_Z {
    textureStore(spec_out, px, vec4f(s_in.rgb, 0.0));
    textureStore(spec_hist_out, px, vec4f(s_in.rgb, 0.0));
    return;
  }

  let cur = read_spec_clamped(px, sz);                       // firefly-clamped specular
  let n = nd.xyz;
  let rough = textureLoad(in_rough, px, 0).a;
  let world = reconstruct_world(vec2f(px), cz);
  let V = normalize(bp.cam_pos.xyz - world);
  let hitDist = exp2(clamp(s_in.a, 0.0, 1.0) * 8.0) - 1.0;   // decode normalized hit distance

  // Surface + virtual-motion reprojection, blended by roughness (glossy -> virtual).
  // NRD virtual image point: BEHIND the surface along the view ray (X - V*hitDist),
  // matching reblur.wgsl get_x_virtual's flat-mirror case — NOT the reflected hit point.
  let surf = project_to_uv(world, bp.prev_right.xyz, bp.prev_up.xyz, bp.prev_fwd.xyz, bp.prev_pos.xyz);
  let virt = project_to_uv(world - V * hitDist, bp.prev_right.xyz, bp.prev_up.xyz, bp.prev_fwd.xyz, bp.prev_pos.xyz);
  let smc = get_spec_magic_curve(rough);
  // No virtual motion when the hit distance is the miss/far sentinel (encoded a ~ 1)
  // or the virtual point projects behind the previous camera.
  var virtualAmount = clamp(1.0 - smc, 0.0, 1.0);
  if s_in.a >= 0.99 || virt.z <= 0.0 { virtualAmount = 0.0; }
  let ruv = mix(surf.xy, virt.xy, virtualAmount);

  var blended = cur;
  var hist_len = 0.0;
  if surf.z > 0.0 && ruv.x >= 0.0 && ruv.x <= 1.0 && ruv.y >= 0.0 && ruv.y <= 1.0 {
    let ppx = clamp(vec2i(ruv * bp.resolution), vec2i(0), sz - 1);
    let h = textureLoad(in_spec_hist, ppx, 0);
    // Shorter, roughness-scaled history (reflections change faster than diffuse).
    let smax = max(2.0, mix(6.0, bp.max_history * 0.5, clamp(bp.frames_still / 32.0, 0.0, 1.0)) * (0.3 + 0.7 * smc));
    hist_len = min(max(h.a, 0.0) + 1.0, smax);
    // Neighbourhood clamp (specular is noisy -> always on) to reject ghosts.
    var mn = cur; var mx = cur;
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let sp = clamp(px + vec2i(dx, dy), vec2i(0), sz - 1);
        let cc = textureLoad(in_spec, sp, 0).rgb;
        mn = min(mn, cc); mx = max(mx, cc);
      }
    }
    let ext = (mx - mn) + vec3f(0.005);
    let hc = clamp(h.rgb, mn - ext, mx + ext);
    let alpha = max(1.0 / (hist_len + 1.0), bp.min_alpha);
    blended = mix(hc, cur, alpha);
  }

  textureStore(spec_out, px, vec4f(blended, s_in.a));        // .a keeps normalized hit distance
  textureStore(spec_hist_out, px, vec4f(blended, hist_len));
}
