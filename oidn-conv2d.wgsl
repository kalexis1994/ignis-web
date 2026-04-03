// OIDN Conv2D 3x3 — im2col matmul with 4×4 register tiling
// Each thread computes 4×4 = 16 output elements → 16x better arithmetic intensity
// Output[M,N] = Weight[M,K] × Im2col[K,N], M=c_out, K=c_in*9, N=H*W

enable f16;

struct ConvParams {
  width: u32, height: u32, c_in: u32, c_out: u32,
  weight_off: u32, bias_off: u32, apply_relu: u32, split_ch: u32,
};

@group(0) @binding(0) var<uniform> params: ConvParams;
@group(0) @binding(1) var<storage, read> weights: array<f16>;
@group(0) @binding(2) var<storage, read> buf_a: array<f16>;
@group(0) @binding(3) var<storage, read_write> buf_out: array<f16>;
@group(0) @binding(4) var<storage, read> buf_b: array<f16>;

const BM: u32 = 32u;  // tile M (output channels per workgroup)
const BN: u32 = 32u;  // tile N (spatial positions per workgroup)
const BK: u32 = 32u;  // tile K (inner dimension per iteration)
const TM: u32 = 4u;   // elements per thread in M
const TN: u32 = 4u;   // elements per thread in N
// Workgroup: (BN/TN) × (BM/TM) = 8 × 8 = 64 threads

var<workgroup> w_sm: array<f16, 1024>;  // BM * BK = 32*32
var<workgroup> i_sm: array<f16, 1024>;  // BK * BN = 32*32
// Total shared memory: 4096 bytes

@compute @workgroup_size(8, 8)
fn conv2d_3x3(
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(workgroup_id) wid: vec3u
) {
  let W = params.width;
  let H = params.height;
  let HW = H * W;
  let c_in = params.c_in;
  let c_out = params.c_out;
  let K = c_in * 9u;
  let split = select(c_in, params.split_ch, params.split_ch > 0u);

  let m_base = wid.x * BM;  // output channel base
  let n_base = wid.y * BN;  // spatial position base

  // Thread position: lid.x = N thread (0..7), lid.y = M thread (0..7)
  // Each thread computes TM × TN = 4×4 = 16 output elements
  let tm_base = lid.y * TM;  // M offset within tile (0,4,8,...,28)
  let tn_base = lid.x * TN;  // N offset within tile (0,4,8,...,28)

  // 16 accumulators (4M × 4N)
  var c00:f32=0.;var c01:f32=0.;var c02:f32=0.;var c03:f32=0.;
  var c10:f32=0.;var c11:f32=0.;var c12:f32=0.;var c13:f32=0.;
  var c20:f32=0.;var c21:f32=0.;var c22:f32=0.;var c23:f32=0.;
  var c30:f32=0.;var c31:f32=0.;var c32:f32=0.;var c33:f32=0.;

  let flat = lid.y * 8u + lid.x;  // 0..63

  for (var k_base = 0u; k_base < K; k_base += BK) {
    // --- Load weight tile w_sm[BM][BK] = 512 values, 64 threads → 8 per thread ---
    for (var i = flat; i < 1024u; i += 64u) {
      let row = i / BK;  // M offset (0..31)
      let col = i % BK;  // K offset (0..15)
      let gm = m_base + row;
      let gk = k_base + col;
      w_sm[i] = select(0.0h, weights[params.weight_off + gm * K + gk], gm < c_out && gk < K);
    }

    // --- Load input tile i_sm[BK][BN] = 512 values via implicit im2col ---
    for (var i = flat; i < 1024u; i += 64u) {
      let row = i / BN;  // K offset (0..15)
      let col = i % BN;  // N offset (0..31)
      let gk = k_base + row;
      let gn = n_base + col;
      var val: f16 = 0.0h;
      if gk < K && gn < HW {
        let gy = gn / W;
        let gx = gn - gy * W;
        let ic = gk / 9u;
        let rem = gk - ic * 9u;
        let ky = rem / 3u;
        let kx = rem - ky * 3u;
        let sy = clamp(i32(gy) + i32(ky) - 1, 0, i32(H) - 1);
        let sx = clamp(i32(gx) + i32(kx) - 1, 0, i32(W) - 1);
        let src = u32(sy) * W + u32(sx);
        if ic < split { val = buf_a[ic * HW + src]; }
        else { val = buf_b[(ic - split) * HW + src]; }
      }
      i_sm[i] = val;
    }

    workgroupBarrier();

    // --- 4×4 register-tiled matmul from shared memory ---
    for (var k = 0u; k < BK; k++) {
      // Load 4 weight values (M dimension)
      let w0 = f32(w_sm[(tm_base    ) * BK + k]);
      let w1 = f32(w_sm[(tm_base + 1u) * BK + k]);
      let w2 = f32(w_sm[(tm_base + 2u) * BK + k]);
      let w3 = f32(w_sm[(tm_base + 3u) * BK + k]);
      // Load 4 input values (N dimension)
      let i0 = f32(i_sm[k * BN + tn_base    ]);
      let i1 = f32(i_sm[k * BN + tn_base + 1u]);
      let i2 = f32(i_sm[k * BN + tn_base + 2u]);
      let i3 = f32(i_sm[k * BN + tn_base + 3u]);
      // 4×4 outer product (16 FMAs)
      c00+=w0*i0; c01+=w0*i1; c02+=w0*i2; c03+=w0*i3;
      c10+=w1*i0; c11+=w1*i1; c12+=w1*i2; c13+=w1*i3;
      c20+=w2*i0; c21+=w2*i1; c22+=w2*i2; c23+=w2*i3;
      c30+=w3*i0; c31+=w3*i1; c32+=w3*i2; c33+=w3*i3;
    }

    workgroupBarrier();
  }

  // --- Write 4×4 output block with bias + ReLU ---
  let relu = params.apply_relu > 0u;
  for (var dm = 0u; dm < TM; dm++) {
    let gm = m_base + tm_base + dm;
    if gm >= c_out { continue; }
    let bias = f32(weights[params.bias_off + gm]);
    for (var dn = 0u; dn < TN; dn++) {
      let gn = n_base + tn_base + dn;
      if gn >= HW { continue; }
      var val: f32;
      switch dm * 4u + dn {
        case 0u: { val = c00; } case 1u: { val = c01; } case 2u: { val = c02; } case 3u: { val = c03; }
        case 4u: { val = c10; } case 5u: { val = c11; } case 6u: { val = c12; } case 7u: { val = c13; }
        case 8u: { val = c20; } case 9u: { val = c21; } case 10u: { val = c22; } case 11u: { val = c23; }
        case 12u: { val = c30; } case 13u: { val = c31; } case 14u: { val = c32; } default: { val = c33; }
      }
      val += bias;
      if relu { val = max(val, 0.0); }
      buf_out[gm * HW + gn] = f16(val);
    }
  }
}
