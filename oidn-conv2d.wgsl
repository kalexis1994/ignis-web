// OIDN Conv2D 3x3 — Shared-memory tiled, 8 output channels per workgroup
// 16x16 threads, z = ceil(c_out/8). Loads input tile ONCE, applies to 8 output channels.

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

const TILE: u32 = 16u;
const PAD: u32 = 18u;
const CH_TILE: u32 = 16u;
const OC_BATCH: u32 = 8u;

var<workgroup> sm: array<f16, 5184>; // PAD*PAD*CH_TILE = 18*18*16

@compute @workgroup_size(16, 16)
fn conv2d_3x3(
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(workgroup_id) wid: vec3u
) {
  let W = params.width;
  let H = params.height;
  let HW = H * W;
  let c_in = params.c_in;
  let c_out = params.c_out;
  let ox = wid.x * TILE + lid.x;
  let oy = wid.y * TILE + lid.y;
  let oob = ox >= W || oy >= H;
  let oc_base = wid.z * OC_BATCH;

  // 8 accumulators — one per output channel
  var a0: f32=0.; var a1: f32=0.; var a2: f32=0.; var a3: f32=0.;
  var a4: f32=0.; var a5: f32=0.; var a6: f32=0.; var a7: f32=0.;

  // Load biases
  if !oob {
    let bo = params.bias_off + oc_base;
    if oc_base      < c_out { a0 = f32(weights[bo]);      }
    if oc_base + 1u < c_out { a1 = f32(weights[bo + 1u]); }
    if oc_base + 2u < c_out { a2 = f32(weights[bo + 2u]); }
    if oc_base + 3u < c_out { a3 = f32(weights[bo + 3u]); }
    if oc_base + 4u < c_out { a4 = f32(weights[bo + 4u]); }
    if oc_base + 5u < c_out { a5 = f32(weights[bo + 5u]); }
    if oc_base + 6u < c_out { a6 = f32(weights[bo + 6u]); }
    if oc_base + 7u < c_out { a7 = f32(weights[bo + 7u]); }
  }

  let tile_x = i32(wid.x * TILE) - 1i;
  let tile_y = i32(wid.y * TILE) - 1i;
  let flat_id = lid.y * TILE + lid.x;
  let split = select(c_in, params.split_ch, params.split_ch > 0u);
  let lx = lid.x + 1u;
  let ly = lid.y + 1u;
  let oc_count = min(OC_BATCH, c_out - min(oc_base, c_out));

  for (var ch_base = 0u; ch_base < c_in; ch_base += CH_TILE) {
    let ch_count = min(CH_TILE, c_in - ch_base);
    let total_sm = PAD * PAD * ch_count;

    for (var t = flat_id; t < total_sm; t += 256u) {
      let ch_local = t / (PAD * PAD);
      let spatial = t - ch_local * (PAD * PAD);
      let sy = spatial / PAD;
      let sx = spatial - sy * PAD;
      let px = u32(clamp(tile_x + i32(sx), 0, i32(W) - 1));
      let py = u32(clamp(tile_y + i32(sy), 0, i32(H) - 1));
      let ic = ch_base + ch_local;
      var val: f16 = 0.0h;
      if ic < split { val = buf_a[ic * HW + py * W + px]; }
      else if ic < c_in { val = buf_b[(ic - split) * HW + py * W + px]; }
      sm[t] = val;
    }
    workgroupBarrier();

    if !oob {
      for (var ch_local = 0u; ch_local < ch_count; ch_local++) {
        let ic = ch_base + ch_local;
        let s = ch_local * (PAD * PAD);
        let v00=f32(sm[s+(ly-1u)*PAD+(lx-1u)]); let v01=f32(sm[s+(ly-1u)*PAD+lx]); let v02=f32(sm[s+(ly-1u)*PAD+(lx+1u)]);
        let v10=f32(sm[s+ly*PAD+(lx-1u)]);      let v11=f32(sm[s+ly*PAD+lx]);      let v12=f32(sm[s+ly*PAD+(lx+1u)]);
        let v20=f32(sm[s+(ly+1u)*PAD+(lx-1u)]); let v21=f32(sm[s+(ly+1u)*PAD+lx]); let v22=f32(sm[s+(ly+1u)*PAD+(lx+1u)]);

        for (var oi = 0u; oi < oc_count; oi++) {
          let w = params.weight_off + ((oc_base + oi) * c_in + ic) * 9u;
          let d = v00*f32(weights[w])+v01*f32(weights[w+1u])+v02*f32(weights[w+2u])
                + v10*f32(weights[w+3u])+v11*f32(weights[w+4u])+v12*f32(weights[w+5u])
                + v20*f32(weights[w+6u])+v21*f32(weights[w+7u])+v22*f32(weights[w+8u]);
          switch oi {
            case 0u: { a0 += d; }
            case 1u: { a1 += d; }
            case 2u: { a2 += d; }
            case 3u: { a3 += d; }
            case 4u: { a4 += d; }
            case 5u: { a5 += d; }
            case 6u: { a6 += d; }
            default: { a7 += d; }
          }
        }
      }
    }
    workgroupBarrier();
  }

  if !oob {
    let si = oy * W + ox;
    let relu = params.apply_relu > 0u;
    if oc_base      < c_out { buf_out[ oc_base      * HW + si] = f16(select(a0, max(a0,0.), relu)); }
    if oc_base + 1u < c_out { buf_out[(oc_base + 1u)* HW + si] = f16(select(a1, max(a1,0.), relu)); }
    if oc_base + 2u < c_out { buf_out[(oc_base + 2u)* HW + si] = f16(select(a2, max(a2,0.), relu)); }
    if oc_base + 3u < c_out { buf_out[(oc_base + 3u)* HW + si] = f16(select(a3, max(a3,0.), relu)); }
    if oc_base + 4u < c_out { buf_out[(oc_base + 4u)* HW + si] = f16(select(a4, max(a4,0.), relu)); }
    if oc_base + 5u < c_out { buf_out[(oc_base + 5u)* HW + si] = f16(select(a5, max(a5,0.), relu)); }
    if oc_base + 6u < c_out { buf_out[(oc_base + 6u)* HW + si] = f16(select(a6, max(a6,0.), relu)); }
    if oc_base + 7u < c_out { buf_out[(oc_base + 7u)* HW + si] = f16(select(a7, max(a7,0.), relu)); }
  }
}
