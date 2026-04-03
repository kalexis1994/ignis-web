// OIDN Conv2D 3x3 — Shared-memory tiled, 4 output channels per workgroup
// Each workgroup: 16x16 threads, z = ceil(c_out/4), processes 4 output channels
// Amortizes shared memory load across 4 output channels (4x fewer global reads)

enable f16;

struct ConvParams {
  width: u32,
  height: u32,
  c_in: u32,
  c_out: u32,
  weight_off: u32,
  bias_off: u32,
  apply_relu: u32,
  split_ch: u32,
};

@group(0) @binding(0) var<uniform> params: ConvParams;
@group(0) @binding(1) var<storage, read> weights: array<f16>;
@group(0) @binding(2) var<storage, read> buf_a: array<f16>;
@group(0) @binding(3) var<storage, read_write> buf_out: array<f16>;
@group(0) @binding(4) var<storage, read> buf_b: array<f16>;

const TILE: u32 = 16u;
const HALO: u32 = 1u;
const PAD: u32 = 18u;
const CH_TILE: u32 = 8u;
const OC_BATCH: u32 = 4u; // output channels per workgroup

var<workgroup> sm: array<f16, 2592>; // PAD * PAD * CH_TILE

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

  // This workgroup handles output channels [oc_base, oc_base+4)
  let oc_base = wid.z * OC_BATCH;

  // Load biases for up to 4 output channels
  var acc0: f32 = 0.0; var acc1: f32 = 0.0; var acc2: f32 = 0.0; var acc3: f32 = 0.0;
  if !oob {
    if oc_base < c_out { acc0 = f32(weights[params.bias_off + oc_base]); }
    if oc_base + 1u < c_out { acc1 = f32(weights[params.bias_off + oc_base + 1u]); }
    if oc_base + 2u < c_out { acc2 = f32(weights[params.bias_off + oc_base + 2u]); }
    if oc_base + 3u < c_out { acc3 = f32(weights[params.bias_off + oc_base + 3u]); }
  }

  let tile_x = i32(wid.x * TILE) - i32(HALO);
  let tile_y = i32(wid.y * TILE) - i32(HALO);
  let flat_id = lid.y * TILE + lid.x;
  let split = select(c_in, params.split_ch, params.split_ch > 0u);
  let lx = lid.x + HALO;
  let ly = lid.y + HALO;

  for (var ch_base = 0u; ch_base < c_in; ch_base += CH_TILE) {
    let ch_count = min(CH_TILE, c_in - ch_base);
    let total_sm = PAD * PAD * ch_count;

    // Cooperative tile load (same for all 4 output channels — the key optimization)
    for (var t = flat_id; t < total_sm; t += 256u) {
      let ch_local = t / (PAD * PAD);
      let spatial = t - ch_local * (PAD * PAD);
      let sy = spatial / PAD;
      let sx = spatial - sy * PAD;
      let px = u32(clamp(tile_x + i32(sx), 0, i32(W) - 1));
      let py = u32(clamp(tile_y + i32(sy), 0, i32(H) - 1));
      let ic = ch_base + ch_local;
      let src = py * W + px;
      var val: f16 = 0.0h;
      if ic < split { val = buf_a[ic * HW + src]; }
      else if ic < c_in { val = buf_b[(ic - split) * HW + src]; }
      sm[t] = val;
    }
    workgroupBarrier();

    // Accumulate 3x3 for all 4 output channels from the SAME shared memory tile
    if !oob {
      for (var ch_local = 0u; ch_local < ch_count; ch_local++) {
        let ic = ch_base + ch_local;
        let s = ch_local * (PAD * PAD);

        // Read 9 input values from shared memory (shared by all 4 oc)
        let v00 = f32(sm[s + (ly - 1u) * PAD + (lx - 1u)]);
        let v01 = f32(sm[s + (ly - 1u) * PAD +  lx       ]);
        let v02 = f32(sm[s + (ly - 1u) * PAD + (lx + 1u) ]);
        let v10 = f32(sm[s +  ly       * PAD + (lx - 1u) ]);
        let v11 = f32(sm[s +  ly       * PAD +  lx       ]);
        let v12 = f32(sm[s +  ly       * PAD + (lx + 1u) ]);
        let v20 = f32(sm[s + (ly + 1u) * PAD + (lx - 1u) ]);
        let v21 = f32(sm[s + (ly + 1u) * PAD +  lx       ]);
        let v22 = f32(sm[s + (ly + 1u) * PAD + (lx + 1u) ]);

        // Each output channel has its own 9 weights but uses the SAME 9 input values
        if oc_base < c_out {
          let w = params.weight_off + (oc_base * c_in + ic) * 9u;
          acc0 += v00*f32(weights[w])+v01*f32(weights[w+1u])+v02*f32(weights[w+2u])
                + v10*f32(weights[w+3u])+v11*f32(weights[w+4u])+v12*f32(weights[w+5u])
                + v20*f32(weights[w+6u])+v21*f32(weights[w+7u])+v22*f32(weights[w+8u]);
        }
        if oc_base + 1u < c_out {
          let w = params.weight_off + ((oc_base+1u) * c_in + ic) * 9u;
          acc1 += v00*f32(weights[w])+v01*f32(weights[w+1u])+v02*f32(weights[w+2u])
                + v10*f32(weights[w+3u])+v11*f32(weights[w+4u])+v12*f32(weights[w+5u])
                + v20*f32(weights[w+6u])+v21*f32(weights[w+7u])+v22*f32(weights[w+8u]);
        }
        if oc_base + 2u < c_out {
          let w = params.weight_off + ((oc_base+2u) * c_in + ic) * 9u;
          acc2 += v00*f32(weights[w])+v01*f32(weights[w+1u])+v02*f32(weights[w+2u])
                + v10*f32(weights[w+3u])+v11*f32(weights[w+4u])+v12*f32(weights[w+5u])
                + v20*f32(weights[w+6u])+v21*f32(weights[w+7u])+v22*f32(weights[w+8u]);
        }
        if oc_base + 3u < c_out {
          let w = params.weight_off + ((oc_base+3u) * c_in + ic) * 9u;
          acc3 += v00*f32(weights[w])+v01*f32(weights[w+1u])+v02*f32(weights[w+2u])
                + v10*f32(weights[w+3u])+v11*f32(weights[w+4u])+v12*f32(weights[w+5u])
                + v20*f32(weights[w+6u])+v21*f32(weights[w+7u])+v22*f32(weights[w+8u]);
        }
      }
    }
    workgroupBarrier();
  }

  // Write outputs
  if !oob {
    let si = oy * W + ox;
    let relu = params.apply_relu > 0u;
    if oc_base < c_out { buf_out[oc_base * HW + si] = f16(select(acc0, max(acc0, 0.0), relu)); }
    if oc_base+1u < c_out { buf_out[(oc_base+1u) * HW + si] = f16(select(acc1, max(acc1, 0.0), relu)); }
    if oc_base+2u < c_out { buf_out[(oc_base+2u) * HW + si] = f16(select(acc2, max(acc2, 0.0), relu)); }
    if oc_base+3u < c_out { buf_out[(oc_base+3u) * HW + si] = f16(select(acc3, max(acc3, 0.0), relu)); }
  }
}
