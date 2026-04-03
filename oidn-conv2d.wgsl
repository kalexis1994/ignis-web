// OIDN Conv2D 3x3 — Shared-memory tiled convolution
// Each workgroup: 16x16 threads processing one output channel (z-dimension)
// Channel tiling: loads CH_TILE input channels to shared memory per iteration

enable f16;

struct ConvParams {
  width: u32,
  height: u32,
  c_in: u32,
  c_out: u32,
  weight_off: u32,   // f16 element offset for weights
  bias_off: u32,     // f16 element offset for bias
  apply_relu: u32,
  split_ch: u32,     // virtual concat split point (0 = no concat)
};

@group(0) @binding(0) var<uniform> params: ConvParams;
@group(0) @binding(1) var<storage, read> weights: array<f16>;
@group(0) @binding(2) var<storage, read> buf_a: array<f16>;
@group(0) @binding(3) var<storage, read_write> buf_out: array<f16>;
@group(0) @binding(4) var<storage, read> buf_b: array<f16>;

const TILE: u32 = 16u;
const HALO: u32 = 1u;
const PAD: u32 = 18u;   // TILE + 2*HALO
const CH_TILE: u32 = 8u; // channels per shared memory batch

// Shared memory: 18×18 × CH_TILE = 2592 f16 values (5184 bytes)
var<workgroup> sm: array<f16, 2592>;

@compute @workgroup_size(16, 16)
fn conv2d_3x3(
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(workgroup_id) wid: vec3u
) {
  let W = params.width;
  let H = params.height;
  let HW = H * W;
  let c_in = params.c_in;

  let ox = wid.x * TILE + lid.x;
  let oy = wid.y * TILE + lid.y;
  let oc = wid.z;
  let oob = ox >= W || oy >= H || oc >= params.c_out;

  // Bias
  var acc: f32 = select(0.0, f32(weights[params.bias_off + oc]), !oob);

  let tile_x = i32(wid.x * TILE) - i32(HALO);
  let tile_y = i32(wid.y * TILE) - i32(HALO);
  let flat_id = lid.y * TILE + lid.x; // 0..255
  let split = select(c_in, params.split_ch, params.split_ch > 0u);

  let lx = lid.x + HALO;
  let ly = lid.y + HALO;

  for (var ch_base = 0u; ch_base < c_in; ch_base += CH_TILE) {
    let ch_count = min(CH_TILE, c_in - ch_base);
    let total_sm = PAD * PAD * ch_count;

    // Cooperative tile load
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
      if ic < split {
        val = buf_a[ic * HW + src];
      } else if ic < c_in {
        val = buf_b[(ic - split) * HW + src];
      }
      sm[t] = val;
    }
    workgroupBarrier();

    // Accumulate 3x3 from shared memory
    if !oob {
      for (var ch_local = 0u; ch_local < ch_count; ch_local++) {
        let ic = ch_base + ch_local;
        let w_base = params.weight_off + (oc * c_in + ic) * 9u;
        let s = ch_local * (PAD * PAD);

        acc += f32(sm[s + (ly - 1u) * PAD + (lx - 1u)]) * f32(weights[w_base      ]);
        acc += f32(sm[s + (ly - 1u) * PAD +  lx       ]) * f32(weights[w_base + 1u ]);
        acc += f32(sm[s + (ly - 1u) * PAD + (lx + 1u) ]) * f32(weights[w_base + 2u ]);
        acc += f32(sm[s +  ly       * PAD + (lx - 1u) ]) * f32(weights[w_base + 3u ]);
        acc += f32(sm[s +  ly       * PAD +  lx       ]) * f32(weights[w_base + 4u ]);
        acc += f32(sm[s +  ly       * PAD + (lx + 1u) ]) * f32(weights[w_base + 5u ]);
        acc += f32(sm[s + (ly + 1u) * PAD + (lx - 1u) ]) * f32(weights[w_base + 6u ]);
        acc += f32(sm[s + (ly + 1u) * PAD +  lx       ]) * f32(weights[w_base + 7u ]);
        acc += f32(sm[s + (ly + 1u) * PAD + (lx + 1u) ]) * f32(weights[w_base + 8u ]);
      }
    }
    workgroupBarrier();
  }

  if !oob {
    if params.apply_relu > 0u { acc = max(acc, 0.0); }
    buf_out[oc * HW + oy * W + ox] = f16(acc);
  }
}
