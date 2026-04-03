// OIDN Conv2D 3x3 — Core neural denoiser kernel
// Shared-memory tiled convolution with channel tiling and output batching.
// Processes OUT_CH output channels per workgroup in the z-dimension.

enable f16;

struct ConvParams {
  width: u32,            // padded feature map width
  height: u32,           // padded feature map height
  c_in: u32,             // total input channels
  c_out: u32,            // total output channels
  weight_off: u32,       // element offset into weight buffer (f16 elements, not bytes)
  bias_off: u32,         // element offset into weight buffer for bias
  apply_relu: u32,       // 1 = ReLU, 0 = linear
  split_ch: u32,         // for virtual concat: channels from buf_a (rest from buf_b). 0 = no concat.
};

@group(0) @binding(0) var<uniform> params: ConvParams;
@group(0) @binding(1) var<storage, read> weights: array<f16>;
@group(0) @binding(2) var<storage, read> buf_a: array<f16>;       // input (or first part of concat)
@group(0) @binding(3) var<storage, read_write> buf_out: array<f16>; // output
@group(0) @binding(4) var<storage, read> buf_b: array<f16>;       // second part of concat (skip connection)

const TILE: u32 = 16u;
const HALO: u32 = 1u;    // padding=1 for 3x3 conv
const PAD: u32 = 18u;    // TILE + 2*HALO
const CH_TILE: u32 = 9u; // input channels per shared memory load (9 fits 3x3 kernel nicely)

// Shared memory: 18x18 spatial × CH_TILE channels = 2916 f16 values (5832 bytes)
var<workgroup> sm: array<f16, 2916>; // PAD * PAD * CH_TILE

@compute @workgroup_size(16, 16)
fn conv2d_3x3(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(workgroup_id) wid: vec3u
) {
  let W = params.width;
  let H = params.height;
  let HW = H * W;
  let c_in = params.c_in;
  let c_out = params.c_out;

  // Output pixel position
  let ox = wid.x * TILE + lid.x;
  let oy = wid.y * TILE + lid.y;
  let out_of_bounds = (ox >= W || oy >= H);

  // Which output channel this workgroup handles (z-dimension)
  let oc = wid.z;
  if oc >= c_out { return; }

  // Load bias
  var acc: f32 = f32(weights[params.bias_off + oc]);

  // Tile origin in spatial coords (top-left including halo)
  let tile_x = i32(wid.x * TILE) - i32(HALO);
  let tile_y = i32(wid.y * TILE) - i32(HALO);
  let flat_id = lid.y * TILE + lid.x; // 0..255

  // Split channel: where buf_a ends and buf_b begins (for virtual concat)
  let split = select(c_in, params.split_ch, params.split_ch > 0u);

  // Process input channels in tiles of CH_TILE
  for (var ch_base = 0u; ch_base < c_in; ch_base += CH_TILE) {
    let ch_end = min(ch_base + CH_TILE, c_in);
    let ch_count = ch_end - ch_base;

    // --- Cooperative load: 18x18 × ch_count values into shared memory ---
    let total_sm = PAD * PAD * ch_count;
    for (var t = flat_id; t < total_sm; t += 256u) {
      let ch_local = t / (PAD * PAD);
      let spatial = t % (PAD * PAD);
      let sy = i32(spatial / PAD);
      let sx = i32(spatial % PAD);

      let px = clamp(tile_x + sx, 0, i32(W) - 1);
      let py = clamp(tile_y + sy, 0, i32(H) - 1);
      let ic = ch_base + ch_local;
      let spatial_idx = u32(py) * W + u32(px);

      // Read from buf_a or buf_b depending on channel index vs split point
      var val: f16 = 0.0h;
      if ic < split {
        val = buf_a[ic * HW + spatial_idx];
      } else if ic < c_in {
        val = buf_b[(ic - split) * HW + spatial_idx];
      }
      sm[t] = val;
    }
    workgroupBarrier();

    // --- Accumulate 3x3 convolution from shared memory ---
    if !out_of_bounds {
      let lx = lid.x + HALO; // local position in padded tile
      let ly = lid.y + HALO;

      for (var ch_local = 0u; ch_local < ch_count; ch_local++) {
        let ic = ch_base + ch_local;
        // Weight offset: weights are OIHW, each kernel is 9 f16 values
        let w_base = params.weight_off + (oc * c_in + ic) * 9u;
        let sm_ch_base = ch_local * (PAD * PAD);

        // Unrolled 3x3 dot product from shared memory
        var dot: f32 = 0.0;
        dot += f32(sm[sm_ch_base + (ly - 1u) * PAD + (lx - 1u)]) * f32(weights[w_base + 0u]);
        dot += f32(sm[sm_ch_base + (ly - 1u) * PAD + lx])        * f32(weights[w_base + 1u]);
        dot += f32(sm[sm_ch_base + (ly - 1u) * PAD + (lx + 1u)]) * f32(weights[w_base + 2u]);
        dot += f32(sm[sm_ch_base + ly * PAD + (lx - 1u)])         * f32(weights[w_base + 3u]);
        dot += f32(sm[sm_ch_base + ly * PAD + lx])                * f32(weights[w_base + 4u]);
        dot += f32(sm[sm_ch_base + ly * PAD + (lx + 1u)])         * f32(weights[w_base + 5u]);
        dot += f32(sm[sm_ch_base + (ly + 1u) * PAD + (lx - 1u)]) * f32(weights[w_base + 6u]);
        dot += f32(sm[sm_ch_base + (ly + 1u) * PAD + lx])         * f32(weights[w_base + 7u]);
        dot += f32(sm[sm_ch_base + (ly + 1u) * PAD + (lx + 1u)]) * f32(weights[w_base + 8u]);

        acc += dot;
      }
    }
    workgroupBarrier();
  }

  // --- Write output ---
  if !out_of_bounds {
    if params.apply_relu > 0u {
      acc = max(acc, 0.0);
    }
    buf_out[oc * HW + oy * W + ox] = f16(acc);
  }
}
