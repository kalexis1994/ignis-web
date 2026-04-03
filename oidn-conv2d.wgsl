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
  if ox >= W || oy >= H { return; }

  let oc = wid.z;
  if oc >= c_out { return; }

  // Full 3x3 convolution — naive (no shared memory)
  var acc: f32 = f32(weights[params.bias_off + oc]);
  let split = select(c_in, params.split_ch, params.split_ch > 0u);
  let spatial_idx = oy * W + ox;

  for (var ic = 0u; ic < c_in; ic++) {
    let w_base = params.weight_off + (oc * c_in + ic) * 9u;
    for (var ky = 0u; ky < 3u; ky++) {
      for (var kx = 0u; kx < 3u; kx++) {
        let px = clamp(i32(ox) + i32(kx) - 1, 0, i32(W) - 1);
        let py = clamp(i32(oy) + i32(ky) - 1, 0, i32(H) - 1);
        let src_idx = u32(py) * W + u32(px);
        var val: f16 = 0.0h;
        if ic < split {
          val = buf_a[ic * HW + src_idx];
        } else {
          val = buf_b[(ic - split) * HW + src_idx];
        }
        acc += f32(val) * f32(weights[w_base + ky * 3u + kx]);
      }
    }
  }

  if params.apply_relu > 0u { acc = max(acc, 0.0); }
  buf_out[oc * HW + spatial_idx] = f16(acc);
}
