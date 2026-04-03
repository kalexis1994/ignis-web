// OIDN Conv2D 3x3 — NHWC layout, 8x8 workgroup, 8 output channels
// NHWC: all channels per pixel contiguous → coalesced vec4 reads, cache-friendly

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

const TILE: u32 = 8u;
const PAD: u32 = 10u;
const CH_TILE: u32 = 8u;

// Shared memory: 10x10 spatial × CH_TILE channels = 800 f16 (1600 bytes)
var<workgroup> sm: array<f16, 800>;

@compute @workgroup_size(8, 8)
fn conv2d_3x3(
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(workgroup_id) wid: vec3u
) {
  let W = params.width;
  let H = params.height;
  let c_in = params.c_in;
  let c_out = params.c_out;
  let ox = wid.x * TILE + lid.x;
  let oy = wid.y * TILE + lid.y;
  let oob = ox >= W || oy >= H;
  let oc_base = wid.z * 8u;

  var a0:f32=0.;var a1:f32=0.;var a2:f32=0.;var a3:f32=0.;
  var a4:f32=0.;var a5:f32=0.;var a6:f32=0.;var a7:f32=0.;
  if !oob {
    let bo = params.bias_off + oc_base;
    if oc_base     <c_out{a0=f32(weights[bo]);}     if oc_base+1u<c_out{a1=f32(weights[bo+1u]);}
    if oc_base+2u<c_out{a2=f32(weights[bo+2u]);}   if oc_base+3u<c_out{a3=f32(weights[bo+3u]);}
    if oc_base+4u<c_out{a4=f32(weights[bo+4u]);}   if oc_base+5u<c_out{a5=f32(weights[bo+5u]);}
    if oc_base+6u<c_out{a6=f32(weights[bo+6u]);}   if oc_base+7u<c_out{a7=f32(weights[bo+7u]);}
  }

  let tile_x = i32(wid.x * TILE) - 1i;
  let tile_y = i32(wid.y * TILE) - 1i;
  let flat_id = lid.y * TILE + lid.x; // 0..63
  let split = select(c_in, params.split_ch, params.split_ch > 0u);
  let c_a = split;           // channels per pixel in buf_a
  let c_b = c_in - split;    // channels per pixel in buf_b
  let lx = lid.x + 1u;
  let ly = lid.y + 1u;

  for (var ch_base = 0u; ch_base < c_in; ch_base += CH_TILE) {
    let ch_count = min(CH_TILE, c_in - ch_base);
    let total_sm = PAD * PAD * ch_count;

    // Cooperative load: NHWC buffers → shared memory
    // sm layout: sm[spatial * CH_TILE + ch_local] (channels innermost)
    for (var t = flat_id; t < total_sm; t += 64u) {
      let spatial = t / CH_TILE;
      let ch_local = t - spatial * CH_TILE;
      if ch_local >= ch_count { sm[t] = 0.0h; continue; }
      let sy = spatial / PAD;
      let sx = spatial - sy * PAD;
      let px = u32(clamp(tile_x + i32(sx), 0, i32(W) - 1));
      let py = u32(clamp(tile_y + i32(sy), 0, i32(H) - 1));
      let ic = ch_base + ch_local;
      let pixel = py * W + px;
      var val: f16 = 0.0h;
      // NHWC: buf_a[(y*W+x)*C_a + ic], buf_b[(y*W+x)*C_b + (ic-split)]
      if ic < split {
        val = buf_a[pixel * c_a + ic];
      } else if ic < c_in {
        val = buf_b[pixel * c_b + (ic - split)];
      }
      sm[t] = val;
    }
    workgroupBarrier();

    if !oob {
      for (var ch_local = 0u; ch_local < ch_count; ch_local++) {
        let ic = ch_base + ch_local;
        // Read 9 neighbors from shared memory (NHWC: stride = CH_TILE between channels)
        let v00=f32(sm[((ly-1u)*PAD+(lx-1u))*CH_TILE+ch_local]);
        let v01=f32(sm[((ly-1u)*PAD+ lx    )*CH_TILE+ch_local]);
        let v02=f32(sm[((ly-1u)*PAD+(lx+1u))*CH_TILE+ch_local]);
        let v10=f32(sm[( ly    *PAD+(lx-1u))*CH_TILE+ch_local]);
        let v11=f32(sm[( ly    *PAD+ lx    )*CH_TILE+ch_local]);
        let v12=f32(sm[( ly    *PAD+(lx+1u))*CH_TILE+ch_local]);
        let v20=f32(sm[((ly+1u)*PAD+(lx-1u))*CH_TILE+ch_local]);
        let v21=f32(sm[((ly+1u)*PAD+ lx    )*CH_TILE+ch_local]);
        let v22=f32(sm[((ly+1u)*PAD+(lx+1u))*CH_TILE+ch_local]);

        if oc_base     <c_out{let w=params.weight_off+(oc_base     *c_in+ic)*9u;a0+=v00*f32(weights[w])+v01*f32(weights[w+1u])+v02*f32(weights[w+2u])+v10*f32(weights[w+3u])+v11*f32(weights[w+4u])+v12*f32(weights[w+5u])+v20*f32(weights[w+6u])+v21*f32(weights[w+7u])+v22*f32(weights[w+8u]);}
        if oc_base+1u<c_out{let w=params.weight_off+((oc_base+1u)*c_in+ic)*9u;a1+=v00*f32(weights[w])+v01*f32(weights[w+1u])+v02*f32(weights[w+2u])+v10*f32(weights[w+3u])+v11*f32(weights[w+4u])+v12*f32(weights[w+5u])+v20*f32(weights[w+6u])+v21*f32(weights[w+7u])+v22*f32(weights[w+8u]);}
        if oc_base+2u<c_out{let w=params.weight_off+((oc_base+2u)*c_in+ic)*9u;a2+=v00*f32(weights[w])+v01*f32(weights[w+1u])+v02*f32(weights[w+2u])+v10*f32(weights[w+3u])+v11*f32(weights[w+4u])+v12*f32(weights[w+5u])+v20*f32(weights[w+6u])+v21*f32(weights[w+7u])+v22*f32(weights[w+8u]);}
        if oc_base+3u<c_out{let w=params.weight_off+((oc_base+3u)*c_in+ic)*9u;a3+=v00*f32(weights[w])+v01*f32(weights[w+1u])+v02*f32(weights[w+2u])+v10*f32(weights[w+3u])+v11*f32(weights[w+4u])+v12*f32(weights[w+5u])+v20*f32(weights[w+6u])+v21*f32(weights[w+7u])+v22*f32(weights[w+8u]);}
        if oc_base+4u<c_out{let w=params.weight_off+((oc_base+4u)*c_in+ic)*9u;a4+=v00*f32(weights[w])+v01*f32(weights[w+1u])+v02*f32(weights[w+2u])+v10*f32(weights[w+3u])+v11*f32(weights[w+4u])+v12*f32(weights[w+5u])+v20*f32(weights[w+6u])+v21*f32(weights[w+7u])+v22*f32(weights[w+8u]);}
        if oc_base+5u<c_out{let w=params.weight_off+((oc_base+5u)*c_in+ic)*9u;a5+=v00*f32(weights[w])+v01*f32(weights[w+1u])+v02*f32(weights[w+2u])+v10*f32(weights[w+3u])+v11*f32(weights[w+4u])+v12*f32(weights[w+5u])+v20*f32(weights[w+6u])+v21*f32(weights[w+7u])+v22*f32(weights[w+8u]);}
        if oc_base+6u<c_out{let w=params.weight_off+((oc_base+6u)*c_in+ic)*9u;a6+=v00*f32(weights[w])+v01*f32(weights[w+1u])+v02*f32(weights[w+2u])+v10*f32(weights[w+3u])+v11*f32(weights[w+4u])+v12*f32(weights[w+5u])+v20*f32(weights[w+6u])+v21*f32(weights[w+7u])+v22*f32(weights[w+8u]);}
        if oc_base+7u<c_out{let w=params.weight_off+((oc_base+7u)*c_in+ic)*9u;a7+=v00*f32(weights[w])+v01*f32(weights[w+1u])+v02*f32(weights[w+2u])+v10*f32(weights[w+3u])+v11*f32(weights[w+4u])+v12*f32(weights[w+5u])+v20*f32(weights[w+6u])+v21*f32(weights[w+7u])+v22*f32(weights[w+8u]);}
      }
    }
    workgroupBarrier();
  }

  // Write output in NHWC: buf_out[(y*W+x)*c_out + oc]
  if !oob {
    let pixel = oy * W + ox;
    let relu = params.apply_relu > 0u;
    let base = pixel * c_out + oc_base;
    if oc_base     <c_out{buf_out[base     ]=f16(select(a0,max(a0,0.),relu));}
    if oc_base+1u<c_out{buf_out[base+1u]=f16(select(a1,max(a1,0.),relu));}
    if oc_base+2u<c_out{buf_out[base+2u]=f16(select(a2,max(a2,0.),relu));}
    if oc_base+3u<c_out{buf_out[base+3u]=f16(select(a3,max(a3,0.),relu));}
    if oc_base+4u<c_out{buf_out[base+4u]=f16(select(a4,max(a4,0.),relu));}
    if oc_base+5u<c_out{buf_out[base+5u]=f16(select(a5,max(a5,0.),relu));}
    if oc_base+6u<c_out{buf_out[base+6u]=f16(select(a6,max(a6,0.),relu));}
    if oc_base+7u<c_out{buf_out[base+7u]=f16(select(a7,max(a7,0.),relu));}
  }
}
