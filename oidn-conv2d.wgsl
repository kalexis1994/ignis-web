// OIDN Conv2D 3x3 — 8x8 workgroup for high occupancy, 8 output channels unrolled

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
const PAD: u32 = 10u;    // TILE + 2*HALO
const CH_TILE: u32 = 8u; // smaller tile = less shared memory

// Shared memory: 10x10 * 8 channels = 800 f16 = 1600 bytes (tiny → high occupancy)
var<workgroup> sm: array<f16, 800>;

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
  let lx = lid.x + 1u;
  let ly = lid.y + 1u;
  let oc_count = min(8u, c_out - min(oc_base, c_out));

  for (var ch_base = 0u; ch_base < c_in; ch_base += CH_TILE) {
    let ch_count = min(CH_TILE, c_in - ch_base);
    let total_sm = PAD * PAD * ch_count;

    // 64 threads loading up to 800 values = ~12.5 loads/thread
    for (var t = flat_id; t < total_sm; t += 64u) {
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

  if !oob {
    let si = oy * W + ox;
    let relu = params.apply_relu > 0u;
    if oc_base     <c_out{buf_out[ oc_base     *HW+si]=f16(select(a0,max(a0,0.),relu));}
    if oc_base+1u<c_out{buf_out[(oc_base+1u)*HW+si]=f16(select(a1,max(a1,0.),relu));}
    if oc_base+2u<c_out{buf_out[(oc_base+2u)*HW+si]=f16(select(a2,max(a2,0.),relu));}
    if oc_base+3u<c_out{buf_out[(oc_base+3u)*HW+si]=f16(select(a3,max(a3,0.),relu));}
    if oc_base+4u<c_out{buf_out[(oc_base+4u)*HW+si]=f16(select(a4,max(a4,0.),relu));}
    if oc_base+5u<c_out{buf_out[(oc_base+5u)*HW+si]=f16(select(a5,max(a5,0.),relu));}
    if oc_base+6u<c_out{buf_out[(oc_base+6u)*HW+si]=f16(select(a6,max(a6,0.),relu));}
    if oc_base+7u<c_out{buf_out[(oc_base+7u)*HW+si]=f16(select(a7,max(a7,0.),relu));}
  }
}
