// RELAX-inspired Spatial Denoiser + Composite for WebGPU
// 3x3 à-trous wavelet filter with normal + depth + luminance edge-stopping
// Run 3 iterations with step_size = 1, 2, 4
// Then composite: AgX tonemap the denoised HDR result

struct Params {
  resolution: vec2f,
  step_size: f32,
  _pad: f32,
};

// --- À-trous spatial filter pass ---
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var in_color: texture_2d<f32>;          // HDR input (ping or pong)
@group(0) @binding(2) var out_color: texture_storage_2d<rgba16float, write>; // filtered output
@group(0) @binding(3) var gbuf_nd: texture_2d<f32>;           // normal.xyz + depth

fn luma(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }

@compute @workgroup_size(8, 8)
fn atrous(@builtin(global_invocation_id) gid: vec3u) {
  let px = vec2i(gid.xy);
  let sz = vec2i(params.resolution);
  if px.x >= sz.x || px.y >= sz.y { return; }

  let cnd = textureLoad(gbuf_nd, px, 0);
  let cn = cnd.xyz;         // center normal
  let cz = cnd.w;           // center depth
  let cc = textureLoad(in_color, px, 0).rgb;
  let cl = luma(cc);

  // Depth gradient for plane-distance weighting
  let zr = textureLoad(gbuf_nd, min(px + vec2i(1,0), sz-1), 0).w;
  let zu = textureLoad(gbuf_nd, min(px + vec2i(0,1), sz-1), 0).w;
  let gz = max(abs(zr - cz), abs(zu - cz));

  let step = i32(params.step_size);

  // 5x5 B3-spline kernel (eliminates circular patterns from 3x3)
  let kw = array<f32, 25>(
    1.0/256.0,  4.0/256.0,  6.0/256.0,  4.0/256.0, 1.0/256.0,
    4.0/256.0, 16.0/256.0, 24.0/256.0, 16.0/256.0, 4.0/256.0,
    6.0/256.0, 24.0/256.0, 36.0/256.0, 24.0/256.0, 6.0/256.0,
    4.0/256.0, 16.0/256.0, 24.0/256.0, 16.0/256.0, 4.0/256.0,
    1.0/256.0,  4.0/256.0,  6.0/256.0,  4.0/256.0, 1.0/256.0,
  );

  var sum = vec3f(0.0);
  var wsum = 0.0;

  for (var dy = -2; dy <= 2; dy++) {
    for (var dx = -2; dx <= 2; dx++) {
      let ki = u32((dy + 2) * 5 + (dx + 2));
      let sp = clamp(px + vec2i(dx, dy) * step, vec2i(0), sz - 1);
      let snd = textureLoad(gbuf_nd, sp, 0);
      let sc = textureLoad(in_color, sp, 0).rgb;

      // Normal: softer falloff (64 instead of 128) to smooth shadow penumbras
      let wn = pow(max(dot(cn, snd.xyz), 0.0), 64.0);

      // Depth: plane-distance aware
      let dz = abs(cz - snd.w);
      let wz = exp(-dz / (gz * f32(step) + 0.001));

      // Luminance: adaptive — more relaxed in dark/shadow regions
      let dl = abs(cl - luma(sc));
      let sigma_l = 3.0 + 1.0 / (cl + 0.08);
      let wl = exp(-dl / sigma_l);

      let w = kw[ki] * wn * wz * wl;
      sum += sc * w;
      wsum += w;
    }
  }

  textureStore(out_color, px, vec4f(sum / max(wsum, 1e-6), 1.0));
}

// --- Composite pass: AgX tonemap denoised HDR → LDR output ---
@group(0) @binding(4) var composite_out: texture_storage_2d<rgba8unorm, write>;

fn agx(color_in: vec3f) -> vec3f {
  var c = mat3x3f(
    vec3f(0.6274, 0.0691, 0.0164), vec3f(0.3293, 0.9195, 0.0880), vec3f(0.0433, 0.0113, 0.8956)
  ) * color_in;
  c = mat3x3f(
    vec3f(0.856627, 0.137319, 0.111898), vec3f(0.095121, 0.761242, 0.076799), vec3f(0.048252, 0.101439, 0.811302)
  ) * c;
  c = max(c, vec3f(1e-10));
  c = clamp(log2(c), vec3f(-12.47393), vec3f(4.026069));
  c = (c + 12.47393) / (4.026069 + 12.47393);
  let x2 = c * c; let x4 = x2 * x2;
  c = 15.5*x4*x2 - 40.14*x4*c + 31.96*x4 - 6.868*x2*c + 0.4298*x2 + 0.1191*c - 0.00232;
  c = pow(max(vec3f(0.0), c), vec3f(1.35));
  let l = dot(c, vec3f(0.2126, 0.7152, 0.0722));
  c = l + 1.4 * (c - l);
  c = mat3x3f(
    vec3f(1.1271, -0.1413, -0.1413), vec3f(-0.1106, 1.1578, -0.1106), vec3f(-0.0165, -0.0165, 1.2519)
  ) * c;
  c = pow(max(vec3f(0.0), c), vec3f(2.2));
  c = mat3x3f(
    vec3f(1.6605, -0.1246, -0.0182), vec3f(-0.5876, 1.1329, -0.1006), vec3f(-0.0728, -0.0083, 1.1187)
  ) * c;
  return clamp(c, vec3f(0.0), vec3f(1.0));
}

@compute @workgroup_size(8, 8)
fn composite(@builtin(global_invocation_id) gid: vec3u) {
  let px = vec2i(gid.xy);
  let sz = vec2i(params.resolution);
  if px.x >= sz.x || px.y >= sz.y { return; }

  let hdr = textureLoad(in_color, px, 0).rgb; // denoised HDR
  var c = agx(hdr);
  c = pow(max(c, vec3f(0.0)), vec3f(1.0 / 2.2));
  textureStore(composite_out, px, vec4f(c, 1.0));
}
