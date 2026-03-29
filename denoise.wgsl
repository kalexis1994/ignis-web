// SVGF-inspired Denoiser — Variance-Guided À-Trous Wavelet Filter
// Key insight: estimate noise variance per-pixel from the local neighborhood.
// High variance (shadows, indirect) → large sigma → blur more
// Low variance (textures, edges) → small sigma → preserve detail

struct Params {
  resolution: vec2f,
  step_size: f32,
  pass_index: f32, // 0=first pass (computes variance), 1+=subsequent
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var in_color: texture_2d<f32>;
@group(0) @binding(2) var out_color: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var gbuf_nd: texture_2d<f32>;

fn luma(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }

// 5x5 B3-spline kernel weights
const KW = array<f32, 25>(
  1.0/256.0,  4.0/256.0,  6.0/256.0,  4.0/256.0, 1.0/256.0,
  4.0/256.0, 16.0/256.0, 24.0/256.0, 16.0/256.0, 4.0/256.0,
  6.0/256.0, 24.0/256.0, 36.0/256.0, 24.0/256.0, 6.0/256.0,
  4.0/256.0, 16.0/256.0, 24.0/256.0, 16.0/256.0, 4.0/256.0,
  1.0/256.0,  4.0/256.0,  6.0/256.0,  4.0/256.0, 1.0/256.0,
);

@compute @workgroup_size(8, 8)
fn atrous(@builtin(global_invocation_id) gid: vec3u) {
  let px = vec2i(gid.xy);
  let sz = vec2i(params.resolution);
  if px.x >= sz.x || px.y >= sz.y { return; }

  let cnd = textureLoad(gbuf_nd, px, 0);
  let cn = cnd.xyz;
  let cz = cnd.w;
  let cc = textureLoad(in_color, px, 0).rgb;
  let cl = luma(cc);

  // Depth gradient
  let zr = textureLoad(gbuf_nd, min(px + vec2i(1,0), sz-1), 0).w;
  let zu = textureLoad(gbuf_nd, min(px + vec2i(0,1), sz-1), 0).w;
  let gz = max(abs(zr - cz), abs(zu - cz));

  let step = i32(params.step_size);

  // === VARIANCE ESTIMATION (SVGF) ===
  // Compute local variance from 3x3 neighborhood of input
  // This adapts the luminance sigma per-pixel
  var m1 = 0.0; // mean of luminance
  var m2 = 0.0; // mean of luminance²
  for (var vy = -1; vy <= 1; vy++) {
    for (var vx = -1; vx <= 1; vx++) {
      let vp = clamp(px + vec2i(vx, vy), vec2i(0), sz - 1);
      let vl = luma(textureLoad(in_color, vp, 0).rgb);
      m1 += vl;
      m2 += vl * vl;
    }
  }
  m1 /= 9.0;
  m2 /= 9.0;
  let variance = max(m2 - m1 * m1, 0.0);
  let std_dev = sqrt(variance);

  // === FILTER ===
  var sum = vec3f(0.0);
  var wsum = 0.0;

  for (var dy = -2; dy <= 2; dy++) {
    for (var dx = -2; dx <= 2; dx++) {
      let ki = u32((dy + 2) * 5 + (dx + 2));
      let sp = clamp(px + vec2i(dx, dy) * step, vec2i(0), sz - 1);
      let snd = textureLoad(gbuf_nd, sp, 0);
      let sc = textureLoad(in_color, sp, 0).rgb;

      // Normal edge-stopping
      let wn = pow(max(dot(cn, snd.xyz), 0.0), 128.0);

      // Depth edge-stopping
      let dz = abs(cz - snd.w);
      let wz = exp(-dz / (gz * f32(step) + 0.001));

      // === VARIANCE-GUIDED luminance edge-stopping (SVGF key technique) ===
      // sigma = f(std_dev): noisy areas get large sigma (more blur)
      //                     clean areas get small sigma (preserve detail)
      let dl = abs(cl - luma(sc));
      let sigma_l = 4.0 * std_dev + 0.01;
      let wl = exp(-dl / max(sigma_l, 0.01));

      let w = KW[ki] * wn * wz * wl;
      sum += sc * w;
      wsum += w;
    }
  }

  textureStore(out_color, px, vec4f(sum / max(wsum, 1e-6), 1.0));
}

// === COMPOSITE: remodulate albedo + AgX tonemap ===
@group(0) @binding(4) var composite_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(5) var albedo_tex: texture_2d<f32>;

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

  let denoised_irr = textureLoad(in_color, px, 0).rgb;
  let albedo = textureLoad(albedo_tex, px, 0).rgb;
  var hdr = denoised_irr * max(albedo, vec3f(0.02));
  var c = agx(hdr);
  c = pow(max(c, vec3f(0.0)), vec3f(1.0 / 2.2));
  textureStore(composite_out, px, vec4f(c, 1.0));
}
