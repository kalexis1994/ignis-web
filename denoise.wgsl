// SVGF-inspired Denoiser — Dual-Signal Variance-Guided À-Trous Wavelet Filter
// Processes diffuse irradiance and specular radiance in a single dispatch.
// Diffuse: standard variance-guided sigma
// Specular: roughness-dependent sigma (tight for glossy, wide for rough)

struct Params {
  resolution: vec2f,
  step_size: f32,
  frames_still: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var in_color: texture_2d<f32>;       // diffuse input
@group(0) @binding(2) var out_color: texture_storage_2d<rgba16float, write>; // diffuse output
@group(0) @binding(3) var gbuf_nd: texture_2d<f32>;        // normal.xyz + depth
@group(0) @binding(4) var in_spec: texture_2d<f32>;        // specular input
@group(0) @binding(5) var out_spec: texture_storage_2d<rgba16float, write>; // specular output
@group(0) @binding(6) var albedo_tex: texture_2d<f32>;     // albedo.rgb + roughness.a

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
  let diff_sample = textureLoad(in_color, px, 0);
  let cc = diff_sample.rgb;
  let cl = luma(cc);
  let spec_sample = textureLoad(in_spec, px, 0);
  let cs = spec_sample.rgb;
  let csl = luma(cs);
  let hit_dist = spec_sample.a;

  // Roughness for specular filter width
  let roughness = textureLoad(albedo_tex, px, 0).a;

  // Hit distance modulates blur: contact shadows stay sharp, far GI gets smoothed
  let hit_factor = mix(0.5, 1.0, hit_dist);

  // Depth gradient
  let zr = textureLoad(gbuf_nd, min(px + vec2i(1,0), sz-1), 0).w;
  let zu = textureLoad(gbuf_nd, min(px + vec2i(0,1), sz-1), 0).w;
  let gz = max(abs(zr - cz), abs(zu - cz));

  let step = i32(params.step_size);

  // === VARIANCE (SVGF §4.2): propagated through à-trous passes ===
  // Pass 0 (step=1): compute spatial variance, write to alpha
  // Pass 1+ (step>1): read propagated variance from alpha (filtered by previous passes)
  // This ensures sigma reflects ORIGINAL noise, not the already-filtered signal.
  var d_var: f32;
  var s_m1 = 0.0; var s_m2 = 0.0; // specular always uses spatial (no alpha channel)

  if step <= 1 {
    // First pass: compute spatial 3×3 variance, initialize alpha chain
    var d_m1 = 0.0; var d_m2 = 0.0;
    for (var vy = -1; vy <= 1; vy++) {
      for (var vx = -1; vx <= 1; vx++) {
        let vp = clamp(px + vec2i(vx, vy), vec2i(0), sz - 1);
        let vl = luma(textureLoad(in_color, vp, 0).rgb);
        d_m1 += vl; d_m2 += vl * vl;
        let vsl = luma(textureLoad(in_spec, vp, 0).rgb);
        s_m1 += vsl; s_m2 += vsl * vsl;
      }
    }
    d_m1 /= 9.0; d_m2 /= 9.0;
    d_var = max(d_m2 - d_m1 * d_m1, 0.0);
  } else {
    // Subsequent passes: read propagated variance from diffuse alpha
    // Also Gaussian-filter the variance from 3×3 neighbors for stability
    var var_sum = 0.0;
    for (var vy = -1; vy <= 1; vy++) {
      for (var vx = -1; vx <= 1; vx++) {
        let vp = clamp(px + vec2i(vx, vy), vec2i(0), sz - 1);
        var_sum += textureLoad(in_color, vp, 0).a;
        let vsl = luma(textureLoad(in_spec, vp, 0).rgb);
        s_m1 += vsl; s_m2 += vsl * vsl;
      }
    }
    d_var = max(var_sum / 9.0, 0.0);
  }
  let d_std = sqrt(d_var);
  s_m1 /= 9.0; s_m2 /= 9.0;
  let s_std = sqrt(max(s_m2 - s_m1 * s_m1, 0.0));

  // σ_l = 4.0 per SVGF (Schied 2017, eq.5)
  let diff_sigma_scale = 4.0;
  // Specular: roughness modulates (ReLAX, NRD)
  let spec_sigma_scale = mix(4.0 * 0.3, 4.0, roughness);

  // === FILTER (both signals + variance filtering in one pass) ===
  var d_sum = vec3f(0.0); var d_wsum = 0.0;
  var s_sum = vec3f(0.0); var s_wsum = 0.0;
  var d_var_filtered = 0.0; var d_var_wsum = 0.0; // variance filtering

  for (var dy = -2; dy <= 2; dy++) {
    for (var dx = -2; dx <= 2; dx++) {
      let ki = u32((dy + 2) * 5 + (dx + 2));
      let sp = clamp(px + vec2i(dx, dy) * step, vec2i(0), sz - 1);
      let snd = textureLoad(gbuf_nd, sp, 0);
      let s_diff = textureLoad(in_color, sp, 0);
      let sd = s_diff.rgb;
      let ss = textureLoad(in_spec, sp, 0).rgb;

      // Shared edge-stopping (SVGF, Schied 2017):
      // Normal: eq.4, σ_n=128 (exponent). Depth: eq.3, σ_z=1.0.
      let wn = pow(max(dot(cn, snd.xyz), 0.0), 128.0);
      let dz = abs(cz - snd.w);
      let wz = exp(-dz / (gz * f32(step) + 1e-3));

      // Diffuse luminance edge-stopping (hit distance modulated)
      let d_dl = abs(cl - luma(sd));
      let d_sigma = diff_sigma_scale * d_std * hit_factor + 0.01;
      let d_wl = exp(-d_dl / max(d_sigma, 0.01));
      let d_w = KW[ki] * wn * wz * d_wl;
      d_sum += sd * d_w;
      d_wsum += d_w;

      // Filter variance through à-trous kernel (SVGF §4.2)
      // Uses geometry weights only (not luminance) to preserve variance at edges
      let geom_w = KW[ki] * wn * wz;
      d_var_filtered += s_diff.a * geom_w;
      d_var_wsum += geom_w;

      // Specular luminance edge-stopping (roughness + hit distance modulated)
      let s_dl = abs(csl - luma(ss));
      let s_sigma = spec_sigma_scale * s_std * hit_factor + 0.01;
      let s_wl = exp(-s_dl / max(s_sigma, 0.01));
      let s_w = KW[ki] * wn * wz * s_wl;
      s_sum += ss * s_w;
      s_wsum += s_w;
    }
  }

  // Output: color + filtered variance in alpha (propagated to next pass)
  let out_var = d_var_filtered / max(d_var_wsum, 1e-6);
  textureStore(out_color, px, vec4f(d_sum / max(d_wsum, 1e-6), out_var));
  textureStore(out_spec, px, vec4f(s_sum / max(s_wsum, 1e-6), textureLoad(in_spec, px, 0).a));
}

// === PRE-BLUR: anti-firefly percentile clamp + lightweight 3x3 bilateral ===
// Stabilizes temporal AABB and eliminates bright speckles adaptively.
@compute @workgroup_size(8, 8)
fn preblur(@builtin(global_invocation_id) gid: vec3u) {
  let px = vec2i(gid.xy);
  let sz = vec2i(params.resolution);
  if px.x >= sz.x || px.y >= sz.y { return; }

  let cnd = textureLoad(gbuf_nd, px, 0);
  let cn = cnd.xyz;
  let cz = cnd.w;
  let cc_raw = textureLoad(in_color, px, 0).rgb;
  let cs_raw = textureLoad(in_spec, px, 0).rgb;

  // === Anti-firefly: 3σ percentile clamp (HPG 2025, Lalber) ===
  var d_m1 = 0.0; var d_m2 = 0.0;
  var s_m1 = 0.0; var s_m2 = 0.0;
  for (var fy = -1; fy <= 1; fy++) {
    for (var fx = -1; fx <= 1; fx++) {
      let fp = clamp(px + vec2i(fx, fy), vec2i(0), sz - 1);
      let fl = luma(textureLoad(in_color, fp, 0).rgb);
      d_m1 += fl; d_m2 += fl * fl;
      let fsl = luma(textureLoad(in_spec, fp, 0).rgb);
      s_m1 += fsl; s_m2 += fsl * fsl;
    }
  }
  let d_mean = d_m1 / 9.0;
  let d_std = sqrt(max(d_m2 / 9.0 - d_mean * d_mean, 0.0));
  let d_max_lum = d_mean + 3.0 * d_std + 0.1;

  let s_mean = s_m1 / 9.0;
  let s_std = sqrt(max(s_m2 / 9.0 - s_mean * s_mean, 0.0));
  let s_max_lum = s_mean + 3.0 * s_std + 0.1;

  let cl_raw = luma(cc_raw);
  var cc = cc_raw;
  if cl_raw > d_max_lum && cl_raw > 0.01 { cc = cc_raw * (d_max_lum / cl_raw); }
  let csl_raw = luma(cs_raw);
  var cs = cs_raw;
  if csl_raw > s_max_lum && csl_raw > 0.01 { cs = cs_raw * (s_max_lum / csl_raw); }

  let cl = luma(cc);
  let csl = luma(cs);

  // === 3x3 bilateral filter ===
  var d_sum = cc;  var d_wsum = 1.0;
  var s_sum = cs;  var s_wsum = 1.0;

  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      if dx == 0 && dy == 0 { continue; }
      let sp = clamp(px + vec2i(dx, dy), vec2i(0), sz - 1);
      let snd = textureLoad(gbuf_nd, sp, 0);
      let sd = textureLoad(in_color, sp, 0).rgb;
      let ss = textureLoad(in_spec, sp, 0).rgb;

      let sd_l = luma(sd);
      var sd_c = sd;
      if sd_l > d_max_lum && sd_l > 0.01 { sd_c = sd * (d_max_lum / sd_l); }
      let ss_l = luma(ss);
      var ss_c = ss;
      if ss_l > s_max_lum && ss_l > 0.01 { ss_c = ss * (s_max_lum / ss_l); }

      let wn = pow(max(dot(cn, snd.xyz), 0.0), 32.0);
      let wz = exp(-abs(cz - snd.w) / max(cz * 0.05, 1e-2));

      let d_wl = exp(-abs(cl - luma(sd_c)) / max(cl * 0.5 + 0.1, 0.01));
      d_sum += sd_c * (wn * wz * d_wl);
      d_wsum += wn * wz * d_wl;

      let s_wl = exp(-abs(csl - luma(ss_c)) / max(csl * 0.5 + 0.1, 0.01));
      s_sum += ss_c * (wn * wz * s_wl);
      s_wsum += wn * wz * s_wl;
    }
  }

  let center_hit_dist = textureLoad(in_spec, px, 0).a;
  textureStore(out_color, px, vec4f(d_sum / d_wsum, 1.0));
  textureStore(out_spec, px, vec4f(s_sum / s_wsum, center_hit_dist));
}

// === COMPOSITE: albedo * diffuse + specular, then AgX tonemap ===
@group(0) @binding(7) var composite_out: texture_storage_2d<rgba8unorm, write>;

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

  let denoised_diff = textureLoad(in_color, px, 0).rgb;
  let denoised_spec = textureLoad(in_spec, px, 0).rgb;
  let albedo = textureLoad(albedo_tex, px, 0).rgb;

  // Remodulate: albedo * diffuse_irradiance + specular_radiance
  var hdr = max(albedo, vec3f(0.02)) * denoised_diff + denoised_spec;
  var c = agx(hdr);
  c = pow(max(c, vec3f(0.0)), vec3f(1.0 / 2.2));
  textureStore(composite_out, px, vec4f(c, 1.0));
}
