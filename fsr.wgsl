// Upscaling + Sharpening (WebGPU Compute)
// Pass 1: EASU - Edge Adaptive Spatial Upscaling (CatmullRom, sharper than Lanczos)
// Pass 2: RCAS - Robust Contrast Adaptive Sharpening

struct FSRParams {
  input_size: vec2f,
  output_size: vec2f,
  sharpness: f32,
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var tex_sampler: sampler;
@group(0) @binding(3) var<uniform> params: FSRParams;
@group(0) @binding(4) var guide_nd: texture_2d<f32>;  // unused by EASU, needed for layout

// ============================================================
// Catmull-Rom weight (sharper than Lanczos, slight negative lobe)
// ============================================================
fn catmull_rom(x: f32) -> f32 {
  let ax = abs(x);
  if ax >= 2.0 { return 0.0; }
  let ax2 = ax * ax;
  let ax3 = ax2 * ax;
  if ax <= 1.0 { return 1.5 * ax3 - 2.5 * ax2 + 1.0; }
  return -0.5 * ax3 + 2.5 * ax2 - 4.0 * ax + 2.0;
}

// ============================================================
// EASU: CatmullRom 4x4 with edge-aware weighting
// Sharper than Lanczos: negative lobes enhance edges
// ============================================================
@compute @workgroup_size(16, 16)
fn easu(@builtin(global_invocation_id) gid: vec3u) {
  let out_px = vec2i(gid.xy);
  let out_size = vec2i(params.output_size);
  if out_px.x >= out_size.x || out_px.y >= out_size.y { return; }

  let ratio = params.input_size / params.output_size;
  let src_pos = (vec2f(gid.xy) + 0.5) * ratio;
  let src_center = floor(src_pos);
  let f = src_pos - src_center - 0.5;

  // Pre-sample center for edge detection
  let center_uv = (src_center + 0.5) / params.input_size;
  let center_col = textureSampleLevel(input_tex, tex_sampler, center_uv, 0.0).rgb;

  var color = vec3f(0.0);
  var tw = 0.0;

  for (var y = -1; y <= 2; y++) {
    let wy = catmull_rom(f32(y) - f.y);
    for (var x = -1; x <= 2; x++) {
      let wx = catmull_rom(f32(x) - f.x);
      let uv = clamp((src_center + vec2f(f32(x), f32(y)) + 0.5) / params.input_size, vec2f(0.001), vec2f(0.999));
      let s = textureSampleLevel(input_tex, tex_sampler, uv, 0.0).rgb;

      // Edge-aware: reduce ringing near strong edges
      let diff = max(abs(center_col.r - s.r), max(abs(center_col.g - s.g), abs(center_col.b - s.b)));
      let ew = exp(-diff * 6.0);
      let w = wx * wy * mix(1.0, ew, 0.4);

      color += s * w;
      tw += w;
    }
  }

  color = max(color / tw, vec3f(0.0));
  textureStore(output_tex, out_px, vec4f(color, 1.0));
}

// ============================================================
// RCAS: Robust Contrast-Adaptive Sharpening
// Based on AMD FidelityFX RCAS — lobe must be NEGATIVE to sharpen.
//
// Formula: out = (neighbors * lobe + center) / (4*lobe + 1)
// When lobe < 0: neighbors are subtracted, center is boosted → sharpening.
// When lobe = 0: out = center (no change).
//
// Adaptive limits prevent clipping:
//   peakLo = min_neighbors / (4 * max_neighbors)  → prevents output < 0
//   peakHi = (1 - max_neighbors) / (4 * min_neighbors) → prevents output > 1
// Lobe = -min(peakLo, peakHi) * sharpness, clamped to [-3/16, 0]
// ============================================================
@compute @workgroup_size(16, 16)
fn rcas(@builtin(global_invocation_id) gid: vec3u) {
  let px = vec2i(gid.xy);
  let size = vec2i(params.output_size);
  if px.x >= size.x || px.y >= size.y { return; }

  // 5-tap cross: center + NSWE neighbors
  let e = textureLoad(input_tex, px, 0).rgb;
  let n = textureLoad(input_tex, clamp(px + vec2i(0,-1), vec2i(0), size - 1), 0).rgb;
  let s = textureLoad(input_tex, clamp(px + vec2i(0, 1), vec2i(0), size - 1), 0).rgb;
  let w = textureLoad(input_tex, clamp(px + vec2i(-1,0), vec2i(0), size - 1), 0).rgb;
  let ea = textureLoad(input_tex, clamp(px + vec2i(1, 0), vec2i(0), size - 1), 0).rgb;

  // Per-channel min/max of the 4 neighbors
  let mn4 = min(min(n, s), min(w, ea));
  let mx4 = max(max(n, s), max(w, ea));

  // Per-channel peak limits: how far can we push before clipping?
  let peak_lo = mn4 / (4.0 * mx4 + vec3f(1e-6)); // keeps output >= 0
  let peak_hi = (1.0 - mx4) / (4.0 * mn4 + vec3f(1e-6)); // keeps output <= 1

  // Most conservative limit per channel
  let peak = min(peak_lo, peak_hi);

  // Single scalar: minimum across R,G,B (avoids color shifts)
  let max_lobe = min(peak.x, min(peak.y, peak.z));

  // Negative lobe = sharpening. Clamp to AMD's -3/16 limit.
  let lobe = max(-0.1875, -(max_lobe * params.sharpness));

  // Apply: neighbors get negative weight (subtracted), center gets boosted
  let result = (n + s + w + ea) * lobe + e;
  let final_c = result / (4.0 * lobe + 1.0);

  textureStore(output_tex, px, vec4f(saturate(final_c), 1.0));
}
