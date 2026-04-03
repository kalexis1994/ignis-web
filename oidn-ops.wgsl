// OIDN auxiliary operations: MaxPool, Upsample, I/O assembly

enable f16;

struct OpsParams {
  in_width: u32,
  in_height: u32,
  out_width: u32,
  out_height: u32,
  channels: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: OpsParams;
@group(0) @binding(1) var<storage, read> buf_in: array<f16>;
@group(0) @binding(2) var<storage, read_write> buf_out: array<f16>;

// ============================================================
// MaxPool 2x2 stride 2
// Dispatch: (ceil(out_width/16), ceil(out_height/16), channels)
// ============================================================
@compute @workgroup_size(16, 16)
fn maxpool2x2(
  @builtin(global_invocation_id) gid: vec3u
) {
  let ox = gid.x;
  let oy = gid.y;
  let ch = gid.z;
  if ox >= params.out_width || oy >= params.out_height || ch >= params.channels { return; }

  let iW = params.in_width;
  let iHW = params.in_height * iW;
  let ix = ox * 2u;
  let iy = oy * 2u;
  let base = ch * iHW;

  let v00 = buf_in[base + iy * iW + ix];
  let v10 = buf_in[base + iy * iW + min(ix + 1u, iW - 1u)];
  let v01 = buf_in[base + min(iy + 1u, params.in_height - 1u) * iW + ix];
  let v11 = buf_in[base + min(iy + 1u, params.in_height - 1u) * iW + min(ix + 1u, iW - 1u)];

  let oW = params.out_width;
  let oHW = params.out_height * oW;
  buf_out[ch * oHW + oy * oW + ox] = max(max(v00, v10), max(v01, v11));
}

// ============================================================
// Nearest-neighbor upsample 2x
// Dispatch: (ceil(out_width/16), ceil(out_height/16), channels)
// ============================================================
@compute @workgroup_size(16, 16)
fn upsample2x(
  @builtin(global_invocation_id) gid: vec3u
) {
  let ox = gid.x;
  let oy = gid.y;
  let ch = gid.z;
  if ox >= params.out_width || oy >= params.out_height || ch >= params.channels { return; }

  let iW = params.in_width;
  let iHW = params.in_height * iW;
  let ix = ox / 2u;
  let iy = oy / 2u;

  let oW = params.out_width;
  let oHW = params.out_height * oW;
  buf_out[ch * oHW + oy * oW + ox] = buf_in[ch * iHW + iy * iW + ix];
}

// ============================================================
// Input assembly: textures → 9-channel NCHW buffer
// Reads: color(3ch) + albedo(3ch) + normal(3ch) with HDR transfer
// Dispatch: (ceil(width/16), ceil(height/16), 1)
// ============================================================
@group(0) @binding(3) var diffuse_tex: texture_2d<f32>;   // noisy diffuse irradiance
@group(0) @binding(4) var albedo_tex: texture_2d<f32>;    // albedo.rgb + roughness.a
@group(0) @binding(5) var normal_tex: texture_2d<f32>;    // normal.xyz + depth.w
@group(0) @binding(7) var specular_tex: texture_2d<f32>;  // noisy specular radiance

@compute @workgroup_size(16, 16)
fn input_assembly(
  @builtin(global_invocation_id) gid: vec3u
) {
  let W = params.out_width;
  let H = params.out_height;
  if gid.x >= W || gid.y >= H { return; }

  let px = vec2i(gid.xy);
  let HW = H * W;
  let idx = gid.y * W + gid.x;

  // Read textures
  let diffuse = textureLoad(diffuse_tex, px, 0).rgb;
  let albedo = textureLoad(albedo_tex, px, 0).rgb;
  let nd = textureLoad(normal_tex, px, 0);
  let normal = nd.xyz;
  let specular = textureLoad(specular_tex, px, 0).rgb;

  // Combine beauty pass: OIDN expects albedo * diffuse_irradiance + specular
  var beauty = max(albedo, vec3f(0.02)) * diffuse + specular;

  // Inline firefly rejection: clamp luminance to 3σ of 3x3 neighborhood
  // Only clamps extremes — no blur, preserves noise pattern for OIDN
  let center_lum = dot(beauty, vec3f(0.2126, 0.7152, 0.0722));
  var m1 = 0.0; var m2 = 0.0;
  for (var fy = -1; fy <= 1; fy++) {
    for (var fx = -1; fx <= 1; fx++) {
      let fp = clamp(px + vec2i(fx, fy), vec2i(0), vec2i(i32(W)-1, i32(H)-1));
      let fd = textureLoad(diffuse_tex, fp, 0).rgb;
      let fs = textureLoad(specular_tex, fp, 0).rgb;
      let fl = dot(max(textureLoad(albedo_tex, fp, 0).rgb, vec3f(0.02)) * fd + fs, vec3f(0.2126, 0.7152, 0.0722));
      m1 += fl; m2 += fl * fl;
    }
  }
  let mean = m1 / 9.0;
  let std_dev = sqrt(max(m2 / 9.0 - mean * mean, 0.0));
  let max_lum = mean + 2.0 * std_dev + 0.1;
  if center_lum > max_lum && center_lum > 0.01 {
    beauty *= max_lum / center_lum;
  }

  // LDR model: expects sRGB gamma-encoded [0,1] (like a final render)
  let tonemapped = beauty / (beauty + 1.0);          // Reinhard tonemap → [0,1)
  let tf = pow(max(tonemapped, vec3f(0.0)), vec3f(1.0 / 2.2)); // sRGB gamma

  // Write 9 channels NCHW: color(3) + albedo(3) + normal(3)
  buf_out[0u * HW + idx] = f16(tf.x);
  buf_out[1u * HW + idx] = f16(tf.y);
  buf_out[2u * HW + idx] = f16(tf.z);
  buf_out[3u * HW + idx] = f16(albedo.x);
  buf_out[4u * HW + idx] = f16(albedo.y);
  buf_out[5u * HW + idx] = f16(albedo.z);
  buf_out[6u * HW + idx] = f16(normal.x * 0.5 + 0.5);
  buf_out[7u * HW + idx] = f16(normal.y * 0.5 + 0.5);
  buf_out[8u * HW + idx] = f16(normal.z * 0.5 + 0.5);
}

// ============================================================
// Output extraction: 3-channel NCHW buffer → texture
// Dispatch: (ceil(width/16), ceil(height/16), 1)
// ============================================================
@group(0) @binding(6) var out_tex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(16, 16)
fn output_extraction(
  @builtin(global_invocation_id) gid: vec3u
) {
  let W = params.in_width;
  let H = params.in_height;
  if gid.x >= W || gid.y >= H { return; }

  let HW = H * W;
  let idx = gid.y * W + gid.x;

  // Read 3 output channels (denoised beauty pass in transfer space)
  let r = f32(buf_in[0u * HW + idx]);
  let g = f32(buf_in[1u * HW + idx]);
  let b = f32(buf_in[2u * HW + idx]);

  // Inverse: sRGB gamma → linear → inverse Reinhard → HDR
  let srgb = clamp(vec3f(r, g, b), vec3f(0.0), vec3f(1.0));
  let linear = pow(srgb, vec3f(2.2));                    // inverse gamma
  let beauty = linear / max(1.0 - linear, vec3f(0.001)); // inverse Reinhard
  // beauty /= 0.5; // undo exposure — skip, let composite exposure handle it

  // Write full denoised beauty pass — composite handles OIDN mode without remodulation
  textureStore(out_tex, vec2i(gid.xy), vec4f(beauty, 1.0));
}

// ============================================================
// Temporal blend: mix current denoised with previous for stability
// Reads current from one texture, history from another, writes blended
// Dispatch: (ceil(width/16), ceil(height/16), 1)
// Uses params.channels as blend alpha × 1000 (integer encoding)
// ============================================================
@group(0) @binding(8) var current_tex: texture_2d<f32>;
@group(0) @binding(9) var history_tex: texture_2d<f32>;
@group(0) @binding(10) var blend_out: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(16, 16)
fn temporal_blend(
  @builtin(global_invocation_id) gid: vec3u
) {
  let W = params.in_width;
  let H = params.in_height;
  if gid.x >= W || gid.y >= H { return; }
  let px = vec2i(gid.xy);

  let current = textureLoad(current_tex, px, 0).rgb;
  let history = textureLoad(history_tex, px, 0).rgb;

  // Alpha from params.channels (encoded as alpha * 1000)
  let base_alpha = f32(params.channels) / 1000.0;

  // Neighborhood AABB clamp: use 3x3 min/max of current frame to reject stale history
  // This prevents ghosting artifacts from misaligned history pixels
  var mn = current;
  var mx = current;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let sp = clamp(px + vec2i(dx, dy), vec2i(0), vec2i(i32(W) - 1, i32(H) - 1));
      let s = textureLoad(current_tex, sp, 0).rgb;
      mn = min(mn, s);
      mx = max(mx, s);
    }
  }

  // Expand AABB slightly to allow noise variation
  let aabb_size = mx - mn;
  let margin = aabb_size * 0.25 + 0.01;
  let clamped_history = clamp(history, mn - margin, mx + margin);

  // How much was history clamped? More clamping → more weight to current
  let clamp_dist = length(history - clamped_history);
  let clamp_factor = min(clamp_dist * 5.0, 1.0); // 0=no clamp, 1=fully rejected
  let alpha = max(base_alpha, clamp_factor);

  let blended = mix(clamped_history, current, alpha);
  textureStore(blend_out, px, vec4f(blended, 1.0));
}
