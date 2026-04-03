// OIDN auxiliary operations: MaxPool, Upsample, I/O assembly — NHWC layout

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
// MaxPool 2x2 stride 2 — NHWC
// Each thread handles one output pixel, ALL channels at once
// Dispatch: (ceil(out_width/8), ceil(out_height/8), 1)
// ============================================================
@compute @workgroup_size(8, 8)
fn maxpool2x2(
  @builtin(global_invocation_id) gid: vec3u
) {
  let ox = gid.x;
  let oy = gid.y;
  if ox >= params.out_width || oy >= params.out_height { return; }

  let iW = params.in_width;
  let C = params.channels;
  let ix = ox * 2u;
  let iy = oy * 2u;
  let iy1 = min(iy + 1u, params.in_height - 1u);
  let ix1 = min(ix + 1u, iW - 1u);

  // NHWC: pixel at (y,x) has channels at (y*W+x)*C
  let p00 = (iy  * iW + ix ) * C;
  let p10 = (iy  * iW + ix1) * C;
  let p01 = (iy1 * iW + ix ) * C;
  let p11 = (iy1 * iW + ix1) * C;

  let oBase = (oy * params.out_width + ox) * C;
  for (var c = 0u; c < C; c++) {
    buf_out[oBase + c] = max(max(buf_in[p00+c], buf_in[p10+c]), max(buf_in[p01+c], buf_in[p11+c]));
  }
}

// ============================================================
// Nearest-neighbor upsample 2x — NHWC
// Each thread handles one output pixel, ALL channels
// Dispatch: (ceil(out_width/8), ceil(out_height/8), 1)
// ============================================================
@compute @workgroup_size(8, 8)
fn upsample2x(
  @builtin(global_invocation_id) gid: vec3u
) {
  let ox = gid.x;
  let oy = gid.y;
  if ox >= params.out_width || oy >= params.out_height { return; }

  let C = params.channels;
  let iBase = (oy / 2u * params.in_width + ox / 2u) * C;
  let oBase = (oy * params.out_width + ox) * C;
  for (var c = 0u; c < C; c++) {
    buf_out[oBase + c] = buf_in[iBase + c];
  }
}

// ============================================================
// Input assembly: textures → 9-channel NHWC buffer
// Dispatch: (ceil(width/8), ceil(height/8), 1)
// ============================================================
@group(0) @binding(3) var diffuse_tex: texture_2d<f32>;
@group(0) @binding(4) var albedo_tex: texture_2d<f32>;
@group(0) @binding(5) var normal_tex: texture_2d<f32>;
@group(0) @binding(7) var specular_tex: texture_2d<f32>;

@compute @workgroup_size(8, 8)
fn input_assembly(
  @builtin(global_invocation_id) gid: vec3u
) {
  let W = params.out_width;
  let H = params.out_height;
  if gid.x >= W || gid.y >= H { return; }

  let px = vec2i(gid.xy);

  let diffuse = textureLoad(diffuse_tex, px, 0).rgb;
  let albedo = textureLoad(albedo_tex, px, 0).rgb;
  let nd = textureLoad(normal_tex, px, 0);
  let normal = nd.xyz;
  let specular = textureLoad(specular_tex, px, 0).rgb;

  let beauty = max(albedo, vec3f(0.02)) * diffuse + specular;
  let exposed = beauty * 0.5;
  let tonemapped = exposed / (exposed + 1.0);
  let tf = pow(max(tonemapped, vec3f(0.0)), vec3f(1.0 / 2.2));

  // Write 9 channels NHWC: all channels at pixel contiguous
  let base = (gid.y * W + gid.x) * 9u;
  buf_out[base     ] = f16(tf.x);
  buf_out[base + 1u] = f16(tf.y);
  buf_out[base + 2u] = f16(tf.z);
  buf_out[base + 3u] = f16(albedo.x);
  buf_out[base + 4u] = f16(albedo.y);
  buf_out[base + 5u] = f16(albedo.z);
  buf_out[base + 6u] = f16(normal.x * 0.5 + 0.5);
  buf_out[base + 7u] = f16(normal.y * 0.5 + 0.5);
  buf_out[base + 8u] = f16(normal.z * 0.5 + 0.5);
}

// ============================================================
// Output extraction: 3-channel NHWC buffer → texture
// Dispatch: (ceil(width/8), ceil(height/8), 1)
// ============================================================
@group(0) @binding(6) var out_tex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn output_extraction(
  @builtin(global_invocation_id) gid: vec3u
) {
  let W = params.in_width;
  let H = params.in_height;
  if gid.x >= W || gid.y >= H { return; }

  // NHWC: 3 channels per pixel contiguous
  let base = (gid.y * W + gid.x) * 3u;
  let r = f32(buf_in[base]);
  let g = f32(buf_in[base + 1u]);
  let b = f32(buf_in[base + 2u]);

  let srgb = clamp(vec3f(r, g, b), vec3f(0.0), vec3f(1.0));
  let linear = pow(srgb, vec3f(2.2));
  let beauty = linear / max(1.0 - linear, vec3f(0.001));

  textureStore(out_tex, vec2i(gid.xy), vec4f(beauty, 1.0));
}
