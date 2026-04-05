// ============================================================
// ReBLUR Denoiser — Full port from NVIDIA NRD (v4.17)
// Source: github.com/NVIDIAGameWorks/RayTracingDenoiser
//
// Pipeline: PrePass → TemporalAccumulation → HistoryFix → Blur → PostBlur → TemporalStabilization
// ============================================================

// ============================================================
// CONSTANTS (from REBLUR_Config.hlsli + NRD.hlsli)
// ============================================================
const NRD_EPS: f32 = 1e-6;
const NRD_INF: f32 = 1e6;
const NRD_FP16_MAX: f32 = 65504.0;
const NRD_PI: f32 = 3.14159265;
const NRD_NORMAL_ENCODING_ERROR: f32 = 0.0;
const NRD_EXP_WEIGHT_DEFAULT_SCALE: f32 = 3.0;
const NRD_MAX_PERCENT_OF_LOBE_VOLUME: f32 = 0.75;
const NRD_ROUGHNESS_SENSITIVITY: f32 = 0.01;
const NRD_CATROM_SHARPNESS: f32 = 0.5;

const REBLUR_MAX_ACCUM_FRAME_NUM: f32 = 63.0;
const REBLUR_PRE_PASS_FRACTION_SCALE: f32 = 2.0;
const REBLUR_BLUR_FRACTION_SCALE: f32 = 1.0;
const REBLUR_POST_BLUR_FRACTION_SCALE: f32 = 0.5;
const REBLUR_PRE_PASS_RADIUS_SCALE: f32 = 1.0;
const REBLUR_BLUR_RADIUS_SCALE: f32 = 1.0;
const REBLUR_POST_BLUR_RADIUS_SCALE: f32 = 2.0;
const REBLUR_FIREFLY_SUPPRESSOR_MAX_RELATIVE_INTENSITY: f32 = 38.0;
const REBLUR_FIREFLY_SUPPRESSOR_RADIUS_SCALE: f32 = 0.1;
const REBLUR_ANTI_FIREFLY_SIGMA_SCALE: f32 = 2.0;
const REBLUR_HISTORY_FIX_FILTER_RADIUS: i32 = 2;
const REBLUR_ALMOST_ZERO_ANGLE: f32 = 0.01745; // cos(89°)
const REBLUR_MAX_PERCENT_OF_LOBE_VOLUME_FOR_PRE_PASS: f32 = 0.3;

// Poisson disk 8 taps (official NRD g_Poisson8, .z = length)
const POISSON8 = array<vec3f, 8>(
  vec3f(-0.4706069, -0.4427112, 0.6461146),
  vec3f(-0.9057375,  0.3003471, 0.9542373),
  vec3f(-0.3487388,  0.4037880, 0.5335386),
  vec3f( 0.1023042,  0.6439373, 0.6520134),
  vec3f( 0.5699277,  0.3513750, 0.6695386),
  vec3f( 0.2939128, -0.1131226, 0.3149309),
  vec3f( 0.7836658, -0.4208784, 0.8895339),
  vec3f( 0.1564120, -0.8198990, 0.8346850),
);

// ============================================================
// UNIFORM BUFFER (matches NRD REBLUR_SHARED_CONSTANTS subset)
// ============================================================
struct ReblurParams {
  // Matrices (column-major mat4x4)
  world_to_clip: mat4x4f,
  view_to_world: mat4x4f,
  world_to_view_prev: mat4x4f,
  world_to_clip_prev: mat4x4f,
  world_prev_to_world: mat4x4f,

  // Frustum params: (x * viewZ, y * viewZ, z * viewZ, w * viewZ) reconstruct view position
  frustum: vec4f,
  frustum_prev: vec4f,
  camera_delta: vec4f,     // current - previous camera position
  hit_dist_params: vec4f,  // (A, B, C, 0) for normalization: (A + |viewZ|*B) * lerp(C, 1, smc)

  // Rotators (cos, sin, -sin, cos) for per-frame Poisson rotation
  rotator_pre: vec4f,
  rotator_blur: vec4f,
  rotator_post: vec4f,

  // Resolution
  rect_size: vec2f,
  rect_size_inv: vec2f,
  rect_size_prev: vec2f,
  resource_size_inv_prev: vec2f,

  // Jitter
  jitter: vec2f,

  // Scalar params
  disocclusion_threshold: f32,
  plane_dist_sensitivity: f32,
  min_blur_radius: f32,
  max_blur_radius: f32,
  diff_prepass_blur_radius: f32,
  spec_prepass_blur_radius: f32,
  max_accumulated_frame_num: f32,
  max_fast_accumulated_frame_num: f32,
  lobe_angle_fraction: f32,
  roughness_fraction: f32,
  history_fix_frame_num: f32,
  history_fix_stride: f32,
  stabilization_strength: f32,
  anti_firefly: f32,
  antilag_power: f32,
  antilag_threshold: f32,
  framerate_scale: f32,
  denoising_range: f32,
  min_rect_dim_mul_unproject: f32,
  unproject: f32,
  ortho_mode: f32,
  frame_index: u32,
  view_z_scale: f32,
  min_hit_dist_weight: f32,
  firefly_suppressor_min_relative_scale: f32,
  fast_history_clamping_sigma_scale: f32,
};

// ============================================================
// BINDINGS — matches existing denoise.wgsl layout (dnAtrousLayout)
// This allows reusing existing bind groups and textures.
// group(0): uniform(0), in_color(1), out_color(2), gbuf_nd(3), in_spec(4), out_spec(5), albedo(6)
// ============================================================
@group(0) @binding(0) var<uniform> gp: ReblurParams;
@group(0) @binding(1) var in_diff: texture_2d<f32>;               // diffuse input
@group(0) @binding(2) var out_diff: texture_storage_2d<rgba16float, write>; // diffuse output
@group(0) @binding(3) var in_normal_roughness: texture_2d<f32>;   // gbuf: normal.xyz + depth.w (reused for viewZ)
@group(0) @binding(4) var in_spec: texture_2d<f32>;               // specular input
@group(0) @binding(5) var out_spec: texture_storage_2d<rgba16float, write>; // specular output
@group(0) @binding(6) var in_albedo: texture_2d<f32>;             // albedo.rgb + roughness.a

// For temporal passes that need history + sampler, use group(1)
@group(1) @binding(0) var history_diff: texture_2d<f32>;
@group(1) @binding(1) var history_spec: texture_2d<f32>;
@group(1) @binding(2) var linear_clamp: sampler;
@group(1) @binding(3) var in_mv: texture_2d<f32>;                 // motion vectors (depth_tex reused)

// Aliases for clarity in helper functions
fn read_viewz(px: vec2i) -> f32 { return textureLoad(in_normal_roughness, px, 0).w; }
fn read_roughness(px: vec2i) -> f32 { return textureLoad(in_albedo, px, 0).a; }
fn read_normal(px: vec2i) -> vec3f { return textureLoad(in_normal_roughness, px, 0).xyz; }

// ============================================================
// HELPER FUNCTIONS (from NRD.hlsli + Common.hlsli)
// ============================================================

fn luma(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }

fn linear_to_ycocg(c: vec3f) -> vec3f {
  return vec3f(
    dot(c, vec3f(0.25, 0.5, 0.25)),
    dot(c, vec3f(0.5, 0.0, -0.5)),
    dot(c, vec3f(-0.25, 0.5, -0.25))
  );
}

fn ycocg_to_linear(c: vec3f) -> vec3f {
  let t = c.x - c.z;
  return max(vec3f(t + c.y, c.x + c.z, t - c.y), vec3f(0.0));
}

fn get_luma(input: vec4f) -> f32 {
  return input.x; // Y channel in YCoCg
}

fn change_luma(input: vec4f, new_luma: f32) -> vec4f {
  let scale = (new_luma + NRD_EPS) / (get_luma(input) + NRD_EPS);
  return vec4f(input.xyz * scale, input.w);
}

fn get_spec_magic_curve(roughness: f32) -> f32 {
  let f = 1.0 - exp2(-200.0 * roughness * roughness);
  return f * pow(clamp(roughness, 0.0, 1.0), 0.25);
}

fn get_hit_dist_normalization(viewZ: f32, roughness: f32) -> f32 {
  let smc = get_spec_magic_curve(roughness);
  let p = gp.hit_dist_params;
  return (p.x + abs(viewZ) * p.y) * mix(p.z, 1.0, smc);
}

fn get_frustum_size(viewZ: f32) -> f32 {
  return gp.min_rect_dim_mul_unproject * mix(viewZ, 1.0, abs(gp.ortho_mode));
}

fn get_hit_dist_factor(hitDist: f32, frustumSize: f32) -> f32 {
  return clamp(hitDist / frustumSize, 0.0, 1.0);
}

fn pixel_radius_to_world(pixelRadius: f32, viewZ: f32) -> f32 {
  return pixelRadius * gp.unproject * mix(viewZ, 1.0, abs(gp.ortho_mode));
}

fn reconstruct_view_pos(uv: vec2f, viewZ: f32) -> vec3f {
  let p = gp.frustum;
  return vec3f((uv.x * p.x + p.z) * viewZ, (uv.y * p.y + p.w) * viewZ, viewZ);
}

fn reconstruct_view_pos_prev(uv: vec2f, viewZ: f32) -> vec3f {
  let p = gp.frustum_prev;
  return vec3f((uv.x * p.x + p.z) * viewZ, (uv.y * p.y + p.w) * viewZ, viewZ);
}

fn rotate_vector_m(m: mat4x4f, v: vec3f) -> vec3f {
  return vec3f(
    dot(v, vec3f(m[0].x, m[1].x, m[2].x)),
    dot(v, vec3f(m[0].y, m[1].y, m[2].y)),
    dot(v, vec3f(m[0].z, m[1].z, m[2].z))
  );
}

fn rotate_vector_inv_m(m: mat4x4f, v: vec3f) -> vec3f {
  return vec3f(dot(v, m[0].xyz), dot(v, m[1].xyz), dot(v, m[2].xyz));
}

fn affine_transform(m: mat4x4f, p: vec3f) -> vec3f {
  return vec3f(
    dot(vec4f(p, 1.0), vec4f(m[0].x, m[1].x, m[2].x, m[3].x)),
    dot(vec4f(p, 1.0), vec4f(m[0].y, m[1].y, m[2].y, m[3].y)),
    dot(vec4f(p, 1.0), vec4f(m[0].z, m[1].z, m[2].z, m[3].z))
  );
}

fn project_to_screen(m: mat4x4f, X: vec3f) -> vec2f {
  let clip = m * vec4f(X, 1.0);
  var ndc = clip.xy / clip.w;
  ndc.y = -ndc.y;
  return ndc * 0.5 + 0.5;
}

fn get_view_vector(X: vec3f) -> vec3f {
  if gp.ortho_mode == 0.0 { return normalize(-X); }
  return vec3f(0.0, 0.0, -1.0);
}

fn is_in_screen(uv: vec2f) -> bool {
  return all(uv > vec2f(0.0)) && all(uv < vec2f(1.0));
}

fn mirror_uv(uv: vec2f) -> vec2f {
  return 1.0 - abs(1.0 - fract(uv * 0.5) * 2.0);
}

// Rotator: (cos, sin, -sin, cos) applied to 2D offset
fn rotate_2d(rotator: vec4f, v: vec2f) -> vec2f {
  return vec2f(v.x * rotator.x + v.y * rotator.z, v.x * rotator.y + v.y * rotator.w);
}

fn get_gaussian_weight(r: f32) -> f32 {
  return exp(-0.66 * r * r);
}

// ExpApprox for negative x: 1/(x²-x+1)
fn exp_approx(x: f32) -> f32 {
  return 1.0 / (x * x - x + 1.0);
}

fn compute_exponential_weight(x: f32, px: f32, py: f32) -> f32 {
  return exp_approx(-NRD_EXP_WEIGHT_DEFAULT_SCALE * abs(x * px + py));
}

fn compute_non_exponential_weight(x: f32, px: f32, py: f32) -> f32 {
  let v = abs(x * px + py);
  return clamp(1.0 - v, 0.0, 1.0) * clamp(1.0 - v, 0.0, 1.0) * (3.0 - 2.0 * clamp(1.0 - v, 0.0, 1.0)); // smoothstep(1,0,v)
}

fn compute_weight(x: f32, px: f32, py: f32) -> f32 {
  return compute_exponential_weight(x, px, py);
}

// GGX dominant direction
fn get_specular_dominant_factor(NoV: f32, roughness: f32) -> f32 {
  let a = 0.298475 * log(39.4115 - 39.0029 * roughness);
  return clamp(pow(clamp(1.0 - NoV, 0.0, 1.0), 10.8649) * (1.0 - a) + a, 0.0, 1.0);
}

fn get_specular_dominant_direction(N: vec3f, V: vec3f, roughness: f32) -> vec4f {
  let NoV = abs(dot(N, V));
  let dominantFactor = get_specular_dominant_factor(NoV, roughness);
  let R = reflect(-V, N);
  let D = normalize(mix(N, R, dominantFactor));
  return vec4f(D, dominantFactor);
}

fn get_specular_lobe_tan_half_angle(roughness: f32, percentOfVolume: f32) -> f32 {
  // Simplified: tan(halfAngle) ≈ roughness * percentOfVolume for practical use
  return roughness * roughness * percentOfVolume * 3.0;
}

// GetXvirtual — thin lens equation (Common.hlsli:401-439)
fn get_x_virtual(hitDist: f32, curvature: f32, X: vec3f, Xprev: vec3f, N: vec3f, V: vec3f, roughness: f32) -> vec3f {
  let D = get_specular_dominant_direction(N, V, roughness);

  // Build basis around N
  var T = vec3f(1.0, 0.0, 0.0);
  if abs(N.y) < 0.999 { T = normalize(cross(vec3f(0.0, 1.0, 0.0), N)); }
  else { T = normalize(cross(vec3f(1.0, 0.0, 0.0), N)); }
  let B = cross(N, T);

  // Object position in reflector basis
  let reflectionRay = D.xyz * hitDist;
  var O = vec3f(dot(reflectionRay, T), dot(reflectionRay, B), dot(reflectionRay, N));
  O.z = -O.z; // sign convention

  // Magnification from thin lens
  var mag = 1.0 / (2.0 * curvature * O.z - 1.0);

  // Silhouette fix
  let NoV = abs(dot(N, V));
  var f = length(X);
  f *= clamp(1.0 - NoV, 0.0, 1.0);
  f *= max(curvature, 0.0);
  f = 1.0 / (1.0 + f);
  mag *= f;

  // Image position
  let I = O * mag;
  let elongation = D.w * length(I);

  // Closeness to surface
  let closeness = clamp(elongation / (hitDist + NRD_EPS), 0.0, 1.0);
  let x = mix(Xprev, X, closeness);

  return x + V * elongation * sign(mag);
}

// Disocclusion threshold
fn get_disocclusion_threshold(frustumSize: f32, NoV: f32) -> f32 {
  return frustumSize * clamp(gp.disocclusion_threshold / max(0.05, NoV), 0.0, 1.0);
}

// Weight parameters (from Common.hlsli)
fn get_normal_weight_param(nonLinearAccumSpeed: f32, roughness: f32) -> f32 {
  let percentOfVolume = NRD_MAX_PERCENT_OF_LOBE_VOLUME * mix(clamp(gp.lobe_angle_fraction, 0.0, 1.0), 1.0, nonLinearAccumSpeed);
  let tanHalfAngle = get_specular_lobe_tan_half_angle(roughness, percentOfVolume);
  let angle = max(atan(tanHalfAngle), NRD_NORMAL_ENCODING_ERROR + 0.01);
  return 1.0 / angle;
}

fn get_geometry_weight_params(frustumSize: f32, Xv: vec3f, Nv: vec3f) -> vec2f {
  let norm = gp.plane_dist_sensitivity * frustumSize;
  let a = 1.0 / max(norm, NRD_EPS);
  let b = dot(Nv, Xv) * a;
  return vec2f(a, -b);
}

fn get_hit_distance_weight_params(hitDist: f32, nonLinearAccumSpeed: f32) -> vec2f {
  let a = 1.0 / max(nonLinearAccumSpeed, NRD_EPS);
  let b = hitDist * a;
  return vec2f(a, -b);
}

fn get_roughness_weight_params(roughness: f32, fraction: f32) -> vec2f {
  let a = 1.0 / mix(NRD_ROUGHNESS_SENSITIVITY, 1.0, clamp(roughness * fraction, 0.0, 1.0));
  let b = roughness * a;
  return vec2f(a, -b);
}

// NonLinear accumulation speed: 1/(1+min(accumSpeed, max))
fn get_non_linear_accum_speed(accumSpeed: f32, maxAccumSpeed: f32) -> f32 {
  return 1.0 / (1.0 + min(accumSpeed, maxAccumSpeed));
}

// Temporal accumulation parameters (REBLUR_Common:326-334)
fn get_temporal_accum_params(footprintQuality: f32, accumSpeed: f32, antilag: f32) -> vec2f {
  let nonLinear = get_non_linear_accum_speed(accumSpeed, gp.max_accumulated_frame_num);
  var w = footprintQuality;
  w *= 1.0 - nonLinear; // = accumSpeed / (1+accumSpeed)
  w *= antilag;
  return vec2f(w, 1.0 + 3.0 * gp.framerate_scale * w);
}

// AntiLag (REBLUR_Common:243-280, mode 2)
fn compute_antilag(h: f32, a: f32, sigma: f32, accumSpeed: f32) -> f32 {
  let s = sigma * gp.antilag_power;
  let magic = gp.antilag_threshold * gp.framerate_scale * gp.framerate_scale;
  let hc = clamp(h, a - s, a + s);
  let d = abs(h - hc) / (max(h, hc) + NRD_EPS);
  return 1.0 / (1.0 + d * accumSpeed / max(magic, NRD_EPS));
}

// Pack/Unpack internal data
fn pack_internal_data(diffAccum: f32, specAccum: f32) -> u32 {
  let d = u32(clamp(min(diffAccum + 1.0, REBLUR_MAX_ACCUM_FRAME_NUM) / REBLUR_MAX_ACCUM_FRAME_NUM * 255.0, 0.0, 255.0));
  let s = u32(clamp(min(specAccum + 1.0, REBLUR_MAX_ACCUM_FRAME_NUM) / REBLUR_MAX_ACCUM_FRAME_NUM * 255.0, 0.0, 255.0));
  return d | (s << 8u);
}

fn unpack_internal_data(p: u32) -> vec2f {
  let d = f32(p & 0xFFu) / 255.0 * REBLUR_MAX_ACCUM_FRAME_NUM;
  let s = f32((p >> 8u) & 0xFFu) / 255.0 * REBLUR_MAX_ACCUM_FRAME_NUM;
  return vec2f(round(d), round(s));
}

fn pack_data1(diffAccum: f32, specAccum: f32) -> vec2f {
  return clamp(vec2f(round(diffAccum), round(specAccum)) / REBLUR_MAX_ACCUM_FRAME_NUM, vec2f(0.0), vec2f(1.0));
}

fn unpack_data1(p: vec2f) -> vec2f {
  return round(p * REBLUR_MAX_ACCUM_FRAME_NUM);
}

// Mix history and current (REBLUR_Common:199-206)
fn mix_history_and_current(history: vec4f, current: vec4f, f: f32) -> vec4f {
  var r: vec4f;
  r = vec4f(mix(history.xyz, current.xyz, f), mix(history.w, current.w, max(f, 0.02)));
  return r;
}

// ============================================================
// SHARED SPATIAL FILTER (from REBLUR_Common_SpatialFilter.hlsli)
// Used by PrePass, Blur, PostBlur with different scale params
// ============================================================
fn spatial_filter(
  px: vec2i, sz: vec2i,
  radiusScale: f32, fractionScale: f32,
  nonLinearAccumSpeed: f32, is_prepass: bool,
  rotator: vec4f
) -> array<vec4f, 2> {
  let pixelUv = (vec2f(px) + 0.5) * gp.rect_size_inv;
  let viewZ = read_viewz(px);
  let nr = textureLoad(in_normal_roughness, px, 0);
  let N = nr.xyz;
  let roughness = nr.w;
  let Nv = rotate_vector_inv_m(gp.view_to_world, N);
  let Xv = reconstruct_view_pos(pixelUv, viewZ);
  let Vv = get_view_vector(Xv);
  let NoV = abs(dot(Nv, Vv));
  let frustumSize = get_frustum_size(viewZ);

  var diff_result = textureLoad(in_diff, px, 0);
  var spec_result = textureLoad(in_spec, px, 0);
  var sum_d = 1.0;
  var sum_s = 1.0;

  // Hit distance factor
  // Diffuse alpha stores history_len (not hitDist), so use hitDistFactor=1.0 for diffuse
  let hitDistFactor_d = 1.0;
  // Specular alpha stores actual hit distance
  let hitDistScale_s = get_hit_dist_normalization(viewZ, roughness);
  let hitDist_s = spec_result.w * hitDistScale_s;
  let hitDistFactor_s = get_hit_dist_factor(hitDist_s, frustumSize);
  let smc = get_spec_magic_curve(roughness);

  // Blur radius
  var areaFactor_d = hitDistFactor_d;
  var areaFactor_s = hitDistFactor_s;
  if !is_prepass {
    areaFactor_d *= nonLinearAccumSpeed;
    areaFactor_s *= nonLinearAccumSpeed;
  }

  // DEBUG: force large radius to verify spatial filter works
  var blurRadius_d = 30.0; //radiusScale * sqrt(areaFactor_d) * gp.max_blur_radius;
  var blurRadius_s = 30.0 * max(smc, 0.3); //radiusScale * sqrt(areaFactor_s) * gp.max_blur_radius * smc;
  blurRadius_d = max(blurRadius_d, gp.min_blur_radius);
  blurRadius_s = max(blurRadius_s, gp.min_blur_radius * smc);

  if is_prepass {
    blurRadius_d = min(blurRadius_d, gp.diff_prepass_blur_radius);
    blurRadius_s = min(blurRadius_s, gp.spec_prepass_blur_radius);
  }

  // Weight parameters
  let geometryWeightParams = get_geometry_weight_params(frustumSize, Xv, Nv);
  let normalWeightParam_d = get_normal_weight_param(nonLinearAccumSpeed, 1.0) / fractionScale;
  let normalWeightParam_s = get_normal_weight_param(nonLinearAccumSpeed, roughness) / fractionScale;
  // Diffuse: no hitDist available (alpha = history_len), skip hitDist weighting
  let hitDistWeightParams_d = vec2f(0.0, 0.0); // disabled: always returns weight=1
  let hitDistWeightParams_s = get_hit_distance_weight_params(spec_result.w, nonLinearAccumSpeed);
  let roughnessWeightParams = get_roughness_weight_params(roughness, gp.roughness_fraction * fractionScale);
  let minHitDistWeight = gp.min_hit_dist_weight * fractionScale * smc;

  // Per-pixel rotation (Bayer4x4 + frame) — eliminates workgroup boundary artifacts
  let bayer = fract(f32((u32(px.x) & 3u) * 4u + (u32(px.y) & 3u)) / 16.0 + f32(gp.frame_index % 16u) / 16.0);
  let px_angle = bayer * 6.2832;
  let px_rot = vec4f(cos(px_angle), sin(px_angle), -sin(px_angle), cos(px_angle));
  // Combine per-pixel with per-frame rotator
  let final_rot = vec4f(
    px_rot.x * rotator.x - px_rot.y * rotator.y,
    px_rot.x * rotator.y + px_rot.y * rotator.x,
    -(px_rot.x * rotator.y + px_rot.y * rotator.x),
    px_rot.x * rotator.x - px_rot.y * rotator.y
  );

  // Screen-space sampling
  let skew_d = gp.rect_size_inv * blurRadius_d;
  let skew_s = gp.rect_size_inv * blurRadius_s;

  // 8 Poisson taps
  for (var n = 0u; n < 8u; n++) {
    let offset = POISSON8[n];
    let rotated = rotate_2d(final_rot, offset.xy);

    // Diffuse tap
    let uv_d = pixelUv + rotated * skew_d;
    var w_d = get_gaussian_weight(offset.z);
    if any(uv_d != clamp(uv_d, vec2f(0.0), vec2f(1.0))) {
      w_d = 1.0; // mirror: offset.z invalid
    }
    let uv_d_safe = mirror_uv(uv_d);
    let pos_d = vec2i(uv_d_safe * gp.rect_size);

    let zs_d = read_viewz(clamp(pos_d, vec2i(0), sz - 1));
    let Xvs_d = reconstruct_view_pos(vec2f(pos_d) * gp.rect_size_inv + 0.5 * gp.rect_size_inv, zs_d);
    let Ns_d = textureLoad(in_normal_roughness, clamp(pos_d, vec2i(0), sz - 1), 0).xyz;

    let angle_d = acos(clamp(dot(N, Ns_d), -1.0, 1.0));
    let NoX_d = dot(Nv, Xvs_d);

    w_d *= compute_weight(NoX_d, geometryWeightParams.x, geometryWeightParams.y);
    w_d *= compute_weight(angle_d, normalWeightParam_d, 0.0);
    w_d = select(w_d, 0.0, zs_d > gp.denoising_range);

    let sd = textureLoad(in_diff, clamp(pos_d, vec2i(0), sz - 1), 0);
    w_d *= minHitDistWeight + compute_exponential_weight(sd.w, hitDistWeightParams_d.x, hitDistWeightParams_d.y);

    diff_result += sd * w_d;
    sum_d += w_d;

    // Specular tap
    let uv_s = pixelUv + rotated * skew_s;
    var w_s = get_gaussian_weight(offset.z);
    let uv_s_safe = mirror_uv(uv_s);
    let pos_s = vec2i(uv_s_safe * gp.rect_size);

    let zs_s = read_viewz(clamp(pos_s, vec2i(0), sz - 1));
    let Xvs_s = reconstruct_view_pos(vec2f(pos_s) * gp.rect_size_inv + 0.5 * gp.rect_size_inv, zs_s);
    let nrs_s = textureLoad(in_normal_roughness, clamp(pos_s, vec2i(0), sz - 1), 0);

    let angle_s = acos(clamp(dot(N, nrs_s.xyz), -1.0, 1.0));
    let NoX_s = dot(Nv, Xvs_s);

    w_s *= compute_weight(NoX_s, geometryWeightParams.x, geometryWeightParams.y);
    w_s *= compute_weight(angle_s, normalWeightParam_s, 0.0);
    w_s *= compute_weight(nrs_s.w, roughnessWeightParams.x, roughnessWeightParams.y);
    w_s = select(w_s, 0.0, zs_s > gp.denoising_range);

    let ss = textureLoad(in_spec, clamp(pos_s, vec2i(0), sz - 1), 0);
    w_s *= minHitDistWeight + compute_exponential_weight(ss.w, hitDistWeightParams_s.x, hitDistWeightParams_s.y);

    spec_result += ss * w_s;
    sum_s += w_s;
  }

  diff_result /= sum_d;
  spec_result /= sum_s;

  // Preserve alpha: diffuse.w = history_len (pass through), specular.w = hitDist (restore)
  if !is_prepass {
    diff_result.w = textureLoad(in_diff, px, 0).w; // preserve history_len
    spec_result.w = hitDist_s / max(hitDistScale_s, NRD_EPS);
  }

  return array<vec4f, 2>(diff_result, spec_result);
}

// ============================================================
// PASS 1: PRE-PASS
// ============================================================
@compute @workgroup_size(16, 16)
fn prepass(@builtin(global_invocation_id) gid: vec3u) {
  let px = vec2i(gid.xy);
  let sz = vec2i(gp.rect_size);
  if px.x >= sz.x || px.y >= sz.y { return; }

  let viewZ = read_viewz(px);
  if viewZ > gp.denoising_range { return; }

  let result = spatial_filter(px, sz,
    REBLUR_PRE_PASS_RADIUS_SCALE, REBLUR_PRE_PASS_FRACTION_SCALE,
    1.0 / (1.0 + 10.0), true, gp.rotator_pre);

  textureStore(out_diff, px, result[0]);
  textureStore(out_spec, px, result[1]);
}

// ============================================================
// PASS 2: TEMPORAL ACCUMULATION
// Surface motion reprojection + virtual motion for specular
// ============================================================
@compute @workgroup_size(16, 16)
fn temporal_accumulation(@builtin(global_invocation_id) gid: vec3u) {
  let px = vec2i(gid.xy);
  let sz = vec2i(gp.rect_size);
  if px.x >= sz.x || px.y >= sz.y { return; }

  let viewZ = read_viewz(px);
  if viewZ > gp.denoising_range {
    textureStore(out_diff, px, textureLoad(in_diff, px, 0));
    textureStore(out_spec, px, textureLoad(in_spec, px, 0));
    textureStore(out_diff, px, textureLoad(in_diff, px, 0));
    textureStore(out_spec, px, textureLoad(in_spec, px, 0));
    return;
  }

  let pixelUv = (vec2f(px) + 0.5) * gp.rect_size_inv;
  let nr = textureLoad(in_normal_roughness, px, 0);
  let N = nr.xyz;
  let roughness = nr.w;
  let Xv = reconstruct_view_pos(pixelUv, viewZ);
  let X = rotate_vector_m(gp.view_to_world, Xv);
  let V = get_view_vector(X);
  let NoV = abs(dot(N, V));
  let frustumSize = get_frustum_size(viewZ);

  // Current data
  let diff = textureLoad(in_diff, px, 0);
  let spec = textureLoad(in_spec, px, 0);

  // === Surface motion reprojection (compute from matrices, no MV texture) ===
  let Xprev = X; // static world: previous position = current position
  let smbPixelUv = project_to_screen(gp.world_to_clip_prev, Xprev);

  var diff_blend = diff;
  var spec_blend = spec;
  var diffAccumSpeed = 0.0;
  var specAccumSpeed = 0.0;

  if is_in_screen(smbPixelUv) {
    // Sample history
    let diff_hist = textureSampleLevel(history_diff, linear_clamp, smbPixelUv, 0.0);
    let spec_hist = textureSampleLevel(history_spec, linear_clamp, smbPixelUv, 0.0);

    // Disocclusion: compare current viewZ with reprojected viewZ
    // history_diff.w stores normalized accumSpeed (0-1), NOT viewZ
    // Use camera-space Z comparison: current Xvprev.z vs what we'd expect
    let Xvprev = affine_transform(gp.world_to_view_prev, Xprev);
    let expectedViewZ = Xvprev.z;
    // Read neighbor depth at reprojected position from current frame's gbuffer as proxy
    let reproj_px = clamp(vec2i(smbPixelUv * gp.rect_size), vec2i(0), vec2i(gp.rect_size) - 1);
    let reproj_depth = read_viewz(reproj_px);
    let disocclusionThreshold = get_disocclusion_threshold(frustumSize, NoV);
    // Simple depth similarity: if reprojected position has similar depth, it's the same surface
    let depth_valid = abs(viewZ - reproj_depth) < frustumSize * 0.5 || length(gp.camera_delta.xyz) < 0.001;

    if depth_valid {
      // Previous accumulation speeds from history alpha (normalized to [0,1])
      let prevAccumNorm = diff_hist.w;
      let prevData = vec2f(prevAccumNorm * REBLUR_MAX_ACCUM_FRAME_NUM);
      diffAccumSpeed = prevData.x;
      specAccumSpeed = prevData.y;

      // Footprint quality (simplified: based on depth similarity)
      let depthDiff = abs(viewZ - reproj_depth);
      let footprintQuality = clamp(1.0 - depthDiff / (frustumSize * 0.5), 0.0, 1.0);
      diffAccumSpeed *= footprintQuality;
      specAccumSpeed *= footprintQuality;

      // Limit
      diffAccumSpeed = min(diffAccumSpeed, gp.max_accumulated_frame_num);
      specAccumSpeed = min(specAccumSpeed, gp.max_accumulated_frame_num);

      // Mix
      let diffNonLinear = 1.0 / (1.0 + diffAccumSpeed);
      let specNonLinear = 1.0 / (1.0 + specAccumSpeed);
      diff_blend = mix_history_and_current(diff_hist, diff, diffNonLinear);
      spec_blend = mix_history_and_current(spec_hist, spec, specNonLinear);

      // === Virtual motion for specular ===
      let hitDistNorm = get_hit_dist_normalization(viewZ, roughness);
      let hitDist = spec.w * hitDistNorm;
      if hitDist > 0.001 && roughness < 0.9 {
        let curvature = 0.0; // simplified: flat surfaces
        let Xvirtual = get_x_virtual(hitDist, curvature, X, Xprev, N, V, roughness);
        let vmbPixelUv = project_to_screen(gp.world_to_clip_prev, Xvirtual);

        if is_in_screen(vmbPixelUv) {
          let vmb_hist = textureSampleLevel(history_spec, linear_clamp, vmbPixelUv, 0.0);
          let vmb_prevZ = textureSampleLevel(history_spec, linear_clamp, vmbPixelUv, 0.0).w * gp.denoising_range;
          let vmb_valid = vmb_prevZ < gp.denoising_range;

          if vmb_valid {
            // Confidence: roughness-based (smooth = virtual, rough = surface)
            let dominantFactor = get_specular_dominant_factor(NoV, roughness);
            let virtualAmount = clamp(dominantFactor * 2.0 - 0.5, 0.0, 1.0);

            // Blend surface and virtual history
            let blended_hist = mix(spec_hist, vmb_hist, virtualAmount);
            spec_blend = mix_history_and_current(blended_hist, spec, specNonLinear);
          }
        }
      }

      // Firefly suppressor (REBLUR_TemporalAccumulation:793-812)
      let specMaxRelIntensity = gp.firefly_suppressor_min_relative_scale + REBLUR_FIREFLY_SUPPRESSOR_MAX_RELATIVE_INTENSITY / (specAccumSpeed + 1.0);
      let specAntifirefly = specAccumSpeed * gp.max_blur_radius * REBLUR_FIREFLY_SUPPRESSOR_RADIUS_SCALE / (1.0 + specAccumSpeed * gp.max_blur_radius * REBLUR_FIREFLY_SUPPRESSOR_RADIUS_SCALE);
      let spec_luma_result = get_luma(spec_blend);
      let spec_luma_clamped = min(spec_luma_result, get_luma(spec_hist) * specMaxRelIntensity);
      let spec_luma_final = mix(spec_luma_result, spec_luma_clamped, specAntifirefly);
      spec_blend = change_luma(spec_blend, spec_luma_final);
    }
  }

  // Store accumSpeed normalized in diff alpha for downstream passes
  let accumNorm = min(diffAccumSpeed, REBLUR_MAX_ACCUM_FRAME_NUM) / REBLUR_MAX_ACCUM_FRAME_NUM;
  textureStore(out_diff, px, vec4f(diff_blend.xyz, accumNorm));
  textureStore(out_spec, px, spec_blend);
}

// ============================================================
// PASS 3: HISTORY FIX
// Fill disoccluded pixels from neighbors with more history
// ============================================================
@compute @workgroup_size(16, 16)
fn history_fix(@builtin(global_invocation_id) gid: vec3u) {
  let px = vec2i(gid.xy);
  let sz = vec2i(gp.rect_size);
  if px.x >= sz.x || px.y >= sz.y { return; }

  let viewZ = read_viewz(px);
  if viewZ > gp.denoising_range { return; }

  let nr = textureLoad(in_normal_roughness, px, 0);
  let N = nr.xyz;
  let Nv = rotate_vector_inv_m(gp.view_to_world, N);
  let pixelUv = (vec2f(px) + 0.5) * gp.rect_size_inv;
  let Xv = reconstruct_view_pos(pixelUv, viewZ);
  let frustumSize = get_frustum_size(viewZ);

  // Read current data
  var diff = textureLoad(in_diff, px, 0);
  var spec = textureLoad(in_spec, px, 0);
  // AccumSpeed stored in diff alpha (normalized to [0,1])
  let frameNum = textureLoad(in_diff, px, 0).w;

  if frameNum >= gp.history_fix_frame_num {
    textureStore(out_diff, px, diff);
    textureStore(out_spec, px, spec);
    return;
  }

  // Parameters
  let nonLinearAccumSpeed = 1.0 / (1.0 + frameNum);
  let normalWeightParam = get_normal_weight_param(nonLinearAccumSpeed, 1.0);
  let geometryWeightParams = get_geometry_weight_params(frustumSize, Xv, Nv);

  // Stride: reduce if neighbors have longer history
  var stride = gp.history_fix_stride;
  stride *= 2.0 / f32(REBLUR_HISTORY_FIX_FILTER_RADIUS);

  // Weighted reconstruction
  var sum_d = 1.0 + frameNum;
  var sum_s = 1.0 + frameNum;
  diff *= sum_d;
  spec *= sum_s;

  for (var j = -REBLUR_HISTORY_FIX_FILTER_RADIUS; j <= REBLUR_HISTORY_FIX_FILTER_RADIUS; j++) {
    for (var i = -REBLUR_HISTORY_FIX_FILTER_RADIUS; i <= REBLUR_HISTORY_FIX_FILTER_RADIUS; i++) {
      if i == 0 && j == 0 { continue; }
      if abs(i) + abs(j) == REBLUR_HISTORY_FIX_FILTER_RADIUS * 2 { continue; } // skip corners

      let uv = pixelUv + vec2f(f32(i), f32(j)) * stride * gp.rect_size_inv;
      let uv_safe = mirror_uv(uv);
      let pos = vec2i(uv_safe * gp.rect_size);
      let pos_clamped = clamp(pos, vec2i(0), sz - 1);

      let zs = read_viewz(pos_clamped);
      let Xvs = reconstruct_view_pos(vec2f(pos_clamped) * gp.rect_size_inv + 0.5 * gp.rect_size_inv, zs);
      let Ns = textureLoad(in_normal_roughness, pos_clamped, 0).xyz;

      let angle = acos(clamp(dot(N, Ns), -1.0, 1.0));
      let NoX = dot(Nv, Xvs);

      var w = compute_weight(NoX, geometryWeightParams.x, geometryWeightParams.y);
      w *= compute_exponential_weight(angle, normalWeightParam, 0.0);
      w = select(w, 0.0, zs > gp.denoising_range);

      // Boost weight by neighbor's frame number (ReBLUR: w *= 1 + frameNum_sample)
      let s_diff = textureLoad(in_diff, pos_clamped, 0);
      let s_spec = textureLoad(in_spec, pos_clamped, 0);
      let s_frameNum = s_diff.w * REBLUR_MAX_ACCUM_FRAME_NUM;
      w *= (1.0 + s_frameNum);

      // Hit distance weight
      let d_hd = s_diff.w - diff.w / max(sum_d, 1.0);
      w *= exp(-d_hd * d_hd / max(nonLinearAccumSpeed, 0.01));

      diff += s_diff * w;
      spec += s_spec * w;
      sum_d += w;
      sum_s += w;
    }
  }

  textureStore(out_diff, px, diff / max(sum_d, NRD_EPS));
  textureStore(out_spec, px, spec / max(sum_s, NRD_EPS));
}

// ============================================================
// PASS 4: BLUR — main spatial filter
// ============================================================
@compute @workgroup_size(16, 16)
fn blur(@builtin(global_invocation_id) gid: vec3u) {
  let px = vec2i(gid.xy);
  let sz = vec2i(gp.rect_size);
  if px.x >= sz.x || px.y >= sz.y { return; }

  // Adaptive Poisson blur with proven geometry weights
  let cn = read_normal(px);
  let cz = read_viewz(px);
  let gz = max(cz * 0.01, 0.1);
  let history_len = textureLoad(in_diff, px, 0).w;

  // Adaptive radius: large when new, small when converged
  let accumSpeed = max(history_len, 0.0);
  let nonLinear = 1.0 / (1.0 + accumSpeed);
  let radius = max(20.0 * sqrt(nonLinear), 1.0);

  // Per-pixel rotation (Bayer + frame)
  let bayer = fract(f32((u32(px.x) & 3u) * 4u + (u32(px.y) & 3u)) / 16.0 + f32(gp.frame_index % 16u) / 16.0);
  let angle = bayer * 6.2832;
  let rc = cos(angle);
  let rs = sin(angle);

  var d_sum = textureLoad(in_diff, px, 0).rgb;
  var s_sum = textureLoad(in_spec, px, 0).rgb;
  var w_sum = 1.0;

  for (var i = 0u; i < 8u; i++) {
    let tap = POISSON8[i];
    let ox = tap.x * rc - tap.y * rs;
    let oy = tap.x * rs + tap.y * rc;
    let offset = vec2i(vec2f(ox, oy) * radius + 0.5);
    let sp = clamp(px + offset, vec2i(0), sz - 1);

    let sn = read_normal(sp);
    let sz2 = read_viewz(sp);
    let gauss = exp(-0.66 * tap.z * tap.z);
    let wn = pow(max(dot(cn, sn), 0.0), 32.0);
    let wz = exp(-abs(cz - sz2) / (gz * radius + 0.01));
    let w = gauss * wn * wz;

    d_sum += textureLoad(in_diff, sp, 0).rgb * w;
    s_sum += textureLoad(in_spec, sp, 0).rgb * w;
    w_sum += w;
  }

  textureStore(out_diff, px, vec4f(d_sum / w_sum, history_len));
  textureStore(out_spec, px, vec4f(s_sum / w_sum, textureLoad(in_spec, px, 0).w));
}

// ============================================================
// PASS 5: POST-BLUR — final spatial cleanup + write to history
// ============================================================
@compute @workgroup_size(16, 16)
fn post_blur(@builtin(global_invocation_id) gid: vec3u) {
  let px = vec2i(gid.xy);
  let sz = vec2i(gp.rect_size);
  if px.x >= sz.x || px.y >= sz.y { return; }

  // PostBlur: same as Blur but 2x radius, tighter normal weight
  let cn2 = read_normal(px);
  let cz2 = read_viewz(px);
  let gz2 = max(cz2 * 0.01, 0.1);
  let hl2 = textureLoad(in_diff, px, 0).w;
  let accumSpeed2 = max(hl2, 0.0);
  let nonLinear2 = 1.0 / (1.0 + accumSpeed2);
  let radius2 = max(40.0 * sqrt(nonLinear2), 1.0); // 2x radius

  let bayer2 = fract(f32((u32(px.x) & 3u) * 4u + (u32(px.y) & 3u)) / 16.0 + f32((gp.frame_index + 8u) % 16u) / 16.0);
  let angle2 = bayer2 * 6.2832;
  let rc2 = cos(angle2);
  let rs2 = sin(angle2);

  var d_sum2 = textureLoad(in_diff, px, 0).rgb;
  var s_sum2 = textureLoad(in_spec, px, 0).rgb;
  var w_sum2 = 1.0;

  for (var i = 0u; i < 8u; i++) {
    let tap = POISSON8[i];
    let ox2 = tap.x * rc2 - tap.y * rs2;
    let oy2 = tap.x * rs2 + tap.y * rc2;
    let offset2 = vec2i(vec2f(ox2, oy2) * radius2 + 0.5);
    let sp2 = clamp(px + offset2, vec2i(0), sz - 1);

    let sn2 = read_normal(sp2);
    let sz3 = read_viewz(sp2);
    let gauss2 = exp(-0.66 * tap.z * tap.z);
    let wn2 = pow(max(dot(cn2, sn2), 0.0), 64.0); // tighter normals
    let wz2 = exp(-abs(cz2 - sz3) / (gz2 * radius2 + 0.01));
    let w2 = gauss2 * wn2 * wz2;

    d_sum2 += textureLoad(in_diff, sp2, 0).rgb * w2;
    s_sum2 += textureLoad(in_spec, sp2, 0).rgb * w2;
    w_sum2 += w2;
  }

  textureStore(out_diff, px, vec4f(d_sum2 / w_sum2, hl2));
  textureStore(out_spec, px, vec4f(s_sum2 / w_sum2, textureLoad(in_spec, px, 0).w));
}

// ============================================================
// PASS 6: TEMPORAL STABILIZATION — anti-flicker
// Luminance-based neighborhood clamp + antilag
// ============================================================
@compute @workgroup_size(16, 16)
fn temporal_stabilization(@builtin(global_invocation_id) gid: vec3u) {
  let px = vec2i(gid.xy);
  let sz = vec2i(gp.rect_size);
  if px.x >= sz.x || px.y >= sz.y { return; }

  let viewZ = read_viewz(px);
  if viewZ > gp.denoising_range { return; }

  let nr = textureLoad(in_normal_roughness, px, 0);
  let roughness = nr.w;

  // Read current (post-blur) data in YCoCg
  let diff_raw = textureLoad(in_diff, px, 0);
  let spec_raw = textureLoad(in_spec, px, 0);
  let diff_ycc = vec4f(linear_to_ycocg(diff_raw.xyz), diff_raw.w);
  let spec_ycc = vec4f(linear_to_ycocg(spec_raw.xyz), spec_raw.w);

  let diff_luma = diff_ycc.x;
  let spec_luma = spec_ycc.x;

  // Compute neighborhood luma statistics (5x5)
  var d_m1 = diff_luma;
  var d_m2 = diff_luma * diff_luma;
  var s_m1 = spec_luma;
  var s_m2 = spec_luma * spec_luma;
  var count = 1.0;

  for (var j = -2; j <= 2; j++) {
    for (var i = -2; i <= 2; i++) {
      if i == 0 && j == 0 { continue; }
      let sp = clamp(px + vec2i(i, j), vec2i(0), sz - 1);
      let sv = read_viewz(sp);
      if sv > gp.denoising_range { continue; }

      let dl = get_luma(vec4f(linear_to_ycocg(textureLoad(in_diff, sp, 0).xyz), 0.0));
      let sl = get_luma(vec4f(linear_to_ycocg(textureLoad(in_spec, sp, 0).xyz), 0.0));
      d_m1 += dl; d_m2 += dl * dl;
      s_m1 += sl; s_m2 += sl * sl;
      count += 1.0;
    }
  }
  d_m1 /= count; d_m2 /= count;
  s_m1 /= count; s_m2 /= count;
  let d_sigma = sqrt(max(d_m2 - d_m1 * d_m1, 0.0));
  let s_sigma = sqrt(max(s_m2 - s_m1 * s_m1, 0.0));

  // Read accumulation speed
  let accumSpeed = diff_raw.w * REBLUR_MAX_ACCUM_FRAME_NUM;

  // Surface motion reprojection for history sampling
  let pixelUv = (vec2f(px) + 0.5) * gp.rect_size_inv;
  let Xv = reconstruct_view_pos(pixelUv, viewZ);
  let X = rotate_vector_m(gp.view_to_world, Xv);
  let mv = textureLoad(in_mv, px, 0).xyz;
  let Xprev = X + mv;
  let smbPixelUv = project_to_screen(gp.world_to_clip_prev, Xprev);

  var diff_hist_luma = diff_luma;
  var spec_hist_luma = spec_luma;

  if is_in_screen(smbPixelUv) {
    diff_hist_luma = get_luma(vec4f(linear_to_ycocg(textureSampleLevel(history_diff, linear_clamp, smbPixelUv, 0.0).xyz), 0.0));
    spec_hist_luma = get_luma(vec4f(linear_to_ycocg(textureSampleLevel(history_spec, linear_clamp, smbPixelUv, 0.0).xyz), 0.0));
    diff_hist_luma = max(diff_hist_luma, 0.0);
    spec_hist_luma = max(spec_hist_luma, 0.0);
  }

  // Antilag
  let d_antilag = compute_antilag(diff_hist_luma, d_m1, d_sigma, accumSpeed);
  let s_antilag = compute_antilag(spec_hist_luma, s_m1, s_sigma, accumSpeed);

  // Temporal accumulation params with antilag
  let d_params = get_temporal_accum_params(1.0, accumSpeed * d_antilag, d_antilag);
  let s_params = get_temporal_accum_params(1.0, accumSpeed * s_antilag, s_antilag);

  // Clamp history luma to neighborhood range (± sigma * scale)
  let d_hist_clamped = clamp(diff_hist_luma, d_m1 - d_sigma * d_params.y, d_m1 + d_sigma * d_params.y);
  let s_hist_clamped = clamp(spec_hist_luma, s_m1 - s_sigma * s_params.y, s_m1 + s_sigma * s_params.y);

  // Stabilize: blend current with clamped history
  let d_stabilized = mix(diff_luma, d_hist_clamped, min(d_params.x, gp.stabilization_strength));
  let s_stabilized = mix(spec_luma, s_hist_clamped, min(s_params.x, gp.stabilization_strength));

  // Apply stabilized luma back to full color
  var diff_out = change_luma(diff_ycc, d_stabilized);
  var spec_out = change_luma(spec_ycc, s_stabilized);

  // Convert back to linear
  diff_out = vec4f(ycocg_to_linear(diff_out.xyz), diff_out.w);
  spec_out = vec4f(ycocg_to_linear(spec_out.xyz), spec_out.w);

  textureStore(out_diff, px, diff_out);
  textureStore(out_spec, px, spec_out);
}
