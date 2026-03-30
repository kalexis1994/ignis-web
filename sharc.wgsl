// SHaRC Resolve — merges per-frame accumulation into resolved cache
// Extended with L1 SH path guiding: stores dominant light direction per voxel
// Layout: keys_accum = [0..cap) keys + [cap..cap*7) accum RGBS+DxDyDz
//         resolved = [0..cap*7) resolved RGBS+DxDyDz

struct SharcParams {
  capacity: u32,
  frame_index: u32,
  scene_scale: f32,
  stale_max: u32,
  camera_pos: vec3f,
  _pad: f32,
};

const SHARC_SCALE: f32 = 1000.0;
const GUIDE_SCALE: f32 = 10000.0;

@group(0) @binding(0) var<uniform> params: SharcParams;
@group(0) @binding(1) var<storage, read_write> keys_accum: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> resolved_data: array<u32>;

@compute @workgroup_size(256)
fn resolve(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if idx >= params.capacity { return; }

  let key = atomicLoad(&keys_accum[idx]);
  if key == 0u { return; }

  let aBase = params.capacity + idx * 7u;
  let aR = atomicLoad(&keys_accum[aBase]);
  let aG = atomicLoad(&keys_accum[aBase + 1u]);
  let aB = atomicLoad(&keys_accum[aBase + 2u]);
  let aCnt = atomicLoad(&keys_accum[aBase + 3u]);
  let aDx = atomicLoad(&keys_accum[aBase + 4u]);
  let aDy = atomicLoad(&keys_accum[aBase + 5u]);
  let aDz = atomicLoad(&keys_accum[aBase + 6u]);

  let rBase = idx * 7u;
  let prevR = bitcast<f32>(resolved_data[rBase]);
  let prevG = bitcast<f32>(resolved_data[rBase + 1u]);
  let prevB = bitcast<f32>(resolved_data[rBase + 2u]);
  let prevMeta = resolved_data[rBase + 3u];
  let prevSamples = f32(prevMeta & 0xFFFFu);
  let staleFrames = (prevMeta >> 16u) & 0xFFFFu;
  let prevDx = bitcast<f32>(resolved_data[rBase + 4u]);
  let prevDy = bitcast<f32>(resolved_data[rBase + 5u]);
  let prevDz = bitcast<f32>(resolved_data[rBase + 6u]);

  if aCnt > 0u {
    let invScale = 1.0 / SHARC_SCALE;
    let newRad = vec3f(f32(aR), f32(aG), f32(aB)) * invScale / f32(aCnt);
    let prevW = prevSamples / max(prevSamples + f32(aCnt), 1.0);
    let newW = f32(aCnt) / max(prevSamples + f32(aCnt), 1.0);
    let blended = vec3f(prevR, prevG, prevB) * prevW + newRad * newW;
    let totalSamples = min(prevSamples + f32(aCnt), 65535.0);

    // Resolve direction: decode offset encoding → signed direction
    // Stored as: val = (dir * lum + lum) * GUIDE_SCALE
    // sum_lum computed from accumulated RGB
    let sum_lum = (0.2126 * f32(aR) + 0.7152 * f32(aG) + 0.0722 * f32(aB)) * invScale;
    var newDx = 0.0; var newDy = 0.0; var newDz = 0.0;
    if sum_lum > 0.01 {
      newDx = f32(aDx) / GUIDE_SCALE / sum_lum - 1.0;
      newDy = f32(aDy) / GUIDE_SCALE / sum_lum - 1.0;
      newDz = f32(aDz) / GUIDE_SCALE / sum_lum - 1.0;
    }
    let blendDx = prevDx * prevW + newDx * newW;
    let blendDy = prevDy * prevW + newDy * newW;
    let blendDz = prevDz * prevW + newDz * newW;

    resolved_data[rBase] = bitcast<u32>(blended.x);
    resolved_data[rBase + 1u] = bitcast<u32>(blended.y);
    resolved_data[rBase + 2u] = bitcast<u32>(blended.z);
    resolved_data[rBase + 3u] = u32(totalSamples) & 0xFFFFu;
    resolved_data[rBase + 4u] = bitcast<u32>(blendDx);
    resolved_data[rBase + 5u] = bitcast<u32>(blendDy);
    resolved_data[rBase + 6u] = bitcast<u32>(blendDz);

    // Clear accumulators (all 7)
    atomicStore(&keys_accum[aBase], 0u);
    atomicStore(&keys_accum[aBase + 1u], 0u);
    atomicStore(&keys_accum[aBase + 2u], 0u);
    atomicStore(&keys_accum[aBase + 3u], 0u);
    atomicStore(&keys_accum[aBase + 4u], 0u);
    atomicStore(&keys_accum[aBase + 5u], 0u);
    atomicStore(&keys_accum[aBase + 6u], 0u);
  } else {
    let newStale = staleFrames + 1u;
    if newStale > params.stale_max {
      atomicStore(&keys_accum[idx], 0u);
      resolved_data[rBase] = 0u; resolved_data[rBase+1u] = 0u;
      resolved_data[rBase+2u] = 0u; resolved_data[rBase+3u] = 0u;
      resolved_data[rBase+4u] = 0u; resolved_data[rBase+5u] = 0u;
      resolved_data[rBase+6u] = 0u;
    } else {
      resolved_data[rBase + 3u] = (newStale << 16u) | (u32(prevSamples) & 0xFFFFu);
    }
  }
}
