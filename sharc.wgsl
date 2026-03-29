// SHaRC Resolve — merges per-frame accumulation into resolved cache
// Layout: keys_accum = [0..cap) keys + [cap..cap*5) accum RGBS
//         resolved = [0..cap*4) resolved RGBS

struct SharcParams {
  capacity: u32,
  frame_index: u32,
  scene_scale: f32,
  stale_max: u32,
  camera_pos: vec3f,
  _pad: f32,
};

const SHARC_SCALE: f32 = 1000.0;

@group(0) @binding(0) var<uniform> params: SharcParams;
@group(0) @binding(1) var<storage, read_write> keys_accum: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> resolved_data: array<u32>;

@compute @workgroup_size(256)
fn resolve(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if idx >= params.capacity { return; }

  let key = atomicLoad(&keys_accum[idx]);
  if key == 0u { return; }

  let aBase = params.capacity + idx * 4u;
  let aR = atomicLoad(&keys_accum[aBase]);
  let aG = atomicLoad(&keys_accum[aBase + 1u]);
  let aB = atomicLoad(&keys_accum[aBase + 2u]);
  let aCnt = atomicLoad(&keys_accum[aBase + 3u]);

  let rBase = idx * 4u;
  let prevR = bitcast<f32>(resolved_data[rBase]);
  let prevG = bitcast<f32>(resolved_data[rBase + 1u]);
  let prevB = bitcast<f32>(resolved_data[rBase + 2u]);
  let prevMeta = resolved_data[rBase + 3u];
  let prevSamples = f32(prevMeta & 0xFFFFu);
  let staleFrames = (prevMeta >> 16u) & 0xFFFFu;

  if aCnt > 0u {
    let invScale = 1.0 / SHARC_SCALE;
    let newRad = vec3f(f32(aR), f32(aG), f32(aB)) * invScale / f32(aCnt);
    let prevW = prevSamples / max(prevSamples + f32(aCnt), 1.0);
    let newW = f32(aCnt) / max(prevSamples + f32(aCnt), 1.0);
    let blended = vec3f(prevR, prevG, prevB) * prevW + newRad * newW;
    let totalSamples = min(prevSamples + f32(aCnt), 65535.0);

    resolved_data[rBase] = bitcast<u32>(blended.x);
    resolved_data[rBase + 1u] = bitcast<u32>(blended.y);
    resolved_data[rBase + 2u] = bitcast<u32>(blended.z);
    resolved_data[rBase + 3u] = u32(totalSamples) & 0xFFFFu;

    atomicStore(&keys_accum[aBase], 0u);
    atomicStore(&keys_accum[aBase + 1u], 0u);
    atomicStore(&keys_accum[aBase + 2u], 0u);
    atomicStore(&keys_accum[aBase + 3u], 0u);
  } else {
    let newStale = staleFrames + 1u;
    if newStale > params.stale_max {
      atomicStore(&keys_accum[idx], 0u);
      resolved_data[rBase] = 0u; resolved_data[rBase+1u] = 0u;
      resolved_data[rBase+2u] = 0u; resolved_data[rBase+3u] = 0u;
    } else {
      resolved_data[rBase + 3u] = (newStale << 16u) | (u32(prevSamples) & 0xFFFFu);
    }
  }
}
