// WebGPU Wavefront Path Tracer — minimal v1
// Uses wavefront.wgsl (3 compute kernels: generate, bounce, finalize)
// + display.wgsl (fullscreen blit). No denoiser, no FSR, no temporal
// accumulation in this first version — those come back incrementally.

import { loadScene } from './scene-loader.js';

function rlog(...args) {
  const msg = args.map(a => typeof a === 'object' ? JSON.stringify(a) : String(a)).join(' ');
  console.log(msg);
  fetch('/log', { method: 'POST', body: msg }).catch(() => {});
}
window.onerror = (msg, src, line) => rlog(`ERROR: ${msg} at ${src}:${line}`);
window.onunhandledrejection = (e) => rlog(`REJECT: ${e.reason}`);

const info = document.getElementById('info');
const errorDiv = document.getElementById('error');

function showError(msg) {
  errorDiv.style.display = 'block';
  errorDiv.textContent = msg;
  info.style.display = 'none';
}

function createGPUBuffer(device, data, usage) {
  const buf = device.createBuffer({ size: data.byteLength, usage: usage | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buf, 0, data);
  return buf;
}

async function init() {
  if (!navigator.gpu) {
    showError('WebGPU not supported. Try Chrome 113+, Edge 113+, or Firefox Nightly.');
    return;
  }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) { showError('Failed to get GPU adapter.'); return; }

  // Wavefront + shadow queue + compaction needs 11 storage buffers in
  // the compute stage (default limit is 8). Adreno 7xx exposes ≥16.
  // shader-f16: FP16 has 2× ALU throughput on Adreno.
  // subgroups: wave-level primitives (ballot, exclusiveAdd, broadcast)
  // used to replace per-lane atomicAdd in queue compaction with one
  // atomicAdd per wave. Huge win for atomic-heavy kernels on mobile.
  const hasF16 = adapter.features.has('shader-f16');
  const hasSubgroups = adapter.features.has('subgroups');
  if (!hasF16) rlog('WARN: shader-f16 not available; FP16 paths will fail to compile.');
  if (!hasSubgroups) rlog('WARN: subgroups not available; wave-level compaction will fail to compile.');
  const device = await adapter.requestDevice({
    requiredFeatures: [
      ...(hasF16 ? ['shader-f16'] : []),
      ...(hasSubgroups ? ['subgroups'] : []),
    ],
    requiredLimits: {
      maxStorageBuffersPerShaderStage: Math.min(adapter.limits.maxStorageBuffersPerShaderStage, 16),
    },
  });
  device.onuncapturederror = (e) => rlog('GPU_ERROR: ' + e.error.message);
  device.lost.then(i => rlog('DEVICE_LOST: ' + i.message));

  const adapterInfo = adapter.info || {};
  const gpuStr = `${adapterInfo.vendor||''} ${adapterInfo.architecture||''} ${adapterInfo.device||''}`.trim();
  rlog(`GPU: ${gpuStr}`);

  // Canvas + context
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('webgpu');
  const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);

  canvas.width  = Math.floor(window.innerWidth  * dpr);
  canvas.height = Math.floor(window.innerHeight * dpr);
  ctx.configure({ device, format: canvasFormat, alphaMode: 'premultiplied' });

  // Internal render resolution — 50% of display for speed. No FSR upscaling
  // in v1; display pass samples bilinear straight to canvas.
  const SCALE = 0.5;
  const rw = Math.max(1, Math.floor(canvas.width  * SCALE));
  const rh = Math.max(1, Math.floor(canvas.height * SCALE));
  rlog(`Render: ${rw}x${rh} → ${canvas.width}x${canvas.height} (scale ${SCALE})`);

  // Load scene
  info.textContent = 'Loading scene...';
  let scene;
  try {
    scene = await loadScene('scene', m => { info.textContent = m; });
  } catch (e) {
    showError(`Scene load failed: ${e.message}`);
    console.error(e);
    return;
  }
  if (!scene.rasterIndices || !scene.vertMatIds) {
    rlog('Stale cache detected — clearing and reloading...');
    try { indexedDB.deleteDatabase('ignis-scene-cache'); } catch(e) {}
    info.textContent = 'Cache outdated, reloading...';
    setTimeout(() => location.reload(), 500);
    return;
  }
  rlog(`Scene: ${scene.stats.triangles} tris, ${scene.stats.bvhNodes} BVH nodes, ${scene.stats.materials} materials`);

  // Scene buffers (same layout the legacy pathtracer used)
  const vtxBuf = createGPUBuffer(device, scene.gpuPositions, GPUBufferUsage.STORAGE);
  const nrmBuf = createGPUBuffer(device, scene.gpuNormals,   GPUBufferUsage.STORAGE);
  const triBuf = createGPUBuffer(device, scene.gpuTriData,   GPUBufferUsage.STORAGE);
  const bvhBuf = createGPUBuffer(device, scene.gpuBVHNodes,  GPUBufferUsage.STORAGE);
  const bvhScene = scene.bvhScene; // { origin: [x,y,z], scale: [x,y,z] } for uint16 dequant
  const matBuf = createGPUBuffer(device, scene.gpuMaterials, GPUBufferUsage.STORAGE);
  // Emissive-triangle CDF (power-weighted) built in scene-loader. 4 floats
  // per entry: [triIdx(u32), area, cdf, 0]. Empty scene still gets a
  // 16-byte min-size buffer so binding is always valid — shader gates on
  // uniforms.emissive_count > 0 before reading.
  const emissiveTrisBuf = createGPUBuffer(device, scene.gpuEmissiveTris, GPUBufferUsage.STORAGE);
  rlog(`Emissive tris: ${scene.stats.emissiveTris}${scene.stats.emissiveTruncated ? ` (truncated from ${scene.stats.emissiveSourceTris})` : ''}`);

  // Output textures
  const noisyTex = device.createTexture({
    size: [rw, rh], format: 'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });
  const ndTex = device.createTexture({
    size: [rw, rh], format: 'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING,
  });
  const albTex = device.createTexture({
    size: [rw, rh], format: 'rgba8unorm',
    // TEXTURE_BINDING so restir_shade can sample albedo_primary at shade time.
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });
  // Temporal accumulation ping-pong textures. Each frame reads from one
  // and writes to the other; display samples whichever was just written.
  const accumTex = [0, 1].map(() => device.createTexture({
    size: [rw, rh], format: 'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  }));

  // Ray state buffer: 4 vec4f per ray = 64 bytes
  const rayStateBuf = device.createBuffer({
    size: rw * rh * 64,
    usage: GPUBufferUsage.STORAGE,
  });
  // Shadow request buffer: 3 vec4f per pixel = 48 bytes. Written by
  // bounce (NEE request) and consumed by shadow_trace.
  const shadowReqBuf = device.createBuffer({
    size: rw * rh * 48,
    usage: GPUBufferUsage.STORAGE,
  });

  // Compaction queues — ping-pong u32 pixel indices per bounce. counts
  // holds the two atomic counters; dispatch_args is the indirect args
  // buffer that prep_dispatch updates between bounces.
  const queueABuf = device.createBuffer({ size: rw * rh * 4, usage: GPUBufferUsage.STORAGE });
  const queueBBuf = device.createBuffer({ size: rw * rh * 4, usage: GPUBufferUsage.STORAGE });
  const countsBuf = device.createBuffer({
    size: 8, // 2 atomic u32
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const dispatchArgsBuf = device.createBuffer({
    size: 16, // 3 u32 + padding (16-byte alignment for indirect args in some drivers)
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
  });

  // ReSTIR GI candidate buffer — 4 vec4f (64 B) per pixel. Cleared by
  // `generate` each frame, written by `bounce` at b==0/1 and `shadow_trace`,
  // read by `restir_shade`.
  const candidateBuf = device.createBuffer({
    size: rw * rh * 64,
    usage: GPUBufferUsage.STORAGE,
  });
  // ReSTIR GI reservoir — ping-pong, 3 vec4f (48 B) per pixel. Motion
  // reprojection reads from reservoir_prev at the reprojected pixel and
  // writes to reservoir_curr at this pixel; single buffer would race
  // because reads and writes target different pixels.
  const reservoirBuf = [0, 1].map(() => device.createBuffer({
    size: rw * rh * 48,
    usage: GPUBufferUsage.STORAGE,
  }));
  // Persistent G-buffer ping-pong — 2 vec4f (32 B) per pixel.
  //   [0] = (x_v_world.xyz, depth_view_linear)
  //   [1] = (n_v.xyz, valid_flag)
  // Written by bounce at b==0 (Lambertian path); read by restir_temporal
  // to reproject current primary hits into the previous frame's camera
  // and validate the history by normal + plane-distance comparison.
  const gbufBuf = [0, 1].map(() => device.createBuffer({
    size: rw * rh * 32,
    usage: GPUBufferUsage.STORAGE,
  }));

  // Uniforms — 192 bytes. 128 base + 64 for the previous-frame camera
  // pose (pos/forward/right/up + prev_fov_factor) consumed by
  // restir_temporal for motion reprojection.
  const uniformBuf = device.createBuffer({
    size: 192, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Shader modules. wavefront.wgsl + restir_gi.wgsl are concatenated into
  // a single compilation unit so restir_gi can reference Uniforms,
  // Sampler, noisy_out, etc. declared in wavefront, and wavefront can call
  // cand_* helpers / candidate_buf declared in restir_gi (WGSL is order-
  // independent at file scope).
  const [wavefrontSrc, restirSrc] = await Promise.all([
    fetch('wavefront.wgsl').then(r => r.text()),
    fetch('restir_gi.wgsl').then(r => r.text()),
  ]);
  const wavefrontModule = device.createShaderModule({ code: wavefrontSrc + '\n' + restirSrc });
  const displaySrc = await fetch('display.wgsl').then(r => r.text());
  const displayModule = device.createShaderModule({ code: displaySrc });

  // Explicit bind group layouts — shared across all kernels so we can
  // bind the same resources regardless of which bindings each kernel uses.
  const bgl0 = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer:        { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture:{ access: 'write-only', format: 'rgba16float' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture:{ access: 'write-only', format: 'rgba16float' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture:{ access: 'write-only', format: 'rgba8unorm'  } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer:        { type: 'storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer:        { type: 'storage' } },
    ],
  });
  const bgl1 = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    ],
  });
  // bgl2_main: queue_a, queue_b, counts, candidate_buf. NO dispatch_args
  // — so when a bounce kernel dispatches indirectly off dispatchArgsBuf
  // the buffer isn't also bound as storage in the same pass (WebGPU
  // forbids writable + indirect in the same sync scope). candidate_buf
  // (binding 4) is declared in restir_gi.wgsl and used by bounce,
  // shadow_trace, and restir_shade.
  const bgl2_main = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // candidate_buf
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // reservoir_curr
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // reservoir_prev
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // gbuf_curr
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // gbuf_prev
    ],
  });
  // bgl2_prep: same + dispatch_args. Used only by prep_dispatch which
  // doesn't dispatch anything (writes args for the NEXT pass). Mirrors
  // every bg2 binding so the shared shader module compiles against
  // either layout.
  const bgl2_prep = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // dispatch_args
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // candidate_buf
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // reservoir_curr
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // reservoir_prev
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // gbuf_curr
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // gbuf_prev
    ],
  });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl0, bgl1, bgl2_main] });
  // prep_dispatch only reads `counts` (bgl2_prep binding 2) and writes
  // `dispatch_args` (binding 3). It doesn't touch group 0 or group 1 at
  // all. Declaring them in the layout anyway would count ALL their
  // storage buffers against maxStorageBuffersPerShaderStage (Adreno=16):
  //   bgl0 (2) + bgl1 (6) + bgl2_prep (9) = 17 → validation failure.
  // Null group slots contribute 0 buffers, keeping prep_dispatch at 9.
  const prepPipeLayout = device.createPipelineLayout({ bindGroupLayouts: [null, null, bgl2_prep] });

  // Group 3 is only for the composite kernel (temporal accumulation)
  const bgl3 = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } }, // noisy_read
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } }, // accum_prev
      { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } }, // accum_new
    ],
  });
  // composite reads noisyTex as texture_2d via bg3, but bgl0 has noisyTex
  // as write-only storage — binding the same texture with both usages in a
  // single pass is a sync-scope violation. Give composite a minimal bgl0
  // with just uniforms, and skip bgl1/bgl2 (composite doesn't use them).
  const bgl0_composite = device.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }],
  });
  const compositeLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl0_composite, null, null, bgl3] });
  // ReLAX temporal accumulation — replaces the old frames_still-gated
  // composite with per-pixel surface-motion reprojection and
  // disocclusion-aware bilinear blending. Needs bg2 for gbuf access
  // (binding 7/8 are gbuf_curr / gbuf_prev, written by bounce and
  // read here for plane-distance / normal disocclusion checks). bg3
  // layout is shared with composite: noisy_read + accum_prev read,
  // accum_new write, with history_length packed into accum.a so no
  // extra binding is needed.
  const relaxTemporalLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl0_composite, null, bgl2_main, bgl3] });

  // Group 3 view for restir_shade — reads albedo_primary as sampled
  // texture at binding 3 (composite uses 0/1/2 of group 3 with a
  // different layout; WebGPU applies layouts per-pipeline so the two
  // views coexist). bgl0 binds albedo_out as write-only storage for
  // bounce; same texture can't be bound both ways in one pass, so
  // restir_shade uses this separate bg3 with texture_2d access.
  const bgl3_shade = device.createBindGroupLayout({
    entries: [
      { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } }, // albedo_read
    ],
  });
  // Minimal bgl0 for restir_shade: uniforms + noisy_out only. Omits
  // ndTex and albTex storage-writes so binding albTex as sampled texture
  // in bg3 doesn't collide with its write-only binding in bgl0.
  const bgl0_shade = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
    ],
  });
  const shadeLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl0_shade, null, bgl2_main, bgl3_shade] });
  // Temporal reuse needs bg1 (BVH + vertices) for the visibility shadow
  // ray from v_curr to prev.x_s. Shade doesn't, so it keeps shadeLayout.
  const temporalLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl0_shade, bgl1, bgl2_main, bgl3_shade] });

  const genPipeline     = device.createComputePipeline({ layout: pipelineLayout, compute: { module: wavefrontModule, entryPoint: 'generate'     } });
  // bounce pipelines: FIRST variant reads gid.x directly (no queue load, used
  // only for bounce 0 where every ray is alive). A/B alternate for bounces 1+.
  const bouncePipelineFirst = device.createComputePipeline({ layout: pipelineLayout, compute: { module: wavefrontModule, entryPoint: 'bounce', constants: { FIRST_BOUNCE: 1, SRC_QUEUE: 0 } } });
  const bouncePipelineA     = device.createComputePipeline({ layout: pipelineLayout, compute: { module: wavefrontModule, entryPoint: 'bounce', constants: { FIRST_BOUNCE: 0, SRC_QUEUE: 0 } } });
  const bouncePipelineB     = device.createComputePipeline({ layout: pipelineLayout, compute: { module: wavefrontModule, entryPoint: 'bounce', constants: { FIRST_BOUNCE: 0, SRC_QUEUE: 1 } } });
  const prepPipelineA   = device.createComputePipeline({ layout: prepPipeLayout, compute: { module: wavefrontModule, entryPoint: 'prep_dispatch', constants: { READ_IDX: 0 } } });
  const prepPipelineB   = device.createComputePipeline({ layout: prepPipeLayout, compute: { module: wavefrontModule, entryPoint: 'prep_dispatch', constants: { READ_IDX: 1 } } });
  const shadowPipeline    = device.createComputePipeline({ layout: pipelineLayout,  compute: { module: wavefrontModule, entryPoint: 'shadow_trace' } });
  const finPipeline       = device.createComputePipeline({ layout: pipelineLayout,  compute: { module: wavefrontModule, entryPoint: 'finalize'     } });
  const compositePipeline = device.createComputePipeline({ layout: compositeLayout, compute: { module: wavefrontModule, entryPoint: 'composite'    } });
  const relaxTemporalPipeline = device.createComputePipeline({ layout: relaxTemporalLayout, compute: { module: wavefrontModule, entryPoint: 'relax_temporal' } });
  // ReSTIR GI shade — replaces finalize. Combines direct (primary-hit
  // emission + NEE) with the resampled indirect term from the reservoir.
  const shadePipeline     = device.createComputePipeline({ layout: shadeLayout,     compute: { module: wavefrontModule, entryPoint: 'restir_shade' } });
  // ReSTIR GI temporal — WRS merge with motion reprojection + jacobian +
  // visibility validation. Uses temporalLayout (includes bgl1 for BVH).
  const temporalPipeline  = device.createComputePipeline({ layout: temporalLayout,  compute: { module: wavefrontModule, entryPoint: 'restir_temporal' } });
  // ReSTIR GI spatial — k random neighbors' reservoirs merged with jacobian
  // + visibility shadow ray. Shares temporalLayout (same bgl signatures).
  const spatialPipeline   = device.createComputePipeline({ layout: temporalLayout,  compute: { module: wavefrontModule, entryPoint: 'restir_spatial' } });

  const bg0 = device.createBindGroup({
    layout: bgl0,
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: noisyTex.createView() },
      { binding: 2, resource: ndTex.createView() },
      { binding: 3, resource: albTex.createView() },
      { binding: 4, resource: { buffer: rayStateBuf } },
      { binding: 5, resource: { buffer: shadowReqBuf } },
    ],
  });
  const bg1 = device.createBindGroup({
    layout: bgl1,
    entries: [
      { binding: 0, resource: { buffer: vtxBuf } },
      { binding: 1, resource: { buffer: nrmBuf } },
      { binding: 2, resource: { buffer: triBuf } },
      { binding: 3, resource: { buffer: bvhBuf } },
      { binding: 4, resource: { buffer: matBuf } },
      { binding: 5, resource: { buffer: emissiveTrisBuf } },
    ],
  });
  // Reservoir role-fixed across frames, gbuf ping-pongs.
  //   reservoirBuf[0] = temporal stage output (read by spatial as "prev",
  //                     read by next frame's temporal as "curr dest")
  //   reservoirBuf[1] = spatial stage output (read by shade this frame,
  //                     read by next frame's temporal as "prev")
  // A single ping-pong on reservoirs can't accommodate all three stages
  // (temporal, spatial, shade) + frame-to-frame carryover without either
  // a 3rd buffer or role-fixing the two. Role-fixing is simpler and the
  // gbuf still ping-pongs via currIdx for temporal reprojection.
  //
  // bg2_main: used by bounce / shadow / generate / prep / temporal —
  //           reservoir_curr (5) = temporal out, reservoir_prev (6) = spatial out
  // bg2_spatial: used by spatial / shade —
  //           reservoir_curr (5) = spatial out, reservoir_prev (6) = temporal out
  const bg2_main = [0, 1].map(curr => device.createBindGroup({
    layout: bgl2_main,
    entries: [
      { binding: 0, resource: { buffer: queueABuf } },
      { binding: 1, resource: { buffer: queueBBuf } },
      { binding: 2, resource: { buffer: countsBuf } },
      { binding: 4, resource: { buffer: candidateBuf } },
      { binding: 5, resource: { buffer: reservoirBuf[0] } },
      { binding: 6, resource: { buffer: reservoirBuf[1] } },
      { binding: 7, resource: { buffer: gbufBuf[curr] } },
      { binding: 8, resource: { buffer: gbufBuf[1 - curr] } },
    ],
  }));
  const bg2_spatial = [0, 1].map(curr => device.createBindGroup({
    layout: bgl2_main,
    entries: [
      { binding: 0, resource: { buffer: queueABuf } },
      { binding: 1, resource: { buffer: queueBBuf } },
      { binding: 2, resource: { buffer: countsBuf } },
      { binding: 4, resource: { buffer: candidateBuf } },
      { binding: 5, resource: { buffer: reservoirBuf[1] } },
      { binding: 6, resource: { buffer: reservoirBuf[0] } },
      { binding: 7, resource: { buffer: gbufBuf[curr] } },
      { binding: 8, resource: { buffer: gbufBuf[1 - curr] } },
    ],
  }));
  const bg2_prep = [0, 1].map(curr => device.createBindGroup({
    layout: bgl2_prep,
    entries: [
      { binding: 0, resource: { buffer: queueABuf } },
      { binding: 1, resource: { buffer: queueBBuf } },
      { binding: 2, resource: { buffer: countsBuf } },
      { binding: 3, resource: { buffer: dispatchArgsBuf } },
      { binding: 4, resource: { buffer: candidateBuf } },
      { binding: 5, resource: { buffer: reservoirBuf[0] } },
      { binding: 6, resource: { buffer: reservoirBuf[1] } },
      { binding: 7, resource: { buffer: gbufBuf[curr] } },
      { binding: 8, resource: { buffer: gbufBuf[1 - curr] } },
    ],
  }));
  // Group 3 bind group for restir_shade — albedo_primary at binding 3.
  const bg3_shade = device.createBindGroup({
    layout: bgl3_shade,
    entries: [
      { binding: 3, resource: albTex.createView() },
    ],
  });
  // Minimal bg0 for restir_shade (uniforms + noisy_out; no albTex).
  const bg0_shade = device.createBindGroup({
    layout: bgl0_shade,
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: noisyTex.createView() },
    ],
  });
  // Ping-pong bind groups for composite: index `i` reads prev = accumTex[1-i]
  // and writes new = accumTex[i]. JS alternates `i` each frame.
  const bg3 = [0, 1].map(i => device.createBindGroup({
    layout: bgl3,
    entries: [
      { binding: 0, resource: noisyTex.createView() },
      { binding: 1, resource: accumTex[1 - i].createView() },
      { binding: 2, resource: accumTex[i].createView() },
    ],
  }));
  // Minimal bg0 for composite (uniforms only — no storage textures to
  // collide with noisyTex bound as texture in bg3).
  const bg0_composite = device.createBindGroup({
    layout: bgl0_composite,
    entries: [{ binding: 0, resource: { buffer: uniformBuf } }],
  });

  // Display pipeline: fullscreen blit noisy → canvas (linear upscale)
  const displaySampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  const displayPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex:   { module: displayModule, entryPoint: 'vs_main' },
    fragment: { module: displayModule, entryPoint: 'fs_main', targets: [{ format: canvasFormat }] },
    primitive: { topology: 'triangle-list' },
  });
  // Display reads from whichever accumulator composite just wrote to.
  // Ping-pong per frame alongside bg3.
  const displayBG = [0, 1].map(i => device.createBindGroup({
    layout: displayPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: accumTex[i].createView() },
      { binding: 1, resource: displaySampler },
    ],
  }));

  // Camera
  const cfg = window.IGNIS_CONFIG || {};
  const maxBounces = Math.max(1, Math.min(8, cfg.bounces || 3));
  const bb = scene.stats;
  const camera = {
    pos: [
      (bb.sceneMin[0] + bb.sceneMax[0]) * 0.5,
      (bb.sceneMin[1] + bb.sceneMax[1]) * 0.5,
      (bb.sceneMin[2] + bb.sceneMax[2]) * 0.5,
    ],
    yaw: 0, pitch: 0, fov: 75,  // applied to the narrow axis (see writeUniforms)
  };
  function camVectors() {
    const cy = Math.cos(camera.yaw), sy = Math.sin(camera.yaw);
    const cp = Math.cos(camera.pitch), sp = Math.sin(camera.pitch);
    // Legacy convention (post "X-flip fix" commit 22a04c3): negated right
    // vector cancels the horizontal mirror you get otherwise from glTF's
    // right-handed coords + our +Z forward camera. up is computed directly
    // from pitch/yaw (not via cross) to stay consistent with this handedness.
    const forward = [cp*sy, sp, cp*cy];
    const right   = [-cy, 0, sy];
    const up      = [-sp*sy, cp, -sp*cy];
    return { forward, right, up };
  }

  // Input: keyboard + mouse drag (desktop) + virtual joysticks (touch)
  const keys = {};
  window.addEventListener('keydown', e => { keys[e.code] = true; });
  window.addEventListener('keyup',   e => { keys[e.code] = false; });
  let dragging = false, lastX = 0, lastY = 0;
  canvas.addEventListener('mousedown', e => { dragging = true; lastX = e.clientX; lastY = e.clientY; });
  canvas.addEventListener('mouseup',   () => { dragging = false; });
  canvas.addEventListener('mousemove', e => {
    if (!dragging) return;
    const dx = e.clientX - lastX, dy = e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    camera.yaw   -= dx * 0.003;
    camera.pitch -= dy * 0.003;
    camera.pitch = Math.max(-Math.PI*0.49, Math.min(Math.PI*0.49, camera.pitch));
  });

  // Virtual joysticks: touch anywhere in left half → move stick at that
  // point; right half → rotate stick. dx/dy in each stick are normalized
  // [-1, 1] after clamping to KNOB_MAX pixels from the spawn point.
  const stickLeft  = { el: document.getElementById('stick-left'),  knob: document.getElementById('knob-left'),  touchId: null, cx:0, cy:0, dx:0, dy:0 };
  const stickRight = { el: document.getElementById('stick-right'), knob: document.getElementById('knob-right'), touchId: null, cx:0, cy:0, dx:0, dy:0 };
  const KNOB_MAX = 50;
  const halfW = () => window.innerWidth / 2;
  function clampStick(dx, dy) { const l = Math.hypot(dx, dy); if (l > KNOB_MAX) { dx = dx/l*KNOB_MAX; dy = dy/l*KNOB_MAX; } return { dx, dy }; }
  function showStick(s, x, y) {
    if (!s.el) return;
    s.el.style.display = 'block';
    s.el.style.left = (x - 70) + 'px';
    s.el.style.top  = (y - 70) + 'px';
    s.el.style.bottom = 'auto';
    s.el.style.right = 'auto';
    s.knob.style.transform = 'translate(-50%, -50%)';
    s.knob.classList.remove('active');
  }
  function hideStick(s) {
    s.touchId = null; s.dx = 0; s.dy = 0;
    if (!s.el) return;
    s.el.style.display = 'none';
    s.knob.style.transform = 'translate(-50%, -50%)';
    s.knob.classList.remove('active');
  }
  function moveKnob(s, tx, ty) {
    const r = clampStick(tx - s.cx, ty - s.cy);
    s.knob.style.transform = `translate(calc(-50% + ${r.dx}px), calc(-50% + ${r.dy}px))`;
    s.knob.classList.toggle('active', r.dx !== 0 || r.dy !== 0);
    s.dx = r.dx / KNOB_MAX;
    s.dy = r.dy / KNOB_MAX;
  }
  if (stickLeft.el)  stickLeft.el.style.display  = 'none';
  if (stickRight.el) stickRight.el.style.display = 'none';

  document.addEventListener('touchstart', e => {
    for (const t of e.changedTouches) {
      const stick = (t.clientX < halfW()) ? stickLeft : stickRight;
      if (stick.touchId !== null) continue;
      stick.touchId = t.identifier;
      stick.cx = t.clientX; stick.cy = t.clientY;
      showStick(stick, t.clientX, t.clientY);
      e.preventDefault();
    }
  }, { passive: false });
  document.addEventListener('touchmove', e => {
    for (const t of e.changedTouches) {
      if (t.identifier === stickLeft.touchId)  { moveKnob(stickLeft,  t.clientX, t.clientY); e.preventDefault(); }
      else if (t.identifier === stickRight.touchId) { moveKnob(stickRight, t.clientX, t.clientY); e.preventDefault(); }
    }
  }, { passive: false });
  function onTouchEnd(e) {
    for (const t of e.changedTouches) {
      if (t.identifier === stickLeft.touchId)  hideStick(stickLeft);
      if (t.identifier === stickRight.touchId) hideStick(stickRight);
    }
  }
  document.addEventListener('touchend',    onTouchEnd);
  document.addEventListener('touchcancel', onTouchEnd);

  // Sun direction (constant for v1)
  const sunDir = (() => {
    const azim = 0.7, elev = 0.6;
    const ce = Math.cos(elev), se = Math.sin(elev);
    return [ce*Math.sin(azim), se, ce*Math.cos(azim)];
  })();

  // Frame loop
  let frameIdx = 0, lastTime = performance.now(), fps = 0, fpsAccum = 0, fpsCount = 0;
  // Temporal accumulation state
  let framesStill = 0, accumFrame = 0;
  let prevPos = [...camera.pos], prevYaw = camera.yaw, prevPitch = camera.pitch;
  // ReSTIR ping-pong: currIdx picks which {reservoir,gbuf}Buf is the
  // "curr" (this-frame write target) and which is "prev" (last-frame
  // read source). Flipped at end of each frame.
  let currIdx = 0;

  function updateCamera(dt) {
    const { forward, right } = camVectors();
    const speed = (keys['ShiftLeft'] ? 10 : 3) * dt;
    if (keys['KeyW']) for (let i=0;i<3;i++) camera.pos[i] += forward[i] * speed;
    if (keys['KeyS']) for (let i=0;i<3;i++) camera.pos[i] -= forward[i] * speed;
    if (keys['KeyA']) for (let i=0;i<3;i++) camera.pos[i] -= right[i]   * speed;
    if (keys['KeyD']) for (let i=0;i<3;i++) camera.pos[i] += right[i]   * speed;
    if (keys['KeyE']) camera.pos[1] += speed;
    if (keys['KeyQ']) camera.pos[1] -= speed;

    // Virtual joysticks (touch): left = move (XZ planar + strafe),
    // right = rotate (yaw/pitch). Deadzone 0.12, matches old renderer.
    const DZ = 0.12;
    const lx = Math.abs(stickLeft.dx)  > DZ ? stickLeft.dx  : 0;
    const ly = Math.abs(stickLeft.dy)  > DZ ? stickLeft.dy  : 0;
    if (lx || ly) {
      camera.pos[0] += forward[0] * (-ly) * speed + right[0] * lx * speed;
      camera.pos[1] += forward[1] * (-ly) * speed;
      camera.pos[2] += forward[2] * (-ly) * speed + right[2] * lx * speed;
    }
    const rx = Math.abs(stickRight.dx) > DZ ? stickRight.dx : 0;
    const ry = Math.abs(stickRight.dy) > DZ ? stickRight.dy : 0;
    if (rx || ry) {
      const rotSpeed = 2.5 * dt;
      camera.yaw   -= rx * rotSpeed;
      camera.pitch -= ry * rotSpeed;
      camera.pitch = Math.max(-Math.PI*0.49, Math.min(Math.PI*0.49, camera.pitch));
    }
  }

  // Previous-frame camera pose, written into the uniform each frame so
  // restir_temporal can reproject world-space primary hits into the
  // previous frame's NDC. Initialized on first writeUniforms call.
  let prevCamState = null;

  function writeUniforms() {
    const { forward, right, up } = camVectors();
    // camera.fov is applied to the NARROW axis. The shader computes
    //   tan(H/2) = aspect * fovFactor   and   tan(V/2) = fovFactor
    // where aspect = width/height. So:
    //   portrait (aspect<1, narrow=horizontal): want tan(H/2)=tan(fov/2)
    //     → fovFactor = tan(fov/2)/aspect
    //   landscape (aspect>=1, narrow=vertical): want tan(V/2)=tan(fov/2)
    //     → fovFactor = tan(fov/2)
    const tanHalfFov = Math.tan((camera.fov * Math.PI / 180) * 0.5);
    const aspect = rw / rh;
    const fovFactor = (aspect < 1.0) ? (tanHalfFov / aspect) : tanHalfFov;
    // On the very first frame, prev = current. Reprojection will land
    // on-screen but gbuf_prev is zero-init (valid=0) so validation fails
    // and no reuse happens. Fine.
    if (prevCamState === null) {
      prevCamState = { pos: [...camera.pos], forward: [...forward], right: [...right], up: [...up], fovFactor };
    }
    const buf = new ArrayBuffer(192);
    const f = new Float32Array(buf);
    const u = new Uint32Array(buf);
    f[0] = rw; f[1] = rh;
    u[2] = frameIdx; u[3] = maxBounces;
    f[4]  = camera.pos[0]; f[5]  = camera.pos[1]; f[6]  = camera.pos[2];
    f[8]  = forward[0];    f[9]  = forward[1];    f[10] = forward[2];
    f[12] = right[0];      f[13] = right[1];      f[14] = right[2];
    f[16] = up[0];         f[17] = up[1];         f[18] = up[2];        f[19] = fovFactor;
    f[20] = sunDir[0];     f[21] = sunDir[1];     f[22] = sunDir[2];
    u[23] = framesStill;   // replaces former _pad3
    // BVH dequantization: scene_origin (vec3f + emissive_count) and scene_scale (vec3f + pad)
    f[24] = bvhScene.origin[0]; f[25] = bvhScene.origin[1]; f[26] = bvhScene.origin[2];
    u[27] = scene.stats.emissiveTris;  // count of entries in the emissive-tri CDF
    f[28] = bvhScene.scale[0];  f[29] = bvhScene.scale[1];  f[30] = bvhScene.scale[2];
    // Previous-frame camera pose, offsets 128-192 → float slots 32-47.
    f[32] = prevCamState.pos[0];     f[33] = prevCamState.pos[1];     f[34] = prevCamState.pos[2];
    f[36] = prevCamState.forward[0]; f[37] = prevCamState.forward[1]; f[38] = prevCamState.forward[2];
    f[40] = prevCamState.right[0];   f[41] = prevCamState.right[1];   f[42] = prevCamState.right[2];
    f[44] = prevCamState.up[0];      f[45] = prevCamState.up[1];      f[46] = prevCamState.up[2]; f[47] = prevCamState.fovFactor;
    device.queue.writeBuffer(uniformBuf, 0, buf);
    // Save this-frame pose for next frame's "prev".
    prevCamState = { pos: [...camera.pos], forward: [...forward], right: [...right], up: [...up], fovFactor };
  }

  function cameraMoved() {
    const moved =
      camera.pos[0] !== prevPos[0] ||
      camera.pos[1] !== prevPos[1] ||
      camera.pos[2] !== prevPos[2] ||
      camera.yaw !== prevYaw ||
      camera.pitch !== prevPitch;
    if (moved) {
      prevPos = [...camera.pos];
      prevYaw = camera.yaw;
      prevPitch = camera.pitch;
    }
    return moved;
  }

  function frame() {
    const now = performance.now();
    const dt = (now - lastTime) / 1000;
    lastTime = now;
    fpsAccum += dt; fpsCount++;
    if (fpsAccum >= 0.5) {
      fps = Math.round(fpsCount / fpsAccum); fpsAccum = 0; fpsCount = 0;
      if (frameIdx < 10 || frameIdx % 60 === 0)
        rlog(`FPS:${fps} frame:${frameIdx} cam:[${camera.pos.map(v=>v.toFixed(1))}]`);
    }

    updateCamera(dt);
    // framesStill = "samples prior to THIS frame". On the first frame ever,
    // or immediately after any camera motion, it's 0 → alpha=1 → composite
    // overwrites history entirely (so an uninitialized or stale accumulator
    // can't leak through). Incremented AFTER dispatches, below.
    // framesStill is only consumed by the legacy `composite` kernel
    // (not dispatched anymore). relax_temporal uses per-pixel
    // history_length packed in accum.a instead of a global counter.
    // accumFrame MUST keep alternating every frame — resetting it on
    // camera motion was causing the ping-pong to stall (read & write
    // the same buffer pair each frame while moving), so prev content
    // never refreshed and the frame appeared frozen once the camera
    // stopped. Per-pixel disocclusion in relax_temporal handles the
    // fresh-sample case locally.
    if (cameraMoved()) { framesStill = 0; }
    frameIdx++;
    writeUniforms();

    // Reset queue state for the frame: all pixels start alive in queue_a
    // (generate writes queue_a[idx]=idx), so count_a=W*H, count_b=0.
    // dispatch_args seeds the first bounce's indirect dispatch.
    const pixelCount = rw * rh;
    device.queue.writeBuffer(countsBuf, 0, new Uint32Array([pixelCount, 0]));
    device.queue.writeBuffer(dispatchArgsBuf, 0, new Uint32Array([Math.ceil(pixelCount / 64), 1, 1, 0]));

    const enc = device.createCommandEncoder();
    {
      const p = enc.beginComputePass();
      p.setPipeline(genPipeline);
      p.setBindGroup(0, bg0);
      p.setBindGroup(1, bg1);
      p.setBindGroup(2, bg2_main[currIdx]);
      p.dispatchWorkgroups(Math.ceil(rw/8), Math.ceil(rh/8));
      p.end();
    }
    for (let b = 0; b < maxBounces; b++) {
      const srcIsA = (b % 2) === 0;
      // bounce — indirect dispatch, 1D; bg2_main does NOT include
      // dispatchArgsBuf so the same buffer can be read as INDIRECT here
      // without a writable-usage sync conflict.
      // First bounce uses bounceFirst (skips queue read — every ray alive).
      {
        const p = enc.beginComputePass();
        const pipe = (b === 0) ? bouncePipelineFirst : (srcIsA ? bouncePipelineA : bouncePipelineB);
        p.setPipeline(pipe);
        p.setBindGroup(0, bg0);
        p.setBindGroup(1, bg1);
        p.setBindGroup(2, bg2_main[currIdx]);
        p.dispatchWorkgroupsIndirect(dispatchArgsBuf, 0);
        p.end();
      }
      // shadow_trace — still 2D over all pixels (consume flagged requests)
      {
        const p = enc.beginComputePass();
        p.setPipeline(shadowPipeline);
        p.setBindGroup(0, bg0);
        p.setBindGroup(1, bg1);
        p.setBindGroup(2, bg2_main[currIdx]);
        p.dispatchWorkgroups(Math.ceil(rw/8), Math.ceil(rh/8));
        p.end();
      }
      // prep_dispatch: update args + zero next-dst count for the NEXT
      // bounce. Uses bg2_prep which DOES have dispatch_args as storage
      // (only place that writes it). Skip after the final bounce.
      if (b < maxBounces - 1) {
        const nextReadsA = ((b + 1) % 2) === 0;
        const p = enc.beginComputePass();
        p.setPipeline(nextReadsA ? prepPipelineA : prepPipelineB);
        // prep_dispatch's layout is [null, null, bgl2_prep] — group 0/1
        // aren't accessed so we skip binding them (setting them would be
        // harmless but unnecessary).
        p.setBindGroup(2, bg2_prep[currIdx]);
        p.dispatchWorkgroups(1);
        p.end();
      }
    }
    // restir_temporal: reprojects current primary hits into prev frame's
    // NDC, validates history via normal + plane-distance + visibility
    // shadow ray, then WRS-merges the prev reservoir (with reconnection-
    // shift jacobian) against the current candidate. bg1 bound for the
    // shadow ray's BVH traversal. Writes to reservoirBuf[0] (temporal-out).
    {
      const p = enc.beginComputePass();
      p.setPipeline(temporalPipeline);
      p.setBindGroup(0, bg0_shade);
      p.setBindGroup(1, bg1);
      p.setBindGroup(2, bg2_main[currIdx]);
      p.setBindGroup(3, bg3_shade);
      p.dispatchWorkgroups(Math.ceil(rw/8), Math.ceil(rh/8));
      p.end();
    }
    // restir_spatial: k random neighbors merged with jacobian + visibility.
    // Under bg2_spatial mapping: reads reservoirBuf[0] (temporal-out) as
    // "prev", writes reservoirBuf[1] (spatial-out) as "curr".
    {
      const p = enc.beginComputePass();
      p.setPipeline(spatialPipeline);
      p.setBindGroup(0, bg0_shade);
      p.setBindGroup(1, bg1);
      p.setBindGroup(2, bg2_spatial[currIdx]);
      p.setBindGroup(3, bg3_shade);
      p.dispatchWorkgroups(Math.ceil(rw/8), Math.ceil(rh/8));
      p.end();
    }
    // restir_shade: reads reservoirBuf[1] (spatial-out, via bg2_spatial).
    // final = direct + albedo * source_pdf_sel * Lo * W.
    {
      const p = enc.beginComputePass();
      p.setPipeline(shadePipeline);
      p.setBindGroup(0, bg0_shade);
      p.setBindGroup(2, bg2_spatial[currIdx]);
      p.setBindGroup(3, bg3_shade);
      p.dispatchWorkgroups(Math.ceil(rw/8), Math.ceil(rh/8));
      p.end();
    }
    // relax_temporal: ReLAX-style per-pixel temporal accumulation.
    // Reprojects current primary hits into prev-frame NDC, validates
    // via plane distance + normal + in-screen, bilinear-samples
    // accum_prev with custom weights, blends with alpha driven by
    // per-pixel history length (packed in accum.a). Replaces the
    // frames_still-gated composite — camera motion no longer needs
    // a global reset since each pixel's disocclusion logic handles
    // it locally. bg2 carries gbuf_curr/prev as storage buffers.
    {
      const p = enc.beginComputePass();
      p.setPipeline(relaxTemporalPipeline);
      p.setBindGroup(0, bg0_composite);
      p.setBindGroup(2, bg2_main[currIdx]);
      p.setBindGroup(3, bg3[accumFrame]);
      p.dispatchWorkgroups(Math.ceil(rw/8), Math.ceil(rh/8));
      p.end();
    }
    {
      const view = ctx.getCurrentTexture().createView();
      const rp = enc.beginRenderPass({
        colorAttachments: [{ view, loadOp: 'clear', storeOp: 'store', clearValue: { r:0, g:0, b:0, a:1 } }],
      });
      rp.setPipeline(displayPipeline);
      rp.setBindGroup(0, displayBG[accumFrame]);
      rp.draw(3);
      rp.end();
    }
    // Swap ping-pong: next frame reads what we just wrote, writes the other
    accumFrame = 1 - accumFrame;
    // ReSTIR reservoir/gbuf ping-pong: next frame's "prev" is this frame's
    // "curr" (which we just wrote). Flip the index so bg2_main/bg2_prep
    // variants swap which buffer is which next frame.
    currIdx = 1 - currIdx;
    // We just added one sample to the accumulator
    framesStill++;
    device.queue.submit([enc.finish()]);

    info.innerHTML =
      `<b>Ignis Wavefront</b> | ${rw}x${rh} → ${canvas.width}x${canvas.height}<br>` +
      `FPS:${fps} frame:${frameIdx} bounces:${maxBounces}`;

    if (frameIdx === 1) rlog('First frame OK');
    requestAnimationFrame(frame);
  }

  info.textContent = 'Ready.';
  rlog('Wavefront renderer started');
  requestAnimationFrame(frame);
}

init().catch(err => {
  rlog('FATAL:', err.message, err.stack);
  showError(`Error: ${err.message}`);
});
