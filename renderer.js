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
    usage: GPUTextureUsage.STORAGE_BINDING,
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

  // Uniforms — 96 bytes, matches Uniforms struct in wavefront.wgsl.
  // 128 bytes: base 96 + 32 for scene_origin/scene_scale (BVH dequant).
  const uniformBuf = device.createBuffer({
    size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Shader modules
  const wavefrontSrc = await fetch('wavefront.wgsl').then(r => r.text());
  const wavefrontModule = device.createShaderModule({ code: wavefrontSrc });
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
    ],
  });
  // bgl2_main: queue_a, queue_b, counts. NO dispatch_args — so when a
  // bounce kernel dispatches indirectly off dispatchArgsBuf the buffer
  // isn't also bound as storage in the same pass (WebGPU forbids
  // writable + indirect in the same sync scope).
  const bgl2_main = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });
  // bgl2_prep: same + dispatch_args. Used only by prep_dispatch which
  // doesn't dispatch anything (writes args for the NEXT pass).
  const bgl2_prep = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // dispatch_args
    ],
  });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl0, bgl1, bgl2_main] });
  const prepPipeLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl0, bgl1, bgl2_prep] });

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
    ],
  });
  const bg2_main = device.createBindGroup({
    layout: bgl2_main,
    entries: [
      { binding: 0, resource: { buffer: queueABuf } },
      { binding: 1, resource: { buffer: queueBBuf } },
      { binding: 2, resource: { buffer: countsBuf } },
    ],
  });
  const bg2_prep = device.createBindGroup({
    layout: bgl2_prep,
    entries: [
      { binding: 0, resource: { buffer: queueABuf } },
      { binding: 1, resource: { buffer: queueBBuf } },
      { binding: 2, resource: { buffer: countsBuf } },
      { binding: 3, resource: { buffer: dispatchArgsBuf } },
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
    const buf = new ArrayBuffer(128);
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
    // BVH dequantization: scene_origin (vec3f + pad) and scene_scale (vec3f + pad)
    f[24] = bvhScene.origin[0]; f[25] = bvhScene.origin[1]; f[26] = bvhScene.origin[2];
    f[28] = bvhScene.scale[0];  f[29] = bvhScene.scale[1];  f[30] = bvhScene.scale[2];
    device.queue.writeBuffer(uniformBuf, 0, buf);
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
    if (cameraMoved()) { framesStill = 0; accumFrame = 0; }
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
      p.setBindGroup(2, bg2_main);
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
        p.setBindGroup(2, bg2_main);
        p.dispatchWorkgroupsIndirect(dispatchArgsBuf, 0);
        p.end();
      }
      // shadow_trace — still 2D over all pixels (consume flagged requests)
      {
        const p = enc.beginComputePass();
        p.setPipeline(shadowPipeline);
        p.setBindGroup(0, bg0);
        p.setBindGroup(1, bg1);
        p.setBindGroup(2, bg2_main);
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
        p.setBindGroup(0, bg0);
        p.setBindGroup(1, bg1);
        p.setBindGroup(2, bg2_prep);
        p.dispatchWorkgroups(1);
        p.end();
      }
    }
    {
      const p = enc.beginComputePass();
      p.setPipeline(finPipeline);
      p.setBindGroup(0, bg0);
      p.setBindGroup(1, bg1);
      p.setBindGroup(2, bg2_main);
      p.dispatchWorkgroups(Math.ceil(rw/8), Math.ceil(rh/8));
      p.end();
    }
    // composite: blend current noisy with previous accumulator → new accum.
    // Uses minimal bg0_composite (uniforms only) + bg3. Groups 1 and 2 are
    // null in the composite pipeline layout so they're not set here.
    {
      const p = enc.beginComputePass();
      p.setPipeline(compositePipeline);
      p.setBindGroup(0, bg0_composite);
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
