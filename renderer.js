// WebGPU Monte Carlo Path Tracer - Renderer (Sponza GLTF + BVH)

import { loadScene } from './scene-loader.js';

// Remote logging — sends to Python server, viewable in client.log
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

  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBuffersPerShaderStage: Math.min(8, adapter.limits.maxStorageBuffersPerShaderStage),
    }
  });
  device.onuncapturederror = (e) => rlog('GPU_ERROR: ' + e.error.message);
  rlog('Canvas format: ' + navigator.gpu.getPreferredCanvasFormat());
  rlog('Max storage buffers/stage: ' + device.limits.maxStorageBuffersPerShaderStage);
  device.lost.then(info => rlog('DEVICE_LOST: ' + info.message));

  // --- Load scene ---
  info.textContent = 'Loading Sponza scene...';
  let scene;
  try {
    scene = await loadScene('scene', msg => { info.textContent = msg; });
  } catch (e) {
    showError(`Scene load failed: ${e.message}`);
    console.error(e);
    return;
  }
  const stats = scene.stats;
  rlog('Scene loaded:', stats);
  rlog('TriFlat:', scene.gpuTriFlat.byteLength, 'bytes | BVH:', scene.gpuBVHNodes.byteLength, 'bytes');

  // --- Resolution setup (FSR modes) ---
  const canvas = document.getElementById('canvas');
  const context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();

  // FSR quality modes
  const FSR_MODES = {
    performance: { scale: 0.50, label: 'Performance (2x)' },
    balanced:    { scale: 0.58, label: 'Balanced (1.7x)' },
    quality:     { scale: 0.67, label: 'Quality (1.5x)' },
    dlaa:        { scale: 1.00, label: 'DLAA (1x, AA only)' },
  };
  const hashMode = location.hash.replace('#', '');
  let fsrMode = FSR_MODES[hashMode] ? hashMode : 'balanced';

  // Display resolution — native device resolution
  const dpr = window.devicePixelRatio || 1;
  const displayWidth = Math.ceil(Math.floor(window.innerWidth * dpr) / 8) * 8;
  const displayHeight = Math.ceil(Math.floor(window.innerHeight * dpr) / 8) * 8;
  canvas.width = displayWidth;
  canvas.height = displayHeight;
  context.configure({ device, format, alphaMode: 'opaque' });

  // Internal resolution from FSR mode
  let fsrScale = FSR_MODES[fsrMode].scale;
  let width = Math.ceil(displayWidth * fsrScale / 8) * 8;
  let height = Math.ceil(displayHeight * fsrScale / 8) * 8;
  let fsrRatio = displayWidth / width;

  rlog(`FSR ${fsrMode}: internal ${width}x${height} → display ${displayWidth}x${displayHeight} (${fsrRatio.toFixed(1)}x)`);

  // --- Load shaders ---
  const v = Date.now(); // cache bust
  const [ptCode, dispCode, fsrCode, dnCode, tmpCode] = await Promise.all([
    fetch(`pathtracer.wgsl?v=${v}`).then(r => r.text()),
    fetch(`display.wgsl?v=${v}`).then(r => r.text()),
    fetch(`fsr.wgsl?v=${v}`).then(r => r.text()),
    fetch(`denoise.wgsl?v=${v}`).then(r => r.text()),
    fetch(`temporal.wgsl?v=${v}`).then(r => r.text()),
  ]);
  const ptModule = device.createShaderModule({ code: ptCode });
  const dispModule = device.createShaderModule({ code: dispCode });
  const fsrModule = device.createShaderModule({ code: fsrCode });
  const dnModule = device.createShaderModule({ code: dnCode });
  const tmpModule = device.createShaderModule({ code: tmpCode });

  // Check shader compilation
  for (const [name, mod] of [['pathtracer',ptModule],['display',dispModule],['fsr',fsrModule],['denoise',dnModule],['temporal',tmpModule]]) {
    const ci = await mod.getCompilationInfo();
    for (const m of ci.messages) {
      if (m.type === 'error') {
        const err = `${name}.wgsl:${m.lineNum}:${m.linePos} ${m.message}`;
        rlog('SHADER ERROR:', err);
        showError(`Shader: ${err}`);
        return;
      }
      if (m.type === 'warning') rlog(`SHADER WARN: ${name}:${m.lineNum} ${m.message}`);
    }
  }
  rlog(`Shaders OK | Internal:${width}x${height} Display:${displayWidth}x${displayHeight} FSR:${fsrRatio.toFixed(1)}x`);

  // --- Uniform buffer ---
  // Added: sun_dir (vec3f) + emissive_tri_count (u32) = 16 more bytes
  const uniformBufferSize = 128;
  const uniformBuffer = device.createBuffer({
    size: uniformBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // --- Internal textures ---
  const F16 = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING;
  const F16C = F16 | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST; // copyable
  const noisyTex = device.createTexture({ size:[width,height], format:'rgba16float', usage:F16 });   // PT 1spp raw
  const ndTex    = device.createTexture({ size:[width,height], format:'rgba16float', usage:F16 });   // normal+depth
  const hdrTex   = device.createTexture({ size:[width,height], format:'rgba16float', usage:F16 });   // temporal accumulated
  const pingTex  = device.createTexture({ size:[width,height], format:'rgba16float', usage:F16 });   // à-trous ping
  const pongTex  = device.createTexture({ size:[width,height], format:'rgba16float', usage:F16 });   // à-trous pong
  const historyA = device.createTexture({ size:[width,height], format:'rgba16float', usage:F16C });   // prev denoised (read)
  const historyB = device.createTexture({ size:[width,height], format:'rgba16float', usage:F16C });   // prev denoised (write)
  const ptOutputTex = device.createTexture({
    size:[width,height], format:'rgba8unorm',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });

  // --- FSR textures (display resolution) ---
  const upscaledTex = device.createTexture({
    size: [displayWidth, displayHeight], format: 'rgba8unorm',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });
  const finalTex = device.createTexture({
    size: [displayWidth, displayHeight], format: 'rgba8unorm',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });

  // --- FSR uniform buffer ---
  const fsrParamsBuf = device.createBuffer({
    size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const fsrParamsData = new Float32Array([
    width, height,               // input_size
    displayWidth, displayHeight, // output_size
    0.6,                         // sharpness (0=off, 1=max)
    0, 0, 0,                     // padding
  ]);
  device.queue.writeBuffer(fsrParamsBuf, 0, fsrParamsData);

  // --- Load textures into GPU texture array ---
  const TEX_SIZE = 1024;
  const texInfo = scene.textureInfo;
  const texCount = texInfo ? texInfo.count : 0;
  let texArray;

  if (texCount > 0) {
    texArray = device.createTexture({
      size: [TEX_SIZE, TEX_SIZE, texCount],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const BATCH = 6;
    for (let b = 0; b < texCount; b += BATCH) {
      const batch = [];
      for (let i = b; i < Math.min(b + BATCH, texCount); i++) {
        batch.push(
          fetch(`scene/${texInfo.imageURIs[i]}`)
            .then(r => r.blob())
            .then(blob => createImageBitmap(blob, {
              resizeWidth: TEX_SIZE, resizeHeight: TEX_SIZE,
              resizeQuality: 'high',
            }))
            .then(bmp => {
              device.queue.copyExternalImageToTexture(
                { source: bmp },
                { texture: texArray, origin: [0, 0, i] },
                [TEX_SIZE, TEX_SIZE]
              );
              bmp.close();
            })
        );
      }
      await Promise.all(batch);
      info.textContent = `Loading textures... ${Math.min(b + BATCH, texCount)}/${texCount}`;
    }
    rlog(`Loaded ${texCount} textures at ${TEX_SIZE}x${TEX_SIZE}`);
  } else {
    texArray = device.createTexture({
      size: [1, 1, 1], format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture({ texture: texArray }, new Uint8Array([128,128,128,255]), { bytesPerRow: 4 }, [1,1,1]);
  }

  // --- Scene GPU buffers ---
  info.textContent = 'Uploading to GPU...';
  const vtxBuf = createGPUBuffer(device, scene.gpuPositions, GPUBufferUsage.STORAGE);
  const nrmBuf = createGPUBuffer(device, scene.gpuNormals, GPUBufferUsage.STORAGE);
  const triBuf = createGPUBuffer(device, scene.gpuTriData, GPUBufferUsage.STORAGE);
  const bvhBuf = createGPUBuffer(device, scene.gpuBVHNodes, GPUBufferUsage.STORAGE);
  const matBuf = createGPUBuffer(device, scene.gpuMaterials, GPUBufferUsage.STORAGE);
  const emsBuf = createGPUBuffer(device, scene.gpuEmissiveTris, GPUBufferUsage.STORAGE);

  // --- Bind group 0: uniforms + accumulation + output ---
  const bg0Layout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
    ],
  });
  const bg0 = device.createBindGroup({
    layout: bg0Layout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: noisyTex.createView() },
      { binding: 2, resource: ndTex.createView() },
    ],
  });

  // --- Bind group 1: scene data ---
  const bg1Layout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    ],
  });
  const bg1 = device.createBindGroup({
    layout: bg1Layout,
    entries: [
      { binding: 0, resource: { buffer: vtxBuf } },
      { binding: 1, resource: { buffer: nrmBuf } },
      { binding: 2, resource: { buffer: triBuf } },
      { binding: 3, resource: { buffer: bvhBuf } },
      { binding: 4, resource: { buffer: matBuf } },
      { binding: 5, resource: { buffer: emsBuf } },
    ],
  });

  // --- SHaRC radiance cache ---
  // SHaRC radiance cache — keys+accum packed in 1 buffer, resolved in another
  // keys_accum: [0..cap) = keys, [cap..cap+cap*4) = accum RGBS = total cap*5 u32s
  // resolved: [0..cap*4) = resolved RGBS = total cap*4 u32s
  const SHARC_CAPACITY = 131072;
  const sharcParamBuf = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const sharcKeysAccumBuf = device.createBuffer({ size: SHARC_CAPACITY * 5 * 4, usage: GPUBufferUsage.STORAGE });
  const sharcResolvedBuf = device.createBuffer({ size: SHARC_CAPACITY * 4 * 4, usage: GPUBufferUsage.STORAGE });

  // PT reads resolved as read-only, writes keys_accum as read_write
  const bg2Layout = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },        // keys_accum rw
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // resolved ro
  ]});
  const bg2 = device.createBindGroup({ layout: bg2Layout, entries: [
    { binding: 0, resource: { buffer: sharcParamBuf } },
    { binding: 1, resource: { buffer: sharcKeysAccumBuf } },
    { binding: 2, resource: { buffer: sharcResolvedBuf } },
  ]});

  // Resolve needs rw on both
  const bg2ResolveLayout = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
  ]});
  const sharcModule = device.createShaderModule({ code: await fetch(`sharc.wgsl?v=${v}`).then(r=>r.text()) });
  const sharcCI = await sharcModule.getCompilationInfo();
  for (const m of sharcCI.messages) { if(m.type==='error') rlog('SHARC SHADER ERROR:',m.lineNum,m.message); }

  const sharcResolvePipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bg2ResolveLayout] }),
    compute: { module: sharcModule, entryPoint: 'resolve' },
  });
  const bg2Resolve = device.createBindGroup({ layout: bg2ResolveLayout, entries: [
    { binding: 0, resource: { buffer: sharcParamBuf } },
    { binding: 1, resource: { buffer: sharcKeysAccumBuf } },
    { binding: 2, resource: { buffer: sharcResolvedBuf } },
  ]});

  // --- Bind group 3: texture array ---
  const texSampler = device.createSampler({
    magFilter: 'linear', minFilter: 'linear',
    addressModeU: 'repeat', addressModeV: 'repeat',
  });
  const bg3Layout = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d-array' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
  ]});
  const bg3 = device.createBindGroup({ layout: bg3Layout, entries: [
    { binding: 0, resource: texArray.createView({ dimension: '2d-array' }) },
    { binding: 1, resource: texSampler },
  ]});

  // --- Compute pipeline (PT + SHaRC + Textures) ---
  const computePipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bg0Layout, bg1Layout, bg2Layout, bg3Layout] }),
    compute: { module: ptModule, entryPoint: 'main' },
  });

  // --- FSR pipelines (EASU + RCAS) ---
  const fsrBGLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba8unorm' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });
  const fsrPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [fsrBGLayout] });

  const easuPipeline = device.createComputePipeline({
    layout: fsrPipelineLayout,
    compute: { module: fsrModule, entryPoint: 'easu' },
  });
  const rcasPipeline = device.createComputePipeline({
    layout: fsrPipelineLayout,
    compute: { module: fsrModule, entryPoint: 'rcas' },
  });

  const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

  // EASU: reads ptOutput (internal), writes upscaled (display)
  const easuBG = device.createBindGroup({
    layout: fsrBGLayout,
    entries: [
      { binding: 0, resource: ptOutputTex.createView() },
      { binding: 1, resource: upscaledTex.createView() },
      { binding: 2, resource: sampler },
      { binding: 3, resource: { buffer: fsrParamsBuf } },
    ],
  });

  // RCAS: reads upscaled, writes final
  const rcasBG = device.createBindGroup({
    layout: fsrBGLayout,
    entries: [
      { binding: 0, resource: upscaledTex.createView() },
      { binding: 1, resource: finalTex.createView() },
      { binding: 2, resource: sampler },
      { binding: 3, resource: { buffer: fsrParamsBuf } },
    ],
  });

  // --- Temporal reprojection pipeline ---
  const tmpBufSize = 4*4 * 10; // 10 vec4f = 160 bytes
  const tmpBuf = device.createBuffer({ size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const tmpLayout = device.createBindGroupLayout({ entries: [
    { binding:0, visibility:GPUShaderStage.COMPUTE, buffer:{type:'uniform'} },
    { binding:1, visibility:GPUShaderStage.COMPUTE, texture:{sampleType:'float'} },          // current noisy
    { binding:2, visibility:GPUShaderStage.COMPUTE, texture:{sampleType:'float'} },          // prev denoised
    { binding:3, visibility:GPUShaderStage.COMPUTE, texture:{sampleType:'float'} },          // depth (ndTex)
    { binding:4, visibility:GPUShaderStage.COMPUTE, storageTexture:{access:'write-only',format:'rgba16float'} }, // accum out
    { binding:5, visibility:GPUShaderStage.COMPUTE, storageTexture:{access:'write-only',format:'rgba16float'} }, // history out
    { binding:6, visibility:GPUShaderStage.COMPUTE, sampler:{type:'filtering'} },
  ]});
  const tmpPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts:[tmpLayout] }),
    compute: { module:tmpModule, entryPoint:'temporal' },
  });
  // Two bind groups: frame A reads historyA, writes historyB; frame B reads historyB, writes historyA
  const tmpBG_A = device.createBindGroup({ layout:tmpLayout, entries:[
    { binding:0, resource:{buffer:tmpBuf} },
    { binding:1, resource:noisyTex.createView() },
    { binding:2, resource:historyA.createView() },
    { binding:3, resource:ndTex.createView() },
    { binding:4, resource:hdrTex.createView() },
    { binding:5, resource:historyB.createView() },
    { binding:6, resource:sampler },
  ]});
  const tmpBG_B = device.createBindGroup({ layout:tmpLayout, entries:[
    { binding:0, resource:{buffer:tmpBuf} },
    { binding:1, resource:noisyTex.createView() },
    { binding:2, resource:historyB.createView() },
    { binding:3, resource:ndTex.createView() },
    { binding:4, resource:hdrTex.createView() },
    { binding:5, resource:historyA.createView() },
    { binding:6, resource:sampler },
  ]});

  // Previous camera state for temporal reprojection
  let prevCam = { pos:[0,5,0], right:[1,0,0], up:[0,1,0], fwd:[0,0,-1] };
  let historyFrame = 0; // alternates 0/1 for double-buffering

  // --- Denoiser pipelines ---
  // 3 separate param buffers for the 3 à-trous steps (avoid writeBuffer during encoding)
  const dnParamBufs = [0,1,2].map(() => device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
  device.queue.writeBuffer(dnParamBufs[0], 0, new Float32Array([width, height, 1, 0]));
  device.queue.writeBuffer(dnParamBufs[1], 0, new Float32Array([width, height, 2, 0]));
  device.queue.writeBuffer(dnParamBufs[2], 0, new Float32Array([width, height, 4, 0]));
  const dnCompParamBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(dnCompParamBuf, 0, new Float32Array([width, height, 0, 0]));

  // À-trous layout: params + input(tex) + output(storage) + normalDepth(tex)
  const dnAtrousLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
    ],
  });
  const dnAtrousPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [dnAtrousLayout] }),
    compute: { module: dnModule, entryPoint: 'atrous' },
  });

  // Composite layout: reuses atrous bindings 0-3 + adds binding 4 for LDR output
  const dnCompLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } }, // unused but needed
      { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba8unorm' } },
    ],
  });
  const dnCompPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [dnCompLayout] }),
    compute: { module: dnModule, entryPoint: 'composite' },
  });

  // À-trous ping-pong bind groups:
  // Pass 1: hdr → ping
  const dnBG_hdr2ping = device.createBindGroup({ layout: dnAtrousLayout, entries: [
    { binding: 0, resource: { buffer: dnParamBufs[0] } },
    { binding: 1, resource: hdrTex.createView() },
    { binding: 2, resource: pingTex.createView() },
    { binding: 3, resource: ndTex.createView() },
  ]});
  const dnBG_ping2pong = device.createBindGroup({ layout: dnAtrousLayout, entries: [
    { binding: 0, resource: { buffer: dnParamBufs[1] } },
    { binding: 1, resource: pingTex.createView() },
    { binding: 2, resource: pongTex.createView() },
    { binding: 3, resource: ndTex.createView() },
  ]});
  const dnBG_pong2ping = device.createBindGroup({ layout: dnAtrousLayout, entries: [
    { binding: 0, resource: { buffer: dnParamBufs[2] } },
    { binding: 1, resource: pongTex.createView() },
    { binding: 2, resource: pingTex.createView() },
    { binding: 3, resource: ndTex.createView() },
  ]});
  const dnBG_comp = device.createBindGroup({ layout: dnCompLayout, entries: [
    { binding: 0, resource: { buffer: dnCompParamBuf } },
    { binding: 1, resource: pingTex.createView() },
    { binding: 2, resource: pongTex.createView() },
    { binding: 3, resource: ndTex.createView() },
    { binding: 4, resource: ptOutputTex.createView() },
  ]});

  // --- Display pipeline (reads FSR final output) ---
  const dispBGLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  });
  const displayPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [dispBGLayout] }),
    vertex: { module: dispModule, entryPoint: 'vs_main' },
    fragment: { module: dispModule, entryPoint: 'fs_main', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });
  const dispBG = device.createBindGroup({
    layout: dispBGLayout,
    entries: [
      { binding: 0, resource: finalTex.createView() },
      { binding: 1, resource: sampler },
    ],
  });
  // DEBUG: bypass FSR, show ptOutput directly
  const dispBGDirect = device.createBindGroup({
    layout: dispBGLayout,
    entries: [
      { binding: 0, resource: ptOutputTex.createView() },
      { binding: 1, resource: sampler },
    ],
  });

  // --- Camera (positioned inside Sponza) ---
  const scCenter = stats.sceneMin.map((v, i) => (v + stats.sceneMax[i]) / 2);
  const camera = {
    pos: [0, 5, 0],
    yaw: Math.PI,
    pitch: -0.1,
    fov: 75,
    speed: 3.0,
    sensitivity: 0.003,
  };

  // --- Settings (menu-controlled) ---
  const settings = {
    sunElevation: 58,
    sunAzimuth: 53,
    sharpness: 0.6,
    temporalAlpha: 0.02,
  };

  function getSunDir() {
    const el = settings.sunElevation * Math.PI / 180;
    const az = settings.sunAzimuth * Math.PI / 180;
    return [Math.sin(az) * Math.cos(el), Math.sin(el), Math.cos(az) * Math.cos(el)];
  }

  // --- Input handling ---
  const keys = {};
  let pointerLocked = false;
  let menuOpen = false;
  const menuEl = document.getElementById('options-menu');

  function toggleMenu() {
    menuOpen = !menuOpen;
    menuEl.classList.toggle('open', menuOpen);
    if (menuOpen) {
      document.exitPointerLock?.();
      for (const k in keys) keys[k] = false;
      hideStick(stickLeft);
      hideStick(stickRight);
      refreshMenu();
    }
  }

  document.addEventListener('keydown', e => {
    if (e.code === 'Escape') { toggleMenu(); e.preventDefault(); return; }
    if (!menuOpen) keys[e.code] = true;
  });
  document.addEventListener('keyup', e => { keys[e.code] = false; });

  const isTouchDevice = ('ontouchstart' in window) || navigator.maxTouchPoints > 0;

  canvas.addEventListener('click', () => {
    if (!isTouchDevice && !menuOpen) canvas.requestPointerLock();
  });
  document.addEventListener('pointerlockchange', () => {
    pointerLocked = document.pointerLockElement === canvas;
  });
  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    if (menuOpen) return;
    camera.speed *= e.deltaY > 0 ? 0.85 : 1.18;
    camera.speed = Math.max(0.1, Math.min(50, camera.speed));
  }, {passive: false});

  document.addEventListener('mousemove', e => {
    if (!pointerLocked || menuOpen) return;
    camera.yaw += e.movementX * camera.sensitivity;
    camera.pitch -= e.movementY * camera.sensitivity;
    camera.pitch = Math.max(-Math.PI * 0.49, Math.min(Math.PI * 0.49, camera.pitch));
    cameraMoved = true;
  });

  // --- Virtual joysticks (relative: touch anywhere, that point becomes center) ---
  const stickLeft  = { el: document.getElementById('stick-left'),  knob: document.getElementById('knob-left'),  touchId: null, cx:0, cy:0, dx:0, dy:0 };
  const stickRight = { el: document.getElementById('stick-right'), knob: document.getElementById('knob-right'), touchId: null, cx:0, cy:0, dx:0, dy:0 };
  const KNOB_MAX = 50;
  const halfW = () => window.innerWidth / 2;

  function clampStick(dx,dy) { const l=Math.sqrt(dx*dx+dy*dy); if(l>KNOB_MAX){dx=dx/l*KNOB_MAX;dy=dy/l*KNOB_MAX;} return{dx,dy}; }

  function showStick(s, x, y) {
    s.el.style.display = 'block';
    s.el.style.left = (x - 70) + 'px';
    s.el.style.top = (y - 70) + 'px';
    s.el.style.bottom = 'auto';
    s.el.style.right = 'auto';
    s.knob.style.transform = 'translate(-50%,-50%)';
    s.knob.classList.remove('active');
  }
  function hideStick(s) {
    s.touchId = null; s.dx = 0; s.dy = 0;
    s.el.style.display = 'none';
    s.knob.style.transform = 'translate(-50%,-50%)';
    s.knob.classList.remove('active');
  }
  function moveKnob(s, tx, ty) {
    const r = clampStick(tx - s.cx, ty - s.cy);
    s.knob.style.transform = `translate(calc(-50% + ${r.dx}px), calc(-50% + ${r.dy}px))`;
    s.knob.classList.toggle('active', r.dx !== 0 || r.dy !== 0);
    s.dx = r.dx / KNOB_MAX;
    s.dy = r.dy / KNOB_MAX;
  }

  // Initially hidden
  stickLeft.el.style.display = 'none';
  stickRight.el.style.display = 'none';

  let threeFingerStart = null;

  document.addEventListener('touchstart', e => {
    if (e.touches.length >= 3) {
      threeFingerStart = performance.now();
      hideStick(stickLeft);
      hideStick(stickRight);
      e.preventDefault();
      return;
    }
    if (menuOpen) return;
    for (const t of e.changedTouches) {
      const isLeft = t.clientX < halfW();
      const stick = isLeft ? stickLeft : stickRight;
      if (stick.touchId !== null) continue;
      stick.touchId = t.identifier;
      stick.cx = t.clientX;
      stick.cy = t.clientY;
      showStick(stick, t.clientX, t.clientY);
      e.preventDefault();
    }
  }, {passive: false});

  document.addEventListener('touchmove', e => {
    if (threeFingerStart !== null && e.touches.length >= 3) {
      threeFingerStart = null; // drag, not a tap
    }
    if (menuOpen) return;
    for (const t of e.changedTouches) {
      if (t.identifier === stickLeft.touchId) { moveKnob(stickLeft, t.clientX, t.clientY); e.preventDefault(); }
      else if (t.identifier === stickRight.touchId) { moveKnob(stickRight, t.clientX, t.clientY); e.preventDefault(); }
    }
  }, {passive: false});

  function onTouchEnd(e) {
    if (threeFingerStart !== null && e.touches.length === 0) {
      const elapsed = performance.now() - threeFingerStart;
      threeFingerStart = null;
      if (elapsed < 500) { toggleMenu(); return; }
    }
    for (const t of e.changedTouches) {
      if (t.identifier === stickLeft.touchId) hideStick(stickLeft);
      if (t.identifier === stickRight.touchId) hideStick(stickRight);
    }
  }
  document.addEventListener('touchend', onTouchEnd);
  document.addEventListener('touchcancel', onTouchEnd);

  // --- Menu controls ---
  document.getElementById('menu-close').addEventListener('click', toggleMenu);

  const optFsr = document.getElementById('opt-fsr');
  optFsr.value = fsrMode;
  optFsr.addEventListener('change', e => {
    location.hash = e.target.value;
    location.reload();
  });

  function bindSlider(id, valId, getter, setter, formatter) {
    const sl = document.getElementById(id);
    const vl = document.getElementById(valId);
    sl.addEventListener('input', () => {
      setter(Number(sl.value));
      vl.textContent = formatter(Number(sl.value));
    });
  }

  bindSlider('opt-sharp', 'val-sharp',
    () => settings.sharpness * 100,
    v => { settings.sharpness = v / 100; fsrParamsData[4] = settings.sharpness; device.queue.writeBuffer(fsrParamsBuf, 0, fsrParamsData); },
    v => (v / 100).toFixed(2));

  bindSlider('opt-sun-el', 'val-sun-el',
    () => settings.sunElevation,
    v => settings.sunElevation = v,
    v => v + '\u00B0');

  bindSlider('opt-sun-az', 'val-sun-az',
    () => settings.sunAzimuth,
    v => settings.sunAzimuth = v,
    v => v + '\u00B0');

  bindSlider('opt-speed', 'val-speed',
    () => camera.speed * 10,
    v => camera.speed = v / 10,
    v => (v / 10).toFixed(1));

  bindSlider('opt-fov', 'val-fov',
    () => camera.fov,
    v => camera.fov = v,
    v => v + '\u00B0');

  bindSlider('opt-temporal', 'val-temporal',
    () => settings.temporalAlpha * 1000,
    v => settings.temporalAlpha = v / 1000,
    v => (v / 1000).toFixed(3));

  function refreshMenu() {
    const sets = [
      ['opt-sharp', 'val-sharp', settings.sharpness * 100, v => (v / 100).toFixed(2)],
      ['opt-sun-el', 'val-sun-el', settings.sunElevation, v => v + '\u00B0'],
      ['opt-sun-az', 'val-sun-az', settings.sunAzimuth, v => v + '\u00B0'],
      ['opt-speed', 'val-speed', camera.speed * 10, v => (v / 10).toFixed(1)],
      ['opt-fov', 'val-fov', camera.fov, v => v + '\u00B0'],
      ['opt-temporal', 'val-temporal', settings.temporalAlpha * 1000, v => (v / 1000).toFixed(3)],
    ];
    for (const [sid, vid, val, fmt] of sets) {
      document.getElementById(sid).value = val;
      document.getElementById(vid).textContent = fmt(val);
    }
    optFsr.value = fsrMode;
  }

  // --- Gizmo ---
  const gizmoCanvas = document.getElementById('gizmo');
  const gctx = gizmoCanvas.getContext('2d');
  const GZ=120, GC=GZ/2, GAXIS_LEN=40;
  const gdpr = window.devicePixelRatio||1;
  gizmoCanvas.width=GZ*gdpr; gizmoCanvas.height=GZ*gdpr; gctx.scale(gdpr,gdpr);
  const gizmoAxes=[{label:'X',color:'#E84545',neg:'-X',nc:'#7a2222'},{label:'Y',color:'#45E845',neg:'-Y',nc:'#227a22'},{label:'Z',color:'#4585E8',neg:'-Z',nc:'#22447a'}];

  function drawGizmo() {
    gctx.save(); gctx.setTransform(gdpr,0,0,gdpr,0,0); gctx.clearRect(0,0,GZ,GZ);
    gctx.beginPath(); gctx.arc(GC,GC,GC-4,0,Math.PI*2); gctx.fillStyle='rgba(30,30,30,0.65)'; gctx.fill();
    gctx.strokeStyle='rgba(255,255,255,0.1)'; gctx.lineWidth=1; gctx.stroke();
    const cy=Math.cos(camera.yaw),sy=Math.sin(camera.yaw),cp=Math.cos(camera.pitch),sp=Math.sin(camera.pitch);
    function proj(wx,wy,wz){return{x:cy*wx-sy*wz,y:-((-sp*sy)*wx+cp*wy+(-sp*cy)*wz),z:cp*sy*wx+sp*wy+cp*cy*wz};}
    const eps=[];
    [[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]].forEach(([wx,wy,wz],i)=>{
      const p=proj(wx,wy,wz);const pos=i<3;const ai=i%3;
      eps.push({sx:GC+p.x*GAXIS_LEN,sy:GC+p.y*GAXIS_LEN,z:p.z,label:pos?gizmoAxes[ai].label:gizmoAxes[ai].neg,color:pos?gizmoAxes[ai].color:gizmoAxes[ai].nc,pos});
    });
    eps.sort((a,b)=>a.z-b.z);
    for(const e of eps){
      gctx.beginPath();gctx.moveTo(GC,GC);gctx.lineTo(e.sx,e.sy);gctx.strokeStyle=e.color;gctx.lineWidth=e.pos?2.5:1.5;gctx.globalAlpha=e.pos?1:0.5;gctx.stroke();
      gctx.beginPath();gctx.arc(e.sx,e.sy,e.pos?10:6,0,Math.PI*2);gctx.fillStyle=e.color;gctx.globalAlpha=e.pos?1:0.45;gctx.fill();
      gctx.fillStyle=e.pos?'#fff':'rgba(255,255,255,0.7)';gctx.globalAlpha=1;gctx.font=e.pos?'bold 11px Courier New':'9px Courier New';gctx.textAlign='center';gctx.textBaseline='middle';gctx.fillText(e.label,e.sx,e.sy);
    }
    gctx.restore();
  }

  // --- Render state ---
  let frameIndex = 0, cameraMoved = false;
  let lastTime = performance.now(), fps = 0, fpsAccum = 0, fpsCount = 0;

  function getCameraVectors() {
    const fw = [Math.cos(camera.pitch)*Math.sin(camera.yaw), Math.sin(camera.pitch), Math.cos(camera.pitch)*Math.cos(camera.yaw)];
    const rt = [Math.cos(camera.yaw), 0, -Math.sin(camera.yaw)];
    const up = [-Math.sin(camera.pitch)*Math.sin(camera.yaw), Math.cos(camera.pitch), -Math.sin(camera.pitch)*Math.cos(camera.yaw)];
    return { forward:fw, right:rt, up };
  }

  function updateCamera(dt) {
    if (menuOpen) return;
    const {forward,right} = getCameraVectors();
    const speed = camera.speed * dt;
    let moved = false;
    if(keys['KeyW']||keys['ArrowUp']){camera.pos[0]+=forward[0]*speed;camera.pos[1]+=forward[1]*speed;camera.pos[2]+=forward[2]*speed;moved=true;}
    if(keys['KeyS']||keys['ArrowDown']){camera.pos[0]-=forward[0]*speed;camera.pos[1]-=forward[1]*speed;camera.pos[2]-=forward[2]*speed;moved=true;}
    if(keys['KeyA']||keys['ArrowLeft']){camera.pos[0]-=right[0]*speed;camera.pos[2]-=right[2]*speed;moved=true;}
    if(keys['KeyD']||keys['ArrowRight']){camera.pos[0]+=right[0]*speed;camera.pos[2]+=right[2]*speed;moved=true;}
    if(keys['KeyE']){camera.pos[1]+=speed;moved=true;}
    if(keys['KeyQ']){camera.pos[1]-=speed;moved=true;}
    const DZ=0.12;
    const lx=Math.abs(stickLeft.dx)>DZ?stickLeft.dx:0, ly=Math.abs(stickLeft.dy)>DZ?stickLeft.dy:0;
    if(lx||ly){camera.pos[0]+=forward[0]*(-ly)*speed+right[0]*lx*speed;camera.pos[1]+=forward[1]*(-ly)*speed;camera.pos[2]+=forward[2]*(-ly)*speed+right[2]*lx*speed;moved=true;}
    const rx=Math.abs(stickRight.dx)>DZ?stickRight.dx:0, ry=Math.abs(stickRight.dy)>DZ?stickRight.dy:0;
    if(rx||ry){const ls=2.5*dt;camera.yaw+=rx*ls;camera.pitch-=ry*ls;camera.pitch=Math.max(-Math.PI*0.49,Math.min(Math.PI*0.49,camera.pitch));moved=true;}
    if(moved) cameraMoved=true;
  }

  // --- Main loop ---
  function frame() {
    const now = performance.now();
    const dt = (now - lastTime) / 1000;
    lastTime = now;
    fpsAccum += dt; fpsCount++;
    if (fpsAccum >= 0.5) {
      fps = Math.round(fpsCount / fpsAccum); fpsAccum = 0; fpsCount = 0;
      if (frameIndex < 10 || frameIndex % 60 === 0) rlog(`FPS:${fps} frame:${frameIndex} cam:[${camera.pos.map(v=>v.toFixed(1))}]`);
    }

    updateCamera(dt);

    if (cameraMoved) { cameraMoved = false; }

    const {forward, right, up} = getCameraVectors();
    const fovFactor = Math.tan((camera.fov * Math.PI / 180) * 0.5);
    const aspect = width / height;

    frameIndex++;

    // Write PT uniforms
    const ud = new ArrayBuffer(uniformBufferSize);
    const f32 = new Float32Array(ud);
    const u32 = new Uint32Array(ud);
    f32[0] = width; f32[1] = height;
    u32[2] = 1; u32[3] = frameIndex; // always SPP=1
    f32[4] = camera.pos[0]; f32[5] = camera.pos[1]; f32[6] = camera.pos[2]; f32[7] = 0;
    f32[8] = forward[0]; f32[9] = forward[1]; f32[10] = forward[2]; f32[11] = 0;
    f32[12] = right[0]; f32[13] = right[1]; f32[14] = right[2]; f32[15] = 0;
    f32[16] = up[0]; f32[17] = up[1]; f32[18] = up[2]; f32[19] = fovFactor;
    const sun = getSunDir();
    f32[20] = sun[0]; f32[21] = sun[1]; f32[22] = sun[2];
    u32[23] = stats.emissiveTris;
    device.queue.writeBuffer(uniformBuffer, 0, ud);

    // Write temporal uniforms (current + previous camera)
    const td = new Float32Array(40); // 10 vec4f
    td[0] = width; td[1] = height; td[2] = settings.temporalAlpha; td[3] = 0;
    td[4]=right[0];td[5]=right[1];td[6]=right[2];td[7]=0;         // cam_right
    td[8]=up[0];td[9]=up[1];td[10]=up[2];td[11]=0;                // cam_up
    td[12]=forward[0];td[13]=forward[1];td[14]=forward[2];td[15]=0;// cam_fwd
    td[16]=camera.pos[0];td[17]=camera.pos[1];td[18]=camera.pos[2];td[19]=0; // cam_pos
    td[20]=prevCam.right[0];td[21]=prevCam.right[1];td[22]=prevCam.right[2];td[23]=0;
    td[24]=prevCam.up[0];td[25]=prevCam.up[1];td[26]=prevCam.up[2];td[27]=0;
    td[28]=prevCam.fwd[0];td[29]=prevCam.fwd[1];td[30]=prevCam.fwd[2];td[31]=0;
    td[32]=prevCam.pos[0];td[33]=prevCam.pos[1];td[34]=prevCam.pos[2];td[35]=0;
    td[36]=fovFactor;td[37]=aspect;td[38]=0;td[39]=0;
    device.queue.writeBuffer(tmpBuf, 0, td);

    // Save current camera as previous for next frame
    prevCam = { pos:[...camera.pos], right:[...right], up:[...up], fwd:[...forward] };

    try {
      const encoder = device.createCommandEncoder();

      // Pass 1: Path trace → HDR + normal/depth
      // Write SHaRC params
      const sp = new ArrayBuffer(32);
      const spU = new Uint32Array(sp); const spF = new Float32Array(sp);
      spU[0] = SHARC_CAPACITY; spU[1] = frameIndex; spF[2] = 0.5; spU[3] = 128;
      spF[4] = camera.pos[0]; spF[5] = camera.pos[1]; spF[6] = camera.pos[2]; spF[7] = 0;
      device.queue.writeBuffer(sharcParamBuf, 0, sp);

      // Pass 1: Path trace + SHaRC store/lookup
      const ptPass = encoder.beginComputePass();
      ptPass.setPipeline(computePipeline);
      ptPass.setBindGroup(0, bg0);
      ptPass.setBindGroup(1, bg1);
      ptPass.setBindGroup(2, bg2);
      ptPass.setBindGroup(3, bg3);
      ptPass.dispatchWorkgroups(Math.ceil(width/8), Math.ceil(height/8));
      ptPass.end();

      // SHaRC resolve (merge accumulation → resolved cache)
      const sharcPass = encoder.beginComputePass();
      sharcPass.setPipeline(sharcResolvePipeline);
      sharcPass.setBindGroup(0, bg2Resolve);
      sharcPass.dispatchWorkgroups(Math.ceil(SHARC_CAPACITY / 256));
      sharcPass.end();

      // Pass 2: Temporal reprojection (blend noisy + reprojected history → hdrTex)
      const tmpPass = encoder.beginComputePass();
      tmpPass.setPipeline(tmpPipeline);
      tmpPass.setBindGroup(0, historyFrame === 0 ? tmpBG_A : tmpBG_B);
      tmpPass.dispatchWorkgroups(Math.ceil(width/8), Math.ceil(height/8));
      tmpPass.end();
      historyFrame = 1 - historyFrame; // swap double buffer

      // Pass 3-5: À-trous spatial denoise (3 iterations, step 1→2→4)
      const dnBGs = [dnBG_hdr2ping, dnBG_ping2pong, dnBG_pong2ping];
      for (let di = 0; di < 3; di++) {
        const dp = encoder.beginComputePass();
        dp.setPipeline(dnAtrousPipeline);
        dp.setBindGroup(0, dnBGs[di]);
        dp.dispatchWorkgroups(Math.ceil(width/8), Math.ceil(height/8));
        dp.end();
      }

      // Pass 5: Composite (tonemap denoised HDR → LDR ptOutputTex)
      const compPass = encoder.beginComputePass();
      compPass.setPipeline(dnCompPipeline);
      compPass.setBindGroup(0, dnBG_comp);
      compPass.dispatchWorkgroups(Math.ceil(width/8), Math.ceil(height/8));
      compPass.end();

      if (fsrMode !== 'dlaa') {
        // Pass 2: FSR EASU — Lanczos upscale (internal → display)
        const easuPass = encoder.beginComputePass();
        easuPass.setPipeline(easuPipeline);
        easuPass.setBindGroup(0, easuBG);
        easuPass.dispatchWorkgroups(Math.ceil(displayWidth/8), Math.ceil(displayHeight/8));
        easuPass.end();

        // Pass 3: FSR RCAS — Contrast-adaptive sharpening
        const rcasPass = encoder.beginComputePass();
        rcasPass.setPipeline(rcasPipeline);
        rcasPass.setBindGroup(0, rcasBG);
        rcasPass.dispatchWorkgroups(Math.ceil(displayWidth/8), Math.ceil(displayHeight/8));
        rcasPass.end();
      }

      // Display: FSR final output, or ptOutput directly for DLAA
      const tv = context.getCurrentTexture().createView();
      const rp = encoder.beginRenderPass({
        colorAttachments: [{view:tv, loadOp:'clear', storeOp:'store', clearValue:{r:0,g:0,b:0,a:1}}],
      });
      rp.setPipeline(displayPipeline);
      rp.setBindGroup(0, fsrMode === 'dlaa' ? dispBGDirect : dispBG);
      rp.draw(3);
      rp.end();

      device.queue.submit([encoder.finish()]);
      if (frameIndex === 1) rlog('First frame OK');
    } catch(e) {
      rlog('FRAME_ERROR: ' + e.message);
    }

    drawGizmo();

    info.innerHTML =
      `<b>Ignis</b> | ${FSR_MODES[fsrMode].label}<br>` +
      `${width}x${height}\u2192${displayWidth}x${displayHeight} FPS:${fps}<br>` +
      `<span style="font-size:11px">ESC: options</span>`;

    requestAnimationFrame(frame);
  }

  info.textContent = 'Ready. Starting render...';
  requestAnimationFrame(frame);
}

init().catch(err => { rlog('FATAL:', err.message, err.stack); showError(`Error: ${err.message}`); });
