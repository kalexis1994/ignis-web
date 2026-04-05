// WebGPU Monte Carlo Path Tracer - Renderer (Sponza GLTF + BVH)

import { loadScene } from './scene-loader.js';
import { createOIDNPipeline } from './oidn-pipeline.js';
import { CyclesSkyModel, CYCLES_SKY_DEFAULTS } from './sky-model.js';

// Remote logging — sends to Python server, viewable in client.log
function rlog(...args) {
  const msg = args.map(a => typeof a === 'object' ? JSON.stringify(a) : String(a)).join(' ');
  console.log(msg);
  fetch('/log', { method: 'POST', body: msg }).catch(() => {});
}
window.onerror = (msg, src, line) => rlog(`ERROR: ${msg} at ${src}:${line}`);
window.onunhandledrejection = (e) => rlog(`REJECT: ${e.reason}`);

function resolveSceneAssetURL(uri) {
  if (!uri) return null;
  return /^(?:data:|blob:|https?:)/i.test(uri) ? uri : `scene/${uri}`;
}

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
  const MAX_PUNCTUAL_LIGHTS = 16;
  if (!navigator.gpu) {
    showError('WebGPU not supported. Try Chrome 113+, Edge 113+, or Firefox Nightly.');
    return;
  }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) { showError('Failed to get GPU adapter.'); return; }

  const requiredStorageBuffersPerStage = Math.min(adapter.limits.maxStorageBuffersPerShaderStage, 16);
  if (adapter.limits.maxStorageBuffersPerShaderStage < requiredStorageBuffersPerStage) {
    showError(
      `This GPU/browser exposes maxStorageBuffersPerShaderStage=${adapter.limits.maxStorageBuffersPerShaderStage}, `
      + `but the current glTF renderer needs at least ${requiredStorageBuffersPerStage}.`
    );
    return;
  }

  const hasSubgroups = adapter.features.has('subgroups');
  const hasF16 = adapter.features.has('shader-f16');
  const hasSubgroupMatrix = adapter.features.has('chromium-experimental-subgroup-matrix');
  console.log(`GPU Features: subgroups=${hasSubgroups}, f16=${hasF16}, subgroup-matrix=${hasSubgroupMatrix}`);
  if (hasSubgroupMatrix) console.log('🟢 TENSOR CORES AVAILABLE via subgroup-matrix!');
  const requiredFeatures = [];
  if (hasSubgroups) requiredFeatures.push('subgroups');
  if (hasF16) requiredFeatures.push('shader-f16');
  if (hasSubgroupMatrix) requiredFeatures.push('chromium-experimental-subgroup-matrix');

  const device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBuffersPerShaderStage: requiredStorageBuffersPerStage,
      maxComputeWorkgroupStorageSize: Math.min(adapter.limits.maxComputeWorkgroupStorageSize, 32768),
    }
  });
  device.onuncapturederror = (e) => rlog('GPU_ERROR: ' + e.error.message);
  // Adreno detection + GPU info logging
  // GPU detection + profile matching
  const adapterInfo = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
  const gpuStr = ((adapterInfo.vendor||'')+ ' ' +(adapterInfo.architecture||'')+' '+(adapterInfo.device||'')).toLowerCase();
  rlog(`GPU: ${gpuStr}`);
  rlog(`Format:${navigator.gpu.getPreferredCanvasFormat()} StorageBuf:${device.limits.maxStorageBuffersPerShaderStage} StorageTex:${device.limits.maxStorageTexturesPerShaderStage}`);

  // Load GPU profiles and match
  let gpuProfile = {};
  try {
    const profiles = await fetch('gpu-profiles.json').then(r => r.json());
    const defaults = profiles.defaults;
    for (const [name, prof] of Object.entries(profiles.profiles)) {
      for (const pattern of prof.match) {
        if (new RegExp(pattern, 'i').test(gpuStr)) {
          gpuProfile = { ...defaults, ...prof, _name: name };
          rlog(`GPU profile matched: ${name} (${prof.match[0]})`);
          break;
        }
      }
      if (gpuProfile._name) break;
    }
    if (!gpuProfile._name) {
      gpuProfile = { ...defaults, _name: 'default' };
      rlog('No GPU profile matched, using defaults');
    }
    if (gpuProfile.quirks?.length) rlog('Quirks: ' + gpuProfile.quirks.join(', '));
  } catch(e) {
    rlog('GPU profiles not loaded: ' + e.message);
    gpuProfile = { _name: 'fallback' };
  }
  const isAdreno = gpuStr.includes('qualcomm') || gpuStr.includes('adreno');
  device.onuncapturederror = (e) => rlog('GPU_ERROR: ' + e.error.message);
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
  // Validate scene has rasterization data (invalidate stale cache)
  if (!scene.rasterIndices || !scene.vertMatIds) {
    rlog('Stale cache detected — clearing and reloading...');
    try { indexedDB.deleteDatabase('ignis-scene-cache'); } catch(e) {}
    info.textContent = 'Cache outdated, reloading...';
    setTimeout(() => location.reload(), 500);
    return;
  }
  const stats = scene.stats;
  rlog('Scene loaded:', stats);
  const punctualLights = [...(scene.punctualLights || [])]
    .sort((a, b) => {
      const la = (0.2126 * a.color[0] + 0.7152 * a.color[1] + 0.0722 * a.color[2]) * a.intensity * Math.max(a.range || 1, 1);
      const lb = (0.2126 * b.color[0] + 0.7152 * b.color[1] + 0.0722 * b.color[2]) * b.intensity * Math.max(b.range || 1, 1);
      return lb - la;
    })
    .slice(0, MAX_PUNCTUAL_LIGHTS);
  if ((scene.punctualLights || []).length > MAX_PUNCTUAL_LIGHTS) {
    rlog(`Punctual lights: ${punctualLights.length}/${scene.punctualLights.length} (truncated)`);
  } else if (punctualLights.length > 0) {
    rlog(`Punctual lights: ${punctualLights.length}`);
    rlog('Synthetic sun disabled because scene punctual lights are present');
  }
  if (stats.emissiveSourceTris !== undefined) {
    const trunc = stats.emissiveTruncated ? ' (truncated)' : '';
    rlog(`Emissive sampling: ${stats.emissiveTris}/${stats.emissiveSourceTris} tris${trunc}`);
  }

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
  // Read config from splash screen (or defaults)
  const cfg = window.IGNIS_CONFIG || {};
  const isMobile = isAdreno || /Android|iPhone|iPad/i.test(navigator.userAgent);
  // User config overrides GPU profile, which overrides defaults
  let fsrMode = cfg.fsrMode || gpuProfile.fsrMode || 'balanced';
  if (!FSR_MODES[fsrMode]) fsrMode = 'balanced';
  const displayCap = cfg.displayCap || gpuProfile.displayCap || 1080;
  const texSize = cfg.texSize ?? gpuProfile.texSize ?? 512;
  const denoiseMode = 'reblur'; // cfg.denoise || gpuProfile.denoise || 'full';
  const maxBounces = cfg.bounces || gpuProfile.maxBounces || 3;
  const sppPerFrame = cfg.spp || gpuProfile.spp || 1;
  const sharcEnabled = cfg.sharc !== undefined ? cfg.sharc : (gpuProfile.sharc !== false);
  rlog(`Config: fsr=${fsrMode} display=${displayCap}p tex=${texSize} denoise=${denoiseMode} bounces=${maxBounces} spp=${sppPerFrame} sharc=${sharcEnabled}`);

  // Display resolution — capped by user preference
  const dispScale = Math.min(1.0, displayCap / Math.max(window.innerWidth, window.innerHeight));
  const displayWidth = Math.ceil(Math.floor(window.innerWidth * dispScale) / 8) * 8;
  const displayHeight = Math.ceil(Math.floor(window.innerHeight * dispScale) / 8) * 8;
  canvas.width = displayWidth;
  canvas.height = displayHeight;
  context.configure({ device, format, alphaMode: 'opaque' });

  // Letterboxing: fit canvas in window maintaining aspect ratio, black bars
  const canvasAspect = displayWidth / displayHeight;
  function fitCanvas() {
    const winW = window.innerWidth, winH = window.innerHeight;
    const winAspect = winW / winH;
    if (winAspect > canvasAspect) {
      // Window wider than canvas → fit height, black bars on sides
      canvas.style.height = winH + 'px';
      canvas.style.width = Math.round(winH * canvasAspect) + 'px';
    } else {
      // Window taller than canvas → fit width, black bars top/bottom
      canvas.style.width = winW + 'px';
      canvas.style.height = Math.round(winW / canvasAspect) + 'px';
    }
  }
  fitCanvas();
  window.addEventListener('resize', fitCanvas);

  // Internal resolution from FSR mode
  let fsrScale = FSR_MODES[fsrMode].scale;
  let width = Math.ceil(displayWidth * fsrScale / 8) * 8;
  let height = Math.ceil(displayHeight * fsrScale / 8) * 8;
  let fsrRatio = displayWidth / width;

  rlog(`FSR ${fsrMode}: internal ${width}x${height} → display ${displayWidth}x${displayHeight} (${fsrRatio.toFixed(1)}x)`);

  // --- Load shaders ---
  const v = Date.now(); // cache bust
  let [ptCode, dispCode, fsrCode, dnCode, tmpCode, gbCode, smCode, reblurCode] = await Promise.all([
    fetch(`pathtracer.wgsl?v=${v}`).then(r => r.text()),
    fetch(`display.wgsl?v=${v}`).then(r => r.text()),
    fetch(`fsr.wgsl?v=${v}`).then(r => r.text()),
    fetch(`denoise.wgsl?v=${v}`).then(r => r.text()),
    fetch(`temporal.wgsl?v=${v}`).then(r => r.text()),
    fetch(`gbuffer.wgsl?v=${v}`).then(r => r.text()),
    fetch(`shadow-map.wgsl?v=${v}`).then(r => r.text()),
    fetch(`reblur.wgsl?v=${v}`).then(r => r.text()),
  ]);

  if (hasSubgroups) rlog('Subgroups available (reserved for denoiser)');

  // shader-f16: half-precision weights in denoiser/FSR (2x ALU throughput on NVIDIA Turing+)
  if (hasF16) {
    dnCode = 'enable f16;\n' + dnCode;
    fsrCode = 'enable f16;\n' + fsrCode;
    rlog('shader-f16 enabled for denoiser and FSR');
  } else {
    // Fallback: strip f16 types back to f32 so shaders compile without the extension
    function stripF16(code) {
      return code.replace(/\bvec(\d)h\b/g, 'vec$1f')
                 .replace(/\bf16\(/g, 'f32(')
                 .replace(/(\d+\.?\d*)h\b/g, '$1')
                 .replace(/:\s*f16\b/g, ': f32');
    }
    dnCode = stripF16(dnCode);
    fsrCode = stripF16(fsrCode);
  }

  const smOpts = { strictMath: false };
  const ptModule = device.createShaderModule({ code: ptCode, ...smOpts });
  const dispModule = device.createShaderModule({ code: dispCode, ...smOpts });
  const fsrModule = device.createShaderModule({ code: fsrCode, ...smOpts });
  const dnModule = device.createShaderModule({ code: dnCode, ...smOpts });
  const tmpModule = device.createShaderModule({ code: tmpCode, ...smOpts });
  const gbModule = device.createShaderModule({ code: gbCode, ...smOpts });
  const shadowModule = device.createShaderModule({ code: smCode, ...smOpts });
  const reblurModule = device.createShaderModule({ code: reblurCode, ...smOpts });
  rlog('ReBLUR shader compiled: ' + (reblurModule ? 'OK' : 'FAIL'));

  // ReBLUR: history bind group layout (group 1)
  const reblurHistoryLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },          // history_diff
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },          // history_spec
      { binding: 2, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },             // linear_clamp
      { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },          // motion vectors (depth_tex)
    ],
  });

  // Check shader compilation
  for (const [name, mod] of [['pathtracer',ptModule],['display',dispModule],['fsr',fsrModule],['denoise',dnModule],['temporal',tmpModule],['gbuffer',gbModule],['shadow-map',shadowModule]]) {
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
  // Base camera/sun state + fixed-size punctual light array.
  const uniformBufferSize = 1280;
  const uniformBuffer = device.createBuffer({
    size: uniformBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // --- Internal textures ---
  const F16 = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING;
  const F16C = F16 | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST; // copyable
  const F16R = F16 | GPUTextureUsage.RENDER_ATTACHMENT; // rasterization target
  const noisyTex = device.createTexture({ size:[width,height], format:'rgba16float', usage:F16 }); // irradiance
  const albedoTex = device.createTexture({ size:[width,height], format:'rgba8unorm',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING }); // first-hit albedo
  const denoiseNdTex = device.createTexture({ size:[width,height], format:'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING });
  // accumulation handled by temporal pass (no extra buffer — 8 storage buf limit on Adreno)
  const ndTex    = device.createTexture({ size:[width,height], format:'rgba16float', usage:F16R });  // raster writes normal+matId
  const matIdTex = device.createTexture({ size:[width,height], format:'rgba16float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING }); // matId + fract(UV)
  const zBuf = device.createTexture({ size:[width,height], format:'depth32float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT }); // Z-test only

  // --- Rasterization pipeline (G-buffer) ---
  const gbufUniformBuf = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); // mat4x4 + vec3 + pad = 80
  const posVB = createGPUBuffer(device, scene.gpuPositions, GPUBufferUsage.VERTEX);
  const nrmVB = createGPUBuffer(device, scene.gpuNormals, GPUBufferUsage.VERTEX);
  const uvExtraVB = createGPUBuffer(device, scene.gpuUVExtra, GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE);
  const matIdVB = createGPUBuffer(device, scene.vertMatIds, GPUBufferUsage.VERTEX);
  const indexBuf = createGPUBuffer(device, scene.rasterIndices, GPUBufferUsage.INDEX);
  const triCount = stats.triangles;

  const gbufBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
    { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
    { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
  ]});
  // gbufBG created after texture array is loaded (below)

  const gbufPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [gbufBGL] }),
    vertex: {
      module: gbModule, entryPoint: 'vs',
      buffers: [
        { arrayStride: 16, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }] }, // pos.xyz + uv.x
        { arrayStride: 16, attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x4' }] }, // normal.xyz + uv.y
        { arrayStride: 4,  attributes: [{ shaderLocation: 2, offset: 0, format: 'float32' }] },    // matId
        { arrayStride: 24, attributes: [
          { shaderLocation: 3, offset: 0,  format: 'float32x2' },  // UV1.xy
          { shaderLocation: 4, offset: 8,  format: 'float32x2' },  // UV2.xy
          { shaderLocation: 5, offset: 16, format: 'float32x2' },  // UV3.xy
        ]},
      ],
    },
    fragment: {
      module: gbModule, entryPoint: 'fs',
      targets: [
        { format: 'rgba16float' },  // normal.xyz + depth
        { format: 'rgba16float' },  // matId + fract(UV)
      ],
    },
    depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
    primitive: { topology: 'triangle-list', cullMode: 'none' }, // double-sided
  });
  rlog(`Raster pipeline: ${triCount} tris, ${stats.vertices} verts`);

  // --- Shadow map ---
  const SHADOW_RES = 2048;
  const shadowDepthTex = device.createTexture({
    size: [SHADOW_RES, SHADOW_RES],
    format: 'depth32float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  const shadowSampler = device.createSampler({ compare: 'less-equal', magFilter: 'linear', minFilter: 'linear' });
  const shadowUniformBuf = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); // mat4x4f
  // Shadow map uses same BGL structure as gbuffer (uniform + materials + texArray + sampler)
  const shadowBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
    { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
    { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
  ]});
  const shadowPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [shadowBGL] }),
    vertex: {
      module: shadowModule, entryPoint: 'vs',
      buffers: [
        { arrayStride: 16, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }] },
        { arrayStride: 16, attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x4' }] },
        { arrayStride: 4,  attributes: [{ shaderLocation: 2, offset: 0, format: 'float32' }] },
        { arrayStride: 24, attributes: [
          { shaderLocation: 3, offset: 0,  format: 'float32x2' },
          { shaderLocation: 4, offset: 8,  format: 'float32x2' },
          { shaderLocation: 5, offset: 16, format: 'float32x2' },
        ]},
      ],
    },
    fragment: { module: shadowModule, entryPoint: 'fs', targets: [] }, // depth-only
    depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
  });
  // shadowBG created after texture array is loaded (below)
  // Diffuse signal textures
  const hdrTex   = device.createTexture({ size:[width,height], format:'rgba16float', usage:F16 });   // temporal accumulated diffuse
  const pingTex  = device.createTexture({ size:[width,height], format:'rgba16float', usage:F16 });   // à-trous ping diffuse
  const pongTex  = device.createTexture({ size:[width,height], format:'rgba16float', usage:F16 });   // à-trous pong diffuse
  const historyA = device.createTexture({ size:[width,height], format:'rgba16float', usage:F16C });   // prev denoised diffuse (read)
  const historyB = device.createTexture({ size:[width,height], format:'rgba16float', usage:F16C });   // prev denoised diffuse (write)
  // Specular signal textures
  const specNoisyTex = device.createTexture({ size:[width,height], format:'rgba16float', usage:F16 }); // PT specular output
  const specHdrTex   = device.createTexture({ size:[width,height], format:'rgba16float', usage:F16 }); // temporal accumulated specular
  const specPingTex  = device.createTexture({ size:[width,height], format:'rgba16float', usage:F16 }); // à-trous ping specular
  const specPongTex  = device.createTexture({ size:[width,height], format:'rgba16float', usage:F16 }); // à-trous pong specular
  const specHistoryA = device.createTexture({ size:[width,height], format:'rgba16float', usage:F16C }); // prev denoised specular
  const specHistoryB = device.createTexture({ size:[width,height], format:'rgba16float', usage:F16C });
  const ptOutputTex = device.createTexture({
    size:[width,height], format:'rgba8unorm',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
  });
  // Legacy composite input kept bound for layout stability.
  const prevFrameTex = device.createTexture({
    size:[width,height], format:'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
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
  const texInfo = scene.textureInfo;
  const texCount = texInfo ? texInfo.count : 0;
  let texArray;

  if (texCount > 0) {
    const TEX_SIZE = texSize || 512;

    texArray = device.createTexture({
      size: [TEX_SIZE, TEX_SIZE, texCount],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const BATCH = TEX_SIZE >= 2048 ? 2 : 6; // smaller batches for large textures
    for (let b = 0; b < texCount; b += BATCH) {
      const batch = [];
      for (let i = b; i < Math.min(b + BATCH, texCount); i++) {
        const texURL = resolveSceneAssetURL(texInfo.imageURIs[i]);
        if (!texURL) throw new Error(`Texture ${i} is missing a URI`);
        batch.push(
          fetch(texURL)
            .then(r => r.blob())
            .then(blob => createImageBitmap(blob, {
              resizeWidth: TEX_SIZE, resizeHeight: TEX_SIZE,
              resizeQuality: 'high',
              colorSpaceConversion: 'none',
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

  // Create G-buffer bind group (now that texArray exists)
  const matBufForGbuf = createGPUBuffer(device, scene.gpuMaterials, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const gbufSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  const gbufBG = device.createBindGroup({ layout: gbufBGL, entries: [
    { binding: 0, resource: { buffer: gbufUniformBuf } },
    { binding: 1, resource: { buffer: matBufForGbuf } },
    { binding: 2, resource: texArray.createView({ dimension: '2d-array' }) },
    { binding: 3, resource: gbufSampler },
  ]});
  const shadowBG = device.createBindGroup({ layout: shadowBGL, entries: [
    { binding: 0, resource: { buffer: shadowUniformBuf } },
    { binding: 1, resource: { buffer: matBufForGbuf } },
    { binding: 2, resource: texArray.createView({ dimension: '2d-array' }) },
    { binding: 3, resource: gbufSampler },
  ]});

  // --- Scene GPU buffers ---
  info.textContent = 'Uploading to GPU...';
  const vtxBuf = createGPUBuffer(device, scene.gpuPositions, GPUBufferUsage.STORAGE);
  const nrmBuf = createGPUBuffer(device, scene.gpuNormals, GPUBufferUsage.STORAGE);
  const triBuf = createGPUBuffer(device, scene.gpuTriData, GPUBufferUsage.STORAGE);
  const bvhBuf = createGPUBuffer(device, scene.gpuBVHNodes, GPUBufferUsage.STORAGE);
  const matBuf = createGPUBuffer(device, scene.gpuMaterials, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const emsBuf = createGPUBuffer(device, scene.gpuEmissiveTris, GPUBufferUsage.STORAGE);
  const uvExtraBuf = uvExtraVB;

  // --- Bind group 0: uniforms + accumulation + output ---
  const bg0Layout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba8unorm' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'depth' } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'comparison' } },
    ],
  });
  const bg0 = device.createBindGroup({
    layout: bg0Layout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: noisyTex.createView() },
      { binding: 2, resource: ndTex.createView() },
      { binding: 3, resource: matIdTex.createView() },
      { binding: 4, resource: albedoTex.createView() },
      { binding: 5, resource: denoiseNdTex.createView() },
      { binding: 6, resource: specNoisyTex.createView() },
      { binding: 7, resource: shadowDepthTex.createView() },
      { binding: 8, resource: shadowSampler },
    ],
  });

  // --- Environment CDF buffer (created early so bg1 can reference it) ---
  const ENV_W = 512, ENV_H = 256;
  const envCdfSize = ENV_W * ENV_H * 5 + ENV_H + 10;
  const envCdfBuf = device.createBuffer({ size: envCdfSize * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

  // --- Bind group 1: scene data ---
  const bg1Layout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
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
      { binding: 6, resource: { buffer: uvExtraBuf } },
      { binding: 7, resource: { buffer: envCdfBuf } },
    ],
  });

  // --- ReSTIR GI buffers (created before SHaRC so bg2 can reference them) ---
  const restirEnabled = device.limits.maxStorageBuffersPerShaderStage >= requiredStorageBuffersPerStage;
  const restirPixels = width * height;
  const restirBufSize = restirEnabled ? restirPixels * 3 * 16 : 48;
  const restirBufA = device.createBuffer({ size: restirBufSize, usage: GPUBufferUsage.STORAGE });
  const restirBufB = device.createBuffer({ size: restirBufSize, usage: GPUBufferUsage.STORAGE });
  let restirFrame = 0;

  // --- SHaRC radiance cache + ReSTIR bind group ---
  const SHARC_CAPACITY = 131072;
  const sharcParamBuf = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  // Extended: +3 u32 per slot for direction accumulation (L1 SH path guiding)
  const sharcKeysAccumBuf = device.createBuffer({ size: SHARC_CAPACITY * 8 * 4, usage: GPUBufferUsage.STORAGE });
  const sharcResolvedBuf = device.createBuffer({ size: SHARC_CAPACITY * 7 * 4, usage: GPUBufferUsage.STORAGE });

  // PT reads resolved as read-only, writes keys_accum as read_write + ReSTIR buffers
  const bg2Layout = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // keys_accum rw
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // resolved ro
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // restir curr rw
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // restir prev ro
  ]});
  // Double-buffered: bg2_A writes restirA reads restirB, bg2_B swaps
  const bg2_A = device.createBindGroup({ layout: bg2Layout, entries: [
    { binding: 0, resource: { buffer: sharcParamBuf } },
    { binding: 1, resource: { buffer: sharcKeysAccumBuf } },
    { binding: 2, resource: { buffer: sharcResolvedBuf } },
    { binding: 3, resource: { buffer: restirBufA } },
    { binding: 4, resource: { buffer: restirBufB } },
  ]});
  const bg2_B = device.createBindGroup({ layout: bg2Layout, entries: [
    { binding: 0, resource: { buffer: sharcParamBuf } },
    { binding: 1, resource: { buffer: sharcKeysAccumBuf } },
    { binding: 2, resource: { buffer: sharcResolvedBuf } },
    { binding: 3, resource: { buffer: restirBufB } },
    { binding: 4, resource: { buffer: restirBufA } },
  ]});

  // Resolve needs rw on both
  const bg2ResolveLayout = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
  ]});
  const sharcModule = device.createShaderModule({ code: await fetch(`sharc.wgsl?v=${v}`).then(r=>r.text()), ...smOpts });
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

  // Scene bounds from BVH root node (AABB)
  const bvhF = scene.gpuBVHNodes;
  const sceneMin = [bvhF[0], bvhF[1], bvhF[2]];
  const sceneMax = [bvhF[4], bvhF[5], bvhF[6]];
  const sceneCenter = [(sceneMin[0]+sceneMax[0])*0.5, (sceneMin[1]+sceneMax[1])*0.5, (sceneMin[2]+sceneMax[2])*0.5];
  const sceneRadius = Math.sqrt((sceneMax[0]-sceneMin[0])**2 + (sceneMax[1]-sceneMin[1])**2 + (sceneMax[2]-sceneMin[2])**2) * 0.5;
  rlog(`Scene bounds: center=[${sceneCenter.map(v=>v.toFixed(1))}] radius=${sceneRadius.toFixed(1)}`);

  rlog(`ReSTIR GI ${restirEnabled ? 'enabled' : 'disabled (limit<10)'} (${(restirBufSize * 2 / 1048576).toFixed(1)} MB)`);

  // --- Compute pipeline (PT + SHaRC/ReSTIR + Textures) ---
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
      { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } }, // G-buffer ND (high-res guide)
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

  // EASU/JBU: reads ptOutput (internal) + G-buffer guide, writes upscaled (display)
  const easuBG = device.createBindGroup({
    layout: fsrBGLayout,
    entries: [
      { binding: 0, resource: ptOutputTex.createView() },
      { binding: 1, resource: upscaledTex.createView() },
      { binding: 2, resource: sampler },
      { binding: 3, resource: { buffer: fsrParamsBuf } },
      { binding: 4, resource: ndTex.createView() },  // high-res G-buffer normals+depth
    ],
  });

  // RCAS: reads upscaled, writes final (ndTex unused but layout requires it)
  const rcasBG = device.createBindGroup({
    layout: fsrBGLayout,
    entries: [
      { binding: 0, resource: upscaledTex.createView() },
      { binding: 1, resource: finalTex.createView() },
      { binding: 2, resource: sampler },
      { binding: 3, resource: { buffer: fsrParamsBuf } },
      { binding: 4, resource: ndTex.createView() },
    ],
  });

  // --- Temporal reprojection pipeline ---
  const tmpBufSize = 4*4 * 10; // 10 vec4f = 160 bytes
  const tmpBuf = device.createBuffer({ size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const tmpLayout = device.createBindGroupLayout({ entries: [
    { binding:0, visibility:GPUShaderStage.COMPUTE, buffer:{type:'uniform'} },
    { binding:1, visibility:GPUShaderStage.COMPUTE, texture:{sampleType:'float'} },          // current noisy diffuse
    { binding:2, visibility:GPUShaderStage.COMPUTE, texture:{sampleType:'float'} },          // prev denoised diffuse
    { binding:3, visibility:GPUShaderStage.COMPUTE, texture:{sampleType:'float'} },          // depth (ndTex)
    { binding:4, visibility:GPUShaderStage.COMPUTE, storageTexture:{access:'write-only',format:'rgba16float'} }, // diff accum out
    { binding:5, visibility:GPUShaderStage.COMPUTE, storageTexture:{access:'write-only',format:'rgba16float'} }, // diff history out
    { binding:6, visibility:GPUShaderStage.COMPUTE, sampler:{type:'filtering'} },
    { binding:7, visibility:GPUShaderStage.COMPUTE, texture:{sampleType:'float'} },          // current noisy specular
    { binding:8, visibility:GPUShaderStage.COMPUTE, texture:{sampleType:'float'} },          // prev denoised specular
    { binding:9, visibility:GPUShaderStage.COMPUTE, storageTexture:{access:'write-only',format:'rgba16float'} }, // spec accum out
    { binding:10, visibility:GPUShaderStage.COMPUTE, storageTexture:{access:'write-only',format:'rgba16float'} }, // spec history out
  ]});
  const tmpPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts:[tmpLayout] }),
    compute: { module:tmpModule, entryPoint:'temporal' },
  });
  // Two bind groups: frame A reads historyA, writes historyB; frame B reads historyB, writes historyA
  // Temporal reads pre-blurred data from pingTex/specPingTex (written by preblur pass)
  const tmpBG_A = device.createBindGroup({ layout:tmpLayout, entries:[
    { binding:0, resource:{buffer:tmpBuf} },
    { binding:1, resource:pingTex.createView() },           // pre-blurred diffuse
    { binding:2, resource:historyA.createView() },
    { binding:3, resource:denoiseNdTex.createView() },
    { binding:4, resource:hdrTex.createView() },
    { binding:5, resource:historyB.createView() },
    { binding:6, resource:sampler },
    { binding:7, resource:specPingTex.createView() },       // pre-blurred specular
    { binding:8, resource:specHistoryA.createView() },
    { binding:9, resource:specHdrTex.createView() },
    { binding:10, resource:specHistoryB.createView() },
  ]});
  const tmpBG_B = device.createBindGroup({ layout:tmpLayout, entries:[
    { binding:0, resource:{buffer:tmpBuf} },
    { binding:1, resource:pingTex.createView() },           // pre-blurred diffuse
    { binding:2, resource:historyB.createView() },
    { binding:3, resource:denoiseNdTex.createView() },
    { binding:4, resource:hdrTex.createView() },
    { binding:5, resource:historyA.createView() },
    { binding:6, resource:sampler },
    { binding:7, resource:specPingTex.createView() },       // pre-blurred specular
    { binding:8, resource:specHistoryB.createView() },
    { binding:9, resource:specHdrTex.createView() },
    { binding:10, resource:specHistoryA.createView() },
  ]});

  // Previous camera state for temporal reprojection
  let prevCam = { pos:[0,5,0], right:[1,0,0], up:[0,1,0], fwd:[0,0,-1] };
  let historyFrame = 0; // alternates 0/1 for double-buffering

  // --- Denoiser pipelines ---
  // 3 separate param buffers for the 3 à-trous steps (avoid writeBuffer during encoding)
  const denoisePasses = gpuProfile.denoisePasses || 5;
  const dnSteps = [1, 2, 4, 8, 16]; // up to 5 passes
  const dnParamBufs = dnSteps.map((s, i) => {
    const buf = device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buf, 0, new Float32Array([width, height, s, 0, 0, 1.0, 1.0, 0]));
    return buf;
  });
  const dnCompParamBuf = device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); // 8+16 camera floats

  // À-trous layout: dual-signal (diffuse + specular) + normals + albedo(roughness)
  const dnAtrousLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },          // diffuse in
      { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } }, // diffuse out
      { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },          // gbuf normal+depth
      { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },          // specular in
      { binding: 5, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } }, // specular out
      { binding: 6, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },          // albedo + roughness
    ],
  });
  const dnAtrousPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [dnAtrousLayout] }),
    compute: { module: dnModule, entryPoint: 'atrous' },
  });
  // Copy denoised RGB to history, preserving temporal alpha (history_len/cam_z)
  const copyToHistoryPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [dnAtrousLayout] }),
    compute: { module: dnModule, entryPoint: 'copy_to_history' },
  });

  // --- ReBLUR pipelines (6 passes) ---
  const reblurSpatialPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [dnAtrousLayout] });
  const reblurTemporalPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [dnAtrousLayout, reblurHistoryLayout] });

  const reblurPrepassPipeline = device.createComputePipeline({ layout: reblurSpatialPipelineLayout, compute: { module: reblurModule, entryPoint: 'prepass' } });
  const reblurTemporalPipeline = device.createComputePipeline({ layout: reblurTemporalPipelineLayout, compute: { module: reblurModule, entryPoint: 'temporal_accumulation' } });
  const reblurHistoryFixPipeline = device.createComputePipeline({ layout: reblurSpatialPipelineLayout, compute: { module: reblurModule, entryPoint: 'history_fix' } });
  const reblurBlurPipeline = device.createComputePipeline({ layout: reblurSpatialPipelineLayout, compute: { module: reblurModule, entryPoint: 'blur' } });
  const reblurPostBlurPipeline = device.createComputePipeline({ layout: reblurSpatialPipelineLayout, compute: { module: reblurModule, entryPoint: 'post_blur' } });
  const reblurStabilizePipeline = device.createComputePipeline({ layout: reblurTemporalPipelineLayout, compute: { module: reblurModule, entryPoint: 'temporal_stabilization' } });
  // ReBLUR uniform buffer (ReblurParams struct, ~640 bytes)
  const reblurParamBuf = device.createBuffer({ size: 640, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  // ReBLUR bind groups — spatial passes use group(0) only, temporal passes use group(0)+group(1)
  // Reuse dnParamBufs[0] as uniform for spatial passes (step_size + frames_still)
  const reblurBG_prepass = device.createBindGroup({ layout: dnAtrousLayout, entries: [
    { binding: 0, resource: { buffer: reblurParamBuf } },
    { binding: 1, resource: noisyTex.createView() },        // in: noisy diffuse
    { binding: 2, resource: pingTex.createView() },          // out: pre-filtered
    { binding: 3, resource: denoiseNdTex.createView() },     // gbuf (normal+depth)
    { binding: 4, resource: specNoisyTex.createView() },     // in: noisy specular
    { binding: 5, resource: specPingTex.createView() },      // out: pre-filtered spec
    { binding: 6, resource: albedoTex.createView() },        // albedo+roughness
  ]});
  const reblurBG_temporal = device.createBindGroup({ layout: dnAtrousLayout, entries: [
    { binding: 0, resource: { buffer: reblurParamBuf } },
    { binding: 1, resource: pingTex.createView() },          // in: prepass output
    { binding: 2, resource: hdrTex.createView() },           // out: temporal result
    { binding: 3, resource: denoiseNdTex.createView() },
    { binding: 4, resource: specPingTex.createView() },
    { binding: 5, resource: specHdrTex.createView() },
    { binding: 6, resource: albedoTex.createView() },
  ]});
  const reblurHistBG_A = device.createBindGroup({ layout: reblurHistoryLayout, entries: [
    { binding: 0, resource: historyA.createView() },
    { binding: 1, resource: specHistoryA.createView() },
    { binding: 2, resource: sampler },
    { binding: 3, resource: denoiseNdTex.createView() },     // reuse as MV proxy
  ]});
  const reblurHistBG_B = device.createBindGroup({ layout: reblurHistoryLayout, entries: [
    { binding: 0, resource: historyB.createView() },
    { binding: 1, resource: specHistoryB.createView() },
    { binding: 2, resource: sampler },
    { binding: 3, resource: denoiseNdTex.createView() },
  ]});
  const reblurBG_historyfix = device.createBindGroup({ layout: dnAtrousLayout, entries: [
    { binding: 0, resource: { buffer: reblurParamBuf } },
    { binding: 1, resource: hdrTex.createView() },           // in: temporal output
    { binding: 2, resource: pingTex.createView() },          // out: fixed
    { binding: 3, resource: denoiseNdTex.createView() },
    { binding: 4, resource: specHdrTex.createView() },
    { binding: 5, resource: specPingTex.createView() },
    { binding: 6, resource: albedoTex.createView() },
  ]});
  const reblurBG_blur = device.createBindGroup({ layout: dnAtrousLayout, entries: [
    { binding: 0, resource: { buffer: reblurParamBuf } },
    { binding: 1, resource: pingTex.createView() },          // in: history-fixed
    { binding: 2, resource: pongTex.createView() },          // out: blurred
    { binding: 3, resource: denoiseNdTex.createView() },
    { binding: 4, resource: specPingTex.createView() },
    { binding: 5, resource: specPongTex.createView() },
    { binding: 6, resource: albedoTex.createView() },
  ]});
  const reblurBG_postblur = device.createBindGroup({ layout: dnAtrousLayout, entries: [
    { binding: 0, resource: { buffer: reblurParamBuf } },
    { binding: 1, resource: pongTex.createView() },          // in: blurred
    { binding: 2, resource: pingTex.createView() },          // out: post-blurred
    { binding: 3, resource: denoiseNdTex.createView() },
    { binding: 4, resource: specPongTex.createView() },
    { binding: 5, resource: specPingTex.createView() },
    { binding: 6, resource: albedoTex.createView() },
  ]});
  const reblurBG_stabilize = device.createBindGroup({ layout: dnAtrousLayout, entries: [
    { binding: 0, resource: { buffer: reblurParamBuf } },
    { binding: 1, resource: pingTex.createView() },          // in: post-blurred
    { binding: 2, resource: hdrTex.createView() },           // out: stabilized (final)
    { binding: 3, resource: denoiseNdTex.createView() },
    { binding: 4, resource: specPingTex.createView() },
    { binding: 5, resource: specHdrTex.createView() },
    { binding: 6, resource: albedoTex.createView() },
  ]});
  // Copy stabilized to history for next frame
  const reblurBG_copyHist_A = device.createBindGroup({ layout: dnAtrousLayout, entries: [
    { binding: 0, resource: { buffer: reblurParamBuf } },
    { binding: 1, resource: hdrTex.createView() },           // in: stabilized
    { binding: 2, resource: historyB.createView() },         // out: history
    { binding: 3, resource: denoiseNdTex.createView() },
    { binding: 4, resource: specHdrTex.createView() },
    { binding: 5, resource: specHistoryB.createView() },
    { binding: 6, resource: albedoTex.createView() },
  ]});
  const reblurBG_copyHist_B = device.createBindGroup({ layout: dnAtrousLayout, entries: [
    { binding: 0, resource: { buffer: reblurParamBuf } },
    { binding: 1, resource: hdrTex.createView() },
    { binding: 2, resource: historyA.createView() },
    { binding: 3, resource: denoiseNdTex.createView() },
    { binding: 4, resource: specHdrTex.createView() },
    { binding: 5, resource: specHistoryA.createView() },
    { binding: 6, resource: albedoTex.createView() },
  ]});
  rlog('ReBLUR pipelines + bind groups created');
  // Shared-memory tiled atrous for step=1 (first pass): 99 textureLoad → ~5 per thread
  const dnAtrousSMPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [dnAtrousLayout] }),
    compute: { module: dnModule, entryPoint: 'atrous_sm' },
  });
  // Pre-blur pipeline: same layout as à-trous, lightweight 3×3 bilateral
  const preblurPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [dnAtrousLayout] }),
    compute: { module: dnModule, entryPoint: 'preblur' },
  });
  // Shared-memory tiled pre-blur (18x18 tile, ±1 halo): 43 textureLoad → ~4 per thread
  const preblurSMPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [dnAtrousLayout] }),
    compute: { module: dnModule, entryPoint: 'preblur_sm' },
  });
  // Pre-blur bind group: reads noisyTex+specNoisyTex → writes pingTex+specPingTex
  const preblurBG = device.createBindGroup({ layout: dnAtrousLayout, entries: [
    { binding: 0, resource: { buffer: dnParamBufs[0] } }, // step_size=1 (unused by preblur, but layout needs it)
    { binding: 1, resource: noisyTex.createView() },
    { binding: 2, resource: pingTex.createView() },
    { binding: 3, resource: denoiseNdTex.createView() },
    { binding: 4, resource: specNoisyTex.createView() },
    { binding: 5, resource: specPingTex.createView() },
    { binding: 6, resource: albedoTex.createView() },
  ]});

  // Composite layout: denoised signals + legacy extra inputs kept for layout stability
  const dnCompLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },          // denoised diffuse
      { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },          // denoised specular
      { binding: 6, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },          // albedo
      { binding: 7, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba8unorm' } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },          // matId (glass detect)
      { binding: 9, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },          // G-buffer normals
      { binding: 10, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },         // prev frame (reflection)
      { binding: 11, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
      { binding: 12, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // materials
      { binding: 13, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },             // exposure accumulator
    ],
  });

  const exposureBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(exposureBuf, 0, new Float32Array([0, 0, 1.0, 0]));
  const dnCompPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [dnCompLayout] }),
    compute: { module: dnModule, entryPoint: 'composite' },
  });

  // À-trous ping-pong bind groups (dual-signal):
  // Pass 0: diff hdr→ping, spec specHdr→specPing
  // Pass 1: diff ping→pong, spec specPing→specPong
  // Pass 2: diff pong→ping, spec specPong→specPing, etc.
  const dnBGs = [];
  for (let i = 0; i < 5; i++) {
    const isFirst = (i === 0);
    const readFromPing = (i % 2 === 1);
    const dIn = isFirst ? hdrTex : (readFromPing ? pingTex : pongTex);
    const dOut = isFirst ? pingTex : (readFromPing ? pongTex : pingTex);
    const sIn = isFirst ? specHdrTex : (readFromPing ? specPingTex : specPongTex);
    const sOut = isFirst ? specPingTex : (readFromPing ? specPongTex : specPingTex);
    dnBGs.push(device.createBindGroup({ layout: dnAtrousLayout, entries: [
      { binding: 0, resource: { buffer: dnParamBufs[i] } },
      { binding: 1, resource: dIn.createView() },
      { binding: 2, resource: dOut.createView() },
      { binding: 3, resource: denoiseNdTex.createView() },
      { binding: 4, resource: sIn.createView() },
      { binding: 5, resource: sOut.createView() },
      { binding: 6, resource: albedoTex.createView() },
    ]}));
  }
  // For spatial-only: first pass reads noisy directly
  const dnBG_noisy_first = device.createBindGroup({ layout: dnAtrousLayout, entries: [
    { binding: 0, resource: { buffer: dnParamBufs[0] } },
    { binding: 1, resource: noisyTex.createView() },
    { binding: 2, resource: pingTex.createView() },
    { binding: 3, resource: denoiseNdTex.createView() },
    { binding: 4, resource: specNoisyTex.createView() },
    { binding: 5, resource: specPingTex.createView() },
    { binding: 6, resource: albedoTex.createView() },
  ]});
  const dnFinalInPing = (denoisePasses % 2 === 1); // 3 passes → ping, 4 → pong, 5 → ping

  // Copy-to-history bind groups: merge denoised .rgb + temporal .a → history
  // BG_A: reads denoised final, writes to historyB (which temporal reads next as historyFrame=1)
  // BG_B: writes to historyA
  const denoisedDiff = dnFinalInPing ? pingTex : pongTex;
  const denoisedSpec = dnFinalInPing ? specPingTex : specPongTex;
  const copyToHistBG_A = device.createBindGroup({ layout: dnAtrousLayout, entries: [
    { binding: 0, resource: { buffer: dnParamBufs[0] } },
    { binding: 1, resource: denoisedDiff.createView() },        // denoised diffuse .rgb
    { binding: 2, resource: historyB.createView() },            // write → historyB
    { binding: 3, resource: hdrTex.createView() },              // temporal output .a = history_len
    { binding: 4, resource: denoisedSpec.createView() },        // denoised specular .rgb
    { binding: 5, resource: specHistoryB.createView() },        // write → specHistoryB
    { binding: 6, resource: specHdrTex.createView() },          // temporal spec output .a = cam_z
  ]});
  const copyToHistBG_B = device.createBindGroup({ layout: dnAtrousLayout, entries: [
    { binding: 0, resource: { buffer: dnParamBufs[0] } },
    { binding: 1, resource: denoisedDiff.createView() },
    { binding: 2, resource: historyA.createView() },            // write → historyA
    { binding: 3, resource: hdrTex.createView() },
    { binding: 4, resource: denoisedSpec.createView() },
    { binding: 5, resource: specHistoryA.createView() },        // write → specHistoryA
    { binding: 6, resource: specHdrTex.createView() },
  ]});

  // Composite: denoised signals + glass composition
  const compSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  const dnBG_comp = device.createBindGroup({ layout: dnCompLayout, entries: [
    { binding: 0, resource: { buffer: dnCompParamBuf } },
    { binding: 1, resource: (dnFinalInPing ? pingTex : pongTex).createView() },
    { binding: 4, resource: (dnFinalInPing ? specPingTex : specPongTex).createView() },
    { binding: 6, resource: albedoTex.createView() },
    { binding: 7, resource: ptOutputTex.createView() },
    { binding: 8, resource: matIdTex.createView() },
    { binding: 9, resource: ndTex.createView() },
    { binding: 10, resource: prevFrameTex.createView() },
    { binding: 11, resource: compSampler },
    { binding: 12, resource: { buffer: matBuf } },
    { binding: 13, resource: { buffer: exposureBuf } },
  ]});
  // ReBLUR composite: reads from hdrTex/specHdrTex (stabilized output)
  const dnBG_comp_reblur = device.createBindGroup({ layout: dnCompLayout, entries: [
    { binding: 0, resource: { buffer: dnCompParamBuf } },
    { binding: 1, resource: hdrTex.createView() },
    { binding: 4, resource: specHdrTex.createView() },
    { binding: 6, resource: albedoTex.createView() },
    { binding: 7, resource: ptOutputTex.createView() },
    { binding: 8, resource: matIdTex.createView() },
    { binding: 9, resource: ndTex.createView() },
    { binding: 10, resource: prevFrameTex.createView() },
    { binding: 11, resource: compSampler },
    { binding: 12, resource: { buffer: matBuf } },
    { binding: 13, resource: { buffer: exposureBuf } },
  ]});
  const dnBG_comp_noisy = device.createBindGroup({ layout: dnCompLayout, entries: [
    { binding: 0, resource: { buffer: dnCompParamBuf } },
    { binding: 1, resource: noisyTex.createView() },
    { binding: 4, resource: specNoisyTex.createView() },
    { binding: 6, resource: albedoTex.createView() },
    { binding: 7, resource: ptOutputTex.createView() },
    { binding: 8, resource: matIdTex.createView() },
    { binding: 9, resource: ndTex.createView() },
    { binding: 10, resource: prevFrameTex.createView() },
    { binding: 11, resource: compSampler },
    { binding: 12, resource: { buffer: matBuf } },
    { binding: 13, resource: { buffer: exposureBuf } },
  ]});
  // OIDN composite: reads blended denoised from history, alternates A/B
  const dnBG_comp_oidn_A = device.createBindGroup({ layout: dnCompLayout, entries: [
    { binding: 0, resource: { buffer: dnCompParamBuf } },
    { binding: 1, resource: historyB.createView() },  // blend wrote to B when reading A
    { binding: 4, resource: specNoisyTex.createView() },
    { binding: 6, resource: albedoTex.createView() },
    { binding: 7, resource: ptOutputTex.createView() },
    { binding: 8, resource: matIdTex.createView() },
    { binding: 9, resource: ndTex.createView() },
    { binding: 10, resource: prevFrameTex.createView() },
    { binding: 11, resource: compSampler },
    { binding: 12, resource: { buffer: matBuf } },
    { binding: 13, resource: { buffer: exposureBuf } },
  ]});

  const dnBG_comp_oidn_B = device.createBindGroup({ layout: dnCompLayout, entries: [
    { binding: 0, resource: { buffer: dnCompParamBuf } },
    { binding: 1, resource: historyA.createView() },  // blend wrote to A when reading B
    { binding: 4, resource: specNoisyTex.createView() },
    { binding: 6, resource: albedoTex.createView() },
    { binding: 7, resource: ptOutputTex.createView() },
    { binding: 8, resource: matIdTex.createView() },
    { binding: 9, resource: ndTex.createView() },
    { binding: 10, resource: prevFrameTex.createView() },
    { binding: 11, resource: compSampler },
    { binding: 12, resource: { buffer: matBuf } },
    { binding: 13, resource: { buffer: exposureBuf } },
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
  // DEBUG: raw noisy HDR (no denoise, no FSR, no tonemap)
  const dispBGRaw = device.createBindGroup({
    layout: dispBGLayout,
    entries: [
      { binding: 0, resource: noisyTex.createView() },
      { binding: 1, resource: sampler },
    ],
  });

  // --- OIDN Neural Denoiser (optional) ---
  let oidn = null;
  if (denoiseMode.startsWith('oidn')) {
    try {
      const oidnWeights = denoiseMode === 'oidn-fast' ? 'oidn/rt_ldr_alb_nrm_small.tza' : 'oidn/rt_ldr_alb_nrm.tza';
      oidn = await createOIDNPipeline(device, oidnWeights, width, height, hasF16, rlog);

      // Input assembly bind group: textures → 9ch NCHW buffer
      // We feed the combined beauty pass: albedo * diffuse + specular
      const ioParams = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(ioParams, 0, new Uint32Array([width, height, oidn.padW, oidn.padH, 9, 0, 0, 0]));

      // Dummy buffer for unused bind slots (avoids read+write conflict on same buffer)
      const oidnDummy = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });

      oidn.inputBG = device.createBindGroup({ layout: oidn.ioBGL, entries: [
        { binding: 0, resource: { buffer: ioParams } },
        { binding: 1, resource: { buffer: oidnDummy } },          // unused for input (read slot)
        { binding: 2, resource: { buffer: oidn.skipInput } },     // output: 9ch NCHW
        { binding: 3, resource: noisyTex.createView() },          // noisy diffuse irradiance
        { binding: 4, resource: albedoTex.createView() },         // albedo
        { binding: 5, resource: denoiseNdTex.createView() },      // normals
        { binding: 6, resource: pingTex.createView() },           // unused for input (write texture)
        { binding: 7, resource: specNoisyTex.createView() },      // noisy specular radiance
      ]});

      // Input BG reading from TEMPORAL output (pre-converged, more stable for OIDN)
      oidn.inputBG_temporal = device.createBindGroup({ layout: oidn.ioBGL, entries: [
        { binding: 0, resource: { buffer: ioParams } },
        { binding: 1, resource: { buffer: oidnDummy } },
        { binding: 2, resource: { buffer: oidn.skipInput } },
        { binding: 3, resource: hdrTex.createView() },             // temporal accumulated diffuse
        { binding: 4, resource: albedoTex.createView() },
        { binding: 5, resource: denoiseNdTex.createView() },
        { binding: 6, resource: pingTex.createView() },
        { binding: 7, resource: specHdrTex.createView() },         // temporal accumulated specular
      ]});

      // Input BG reading from preblurred textures (pingTex/specPingTex after anti-firefly pass)
      oidn.inputBG_pb = device.createBindGroup({ layout: oidn.ioBGL, entries: [
        { binding: 0, resource: { buffer: ioParams } },
        { binding: 1, resource: { buffer: oidnDummy } },
        { binding: 2, resource: { buffer: oidn.skipInput } },
        { binding: 3, resource: pingTex.createView() },             // preblurred diffuse
        { binding: 4, resource: albedoTex.createView() },
        { binding: 5, resource: denoiseNdTex.createView() },
        { binding: 6, resource: pongTex.createView() },             // unused write slot (not pingTex!)
        { binding: 7, resource: specPingTex.createView() },         // preblurred specular
      ]});

      // Output extraction: 3ch denoised beauty → pingTex
      const outParams = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(outParams, 0, new Uint32Array([oidn.padW, oidn.padH, width, height, 3, 0, 0, 0]));

      oidn.outputBG = device.createBindGroup({ layout: oidn.ioBGL, entries: [
        { binding: 0, resource: { buffer: outParams } },
        { binding: 1, resource: { buffer: oidn.outputBuf } },     // UNet final output (3 channels)
        { binding: 2, resource: { buffer: oidnDummy } },          // unused for output (write slot)
        { binding: 3, resource: noisyTex.createView() },          // unused placeholder
        { binding: 4, resource: albedoTex.createView() },         // albedo (for output_extraction)
        { binding: 5, resource: denoiseNdTex.createView() },      // unused placeholder
        { binding: 6, resource: pingTex.createView() },           // OUTPUT: denoised HDR → pingTex
        { binding: 7, resource: specNoisyTex.createView() },      // unused placeholder
      ]});

      // Temporal blend with reprojection: stabilize OIDN output
      const blendBGL = device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
        { binding: 8, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 9, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 10, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
      ]});
      const blendOpsCode = await fetch(`oidn-ops.wgsl?v=${Date.now()}`).then(r=>r.text());
      const blendPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [blendBGL] }),
        compute: { module: device.createShaderModule({ code: blendOpsCode, strictMath: false }), entryPoint: 'temporal_blend' },
      });
      const blendParams = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      // BlendCameraParams: 4 u32 + 8 vec4f + 1 f32 + 1 f32 + 2 f32 pad = 160 bytes
      const blendCamBuf = device.createBuffer({ size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const dummyReadBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
      const dummyWriteBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
      const blendSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

      // Ping-pong: frame 0 reads historyA, writes historyB. Frame 1: reverse.
      oidn.blendBG_A = device.createBindGroup({ layout: blendBGL, entries: [
        { binding: 0, resource: { buffer: blendParams } },
        { binding: 1, resource: { buffer: dummyReadBuf } },
        { binding: 2, resource: { buffer: dummyWriteBuf } },
        { binding: 3, resource: { buffer: blendCamBuf } },
        { binding: 4, resource: ndTex.createView() },          // depth from G-buffer
        { binding: 5, resource: blendSampler },
        { binding: 8, resource: pingTex.createView() },        // current denoised
        { binding: 9, resource: historyA.createView() },       // previous blended
        { binding: 10, resource: historyB.createView() },      // output
      ]});
      oidn.blendBG_B = device.createBindGroup({ layout: blendBGL, entries: [
        { binding: 0, resource: { buffer: blendParams } },
        { binding: 1, resource: { buffer: dummyReadBuf } },
        { binding: 2, resource: { buffer: dummyWriteBuf } },
        { binding: 3, resource: { buffer: blendCamBuf } },
        { binding: 4, resource: ndTex.createView() },
        { binding: 5, resource: blendSampler },
        { binding: 8, resource: pingTex.createView() },
        { binding: 9, resource: historyB.createView() },
        { binding: 10, resource: historyA.createView() },
      ]});
      oidn.blendPipeline = blendPipeline;
      oidn.blendParams = blendParams;
      oidn.blendCamBuf = blendCamBuf;
      oidn.blendFrame = 0;

      rlog('OIDN neural denoiser initialized');
    } catch (e) {
      rlog('OIDN init failed: ' + e.message + '. Falling back to SVGF.');
      oidn = null;
    }
  }

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
  const TONEMAP_NAMES = ['AgX Punchy', 'ACES', 'Reinhard', 'Uncharted 2', 'PBR Neutral', 'Standard', 'None'];
  const settings = {
    sunElevation: 58,
    sunAzimuth: 0, // DEBUG: zero azimuth to verify sun alignment
    sharpness: 0.6,
    temporalAlpha: 0.02,
    tonemapMode: cfg.tonemap !== undefined ? cfg.tonemap : 0, // default: AgX Punchy (preserves saturation)
    exposure: 1.0,
    saturation: 1.0,
    contrast: 0.0,
  };

  function getSunDir() {
    const el = settings.sunElevation * Math.PI / 180;
    const az = settings.sunAzimuth * Math.PI / 180;
    return [Math.sin(az) * Math.cos(el), Math.sin(el), Math.cos(az) * Math.cos(el)];
  }

  // --- Cycles-like sky environment ---
  let lastEnvSunEl = -999, lastEnvSunAz = -999;
  const skyModel = new CyclesSkyModel(ENV_W, ENV_H, CYCLES_SKY_DEFAULTS);

  function rebuildSkyEnvironment() {
    const t0 = performance.now();
    const skyState = skyModel.update({
      sunElevation: settings.sunElevation * Math.PI / 180,
      sunAzimuth: settings.sunAzimuth * Math.PI / 180,
    });
    device.queue.writeBuffer(envCdfBuf, 0, skyState.buffer);
    lastEnvSunEl = settings.sunElevation;
    lastEnvSunAz = settings.sunAzimuth;
    const parts = [];
    if (skyState.rebuiltAtmosphere) parts.push('atmosphere');
    if (skyState.rebuiltTexture) parts.push('lut');
    if (skyState.rebuiltSunData) parts.push('sun');
    if (skyState.rebuiltPacked) parts.push('cdf');
    rlog(`Cycles sky rebuilt (${parts.join(' + ') || 'cached'}) in ${(performance.now() - t0).toFixed(1)}ms`);
  }

  rebuildSkyEnvironment();

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

  let debugMode = false;
  const debugPanel = document.getElementById('debug-panel');
  const debugContent = document.getElementById('debug-content');
  const debugCrosshair = document.getElementById('debug-crosshair');

  function toggleDebug() {
    debugMode = !debugMode;
    debugPanel.classList.toggle('open', debugMode);
    debugCrosshair.style.display = debugMode ? 'block' : 'none';
    if (debugMode) {
      if (menuOpen) toggleMenu();
      document.exitPointerLock?.();
      for (const k in keys) keys[k] = false;
      debugContent.textContent = 'Click to inspect...';
    }
  }

  // --- CPU ray-BVH picker for debug ---
  function debugPick(cssX, cssY) {
    const rect = canvas.getBoundingClientRect();
    // Map CSS click to NDC [-1,+1]
    // Y inverted: display pipeline flips Y (clip -1=bottom → UV 0=texture top)
    const nx = ((cssX - rect.left) / rect.width) * 2 - 1;
    const ny = -(((cssY - rect.top) / rect.height) * 2 - 1);
    const {forward: fw, right: rt, up: u} = getCameraVectors();
    const ff = Math.tan((camera.fov * Math.PI / 180) * 0.5);
    const asp = rect.width / rect.height;
    const dx = fw[0]+nx*asp*ff*rt[0]+ny*ff*u[0], dy = fw[1]+nx*asp*ff*rt[1]+ny*ff*u[1], dz = fw[2]+nx*asp*ff*rt[2]+ny*ff*u[2];
    const dl = Math.sqrt(dx*dx+dy*dy+dz*dz);
    const dir = [dx/dl, dy/dl, dz/dl];
    const orig = camera.pos;
    const invD = [1/dir[0], 1/dir[1], 1/dir[2]];
    const bF = scene.gpuBVHNodes, bU = new Uint32Array(bF.buffer, bF.byteOffset, bF.length);
    const pos = scene.gpuPositions, nrm = scene.gpuNormals, td = scene.gpuTriData;
    let bestT = 1e30, bestTri = -1, bestBary = [0,0];

    function aabb(o) {
      const t1x=(bF[o]-orig[0])*invD[0], t2x=(bF[o+4]-orig[0])*invD[0];
      const t1y=(bF[o+1]-orig[1])*invD[1], t2y=(bF[o+5]-orig[1])*invD[1];
      const t1z=(bF[o+2]-orig[2])*invD[2], t2z=(bF[o+6]-orig[2])*invD[2];
      const tmn=Math.max(Math.min(t1x,t2x),Math.min(t1y,t2y),Math.min(t1z,t2z));
      const tmx=Math.min(Math.max(t1x,t2x),Math.max(t1y,t2y),Math.max(t1z,t2z));
      return (tmx>=Math.max(tmn,0)&&tmn<bestT)?Math.max(tmn,0):1e30;
    }
    function tri(ti) {
      const i0=td[ti*4]*4, i1=td[ti*4+1]*4, i2=td[ti*4+2]*4;
      const e1=[pos[i1]-pos[i0],pos[i1+1]-pos[i0+1],pos[i1+2]-pos[i0+2]];
      const e2=[pos[i2]-pos[i0],pos[i2+1]-pos[i0+1],pos[i2+2]-pos[i0+2]];
      const h=[dir[1]*e2[2]-dir[2]*e2[1],dir[2]*e2[0]-dir[0]*e2[2],dir[0]*e2[1]-dir[1]*e2[0]];
      const a=e1[0]*h[0]+e1[1]*h[1]+e1[2]*h[2];
      if(Math.abs(a)<1e-8)return;
      const f=1/a, s=[orig[0]-pos[i0],orig[1]-pos[i0+1],orig[2]-pos[i0+2]];
      const bu=f*(s[0]*h[0]+s[1]*h[1]+s[2]*h[2]);
      if(bu<0||bu>1)return;
      const q=[s[1]*e1[2]-s[2]*e1[1],s[2]*e1[0]-s[0]*e1[2],s[0]*e1[1]-s[1]*e1[0]];
      const bv=f*(dir[0]*q[0]+dir[1]*q[1]+dir[2]*q[2]);
      if(bv<0||bu+bv>1)return;
      const t=f*(e2[0]*q[0]+e2[1]*q[1]+e2[2]*q[2]);
      if(t>1e-5&&t<bestT){bestT=t;bestTri=ti;bestBary=[bu,bv];}
    }

    const stk = [0];
    while (stk.length > 0) {
      const ni = stk.pop(), o = ni * 8;
      const cnt = bU[o+7], lf = bU[o+3];
      if (cnt > 0) { for (let i = 0; i < cnt; i++) tri(lf+i); }
      else {
        const tl = aabb(lf*8), tr = aabb((lf+1)*8);
        if(tl<tr){if(tr<bestT)stk.push(lf+1);if(tl<bestT)stk.push(lf);}
        else{if(tl<bestT)stk.push(lf);if(tr<bestT)stk.push(lf+1);}
      }
    }
    if (bestTri < 0) return null;

    const mi = td[bestTri*4+3];
    const v0=td[bestTri*4]*4, v1=td[bestTri*4+1]*4, v2=td[bestTri*4+2]*4;
    const bw=1-bestBary[0]-bestBary[1];
    const huv = [
      bw*pos[v0+3]+bestBary[0]*pos[v1+3]+bestBary[1]*pos[v2+3],
      bw*nrm[v0+3]+bestBary[0]*nrm[v1+3]+bestBary[1]*nrm[v2+3],
    ];
    const hn = [
      bw*nrm[v0]+bestBary[0]*nrm[v1]+bestBary[1]*nrm[v2],
      bw*nrm[v0+1]+bestBary[0]*nrm[v1+1]+bestBary[1]*nrm[v2+1],
      bw*nrm[v0+2]+bestBary[0]*nrm[v1+2]+bestBary[1]*nrm[v2+2],
    ];
    const nl = Math.sqrt(hn[0]*hn[0]+hn[1]*hn[1]+hn[2]*hn[2]) || 1;
    const hitPos = [orig[0]+dir[0]*bestT, orig[1]+dir[1]*bestT, orig[2]+dir[2]*bestT];
    return { triIdx:bestTri, matIdx:mi, uv:huv, t:bestT, normal:[hn[0]/nl,hn[1]/nl,hn[2]/nl], hitPos };
  }

  const pickModal = document.getElementById('pick-modal');
  const pickTitle = document.getElementById('pick-title');
  const pickProps = document.getElementById('pick-props');
  const pickTextures = document.getElementById('pick-textures');
  document.getElementById('pick-close').addEventListener('click', () => { pickModal.style.display = 'none'; });
  pickModal.addEventListener('click', (e) => { if (e.target === pickModal) pickModal.style.display = 'none'; });

  // Upload material changes to GPU (both buffers)
  function uploadMaterial(matIdx) {
    const stride = scene.materialStride || 80;
    const byteOffset = matIdx * stride * 4;
    const slice = new Float32Array(scene.gpuMaterials.buffer, byteOffset, stride);
    device.queue.writeBuffer(matBuf, byteOffset, slice);
    device.queue.writeBuffer(matBufForGbuf, byteOffset, slice);
  }

  function showPickInfo(pick) {
    if (!pick) { debugContent.textContent = 'No hit (sky)'; return; }
    const stride = scene.materialStride || 80;
    const m = scene.gpuMaterials, o = pick.matIdx * stride;
    const names = scene.materialNames || [];
    const types = ['PBR','Unlit','Reserved','Transmission'];
    const alphaModes = ['Opaque','Mask','Blend'];
    const name = names[pick.matIdx] || 'unknown';
    const mi = pick.matIdx;

    debugContent.textContent = `${name} | Tri #${pick.triIdx} | d=${pick.t.toFixed(2)}`;
    pickTitle.textContent = `#${mi} "${name}"`;

    const albHex = '#' + [m[o],m[o+1],m[o+2]].map(v => Math.round(Math.min(v,1)*255).toString(16).padStart(2,'0')).join('');
    const isEmissive = (m[o+4] + m[o+5] + m[o+6]) > 0.001 || m[o+25] >= 0;

    // Helper: create editable slider row
    function sliderRow(label, value, min, max, step, onChange) {
      const id = `mat-${label.replace(/\s/g,'-').toLowerCase()}-${mi}`;
      return `<div style="margin-bottom:6px; display:flex; align-items:center; gap:8px;">
        <span style="color:#888; flex:0 0 80px; font-size:11px;">${label}</span>
        <input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${value}"
          style="flex:1; accent-color:#0c0; height:4px;">
        <span id="${id}-v" style="color:#fff; flex:0 0 40px; text-align:right; font-size:11px;">${Number(value).toFixed(2)}</span>
      </div>`;
    }

    // Build editable properties
    let html = `
      <div style="margin-bottom:10px;">
        <span style="color:#888;">Type:</span>
        <select id="mat-type-${mi}" style="background:#222; color:#fff; border:1px solid #444; padding:2px 6px; border-radius:3px; font-size:11px;">
          ${types.map((t,i) => `<option value="${i}" ${Math.round(m[o+3])===i?'selected':''}>${t}</option>`).join('')}
        </select>
        &nbsp;&nbsp;<span style="color:#555;">Tri #${pick.triIdx} | Dist: ${pick.t.toFixed(2)}</span>
      </div>
      <div style="margin-bottom:10px;">
        <span style="color:#888; font-size:11px;">Albedo</span>
        <input type="color" id="mat-albedo-${mi}" value="${albHex}"
          style="vertical-align:middle; width:28px; height:20px; border:1px solid #444; border-radius:3px; cursor:pointer;">
      </div>
      ${sliderRow('Roughness', m[o+7], 0.01, 1, 0.01)}
      ${sliderRow('Metallic', m[o+8], 0, 1, 0.01)}
      ${sliderRow('IoR', m[o+14], 1, 3, 0.01)}
    `;

    if (isEmissive) {
      html += `
        <div style="margin-top:8px; padding-top:8px; border-top:1px solid rgba(255,255,255,0.05);">
          ${sliderRow('Strength', m[o+15], 0, 100, 0.1)}
          ${sliderRow('Emission R', m[o+4], 0, 1, 0.01)}
          ${sliderRow('Emission G', m[o+5], 0, 1, 0.01)}
          ${sliderRow('Emission B', m[o+6], 0, 1, 0.01)}
        </div>
      `;
    }

    html += `
      <div style="margin-top:6px; color:#444; font-size:10px;">
        Normal: [${pick.normal.map(v=>v.toFixed(3)).join(', ')}] | UV: [${pick.uv.map(v=>v.toFixed(3)).join(', ')}]
      </div>
    `;

    pickProps.innerHTML = html;

    // Bind change events
    function bindSliderProp(label, offset) {
      const id = `mat-${label.replace(/\s/g,'-').toLowerCase()}-${mi}`;
      const sl = document.getElementById(id);
      const vl = document.getElementById(id + '-v');
      if (!sl) return;
      sl.addEventListener('input', () => {
        const v = Number(sl.value);
        vl.textContent = v.toFixed(2);
        m[o + offset] = v;
        uploadMaterial(mi);
      });
    }

    bindSliderProp('Roughness', 7);
    bindSliderProp('Metallic', 8);
    bindSliderProp('IoR', 14);
    bindSliderProp('Strength', 15);
    bindSliderProp('Emission R', 4);
    bindSliderProp('Emission G', 5);
    bindSliderProp('Emission B', 6);

    // Albedo color picker
    const albPicker = document.getElementById(`mat-albedo-${mi}`);
    if (albPicker) {
      albPicker.addEventListener('input', () => {
        const hex = albPicker.value;
        m[o]   = parseInt(hex.slice(1,3), 16) / 255;
        m[o+1] = parseInt(hex.slice(3,5), 16) / 255;
        m[o+2] = parseInt(hex.slice(5,7), 16) / 255;
        uploadMaterial(mi);
      });
    }

    // Type selector
    const typeSel = document.getElementById(`mat-type-${mi}`);
    if (typeSel) {
      typeSel.addEventListener('change', () => {
        m[o+3] = Number(typeSel.value);
        uploadMaterial(mi);
        showPickInfo(pick); // rebuild UI (show/hide emission)
      });
    }

    // Texture previews
    const texInfo = scene.textureInfo;
    const texIds = { base: m[o+9], mr: m[o+10], normal: m[o+11] };
    const texLabels = { base: 'Base Color', mr: 'Metal/Rough', normal: 'Normal' };
    let texHTML = '';
    if (texInfo) {
      for (const [key, idx] of Object.entries(texIds)) {
        const ti = Math.round(idx);
        if (ti >= 0 && ti < texInfo.count) {
          const texURL = resolveSceneAssetURL(texInfo.imageURIs[ti]);
          texHTML += `<div style="margin-bottom:8px;">
            <div style="font-size:10px; text-transform:uppercase; letter-spacing:1px; color:#0a0; margin-bottom:4px;">${texLabels[key]} <span style="color:#444;">(#${ti})</span></div>
            <img src="${texURL || ''}" style="width:100%; max-height:140px; object-fit:contain; border-radius:4px; border:1px solid rgba(255,255,255,0.08); background:#111;">
          </div>`;
        } else {
          texHTML += `<div style="margin-bottom:4px;"><span style="font-size:10px; text-transform:uppercase; color:#333;">${texLabels[key]}: none</span></div>`;
        }
      }
    }
    pickTextures.innerHTML = texHTML;
    pickModal.style.display = 'flex';
  }

  document.addEventListener('keydown', e => {
    if (e.code === 'Escape') { toggleMenu(); e.preventDefault(); return; }
    if (e.code === 'Tab') { toggleDebug(); e.preventDefault(); return; }
    if (!menuOpen && !debugMode) keys[e.code] = true;
  });
  document.addEventListener('keyup', e => { keys[e.code] = false; });

  const isTouchDevice = ('ontouchstart' in window) || navigator.maxTouchPoints > 0;

  canvas.addEventListener('click', (e) => {
    if (debugMode) {
      showPickInfo(debugPick(e.clientX, e.clientY));
      return;
    }
    if (!isTouchDevice && !menuOpen) canvas.requestPointerLock();
  });
  document.addEventListener('pointerlockchange', () => {
    pointerLocked = document.pointerLockElement === canvas;
  });
  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    if (menuOpen || debugMode) return;
    camera.speed *= e.deltaY > 0 ? 0.85 : 1.18;
    camera.speed = Math.max(0.1, Math.min(50, camera.speed));
  }, {passive: false});

  document.addEventListener('mousemove', e => {
    if (!pointerLocked || menuOpen || debugMode) return;
    camera.yaw -= e.movementX * camera.sensitivity;
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

  // Long-press debug pick (hold 600ms without moving)
  let longPressTimer = null;
  let longPressPos = null;
  let longPressFired = false;

  function startLongPress(x, y) {
    longPressPos = { x, y };
    longPressFired = false;
    clearTimeout(longPressTimer);
    longPressTimer = setTimeout(() => {
      longPressFired = true;
      // Activate debug mode and pick
      if (!debugMode) {
        debugMode = true;
        debugPanel.classList.add('open');
        debugCrosshair.style.display = 'block';
      }
      showPickInfo(debugPick(x, y));
      // Haptic feedback if available
      if (navigator.vibrate) navigator.vibrate(30);
    }, 600);
  }

  function cancelLongPress() {
    clearTimeout(longPressTimer);
    longPressTimer = null;
  }

  document.addEventListener('touchstart', e => {
    if (e.touches.length >= 3) {
      threeFingerStart = performance.now();
      cancelLongPress();
      hideStick(stickLeft);
      hideStick(stickRight);
      e.preventDefault();
      return;
    }
    if (menuOpen) return;

    // Start long-press detection on single touch
    if (e.touches.length === 1) {
      startLongPress(e.touches[0].clientX, e.touches[0].clientY);
    }

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
      threeFingerStart = null;
    }
    // Cancel long-press if finger moves more than 10px
    if (longPressPos && e.touches.length === 1) {
      const dx = e.touches[0].clientX - longPressPos.x;
      const dy = e.touches[0].clientY - longPressPos.y;
      if (dx*dx + dy*dy > 100) cancelLongPress();
    }
    if (menuOpen) return;
    for (const t of e.changedTouches) {
      if (t.identifier === stickLeft.touchId) { moveKnob(stickLeft, t.clientX, t.clientY); e.preventDefault(); }
      else if (t.identifier === stickRight.touchId) { moveKnob(stickRight, t.clientX, t.clientY); e.preventDefault(); }
    }
  }, {passive: false});

  function onTouchEnd(e) {
    cancelLongPress();
    if (threeFingerStart !== null && e.touches.length === 0) {
      const elapsed = performance.now() - threeFingerStart;
      threeFingerStart = null;
      if (elapsed < 500) { toggleMenu(); return; }
    }
    // If long-press just fired, don't process as stick release
    if (longPressFired) { longPressFired = false; return; }
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

  // Color controls
  bindSlider('opt-exposure', 'val-exposure',
    () => settings.exposure * 100,
    v => settings.exposure = v / 100,
    v => (v / 100).toFixed(2));

  bindSlider('opt-saturation', 'val-saturation',
    () => settings.saturation * 100,
    v => settings.saturation = v / 100,
    v => (v / 100).toFixed(2));

  bindSlider('opt-contrast', 'val-contrast',
    () => settings.contrast * 100,
    v => settings.contrast = v / 100,
    v => (v / 100).toFixed(2));

  const optTonemap = document.getElementById('opt-tonemap');
  if (optTonemap) {
    optTonemap.value = settings.tonemapMode;
    optTonemap.addEventListener('change', e => settings.tonemapMode = Number(e.target.value));
  }

  function refreshMenu() {
    const sets = [
      ['opt-sharp', 'val-sharp', settings.sharpness * 100, v => (v / 100).toFixed(2)],
      ['opt-sun-el', 'val-sun-el', settings.sunElevation, v => v + '\u00B0'],
      ['opt-sun-az', 'val-sun-az', settings.sunAzimuth, v => v + '\u00B0'],
      ['opt-speed', 'val-speed', camera.speed * 10, v => (v / 10).toFixed(1)],
      ['opt-fov', 'val-fov', camera.fov, v => v + '\u00B0'],
      ['opt-temporal', 'val-temporal', settings.temporalAlpha * 1000, v => (v / 1000).toFixed(3)],
      ['opt-exposure', 'val-exposure', settings.exposure * 100, v => (v / 100).toFixed(2)],
      ['opt-saturation', 'val-saturation', settings.saturation * 100, v => (v / 100).toFixed(2)],
      ['opt-contrast', 'val-contrast', settings.contrast * 100, v => (v / 100).toFixed(2)],
    ];
    for (const [sid, vid, val, fmt] of sets) {
      const el = document.getElementById(sid);
      if (el) { el.value = val; document.getElementById(vid).textContent = fmt(val); }
    }
    optFsr.value = fsrMode;
    if (optTonemap) optTonemap.value = settings.tonemapMode;
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
    function proj(wx,wy,wz){return{x:cy*wx-sy*wz,y:-(sp*sy*wx+cp*wy+sp*cy*wz),z:-(cp*sy*wx+sp*wy+cp*cy*wz)};}
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

  // Halton sequence for sub-pixel jitter (base 2 and 3, cycles every 256 frames)
  function halton(index, base) {
    let f = 1, r = 0;
    let i = index;
    while (i > 0) { f /= base; r += f * (i % base); i = Math.floor(i / base); }
    return r;
  }

  // --- Render state ---
  let frameIndex = 0, cameraMoved = false, framesStill = 0, autoExposure = 1.0;
  // Reusable staging buffer for autoexposure readback (avoids creating one per frame)
  let exposureStagingBusy = false;
  const exposureStagingBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  let lastTime = performance.now(), fps = 0, fpsAccum = 0, fpsCount = 0;

  // --- Per-pass GPU timing (CPU-side submit timing) ---
  const passTimers = { gbuffer: 0, shadow: 0, pathtrace: 0, temporal: 0, denoise: 0, composite: 0, fsr: 0, oidn: 0, total: 0 };
  let timerLogCounter = 0;

  function getCameraVectors() {
    const fw = [Math.cos(camera.pitch)*Math.sin(camera.yaw), Math.sin(camera.pitch), Math.cos(camera.pitch)*Math.cos(camera.yaw)];
    const rt = [-Math.cos(camera.yaw), 0, Math.sin(camera.yaw)];
    const up = [-Math.sin(camera.pitch)*Math.sin(camera.yaw), Math.cos(camera.pitch), -Math.sin(camera.pitch)*Math.cos(camera.yaw)];
    return { forward:fw, right:rt, up };
  }

  function buildLightViewProj(sunDir, sceneCenter, sceneRadius) {
    // Orthographic projection from sun's direction covering the scene
    const lx = sunDir[0], ly = sunDir[1], lz = sunDir[2];
    // Light position: far back along sun direction
    const lp = [sceneCenter[0] - lx * sceneRadius * 2, sceneCenter[1] - ly * sceneRadius * 2, sceneCenter[2] - lz * sceneRadius * 2];
    // Build look-at: forward = sunDir, compute right and up
    const fw = [lx, ly, lz];
    // Pick a stable up vector that's not parallel to sun
    const worldUp = Math.abs(ly) < 0.99 ? [0, 1, 0] : [1, 0, 0];
    const rt = [fw[1]*worldUp[2]-fw[2]*worldUp[1], fw[2]*worldUp[0]-fw[0]*worldUp[2], fw[0]*worldUp[1]-fw[1]*worldUp[0]];
    const rl = Math.sqrt(rt[0]*rt[0]+rt[1]*rt[1]+rt[2]*rt[2]);
    rt[0] /= rl; rt[1] /= rl; rt[2] /= rl;
    const up = [rt[1]*fw[2]-rt[2]*fw[1], rt[2]*fw[0]-rt[0]*fw[2], rt[0]*fw[1]-rt[1]*fw[0]];
    // View matrix
    const view = new Float32Array([
      rt[0], up[0], -fw[0], 0,
      rt[1], up[1], -fw[1], 0,
      rt[2], up[2], -fw[2], 0,
      -(rt[0]*lp[0]+rt[1]*lp[1]+rt[2]*lp[2]),
      -(up[0]*lp[0]+up[1]*lp[1]+up[2]*lp[2]),
      (fw[0]*lp[0]+fw[1]*lp[1]+fw[2]*lp[2]),
      1,
    ]);
    // Orthographic projection: [-r, r] x [-r, r] x [near, far] → NDC z [0, 1]
    // View-space z is negative (convention: -fw in view matrix), so negate in projection
    const r = sceneRadius;
    const near = 0.01, far = sceneRadius * 4;
    const proj = new Float32Array(16);
    proj[0] = 1 / r;
    proj[5] = -1 / r;  // flip Y for WebGPU
    proj[10] = -1 / (far - near);         // negate: view.z is negative for objects in front
    proj[14] = -near / (far - near);
    proj[15] = 1;
    // viewProj = proj * view
    const vp = new Float32Array(16);
    for (let c = 0; c < 4; c++)
      for (let r2 = 0; r2 < 4; r2++)
        vp[c*4+r2] = proj[r2]*view[c*4] + proj[4+r2]*view[c*4+1] + proj[8+r2]*view[c*4+2] + proj[12+r2]*view[c*4+3];
    return vp;
  }

  function buildViewProj(cam, fw, rt, up, w, h) {
    // View matrix (column-major, WebGPU convention)
    const p = cam.pos;
    const view = new Float32Array([
      rt[0], up[0], -fw[0], 0,
      rt[1], up[1], -fw[1], 0,
      rt[2], up[2], -fw[2], 0,
      -(rt[0]*p[0]+rt[1]*p[1]+rt[2]*p[2]),
      -(up[0]*p[0]+up[1]*p[1]+up[2]*p[2]),
      (fw[0]*p[0]+fw[1]*p[1]+fw[2]*p[2]),
      1,
    ]);
    // Perspective projection (no jitter — requires reprojection compensation to work)
    const fov = cam.fov * Math.PI / 180;
    const asp = w / h;
    const near = 0.01, far = 200.0;
    const f = 1 / Math.tan(fov / 2);
    const proj = new Float32Array(16);
    proj[0] = f / asp;
    proj[5] = -f;  // negate Y: match compute shader convention
    proj[10] = far / (near - far); proj[11] = -1;
    proj[14] = (near * far) / (near - far);
    // viewProj = proj * view (column-major multiply)
    const vp = new Float32Array(16);
    for (let c = 0; c < 4; c++)
      for (let r = 0; r < 4; r++)
        vp[c*4+r] = proj[r]*view[c*4] + proj[4+r]*view[c*4+1] + proj[8+r]*view[c*4+2] + proj[12+r]*view[c*4+3];
    return vp;
  }

  function updateCamera(dt) {
    if (menuOpen || debugMode) return;
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
    if(rx||ry){const ls=2.5*dt;camera.yaw-=rx*ls;camera.pitch-=ry*ls;camera.pitch=Math.max(-Math.PI*0.49,Math.min(Math.PI*0.49,camera.pitch));moved=true;}
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

    if (cameraMoved) { cameraMoved = false; framesStill = 0; } else { framesStill++; }

    const {forward, right, up} = getCameraVectors();
    const fovFactor = Math.tan((camera.fov * Math.PI / 180) * 0.5);
    const aspect = width / height;

    frameIndex++;
    restirFrame = 1 - restirFrame;

    const t_frame_start = performance.now();

    // --- G-buffer rasterization (replaces primary ray BVH traversal) ---
    const viewProj = buildViewProj(camera, forward, right, up, width, height);
    const gbufUd = new Float32Array(20); // mat4x4 + vec3 + pad = 20 floats = 80 bytes
    gbufUd.set(viewProj, 0);
    gbufUd[16] = camera.pos[0]; gbufUd[17] = camera.pos[1]; gbufUd[18] = camera.pos[2]; gbufUd[19] = 0;
    device.queue.writeBuffer(gbufUniformBuf, 0, gbufUd);

    {
      const encoder = device.createCommandEncoder();
      const rp = encoder.beginRenderPass({
        colorAttachments: [
          { view: ndTex.createView(), loadOp: 'clear', storeOp: 'store', clearValue: {r:0,g:0,b:0,a:65000} },
          { view: matIdTex.createView(), loadOp: 'clear', storeOp: 'store', clearValue: {r:-1,g:0,b:0,a:0} },
        ],
        depthStencilAttachment: {
          view: zBuf.createView(), depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 1.0,
        },
      });
      rp.setPipeline(gbufPipeline);
      rp.setBindGroup(0, gbufBG);
      rp.setVertexBuffer(0, posVB);
      rp.setVertexBuffer(1, nrmVB);
      rp.setVertexBuffer(2, matIdVB);
      rp.setVertexBuffer(3, uvExtraVB);
      rp.setIndexBuffer(indexBuf, 'uint32');
      rp.drawIndexed(triCount * 3);
      rp.end();
      device.queue.submit([encoder.finish()]);
    }
    const t_gbuffer = performance.now();

    // --- Rebuild Cycles sky environment when the sun moves ---
    const sun = getSunDir();
    if (settings.sunElevation !== lastEnvSunEl || settings.sunAzimuth !== lastEnvSunAz) {
      rebuildSkyEnvironment();
      cameraMoved = true;
    }

    // --- Shadow map rasterization ---
    const lightViewProj = buildLightViewProj(sun, sceneCenter, sceneRadius);
    device.queue.writeBuffer(shadowUniformBuf, 0, lightViewProj);
    {
      const encoder = device.createCommandEncoder();
      const rp = encoder.beginRenderPass({
        colorAttachments: [],
        depthStencilAttachment: {
          view: shadowDepthTex.createView(), depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 1.0,
        },
      });
      rp.setPipeline(shadowPipeline);
      rp.setBindGroup(0, shadowBG);
      rp.setVertexBuffer(0, posVB);
      rp.setVertexBuffer(1, nrmVB);
      rp.setVertexBuffer(2, matIdVB);
      rp.setVertexBuffer(3, uvExtraVB);
      rp.setIndexBuffer(indexBuf, 'uint32');
      rp.drawIndexed(triCount * 3);
      rp.end();
      device.queue.submit([encoder.finish()]);
    }
    const t_shadow = performance.now();

    // Write PT uniforms
    const ud = new ArrayBuffer(uniformBufferSize);
    const f32 = new Float32Array(ud);
    const u32 = new Uint32Array(ud);
    f32[0] = width; f32[1] = height;
    u32[2] = 1; u32[3] = frameIndex;
    f32[4] = camera.pos[0]; f32[5] = camera.pos[1]; f32[6] = camera.pos[2]; f32[7] = 0;
    f32[8] = forward[0]; f32[9] = forward[1]; f32[10] = forward[2]; f32[11] = 0;
    f32[12] = right[0]; f32[13] = right[1]; f32[14] = right[2]; f32[15] = 0;
    f32[16] = up[0]; f32[17] = up[1]; f32[18] = up[2]; f32[19] = fovFactor;
    f32[20] = sun[0]; f32[21] = sun[1]; f32[22] = sun[2];
    u32[23] = stats.emissiveTris;
    u32[24] = maxBounces;
    u32[25] = framesStill;
    f32[26] = aspect;
    u32[27] = restirEnabled ? 1 : 0;
    // Previous camera for ReSTIR reprojection
    f32[28] = prevCam.pos[0]; f32[29] = prevCam.pos[1]; f32[30] = prevCam.pos[2]; f32[31] = 0;
    f32[32] = prevCam.fwd[0]; f32[33] = prevCam.fwd[1]; f32[34] = prevCam.fwd[2]; f32[35] = 0;
    f32[36] = prevCam.right[0]; f32[37] = prevCam.right[1]; f32[38] = prevCam.right[2]; f32[39] = 0;
    f32[40] = prevCam.up[0]; f32[41] = prevCam.up[1]; f32[42] = prevCam.up[2]; f32[43] = 0;
    u32[44] = punctualLights.length;
    u32[45] = punctualLights.length === 0 ? 1 : 0;
    u32[46] = 0;
    u32[47] = 0;
    // light_view_proj matrix (mat4x4f at float offset 48)
    f32.set(lightViewProj, 48);
    for (let i = 0; i < MAX_PUNCTUAL_LIGHTS; i++) {
      const light = punctualLights[i];
      const base = 64 + i * 16;
      if (!light) {
        for (let j = 0; j < 16; j++) f32[base + j] = 0;
        continue;
      }
      f32[base] = light.position[0];
      f32[base + 1] = light.position[1];
      f32[base + 2] = light.position[2];
      f32[base + 3] = light.range || 0;
      f32[base + 4] = light.direction[0];
      f32[base + 5] = light.direction[1];
      f32[base + 6] = light.direction[2];
      f32[base + 7] = Math.cos(light.innerConeAngle || 0);
      f32[base + 8] = light.color[0];
      f32[base + 9] = light.color[1];
      f32[base + 10] = light.color[2];
      f32[base + 11] = light.intensity || 1;
      f32[base + 12] = light.type === 'directional' ? 0 : (light.type === 'spot' ? 2 : 1);
      f32[base + 13] = Math.cos(light.outerConeAngle ?? (Math.PI * 0.25));
      f32[base + 14] = 0;
      f32[base + 15] = 0;
    }
    device.queue.writeBuffer(uniformBuffer, 0, ud);

    // Write temporal uniforms (current + previous camera)
    const td = new Float32Array(40); // 10 vec4f
    td[0] = width; td[1] = height; td[2] = settings.temporalAlpha; td[3] = framesStill;
    td[4]=right[0];td[5]=right[1];td[6]=right[2];td[7]=0;         // cam_right
    td[8]=up[0];td[9]=up[1];td[10]=up[2];td[11]=0;                // cam_up
    td[12]=forward[0];td[13]=forward[1];td[14]=forward[2];td[15]=0;// cam_fwd
    td[16]=camera.pos[0];td[17]=camera.pos[1];td[18]=camera.pos[2];td[19]=0; // cam_pos
    td[20]=prevCam.right[0];td[21]=prevCam.right[1];td[22]=prevCam.right[2];td[23]=0;
    td[24]=prevCam.up[0];td[25]=prevCam.up[1];td[26]=prevCam.up[2];td[27]=0;
    td[28]=prevCam.fwd[0];td[29]=prevCam.fwd[1];td[30]=prevCam.fwd[2];td[31]=0;
    td[32]=prevCam.pos[0];td[33]=prevCam.pos[1];td[34]=prevCam.pos[2];td[35]=0;
    td[36]=fovFactor;td[37]=aspect;td[38]=0.1;td[39]=512.0; // depth_reject_scale, max_history
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

      // Multi-SPP loop: trace multiple samples per frame
      for (let spp = 0; spp < sppPerFrame; spp++) {
        if (spp > 0) {
          frameIndex++;
          u32[3] = frameIndex; // update seed for different noise pattern
          device.queue.writeBuffer(uniformBuffer, 0, ud);
        }
        const ptPass = encoder.beginComputePass();
        ptPass.setPipeline(computePipeline);
        ptPass.setBindGroup(0, bg0);
        ptPass.setBindGroup(1, bg1);
        ptPass.setBindGroup(2, restirFrame === 0 ? bg2_A : bg2_B);
        ptPass.setBindGroup(3, bg3);
        ptPass.dispatchWorkgroups(Math.ceil(width/16), Math.ceil(height/16));
        ptPass.end();
      }

      // SHaRC resolve
      if (sharcEnabled) {
        const sharcPass = encoder.beginComputePass();
        sharcPass.setPipeline(sharcResolvePipeline);
        sharcPass.setBindGroup(0, bg2Resolve);
        sharcPass.dispatchWorkgroups(Math.ceil(SHARC_CAPACITY / 256));
        sharcPass.end();
      }

      if (denoiseMode.startsWith('oidn') && oidn) {
        // === Temporal BEFORE OIDN: stabilizes input → stable OIDN output ===
        // 1. Run temporal on noisy input → partially converged in hdrTex/specHdrTex
        // 2. Feed temporal output to OIDN → clean + stable result
        // 3. No post-OIDN blend needed (temporal already handled stability)

        // Step 1: Temporal accumulation (same as 'full' mode)
        const pbPass = encoder.beginComputePass();
        pbPass.setPipeline(preblurSMPipeline);
        pbPass.setBindGroup(0, preblurBG);
        pbPass.dispatchWorkgroups(Math.ceil(width/16), Math.ceil(height/16));
        pbPass.end();

        const tmpPass = encoder.beginComputePass();
        tmpPass.setPipeline(tmpPipeline);
        tmpPass.setBindGroup(0, historyFrame === 0 ? tmpBG_A : tmpBG_B);
        tmpPass.dispatchWorkgroups(Math.ceil(width/16), Math.ceil(height/16));
        tmpPass.end();
        historyFrame = 1 - historyFrame;

        // Step 2: OIDN reads temporal output (stable, partially converged)
        // Run every frame when moving, skip when still (temporal already converged)
        const runOidn = framesStill < 4 || (framesStill < 32 && frameIndex % 4 === 0) || framesStill === 4;
        if (runOidn) {
          oidn.encode(encoder, oidn.inputBG_temporal, oidn.outputBG);
          // Copy OIDN result to history for composite to read
          const blendAlpha = 1.0; // 100% OIDN (no blend, just store)
          const bcam = new Float32Array(40);
          const bcamU = new Uint32Array(bcam.buffer);
          bcamU[0] = width; bcamU[1] = height; bcamU[2] = Math.round(blendAlpha * 1000); bcamU[3] = 0;
          bcam[4] = right[0]; bcam[5] = right[1]; bcam[6] = right[2]; bcam[7] = 0;
          bcam[8] = up[0]; bcam[9] = up[1]; bcam[10] = up[2]; bcam[11] = 0;
          bcam[12] = forward[0]; bcam[13] = forward[1]; bcam[14] = forward[2]; bcam[15] = 0;
          bcam[16] = camera.pos[0]; bcam[17] = camera.pos[1]; bcam[18] = camera.pos[2]; bcam[19] = 0;
          bcam[20] = prevCam.right[0]; bcam[21] = prevCam.right[1]; bcam[22] = prevCam.right[2]; bcam[23] = 0;
          bcam[24] = prevCam.up[0]; bcam[25] = prevCam.up[1]; bcam[26] = prevCam.up[2]; bcam[27] = 0;
          bcam[28] = prevCam.fwd[0]; bcam[29] = prevCam.fwd[1]; bcam[30] = prevCam.fwd[2]; bcam[31] = 0;
          bcam[32] = prevCam.pos[0]; bcam[33] = prevCam.pos[1]; bcam[34] = prevCam.pos[2]; bcam[35] = 0;
          bcam[36] = fovFactor; bcam[37] = aspect; bcam[38] = 0; bcam[39] = 0;
          device.queue.writeBuffer(oidn.blendCamBuf, 0, bcam);
          const blendPass = encoder.beginComputePass();
          blendPass.setPipeline(oidn.blendPipeline);
          blendPass.setBindGroup(0, oidn.blendFrame === 0 ? oidn.blendBG_A : oidn.blendBG_B);
          blendPass.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16));
          blendPass.end();
          oidn.blendFrame = 1 - oidn.blendFrame;
        }
        // When OIDN skipped: composite reads last OIDN result from history (still clean)
      } else if (denoiseMode === 'reblur') {
        // === ReBLUR pipeline: 6 passes ===
        const wg = [Math.ceil(width/16), Math.ceil(height/16)];
        const histBG = historyFrame === 0 ? reblurHistBG_A : reblurHistBG_B;

        // Fill ReblurParams: 5 mat4x4 + 4 vec4 + 3 vec4 rotators + 6 vec2 + ~25 scalars
        // WGSL layout (byte offsets):
        //   0-63:   world_to_clip (mat4x4f)
        //  64-127:  view_to_world
        // 128-191:  world_to_view_prev
        // 192-255:  world_to_clip_prev
        // 256-319:  world_prev_to_world (identity for now)
        // 320-335:  frustum (vec4f)
        // 336-351:  frustum_prev
        // 352-367:  camera_delta
        // 368-383:  hit_dist_params
        // 384-399:  rotator_pre
        // 400-415:  rotator_blur
        // 416-431:  rotator_post
        // 432-439:  rect_size (vec2f)
        // 440-447:  rect_size_inv
        // 448-455:  rect_size_prev
        // 456-463:  resource_size_inv_prev
        // 464-471:  jitter
        // 472-475:  padding (align to 4)
        // 476+:     scalars (f32 each, u32 for frame_index)
        const rp = new Float32Array(160); // 640 bytes

        // Build view matrix (column-major) — simplified using our camera vectors
        // view_to_world = [right | up | -forward | pos] columns
        const vtw = rp.subarray(16, 32); // offset 64 = float[16]
        vtw[0]=right[0]; vtw[1]=right[1]; vtw[2]=right[2]; vtw[3]=0;
        vtw[4]=up[0]; vtw[5]=up[1]; vtw[6]=up[2]; vtw[7]=0;
        vtw[8]=-forward[0]; vtw[9]=-forward[1]; vtw[10]=-forward[2]; vtw[11]=0;
        vtw[12]=camera.pos[0]; vtw[13]=camera.pos[1]; vtw[14]=camera.pos[2]; vtw[15]=1;

        // frustum params (offset 320 = float[80]): reconstructs view pos from UV
        // Xv = (uv.x * frustum.x + frustum.z, uv.y * frustum.y + frustum.w, 1) * viewZ
        // For perspective: frustum = (2*tan(fovH/2), 2*tan(fovV/2), -tan(fovH/2), -tan(fovV/2))
        const tanH = aspect * fovFactor;
        const tanV = fovFactor;
        rp[80] = 2*tanH; rp[81] = 2*tanV; rp[82] = -tanH; rp[83] = -tanV; // frustum
        rp[84] = 2*tanH; rp[85] = 2*tanV; rp[86] = -tanH; rp[87] = -tanV; // frustum_prev (same for now)

        // world_to_view_prev (offset 128 = float[32]): column-major
        // viewPrev = [prevRight | prevUp | -prevFwd | prevPos] inverse
        {
          const pr = prevCam.right, pu = prevCam.up, pf = prevCam.fwd, pp = prevCam.pos;
          const wtvp = rp.subarray(32, 48);
          wtvp[0]=pr[0]; wtvp[1]=pu[0]; wtvp[2]=-pf[0]; wtvp[3]=0;
          wtvp[4]=pr[1]; wtvp[5]=pu[1]; wtvp[6]=-pf[1]; wtvp[7]=0;
          wtvp[8]=pr[2]; wtvp[9]=pu[2]; wtvp[10]=-pf[2]; wtvp[11]=0;
          wtvp[12]=-(pr[0]*pp[0]+pr[1]*pp[1]+pr[2]*pp[2]);
          wtvp[13]=-(pu[0]*pp[0]+pu[1]*pp[1]+pu[2]*pp[2]);
          wtvp[14]=(pf[0]*pp[0]+pf[1]*pp[1]+pf[2]*pp[2]);
          wtvp[15]=1;
        }

        // world_to_clip_prev (offset 192 = float[48]): proj * viewPrev
        // Perspective proj: similar to our buildViewProj
        {
          const near = 0.1, far = 1000.0;
          const proj = new Float32Array(16);
          proj[0] = 1.0 / (aspect * fovFactor); proj[5] = -1.0 / fovFactor;
          proj[10] = -far / (far - near); proj[11] = -1;
          proj[14] = -(far * near) / (far - near);
          // Multiply proj * viewPrev
          const vp = rp.subarray(32, 48); // world_to_view_prev
          const wcp = rp.subarray(48, 64); // world_to_clip_prev
          for (let c = 0; c < 4; c++)
            for (let r = 0; r < 4; r++)
              wcp[c*4+r] = proj[r]*vp[c*4] + proj[4+r]*vp[c*4+1] + proj[8+r]*vp[c*4+2] + proj[12+r]*vp[c*4+3];
        }

        // camera_delta (offset 352 = float[88])
        rp[88] = camera.pos[0]-prevCam.pos[0]; rp[89] = camera.pos[1]-prevCam.pos[1];
        rp[90] = camera.pos[2]-prevCam.pos[2]; rp[91] = 0;

        // hit_dist_params (offset 368 = float[92]): (A=3, B=0.1, C=1, 0)
        rp[92] = 3.0; rp[93] = 0.1; rp[94] = 1.0; rp[95] = 0;

        // Rotators (offset 384,400,416 = float[96,100,104])
        const angle = (frameIndex % 256) * (6.2832 / 256.0);
        const rc = Math.cos(angle), rs = Math.sin(angle);
        rp[96] = rc; rp[97] = rs; rp[98] = -rs; rp[99] = rc;
        rp[100] = rc; rp[101] = rs; rp[102] = -rs; rp[103] = rc;
        rp[104] = rc; rp[105] = rs; rp[106] = -rs; rp[107] = rc;

        // rect_size (offset 432 = float[108])
        rp[108] = width; rp[109] = height;
        rp[110] = 1.0/width; rp[111] = 1.0/height;
        rp[112] = width; rp[113] = height;
        rp[114] = 1.0/width; rp[115] = 1.0/height;
        // jitter (offset 464 = float[116])
        rp[116] = 0; rp[117] = 0;

        // Scalars (offset 472 = float[118], but we need padding for align)
        // After vec2f jitter (8 bytes), next f32 starts at offset 472
        // But the struct has f32 fields starting at disocclusion_threshold
        // With WGSL alignment: after vec2f, f32 aligns to 4 → offset 472
        let si = 118; // float index for offset 472
        rp[si++] = 0.015;  // disocclusion_threshold
        rp[si++] = 2.0;    // plane_dist_sensitivity
        rp[si++] = 0.5;    // min_blur_radius
        rp[si++] = 20.0;   // max_blur_radius
        rp[si++] = 30.0;   // diff_prepass_blur_radius
        rp[si++] = 30.0;   // spec_prepass_blur_radius
        rp[si++] = 63.0;   // max_accumulated_frame_num
        rp[si++] = 8.0;    // max_fast_accumulated_frame_num
        rp[si++] = 0.5;    // lobe_angle_fraction
        rp[si++] = 0.5;    // roughness_fraction
        rp[si++] = 7.0;    // history_fix_frame_num
        rp[si++] = 14.0;   // history_fix_stride
        rp[si++] = 1.0;    // stabilization_strength
        rp[si++] = 1.0;    // anti_firefly
        rp[si++] = 1.0;    // antilag_power
        rp[si++] = 1.0;    // antilag_threshold
        rp[si++] = 1.0;    // framerate_scale
        rp[si++] = 10000.0;// denoising_range
        rp[si++] = Math.min(width,height) * fovFactor / width; // min_rect_dim_mul_unproject
        rp[si++] = fovFactor; // unproject
        rp[si++] = 0.0;    // ortho_mode
        new Uint32Array(rp.buffer)[si] = frameIndex; si++; // frame_index (u32)
        rp[si++] = 1.0;    // view_z_scale
        rp[si++] = 0.2;    // min_hit_dist_weight
        rp[si++] = 1.0;    // firefly_suppressor_min_relative_scale
        rp[si++] = 1.0;    // fast_history_clamping_sigma_scale
        device.queue.writeBuffer(reblurParamBuf, 0, rp);

        // DEBUG: ONLY temporal — everything else skipped to find bloom source
        // PrePass SKIPPED: noisy goes directly to temporal via reblurBG_temporal
        // which reads from pingTex. Need to copy noisy→ping first.
        // Actually reblurBG_prepass: noisy→ping, so run it as passthrough:
        // NO — the prepass does spatial filtering. Need a raw copy.
        // Simplest: change reblurBG_temporal to read from noisy directly.
        // For now: just run temporal with the prepass BG textures swapped.
        // Actually the temporal BG reads pingTex. PrePass writes pingTex from noisy.
        // If we skip PrePass, pingTex has stale data. We need to copy noisy→ping.
        // Easiest hack: run PrePass but it will filter. Let's just accept that for now
        // and skip everything AFTER temporal.

        // PrePass: noisy → ping (keeps spatial filter for now)
        { const p = encoder.beginComputePass(); p.setPipeline(reblurPrepassPipeline); p.setBindGroup(0, reblurBG_prepass); p.dispatchWorkgroups(...wg); p.end(); }
        // Temporal: ping + history → hdr (ONLY pass that matters)
        { const p = encoder.beginComputePass(); p.setPipeline(reblurTemporalPipeline); p.setBindGroup(0, reblurBG_temporal); p.setBindGroup(1, histBG); p.dispatchWorkgroups(...wg); p.end(); }
        // ALL other passes SKIPPED
        // Stabilize: SKIPPED (testing bloom source)
        // Copy temporal output (hdrTex) to history for next frame
        { const p = encoder.beginComputePass(); p.setPipeline(copyToHistoryPipeline); p.setBindGroup(0, historyFrame === 0 ? reblurBG_copyHist_A : reblurBG_copyHist_B); p.dispatchWorkgroups(...wg); p.end(); }
        historyFrame = 1 - historyFrame;

      } else if (denoiseMode === 'full') {
        // Pre-blur: lightweight 3×3 bilateral → stabilizes temporal AABB + removes fireflies
        const pbPass = encoder.beginComputePass();
        pbPass.setPipeline(preblurSMPipeline);
        pbPass.setBindGroup(0, preblurBG);
        pbPass.dispatchWorkgroups(Math.ceil(width/16), Math.ceil(height/16));
        pbPass.end();

        // Temporal reprojection: reads pre-blurred ping → writes hdrTex
        const tmpPass = encoder.beginComputePass();
        tmpPass.setPipeline(tmpPipeline);
        tmpPass.setBindGroup(0, historyFrame === 0 ? tmpBG_A : tmpBG_B);
        tmpPass.dispatchWorkgroups(Math.ceil(width/16), Math.ceil(height/16));
        tmpPass.end();
        historyFrame = 1 - historyFrame;
      }

      if (denoiseMode !== 'off' && denoiseMode !== 'oidn' && denoiseMode !== 'reblur') {
        // Update denoise params with framesStill for motion-adaptive filtering
        for (let di = 0; di < denoisePasses; di++) {
          device.queue.writeBuffer(dnParamBufs[di], 0, new Float32Array([width, height, dnSteps[di], framesStill]));
        }
        // Variance-guided à-trous (GPU profile controls pass count: 3-5)
        for (let di = 0; di < denoisePasses; di++) {
          const bg = (di === 0 && denoiseMode !== 'full') ? dnBG_noisy_first : dnBGs[di];
          const dp = encoder.beginComputePass();
          dp.setPipeline(di === 0 ? dnAtrousSMPipeline : dnAtrousPipeline);
          dp.setBindGroup(0, bg);
          dp.dispatchWorkgroups(Math.ceil(width/16), Math.ceil(height/16));
          dp.end();
        }

        // Temporal Stabilization (ReBLUR anti-flicker): clamps luminance to neighborhood
        // Uses copyToHistBG which reads denoised final and writes to history
        const stabPass = encoder.beginComputePass();
        stabPass.setPipeline(copyToHistoryPipeline);
        stabPass.setBindGroup(0, historyFrame === 0 ? copyToHistBG_B : copyToHistBG_A);
        stabPass.dispatchWorkgroups(Math.ceil(width/16), Math.ceil(height/16));
        stabPass.end();
      }

      // Composite (tonemap → LDR)
      // Write composite params: resolution + color controls.
      const cp = new ArrayBuffer(96);
      const cpf = new Float32Array(cp);
      const cpu = new Uint32Array(cp);
      cpf[0] = width; cpf[1] = height;
      cpf[2] = (denoiseMode.startsWith('oidn') && oidn) ? -1.0 : 0.0; // negative = OIDN mode (skip remodulation)
      cpf[3] = 0;
      cpu[4] = settings.tonemapMode; cpf[5] = settings.exposure * autoExposure;
      cpf[6] = settings.saturation; cpf[7] = settings.contrast;
      // Legacy camera fields remain populated to preserve uniform layout.
      cpf[8] = camera.pos[0]; cpf[9] = camera.pos[1]; cpf[10] = camera.pos[2]; cpf[11] = 0;
      cpf[12] = forward[0]; cpf[13] = forward[1]; cpf[14] = forward[2]; cpf[15] = fovFactor;
      cpf[16] = right[0]; cpf[17] = right[1]; cpf[18] = right[2]; cpf[19] = aspect;
      cpf[20] = up[0]; cpf[21] = up[1]; cpf[22] = up[2]; cpf[23] = 0;
      device.queue.writeBuffer(dnCompParamBuf, 0, cp);

      const compPass = encoder.beginComputePass();
      compPass.setPipeline(dnCompPipeline);
      const compBG = (denoiseMode.startsWith('oidn') && oidn)
                   ? (oidn.blendFrame === 0 ? dnBG_comp_oidn_A : dnBG_comp_oidn_B)
                   : denoiseMode === 'reblur' ? dnBG_comp_reblur
                   : denoiseMode !== 'off' ? dnBG_comp : dnBG_comp_noisy;
      compPass.setBindGroup(0, compBG);
      compPass.dispatchWorkgroups(Math.ceil(width/16), Math.ceil(height/16));
      compPass.end();

      if (fsrMode !== 'dlaa') {
        // Pass 2: FSR EASU — Lanczos upscale (internal → display)
        const easuPass = encoder.beginComputePass();
        easuPass.setPipeline(easuPipeline);
        easuPass.setBindGroup(0, easuBG);
        easuPass.dispatchWorkgroups(Math.ceil(displayWidth/16), Math.ceil(displayHeight/16));
        easuPass.end();

        // Pass 3: FSR RCAS — Contrast-adaptive sharpening
        const rcasPass = encoder.beginComputePass();
        rcasPass.setPipeline(rcasPipeline);
        rcasPass.setBindGroup(0, rcasBG);
        rcasPass.dispatchWorkgroups(Math.ceil(displayWidth/16), Math.ceil(displayHeight/16));
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

      // Auto-exposure: copy accumulator to staging for async readback (reuse buffer)
      const t_submit = performance.now();
      if (!exposureStagingBusy) {
        const copyEncoder = device.createCommandEncoder();
        copyEncoder.copyBufferToBuffer(exposureBuf, 0, exposureStagingBuf, 0, 16);
        device.queue.submit([encoder.finish(), copyEncoder.finish()]);
        exposureStagingBusy = true;

        exposureStagingBuf.mapAsync(GPUMapMode.READ).then(() => {
          const data = new Uint32Array(exposureStagingBuf.getMappedRange());
          const logLumSum = data[0];
          const pixelCount = data[1];
          exposureStagingBuf.unmap();
          exposureStagingBusy = false;

        if (pixelCount > 0) {
          const avgLogLum = logLumSum / (16.0 * pixelCount) - 20.0;
          const avgLum = Math.pow(2, avgLogLum);
          const keyValue = 0.18;
          const targetExposure = Math.min(Math.max(keyValue / Math.max(avgLum, 0.001), 0.1), 10.0);
          // EMA smoothing: fast response (20%) for large changes, smooth (5%) for small
          const expDiff = Math.abs(targetExposure - autoExposure) / Math.max(autoExposure, 0.01);
          const expSpeed = expDiff > 0.3 ? 0.3 : 0.08;
          autoExposure = autoExposure + (targetExposure - autoExposure) * expSpeed;
        }
        // Reset accumulators for next frame
        device.queue.writeBuffer(exposureBuf, 0, new Uint32Array([0, 0]));
      }).catch(() => { exposureStagingBusy = false; });
      } else {
        // Staging buffer busy — just submit without exposure readback this frame
        device.queue.submit([encoder.finish()]);
      }

      // GPU timing
      device.queue.onSubmittedWorkDone().then(() => {
        const t_gpu_done = performance.now();
        passTimers.total = t_gpu_done - t_frame_start;
        passTimers.gbuffer = t_gbuffer - t_frame_start;
        passTimers.shadow = t_shadow - t_gbuffer;
        passTimers.pathtrace = t_submit - t_shadow;
        timerLogCounter++;
        if (timerLogCounter % 120 === 0) {
          rlog(`TIMING: total=${passTimers.total.toFixed(1)}ms gbuf=${passTimers.gbuffer.toFixed(1)}ms shadow=${passTimers.shadow.toFixed(1)}ms encode=${passTimers.pathtrace.toFixed(1)}ms gpu=${(t_gpu_done - t_submit).toFixed(1)}ms`);
        }
      });

      if (frameIndex === 1) rlog('First frame OK');
    } catch(e) {
      rlog('FRAME_ERROR: ' + e.message);
    }

    drawGizmo();

    const tracePercent = framesStill > 30 ? 12 : framesStill > 10 ? 25 : 50;
    info.innerHTML =
      `<b>Ignis</b> | ${FSR_MODES[fsrMode].label} | ${denoiseMode}<br>` +
      `${width}x${height}\u2192${displayWidth}x${displayHeight} FPS:${fps}<br>` +
      `<span style="font-size:11px">frame:${passTimers.total.toFixed(1)}ms gpu:${(passTimers.total - passTimers.gbuffer - passTimers.shadow).toFixed(1)}ms</span><br>` +
      `<span style="font-size:11px">ESC: options | TAB: debug</span>`;

    requestAnimationFrame(frame);
  }

  info.textContent = 'Ready. Starting render...';
  requestAnimationFrame(frame);
}

init().catch(err => { rlog('FATAL:', err.message, err.stack); showError(`Error: ${err.message}`); });
