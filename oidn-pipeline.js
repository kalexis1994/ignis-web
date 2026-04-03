// OIDN UNet Pipeline — orchestrates the full neural denoiser forward pass
// 16 conv layers + 4 maxpool + 4 upsample + input/output assembly = 26 dispatches

import { loadOIDNWeights, logWeightSummary } from './oidn-weights.js';

// UNet layer definitions extracted from rt_hdr_alb_nrm.tza
// Format: [name, c_in, c_out, relu]
const ENCODER_CONVS = [
  ['enc_conv0',  9,  32, true],
  ['enc_conv1',  32, 32, true],  // → skip_0 (full res)
  // maxpool
  ['enc_conv2',  32, 48, true],  // → skip_1 (1/2)
  // maxpool
  ['enc_conv3',  48, 64, true],  // → skip_2 (1/4)
  // maxpool
  ['enc_conv4',  64, 80, true],  // → skip_3 (1/8)
  // maxpool
  ['enc_conv5a', 80, 96, true],  // bottleneck
  ['enc_conv5b', 96, 96, true],
];

const DECODER_CONVS = [
  // upsample → concat with skip_3(80ch) → 96+80=176... wait, actual TZA shows dec_conv4a has c_in=160
  // Re-check: enc_conv4 outputs 80ch but skip_3 stores enc_conv3 output=64ch
  // Actually let me re-derive from the TZA shapes:
  // dec_conv4a.weight: [112, 160, 3, 3] → c_in=160 (96 from bottleneck upsample + 64 from skip after enc_conv3... no)
  // Wait: 96 (from enc_conv5b) upsampled + 64 (from enc_conv4's INPUT, which is enc_conv3 output=64)
  // Actually enc_conv4 takes 64→80. Skip_3 saves BEFORE enc_conv4, at enc_conv3 output=64? No...
  // Let me re-derive: encoder saves skip AFTER the conv, before pool:
  //   enc_conv1(32) → skip_0(32)  → pool
  //   enc_conv2(48) → skip_1(48)  → pool
  //   enc_conv3(64) → skip_2(64)  → pool
  //   enc_conv4(80) → skip_3(80)  → pool
  // decoder: upsample(96) + skip_3(80) = 176? But TZA says 160...
  // Hmm, TZA: dec_conv4a [112, 160, 3, 3] → 160 input channels
  //          dec_conv3a [96, 160, 3, 3] → 160 input channels
  //          dec_conv2a [64, 128, 3, 3] → 128 input channels
  //          dec_conv1a [64, 73, 3, 3] → 73 input channels (9 input + 64)
  // So: 160 = 96 + 64 (skip from enc_conv3=64, NOT enc_conv4=80)
  //     160 = 112 + 48 (skip from enc_conv2=48)... 112+48=160 yes!
  //     128 = 64 + 64?? no, dec_conv3b outputs 96... let me re-trace
  //
  // Actually the skip connections save the POOLED output, not the conv output.
  // No— UNet saves BEFORE pooling. So:
  //   Level 0 (full): enc_conv1 → 32ch → skip_0 → pool → 32ch at 1/2
  //   Level 1 (1/2):  enc_conv2 → 48ch → skip_1 → pool → 48ch at 1/4
  //   Level 2 (1/4):  enc_conv3 → 64ch → skip_2 → pool → 64ch at 1/8
  //   Level 3 (1/8):  enc_conv4 → 80ch → skip_3 → pool → 80ch at 1/16
  //   Bottleneck:     enc_conv5a → 96ch, enc_conv5b → 96ch
  //
  //   Decoder:
  //   upsample(96 at 1/16 → 96 at 1/8) + skip_3(80) = 176 → but TZA says 160
  //
  // Something doesn't match. Let me just trust the TZA file shapes directly:
  //   dec_conv4a: [112, 160, 3, 3] → concat input is 160ch
  //   dec_conv4b: [112, 112, 3, 3]
  //   dec_conv3a: [96, 160, 3, 3]  → concat input is 160ch = 112 + 48
  //   dec_conv3b: [96, 96, 3, 3]
  //   dec_conv2a: [64, 128, 3, 3]  → concat input is 128ch = 96 + 32
  //   dec_conv2b: [64, 64, 3, 3]
  //   dec_conv1a: [64, 73, 3, 3]   → concat input is 73ch = 64 + 9 (input image!)
  //   dec_conv1b: [32, 64, 3, 3]
  //   dec_conv0:  [3, 32, 3, 3]    → final output
  //
  // So: dec_conv4a concat = 96 + 64 = 160 → skip is enc_conv3 output (64ch), NOT enc_conv4 (80ch)
  // This means the skip connections are OFFSET by one level!
  // Actually no. Let me reconsider. Maybe the architecture is:
  //   enc_conv0(9→32) enc_conv1(32→32) → pool  (skip saves 32ch)
  //   enc_conv2(32→48) → pool                   (skip saves 48ch)
  //   enc_conv3(48→64) → pool                   (skip saves 64ch)
  //   enc_conv4(64→80) → pool
  //   enc_conv5a(80→96) enc_conv5b(96→96)
  //
  // dec_conv4a input = upsample(96) + skip_enc_conv3(64) = 160 ✓
  // dec_conv3a input = upsample(112) + skip_enc_conv2(48) = 160 ✓
  // dec_conv2a input = upsample(96) + skip_enc_conv1(32) = 128 ✓
  // dec_conv1a input = upsample(64) + input(9) = 73 ✓
  //
  // So skips come from: enc_conv3(64), enc_conv2(48), enc_conv1(32), and original input(9)
  // enc_conv4's output(80) is NOT saved as a skip — it just gets pooled and fed to bottleneck.
];

// Derived from TZA shapes: [name, c_in_total, c_out, relu, upsample_ch, skip_ch]
// upsample_ch = channels from upsampled decoder, skip_ch = channels from encoder skip
const DECODER_LAYERS = [
  // After bottleneck: upsample 96ch to 1/8
  { name: 'dec_conv4a', c_in: 160, c_out: 112, relu: true, up_ch: 96, skip_ch: 64 }, // skip from enc_conv3
  { name: 'dec_conv4b', c_in: 112, c_out: 112, relu: true },
  // upsample 112ch to 1/4
  { name: 'dec_conv3a', c_in: 160, c_out: 96,  relu: true, up_ch: 112, skip_ch: 48 }, // skip from enc_conv2
  { name: 'dec_conv3b', c_in: 96,  c_out: 96,  relu: true },
  // upsample 96ch to 1/2
  { name: 'dec_conv2a', c_in: 128, c_out: 64,  relu: true, up_ch: 96, skip_ch: 32 }, // skip from enc_conv1
  { name: 'dec_conv2b', c_in: 64,  c_out: 64,  relu: true },
  // upsample 64ch to full
  { name: 'dec_conv1a', c_in: 73,  c_out: 64,  relu: true, up_ch: 64, skip_ch: 9 }, // skip from input(9ch)
  { name: 'dec_conv1b', c_in: 64,  c_out: 32,  relu: true },
  { name: 'dec_conv0',  c_in: 32,  c_out: 3,   relu: false }, // linear output
];

/**
 * Create the full OIDN UNet pipeline.
 */
export async function createOIDNPipeline(device, weightsUrl, width, height, hasF16, logFn) {
  const log = logFn || console.log;

  // Load weights
  const wt = await loadOIDNWeights(weightsUrl, device);
  logWeightSummary(wt, log);

  // Pad dimensions to multiple of 16 (4 pooling levels)
  const padW = Math.ceil(width / 16) * 16;
  const padH = Math.ceil(height / 16) * 16;
  log(`OIDN: internal ${width}x${height} → padded ${padW}x${padH}`);

  // Resolution at each level
  const levels = [];
  let lw = padW, lh = padH;
  for (let i = 0; i < 5; i++) {
    levels.push({ w: lw, h: lh });
    lw = Math.ceil(lw / 2);
    lh = Math.ceil(lh / 2);
  }

  // --- Allocate feature map buffers ---
  function bufSize(channels, level) {
    return levels[level].w * levels[level].h * channels * 2; // f16 = 2 bytes
  }

  const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;

  // Skip buffers (persist encoder → decoder) — saved AFTER pooling (at decoder resolution)
  const skipInput = device.createBuffer({ size: bufSize(9, 0), usage: STORAGE });   // original 9ch input (L0)
  const skip1 = device.createBuffer({ size: bufSize(32, 1), usage: STORAGE });      // pooled enc_conv1 (L1)
  const skip2 = device.createBuffer({ size: bufSize(32, 2), usage: STORAGE });      // pooled enc_conv2 (L2)
  const skip3 = device.createBuffer({ size: bufSize(32, 3), usage: STORAGE });      // pooled enc_conv3 (L3)

  // Working buffers (ping-pong, sized for largest need)
  const maxBufSize = Math.max(
    bufSize(96, 0),   // largest: bottleneck upsampled to full res (won't happen, but safe)
    bufSize(112, 3),  // dec_conv4a output at 1/8
    bufSize(96, 0),   // various full-res intermediates
    bufSize(32, 0),   // enc_conv0/1 output
    bufSize(48, 1),   // enc_conv2 output
    bufSize(64, 2),   // enc_conv3 output
    bufSize(80, 3),   // enc_conv4 output
    bufSize(96, 4),   // bottleneck
  );
  // Two large working buffers for ping-pong between layers
  const workA = device.createBuffer({ size: maxBufSize, usage: STORAGE });
  const workB = device.createBuffer({ size: maxBufSize, usage: STORAGE });

  // Output buffer (3ch at full res)
  const outputBuf = device.createBuffer({ size: bufSize(3, 0), usage: STORAGE });

  // --- Load shaders ---
  const v = Date.now();
  const smOpts = { strictMath: false };
  const convModule = device.createShaderModule({
    code: (hasF16 ? 'enable f16;\n' : '') + await fetch(`oidn-conv2d.wgsl?v=${v}`).then(r => r.text()).then(c =>
      hasF16 ? c.replace('enable f16;\n', '') : c.replace(/\bvec(\d)h\b/g, 'vec$1f').replace(/\bf16\(/g, 'f32(').replace(/(\d+\.?\d*)h\b/g, '$1').replace(/:\s*f16\b/g, ': f32')
    ),
    ...smOpts
  });
  const opsCode = await fetch(`oidn-ops.wgsl?v=${v}`).then(r => r.text());
  const opsModule = device.createShaderModule({
    code: hasF16 ? 'enable f16;\n' + opsCode.replace('enable f16;\n', '') : opsCode.replace(/\bvec(\d)h\b/g, 'vec$1f').replace(/\bf16\(/g, 'f32(').replace(/(\d+\.?\d*)h\b/g, '$1').replace(/:\s*f16\b/g, ': f32'),
    ...smOpts
  });

  // --- Create pipeline layouts ---
  // Conv2D layout: params + weights + buf_a + buf_out + buf_b
  const convBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
  ]});
  const convPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [convBGL] }),
    compute: { module: convModule, entryPoint: 'conv2d_3x3' },
  });

  // Pool/Upsample layout: params + buf_in + buf_out
  const opsBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
  ]});
  const poolPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [opsBGL] }),
    compute: { module: opsModule, entryPoint: 'maxpool2x2' },
  });
  const upsamplePipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [opsBGL] }),
    compute: { module: opsModule, entryPoint: 'upsample2x' },
  });

  // I/O layout: params + buffers + diffuse_tex + albedo_tex + normal_tex + out_tex + specular_tex
  const ioBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
    { binding: 6, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
    { binding: 7, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
  ]});
  const inputPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [ioBGL] }),
    compute: { module: opsModule, entryPoint: 'input_assembly' },
  });
  const outputPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [ioBGL] }),
    compute: { module: opsModule, entryPoint: 'output_extraction' },
  });

  // --- Helper: create conv uniform buffer ---
  function makeConvParams(w, h, c_in, c_out, layerName, relu, split_ch) {
    const t_w = wt.tensors.get(layerName + '.weight');
    const t_b = wt.tensors.get(layerName + '.bias');
    if (!t_w || !t_b) throw new Error(`Missing weights for ${layerName}`);
    const data = new Uint32Array([
      w, h, c_in, c_out,
      t_w.offset / 2,  // f16 element offset (byte offset / 2)
      t_b.offset / 2,
      relu ? 1 : 0,
      split_ch || 0,
    ]);
    const buf = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buf, 0, data);
    return buf;
  }

  function makeOpsParams(inW, inH, outW, outH, channels) {
    const data = new Uint32Array([inW, inH, outW, outH, channels, 0, 0, 0]);
    const buf = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buf, 0, data);
    return buf;
  }

  // Dummy buffer for unused concat input
  const dummyBuf = device.createBuffer({ size: 16, usage: STORAGE });

  // --- Build dispatch list ---
  // Each entry: { pipeline, bindGroup, dispatch: [x, y, z] }
  const dispatches = [];

  function addConv(layerName, c_in, c_out, relu, inBuf, outBuf, level, splitCh, skipBuf) {
    const { w, h } = levels[level];
    const params = makeConvParams(w, h, c_in, c_out, layerName, relu, splitCh || 0);
    const bg = device.createBindGroup({ layout: convBGL, entries: [
      { binding: 0, resource: { buffer: params } },
      { binding: 1, resource: { buffer: wt.buffer } },
      { binding: 2, resource: { buffer: inBuf } },
      { binding: 3, resource: { buffer: outBuf } },
      { binding: 4, resource: { buffer: skipBuf || dummyBuf } },
    ]});
    dispatches.push({
      pipeline: convPipeline, bindGroup: bg,
      dispatch: [Math.ceil(w / 16), Math.ceil(h / 16), c_out],
      label: layerName,
    });
  }

  function addPool(inBuf, outBuf, channels, level) {
    const { w: inW, h: inH } = levels[level];
    const { w: outW, h: outH } = levels[level + 1];
    const params = makeOpsParams(inW, inH, outW, outH, channels);
    const bg = device.createBindGroup({ layout: opsBGL, entries: [
      { binding: 0, resource: { buffer: params } },
      { binding: 1, resource: { buffer: inBuf } },
      { binding: 2, resource: { buffer: outBuf } },
    ]});
    dispatches.push({
      pipeline: poolPipeline, bindGroup: bg,
      dispatch: [Math.ceil(outW / 16), Math.ceil(outH / 16), channels],
      label: `maxpool_L${level}→${level + 1}`,
    });
  }

  function addUpsample(inBuf, outBuf, channels, fromLevel) {
    const { w: inW, h: inH } = levels[fromLevel];
    const { w: outW, h: outH } = levels[fromLevel - 1];
    const params = makeOpsParams(inW, inH, outW, outH, channels);
    const bg = device.createBindGroup({ layout: opsBGL, entries: [
      { binding: 0, resource: { buffer: params } },
      { binding: 1, resource: { buffer: inBuf } },
      { binding: 2, resource: { buffer: outBuf } },
    ]});
    dispatches.push({
      pipeline: upsamplePipeline, bindGroup: bg,
      dispatch: [Math.ceil(outW / 16), Math.ceil(outH / 16), channels],
      label: `upsample_L${fromLevel}→${fromLevel - 1}`,
    });
  }

  // --- ENCODER (small model: all 32ch) ---
  addConv('enc_conv0', 9, 32, true, skipInput, workA, 0);
  addConv('enc_conv1', 32, 32, true, workA, workB, 0);
  addPool(workB, skip1, 32, 0);  // skip1: 32ch at L1
  addConv('enc_conv2', 32, 32, true, skip1, workA, 1);
  addPool(workA, skip2, 32, 1);  // skip2: 32ch at L2
  addConv('enc_conv3', 32, 32, true, skip2, workB, 2);
  addPool(workB, skip3, 32, 2);  // skip3: 32ch at L3
  addConv('enc_conv4', 32, 32, true, skip3, workA, 3);
  addPool(workA, workB, 32, 3);
  addConv('enc_conv5a', 32, 32, true, workB, workA, 4);
  addConv('enc_conv5b', 32, 32, true, workA, workB, 4);

  // --- DECODER (small model: all 64ch, skips are 32ch) ---
  addUpsample(workB, workA, 32, 4);                              // 32ch L4→L3
  addConv('dec_conv4a', 64, 64, true, workA, workB, 3, 32, skip3); // 32+32=64
  addConv('dec_conv4b', 64, 64, true, workB, workA, 3);
  addUpsample(workA, workB, 64, 3);                              // 64ch L3→L2
  addConv('dec_conv3a', 96, 64, true, workB, workA, 2, 64, skip2); // 64+32=96
  addConv('dec_conv3b', 64, 64, true, workA, workB, 2);
  addUpsample(workB, workA, 64, 2);                              // 64ch L2→L1
  addConv('dec_conv2a', 96, 64, true, workA, workB, 1, 64, skip1); // 64+32=96
  addConv('dec_conv2b', 64, 32, true, workB, workA, 1);
  addUpsample(workA, workB, 32, 1);                              // 32ch L1→L0
  addConv('dec_conv1a', 41, 32, true, workB, workA, 0, 32, skipInput); // 32+9=41
  addConv('dec_conv1b', 32, 32, true, workA, workB, 0);
  addConv('dec_conv0', 32, 3, false, workB, outputBuf, 0);

  log(`OIDN pipeline: ${dispatches.length} dispatches, ${levels.map(l => l.w + 'x' + l.h).join(' → ')}`);

  return {
    padW, padH, width, height, levels,
    skipInput, outputBuf, workA, workB,
    dispatches,
    // Pipelines for I/O (need textures, created at integration time)
    inputPipeline, outputPipeline, ioBGL,

    /**
     * Encode all UNet dispatches into a command encoder.
     * Call createIOBindGroups first to set up texture bindings.
     */
    encode(encoder, inputBG, outputBG) {
      // Input assembly: textures → skipInput buffer
      const ioPass = encoder.beginComputePass();
      ioPass.setPipeline(inputPipeline);
      ioPass.setBindGroup(0, inputBG);
      ioPass.dispatchWorkgroups(Math.ceil(padW / 16), Math.ceil(padH / 16));
      ioPass.end();

      // UNet forward pass (all layers)
      for (const d of dispatches) {
        const pass = encoder.beginComputePass();
        pass.setPipeline(d.pipeline);
        pass.setBindGroup(0, d.bindGroup);
        pass.dispatchWorkgroups(d.dispatch[0], d.dispatch[1], d.dispatch[2]);
        pass.end();
      }

      // Output extraction: outputBuf → texture
      const outPass = encoder.beginComputePass();
      outPass.setPipeline(outputPipeline);
      outPass.setBindGroup(0, outputBG);
      outPass.dispatchWorkgroups(Math.ceil(padW / 16), Math.ceil(padH / 16));
      outPass.end();
    },
  };
}
