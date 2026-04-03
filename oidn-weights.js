// OIDN TZA Weight Loader — parses Intel Open Image Denoise .tza tensor archives
// and uploads all weights into a single GPUBuffer for use by the neural denoiser.
// Weights: Apache 2.0 license (Intel Corporation)

/**
 * Parse a .tza (Tensor Archive) file and upload weights to GPU.
 * @param {string} url - URL to the .tza weight file
 * @param {GPUDevice} device - WebGPU device
 * @returns {{ buffer: GPUBuffer, tensors: Map<string, {offset:number, shape:number[], dtype:string}>, totalBytes: number }}
 */
export async function loadOIDNWeights(url, device) {
  const arrayBuffer = await fetch(url).then(r => {
    if (!r.ok) throw new Error(`Failed to load OIDN weights: ${r.status} ${r.statusText}`);
    return r.arrayBuffer();
  });

  const tensors = parseTZA(arrayBuffer);

  // Calculate total size and pack all tensor data contiguously
  let totalBytes = 0;
  for (const t of tensors.values()) {
    // Align each tensor to 4 bytes for GPU buffer compatibility
    totalBytes = (totalBytes + 3) & ~3;
    t.gpuOffset = totalBytes;
    totalBytes += t.data.byteLength;
  }
  totalBytes = (totalBytes + 3) & ~3;

  // Pack into a single ArrayBuffer
  const packed = new Uint8Array(totalBytes);
  for (const t of tensors.values()) {
    packed.set(t.data, t.gpuOffset);
  }

  // Upload to GPU
  const buffer = device.createBuffer({
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, packed);

  // Build metadata map (drop raw data to free memory)
  const meta = new Map();
  for (const [name, t] of tensors) {
    let numElements = 1;
    for (const d of t.shape) numElements *= d;
    meta.set(name, {
      offset: t.gpuOffset,          // byte offset in the GPU buffer
      shape: t.shape,                // e.g. [32, 9, 3, 3] for weight, [32] for bias
      layout: t.layout,             // "oihw" or "x"
      dtype: t.dtypeChar,           // "h" (fp16) or "f" (fp32)
      bytesPerElement: t.bytesPerElement,
      numElements,
      byteSize: t.data.byteLength,
    });
  }

  return { buffer, tensors: meta, totalBytes };
}

/**
 * Parse TZA binary format.
 * Format: header(12 bytes) → tensor data blobs (64-byte aligned) → tensor table at tableOffset.
 */
function parseTZA(arrayBuffer) {
  const view = new DataView(arrayBuffer);
  const bytes = new Uint8Array(arrayBuffer);
  let pos = 0;

  const readU8  = () => { const v = view.getUint8(pos);  pos += 1; return v; };
  const readU16 = () => { const v = view.getUint16(pos, true); pos += 2; return v; };
  const readU32 = () => { const v = view.getUint32(pos, true); pos += 4; return v; };
  const readU64 = () => {
    // Read as two u32 (safe for offsets < 2^53)
    const lo = view.getUint32(pos, true);
    const hi = view.getUint32(pos + 4, true);
    pos += 8;
    return lo + hi * 0x100000000;
  };

  // --- Header (12 bytes) ---
  const magic = readU16();
  if (magic !== 0x41D7) throw new Error(`Invalid TZA magic: 0x${magic.toString(16)}`);

  const majorVersion = readU8();
  const minorVersion = readU8();
  if (majorVersion !== 2) throw new Error(`Unsupported TZA version: ${majorVersion}.${minorVersion}`);

  const tableOffset = readU64();

  // --- Tensor table (at tableOffset) ---
  pos = tableOffset;
  const numTensors = readU32();

  const tensors = new Map();
  const decoder = new TextDecoder();

  for (let i = 0; i < numTensors; i++) {
    // Name
    const nameLen = readU16();
    const name = decoder.decode(bytes.subarray(pos, pos + nameLen));
    pos += nameLen;

    // Dimensions
    const ndims = readU8();
    const shape = [];
    for (let j = 0; j < ndims; j++) shape.push(readU32());

    // Layout (ndims ASCII chars, e.g. "oihw" or "x")
    const layout = decoder.decode(bytes.subarray(pos, pos + ndims));
    pos += ndims;

    // Data type (1 ASCII char)
    const dtypeChar = String.fromCharCode(readU8());
    let bytesPerElement;
    switch (dtypeChar) {
      case 'f': bytesPerElement = 4; break;
      case 'h': bytesPerElement = 2; break;
      default: throw new Error(`Unknown TZA dtype: '${dtypeChar}' in tensor '${name}'`);
    }

    // Data offset (absolute position in file)
    const dataOffset = readU64();

    // Compute total byte size and slice raw data
    let numElements = 1;
    for (const d of shape) numElements *= d;
    const byteSize = numElements * bytesPerElement;
    const data = bytes.slice(dataOffset, dataOffset + byteSize);

    tensors.set(name, { name, shape, layout, dtypeChar, bytesPerElement, dataOffset, data });
  }

  return tensors;
}

/**
 * Log loaded weight summary for debugging.
 */
export function logWeightSummary(meta, logFn = console.log) {
  logFn(`OIDN weights: ${meta.tensors.size} tensors, ${(meta.totalBytes / 1024).toFixed(1)} KB`);
  for (const [name, t] of meta.tensors) {
    logFn(`  ${name}: [${t.shape.join(',')}] ${t.dtype} @ offset ${t.offset}`);
  }
}
