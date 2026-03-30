// GLTF Scene Loader + BVH Builder for WebGPU Path Tracer

const MAX_LEAF_SIZE = 8;
const MAT_FLAG_THIN_TRANSMISSION = 1;
const MAT_FLAG_DOUBLE_SIDED = 2;
const MAT_FLAG_UNLIT = 4;

// ============================================================
// Matrix utilities (column-major, GLTF convention)
// ============================================================
function mat4Identity() {
  return new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
}

function mat4Multiply(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++)
    for (let r = 0; r < 4; r++)
      o[c*4+r] = a[r]*b[c*4] + a[4+r]*b[c*4+1] + a[8+r]*b[c*4+2] + a[12+r]*b[c*4+3];
  return o;
}

function mat4FromTRS(t, q, s) {
  const [qx,qy,qz,qw] = q;
  const x2=qx+qx, y2=qy+qy, z2=qz+qz;
  const xx=qx*x2, xy=qx*y2, xz=qx*z2;
  const yy=qy*y2, yz=qy*z2, zz=qz*z2;
  const wx=qw*x2, wy=qw*y2, wz=qw*z2;
  return new Float32Array([
    (1-yy-zz)*s[0], (xy+wz)*s[0], (xz-wy)*s[0], 0,
    (xy-wz)*s[1], (1-xx-zz)*s[1], (yz+wx)*s[1], 0,
    (xz+wy)*s[2], (yz-wx)*s[2], (1-xx-yy)*s[2], 0,
    t[0], t[1], t[2], 1,
  ]);
}

function nodeLocalMatrix(node) {
  if (node.matrix) return new Float32Array(node.matrix);
  return mat4FromTRS(
    node.translation || [0,0,0],
    node.rotation || [0,0,0,1],
    node.scale || [1,1,1]
  );
}

// ============================================================
// GLTF data reading
// ============================================================
function readAccessor(gltf, bin, accIdx) {
  const acc = gltf.accessors[accIdx];
  const bv = gltf.bufferViews[acc.bufferView];
  const byteOff = (bv.byteOffset || 0) + (acc.byteOffset || 0);
  const compCount = {SCALAR:1, VEC2:2, VEC3:3, VEC4:4}[acc.type];
  const count = acc.count;

  if (acc.componentType === 5126) { // float32
    const stride = bv.byteStride || (4 * compCount);
    if (stride === 4 * compCount && (byteOff % 4) === 0) {
      return new Float32Array(bin, byteOff, count * compCount);
    }
    const out = new Float32Array(count * compCount);
    const dv = new DataView(bin);
    for (let i = 0; i < count; i++) {
      const base = byteOff + i * stride;
      for (let c = 0; c < compCount; c++)
        out[i * compCount + c] = dv.getFloat32(base + c * 4, true);
    }
    return out;
  }
  if (acc.componentType === 5125) { // uint32
    const stride = bv.byteStride || 4;
    if (stride === 4 && (byteOff % 4) === 0) {
      return new Uint32Array(bin, byteOff, count);
    }
    const out = new Uint32Array(count);
    const dv = new DataView(bin);
    for (let i = 0; i < count; i++)
      out[i] = dv.getUint32(byteOff + i * stride, true);
    return out;
  }
  if (acc.componentType === 5123) { // uint16
    const stride = bv.byteStride || 2;
    const out = new Uint32Array(count);
    const dv = new DataView(bin);
    for (let i = 0; i < count; i++)
      out[i] = dv.getUint16(byteOff + i * stride, true);
    return out;
  }
  throw new Error(`Unsupported component type: ${acc.componentType}`);
}

function textureBinding(gltf, texInfo) {
  if (!texInfo) return { index: -1, texCoord: 0 };
  const tex = gltf.textures?.[texInfo.index];
  return {
    index: tex?.source ?? -1,
    texCoord: texInfo.texCoord ?? 0,
  };
}

// ============================================================
// Material extraction (PBR metallic-roughness from GLTF)
// ============================================================
function extractMaterials(gltf) {
  const defaultMat = {
    albedo:[1,1,1], type:0, emission:[0,0,0], roughness:1,
    metallic:1, baseTex:-1, mrTex:-1, normalTex:-1,
    alphaMode:0, alphaCutoff:0.5, ior:1.5,
    transmission:0, transmissionTex:-1, thickness:0, flags:0,
    baseAlpha:1.0, baseTexCoord:0, mrTexCoord:0, normalTexCoord:0,
    normalScale:1.0, emissiveTex:-1, emissiveTexCoord:0,
    occlusionTex:-1, occlusionTexCoord:0, occlusionStrength:1.0,
    thicknessTex:-1, thicknessTexCoord:0, transmissionTexCoord:0,
    attenuationColor:[1,1,1], attenuationDistance:1e30,
  };
  if (!gltf.materials) return [defaultMat];

  return gltf.materials.map(mat => {
    const pbr = mat.pbrMetallicRoughness || {};

    // Base color factor (multiplied with texture in shader)
    const bcf = pbr.baseColorFactor || [1, 1, 1, 1];
    const albedo = [bcf[0], bcf[1], bcf[2]];
    const baseAlpha = bcf[3] ?? 1.0;

    let metallic = pbr.metallicFactor ?? 1.0;
    const roughness = pbr.roughnessFactor ?? 1.0;

    const baseBinding = textureBinding(gltf, pbr.baseColorTexture);
    const mrBinding = textureBinding(gltf, pbr.metallicRoughnessTexture);
    const normalBinding = textureBinding(gltf, mat.normalTexture);
    const emissiveBinding = textureBinding(gltf, mat.emissiveTexture);
    const occlusionBinding = textureBinding(gltf, mat.occlusionTexture);

    const baseTex = baseBinding.index;
    const mrTex = mrBinding.index;
    const normalTex = normalBinding.index;
    const emissiveTex = emissiveBinding.index;
    const occlusionTex = occlusionBinding.index;
    const baseTexCoord = baseBinding.texCoord;
    const mrTexCoord = mrBinding.texCoord;
    const normalTexCoord = normalBinding.texCoord;
    const emissiveTexCoord = emissiveBinding.texCoord;
    const occlusionTexCoord = occlusionBinding.texCoord;
    const normalScale = mat.normalTexture?.scale ?? 1.0;
    const occlusionStrength = mat.occlusionTexture?.strength ?? 1.0;

    let type = 0; // 0=PBR, 1=unlit, 3=transmission
    let emission = [0, 0, 0];
    let ior = 1.5;
    let transmission = 0.0;
    let transmissionTex = -1;
    let transmissionTexCoord = 0;
    let thickness = 0.0;
    let thicknessTex = -1;
    let thicknessTexCoord = 0;
    let flags = 0;
    let attenuationColor = [1, 1, 1];
    let attenuationDistance = 1e30;

    if (mat.emissiveFactor) {
      emission = [mat.emissiveFactor[0], mat.emissiveFactor[1], mat.emissiveFactor[2]];
    }
    if (mat.doubleSided) flags |= MAT_FLAG_DOUBLE_SIDED;
    if (mat.extensions?.KHR_materials_unlit) {
      flags |= MAT_FLAG_UNLIT;
      type = 1;
    }

    const transmissionExt = mat.extensions?.KHR_materials_transmission;
    const volumeExt = mat.extensions?.KHR_materials_volume;
    if (transmissionExt) {
      transmission = transmissionExt.transmissionFactor ?? 0.0;
      const txBinding = textureBinding(gltf, transmissionExt.transmissionTexture);
      transmissionTex = txBinding.index;
      transmissionTexCoord = txBinding.texCoord;
      thickness = volumeExt?.thicknessFactor ?? 0.0;
      const thBinding = textureBinding(gltf, volumeExt?.thicknessTexture);
      thicknessTex = thBinding.index;
      thicknessTexCoord = thBinding.texCoord;
      attenuationColor = volumeExt?.attenuationColor || attenuationColor;
      attenuationDistance = Number.isFinite(volumeExt?.attenuationDistance)
        ? volumeExt.attenuationDistance : attenuationDistance;

      if (transmission > 0.001 || transmissionTex >= 0) {
        type = 3;
        ior = mat.extensions?.KHR_materials_ior?.ior || 1.5;
        // glTF volume becomes active only when thicknessFactor > 0.
        if (thickness <= 0.0) flags |= MAT_FLAG_THIN_TRANSMISSION;
      }
    }

    let alphaMode = 0; // OPAQUE
    if (mat.alphaMode === 'MASK') alphaMode = 1;
    else if (mat.alphaMode === 'BLEND') alphaMode = 2;
    let alphaCutoff = mat.alphaCutoff ?? 0.5;

    // Emission strength: KHR_materials_emissive_strength or extract from magnitude
    let emissionStrength = mat.extensions?.KHR_materials_emissive_strength?.emissiveStrength || 0;
    const emLum = Math.max(emission[0], emission[1], emission[2]);
    if (emissionStrength <= 0 && emLum > 1.0) {
      // Hardcoded override with values > 1: extract magnitude as strength
      emissionStrength = emLum;
    }
    if (emissionStrength <= 0) emissionStrength = 1.0;
    // Normalize emission color to 0-1 range
    if (emLum > 1.0) {
      emission[0] /= emLum; emission[1] /= emLum; emission[2] /= emLum;
    }
    return {
      albedo, type, emission, roughness, metallic, baseTex, mrTex, normalTex,
      alphaMode, alphaCutoff, ior, emissionStrength,
      transmission, transmissionTex, thickness, flags,
      baseAlpha, baseTexCoord, mrTexCoord, normalTexCoord, normalScale,
      emissiveTex, emissiveTexCoord, occlusionTex, occlusionTexCoord, occlusionStrength,
      thicknessTex, thicknessTexCoord, transmissionTexCoord,
      attenuationColor, attenuationDistance,
    };
  });
}

// Inverse-transpose of upper-left 3x3 for correct normal transform
function normalMatrix3x3(m) {
  const a=m[0],b=m[4],c=m[8], d=m[1],e=m[5],f=m[9], g=m[2],h=m[6],k=m[10];
  return [e*k-f*h, f*g-d*k, d*h-e*g, c*h-b*k, a*k-c*g, b*g-a*h, b*f-c*e, c*d-a*f, a*e-b*d];
}

// ============================================================
// BVH Builder — Binned SAH (Surface Area Heuristic)
// Much better split quality than midpoint for irregular geometry.
// Reduces BVH traversal steps 30-50% → directly less memory traffic.
// ============================================================
const SAH_BINS = 8;

function surfaceArea(mnx,mny,mnz, mxx,mxy,mxz) {
  const dx = mxx-mnx, dy = mxy-mny, dz = mxz-mnz;
  return 2*(dx*dy + dy*dz + dz*dx);
}

function buildBVH(positions, triDataArr, triCount, onProgress) {
  const centroids = new Float32Array(triCount * 3);
  const tMin = new Float32Array(triCount * 3);
  const tMax = new Float32Array(triCount * 3);

  for (let i = 0; i < triCount; i++) {
    const base = i * 4;
    const i0 = triDataArr[base]*3, i1 = triDataArr[base+1]*3, i2 = triDataArr[base+2]*3;
    const ax=positions[i0],ay=positions[i0+1],az=positions[i0+2];
    const bx=positions[i1],by=positions[i1+1],bz=positions[i1+2];
    const cx=positions[i2],cy=positions[i2+1],cz=positions[i2+2];
    centroids[i*3]=(ax+bx+cx)/3; centroids[i*3+1]=(ay+by+cy)/3; centroids[i*3+2]=(az+bz+cz)/3;
    tMin[i*3]=Math.min(ax,bx,cx); tMin[i*3+1]=Math.min(ay,by,cy); tMin[i*3+2]=Math.min(az,bz,cz);
    tMax[i*3]=Math.max(ax,bx,cx); tMax[i*3+1]=Math.max(ay,by,cy); tMax[i*3+2]=Math.max(az,bz,cz);
  }

  const triOrder = new Uint32Array(triCount);
  for (let i = 0; i < triCount; i++) triOrder[i] = i;

  let nodeCap = Math.max(1024, Math.ceil(triCount / MAX_LEAF_SIZE) * 3);
  let nodesF32 = new Float32Array(nodeCap * 8);
  let nodesU32 = new Uint32Array(nodesF32.buffer);
  let nodeCount = 0;

  function allocNode() {
    if (nodeCount >= nodeCap) {
      nodeCap *= 2;
      const nf = new Float32Array(nodeCap * 8);
      nf.set(nodesF32);
      nodesF32 = nf;
      nodesU32 = new Uint32Array(nf.buffer);
    }
    return nodeCount++;
  }

  function computeAABB(start, end) {
    let mnx=1e30,mny=1e30,mnz=1e30, mxx=-1e30,mxy=-1e30,mxz=-1e30;
    for (let i = start; i < end; i++) {
      const t = triOrder[i], t3 = t*3;
      if(tMin[t3]<mnx)mnx=tMin[t3]; if(tMin[t3+1]<mny)mny=tMin[t3+1]; if(tMin[t3+2]<mnz)mnz=tMin[t3+2];
      if(tMax[t3]>mxx)mxx=tMax[t3]; if(tMax[t3+1]>mxy)mxy=tMax[t3+1]; if(tMax[t3+2]>mxz)mxz=tMax[t3+2];
    }
    return [mnx,mny,mnz, mxx,mxy,mxz];
  }

  // Pre-allocate bin arrays (reused per node to avoid GC pressure)
  const binCount = new Int32Array(SAH_BINS);
  const binMin = new Float32Array(SAH_BINS * 3);
  const binMax = new Float32Array(SAH_BINS * 3);

  function findSAHSplit(start, end, nmnx,nmny,nmnz, nmxx,nmxy,nmxz) {
    const count = end - start;
    const parentSA = surfaceArea(nmnx,nmny,nmnz, nmxx,nmxy,nmxz);
    const leafCost = count; // cost of testing all tris in a leaf
    let bestCost = leafCost;
    let bestAxis = -1, bestSplitPos = 0;

    // Compute centroid bounds
    let cmin = [1e30,1e30,1e30], cmax = [-1e30,-1e30,-1e30];
    for (let i = start; i < end; i++) {
      const t = triOrder[i], t3 = t*3;
      for (let a = 0; a < 3; a++) {
        if(centroids[t3+a]<cmin[a]) cmin[a]=centroids[t3+a];
        if(centroids[t3+a]>cmax[a]) cmax[a]=centroids[t3+a];
      }
    }

    for (let axis = 0; axis < 3; axis++) {
      const ext = cmax[axis] - cmin[axis];
      if (ext < 1e-8) continue;
      const scale = SAH_BINS / ext;

      // Reset bins
      binCount.fill(0); binMin.fill(1e30); binMax.fill(-1e30);

      // Fill bins
      for (let i = start; i < end; i++) {
        const t = triOrder[i], t3 = t*3;
        let b = ((centroids[t3+axis] - cmin[axis]) * scale)|0;
        if (b >= SAH_BINS) b = SAH_BINS - 1;
        binCount[b]++;
        const b3 = b*3;
        for (let a = 0; a < 3; a++) {
          if(tMin[t3+a]<binMin[b3+a]) binMin[b3+a]=tMin[t3+a];
          if(tMax[t3+a]>binMax[b3+a]) binMax[b3+a]=tMax[t3+a];
        }
      }

      // Left sweep: prefix counts + surface areas
      const lCnt = new Int32Array(SAH_BINS-1);
      const lSA = new Float32Array(SAH_BINS-1);
      let lc=0, l0=1e30,l1=1e30,l2=1e30, l3=-1e30,l4=-1e30,l5=-1e30;
      for (let i = 0; i < SAH_BINS-1; i++) {
        lc += binCount[i]; lCnt[i] = lc;
        const i3=i*3;
        if(binMin[i3]<l0)l0=binMin[i3]; if(binMin[i3+1]<l1)l1=binMin[i3+1]; if(binMin[i3+2]<l2)l2=binMin[i3+2];
        if(binMax[i3]>l3)l3=binMax[i3]; if(binMax[i3+1]>l4)l4=binMax[i3+1]; if(binMax[i3+2]>l5)l5=binMax[i3+2];
        lSA[i] = (lc>0) ? surfaceArea(l0,l1,l2,l3,l4,l5) : 0;
      }

      // Right sweep + evaluate SAH
      let rc=0, r0=1e30,r1=1e30,r2=1e30, r3=-1e30,r4=-1e30,r5=-1e30;
      for (let i = SAH_BINS-1; i > 0; i--) {
        rc += binCount[i];
        const i3=i*3;
        if(binMin[i3]<r0)r0=binMin[i3]; if(binMin[i3+1]<r1)r1=binMin[i3+1]; if(binMin[i3+2]<r2)r2=binMin[i3+2];
        if(binMax[i3]>r3)r3=binMax[i3]; if(binMax[i3+1]>r4)r4=binMax[i3+1]; if(binMax[i3+2]>r5)r5=binMax[i3+2];
        if (lCnt[i-1]===0 || rc===0) continue;
        const rSA = surfaceArea(r0,r1,r2,r3,r4,r5);
        const cost = 1.0 + (lCnt[i-1] * lSA[i-1] + rc * rSA) / parentSA;
        if (cost < bestCost) {
          bestCost = cost;
          bestAxis = axis;
          bestSplitPos = cmin[axis] + ext * (i / SAH_BINS);
        }
      }
    }
    return { axis: bestAxis, pos: bestSplitPos, cost: bestCost };
  }

  const rootIdx = allocNode();
  const workStack = [{ ni: rootIdx, start: 0, end: triCount }];
  let built = 0;

  while (workStack.length > 0) {
    const { ni, start, end } = workStack.pop();
    const count = end - start;
    const [mnx,mny,mnz, mxx,mxy,mxz] = computeAABB(start, end);
    const off = ni * 8;
    nodesF32[off]=mnx; nodesF32[off+1]=mny; nodesF32[off+2]=mnz;
    nodesF32[off+4]=mxx; nodesF32[off+5]=mxy; nodesF32[off+6]=mxz;

    if (count <= MAX_LEAF_SIZE) {
      nodesU32[off+3] = start;
      nodesU32[off+7] = count;
      built += count;
      if (built % 500000 < MAX_LEAF_SIZE) onProgress?.(`Building SAH BVH... ${((built/triCount)*100)|0}%`);
      continue;
    }

    // SAH split
    const sah = findSAHSplit(start, end, mnx,mny,mnz, mxx,mxy,mxz);

    if (sah.axis < 0) {
      // SAH says leaf is cheaper, or degenerate centroids — make leaf (capped)
      if (count <= 16) {
        nodesU32[off+3] = start; nodesU32[off+7] = count;
        built += count; continue;
      }
      // Too many for leaf, fallback midpoint on longest axis
      const ext = [mxx-mnx, mxy-mny, mxz-mnz];
      sah.axis = ext[1]>ext[0] ? (ext[2]>ext[1]?2:1) : (ext[2]>ext[0]?2:0);
      sah.pos = [mnx,mny,mnz][sah.axis] + ext[sah.axis]*0.5;
    }

    // Partition around SAH split
    let i = start, j = end - 1;
    while (i <= j) {
      if (centroids[triOrder[i]*3 + sah.axis] < sah.pos) { i++; }
      else { const tmp=triOrder[i]; triOrder[i]=triOrder[j]; triOrder[j]=tmp; j--; }
    }
    let mid = i;
    if (mid === start || mid === end) mid = (start + end) >> 1;

    const leftIdx = allocNode();
    const rightIdx = allocNode();
    nodesU32[off+3] = leftIdx;
    nodesU32[off+7] = 0;

    workStack.push({ ni: rightIdx, start: mid, end: end });
    workStack.push({ ni: leftIdx, start: start, end: mid });
  }

  // Reorder triData by BVH leaf order
  const sortedTriData = new Uint32Array(triCount * 4);
  for (let i = 0; i < triCount; i++) {
    const src = triOrder[i] * 4;
    const dst = i * 4;
    sortedTriData[dst]   = triDataArr[src];
    sortedTriData[dst+1] = triDataArr[src+1];
    sortedTriData[dst+2] = triDataArr[src+2];
    sortedTriData[dst+3] = triDataArr[src+3];
  }

  return {
    nodesF32: nodesF32.subarray(0, nodeCount * 8),
    nodesU32: nodesU32.subarray(0, nodeCount * 8),
    sortedTriData,
    nodeCount,
  };
}

// ============================================================
// Main loader
// ============================================================
// ============================================================
// IndexedDB cache for processed scene data
// ============================================================
function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open('ignis-scene-cache', 2); // bump version to invalidate stale cache
    req.onupgradeneeded = () => req.result.createObjectStore('scenes');
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}
function dbGet(db, key) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction('scenes', 'readonly');
    const req = tx.objectStore('scenes').get(key);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}
function dbPut(db, key, val) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction('scenes', 'readwrite');
    const req = tx.objectStore('scenes').put(val, key);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}

// Fast hash of ArrayBuffer (FNV-1a on first+last 1MB + size)
async function hashBuffer(buf) {
  const bytes = new Uint8Array(buf);
  const len = bytes.length;
  const chunk = Math.min(len, 1024 * 1024);
  let h = 2166136261 >>> 0;
  // Hash first chunk
  for (let i = 0; i < chunk; i++) { h ^= bytes[i]; h = Math.imul(h, 16777619) >>> 0; }
  // Hash last chunk
  for (let i = Math.max(0, len - chunk); i < len; i++) { h ^= bytes[i]; h = Math.imul(h, 16777619) >>> 0; }
  // Mix in length
  h ^= len; h = Math.imul(h, 16777619) >>> 0;
  return h.toString(16);
}

export async function loadScene(basePath, onProgress) {
  onProgress?.('Loading scene.gltf...');
  const gltf = await (await fetch(`${basePath}/scene.gltf`)).json();

  // Check cache first
  let db, cacheKey, cached;
  try {
    db = await openDB();
    // Hash based on gltf file size + accessor count (changes if scene changes)
    const gltfSig = `${JSON.stringify(gltf.accessors?.length)}-${gltf.buffers?.[0]?.byteLength}`;
    cacheKey = 'scene-v4-' + gltfSig;
    cached = await dbGet(db, cacheKey);
  } catch(e) { /* IndexedDB not available, proceed without cache */ }

  if (cached) {
    onProgress?.('Loading from cache (instant)...');
    return cached;
  }

  onProgress?.('Loading scene.bin (147 MB, first time only)...');
  const binResp = await fetch(`${basePath}/scene.bin`);
  const bin = await binResp.arrayBuffer();

  onProgress?.('Extracting meshes...');

  // Compute world transforms
  const worldMats = new Array(gltf.nodes.length).fill(null);
  function walkNodes(ni, parentMat) {
    const node = gltf.nodes[ni];
    const local = nodeLocalMatrix(node);
    const world = mat4Multiply(parentMat, local);
    worldMats[ni] = world;
    for (const ch of (node.children || [])) walkNodes(ch, world);
  }
  const scene = gltf.scenes[gltf.scene || 0];
  for (const ri of scene.nodes) walkNodes(ri, mat4Identity());

  // Count totals first
  let totalVerts = 0, totalTris = 0;
  for (let ni = 0; ni < gltf.nodes.length; ni++) {
    const node = gltf.nodes[ni];
    if (node.mesh === undefined) continue;
    const mesh = gltf.meshes[node.mesh];
    for (const prim of mesh.primitives) {
      if (prim.mode !== undefined && prim.mode !== 4) continue; // only TRIANGLES
      const posAcc = gltf.accessors[prim.attributes.POSITION];
      const idxAcc = gltf.accessors[prim.indices];
      totalVerts += posAcc.count;
      totalTris += idxAcc.count / 3;
    }
  }

  onProgress?.(`Extracting ${totalTris|0} triangles, ${totalVerts} vertices...`);

  // Allocate flat arrays
  const allPos = new Float32Array(totalVerts * 3);
  const allNrm = new Float32Array(totalVerts * 3);
  const allUV0 = new Float32Array(totalVerts * 2);
  const allUV1 = new Float32Array(totalVerts * 2);
  const allTriData = new Uint32Array(totalTris * 4);
  let vOff = 0, tOff = 0;

  for (let ni = 0; ni < gltf.nodes.length; ni++) {
    const node = gltf.nodes[ni];
    if (node.mesh === undefined) continue;
    const mesh = gltf.meshes[node.mesh];
    const wm = worldMats[ni];
    const nm = normalMatrix3x3(wm);

    for (const prim of mesh.primitives) {
      if (prim.mode !== undefined && prim.mode !== 4) continue;

      const positions = readAccessor(gltf, bin, prim.attributes.POSITION);
      const normals = prim.attributes.NORMAL != null ? readAccessor(gltf, bin, prim.attributes.NORMAL) : null;
      const uv0s = prim.attributes.TEXCOORD_0 != null ? readAccessor(gltf, bin, prim.attributes.TEXCOORD_0) : null;
      const uv1s = prim.attributes.TEXCOORD_1 != null ? readAccessor(gltf, bin, prim.attributes.TEXCOORD_1) : null;
      const indices = readAccessor(gltf, bin, prim.indices);
      const matIdx = prim.material ?? 0;
      const vertCount = gltf.accessors[prim.attributes.POSITION].count;

      for (let v = 0; v < vertCount; v++) {
        const sx = positions[v*3], sy = positions[v*3+1], sz = positions[v*3+2];
        // Transform position
        allPos[(vOff+v)*3]   = wm[0]*sx + wm[4]*sy + wm[8]*sz + wm[12];
        allPos[(vOff+v)*3+1] = wm[1]*sx + wm[5]*sy + wm[9]*sz + wm[13];
        allPos[(vOff+v)*3+2] = wm[2]*sx + wm[6]*sy + wm[10]*sz + wm[14];
        // Transform normal (inverse-transpose for non-uniform scale)
        if (normals) {
          const nx = normals[v*3], ny = normals[v*3+1], nz = normals[v*3+2];
          const rx = nm[0]*nx + nm[1]*ny + nm[2]*nz;
          const ry = nm[3]*nx + nm[4]*ny + nm[5]*nz;
          const rz = nm[6]*nx + nm[7]*ny + nm[8]*nz;
          const len = Math.sqrt(rx*rx + ry*ry + rz*rz) || 1;
          allNrm[(vOff+v)*3]   = rx/len;
          allNrm[(vOff+v)*3+1] = ry/len;
          allNrm[(vOff+v)*3+2] = rz/len;
        } else {
          allNrm[(vOff+v)*3+1] = 1;
        }
        // UV coordinates
        if (uv0s) {
          allUV0[(vOff+v)*2]   = uv0s[v*2];
          allUV0[(vOff+v)*2+1] = uv0s[v*2+1];
        }
        if (uv1s) {
          allUV1[(vOff+v)*2]   = uv1s[v*2];
          allUV1[(vOff+v)*2+1] = uv1s[v*2+1];
        }
      }

      const triCountPrim = indices.length / 3;
      for (let t = 0; t < triCountPrim; t++) {
        allTriData[(tOff+t)*4]   = indices[t*3]   + vOff;
        allTriData[(tOff+t)*4+1] = indices[t*3+1] + vOff;
        allTriData[(tOff+t)*4+2] = indices[t*3+2] + vOff;
        allTriData[(tOff+t)*4+3] = matIdx;
      }
      vOff += vertCount;
      tOff += triCountPrim;
    }
  }

  onProgress?.(`Building BVH for ${totalTris|0} triangles...`);

  // Build BVH
  const bvh = buildBVH(allPos, allTriData, totalTris, onProgress);

  // Pre-flatten triangle positions for cache-friendly BVH traversal
  // Layout: 3 × vec4f per triangle = 48 bytes, positions contiguous
  // v0.xyz + bitcast(mat_id), v1.xyz + 0, v2.xyz + 0
  onProgress?.('Flattening triangle data for GPU...');
  const gpuTriFlat = new Float32Array(totalTris * 12); // 3 vec4f per tri
  const gpuTriFlatU32 = new Uint32Array(gpuTriFlat.buffer);
  for (let i = 0; i < totalTris; i++) {
    const td = bvh.sortedTriData;
    const i0 = td[i*4]*3, i1 = td[i*4+1]*3, i2 = td[i*4+2]*3;
    const base = i * 12;
    gpuTriFlat[base]    = allPos[i0];   gpuTriFlat[base+1]  = allPos[i0+1]; gpuTriFlat[base+2]  = allPos[i0+2];
    gpuTriFlatU32[base+3] = td[i*4+3]; // material_id as u32
    gpuTriFlat[base+4]  = allPos[i1];   gpuTriFlat[base+5]  = allPos[i1+1]; gpuTriFlat[base+6]  = allPos[i1+2];
    gpuTriFlat[base+7]  = 0;
    gpuTriFlat[base+8]  = allPos[i2];   gpuTriFlat[base+9]  = allPos[i2+1]; gpuTriFlat[base+10] = allPos[i2+2];
    gpuTriFlat[base+11] = 0;
  }

  // Pack vertex data: position.xyz + UV0.x, normal.xyz + UV0.y, UV1 in separate buffer
  const gpuPositions = new Float32Array(totalVerts * 4);
  const gpuNormals = new Float32Array(totalVerts * 4);
  const gpuUV1 = new Float32Array(totalVerts * 2);
  for (let v = 0; v < totalVerts; v++) {
    gpuPositions[v*4]   = allPos[v*3];
    gpuPositions[v*4+1] = allPos[v*3+1];
    gpuPositions[v*4+2] = allPos[v*3+2];
    gpuPositions[v*4+3] = allUV0[v*2];    // UV0.x packed in position.w
    gpuNormals[v*4]   = allNrm[v*3];
    gpuNormals[v*4+1] = allNrm[v*3+1];
    gpuNormals[v*4+2] = allNrm[v*3+2];
    gpuNormals[v*4+3] = allUV0[v*2+1];   // UV0.y packed in normal.w
    gpuUV1[v*2] = allUV1[v*2];
    gpuUV1[v*2+1] = allUV1[v*2+1];
  }

  // Materials -> GPU format (160 bytes / 40 floats per material)
  const materials = extractMaterials(gltf);
  const gpuMaterials = new Float32Array(materials.length * 40);
  for (let i = 0; i < materials.length; i++) {
    const m = materials[i], o = i * 40;
    gpuMaterials[o]    = m.albedo[0];
    gpuMaterials[o+1]  = m.albedo[1];
    gpuMaterials[o+2]  = m.albedo[2];
    gpuMaterials[o+3]  = m.type;
    gpuMaterials[o+4]  = m.emission[0];
    gpuMaterials[o+5]  = m.emission[1];
    gpuMaterials[o+6]  = m.emission[2];
    gpuMaterials[o+7]  = m.roughness;
    gpuMaterials[o+8]  = m.metallic;
    gpuMaterials[o+9]  = m.baseTex;
    gpuMaterials[o+10] = m.mrTex;
    gpuMaterials[o+11] = m.normalTex;
    gpuMaterials[o+12] = m.alphaMode;
    gpuMaterials[o+13] = m.alphaCutoff;
    gpuMaterials[o+14] = m.ior;
    gpuMaterials[o+15] = m.emissionStrength || 1.0;
    gpuMaterials[o+16] = m.transmission || 0.0;
    gpuMaterials[o+17] = m.transmissionTex ?? -1;
    gpuMaterials[o+18] = m.thickness || 0.0;
    gpuMaterials[o+19] = m.flags || 0;
    gpuMaterials[o+20] = m.baseAlpha ?? 1.0;
    gpuMaterials[o+21] = m.baseTexCoord ?? 0;
    gpuMaterials[o+22] = m.mrTexCoord ?? 0;
    gpuMaterials[o+23] = m.normalTexCoord ?? 0;
    gpuMaterials[o+24] = m.normalScale ?? 1.0;
    gpuMaterials[o+25] = m.emissiveTex ?? -1;
    gpuMaterials[o+26] = m.occlusionTex ?? -1;
    gpuMaterials[o+27] = m.thicknessTex ?? -1;
    gpuMaterials[o+28] = m.transmissionTexCoord ?? 0;
    gpuMaterials[o+29] = m.emissiveTexCoord ?? 0;
    gpuMaterials[o+30] = m.occlusionTexCoord ?? 0;
    gpuMaterials[o+31] = m.thicknessTexCoord ?? 0;
    gpuMaterials[o+32] = m.occlusionStrength ?? 1.0;
    gpuMaterials[o+33] = m.attenuationDistance ?? 1e30;
    gpuMaterials[o+34] = m.attenuationColor?.[0] ?? 1.0;
    gpuMaterials[o+35] = m.attenuationColor?.[1] ?? 1.0;
    gpuMaterials[o+36] = m.attenuationColor?.[2] ?? 1.0;
    gpuMaterials[o+37] = 0.0;
    gpuMaterials[o+38] = 0.0;
    gpuMaterials[o+39] = 0.0;
  }

  // Collect emissive triangle indices for NEE
  // Build emissive triangle buffer:
  // 1 vec4 per tri: [tri_idx(bitcast u32), area, CDF, 0]
  // Sorted by power, CDF for importance sampling.
  // Keep a generous fixed budget (~2 MB) so large area lights are represented
  // without exploding memory on pathological fully-emissive scenes.
  const EMISSIVE_TRI_STRIDE_BYTES = 16;
  const MAX_EMISSIVE_BUFFER_BYTES = 2 * 1024 * 1024;
  const MAX_EMISSIVE = Math.max(256, Math.floor(MAX_EMISSIVE_BUFFER_BYTES / EMISSIVE_TRI_STRIDE_BYTES));
  const emissiveCandidates = [];
  for (let i = 0; i < totalTris; i++) {
    const matIdx = bvh.sortedTriData[i * 4 + 3];
    if (matIdx >= materials.length) continue;
    const m = materials[matIdx];
    if ((m.emission[0] <= 0.0 && m.emission[1] <= 0.0 && m.emission[2] <= 0.0) || (m.emissionStrength || 0) <= 0.0) continue;
    const vi0 = bvh.sortedTriData[i * 4], vi1 = bvh.sortedTriData[i * 4 + 1], vi2 = bvh.sortedTriData[i * 4 + 2];
    const v0 = [allPos[vi0*3], allPos[vi0*3+1], allPos[vi0*3+2]];
    const v1 = [allPos[vi1*3], allPos[vi1*3+1], allPos[vi1*3+2]];
    const v2 = [allPos[vi2*3], allPos[vi2*3+1], allPos[vi2*3+2]];
    const e1 = [v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]];
    const e2 = [v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]];
    const cx = e1[1]*e2[2]-e1[2]*e2[1], cy = e1[2]*e2[0]-e1[0]*e2[2], cz = e1[0]*e2[1]-e1[1]*e2[0];
    const area = 0.5 * Math.sqrt(cx*cx + cy*cy + cz*cz);
    if (area < 1e-8) continue;
    // Store color only (no strength) — shader reads live strength from material_buf
    const em = [m.emission[0], m.emission[1], m.emission[2]];
    const str = m.emissionStrength || 1;
    const lum = 0.2126*em[0]*str + 0.7152*em[1]*str + 0.0722*em[2]*str;
    const power = area * lum;
    if (power < 1e-6) continue;
    emissiveCandidates.push({ power, triIdx: i, area });
  }
  // Sort by power descending, keep top MAX_EMISSIVE if needed.
  emissiveCandidates.sort((a, b) => b.power - a.power);
  const emissiveSourceCount = emissiveCandidates.length;
  const emissiveCount = Math.min(emissiveSourceCount, MAX_EMISSIVE);
  const emissiveTruncated = emissiveSourceCount > emissiveCount;
  const totalPower = emissiveCandidates.slice(0, emissiveCount).reduce((s, t) => s + t.power, 0);

  // Build GPU buffer: 4 floats per triangle
  const gpuEmissiveTris = new Float32Array(Math.max(emissiveCount * 4, 4));
  const gpuEmissiveTrisU32 = new Uint32Array(gpuEmissiveTris.buffer);
  let cumulative = 0;
  for (let i = 0; i < emissiveCount; i++) {
    const t = emissiveCandidates[i], o = i * 4;
    cumulative += t.power / totalPower;
    gpuEmissiveTrisU32[o+0] = t.triIdx;
    gpuEmissiveTris[o+1] = t.area;
    gpuEmissiveTris[o+2] = cumulative;
    gpuEmissiveTris[o+3] = 0.0;
  }

  // Scene bounding box for camera
  let sceneMin = [1e30,1e30,1e30], sceneMax = [-1e30,-1e30,-1e30];
  for (let v = 0; v < totalVerts; v++) {
    for (let a = 0; a < 3; a++) {
      sceneMin[a] = Math.min(sceneMin[a], allPos[v*3+a]);
      sceneMax[a] = Math.max(sceneMax[a], allPos[v*3+a]);
    }
  }

  // Material names for debug
  const materialNames = gltf.materials ? gltf.materials.map(m => m.name || 'unnamed') : ['default'];

  // Texture info for renderer (image URIs for async loading)
  let textureInfo = null;
  if (gltf.images && gltf.images.length > 0) {
    textureInfo = {
      imageURIs: gltf.images.map(img => img.uri),
      count: gltf.images.length,
    };
  }

  const stats = {
    triangles: totalTris,
    vertices: totalVerts,
    bvhNodes: bvh.nodeCount,
    materials: materials.length,
    emissiveTris: emissiveCount,
    emissiveSourceTris: emissiveSourceCount,
    emissiveTruncated,
    sceneMin, sceneMax,
  };

  onProgress?.(`Scene ready: ${totalTris|0} tris, ${bvh.nodeCount} BVH nodes, ${textureInfo?.count || 0} textures`);

  // --- Rasterization data: separate opaque + blend index buffers ---
  onProgress?.('Building rasterization buffers...');
  const opaqueIdx = [];
  const blendIdx = [];
  for (let i = 0; i < totalTris; i++) {
    const td = bvh.sortedTriData;
    const matIdx = td[i*4+3];
    const tri = [td[i*4], td[i*4+1], td[i*4+2]];
    if (matIdx < materials.length && materials[matIdx].alphaMode === 2) {
      blendIdx.push(...tri);
    } else {
      opaqueIdx.push(...tri);
    }
  }
  const rasterIndices = new Uint32Array([...opaqueIdx, ...blendIdx]);
  const opaqueIndexCount = opaqueIdx.length;
  const blendIndexCount = blendIdx.length;

  // Per-vertex material ID
  const vertMatIds = new Float32Array(totalVerts);
  for (let i = 0; i < totalTris; i++) {
    const td = bvh.sortedTriData;
    const matIdx = td[i*4+3];
    vertMatIds[td[i*4]]   = matIdx;
    vertMatIds[td[i*4+1]] = matIdx;
    vertMatIds[td[i*4+2]] = matIdx;
  }

  const result = {
    gpuPositions,
    gpuNormals,
    gpuUV1,
    gpuTriData: bvh.sortedTriData,
    gpuTriFlat,
    gpuBVHNodes: bvh.nodesF32,
    gpuMaterials,
    materialStride: 40,
    gpuEmissiveTris,
    rasterIndices,
    vertMatIds,
    textureInfo,
    materialNames,
    stats,
  };

  // Cache processed scene in IndexedDB (skip 147MB download + BVH build next time)
  if (db) {
    try {
      onProgress?.('Caching processed scene...');
      await dbPut(db, cacheKey, result);
      onProgress?.('Cached! Next load will be instant.');
    } catch(e) { /* cache write failed, no problem */ }
  }

  return result;
}
