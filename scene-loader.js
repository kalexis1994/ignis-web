// GLTF Scene Loader + BVH Builder for WebGPU Path Tracer

const MAX_LEAF_SIZE = 8;

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

// ============================================================
// Material extraction
// ============================================================
const FALLBACK_COLORS = {
  'roof_tiles_01': [0.55, 0.30, 0.15],
  'stones_01_tile': [0.65, 0.62, 0.55],
  'metal_door': [0.30, 0.28, 0.25],
  'brickwall_02': [0.60, 0.42, 0.32],
  'brickwall_01': [0.65, 0.48, 0.35],
  'window_frame_01': [0.28, 0.26, 0.23],
  'glass': [0.92, 0.92, 0.95],
  'door_stoneframe_02': [0.65, 0.60, 0.50],
  'door_stoneframe_01': [0.67, 0.62, 0.52],
  'stones_2ndfloor': [0.62, 0.58, 0.50],
  'ornament_lion': [0.65, 0.60, 0.50],
  'arch_stone_wall_01': [0.70, 0.67, 0.60],
  'stone_trims_01': [0.67, 0.65, 0.58],
  'ornament_01': [0.63, 0.58, 0.50],
  'floor_01': [0.60, 0.57, 0.50],
  'wood_01': [0.50, 0.32, 0.18],
  'wood_door_01': [0.45, 0.28, 0.15],
  'ceiling_plaster_02': [0.78, 0.76, 0.70],
  'ceiling_plaster_01': [0.80, 0.78, 0.72],
  'stone_trims_02': [0.65, 0.63, 0.55],
  'dirt_decal': [0.30, 0.25, 0.18],
  'column_head_1stfloor': [0.67, 0.63, 0.55],
  'column_1stfloor': [0.70, 0.67, 0.60],
  'column_head_2ndfloor_03': [0.65, 0.62, 0.55],
  'column_brickwall_01': [0.63, 0.50, 0.38],
  'column_head_2ndfloor_02': [0.65, 0.62, 0.55],
};

function extractMaterials(gltf) {
  if (!gltf.materials) return [{ albedo:[0.7,0.7,0.7], type:0, emission:[0,0,0], roughness:1, ior:1.5 }];

  return gltf.materials.map(mat => {
    const pbr = mat.pbrMetallicRoughness || {};
    const name = mat.name || '';
    let albedo = [0.7, 0.7, 0.7];
    const factor = pbr.baseColorFactor;

    if (factor && (factor[0] !== 1 || factor[1] !== 1 || factor[2] !== 1)) {
      albedo = [factor[0], factor[1], factor[2]];
    } else if (pbr.baseColorTexture && FALLBACK_COLORS[name]) {
      albedo = FALLBACK_COLORS[name];
    } else if (factor) {
      albedo = [factor[0], factor[1], factor[2]];
    }

    const metallic = pbr.metallicFactor ?? 1.0;
    const roughness = pbr.roughnessFactor ?? 1.0;

    let type = 0; // diffuse
    let emission = [0, 0, 0];
    let ior = 1.5;

    if (name === 'light_bulb') {
      type = 2; emission = [8.0, 7.5, 6.0];
    } else if (name === 'lamp_glass_01') {
      type = 2; emission = [5.0, 3.0, 1.0];
    } else if (name === 'glass') {
      type = 3; albedo = [0.97, 0.97, 0.98]; ior = 1.5;
    } else if (metallic > 0.5 && !pbr.metallicRoughnessTexture) {
      type = 1; // metal
    }

    return { albedo, type, emission, roughness, ior };
  });
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
    const req = indexedDB.open('ignis-scene-cache', 1);
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
    cacheKey = 'scene-' + gltfSig;
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
  const allTriData = new Uint32Array(totalTris * 4);
  let vOff = 0, tOff = 0;

  for (let ni = 0; ni < gltf.nodes.length; ni++) {
    const node = gltf.nodes[ni];
    if (node.mesh === undefined) continue;
    const mesh = gltf.meshes[node.mesh];
    const mat = worldMats[ni];

    for (const prim of mesh.primitives) {
      if (prim.mode !== undefined && prim.mode !== 4) continue;

      const positions = readAccessor(gltf, bin, prim.attributes.POSITION);
      const hasNormals = prim.attributes.NORMAL !== undefined;
      const normals = hasNormals ? readAccessor(gltf, bin, prim.attributes.NORMAL) : null;
      const indices = readAccessor(gltf, bin, prim.indices);
      const matIdx = prim.material ?? 0;
      const vertCount = gltf.accessors[prim.attributes.POSITION].count;

      // Transform and store vertices
      for (let v = 0; v < vertCount; v++) {
        const sx = positions[v*3], sy = positions[v*3+1], sz = positions[v*3+2];
        // Transform position
        allPos[(vOff+v)*3]   = mat[0]*sx + mat[4]*sy + mat[8]*sz + mat[12];
        allPos[(vOff+v)*3+1] = mat[1]*sx + mat[5]*sy + mat[9]*sz + mat[13];
        allPos[(vOff+v)*3+2] = mat[2]*sx + mat[6]*sy + mat[10]*sz + mat[14];
        // Transform normal
        if (normals) {
          const nx = normals[v*3], ny = normals[v*3+1], nz = normals[v*3+2];
          const rx = mat[0]*nx + mat[4]*ny + mat[8]*nz;
          const ry = mat[1]*nx + mat[5]*ny + mat[9]*nz;
          const rz = mat[2]*nx + mat[6]*ny + mat[10]*nz;
          const len = Math.sqrt(rx*rx + ry*ry + rz*rz) || 1;
          allNrm[(vOff+v)*3]   = rx/len;
          allNrm[(vOff+v)*3+1] = ry/len;
          allNrm[(vOff+v)*3+2] = rz/len;
        } else {
          allNrm[(vOff+v)*3+1] = 1; // default up
        }
      }

      // Store triangle data (re-index to global vertex offset)
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

  // Keep indexed normals + tri_data for post-hit normal lookup only
  const gpuPositions = new Float32Array(totalVerts * 4);
  const gpuNormals = new Float32Array(totalVerts * 4);
  for (let v = 0; v < totalVerts; v++) {
    gpuPositions[v*4]   = allPos[v*3];
    gpuPositions[v*4+1] = allPos[v*3+1];
    gpuPositions[v*4+2] = allPos[v*3+2];
    gpuPositions[v*4+3] = 0;
    gpuNormals[v*4]   = allNrm[v*3];
    gpuNormals[v*4+1] = allNrm[v*3+1];
    gpuNormals[v*4+2] = allNrm[v*3+2];
    gpuNormals[v*4+3] = 0;
  }

  // Materials -> GPU format (32 bytes per material)
  const materials = extractMaterials(gltf);
  const gpuMaterials = new Float32Array(materials.length * 8);
  for (let i = 0; i < materials.length; i++) {
    const m = materials[i];
    gpuMaterials[i*8]   = m.albedo[0];
    gpuMaterials[i*8+1] = m.albedo[1];
    gpuMaterials[i*8+2] = m.albedo[2];
    gpuMaterials[i*8+3] = m.type;       // mat_type
    gpuMaterials[i*8+4] = m.emission[0];
    gpuMaterials[i*8+5] = m.emission[1];
    gpuMaterials[i*8+6] = m.emission[2];
    gpuMaterials[i*8+7] = m.roughness;
  }

  // Collect emissive triangle indices for NEE
  const emissiveTris = [];
  for (let i = 0; i < totalTris; i++) {
    const matIdx = bvh.sortedTriData[i * 4 + 3];
    if (matIdx < materials.length && materials[matIdx].type === 2) {
      emissiveTris.push(i);
    }
  }
  const gpuEmissiveTris = new Uint32Array(emissiveTris.length > 0 ? emissiveTris : [0]);

  // Scene bounding box for camera
  let sceneMin = [1e30,1e30,1e30], sceneMax = [-1e30,-1e30,-1e30];
  for (let v = 0; v < totalVerts; v++) {
    for (let a = 0; a < 3; a++) {
      sceneMin[a] = Math.min(sceneMin[a], allPos[v*3+a]);
      sceneMax[a] = Math.max(sceneMax[a], allPos[v*3+a]);
    }
  }

  const stats = {
    triangles: totalTris,
    vertices: totalVerts,
    bvhNodes: bvh.nodeCount,
    materials: materials.length,
    emissiveTris: emissiveTris.length,
    sceneMin, sceneMax,
  };

  onProgress?.(`Scene ready: ${totalTris|0} tris, ${bvh.nodeCount} BVH nodes`);

  const result = {
    gpuPositions,
    gpuNormals,
    gpuTriData: bvh.sortedTriData,
    gpuTriFlat,
    gpuBVHNodes: bvh.nodesF32,
    gpuMaterials,
    gpuEmissiveTris,
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
