# Plan de optimización — Path Tracer para SD 8 Gen 3 (Adreno 750)

**Objetivo:** 30 fps estables en Snapdragon 8 Gen 3.

## Contexto del hardware: Adreno 750

- ~3 TFLOPS FP32, **~6 TFLOPS FP16** (ratio 2:1 nativo)
- Wave size: **64 lanes**
- VGPR budget: **~128 registros/lane** antes de caer ocupación
- LPDDR5X: 51-67 GB/s compartido con CPU
- Sin RT por hardware → BVH traversal 100% ALU
- Framebuffer compression no ayuda en compute shaders (solo render passes)
- **Budget a 30 fps en 540p internos:** ~60 ns/pixel total

---

## Bottlenecks identificados

### 1. Mega-kernel `path_trace` = matanza de registros
**Ubicación:** `pathtracer.wgsl:1865-2275`

410 líneas en un único kernel que incluye:
- Glass thin + thick, alpha blend, clearcoat, specular, diffuse closures
- Stacks `sharc_pos/nrm/rad/dir[4]` + `medium_*[4]` = **~400 bytes stack/lane**
- El compilador no puede encajar esto en 128 VGPRs → **spill a memoria** → occupancy 1-2 waves

**Es la causa #1 de que no corra rápido.**

**Fix:** Wavefront path tracing (partir en pases):
- Pass A: primary ray + gbuffer lookup
- Pass B: NEE sun + env (kernel pequeño)
- Pass C: BSDF sample + bounce (stream compaction)
- Pass D: shade bounce → SHaRC

Cost: bandwidth extra (ray buffers). Gain: **ocupación 3-4×**, ALU se llena.

---

### 2. Material struct 320 bytes leído por hit
**Ubicación:** `pathtracer.wgsl:52-72`

20× vec4f = 5 cache lines cargadas siempre, aunque el shader use 5 campos. En alpha shadow test (`pathtracer.wgsl:1280-1291`) se lee material completo para descartar.

**Fix:** Partir en:
- **Material core** (32B): `albedo + mat_type`, `emission + rough`, `metallic + flags`, `ior + trans`
- **Material extras** (indirecto): clearcoat/sheen/aniso/iridescence, cargado solo tras chequeo `if flags & X`

Ahorras 4 cache lines por hit y baja presión de registros.

---

### 3. Cuatro UV sets siempre calculados
**Ubicación:** `pathtracer.wgsl:1935-1938`

```wgsl
let uv0 = get_uv0(...); uv1 = get_uv1(...); uv2 = ...; uv3 = ...;
```

99% de glTFs usan solo uv0. Cada `load_uv_extra` toca storage buffer → 4× loads por hit.

**Fix:** Cargar `uv0` siempre. Cargar `uv1-3` solo si `mat_texcoord_mask != 0` (precomputado en CPU al construir el buffer de materiales).

---

### 4. FP32 en todas partes → pierdes 2× throughput
**Ubicación:** todo el shader

Adreno 750 tiene FP16 al doble de rate. Mantener FP32 en BRDF/colores/throughput es tirar performance.

**Fix quirúrgico:**
- A `vec3h` (f16): `throughput`, `diff_rad`, `spec_rad`, emissive, base_color, Fresnel
- Mantener FP32: posiciones mundiales, direcciones tras `normalize`, `t` de intersección, PDFs
- `sample_ggx_vndf` y `fresnel_real` → mitad en f16

**Solo esto = 20-40% speedup.**

---

### 5. Doble función de traversal duplicada
**Ubicación:** `pathtracer.wgsl:1266-1308` (`trace_bvh` y `trace_shadow`)

Misma lógica con única diferencia: `any_hit` vs `closest_hit`. Código duplicado → compilador no comparte paths → más I-cache pressure + VGPRs totales.

**Fix:** Una función `traverse<ANY_HIT>(...)` parametrizada.

---

### 6. BVH stack de 16 slots en VGPR
**Ubicación:** `pathtracer.wgsl:1268`

`stk: array<u32, 16>` = 64 bytes en registros. Puede caer a memoria en Adreno. Además cada nodo hace 2 intersecciones AABB antes de decidir orden (`tl < tr`).

**Fix opciones:**
- Stackless rope BVH (Hapala) o short-stack 4-6 con restart → reduce VGPRs
- BVH8 (8-wide) con SIMD AABB test → menor profundidad, mejor coherencia

---

### 7. Workgroup 16×16 (256 threads) posiblemente subóptimo
**Ubicación:** `pathtracer.wgsl:2347`

4 waves/WG en Adreno. Con mega-kernel + stacks pesados puede que no quepa ni 1 WG completo por SP. Coherencia de rayos muere tras bounce 1 de todas formas.

**Fix:** Probar **8×8 (1 wave/WG)**. Menos competencia por VGPR file → mejor ocupación real. Contraintuitivo pero común en móvil.

---

### 8. Texturas sin control de mip
**Ubicación:** `sample_base_rgba` y similares (línea 1619+)

Sin `textureSampleLevel` explícito para hits indirectos, ni ray cones/differentials. En bounces la divergencia de UV explota → cache thrashing en TMU.

**Fix:** Ray cones (propagar `cone_width` por bounce) → calcular LoD explícito. En bounces ≥2, forzar mip 2+. El denoiser se come la pérdida de detalle.

---

### 9. SHaRC atomics en cada store
**Ubicación:** `pathtracer.wgsl:355` (`sharc_accum_rgbs`)

Atomic add a global memory = **lentísimo en móvil**, serializa waves. `sharc_count` hasta 4 → hasta 4 atomics por pixel.

**Fix:**
- SHaRC solo para bounces ≥1 (no primary)
- Bufferizar en shared memory del workgroup + flush con 1 atomic/WG al final
- Alternativa pragmática: **SHaRC off en modo 30fps**, confiar en ReBLUR + 1 bounce

---

### 10. ReBLUR 6 pasadas a res completa
**Ubicación:** `reblur.wgsl`, integración en `renderer.js:2181-2335`

Para 30 fps mobile, ReBLUR completo ~4-6ms/frame solo. **Muy caro.**

**Fix:** ReBLUR Lite:
- Temporal + 1 blur + stabilize (saltar prepass y post-blur)
- O mitad de pasadas a half-res con upsample bilateral
- Alternativa: A-SVGF variante ligera

---

## Roadmap por impacto

| # | Cambio | Esfuerzo | Speedup estimado |
|---|--------|----------|------------------|
| 1 | FP16 en throughput/colores/BRDF | Medio | 1.3-1.6× |
| 2 | Material split core/extras + UV mask | Bajo | 1.2× |
| 3 | Workgroup 8×8 + medir ocupación | Trivial | 1.1-1.3× |
| 4 | Unificar `trace_bvh`/`trace_shadow`, stack más corto | Medio | 1.1× |
| 5 | Ray cones + LoD forzado en bounces | Medio | 1.2× |
| 6 | ReBLUR Lite (3 pasadas) | Medio | 1.5× en denoise |
| 7 | Wavefront PT (2-4 kernels) | **Alto** | 1.8-2.5× |

## Estrategia recomendada

**Fase 1 (low-hanging, 2-3 días):**
Cambios #2, #3, #4 → refactor puro sin riesgo arquitectónico. Libera registros suficientes para notar diferencia incluso antes de tocar FP16.

**Fase 2 (impacto alto, 1 semana):**
Cambios #1, #5, #6 → FP16 + ray cones + ReBLUR Lite. Aquí es donde empiezas a ver 30 fps asomar.

**Fase 3 (la apuesta grande):**
Cambio #7 → wavefront PT. Inversión alta pero es lo que da margen estable.

## Configuración target para 30 fps

- Primary rays @ 540p internos
- 1 bounce indirecto (max_bounces = 2)
- SHaRC off
- ReBLUR Lite
- FSR upscale a 1080p
- Materiales: glTF core (sin clearcoat/sheen/aniso/iridescence en escena primaria)

---

## Notas de implementación

**Sobre medir:**
- WebGPU `timestamp-query` NO es viable en Adreno/Mali: cada `timestampWrites` en un pass fuerza un pipeline flush del orden de 20-50ms. Con 11 passes instrumentados el frame cae a ~3 fps (barreras dominan todo). Descartado.
- Baseline disponible: log ya imprime `TIMING: total=X gbuf=X shadow=X encode=X gpu=X` cada 120 frames via `performance.now()` + `onSubmittedWorkDone()`. Resolución coarse pero sin overhead. Suficiente para ver si un cambio mueve la aguja.
- Alternativa profesional: Snapdragon Profiler (PC tethered al teléfono, captura RenderDoc-style de frames).
- Comparar VGPR usage antes/después via `tint` dump cuando sea posible.

**Invariantes a preservar:**
- Shadow terminator fix (`pathtracer.wgsl:120-127`)
- Ray offset Wächter & Binder (`pathtracer.wgsl:130-147`)
- PCG3D decorrelación (`pathtracer.wgsl:194-199`)
- AgX tonemap en composite
