# Plan de implementación — Denoiser BMFR para Ignis Web

> Reconstrucción por regresión (Blockwise Multi-Order Feature Regression, Koskela et al. TOG 2019)
> como pase primario del **diffuse demodulado**. El **specular** permanece en ReBLUR.
> Objetivo: imagen limpia a 1 spp **sin** RT/tensor cores, usando sólo subgroups + f16 (lo que
> el 4060m expone hoy en WebGPU/D3D12).

---

## 0. Decisiones de arquitectura

| Decisión | Elección | Por qué |
|---|---|---|
| Señal a denoisar con BMFR | **Diffuse irradiance demodulada** (`noisy_out.rgb`) | Ya viene sin albedo → es suave → ideal para regresión. El specular es view-dependent y BMFR lo maneja mal. |
| Specular | Se queda en **ReBLUR** (sacado de debug, Tier 0) | Virtual-motion + roughness sigma ya están en `reblur.wgsl`. |
| Solver de la regresión (v1) | **Ecuaciones normales + Cholesky + ridge (λI)** | El paso caro (formar TᵀT) es una reducción paralela trivial en GPU; el solve 10×10 es minúsculo. Mucho más simple que Householder QR. |
| Solver (v2, si falta precisión) | **Householder QR aumentada** (como el paper/DXR) | Mejor condicionamiento; sólo si Cholesky+ridge muestra artefactos en bloques planos. |
| Features | `[1, nx, ny, nz, px, py, pz, px², py², pz²]` (10) | Igual que BMFR-DXR. Captura sombreado suave de 2º orden; no puede representar ruido de alta frecuencia. |
| Cambios al path tracer (v1) | **Ninguno** | World position se reconstruye en el shader desde `denoise_nd_out.w` (hit-dist) + cámara, como ya hace `temporal.wgsl:80-97`. |
| Precisión | Carga f16, **acumulación y solve en f32** | Las ecuaciones normales son sensibles; f32 evita pérdida de rango. |

---

## 1. Algoritmo BMFR (resumen técnico)

Tres etapas por frame:

1. **Pre-acumulación temporal del ruido**: EMA del diffuse 1spp con la historia reproyectada
   (α≈0.2; para <4 muestras, promedio acumulativo). Baja la varianza del input antes de ajustar.
2. **Regresión por bloques**: pantalla dividida en bloques de **32×32**, con **offset aleatorio por
   frame** (rompe las costuras de bloque). Por bloque:
   - Cada píxel aporta una fila `fᵢ = [1, n, p', p'²]` (10) y un target `cᵢ` (diffuse acumulado, RGB).
   - `p'` = posición normalizada al bounding-box del bloque (→ O(1), crítico para condicionamiento).
   - Resolver mínimos cuadrados `min ‖T·α − c‖²` → `α` (10×3).
   - Reconstruir `cᵢ' = fᵢᵀ·α`. Como `T` no tiene ruido, el ajuste **promedia el ruido sin
     emborronar bordes** (los features cambian de golpe en los bordes geométricos).
3. **Post-acumulación temporal**: EMA del resultado de regresión con su historia (quita el
   parpadeo residual de bloque). Esta es la señal diffuse final.

**Por qué converge rápido y sin manchas:** no promedia vecinos ruidosos (como à-trous/ReBLUR),
sino que **reconstruye desde la geometría sin ruido**. Depende mucho menos del temporal → ataca
directo la queja "ruidoso incluso quieto".

---

## 2. Qué se reutiliza (no tocar)

- **G-buffer NRD-ready** — `pathtracer.wgsl:2487-2490`:
  - `noisy_out`     = diffuse irradiance demodulada (rgb) — **input/target de BMFR**
  - `denoise_nd_out`= normal mundo (xyz) + hit-distance (w) — **features de BMFR**
  - `albedo_out`    = albedo (rgb) + roughness (a) — **remodulación en composite**
  - `specular_out`  = specular radiance (rgb) + hitDist norm (a) — **va a ReBLUR**
- **Demodulación por albedo** — `pathtracer.wgsl:2240-2244` (correcta, NRD-style). No tocar.
- **Reconstrucción de world-pos desde depth+cámara** — copiar el patrón de `temporal.wgsl:80-97`.
- **Composite/remodulación** — `denoise.wgsl:610` (`hdr = albedo*diff + spec`). Sólo cambia
  *qué textura* es `diff`.
- **Specular ReBLUR** — `reblur.wgsl` (tras sacarlo de debug en Tier 0).

---

## 3. Cambios por archivo

### 3.1. `bmfr.wgsl` — **NUEVO** (núcleo)

Tres `@compute` entrypoints:

#### `pre_accumulate` (opcional — se puede reusar `temporal.wgsl`)
- Reproyecta historia del diffuse (depth-based, como temporal.wgsl) y hace EMA.
- Salida: `bmfr_accum` (rgba16float): rgb = diffuse acumulado, a = sample count.
- **Recomendación:** en v1 reutilizar el pase temporal existente para no duplicar; el `fit`
  lee directo la salida temporal actual.

#### `fit` (la regresión — el kernel central)
- **Dispatch:** 1 workgroup por bloque de 32×32. `@workgroup_size(16,16)` = 256 threads;
  cada thread procesa 1024/256 = **4 píxeles**.
- **Uniforms (`BmfrParams`):** resolución, `block_offset: vec2i` (por frame), matrices/vectores de
  cámara (right/up/fwd/pos), `fov_factor`, `aspect`, `ridge_lambda`, `temporal_alpha`.
- **Fase A — acumular `A=TᵀT` (10×10 sim) y `b=Tᵀc` (10×3):**
  - Cada thread, sobre sus 4 píxeles: reconstruye world-pos (hit-dist+cámara), normaliza al bloque,
    arma `f` (10) y `c` (3), acumula en **registros f32** los 55 únicos de `A` + 30 de `b`.
  - Reducción workgroup vía **`subgroupAdd`** por subgrupo → `shared[numSubgroups][85]` (≈2.7 KB)
    → thread 0 suma los subgrupos. (Evita 256×85 f32 en shared, que excedería los 32 KB.)
- **Fase B — solve (thread 0 cooperativo):**
  - `A += λ·I` (ridge; `λ ≈ 1e-3·trace/10`).
  - Cholesky `A=LLᵀ`; resolver `L y = b`, `Lᵀ α = y` para los 3 canales.
  - Guard de rango: si algún `diag(L)` < ε → subir λ o caer a media de bloque (coef `[1]`).
  - `α` (10×3) → shared.
- **Fase C — reconstruir:** cada thread evalúa `cᵢ' = fᵢᵀ·α` para sus 4 píxeles → `bmfr_fit_out`.
- **Skip de cielo:** píxeles con hit-dist “infinito” (`cz > 1e5`, ver `temporal.wgsl:207`) se copian
  sin tocar.
- **Normalización de features:** primera mini-reducción del AABB de world-pos del bloque (min/max),
  o usar el centro del bloque + extent fijo derivado del depth. (Detalle de implementación; el AABB
  exacto da mejor condicionamiento.)

#### `post_accumulate`
- EMA de `bmfr_fit_out` con su historia (`bmfr_history`), depth-reject como temporal.
- Salida: `bmfr_out` (diffuse final denoisado) + copia a `bmfr_history`.

### 3.2. `renderer.js` — integración

- **Modo nuevo:** agregar rama `denoiseMode === 'bmfr'` en el dispatch (`~2355-2627`), en paralelo
  a las ramas `reblur`/`full`/`oidn` existentes.
- **Texturas nuevas** (todas rgba16float, tamaño interno `width×height`):
  `bmfr_accum`, `bmfr_fit_out`, `bmfr_out`, `bmfr_history_A/B` (ping-pong).
- **Pipelines:** `bmfrFitPipeline`, `bmfrPostPipeline` (+ `bmfrPrePipeline` si no se reusa temporal).
  Crear cerca de `~799-804` donde se crean los de reblur.
- **Bind groups:** `fit` lee `noisy_out`(o accum) + `denoise_nd_out`; escribe `bmfr_fit_out` +
  uniform `BmfrParams`. `post` hace ping-pong con history.
- **Uniform de offset por frame:** `block_offset = (hash(frameIndex) & 31, (hash>>5) & 31)`
  (Math.random/Date.now válidos acá). Subir a `BmfrParams` (patrón `Float32Array` como `~2430`).
- **Dispatch del `fit`:** `dispatchWorkgroups(ceil(width/32), ceil(height/32))` — **1 wg por bloque**
  (≠ el `ceil(/16)` de los otros pases).
- **Composite:** apuntar el bind group de composición al `bmfr_out` como señal diffuse
  (nuevo `dnBG_comp_bmfr`, análogo a `dnBG_comp_reblur` en `~2623`). Specular sigue del path ReBLUR.

### 3.3. `denoise.wgsl` — composite
- Sin cambios de lógica: `hdr = albedo * diff_bmfr + spec_reblur` ya es lo que hace `:610`.
  Sólo cambia la textura `diff` por binding (se resuelve en el bind group de renderer.js).

### 3.4. `index.html` — splash
- Agregar `<option value="bmfr">BMFR (regression)</option>` al `#splash-denoise` (`~219-225`).

### 3.5. `pathtracer.wgsl` — **sólo v2 (opcional)**
- v1 no lo toca. Para v2/calidad: emitir **world-position** y **viewZ lineal** dedicados (en vez de
  reconstruir), y **motion vectors** (habilita escenas dinámicas y mejor reproyección). Añadir 1
  textura `gbuf_pos_out`. Es deuda estructural ya identificada en la auditoría (no hay MV).

---

## 4. Diseño del kernel de regresión (resumen de recursos)

| Recurso | Uso | Presupuesto |
|---|---|---|
| Workgroup storage | `shared[subgroups][85] f32` (reducción) + `α[30] f32` | ≈3 KB de 32 KB ✓ |
| Threads/wg | 256 (16×16), 4 píxeles c/u | ✓ |
| Subgroups | `subgroupAdd` para reducir A/b | feature ya disponible ✓ |
| Precisión | load f16 → acumular/solve f32 → store f16 | ✓ |
| Cómputo dominante | formar `A` (Σ outer-products) | O(blockPx·55), paralelo |

---

## 5. Edge cases y parámetros

- **Bloques planos (rank-deficiency):** pos/pos² colineales → ridge `λ` lo estabiliza; guard de
  `diag(L)` cae a media de bloque.
- **Costuras de bloque:** offset aleatorio por frame + post-acumulación temporal las disuelven.
- **Bordes de pantalla:** bloques parciales → clamp del rango de píxeles.
- **Cielo / miss:** copia passthrough (no regresión).
- **Parámetros iniciales:** block=32, λ=1e-3·(trace/10), pre-α=0.2, post-α=0.1, features=10.

---

## 6. Milestones y validación

1. **M0 — Tier 0 (prerequisito):** sacar ReBLUR de debug (`reblur.wgsl:441-443`) para tener un
   specular sano de referencia. *Validar:* specular deja de sobre-difuminar.
2. **M1 — `fit` solo (sin temporal):** modo `bmfr`, regresión cruda sobre `noisy_out`.
   *Validar:* el diffuse se ve **reconstruido** (limpio en superficies, bordes nítidos) aunque
   parpadee entre frames. Comparar contra `off` (crudo) y `full`.
3. **M2 — + pre/post acumulación temporal:** agregar EMA. *Validar:* el parpadeo desaparece,
   converge en pocos frames con cámara quieta. **Esta es la métrica de éxito principal.**
4. **M3 — tuning:** λ, block-offset, α; medir ms del `fit` en el HUD (`TIMING`).
   *Objetivo:* `fit` < ~3-4 ms a 968×448 en el 4060m.
5. **M4 (opcional) — v2:** QR si Cholesky muestra artefactos; world-pos/MV dedicados en el PT.

*Cómo medir:* el log ya emite `TIMING: total/gbuf/...` (`client.log`); agregar un timestamp del
pase BMFR. Comparar visualmente `off` vs `bmfr` vs `full` con el selector (ya funcional).

---

## 7. Riesgos y mitigaciones

| Riesgo | Prob. | Mitigación |
|---|---|---|
| `fit` demasiado lento (reducción A) | Media | Bajar a 8 features (sin pos²), block 16×16, o tilear con subgroups; medir M3. |
| Inestabilidad numérica (Cholesky) | Media | Ridge λ + normalización de pos por bloque; fallback QR (v2). |
| Costuras de bloque visibles | Baja | Offset por frame + post-temporal; si persiste, blend de 2 grids desfasados. |
| Sin motion vectors → ghosting en dinámico | Alta (escenas dinámicas) | v1 asume estático (Sponza ok); v2 agrega MV al PT. |
| Specular sigue ruidoso | Media | Es independiente; lo cubre el Tier 0 de ReBLUR. |

---

## 8. Resumen de archivos tocados

| Archivo | Acción |
|---|---|
| `bmfr.wgsl` | **NUEVO** — `fit` (+ `pre/post_accumulate`) |
| `renderer.js` | Modo `bmfr`, pipelines, texturas, bind groups, uniform offset, dispatch /32 |
| `denoise.wgsl` | Sin cambios de lógica (composite vía bind group) |
| `index.html` | Opción `bmfr` en el splash |
| `reblur.wgsl` | Tier 0 (prerequisito): sacar radio de debug |
| `pathtracer.wgsl` | Sólo v2: world-pos/viewZ/MV dedicados |
