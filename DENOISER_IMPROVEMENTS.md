# Denoiser — mejoras propuestas y estado

Auditoría del denoiser (SVGF temporal + à-trous, ReBLUR specular, BMFR, orquestación
en `renderer.js`). Objetivo: performance y calidad a 1 spp en desktop (4060m) y móvil
(Adreno 750, límite 8 storage buffers).

Leyenda de estado: ✅ aplicado y validado · ⏳ pendiente · 💤 decisión pendiente.

---

## ✅ Aplicado (commit `3f73fca`)

### Bugs activos
| # | Fix | Archivo | Impacto |
|---|---|---|---|
| 1 | `in_mv` en `temporal_stabilization` leía la textura de **normales** como motion vectors → reproyectaba la historia 1 unidad a lo largo de la normal. Escena estática: `Xprev = X`. | `reblur.wgsl:997` | Calidad alto |
| 2 | El hitT **especular** (`in_spec.a`) encogía el radio y rechazaba taps del filtro **difuso**. El difuso no tiene hitT propio → eliminado de ese camino (−12 loads/px/pase). | `denoise.wgsl:115,128,180` | Calidad alto + perf |
| 3 | Radio de blur specular = mismo que diffuse (sin roughness). Ahora `radius * spec_magic_curve(roughness)`, taps separados diffuse/spec, peso de normal lobe-aware. | `reblur.wgsl` prepass/blur/post_blur | Calidad alto |
| 4 | Offsets Poisson con `vec2i(x + 0.5)` truncan hacia cero → sesgo de los taps negativos. Cambiado a `round()`. | `denoise.wgsl`, `reblur.wgsl` | Calidad medio |
| 5 | Anti-firefly a 1.5σ (el comentario decía 3σ) recortaba energía legítima a 1 spp. Subido a 3σ. | `denoise.wgsl` preblur ×2 | Calidad medio |

### Calidad
| # | Fix | Archivo |
|---|---|---|
| 6 | Radio de preblur escalado por `frames_still` (`mix(3, 15)` px) en vez de 15px fijo → la historia convergida deja de hornear blur permanente; la nitidez vuelve con la cámara quieta. | `denoise.wgsl` preblur + preblur_sm |
| 7 | AABB temporal ahora **siempre activo** con expansión que crece con la historia (0.5×→4×): captura cambios de luz gruesos en zonas convergidas sin clampear el rango de ruido 1-spp. | `temporal.wgsl` |

### Performance
| # | Fix | Archivo | Impacto |
|---|---|---|---|
| 8 | Auto-exposure jerárquico: reducción en memoria de workgroup + 1 `atomicAdd` global por workgroup en vez de 2 por píxel (~4M atomics globales contendidos/frame). | `denoise.wgsl` composite | Perf **alto en Adreno** |
| — | Dedup de loads `normal`+`viewZ` del mismo texel en los loops de blur de reblur (~25% menos fetches). | `reblur.wgsl` | Perf medio |

---

## ⏳ Pendiente — Performance

### P1. Timestamp queries GPU reales *(prerequisito de todo lo demás)*
- **Dónde:** `renderer.js:2105-2107, 2826-2851`; `requiredFeatures` L60-63.
- **Hoy:** `passTimers` declara temporal/denoise/composite/fsr/oidn pero **nunca se escriben**; el HUD TIMING mide tiempo de *encoding CPU*, no GPU. No se puede saber cuánto cuesta cada pase en Adreno.
- **Mejora:** pedir `'timestamp-query'` si está disponible, `timestampWrites` por compute pass, resolver a buffer con readback asíncrono.
- **Impacto:** Alto (habilita medir todo lo demás).

### P2. Eliminar pases redundantes por frame
- **Passthrough con dispatch completo** (`denoise.wgsl:99-103`): con `step_size ≥ 7.5` el pase entero es load→store de 2 texturas. No despacharlo desde CPU. (~66 MB de tráfico/frame a 1080p tirados.)
- **`copy_to_history` fusionable** (`denoise.wgsl:667`, `renderer.js:2693/2730`): el último à-trous podría escribir directo a `historyA/B` propagando el alfa. −1 dispatch full-res/frame.
- **Historia escrita 2× por frame** (`renderer.js:753/758` vs `copy_to_history`): el temporal escribe `history_out` y luego `copy_to_history` lo sobrescribe. Quitar el store del temporal cuando el modo despacha `copy_to_history`.
- **À-trous fijo por perfil** (`renderer.js:780,2715`): con `framesStill > ~30` los pases 4-5 (step 8/16) aportan casi nada; saltarlos en estado estacionario. −2 pases/frame en desktop.
- **Impacto:** −3/4 dispatches full-res/frame en estado estacionario.

### P3. f16 en los kernels de filtrado
- **Dónde:** ambos shaders (solo el composite usa `vec3h`).
- **Mejora:** `enable f16;` + colores/pesos/stats en `vec3h/f16` en atrous/preblur/stabilization; mantener `w_sum` y momentos en f32.
- **Impacto:** Perf medio-alto en Adreno (fp16 double-rate, menos presión de registros), bajo en 4060m.

### P4. Tiling en shared memory de los pases pesados de reblur
- `temporal_stabilization` hace 72 loads/px (ventana 5×5 × 3 texturas) → tile 20×20 en workgroup ≈ 1.6 loads/px.
- Los blurs de reblur no tienen variante shared-memory (à-trous/preblur de denoise.wgsl sí: `atrous_sm`, `preblur_sm`).
- **Impacto:** Perf alto en móvil.

### P5. Memoria: alocar recursos por modo
- **Dónde:** `renderer.js:824-834, 899-917`.
- **Hoy:** 5 texturas BMFR full-res + 7 pipelines se crean aunque `denoiseMode !== 'bmfr'`; ídem los 6 pipelines ReBLUR en modo full/oidn.
- **Mejora:** alocación lazy por modo (como ya hace OIDN, L1392). ~10-20 MB + tiempo de init en Adreno.

### P6. Formatos de textura sobredimensionados
- `matIdTex` rgba16float (8 B/px) para matId+fract(UV) → cabría en `rg16uint`+`rg8unorm`.
- `denoiseNdTex` + `ndTex` guardan la normal **dos veces** → unificable (octaédrica rg16f + depth).
- `prevFrameTex` full-res nunca escrita ("legacy… kept for layout stability") → sacar del layout.
- historia `rgba16float` → evaluar `rg11b10ufloat` + alpha aparte (−50% BW del pase más caro).

---

## ⏳ Pendiente — Calidad

### C1. Varianza SVGF real *(la mejora de fondo)*
- **Dónde:** `denoise.wgsl:176-178` (peso de luma heurístico), `temporal.wgsl` (no acumula momentos).
- **Hoy:** pese al README ("variance-guided"), no hay varianza en ningún sitio. El peso de luma usa una tolerancia proporcional al brillo → sobre-rechazo en zonas oscuras (ruido residual), overblur en brillantes.
- **Mejora:** (1) acumular μ y μ² de luma en el temporal (canal extra `rg16float`); (2) estimación espacial 7×7 cuando `history < 4` (SVGF §4.2); (3) peso `exp(-|Δl| / (σ_l·√(gauss(var)) + ε))` con σ_l≈4; (4) propagar `var' = Σw²·var / (Σw)²`.
- **Impacto:** Alto. Casi todos los heurísticos frágiles (deadzone 0.3, corte de AABB, clamp 1.5σ) son compensaciones de esta ausencia y se simplifican al añadirla.

### C2. Reproyección con validez por tap
- **Dónde:** `temporal.wgsl:233-237`.
- **Hoy:** historia muestreada con Catmull-Rom sin test de geometría por tap; el `prev_z` del rechazo sale del canal `.a` **interpolado por CatRom** (mezcla foreground/background en siluetas) y clampeado por `max(color,0)`.
- **Mejora:** bilineal 2×2 con peso de validez por tap (`w_bilinear · valid(depth,normal)`), renormalizado; z e history_len con `textureLoad` nearest, nunca interpolados; CatRom solo si los 4 taps del footprint son válidos.
- **Impacto:** Alto — causa nº1 de ghosting/halo en bordes.

### C3. Rechazo por normal + plane-distance en el temporal
- **Dónde:** `temporal.wgsl:236-237`.
- **Hoy:** solo `abs(Δz) < max(|z|·0.1, 0.5)`. 10% relativo es muy laxo; el suelo `0.5` depende de la escala de escena; no hay rechazo por normal.
- **Mejora:** guardar normal previa, rechazar con `pow(dot(n_prev,n_cur), k)`; sustituir Δz por distancia al plano `|dot(N, Xprev - X)| < 0.01·frustumSize(z)`.
- **Impacto:** Alto (ghosting en esquinas/geometría fina).

### C4. AntiLag: comparar contra media filtrada + bajar deadzone 💤
- **Dónde:** `temporal.wgsl:269-283`.
- **Estado:** el AABB ya se dejó siempre activo (fix #7). El **deadzone de 0.3 NO se tocó a propósito**: el autor documentó que lo puso para frenar jitter/splotches reales (comparación contra 1-spp ruidoso). Bajarlo a ~0.05-0.1 requiere primero la varianza real (C1) o A/B en device.
- **Impacto:** Medio-alto con luz dinámica; **riesgo de regresión** — no aplicar sin medir.

### C5. Specular ReBLUR: historia propia + roughness en el temporal
- **Dónde:** `temporal.wgsl:286-322`.
- **Hoy:** el specular usa el mismo alpha que el difuso; la historia virtual (`vmb_hist`) se muestrea sin ningún test de validez; sin roughness en el pase.
- **Mejora:** pasar roughness (está en `albedo_tex.a`); `vmb_weight ∝ specMagicCurve(roughness)`; AABB también sobre la historia virtual; alpha specular propio; antilag separado con luma especular.
- **Impacto:** Alto en escenas con especular prominente.

### C6. HistoryFix con stride en disocclusiones grandes
- **Dónde:** `denoise.wgsl:263-300`.
- **Hoy:** píxeles con `history < 8` ponderan vecinos en un 5×5 fijo; en una disocclusion grande todo el 5×5 tiene historia ~0.
- **Mejora:** taps extra con stride `∝ (1 - history/8)` fuera del tile (ReBLUR HistoryFix, alcance ~14px+). Alternativa: la varianza de C1 lo resuelve en gran parte.
- **Impacto:** Medio (parches ruidosos ~5-10 frames tras cada disocclusion).

### C7. Sin AA antes de EASU (FSR)
- **Dónde:** `renderer.js:2169, 2637`; "DLAA" en L158.
- **Hoy:** la proyección raster no jittea ni el reproject lo compensa; FSR1 espera entrada anti-aliased → bordes aliased + RCAS los realza (crawling). "DLAA" es 1× sin ningún AA (nombre engañoso).
- **Mejora:** jitter Halton en la matriz del G-buffer + des-jitter en el temporal (la infra Halton ya existe, L2089), o un resolve tipo TAA sobre el composite antes de EASU. *El orden tonemap→EASU es correcto para FSR1, no cambiarlo.*
- **Impacto:** Alto.

### C8. Motion vectors de objetos
- **Dónde:** `temporal.wgsl:215-227`, `reblur.wgsl` (asume `world_prev = identidad`).
- **Hoy:** reproyección solo de cámara → cualquier objeto animado hace ghosting (el depth-reject solo salva disocclusiones, no movimiento tangencial).
- **Mejora:** canal de velocity en el G-buffer raster. Deuda estructural para escenas dinámicas (hoy Sponza estática es aceptable).
- **Impacto:** Alto si hay animación; N/A hoy.

---

## 💤 Decisiones pendientes

### D1. ReBLUR: ~450 líneas de código muerto
- **Dónde:** `reblur.wgsl` — `temporal_accumulation` (633-750), `history_fix` (756-839), `spatial_filter` (402-544) **nunca se despachan**. La rama `reblur` real usa `prepass → temporal (de temporal.wgsl) → blur → post_blur → stabilization`.
- **Decidir:** (a) cablear `temporal_accumulation` + `history_fix` tras arreglar sus bugs latentes (stores duplicados; `specAccumSpeed` leído del canal difuso; `frameNum` en unidades incorrectas en history_fix `:776`); o (b) borrar las ~450 líneas y asumir el pipeline simplificado.
- **Impacto:** Calidad alto (si se cablea) / mantenibilidad alto (si se borra).

### D2. `DENOISER_BMFR_PLAN.md` desactualizado
- El `bmfr.wgsl` real **no** es BMFR Koskela (bloques 32×32, 10 features con p², subgroupAdd, offset por frame). Es una **regresión lineal local estilo guided filter**: 7 features `[1,n,p]`, momentos por celda 8×8, Cholesky 6×6, upsample bilineal de coeficientes. Enfoque válido; el plan y sus milestones M3/M4 ya no aplican tal cual.

---

## ⏳ Pendiente — BMFR (modo `bmfr`)

| # | Hallazgo | Archivo | Impacto |
|---|---|---|---|
| B1 | **Specular sin filtro espacial** (solo temporal) → glossy ruidoso en movimiento. Mini-Poisson 5-8 taps post-temporal con radio `∝ smc/(1+hist)`. | `bmfr.wgsl:554-617` | Calidad **alto** (eslabón débil del modo) |
| B2 | Reducción de momentos serializada en thread 0 (3136 adds en serie, 63 threads esperando). Reducción en árbol en shared → 5-10×. | `bmfr.wgsl:237-247` | Perf medio-alto |
| B3 | `apply` hace ~100 loads escalares de storage/px. Empaquetar `coefs` como `array<vec4f>` (→28 loads) o volcar a textura. | `bmfr.wgsl:408-456` | Perf medio (alto móvil) |
| B4 | Guard de Cholesky débil en celdas planas (suelo absoluto 1e-7 fijo). Suelo `∝ trace/6` o drop-de-feature por pivote. | `bmfr.wgsl:348,360` | Calidad medio |
| B5 | `depth_reject` declarado pero **sin usar**: reproyección temporal sin test de disocclusión (clamp se apaga con `hist ≥ 16`). | `bmfr.wgsl:36,524-546` | Calidad medio |
| B6 | Reproyección nearest con truncado (`vec2i(puv*res)`); sin bilinear → shimmering/drift en movimiento. | `bmfr.wgsl:150,525,595` | Calidad medio |
| B7 | `spec_temporal` mezcla **UVs** en vez de historias (`mix(surf.xy, virt.xy)`) → muestrea un punto que no corresponde a ningún motion real. Mezclar colores. | `bmfr.wgsl:583-590` | Calidad medio |

---

## Notas de entorno

- **Escena:** `scene/scene.bin` (147 MB) está gitignored y no viene en el clon. Se sirve
  desde Android Downloads vía `SCENE_PATH` en `.env` (gitignored). **El server debe correr
  en Termux real** (`python serve.py`) — solo Termux ve `/storage`.
- **Validación de shaders offline:** wgpu-py + lavapipe (Vulkan software) en el proot Debian.
  `denoise.wgsl` y `fsr.wgsl` necesitan `enable f16;` antepuesto (lo hace `renderer.js:231`
  en runtime; hay que replicarlo al validar suelto).
