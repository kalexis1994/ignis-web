# Ignis Web — WebGPU Monte Carlo Path Tracer

Real-time path tracer running entirely in the browser via WebGPU. Renders the Sponza atrium (3.7M triangles) with physically-based materials, global illumination, and temporal stability.

## Features

### Rendering
- **Monte Carlo path tracing** with configurable bounces (1-4) and samples per pixel
- **Hybrid raster G-buffer** replaces primary ray BVH traversal (~60-80% faster)
- **Cook-Torrance GGX BRDF** with real dielectric Fresnel, VNDF sampling (Heitz 2018)
- **Perceptual √ Russian Roulette** (NRD guideline, min 0.05 survival)
- **Separate glass bounce budget** (up to 16, doesn't consume main bounces)
- **Stochastic alpha blending** for BLEND materials (decals, foliage)
- **R2 quasi-random sub-pixel jitter** for temporal super-resolution

### Ignis SVGF Denoiser
Hybrid denoiser combining [SVGF](https://research.nvidia.com/publication/2017-07_Spatiotemporal-Variance-Guided-Filtering:) (Schied 2017) with [NRD/ReBLUR/ReLAX](https://github.com/NVIDIA-RTX/NRD) (Zhdan 2021) techniques:

- **Dual-signal pipeline**: separated demodulated diffuse irradiance + specular radiance
- **À-trous wavelet filter**: 5×5 B3-spline kernel, 3-5 passes (σ_n=128, σ_l=4.0×√var)
- **Variance filtering**: propagated through à-trous passes (SVGF §4.2)
- **Per-pixel history length**: replaces global frame counter for adaptive temporal/spatial balance
- **Depth-based disocclusion**: camera-space Z comparison rejects invalid history
- **Pre-blur pass**: 3×3 bilateral with anti-firefly 3σ percentile clamp ([HPG 2025](https://www.lalber.org/2025/06/percentile-based-adaptive-svgf/))
- **Hit distance blur radius**: contact shadows stay sharp, far GI gets smoothed (ReBLUR hitT)
- **Roughness-dependent specular sigma**: tight for glossy, wide for rough surfaces (ReLAX)
- **AABB clip** (Salvi) with adaptive expansion for temporal accumulation
- **Composite**: albedo × diffuse + specular → configurable tone mapping

### Global Illumination
- **SHaRC radiance cache**: spatial hashing with backpropagation (4 bounce points per path)
- **ReSTIR GI**: temporal radiance reuse via Weighted Reservoir Sampling (Talbot et al.)
- Desktop-only (requires ≥10 storage buffers per shader stage)

### Tone Mapping & Color
7 configurable tone mapping operators:
- **AgX Punchy** (Blender 4 / Troy Sobotka)
- **ACES** (Narkowicz 2015 fit)
- **Reinhard** (luminance-preserving)
- **Uncharted 2** (Hable filmic)
- **[Khronos PBR Neutral](https://github.com/KhronosGroup/ToneMapping)** (2024, true-to-life)
- **Standard** (linear clamp, default)
- **None** (debug)

Plus exposure, saturation, contrast controls and triangular dither.

### Upscaling
- **FSR3-inspired** EASU (Lanczos 2-lobe) + RCAS (contrast-adaptive sharpening)
- Modes: Performance (2×), Balanced (1.7×), Quality (1.5×), DLAA (1×)

### Platform Support
- **Desktop**: NVIDIA (Ada/Ampere/Turing), AMD (RDNA), Intel Arc
- **Mobile**: Qualcomm Adreno (8 storage buffer limit respected)
- **20+ GPU profiles** with auto-detection and tailored settings

## Architecture

```
Path Tracer (1SPP) → Pre-blur (3×3 bilateral + anti-firefly)
    ↓
Temporal Reprojection (per-pixel history, AABB clip)
    ↓
Spatial Denoise (à-trous wavelet × 5 passes, dual-signal)
    ↓
Composite (albedo × diffuse + specular → tonemap → gamma → contrast → dither)
    ↓
FSR Upscale (EASU + RCAS) → Display
```

## References

- Schied et al., *Spatiotemporal Variance-Guided Filtering*, 2017
- Zhdan, *ReBLUR: A Hierarchical Recurrent Denoiser*, 2021
- Heitz, *Sampling the GGX Distribution of Visible Normals*, 2018
- Dammertz et al., *Edge-Avoiding À-Trous Wavelet Transform*, 2010
- Talbot et al., *Importance Resampling for Global Illumination*, 2005
- Lalber, *Percentile-based Adaptive SVGF*, HPG 2025
- Khronos Group, *PBR Neutral Tone Mapper*, 2024
- Narkowicz, *ACES Filmic Tone Mapping Curve*, 2015

## Running

```bash
python serve.py [port]  # default 8080
```

Requires Chrome/Edge with WebGPU support. Scene files in `scene/` directory.
