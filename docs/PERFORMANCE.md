# Performance Analysis & Optimization Guide

## Current Performance Profile

### Detection Pipeline (Release Build, 640×640, GTX 1080 Ti / Vulkan)

| Stage              | CPU-only   | GPU-enabled | Notes                                 |
|--------------------|------------|-------------|---------------------------------------|
| Model loading      | 0.11–0.29s | 0.11–0.29s  | Cached after first load               |
| Preprocessing      | ~43ms      | ~51ms¹      | Criterion: CPU 162ms, GPU 51ms        |
| ONNX inference     | ~82ms      | N/A²        | Custom WGPU inference available       |
| Postprocessing     | <1ms       | <1ms        | Grid-based NMS, already optimal       |
| Enhancement (full) | ~798ms     | GPU shaders | Criterion: 895ms→798ms after LUT/SIMD |

¹ CLI `--benchmark-preprocess` GPU path is currently bottlenecked by host map/poll latency;
  Criterion GPU benchmark (device-resident) measures ~51ms.  
² Custom WGPU GPU inference (Phase 12) available via `--gpu-inference`; timing varies by image
  size and driver scheduling.

### Bottleneck Summary

- **Preprocessing** (33%): rayon, buffer pooling, GPU preprocessing all implemented.
- **Inference** (52%): tract CPU baseline ~82ms; custom WGPU inference reduces this further.
- **Enhancement**: LUT + autovectorisation pass achieved −10.9% (895ms→798ms); GPU shaders give
  larger gains. Hand-written SIMD is not the lever here — see "What Did Not Work".
- **Postprocessing** (<1%): spatial grid NMS, optimal.

---

## Optimisations Implemented

### Phase 1 — Quick Wins

| Optimization                  | Impact                   |
|-------------------------------|--------------------------|
| `image` crate `rayon` feature | 2–4ms on first load      |
| App-level image cache (GUI)   | 34ms saved per cache hit |
| tract already uses rayon pool | Baseline (no change)     |

### Phase 2 — Medium Effort

| Optimization                                | Impact                     |
|---------------------------------------------|----------------------------|
| Thread-local preprocessing buffer pool      | 2–5ms per detection        |
| GPU preprocessing (`WgpuPreprocessor`)      | 20–25ms when GPU available |
| Conditional resize bypass (`Cow<RgbImage>`) | 15–20ms for 640×640 inputs |

### Phase 11 — Enhancement Pipeline

| Optimization                     | Impact                   |
|----------------------------------|--------------------------|
| Skin smoothing rayon parallelism | 4.25s → 116ms (36×)      |
| Background blur rayon + no-sqrt  | 182ms → 136ms (25%)      |
| Exposure/brightness/contrast LUT | Criterion: 895ms → 798ms |
| Plain ops + saturating cast in hot loops (replaced `mul_add`/`round`/`wide::f32x4`; see below) | Saturation 7.6→6.2ms, skin 8.0→2.3ms, unsharp 4.9→1.9ms |
| `target-cpu=x86-64-v3` on x86_64  | Lets LLVM autovectorise the above |

### Phase 12 — GPU Acceleration

| Optimization                      | Status     | Impact                                      |
|-----------------------------------|------------|---------------------------------------------|
| GPU preprocessing (WGSL shader)   | ✅ Shipped | CPU ~162ms → GPU ~51ms (Criterion)          |
| GPU enhancement pipeline          | ✅ Shipped | All filters have WGSL kernels               |
| Custom WGPU YuNet inference graph | ✅ Shipped | Conv2D/BN/Activation on GPU, parity-tested  |
| GPU batch crop extraction         | ✅ Shipped | Parallel crop regions as GPU draw calls     |
| GPU buffer/texture pool           | ✅ Shipped | Avoids repeated allocation overhead         |

---

## Benchmark Infrastructure

```bash
# CPU vs GPU preprocessing (Criterion)
cargo bench -p fcs-core --bench preprocessing

# Lightweight CLI benchmark over an image set
cargo run -p fcs-cli -- --input fixtures/images --benchmark-preprocess

# Full pipeline example
cargo run --release --example profile_pipeline -p fcs-core

# GPU/CPU parity validation
cargo test -p fcs-core gpu_inference_matches_cpu_baseline -- --nocapture
```

Criterion results are written to `target/criterion/`. Do not commit benchmark output text files.

---

## What Did Not Work

### Nested loop parallelisation in `decode_yunet_outputs`

Parallelizing the inner `row`/`col` loops within each stride added overhead rather than saving
time. Thread coordination cost exceeded the per-row work. The existing stride-level parallelism
(3 parallel tasks for strides 8/16/32) is the right granularity for this workload.

### Hand-written SIMD via the `wide` crate

`wide::f32x4` was used for the saturation pass in Phase 11 and removed again in `616ffa0`: the
plain scalar loop benched **18% faster**, so the dependency was dropped. Two reasons it lost, both
still true:

- The build now sets `target-cpu=x86-64-v3` on x86_64, so LLVM autovectorises the scalar loops at
  AVX2 width. Hand-written `wide` has to beat the autovectoriser, not scalar code.
- Every remaining hot loop is either an interleaved-pixel deinterleave (RGBA saturation, RGB→BGR
  CHW) or a table lookup (tone LUTs, histogram equalisation, the bilateral colour LUT). Those need
  `pshufb`/`vgather`, which `wide` deliberately does not expose — it is an elementwise-lanes crate.
  The loops `wide` *could* express are already autovectorised, or live inside `tract` and
  `fast_image_resize`, which are hand-vectorised already.

Related: `f32::mul_add` and `f32::round` were removed from the same loops. On the pre-v3 SSE2
baseline both compiled to software libm calls; `mul_add` stayed slower even with FMA enabled,
because its single-rounding guarantee blocks reassociation. The hot loops use plain `a * b + c`
and a saturating `(x + 0.5) as u8` cast instead, marked with `ponytail:` comments at each site.

---

## Future Opportunities

| Opportunity                   | Est. Gain       | Notes                                                     |
|-------------------------------|-----------------|-----------------------------------------------------------|
| INT8 model quantisation       | Unmeasured      | Independent of `ort` — `tract` already parses QDQ graphs; see below  |
| `ort` crate (DirectML/CoreML) | Unmeasured      | Adds ~160MB runtime; see ONNX_RUNTIME_OPTIONS.md          |
| Wider CPU SIMD (`wide` crate) | None expected   | Tried and reverted; see "What Did Not Work"                |
| macOS/Linux GPU testing       | Validation only | Metal and Vulkan paths exist; untested on hardware        |
| CLI GPU map/poll latency      | ~20ms           | Staging buffer strategy; deferred                         |

Figures in the "Est. Gain" column are measured only where a row says so. "Unmeasured" means no
benchmark exists in this repo for that idea — treat those rows as directions to investigate, not
as predictions.

### INT8 quantisation is not the `ort` question

These are often conflated. They are separate decisions:

- **`ort`** replaces the inference runtime and adds a ~160MB shared-library dependency. That is
  the subject of ONNX_RUNTIME_OPTIONS.md.
- **INT8** is a change to the *model file*, and needs no runtime change at all. `tract-onnx`
  already registers `QuantizeLinear`, `DequantizeLinear`, `QLinearConv` and `QLinearMatMul`, and
  `tract-linalg` ships x86_64 i8 GEMM kernels (`avx2_mmm_i32_8x8`, `avxvnni_mmm_i32_8x8`,
  `avx512vnni_mmm_i32_*`) selected by runtime CPUID. Quantising is a one-off offline step whose
  output is a committed `.onnx`, exactly like `face_detection_yunet_2023mar_640.onnx` — nothing
  extra ships and the pure-Rust build story is unchanged.

No gain figure is quoted above because none has been measured here. Three things would decide it,
and they are listed in the order that kills the idea cheapest:

1. **Does the graph still load?** This model already needed a fixed-shape re-export to satisfy
   `into_optimized()` at all (see models/README.md). A quantised export may not survive it.
2. **Are the i8 kernels even on the critical path?** YuNet is a depthwise-separable backbone
   (53 convs, all carrying a `group` attribute). Depthwise convolution does not lower to GEMM
   cleanly, so the i8 GEMM kernels above may contribute little.
3. **What does it cost in recall?** Post-training quantisation on a small detector loses
   detections, and the quality thresholds are tuned against f32 behaviour.

Two further costs are easy to overlook. INT8 gains depend strongly on the CPU: with AVX512-VNNI
(Zen 4, Cascade Lake+, Alder Lake+) `VPDPBUSD` gives a 4-way i8 dot per lane, but on the shipped
`x86-64-v3` baseline the AVX2 fallback computes i8 products in i16 lanes — the same lane count as
f32 FMA, so the win is cache footprint rather than arithmetic. Benchmarking only on a VNNI
developer machine will overstate what most users get. And the WGSL GPU inference path gains
nothing, so an INT8 CPU path diverges numerically from the f32 GPU path that
`gpu_inference_matches_cpu_baseline` and docs/parity_report.md compare against.
