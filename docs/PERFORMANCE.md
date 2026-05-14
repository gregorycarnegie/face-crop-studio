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
- **Enhancement**: LUT/SIMD pass achieved −10.9% (895ms→798ms); GPU shaders give larger gains.
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
| Saturation `wide::f32x4` SIMD    | Included in above        |
| `mul_add` audit across hotspots  | Minor precision + perf   |

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

---

## Future Opportunities

| Opportunity                   | Est. Gain       | Notes                                                     |
|-------------------------------|-----------------|-----------------------------------------------------------|
| INT8 model quantisation       | 2–4× inference  | Requires Python ONNX tooling; see ONNX_RUNTIME_OPTIONS.md |
| `ort` crate (DirectML/CoreML) | 2–5× inference  | Adds ~160MB runtime; see ONNX_RUNTIME_OPTIONS.md          |
| macOS/Linux GPU testing       | Validation only | Metal and Vulkan paths exist; untested on hardware        |
| CLI GPU map/poll latency      | ~20ms           | Staging buffer strategy; deferred                         |
