# Performance Analysis & Optimization Guide

## Current Performance Profile

### Detection Pipeline (Release Build, 640x640, RTX 4090 / Ryzen 9 7950X)

Measured per stage on `fixtures/images/006.jpg` (2384x4240 -> 640x640) with
`cargo run --release -p fcs-core --example stage_breakdown`, mimalloc enabled as
in the shipped binaries.

| Stage              | tract    | ONNX Runtime | GPU (WGSL) |
|--------------------|----------|--------------|------------|
| JPEG decode        | 20.7 ms  | 20.7 ms      | 20.7 ms    |
| Preprocessing      | 2.9 ms   | 2.9 ms       | on device  |
| Inference          | 65.9 ms  | 6.7 ms       | —          |
| Postprocessing     | <1 ms    | <1 ms        | <1 ms      |
| **`detect_image`** | 69.6 ms  | **10.7 ms**  | **8.2 ms** |
| End-to-end / image | 86.9 ms  | 31.3 ms      | ~29 ms     |

Batch, 20 images through rayon over one shared detector: tract 236 ms,
ONNX Runtime 118 ms. The batch gain (2.0x) is far smaller than the single-image
gain (6.5x) because batch was already parallel across images and becomes memory
bound once inference stops being the constraint — worth remembering before
quoting the single-image ratio at anyone.

### Bottleneck Summary

- **Inference** (76% of end-to-end on tract, 95% of `detect_image`): the one
  stage worth optimising. See the ONNX Runtime backend below.
- **JPEG decode** (~21 ms, and the largest stage once inference drops): looks
  like the obvious next target and is not one. See "What Did Not Work".
- **Preprocessing** (3%): not a bottleneck, despite earlier revisions of this
  document claiming 33%. That figure was wrong and sent at least one
  optimisation hunt at a stage that costs 2.9 ms.
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

### Decoding JPEGs at reduced scale

The arithmetic is tempting: detection runs at 640x640, sources are routinely
2384x4240, so decode appears to produce 93% pixels that are thrown away. JPEG
supports 1/2, 1/4 and 1/8 scale decode natively in the DCT at a fraction of the
cost.

It does not work here, because those pixels are not thrown away — they are the
product. `process_single_image` decodes once and passes the same image to both
detection and `crop_face_from_image`, and the output crop is cut from the
full-resolution pixels. Detection only needs 640x640; cropping needs everything.
Decoding at reduced scale would degrade every crop the app produces. Decoding
twice (scaled for detection, full for cropping) is strictly more work for any
image that actually contains a face, which is most of them.

Two further measurements close the stage off as a target entirely:

- **The decoder is already fast.** zune-jpeg, via `image`, runs at 390-570
  Mpx/s on this hardware. The ~21 ms is simply what 10.1 megapixels costs; it is
  not overhead waiting to be removed.
- **It cannot be parallelised for these files.** Decode is single-threaded —
  identical timings under `RAYON_NUM_THREADS=1` and 32 — and splitting one image
  across threads requires restart markers to give independent entry points into
  the entropy-coded stream. None of the fixtures have any (no DRI segment, zero
  RST markers): baseline sequential JPEG is one continuous Huffman run, so no
  MCU can be decoded without decoding every MCU before it.

Batch throughput is unaffected either way, since whole images already decode in
parallel across rayon workers. This is a single-image latency figure only.

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
| `ort` DirectML/CoreML EPs     | None expected   | Measured: ties the WGSL path already shipped; see below    |
| Wider CPU SIMD (`wide` crate) | None expected   | Tried and reverted; see "What Did Not Work"                |
| macOS/Linux GPU testing       | Validation only | Metal and Vulkan paths exist; untested on hardware        |
| CLI GPU map/poll latency      | ~20ms           | Staging buffer strategy; deferred                         |

Figures in the "Est. Gain" column are measured only where a row says so. "Unmeasured" means no
benchmark exists in this repo for that idea — treat those rows as directions to investigate, not
as predictions.

### The ONNX Runtime CPU backend (shipped in fcs-core)

`tract` lowers YuNet's 3x3 depthwise convolutions to a scalar fallback — it only
unrolls zones with at most 4 taps — which 1.5.3 measured at 59-61% of a CPU
detection and concluded was "not reachable from this side". That was true within
tract. Swapping the runtime reaches it: ONNX Runtime vectorises those
convolutions and runs the same graph in 6.7 ms against 65.9 ms.

Scope deliberately excludes DirectML. Measured end to end it lands at ~9.8 ms
against the 8.2 ms the WGSL graph already achieves with no dependency at all, so
the GPU execution providers buy nothing here. Only the CPU EP is used, which
also means one library (20.1 MB) rather than the ~38 MB a DirectML build needs,
and no per-platform execution-provider matrix.

Loaded dynamically rather than linked: the prebuilt static library is built
against the dynamic CRT and collides with the workspace's `+crt-static`, and
dynamic loading keeps DirectML out of the build entirely.

The binding lives in `fcs-ort` and is ours — the `ort` crate is no longer a
dependency. `fcs-core/src/ort_backend.rs` only adapts between tract tensors and
that binding. Two things to know before touching either:

1. **`sys::OrtApi` is a hand-maintained prefix of a 424-entry function table.**
   A wrong offset is undefined behaviour, not a compile error. Read the rules in
   `fcs-ort/src/sys.rs` before adding a field, and note that generated bindings
   declare some fields twice under `#[cfg]` — counting those duplicates shifts
   every later offset. Two guards exist: a `const` assertion that the struct is
   exactly one pointer per declared field, and `fcs-ort/tests/end_to_end.rs`,
   which calls through offsets 3 to 100 against a real runtime.
2. **A bad runtime must never reach the FFI.** `fcs_ort::locate` reproduces the
   version rule and adds an ABI check, because a machine with an unrelated
   `onnxruntime.dll` on PATH will otherwise be resolved and used. This mattered
   more when `ort` was in the picture — its version rejection aborted the
   process — but a mismatched library is still worth refusing outright.

Sessions are shared, not pooled: ONNX Runtime permits concurrent `Run` on one
session, which `concurrent_runs_match_sequential` verifies against eight
threads. `SessionOptions::intra_threads` defaults to 1 because parallelism comes
from rayon running whole images at once.

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
