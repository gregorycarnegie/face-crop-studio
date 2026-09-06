# Performance Analysis & Optimization Guide

## Current Performance Profile

### Latest GPU results (2026-09-05)

The completed shader experiments reduced full-graph GPU compute from **0.910 ms
to about 0.536 ms (41% less time)** on RTX 4090 / D3D12 at 640x640. Three f32
changes were retained: pointwise specialization, depthwise input reuse, and a
four-output-channel pointwise tile. FP16 storage and subgroup reduction were
measured but not adopted. These changes are in the working tree at the time of
this report; the experiment baseline is commit `c332075`.

The final whole-detection A/B/B/A estimates were **3.561 / 3.504 / 3.495 /
3.463 ms**, with overlapping intervals. They establish **no reliable
whole-detection speedup** from this shader round. This Criterion case is
`inference_pipeline/detect_image/gpu`: CPU speed resize followed by GPU inference,
excluding image decode. It is distinct from `gpu_on_device` and `gpu_quality`.
No new batch-throughputput measurement was made.

The [GPU experiment results](#gpu-experiment-results) below consolidate
the findings; [experimentation.md](../experimentation.md) retains the sequential
checklist, individual runs and reproduction commands. Finishing that checklist
does **not** mean the performance search is exhausted.

### Where warm detection actually spends its time (2026-09-05, later round)

Phase timings from `cargo run --release -p fcs-core --example phase_timings`,
RTX 4090 / D3D12, warm, single request. Indentation is containment; children
must not be summed with their parent, and `readback_wait` **contains** the
forward pass rather than idling beside it.

Two different paths, chosen by source size. `upload_pays_for_source` declines
above 2.5 MP, so a large photo preprocesses on the CPU and a small one does not.

| Phase | 0.17 MP | 10 MP |
| --- | ---: | ---: |
| detect_image | 1.57 | 4.68 |
| - CPU preprocess (large only) | - | **2.65** |
| - on-device preprocess (small only) | ~0.21 | - |
| - inference | 1.35 | 1.97 |
| - - record (host, before submit) | 0.21 | 0.28 |
| - - finish + submit | 0.13 | 0.16 |
| - - readback (incl. GPU execution) | 0.79 | 1.04 |
| - - CHW to HWC + sigmoid | 0.11 | 0.11 |
| - - decode | 0.19 | 0.19 |

**Large-image detection is a preprocessing problem, not a shader problem.**
The ~0.54 ms of profiled GPU compute is a minority of even the small-image
case, and on a 10 MP source CPU preprocessing alone is 57% of the whole
detection.

**Host work between the inference submit and the readback wait is free.** The
GPU is busy through that window, so shortening it does not shorten detection:
removing the staging allocation shed 0.039 ms and the wait grew by 0.046 ms.
Moving work *into* that window helps, and moving work out of it hurts -
encoding the head copies with inference cost 0.09 ms. The recoverable time is
before the submit, in the GPU work itself, and after the wait.

**Measurement caveats that invalidated earlier attempts.** Cross-process
comparison cannot resolve anything at this scale: repeated identical GPU runs
drift +/-0.08 ms as clocks ramp, and CPU throughput on this machine moved by
**1.6x between two builds of identical code**. Both are handled by alternating
variants inside one warm process (`phase_timings --ab VAR`), whose A/A control
sits within +/-0.003 ms per phase.

### The biggest saving found is switching cropping off the GPU

`GpuBatchCropper::crop` converts the full-resolution source to RGBA, packs every
pixel to `u32` and uploads all of it -- 40 MB for a 10 MP photo -- to produce one
512x512 crop. It accounts for 11% of batch CPU through `memcpy` alone. Forcing
the CPU path takes a 1239-image folder from a median **17.25 s to 9.8 s, 43%**,
order-alternated and winning every pair.

Adopted, and `GpuBatchCropper` deleted with it -- about 740 lines. The saving
holds for the shaped crops too (38% rectangle, 47% koch snowflake, 38% star),
since the mask costs the same on both sides and lands on the finished 512x512
image. Crops changed: `crop.wgsl` sampled a fixed 2x2 neighbourhood where the
CPU uses Lanczos3, so 18% of crops shifted quality label and one image in 171
selects a different face. See experiment 71.

### The batch path is not serialised

Profiled over 1239 real images, all threads: **194.7 s of CPU across 51 threads
in about 16 s of wall time**, roughly 12.5 cores busy on a 16-core machine, with
the top ten workers within 15% of each other. Nothing -- not the GPU queue, not
the buffer-pool mutex, not export -- serialises the batch, so experiments 60, 61
and 64 have no contention to fix. Warm throughput is about 65 images/s.

Two hazards for anyone measuring this path. `fcs-cli` loads
`config/gui_settings.json` **from the working directory**, so the same command
detects 1020 faces from the repository root and 423 from `target/release`;
always pass `--config`. And batch wall time swings about 10% run to run, so
comparisons need their order alternated -- see experiment 63, where an
unalternated pair produced a confident conclusion that reversed under
alternation.

### Decode, not detection, sets what a folder costs

Warm, from memory, 10 MP fixture: **JPEG decode 24.0 ms against 3.1 ms of
`detect_image`**. Every remaining item in the detection path is small against
that, so experiments 67 and 68 outrank them for folder work. Interactive
single-image latency is the case where the detection numbers below still
dominate.

libjpeg-turbo, already linked in through `nokhwa`, decodes the same corpus
**1.23x faster** (517 ms vs 420 ms over 15 images of 8-22 MP) and is now the
default for `.jpg`/`.jpeg`. Decoded pixels differ from the previous decoder by
up to 5/255, which moves detection boxes 0.28-1.30 px and an exported crop by
about a pixel; the crops were reviewed before adopting it, and the CLI JSON
snapshot was updated by at most 0.26 px. See experiment 67.

A build issue found along the way and fixed: `mozjpeg-sys` silently compiles
`jsimd_none.c` when NASM is absent, and the Windows and macOS release legs did
not install it, so released binaries decoded webcam MJPEG frames ~2.2x slower
than they should. Both legs now install NASM and the Windows leg fails if it is
not on `PATH`.

### The source resize is now a third of a folder job

After the changes above, a 1239-image folder spends 112 s of CPU, and
`resize_image` on the way into the detector is 32.3% of it. `detect_image` is the
same 32.3%, because a GPU inference wait costs no CPU: detection *is* the resize.

Feeding the detector a reduced-scale JPEG decode instead was measured and
rejected (experiment 90). The arithmetic works -- a 1/2 or 1/4 DCT decode costs
less than the resize it removes, about 13% off the folder -- but it moves
landmarks by up to 158 px, and backing off to where nothing moves past 35 px
leaves about 4%. `examples/scaled_decode.rs` re-runs the cost half.

### The resize is at its floor

After the changes above, the CPU source resize is the largest single cost in
detecting a large image -- 1.39 ms at 10 MP, 2.78 ms at 22 MP -- and experiment
51 establishes that it cannot be made faster without changing what is detected.
`SuperSampling` measured **slower** (its nearest-neighbour first pass still
reads every source pixel, then adds a convolution over the intermediate).
`Interpolation` is 0.7-2.0 ms faster precisely because its fixed two-tap kernel
reads about four source pixels per output instead of the ~14 the ratio calls
for, and it moved landmarks by up to **35 px** over the fixture corpus.

At a fixed source resolution the resize is bounded below by reading the source
once, which production's adaptive-kernel convolution already does. The remaining
lever is fewer source pixels -- a decode-stage change (experiment 86), not a
resize-stage one.

`examples/resize_quality.rs` is what makes that kind of question decidable: it
runs the detector twice over the corpus and reports faces lost or gained,
landmark displacement in source pixels, box IoU and score deltas, against an
A/A control that reads exactly 0.00 px and IoU 1.0000.

### Changes retained in this round

| Change | Effect | Where |
| --- | --- | --- |
| Upload resized bytes and convert on the GPU | **-0.6 to -1.0 ms** on large images | `preprocess.rs`, `rgb_to_chw.wgsl` |
| Threaded source resize above 4 MP | **-0.6 ms** on large images | `fcs-utils/src/image_utils.rs` |
| No one-thread-pool hop inside a rayon worker | **-18%** of batch wall time | `fcs-utils/src/image_utils.rs` |
| RGBA fast resize for crops | -2% of batch wall time | `image_utils.rs`, `face_cropper.rs` |
| RGBA fast resize for the quality metric | **-9%** of batch wall time | `quality.rs` |
| Borrow rather than clone to encode an export | below the noise floor; one less copy | `output/encoders.rs` |
| One `fir::Resizer` per thread | -0.1 to -0.15 ms on large images | `fcs-utils/src/image_utils.rs` |
| Stop zeroing the BGR/CHW buffer | -0.1 to -0.2 ms on large images | `fcs-utils/src/image_utils.rs` |
| Decode straight from the GPU's channel-major heads | **-0.15 to -0.2 ms** | `model.rs`, `gpu/runtime.rs` |
| `Tensor::from_vec` instead of copying | -0.02 to -0.03 ms | `gpu/runtime.rs`, `model.rs` |
| One readback poll instead of two | no speed change; less code | `gpu/runtime.rs` |

All are bit-exact except the RGBA crop resize, which swaps one Lanczos3
implementation for another and so differs by rounding -- at most 23 per channel
over a 1239-image folder, one crop in 901 shifting quality label (experiment 88).
The rest: the raw 126000-float output fingerprint is unchanged
(`readback_parity` probe), resize output is byte-identical between one thread
and many (`threaded_and_single_threaded_resize_agree`), and the full workspace
suite passes under `FCS_STRICT_TESTS=1` with ONNX Runtime 1.24.4.

### Historical stage breakdown (before the latest GPU optimizations)

Release build, 640x640, RTX 4090 / Ryzen 9 7950X. The following measurements
belong to an earlier revision and are retained for CPU/decode context. The GPU
8.2 ms figure is not the current baseline; do not combine this table with the
latest shader timings to calculate stage shares or gains.

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

- **Current GPU inference:** convolution still dominates GPU compute, but shader
  time is only one part of detection. Measure CPU recording, submission, buffer
  allocation, synchronization, readback and output conversion separately before
  selecting the next latency change.
- **Remaining shader budget:** about 0.536 ms in this measured profiled graph.
  Halving all shader time would remove about 0.268 ms of GPU work; eliminating
  it entirely would remove about 0.536 ms. These are fixed-workload arithmetic
  ceilings, not promised wall-time savings. Profiling uses separate passes;
  normal inference uses one merged pass, so validate gains in the normal path.
- **Full-resolution image processing:** JPEG decode was about 21 ms in the
  historical fixture measurement and remains a separate workload from
  `detect_image`. Reduced-resolution decode does not preserve crop quality.
- **Batch and interactive work:** CPU parallelism, GPU queue occupancy and
  latency overlap differ. Remeasure the actual export or webcam/preview path;
  a single-image shader result cannot select the fastest batch backend.
- **Preprocessing and postprocessing:** now measured, not assumed. CPU
  preprocessing is 2.65 ms of a 4.68 ms detection at 10 MP and is the single
  largest cost in the application; the resize inside it is now threaded above
  4 MP, and unconditionally when a rayon worker is already running it -- holding
  it to one core there means a cross-registry `install()` hop that cost 18% of
  batch wall time (experiment 88). Postprocessing is 0.005-0.010 ms and is not worth attention. Output
  conversion and decode together are about 0.3 ms and sit on the critical path
  after the readback wait -- now about 0.08 ms, since the CHW-to-HWC transpose
  the decoder used to require has been removed rather than optimised.

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

### GPU experiment results

Baseline `c332075` already contains convolution uniform caching and the merged
compute pass. The latter previously saved about 0.40 ms in paired
encode/finish/submit/wait measurements while retaining all 61 per-op profiling
records. That CPU/driver saving is separate from the shader results below.
See [CHANGELOG.md](../CHANGELOG.md) for those earlier measurements.

| Experiment | Paired full-graph GPU time, A/B/B/A | Decision |
| --- | --- | --- |
| 0. Timestamp families and genuine shader A/B probe | Baseline about 0.910 ms | Keep measurement tools; retire duplicate standard/vec4 benchmark labels |
| 1. Specialized 1x1 convolution | 0.909 / 0.673 / 0.672 / 0.910 ms | Keep; about 0.237 ms saved |
| 2. Depthwise 3x3 overlapping-input reuse | 0.673 / 0.648 / 0.646 / 0.674 ms | Keep; about 0.027 ms saved beyond step 1 |
| 3. Four-channel pointwise register tile | 0.647 / 0.535 / 0.538 / 0.651 ms | Keep; about 0.113 ms saved beyond step 2 |
| 4a. FP16 storage, f32 accumulation | Microbenchmarks only; eight cases 3-15% slower, two unchanged | Reject this candidate; no full-graph trial warranted |
| 4b. Subgroup channel reduction | Microbenchmarks only; large cases 2.9-9.9x as slow, some small cases faster | Reject as a general replacement; selective use remains untested |

Each A/B/B/A row compares the previous retained version with that experiment's
candidate. The savings are rounded from separate comparisons; they are not
independent percentages to add. The final comparison with the original baseline
is about 41% less GPU time, not 41% less detection time.

Pointwise specialization requires a 1x1 kernel, unit stride, zero padding and
one group. The four-channel tile shares each loaded four-pixel input vector
across four accumulators; dispatch-z rounds up and guards channel tails.
Depthwise specialization requires a 3x3 kernel, unit stride, pad 1 and channel
multiplier 1. It loads six values per row for four adjacent outputs, expressing
18 distinct input loads rather than 36. Other configurations retain the general
shader. Fused activations and the shared graph/profiling path are preserved.

A representative final profiled graph breaks down as follows (20 runs; values
rounded, so sums can differ slightly):

| Family | Dispatches | Before shaders | After shaders | Current GPU share |
| --- | ---: | ---: | ---: | ---: |
| Pointwise | 26 | 673.8 us | 320.5 us | 60.0% |
| Depthwise | 26 | 145.4 us | 123.9 us | 23.2% |
| General/stem | 1 | 42.0 us | 41.0 us | 7.7% |
| Pool/resize/add | 8 | 49.1 us | 49.2 us | 9.2% |
| **Total** | **61** | **about 910 us** | **about 535 us** | **100%** |

The reusable shader probe checks finite raw outputs, warms up for 20 pairs,
then measures 50 pairs with alternating order in one process. Compilation,
upload, validation readback and timestamp resolution are outside its GPU timer.
Identical-file controls established roughly 1.024 us timestamp granularity;
one-tick changes on tiny layers are weak evidence. Full-graph runs are separate
binary A/B/B/A comparisons. The final Criterion latency check used 3 seconds of
warm-up and 30 samples over at least 6 seconds per run; historical cached
Criterion change percentages were not used for the conclusion.

Final validation: **823 strict workspace tests plus two doctests passed**, with
eight workspace tests and five doctests skipped/ignored. Coverage included
ONNX raw-output and detection parity, profiled/merged equality, concurrent
inference, spatial/channel tails, exposed activations and stride/pad/group
fallbacks. Workspace and core all-target Clippy, formatting and diff checks
passed. This is validation on the tested hardware, not cross-adapter performance
validation.

---

## Benchmark Infrastructure

```bash
# CPU vs GPU preprocessing (Criterion)
cargo bench -p fcs-core --bench preprocessing

# Lightweight CLI benchmark over an image set
cargo run -p fcs-cli -- --input fixtures/images --benchmark-preprocess

# Full pipeline example
cargo run --release --example profile_pipeline -p fcs-core

# Full-graph GPU timestamp families (profiling path)
cargo run --release -p fcs-core --example gpu_pass_breakdown

# Actual shader A/B, current four-channel pointwise grid on both sides
cargo run --release -p fcs-core --example conv2d_experiment -- fcs-core/src/gpu/conv2d.wgsl fcs-core/src/gpu/conv2d.wgsl 32 8 4 32 8 4

# Paired separate/merged pass encoding costs
cargo run --release -p fcs-core --example gpu_encode_comparison

# Wall-clock phase breakdown of one detection, and in-process A/B of a flag
cargo run --release -p fcs-core --example phase_timings [image]
cargo run --release -p fcs-core --example phase_timings [image] --ab SOME_ENV_FLAG

# Bit-exact fingerprint of the raw head outputs, for readback changes
cargo run --release -p fcs-core --example readback_parity

# Whether threading the source resize pays, at several megapixel counts
cargo run --release -p fcs-core --example resize_threading

# Whole detection; keep gpu, gpu_on_device and gpu_quality results separate
cargo bench -p fcs-core --bench inference_pipeline

# GPU/CPU parity validation (set FCS_STRICT_TESTS=1 to reject missing prerequisites)
cargo test -p fcs-core gpu_inference_matches_cpu_baseline -- --nocapture
```

Criterion results are written to `target/criterion/`. Do not commit benchmark output text files.

---

## What Did Not Work

### FP16 storage and a general subgroup replacement

The default D3D12 context selected FXC and exposed neither optional shader
feature. Making Windows SDK DXC 1.9.2602.17 (SDK 10.0.28000.0) available on the
probe's process-local PATH enabled both SHADER_F16 and SUBGROUP on the same
RTX 4090. Lack of compiler support was not lack of physical GPU support.

The FP16 candidate packed input, weights and bias to f16, accumulated in f32,
and returned f32. Its maximum synthetic raw error was 0.000824, within the
probe's absolute 1e-3 screening budget. It was slower even with packing excluded
from timing, so it did not advance to full-graph or detection parity testing.
This rejects that storage/conversion strategy, not every FP16 arithmetic, packed
layout or mixed-precision implementation.

The subgroup candidate reduced input channels across one 32-lane subgroup for
four output pixels. Raw f32 comparisons passed, but 320x320 16 -> 16 cost
201.728 us versus 20.480 us for the f32 tile under DXC. Small layers did improve:
20x20 64 -> 64 fell from 29.696 to 8.192 us. It was not suitable across the graph.

DXC also slowed the retained f32 tile: 160x160 64 -> 64 measured 67.584 us
versus 27.648 us under the existing FXC setup. A whole-context compiler switch
would sacrifice large-layer performance to enable the small-layer subgroup
candidate. Selective kernels need a compiler/deployment solution and a full-graph
measurement before adoption. Other subgroup algorithms have not been exhausted.

The standalone [FP16](../fcs-core/examples/shaders/pointwise_f16.wgsl) and
[subgroup](../fcs-core/examples/shaders/pointwise_subgroup.wgsl) probes and exact
commands remain in [experimentation.md](../experimentation.md). Neither adds a
production feature requirement.

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

Two further measurements explain why reduced-scale or within-image parallel
decode was not pursued for these fixtures:

- **The decoder is already fast.** zune-jpeg, via `image`, runs at 390-570
  Mpx/s on this hardware. The ~21 ms is simply what 10.1 megapixels costs; it is
  not evidence of avoidable overhead. No faster decoder has been established here.
- **It cannot be parallelised for these files.** Decode is single-threaded —
  identical timings under `RAYON_NUM_THREADS=1` and 32 — and splitting one image
  across threads requires restart markers to give independent entry points into
  the entropy-coded stream. None of the fixtures have any (no DRI segment, zero
  RST markers): baseline sequential JPEG is one continuous Huffman run, so no
  MCU can be decoded without decoding every MCU before it.

Whole images already decode in parallel across rayon workers. Any new decoder
or scheduling strategy needs a separate batch measurement; the figures above
are single-image latency measurements.

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

**No, the avenues are not exhausted.** This round tested a small set of kernels
on one GPU/backend/compiler configuration. It did not sweep workgroup sizes,
implement cooperative workgroup tiling, fuse adjacent layers, or redesign
readback and frame scheduling. The following is a ranked investigation list,
not a promise that each idea will win.

The full [experiment backlog](../experimentation.md#remaining-experiments) assigns
stable IDs, prerequisites and acceptance checks to 81 further experiments. Start
with phase measurements (5), then the preliminary readback wait (11); the table
below is the shorter priority summary.

| Priority | Next experiment | Evidence and acceptance condition |
| --- | --- | --- |
| 1 | Measure and simplify head readback | `batch_download` still creates 12 staging buffers, submits a second command buffer, waits, starts 12 maps, then waits again. Measure those components; try mapping before the first wait, then pooled/packed staging separately. Require raw-head equality and concurrent-inference safety. |
| 2 | Sweep small f32 workgroup and tile choices | Only 1/2/4 output-channel register tiles were compared. Sweep a bounded set of workgroup shapes and pixels/channels per thread on the expensive pointwise shapes. Keep a shape-specific variant only if full-graph gain pays for its complexity. |
| 3 | Reuse inputs/weights across a workgroup or fuse adjacent layers | Cooperative pointwise tiles and depthwise-to-pointwise fusion remain untested. They may save reads and dispatches, but add barriers, storage/register pressure or redundant work. Benchmark one hotspot first; preserve activation boundaries and parity. |
| 4 | Measure batch and webcam scheduling | Test bounded frames/images in flight and CPU/GPU work overlap against the current path. Preserve per-inference buffer ownership through completion. Report throughput and frame latency separately; more concurrent submissions alone are not a gain. |
| 5 | Resolve compiler effects, then revisit selective subgroups/precision | Small-layer subgroup gains exist, but DXC regressed the f32 baseline. Test compiler/code-generation and backend variants before introducing optional production kernels. FP16 arithmetic and packed layouts are different, untried candidates with accuracy/deployment costs. |
| 6 | Reprofile preprocessing, output conversion and command recording | Preprocess-to-inference GPU residency and merged passes already exist. Time the remaining upload, CHW/HWC conversion, sigmoid/decode, uniform and bind-group work. Avoid rebuilding those already-completed optimizations. |
| 7 | Broaden hardware and model experiments | Measure AMD/Intel, Metal/Vulkan and representative image sets. Smaller detector input, model changes or INT8 require explicit recall/landmark/crop-quality evaluation as well as speed measurements. The current GPU implementation is f32. |

For priority 1, inspect [runtime.rs](../fcs-core/src/gpu/runtime.rs), especially
`run_inference`, `build_decode_tensors` and `batch_download`. The proposed
single-wait experiment follows wgpu's documented asynchronous mapping behavior:
a map can wait for preceding GPU work, with callbacks driven by polling. That
supports a test, not a claim of measured savings. Mapped buffers cannot be used
by the GPU until unmapped. [wgpu buffer mapping documentation](https://docs.rs/wgpu/latest/wgpu/struct.Buffer.html#mapping-buffers).

For cooperative tiling, [ONNX Runtime's packed WebGPU matmul](https://github.com/microsoft/onnxruntime/blob/main/js/web/lib/wasm/jsep/webgpu/ops/3rd-party/matmul_packed_webgpu.ts)
provides a concrete primary-source implementation using workgroup tiles and
barriers. Applying it to our pointwise layers is an unmeasured hypothesis; the
current four-channel register tile does not exhaust that design space.

Bind-group caching alone previously accounted for only about 0.107 ms of encode
cost, and fixed intermediate buffers would be a larger architectural change.
Remeasure before pursuing it. The old unverified "~20 ms CLI map/poll" estimate
has been retired; it is not compatible with using today's roughly 3.5 ms
single-detection result as the reference workload.

DirectML was measured without a win in the historical comparison below; CoreML
was not established by that comparison. Hand-written `wide` SIMD was tried and
reverted. These are scoped negative results, not proofs that every runtime,
compiler or architecture will behave the same way.

### The ONNX Runtime CPU backend (shipped in fcs-core)

`tract` lowers YuNet's 3x3 depthwise convolutions to a scalar fallback — it only
unrolls zones with at most 4 taps — which 1.5.3 measured at 59-61% of a CPU
detection and concluded was "not reachable from this side". That was true within
tract. Swapping the runtime reaches it: ONNX Runtime vectorises those
convolutions and runs the same graph in 6.7 ms against 65.9 ms.

Scope deliberately excludes DirectML. Measured end to end it lands at ~9.8 ms
against the then-current 8.2 ms WGSL graph without an additional runtime DLL,
so that comparison did not justify adopting DirectML. It did not measure CoreML
or establish a permanent ceiling for other execution providers. Only the CPU EP
is used, which also means one library (20.1 MB) rather than the ~38 MB a DirectML build needs,
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

- **Runtime selection** changes the inference implementation. This workspace now
  uses its own `fcs-ort` binding and a roughly 20 MB dynamically loaded ONNX
  Runtime library; the external `ort` crate is no longer the integration.
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
3. **What does it cost in recall?** Post-training quantisation on a small detector can lose
   detections, and the quality thresholds are tuned against f32 behaviour.

Two further costs are easy to overlook. INT8 gains depend strongly on the CPU: with AVX512-VNNI
(Zen 4, Cascade Lake+, Alder Lake+) `VPDPBUSD` gives a 4-way i8 dot per lane, but on the shipped
`x86-64-v3` baseline the AVX2 fallback computes i8 products in i16 lanes — the same lane count as
f32 FMA, so the win is cache footprint rather than arithmetic. Benchmarking only on a VNNI
developer machine will overstate what most users get. And the WGSL GPU inference path gains
nothing, so an INT8 CPU path diverges numerically from the f32 GPU path that
`gpu_inference_matches_cpu_baseline` and docs/parity_report.md compare against.
