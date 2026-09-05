# Performance experiments

Results are consolidated in [docs/PERFORMANCE.md](docs/PERFORMANCE.md), including
the remaining opportunities. This file retains the experiment-by-experiment log.

Experiments 0-4 used baseline `c332075` on `master`, RTX 4090 / D3D12. New
experiments start from the latest retained implementation, with its exact commit
and working-tree diff recorded. Work on **one experiment at a time**. A checked
box means **tried and assessed**, not necessarily shipped. Record evidence and
the keep/revert decision before checking it off or starting the next experiment.
Do not infer gains on other GPUs or batch work from single-image measurements.

This backlog covers the major performance avenues in the current application;
it is not a claim that every possible algorithm or parameter combination is
listed. Unticked items are hypotheses, not established defects or promised
speedups. Completing one can make others unnecessary.

## How to use this backlog

- **P0:** establish the current bottleneck and test the smallest plausible fixes.
- **P1:** follow measured hotspots; bounded shader, allocation and scheduling work.
- **P2:** larger architecture, compiler or cross-platform experiments; require
  evidence that the relevant stage matters before implementing.
- **Q:** can change detections, image quality or frame semantics. Define the
  acceptable quality/latency trade-off first; never weaken existing parity tests
  to make a candidate pass. Benchmark a separate candidate/configuration.
- IDs are stable. Add new IDs rather than renumbering completed experiments.
  Follow the suggested route below, then pick by evidence rather than treating
  all later items as mandatory. A deferred or blocked item stays unchecked with
  its reason. A tested rejection is checked and labelled rejected in Results.

**Suggested next route:** 5 (fresh phase timings), 11 (remove the preliminary
readback wait), 12 (reuse staging buffers), 13 (pack head readback), then 14
(combine inference and copy submission), only while measurements justify it.
Re-establish the baseline after each retained change. Shader tuning starts with
26/27 if GPU compute remains the target; batch/webcam work starts with 60.

## Experiment coverage

| Area | IDs | Initial priority |
| --- | --- | --- |
| Completed shader round | 0-4 | Assessed; results below |
| Measurement and controls | 5-10 | P0 |
| Readback and synchronization | 11-18 | P0/P1 |
| CPU recording and resource allocation | 19-25 | P1 |
| Shader geometry and memory access | 26-34 | P1 |
| Graph fusion and intermediate traffic | 35-40 | P2 |
| Compilers, backends and precision | 41-47 | P2 |
| Preprocessing and image upload | 48-54 | P1/P2, some Q |
| Detection output conversion and filtering | 55-59 | P1/P2, some Q |
| Batch and webcam scheduling | 60-66 | P1/P2, some Q |
| Decode, CPU execution and export | 67-73 | P1/P2, some Q |
| Model and algorithm changes | 74-79 | Q/P2 |
| Startup, caching and resource lifetime | 80-85 | P1/P2 |

## Already done: do not rediscover these gains

Uniform caching, merged non-profiled compute passes, fused normalization and
activations, GPU-resident preprocessing-to-inference handoff, buffer/texture
pooling and grouped head download already exist. The current head download
still allocates separate staging buffers and performs two polls; grouping the
copies was not the same as completing the readback work below. Register tiling
was tested; cooperative workgroup tiling was not.

The rejected FP16 storage and general subgroup probes do not rule out all
reduced-precision or subgroup algorithms. The rejected `wide` CPU SIMD and
reduced-scale JPEG crop workflow are historical negatives in
[PERFORMANCE.md](docs/PERFORMANCE.md); revisit only with a materially different
implementation or workload.

## Completed first round

- [x] 0. Establish trustworthy shader measurements. Split GPU timestamps into
  pointwise, depthwise and general convolution. Replace the misleading
  standard/vec4 comparison (both call the same pipeline) with real A/B variants.
- [x] 1. Specialize 1x1 convolution. Remove irrelevant spatial kernel loops,
  grouping arithmetic and padding checks; retain the general fallback.
- [x] 2. Reuse overlapping inputs in depthwise 3x3 convolution. Four adjacent
  outputs need only 18 distinct interior input values versus 36 scalar loads
  expressed by the current shader. Test explicit register reuse and fixed loops.
- [x] 3. Tile pointwise work across output channels. Compare a small register
  tile and, if justified, cooperative workgroup storage against experiment 1.
  Account for barriers, occupancy and register pressure.
- [x] 4. Evaluate FP16 and subgroups separately. Feature-check the actual
  adapter/backend; gate reduced precision on raw-output accuracy and speed before
  detection parity, and measure subgroup sharing against the best f32 kernel.
  Both candidates were measured and rejected for production in this round.

## Remaining experiments

### Measurement and controls (P0)

- [ ] **5. Reprofile the current normal detection path.** Split preprocessing,
  input upload, resource preparation, recording, finish, submit, GPU execution,
  readback allocation/copy/map/waits, output conversion and decode. Measure GPU
  and wall intervals separately; waiting includes preceding GPU work and must
  not be added to it as an independent cost.
- [ ] **6. Build a representative workload matrix.** Cover small/large images,
  portrait/landscape, supported formats, no/one/many faces, difficult small
  faces, warm preview, first detection, webcam and folder export. Record p50/p95
  latency, images/s, CPU use and peak RAM/VRAM where relevant.
- [ ] **7. Quantify timing noise and minimum detectable gains.** Repeat
  identical A/A controls, alternating A/B order and independent runs; record
  clocks, power mode, thermals and competing GPU work. Select sample duration
  from observed variance; treat isolated timestamp ticks as inconclusive.
- [ ] **8. Compare profiled and normal execution costs.** Measure one merged
  graph timestamp where supported versus summed per-op timestamps and normal
  wall time. Separate query resolution/profiler overhead; verify identical raw
  heads. Do not optimize an artifact of the separate-pass profiling path.
- [ ] **9. Capture a native CPU/GPU timeline for one bottleneck.** Use an
  available platform/vendor profiler to distinguish allocation/driver work,
  queue idle gaps, memory traffic and GPU occupancy. Record tool overhead and
  return to the uninstrumented release benchmark to validate any conclusion.
- [ ] **10. Establish hardware/backend baselines.** Measure representative
  NVIDIA, AMD, Intel and integrated/unified-memory adapters as hardware becomes
  available, including D3D12, Vulkan and Metal where supported. Record compiler,
  driver, features and limits; missing hardware is a documented deferral.

### Readback and synchronization (P0 first, then P1)

- [ ] **11. Request maps before the preliminary blocking poll.** In
  `batch_download`, submit copies, start all maps, then drive completion with
  one wait. Compare with the existing wait-map-wait sequence. Require exact raw
  head equality, map-error propagation and concurrent-inference safety.
- [ ] **12. Reuse the 12 staging buffers.** Compare per-run allocation with
  size-aware reuse. Keep each buffer owned until its mapped view is dropped
  and it is unmapped; test concurrent requests and changing output sizes.
  Measure allocation time, wall latency and retained memory separately.
- [ ] **13. Pack all head outputs into one staging buffer.** Compare 12 maps
  against one aligned allocation/map plus offsets. Include CPU slicing/copy
  cost, odd sizes and changing resolutions; validate every raw head and avoid
  reading unused capacity from a larger reused buffer.
- [ ] **14. Encode output copies with inference.** Append copies after the
  compute pass ends and submit once instead of creating a second encoder and
  submission. Preserve tensor/pool lifetimes through completion; compare with
  the latest retained readback strategy, not the original baseline.
- [ ] **15. Compare mapping-on-submit with explicit map requests.** Use
  `map_buffer_on_submit` if it simplifies the retained path; compare CPU setup,
  callback and completion overhead. This is an alternative to explicit mapping,
  not an assumed additional saving on top of experiment 11.
- [ ] **16. Wait only for the relevant work.** Compare submission-index waits
  or mapping completion with a device-wide wait under other in-flight work.
  Measure unrelated-work interference; no buffer may be read or recycled
  before its own submission completes.
- [ ] **17. Reuse CPU output storage.** Compare new vectors/channels/temporary
  tensors per detection with appropriately scoped reuse or decoding from a
  mapped view. Measure copies and allocation cost; release mapped memory
  promptly and preserve error handling and concurrent callers.
- [ ] **18. Overlap readback with the next input.** After 11-17, test a bounded
  two/three-slot staging ring if throughput warrants it. Measure frame latency,
  throughput and VRAM, including slow consumers and cancellation. Depends on
  explicit per-request ownership; not a single-image latency claim.

### CPU recording and resource allocation (P1)

- [ ] **19. Profile and reduce per-layer host bookkeeping.** Measure shape
  validation, graph traversal, weight-name lookup, temporary collections,
  labels and reference counting in warm inference. Pre-resolve only repeated
  immutable data that is a measured cost; keep public input validation.
- [ ] **20. Cache remaining small-operation uniforms.** Measure max-pool,
  add and resize uniform creation after convolution caching. Reuse immutable
  content-keyed values only if the eight dispatches contribute a repeatable
  cost; include cache growth under varied resolutions.
- [ ] **21. Compare per-inference fixed intermediate plans with pooling.**
  Precompute shapes/lifetimes for the fixed YuNet graph and allocate a bounded
  workspace per in-flight request. Measure recording time and peak memory;
  protect current safe reuse and never share writable intermediates concurrently.
- [ ] **22. Cache bind groups with stable buffer identities.** Depends on
  evidence from 19/21. Compare construction cost against retained groups per
  workspace, with correct input/output identities and invalidation. The old
  0.107 ms figure is historical, not an expected current saving.
- [ ] **23. Test buffer arenas or dynamic offsets.** If 21/22 justify it,
  compare individually bound buffers with aligned suballocations and offsets.
  Account for binding limits, aliasing rules, internal fragmentation and CPU
  indexing. Reject if complexity exceeds a repeatable whole-path gain.
- [ ] **24. Measure pool and shared-cache contention.** Compare lock wait and
  allocation behavior under actual batch concurrency; test per-worker/per-slot
  ownership only where contention is observed. Retain bounded memory and
  correctness during cancellation, failure and changing image sizes.
- [ ] **25. Cache a host-side dispatch plan.** Compare graph traversal with
  pre-resolved pipelines, shapes and resource slots after 19. Include warm and
  cold cost. This means reusing metadata, not assuming a submitted wgpu compute
  command buffer can be replayed.

### Shader geometry and memory access (P1)

- [ ] **26. Sweep pointwise workgroup shapes.** Start with a small explicit
  set such as 8x8, 16x4, 32x2 and 8x4, adjusting dispatch coverage consistently.
  Measure large layers and tiny heads, raw tails and full graph; select per-shape
  kernels only when the gain survives dispatch/selection overhead.
- [ ] **27. Sweep pixels and output channels per thread.** Extend the existing
  1/2/4-channel comparison to bounded combinations of spatial width and channel
  tiles, including 8 channels where limits permit. Compare register pressure,
  tail waste and occupancy; do not assume a larger tile wins.
- [ ] **28. Specialize fixed dimensions at pipeline creation.** Compare runtime
  uniform loops with constants/overrides or generated kernels for recurring
  shapes. Measure compiler unrolling, warm speed, compile time and pipeline
  count; retain the general fallback and bound specialization growth.
- [ ] **29. Cooperatively tile pointwise inputs and weights.** Stage reusable
  tiles in workgroup memory and compare with the retained register tile. Sweep
  a small inner-channel tile set; all invocations must reach barriers uniformly,
  including edge groups. Count initialization/barrier cost and full-graph gain.
- [ ] **30. Prepack immutable weights for vector/coalesced loads.** Try a
  layout suited to the winning pointwise mapping, paying packing once at model
  load. Measure cache behavior and warm kernels plus startup/memory overhead;
  preserve channel tails and the source model's numerical values.
- [ ] **31. Compare activation layouts across a graph segment.** Test NCHW
  against blocked channels or NHWC only on a representative connected segment.
  Include every required transpose/packing conversion and other affected ops;
  reject isolated kernel gains that lose end to end.
- [ ] **32. Tune depthwise workgroups and tile reuse.** Compare more horizontal
  pixels or two-dimensional register/workgroup tiles against the retained six-
  value row reuse. Measure halo duplication, barriers, borders and tiny maps;
  verify stride/padding fallbacks remain correct.
- [ ] **33. Separate interior and edge handling.** Compare bounds-checked
  kernels with an interior fast path or explicit border dispatch for large
  maps. Include extra dispatch cost and safe accesses on small/odd inputs;
  never turn off validation or rely on out-of-bounds behavior.
- [ ] **34. Specialize the remaining general/stem convolution.** Measure the
  3x3, stride-2, three-input-channel stem separately; try fixed loops and input
  reuse. Its measured roughly 41 us is a small ceiling, so stop if gains do not
  survive whole-graph measurement.

### Graph fusion and intermediate traffic (P2)

- [ ] **35. Fuse depthwise then pointwise.** Prototype one expensive adjacent
  pair, keeping any intervening activation in its original position. Compare
  fewer intermediate reads/writes with recomputation, registers and halo costs;
  require intermediate/raw-head and final-detection parity.
- [ ] **36. Fuse compatible pointwise/add/resize or pool boundaries.** Pick
  one actual graph pattern with measured traffic/dispatch cost. Preserve
  operation order and fan-out consumers; compare with already-merged passes,
  since eliminating a pass is not a new saving here.
- [ ] **37. Compute detection head branches together.** Test sharing input
  loads across cls/obj/bbox/keypoint outputs at one level. Include small and
  mismatched channel counts, occupancy and output layout; validate all 12 heads.
- [ ] **38. Write head outputs in CPU decode order.** Compare final-layer
  direct HWC/packed output with CHW followed by CPU reorder. Include GPU store
  coalescing, downstream binding changes and decode cost; do not move sigmoid
  across other operations or apply it twice.
- [ ] **39. Fuse compatible preprocessing and stem work.** Prototype only
  after 48 identifies a relevant cost. Account for source texture sampling,
  resize semantics, border handling and source-pixel reuse; keep the exact
  preprocessor contract unless explicitly evaluating a Q variant.
- [ ] **40. Reduce intermediate lifetimes and unnecessary traffic.** Use graph
  liveness to find copies or buffers that can be removed or safely reused.
  Compare VRAM and GPU time with the existing pool; avoid assuming every
  allocation causes a copy, or reusing memory before all consumers finish.

### Compilers, backends and precision (P2)

- [ ] **41. Isolate the FXC/DXC regression.** Compare the same f32 source,
  shape, driver and context settings; inspect generated code and bounded source
  variants around dynamic array indexing/loops. Record the exact compiler
  binaries. A fix must retain the current f32 baseline before enabling features.
- [ ] **42. Compare supported backend/compiler versions.** Test D3D12 versus
  Vulkan on the same supported adapter and later Metal on available hardware.
  Include warm latency, compilation, correctness and stability; do not remove
  platform guards or change shipped defaults solely for a benchmark.
- [ ] **43. Revisit subgroups only on winning small shapes.** After 41/42,
  compare selective subgroup reduction with the best non-subgroup kernel,
  including selection and compiler effects across the full graph. Provide a
  valid fallback and verify the actual subgroup width.
- [ ] **44. Try subgroup sharing without channel reduction.** Compare
  broadcast/shuffle-based sharing of inputs or weights within a suitable tile
  against register/workgroup reuse. Avoid the uncoalesced accesses of the
  rejected candidate; measure across available subgroup sizes and adapters.
- [ ] **45. Test FP16 arithmetic and packed layouts separately.** The rejected
  experiment changed storage with f32 accumulation. Compare native half
  arithmetic or packed vectors as distinct candidates, including conversions;
  screen raw error before expensive full-model/detection validation. Q if parity
  cannot be preserved; no production tolerance relaxation.
- [ ] **46. Test selective mixed precision.** After a useful candidate in 45,
  keep sensitive layers/heads in f32 and test lower precision only where error
  and speed permit. Include boundary conversion cost and a representative
  difficult-face corpus; synthetic one-layer accuracy is insufficient. Q.
- [ ] **47. Investigate accelerated matrix/native-runtime paths.** Check actual
  supported features or a platform runtime before prototyping matrix hardware,
  CUDA/TensorRT/DirectML/CoreML or another backend. Compare the complete workload,
  packaging/startup and maintenance cost; support and speed are unproven here.
  This is a larger optional fork, not a reason to add dependencies pre-emptively.

### Preprocessing and image upload (P1/P2)

- [ ] **48. Refresh CPU versus on-device preprocessing measurements.** Compare
  `gpu`, `gpu_on_device` and `gpu_quality` with matched input/resize semantics.
  Split upload, conversion, resize and inference handoff; warm and cold paths
  need separate results. Do not recreate the already-removed tensor round trip.
- [ ] **49. Reduce source pixel conversion and copies.** Trace decoded RGB/RGBA,
  row layout and upload buffers; remove only measured redundant conversions or
  copies. Compare throughput on large images and verify colour order, stride,
  alpha handling and orientation.
- [ ] **50. Compare upload strategies and resource reuse.** Measure existing
  queue writes against reusable staging/texture resources at representative
  sizes. Include allocation and transfer cost, alignment rules and concurrent
  ownership; a discrete-GPU result need not apply to unified memory.
- [ ] **51. Tune preprocessing shader geometry and sampling.** Compare bounded
  workgroups and explicit vector loads or texture sampling under the same
  resize contract. Validate borders and CPU/GPU parity; any different filter or
  coordinate convention is a separately assessed Q candidate.
- [ ] **52. Cache preprocessing for unchanged preview input.** First inspect
  current GUI caches and invalidation. Test reuse across changes that only affect
  crop presentation/enhancement; invalidate for image, orientation, input-size
  or detector changes. Measure repeated-interaction latency and retained memory.
- [ ] **53. Keep webcam frames on the GPU where capture permits.** Investigate
  native texture/frame import or fewer colour-conversion copies. Count capture-
  to-result latency, synchronization and format conversion; retain the portable
  fallback and do not assume the capture API exposes compatible device memory.
- [ ] **54. Evaluate adaptive resize/input routing.** Test a measured CPU/GPU
  cutoff by source size/device and, separately, cheaper preview resize quality.
  Include routing overhead. Quality changes require detection/landmark/crop
  evaluation and distinct settings; smaller input is also covered by 74. Q.

### Detection output conversion and filtering (P1/P2)

- [ ] **55. Combine CPU reorder, sigmoid and decode work.** Profile current
  traversals and temporary tensors, then fuse one measured redundant pass.
  Compare exact output ordering and established numerical tolerances; avoid
  reproducing the already-rejected fine-grained loop parallelism.
- [ ] **56. Move score/box decoding to the GPU.** Compare GPU decode plus a
  readback against current CPU decoding, including dispatch and map overhead.
  Preserve score calculation, anchors, dimensions and landmark conventions;
  merely moving tiny CPU work to the GPU can lose.
- [ ] **57. Compact valid candidates before readback.** Use the existing score
  rule and threshold to reduce output bytes; measure sparse and crowded cases,
  counter/scan overhead and overflow handling. Preserve ordering/tie semantics
  where observable; no arbitrary top-K cap to manufacture speed.
- [ ] **58. Reprofile NMS on worst-case candidate counts.** Compare current
  spatial-grid NMS with bounded CPU or GPU alternatives only if it is material.
  Validate overlaps, equal-score ties and dense scenes; GPU transfer/dispatch
  cost belongs in the comparison.
- [ ] **59. Evaluate approximate candidate pruning separately.** Try top-K,
  alternate thresholds or approximate NMS only against an explicit recall and
  landmark-quality budget. Measure difficult and crowded scenes, not just
  average timing. Q; never substitute these for exact-path optimization.

### Batch and webcam scheduling (P1/P2)

- [ ] **60. Measure bounded GPU concurrency.** Compare 1/2/3/4 in-flight
  requests with fixed input sets and safe per-request buffers. Record images/s,
  p50/p95 latency, peak memory and queue delay; do not assume more rayon workers
  create useful GPU parallelism.
- [ ] **61. Pipeline CPU decode/upload/GPU inference/export.** Use measured
  stage costs to test a bounded producer/consumer schedule. Include backpressure,
  errors and cancellation; compare total folder completion time, not a stage in
  isolation. Depends on safe ownership from 18/21/60 as applicable.
- [ ] **62. Submit true small inference batches.** Compare a batch dimension
  of 2/4/8 with independent in-flight requests. Validate every kernel, head,
  decode and memory plan for batch indexing; current single-image behavior is
  not evidence that batching already works or improves throughput.
- [ ] **63. Tune CPU thread budgets alongside GPU work.** Compare rayon and
  runtime thread counts on single/batch workloads; detect oversubscription,
  driver starvation and memory-bandwidth contention. Record CPU-only results
  as well; avoid a global setting chosen from one developer machine.
- [ ] **64. Compare a GPU submission worker with caller-thread submission.**
  Only if 9/24/60 show contention or idle gaps, test a bounded dispatcher.
  Include handoff latency and fairness; do not introduce a worker/thread solely
  as an abstraction or serialize independent CPU work unnecessarily.
- [ ] **65. Prefer fresh webcam frames under overload.** Compare queued-all
  processing with bounded latest-frame scheduling, skipping stale detections
  when a newer frame supersedes them. Report capture-to-display age, dropped
  frames and detection cadence as well as fps. Q; export must remain complete.
- [ ] **66. Share work across identical preview requests.** Inspect existing
  cancellation/caches, then test coalescing duplicate in-flight detections.
  Measure rapid UI edits and mixed images; prevent stale results from replacing
  current ones and preserve errors/cancellation for each requester.

### Decode, CPU execution and export (P1/P2)

- [ ] **67. Benchmark alternative full-resolution decoders.** Compare available
  implementations on the actual format corpus, including orientation/colour
  fidelity, cold I/O, warm cache and batch throughput. Preserve full-resolution
  crop pixels; this differs from the rejected reduced-scale decode workflow.
- [ ] **68. Tune file I/O and bounded decode prefetch.** Measure cold disk,
  cached files and large folders separately. Compare modest prefetch depth and
  reuse of read buffers; include peak RAM and cancellation. Skip if decode or
  inference dominates and I/O is already hidden.
- [ ] **69. Compare CPU inference settings and layout costs.** Benchmark the
  shipped tract and ONNX Runtime paths with bounded thread/optimization settings,
  warm sessions and real batches. Profile tensor conversion and memory copies;
  a faster isolated runtime is not necessarily a faster application.
- [ ] **70. Apply measured CPU vectorization/PGO changes.** Inspect hot-loop
  assembly first; compare compiler flags or representative profile-guided
  optimization with the current x86-64-v3/autovectorized build. Include portable
  fallbacks and startup/binary-size costs; do not re-add `wide` without evidence.
- [ ] **71. Keep crop/enhancement intermediates on device.** Trace actual
  filters and crop batches, then remove measured intermediate downloads/uploads
  or fuse compatible filter passes. Include final export readback and verify
  pixel quality plus operation ordering at production image sizes.
- [ ] **72. Measure export encoding and concurrency.** Compare encoder settings
  and bounded encode/write parallelism after detection accelerates. Report total
  export time, output size and peak memory. Compression/quality changes are Q;
  identical settings and pixels are the baseline.
- [ ] **73. Evaluate two-stage decoding only for suitable workloads.** For
  mostly no-face or detection-only input, test a cheap screening decode followed
  by full decode only when needed. Count both decodes for positives and all
  missed faces. Q; rejected for normal crop-heavy work unless new evidence
  changes that workload assumption.

### Model and algorithm changes (Q/P2)

- [ ] **74. Sweep detector input resolution.** Compare supported smaller/larger
  inputs with latency, recall by face size, landmark error and final crop
  quality. Include rescaling/model-shape compatibility; do not silently change
  the quality contract of the current 640x640 detector.
- [ ] **75. Test coarse-to-fine or region-of-interest detection.** Evaluate a
  cheap first pass with targeted higher-resolution follow-up. Include failures
  of the first pass, crowded scenes, edge faces and total follow-up work.
  Coordinates and full-resolution crop accuracy must remain correct.
- [ ] **76. Test tracking between webcam detections.** Compare periodic
  detection plus tracking against detecting every frame. Evaluate new entrants,
  occlusion, rapid motion, scene cuts and recovery; report detection latency and
  missed faces as well as throughput. Does not apply to independent exports.
- [ ] **77. Evaluate calibrated INT8/QDQ models.** First check graph loading
  and actual quantized-kernel use, then speed and quality on a held-out corpus.
  Separate calibration data from evaluation; compare CPU and any supported GPU
  path honestly. A smaller model file alone is not an inference speedup.
- [ ] **78. Compare smaller models or structured pruning/distillation.** Treat
  as model research with training/calibration costs and a reproducible quality
  benchmark. Measure deployment size, load time and all target runtimes;
  include small-face/landmark failures and preserve the existing model option.
- [ ] **79. Test content-aware early exits or cascades.** Evaluate cheap
  no-face screening or confidence-based refinement with explicit false-negative
  budgets. Measure worst-case work when all stages run; never infer safety from
  only easy single-face images.

### Startup, caching and resource lifetime (P1/P2)

- [ ] **80. Split cold start into adapter, model and pipeline costs.** Measure
  process launch, device selection, model read/parse/weight upload, shader compile
  and first detection separately from steady state. Compare cached and clean
  runs; prevent lazy initialization from hiding cost in the first user action.
- [ ] **81. Reuse device/model/pipelines across real UI lifetimes.** Check
  existing sharing, then eliminate measured accidental recreation across preview,
  webcam and export. Include device loss, switching models and shutting down;
  longer lifetime must not produce unbounded retained GPU memory.
- [ ] **82. Evaluate supported pipeline caches or controlled prewarming.**
  Feature-check the active backend, compare cold/warm startup and first-frame
  latency, and invalidate persisted data by compatible device/driver/shader
  identity. Include cache size and total work; moving compilation earlier is
  not the same as making startup cheaper.
- [ ] **83. Bound specialization and uniform-cache growth.** Sweep many input
  resolutions/configurations and compare memory, hit rate and eviction cost.
  Test a bounded cache or known-model prepopulation only if growth matters;
  keep content-based correctness and concurrent access safety.
- [ ] **84. Tune pool retention under mixed workloads.** Compare current
  retention with bounded high-water marks or idle trimming across large images,
  smaller follow-up runs and concurrent exports. Measure allocation churn,
  p95 latency, VRAM pressure and device failures; memory savings may trade speed.
- [ ] **85. Validate sustained operation and power efficiency.** Run the best
  candidates through long webcam sessions and large exports, including VRAM
  pressure and background GPU work. Track drift, thermals, memory growth,
  responsiveness and energy/image where measurable; short warm microbenchmarks
  can miss production regressions.

## Result record for each new experiment

Append a result under the stable ID, then update its checkbox and the roll-up
in [PERFORMANCE.md](docs/PERFORMANCE.md) when assessed. Use this compact template:

```text
ID / title / date:
Status: kept | rejected | inconclusive | deferred | blocked
Baseline commit + working-tree diff; candidate diff/probe path:
Hardware / OS / driver / backend / compiler / enabled features:
Workload / input dimensions / warm or cold / concurrency / exact commands:
Hypothesis and changed variable; prerequisite experiment results:
A/A noise and A/B order; sample counts; independent repetitions:
GPU time; whole-path p50/p95; throughput; memory (only relevant metrics):
Raw parity / final detections / quality checks and failures:
Decision, measured limits and what would justify revisiting:
```

A quality-changing candidate needs its own explicit evaluation criteria; do not
call it parity-preserving. An inconclusive experiment may be checked as tried,
but remains unadopted and records what additional evidence is needed. No projected
saving becomes a measured result until the experiment supplies it.

## Measurement and acceptance

- Warm pipelines and buffers; alternate A/B order in the same process. Use GPU
  timestamps around resident dispatches, with upload/readback outside the timer.
- Use actual YuNet shapes plus borders and non-multiple dimensions for correctness.
  Compare raw outputs before timing; retain existing ONNX/detection tolerances.
- A faster microbenchmark earns a full-graph trial, not immediate adoption.
  Validate the merged production path, profiled path and concurrent inference.
- Record negative results. Remove losing production changes. Keep the smallest
  reproducible probe that makes the decision reviewable.
- The retained graph is about 0.536 ms of profiled GPU compute on this setup.
  Halving all shader work removes about 0.268 ms of GPU work at fixed workload;
  it does not promise that wall-time saving. The historical 0.9 ms baseline
  below is not the starting point for new experiments.
- Do not run competing GPU benchmarks concurrently. Match preprocessing and
  image sets; separate latency, throughput, startup and memory goals. Record
  p95 only with enough samples to support it and report uncertainty.
- New low-impact code changes need relevant checks, not an automatic full-suite
  rerun per microvariant. Retained graph/resource/precision changes must pass
  the applicable strict parity, detection and concurrency checks before adoption.
- Rank by measured recoverable time and complexity. Reprofile after a win,
  stop an unproductive sweep, and keep prior negative results visible.

## Research references

- [ONNX Runtime convolution selection](https://github.com/microsoft/onnxruntime/blob/main/js/web/lib/wasm/jsep/webgpu/ops/conv.ts): eligible 1x1 convolutions become matrix multiplication, including NCHW.
- [ONNX Runtime depthwise implementation](https://github.com/microsoft/onnxruntime/blob/main/js/web/lib/wasm/jsep/webgpu/ops/conv-grouped.ts): overlapping-input reuse and constant kernel dimensions.
- [ONNX Runtime packed matmul](https://github.com/microsoft/onnxruntime/blob/main/js/web/lib/wasm/jsep/webgpu/ops/3rd-party/matmul_packed_webgpu.ts): workgroup tiles and multiple accumulators per thread.
- [Chrome engineering measurements](https://developer.chrome.com/blog/io24-webassembly-webgpu-2): FP16, subgroups and memory access optimizations vary substantially by GPU.
- [wgpu feature documentation](https://wgpu.rs/doc/wgpu/struct.Features.html): native feature/backend availability must be checked independently of browser support.

Additional primary references for the backlog (technique support, not evidence
of speedups in this application):

- [wgpu mapping on submit](https://docs.rs/wgpu/30.0.1/wgpu/struct.CommandEncoder.html#method.map_buffer_on_submit): schedules mapping after the command submission; CPU/GPU buffer ownership and callback completion still apply (11/15).
- [wgpu buffer mapping](https://docs.rs/wgpu/30.0.1/wgpu/struct.Buffer.html#mapping-buffers): asynchronous mapping, polling and mapped-view lifetimes (11-18).
- [ONNX Runtime quantization guide](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html): calibration, quantization formats and accuracy troubleshooting (77). Availability of a format does not guarantee an accelerated kernel in our selected runtime.

## Results

### 0. Measurement prerequisite — kept

The full graph's GPU timestamps on the baseline are 0.910 ms total:

| Family | Dispatches | GPU time | Share |
| --- | ---: | ---: | ---: |
| Pointwise | 26 | 673.8 us | 74.0% |
| Depthwise | 26 | 145.4 us | 16.0% |
| General/stem | 1 | 42.0 us | 4.6% |
| Pool/resize/add | 8 | 49.1 us | 5.4% |

`gpu_pass_breakdown` now reports those families. The old Criterion comparison
is now explicitly one `gpu_with_readback` measurement. Both previous labels
called the same pipeline and could not compare vectorization.

`conv2d_experiment` loads two actual WGSL files, checks finite raw outputs
within `1e-4 + abs(reference)*1e-4`, and measures 50 alternating pairs after
20 warm-ups per shape. Identical-file control: 13 of 14 shape medians matched;
one differed by 3.2%. Timestamps here are quantized in roughly 1.024 us steps,
so small differences on tiny layers are not actionable.

Commands (from the workspace root):

```powershell
cargo run --release -p fcs-core --example gpu_pass_breakdown
cargo run --release -p fcs-core --example conv2d_experiment -- baseline.wgsl candidate.wgsl
```

### 1. Pointwise specialization - kept

Added a uniform fast-path branch for 1x1, unit stride, zero padding and one
group. It bypasses the general spatial loops without adding another production
pipeline. Other convolutions retain the original path. The shader grew by 29
lines, sharing activation and output writes with the fallback.

Two microbenchmark runs showed 10-46% lower time on the larger pointwise cases;
small one-tick changes were disregarded. Full graph A/B/B/A GPU totals were
0.909 / 0.673 / 0.672 / 0.910 ms. Pointwise itself fell from 671 to 433-434 us;
the other families stayed close. That is about **0.237 ms (26%) less total GPU
compute**, not a 26% detection-latency claim.

Whole-detection run estimates were 3.365 / 3.037 / 3.324 / 3.221 ms. Their drift
and overlap do not establish a reliable end-to-end percentage improvement.
Keep the repeatable GPU saving; do not quote a stronger wall-time claim.

Validation: all 187 core tests passed under strict fixture/model/runtime
checks, including ONNX parity, concurrent and profiled inference. The new
CPU-reference test covers widths 1/3/4/5/37, every exposed activation, and
independent stride/padding/group fallbacks. Core all-target Clippy, formatting
and the raw-output A/B probe passed.

### 2. Depthwise overlapping-input reuse - kept

Changed only the depthwise 3x3/unit-stride/pad-1/channel-multiplier-1 path.
Six input values per row are reused for four outputs. Microbenchmarks saved
1-2 us on the measured nontrivial depthwise shapes; the one-pixel case was
unchanged. All 19 raw-output cases passed, including borders and odd sizes.

Full-graph A/B/B/A GPU totals: **0.673 / 0.648 / 0.646 / 0.674 ms**.
Depthwise itself: 149.5 / 122.9 / 120.8 / 150.5 us. The repeatable saving is
about **0.027 ms**, roughly 4% beyond experiment 1. No wall-time percentage
claim is warranted at this scale. All 187 core tests passed, including an
expanded CPU-reference test for depthwise borders, activations and fallbacks.

### 3. Pointwise channel tiling - kept (four channels)

Compared one/two/four output channels per thread. The one-channel standalone
control matched the retained shader to within one timestamp tick. Two channels
saved 32-45% on the large shapes. Four channels saved 42-71%, with no clear
change on the smaller shapes. Raw outputs passed, including channel tails.

Retained a four-channel register tile and matching Rust dispatch-z rounding.
The fallback still dispatches one channel per z group. No workgroup memory or
barriers were necessary. All 187 core tests passed, including odd channel and
spatial tails. Full-graph A/B/B/A totals were **0.647 / 0.535 / 0.538 /
0.651 ms**; pointwise was 433.2 / 320.5 / 323.6 / 432.1 us. This saves
about **0.113 ms** beyond experiment 2. The three retained changes together
reduce GPU compute from about **0.910 to 0.536 ms (41%)** on this setup.
This is a GPU timestamp result, not a whole-detection or batch-throughputput gain.

The probe accepts output coverage (x, y, channels) for B only or both A and B.
Providing coverage selects pointwise-only cases. The defaults are 32 x 8 x 1;
**the retained tiled shader requires 32 x 8 x 4 for pointwise comparisons**.
For example: `conv2d_experiment baseline.wgsl tile4.wgsl 32 8 4` compares an
old one-channel baseline to a four-channel candidate.

### 4. FP16 and subgroups - tested, not adopted

The default D3D12 setup selected FXC and did not expose either optional feature.
Adding the installed Windows SDK DXC directory to the probe's process-local
PATH enabled both SHADER_F16 and SUBGROUP. This GPU has 32-lane subgroups.
The tested DXC was 1.9.2602.17, SDK 10.0.28000.0; production compiler settings
and feature requirements remain unchanged.

**FP16 storage with f32 accumulation:** input, weights and bias were packed to
f16 outside the timing interval; output and accumulators stayed f32. All ten
synthetic pointwise cases stayed within an absolute 1e-3 screening budget
(maximum error 0.000824). This does not establish detection parity. Compared
with the retained f32 shader under the same DXC compiler, eight cases were
3-15% slower and two tiny cases unchanged. A repeat using the standalone f32
tile gave the same decision. Reject: even excluding conversion costs, there
was no speed benefit to justify full-graph FP16 integration or parity testing.
Existing production tolerances were not changed.

**Subgroup reduction:** one 32-lane subgroup accumulates input channels for
four adjacent output pixels, then uses subgroupAdd. All ten cases passed the
normal raw f32 comparison. Large shapes were 2.9-9.9 times as slow; some small
shapes were faster. Selected production-f32/candidate medians under DXC:

| Pointwise shape | f32 tile | Subgroup | Decision |
| --- | ---: | ---: | --- |
| 320x320, 16 -> 16 | 20.480 us | 201.728 us | Regression |
| 160x160, 64 -> 64 | 67.584 us | 370.688 us | Regression |
| 80x80, 64 -> 64 | 32.768 us | 95.232 us | Regression |
| 20x20, 64 -> 64 | 29.696 us | 8.192 us | Microbenchmark gain |
| 80x80, 64 -> 1 | 20.480 us | 5.120 us | Microbenchmark gain |
| 40x40, 64 -> 10 | 29.696 us | 7.168 us | Microbenchmark gain |

Reject as a general replacement. A shape-selective subgroup path remains a
possible future experiment, but it also needs a compiler/deployment solution:
DXC itself regressed the retained f32 tile here. For example, 160x160 64 -> 64
was 27.648 us under the existing FXC setup versus 67.584 us under DXC. A fresh
FXC identical-file control matched all ten medians. Switching the whole context
to DXC just for the small heads would sacrifice large-layer performance.
No full-graph subgroup integration or detection-parity claim is made.

Reproduce the rejected candidates after building the probe (adjust the SDK path
for the local installation). Run in a fresh PowerShell process so PATH affects
only these experiments:

```powershell
cargo build --release -p fcs-core --example conv2d_experiment
$env:PATH='C:/Program Files (x86)/Windows Kits/10/bin/10.0.28000.0/x64;' + $env:PATH
& target/release/examples/conv2d_experiment.exe fcs-core/src/gpu/conv2d.wgsl fcs-core/examples/shaders/pointwise_f16.wgsl 32 8 4 32 8 4 --f16-storage
& target/release/examples/conv2d_experiment.exe fcs-core/src/gpu/conv2d.wgsl fcs-core/examples/shaders/pointwise_subgroup.wgsl 32 8 4 4 1 1
```

The subgroup fixture is specifically for a 32-lane subgroup; it is not a portable
production kernel. With this Naga version the native SUBGROUP feature enables
the builtins directly; `enable subgroups;` is rejected by its parser.

### Final validation and whole-detection check

After all three retained changes, the release Criterion A/B/B/A run estimates
were **3.561 / 3.504 / 3.495 / 3.463 ms**, with overlapping intervals. Each run
used 3 seconds of warm-up and 30 samples over at least 6 seconds. This round
therefore establishes no reliable whole-detection improvement, despite the
repeatable GPU timestamp gain. Criterion's cached historical change percentages
are not used as evidence for this comparison.

Validation after the final source changes: **823 workspace tests and two
doctests passed** (eight workspace tests and five doctests were skipped/ignored).
The workspace run used `--all-features`, `FCS_STRICT_TESTS=1` and the compatible
ONNX Runtime DLL, covering raw ONNX parity, final detections, concurrent
inference, and profiled/merged equality. Workspace Clippy, core all-target
Clippy, formatting and `git diff --check` also passed.

### Previous work

The compute-pass merge is already shipped in the baseline: roughly 0.40 ms saved in paired
encode/finish/submit/wait measurements, with all 61 profiling records retained.
