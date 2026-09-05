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
- Box states: `[ ]` not yet tried, `[x]` tried and assessed (kept or rejected),
  `[~]` **premise removed** -- another result eliminated the cost this item was
  written to attack, so there is nothing left for it to win. `[~]` is not a
  skip: the reason belongs in Results like any other outcome, and the item can
  be reopened if a different motivation for it appears.

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

- [x] **5. Reprofile the current normal detection path.** Split preprocessing,
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
- [x] **9. Capture a native CPU/GPU timeline for one bottleneck.** Use an
  available platform/vendor profiler to distinguish allocation/driver work,
  queue idle gaps, memory traffic and GPU occupancy. Record tool overhead and
  return to the uninstrumented release benchmark to validate any conclusion.
- [ ] **10. Establish hardware/backend baselines.** Measure representative
  NVIDIA, AMD, Intel and integrated/unified-memory adapters as hardware becomes
  available, including D3D12, Vulkan and Metal where supported. Record compiler,
  driver, features and limits; missing hardware is a documented deferral.

### Readback and synchronization (P0 first, then P1)

- [x] **11. Request maps before the preliminary blocking poll.** In
  `batch_download`, submit copies, start all maps, then drive completion with
  one wait. Compare with the existing wait-map-wait sequence. Require exact raw
  head equality, map-error propagation and concurrent-inference safety.
- [x] **12. Reuse the 12 staging buffers.** Compare per-run allocation with
  size-aware reuse. Keep each buffer owned until its mapped view is dropped
  and it is unmapped; test concurrent requests and changing output sizes.
  Measure allocation time, wall latency and retained memory separately.
- [x] **13. Pack all head outputs into one staging buffer.** Compare 12 maps
  against one aligned allocation/map plus offsets. Include CPU slicing/copy
  cost, odd sizes and changing resolutions; validate every raw head and avoid
  reading unused capacity from a larger reused buffer.
- [x] **14. Encode output copies with inference.** Append copies after the
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

- [x] **19. Profile and reduce per-layer host bookkeeping.** Measure shape
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
- [~] **38. Write head outputs in CPU decode order.** Compare final-layer
  direct HWC/packed output with CHW followed by CPU reorder. Include GPU store
  coalescing, downstream binding changes and decode cost; do not move sigmoid
  across other operations or apply it twice. **Premise removed by 55:** there
  is no CPU reorder left to save. Reopen only if GPU store coalescing alone
  justifies it.
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

- [x] **48. Refresh CPU versus on-device preprocessing measurements.** Compare
  `gpu`, `gpu_on_device` and `gpu_quality` with matched input/resize semantics.
  Split upload, conversion, resize and inference handoff; warm and cold paths
  need separate results. Do not recreate the already-removed tensor round trip.
- [x] **49. Reduce source pixel conversion and copies.** Trace decoded RGB/RGBA,
  row layout and upload buffers; remove only measured redundant conversions or
  copies. Compare throughput on large images and verify colour order, stride,
  alpha handling and orientation.
- [x] **50. Compare upload strategies and resource reuse.** Measure existing
  queue writes against reusable staging/texture resources at representative
  sizes. Include allocation and transfer cost, alignment rules and concurrent
  ownership; a discrete-GPU result need not apply to unified memory.
- [x] **51. Tune preprocessing shader geometry and sampling.** Compare bounded
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

- [x] **55. Combine CPU reorder, sigmoid and decode work.** Profile current
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
- [ ] **86. Detect from a reduced-scale decode, crop from the full one.**
  Experiment 51 showed the resize is bounded below by reading the source once,
  so the only remaining lever on it is fewer source pixels. A JPEG decoded at
  1/2 via DCT scaling is a proper low-pass, not a dropped-pixel approximation,
  and at 1/2 a 10 MP source is still 4x the 640x640 input. Detection would read
  2.5 MP instead of 10; crops keep the full-resolution decode, so this is not
  the rejected reduced-scale crop workflow. Measure both decodes where a face is
  found, and evaluate recall on small faces with `resize_quality.rs`. Q, and it
  changes the decode stage rather than preprocessing. Distinct from 73, which
  screens for whether faces exist at all.

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

### 5. Phase timings for the normal detection path - kept (measurement)

Split the existing three GPU guards into eleven, so recording, submission and
each readback phase are separately visible, and added
`fcs-core/examples/phase_timings.rs`, which captures the guard log lines
instead of printing them and reports a p50/p95 per label over 30 warm runs.

```powershell
cargo run --release -p fcs-core --example phase_timings                          # 10 MP
cargo run --release -p fcs-core --example phase_timings fixtures/images/249_o.jpg # 0.17 MP
```

RTX 4090 / D3D12 / FXC, warm, single request, baseline `9906987` plus the guard
diff. Milliseconds; indentation is containment, so children must not be summed
with their parent. `readback_wait` **contains the forward pass**: it is GPU
execution plus the head copies, not idle time to be added to a GPU total.

| Phase | 0.17 MP p50 | 0.17 MP p95 | 10 MP p50 | 10 MP p95 |
| --- | ---: | ---: | ---: | ---: |
| detect_image | 1.570 | 1.770 | 4.680 | 5.610 |
| - preprocess (CPU, large only) | - | - | 2.650 | - |
| - on-device preprocess (small only) | ~0.21 | - | - | - |
| - inference | 1.350 | 1.560 | 1.970 | 2.470 |
| - - encode | 0.343 | 0.417 | 0.449 | 0.768 |
| - - - record | 0.213 | 0.263 | 0.281 | 0.464 |
| - - - finish + submit | 0.128 | 0.160 | 0.160 | 0.296 |
| - - readback | 0.792 | 1.030 | 1.040 | 1.430 |
| - - - staging alloc | 0.044 | 0.053 | 0.052 | 0.073 |
| - - - copy encode + submit | 0.045 | 0.059 | 0.059 | 0.092 |
| - - - wait (contains GPU execution) | 0.501 | 0.739 | 0.767 | 1.130 |
| - - - map request | 0.001 | 0.002 | 0.003 | 0.004 |
| - - - map wait | 0.004 | 0.006 | 0.005 | 0.008 |
| - - - collect + unmap | 0.064 | 0.080 | 0.029 | 0.039 |
| - - - CHW to HWC + sigmoid | 0.108 | 0.131 | 0.111 | 0.133 |
| - - decode | 0.205 | 0.217 | 0.188 | 0.209 |
| - - host upload of input tensor | - | - | 0.280 | 0.359 |
| - postprocess | 0.010 | 0.023 | 0.005 | 0.007 |

Two different paths, not one. At 0.17 MP `detect_on_device` applies and
preprocessing costs about 0.21 ms. At 10 MP `upload_pays_for_source` declines,
so the CPU preprocessor runs at **2.650 ms — 57% of the whole detection** — and
the input tensor is then uploaded for another 0.280 ms. The 22 MP fixtures take
the same route. Large-image detection is therefore a preprocessing problem, not
a shader problem.

Inside inference the ~0.536 ms of profiled GPU compute sits inside the 0.501 ms
small-image wait, and the host work around it is comparable in size:

- recording 0.213-0.281 ms of pure CPU bookkeeping before anything is submitted
- decode 0.188-0.205 ms plus CHW-to-HWC conversion 0.108-0.111 ms on the CPU
- readback overhead outside the wait: alloc 0.044-0.052, copy 0.045-0.059,
  collect 0.029-0.064 ms, so about 0.15 ms total

Ranked recoverable time on this evidence: CPU preprocessing for large images
(48-51), host recording (19-25), CPU output conversion and decode (55), then
readback bookkeeping (11-13). Shader work is not the top target for either
image size. Nothing was adopted or rejected here; the guards and the probe are
retained as measurement infrastructure.

### 11. One readback poll instead of two - kept (no speed gain)

`batch_download` submitted the head copies, blocked until they landed, then
started the maps and blocked again. `map_async` on a buffer with a pending
submission is already deferred until that submission completes, so the maps can
be requested first and one wait drives both.

A/B/B/A in alternating processes, RTX 4090 / D3D12 / FXC, warm, single request,
0.17 MP fixture (the on-device path), 30 samples each. `gpu_readback` p50:

| Order | Variant | gpu_readback p50 | detect_image p50 |
| --- | --- | ---: | ---: |
| A | two polls | 0.700 ms | 1.345 ms |
| B | one poll | 0.697 ms | 1.363 ms |
| B | one poll | 0.693 ms | 1.323 ms |
| A | two polls | 0.717 ms | 1.432 ms |

**No measurable gain.** The reason is visible in the baseline breakdown: with
the copies already waited for, the second poll measured 0.003-0.005 ms, because
the map callbacks had nothing left to wait on. Removing it recovers that and
nothing more, which is below the noise on this path.

Retained anyway on simplicity, not speed: one blocking call instead of two,
twelve fewer lines, and the ordering the rest of the readback group (12-14)
builds on. No performance claim is made for it.

Validation: new `fcs-core/examples/readback_parity.rs` prints an FNV-1a
fingerprint of all 126000 raw output floats over five runs. Both variants gave
`0xa116e42f7c2dabdb` on every run, so the readback is bit-exact, not merely
within tolerance. The 46 GPU tests (ONNX parity, profiled/merged equality) and
`concurrent_inference_matches_sequential` passed under the candidate with
`FCS_STRICT_TESTS=1`. Map errors still propagate per buffer through the same
channel receive.

### 7 (partial). In-process A/B and the noise floor - kept (measurement)

Comparing candidates across separate processes could not resolve this group:
repeated identical runs drifted 0.696-0.779 ms on `gpu_readback` p50 as the GPU
clocks ramped, swamping every effect being tested. `phase_timings --ab VAR` now
alternates an environment flag between 15-run blocks inside one warm process,
8 blocks per variant, and reports a per-phase p50 for each.

A/A control (`--ab FCS_AA_CONTROL`, a flag nothing reads): every phase delta
within **+/-0.003 ms**, `detect_image` +0.000 ms, off-block spread
1.308-1.388 ms. Effects at or below about 0.005 ms stay inconclusive; anything
above roughly 0.01 ms is now resolvable. Experiment 7 is only partly covered:
this is one machine, one thermal state, and no independent-repetition or
power-mode record.

### 12. Pooled readback staging buffers - rejected

Acquired the 12 staging buffers from the existing `GpuBufferPool` and recycled
them after unmap, instead of `create_buffer` per run. In-process A/B, 0.17 MP
fixture, RTX 4090 / D3D12:

| Phase | fresh | pooled | delta |
| --- | ---: | ---: | ---: |
| readback_alloc | 0.042 | 0.003 | **-0.039** |
| readback_wait | 0.487 | 0.533 | **+0.046** |
| gpu_readback | 0.701 | 0.697 | -0.004 |
| detect_image | 1.350 | 1.350 | +0.000 |

The allocation cost is real and the pool removes essentially all of it. It buys
nothing, because it sits in the window between submit and the blocking wait,
where the host is already ahead of the GPU: the wait grows by what the
allocation shed, to within 0.007 ms. Whole-path delta is zero against a
+/-0.003 ms control.

Reverted. Revisit only under actual batch concurrency (60), where host time is
not hidden behind one request's GPU work and allocation churn may matter.
Bit-exact under the fingerprint probe; 46 GPU tests and the concurrency test
passed with it enabled.

### 13. One packed staging buffer - rejected

One pooled or fresh allocation sized to the sum of the 12 head outputs, each
copied to its own `COPY_BUFFER_ALIGNMENT` offset, one `map_async`, one unmap,
and CPU slicing per head. Only each tensor's own byte range is read, so a
larger pooled buffer contributes no stale capacity.

| Phase | 12 buffers | 1 packed | delta |
| --- | ---: | ---: | ---: |
| readback_alloc | 0.043 | 0.009 | **-0.034** |
| readback_wait | 0.483 | 0.521 | **+0.038** |
| readback_collect | 0.020 | 0.018 | -0.002 |
| gpu_readback | 0.698 | 0.690 | -0.008 |
| detect_image | 1.330 | 1.320 | -0.010 |

Same outcome and same cause as 12: one allocation instead of twelve, and the
wait absorbs the difference. The -0.010 ms on `detect_image` is at the edge of
the control's resolution and is not claimed as a gain. CPU slicing cost nothing
measurable, so packing is not what fails here.

Reverted. **The finding that generalises: on the single-image path, host work
between the inference submit and the readback wait is free.** Experiments 12,
13 and anything else that only shortens that window cannot improve latency. The
critical path is what happens before the submit (recording, 0.18 ms), the GPU
work itself (about 0.54 ms), and what happens after the wait (collect 0.02 ms,
CHW-to-HWC 0.095 ms, decode 0.176 ms) - plus CPU preprocessing on large images.

### 14. Head copies encoded with inference - rejected (slower)

Allocated the staging buffers and appended the 12 head copies to the inference
encoder, so one submission carried the forward pass and the readback copies
instead of building a second encoder and submitting again. In-process A/B,
0.17 MP fixture, 8 blocks per variant:

| Phase | separate submit | merged submit | delta |
| --- | ---: | ---: | ---: |
| gpu_encode | 0.314 | 0.380 | **+0.066** |
| gpu_submit | 0.120 | 0.140 | +0.020 |
| readback_alloc | 0.042 | - | removed |
| readback_copy | 0.042 | 0.038 | -0.004 |
| readback_wait | 0.489 | 0.582 | **+0.093** |
| onnx_inference | 1.210 | 1.290 | **+0.080** |
| detect_image | 1.390 | 1.480 | **+0.090** |

**Rejected: 0.09 ms slower**, repeated across two independent runs (+0.070 and
+0.090 ms) against a +/-0.003 ms control. The same effect as 12 and 13, with
the sign reversed. Allocating the staging buffers and recording the copies
costs about 0.084 ms of host time either way. In the existing arrangement that
work happens after the inference submit, while the GPU is busy, and is free. In
the merged version it happens before the submit, where it delays the point at
which the GPU can start, and the wait grows accordingly.

One submission instead of two saved nothing observable in return: the second
submit was itself inside the free window.

Bit-exact under the fingerprint probe (`0xa116e42f7c2dabdb` in both variants),
so this is a latency decision, not a correctness one. Reverted.

**Readback group (11-14) conclusion.** Only the ordering change in 11 was kept,
and it was kept for simplicity rather than speed. The group's shared premise -
that host-side readback bookkeeping costs whole-path time - does not hold on
this path: about 0.13 ms of it sits in a window the GPU is busy through. Any
remaining item in this group that only moves work within that window (15, 16,
17 in its allocation-cost aspect) should be expected to measure zero, and 18's
case has to be made on throughput under concurrency rather than latency. The
recoverable time is before the submit and after the wait.

### 48/49. Threading the source resize - kept

Experiment 5 put CPU preprocessing at 2.65 ms of a 4.68 ms detection on a 10 MP
image, the largest single cost anywhere in the application. `fast_image_resize`
already does the resize with SIMD, but runs it on **one core**: its `rayon`
feature is optional and the workspace did not enable it.

**The measurement had to be fixed first.** Enabling the feature and comparing
two binaries gave a contradictory answer - 10 MP looked slightly better, 22 MP
looked 0.6 ms worse - because this machine's CPU throughput moved by **1.6x
between two builds of identical code** (10 MP preprocessing measured 2.66 ms in
one state and 4.82 ms in another, stable within each). No cross-build CPU
comparison on this machine is trustworthy.

`fast_image_resize` reads its thread count from `rayon::current_num_threads()`,
so both variants can run in one process: a one-thread pool is the
single-threaded build, the default pool is the threaded one.
`fcs-core/examples/resize_threading.rs` alternates them, 8 blocks of 10 runs
each, timing `resize_image` alone - not through `preprocess_dynamic_image`,
whose BGR/CHW conversion is already rayon-parallel and reacts to the same pool.

**Ungated, feature on** (640x640 output, 32-thread pool):

| Source | Filter | 1 thread | all cores | speedup |
| --- | --- | ---: | ---: | ---: |
| 0.1 MP | Bilinear | 0.467 | 0.683 | 0.68x |
| 0.2 MP | Bilinear | 0.478 | 0.816 | 0.59x |
| 0.6 MP | Bilinear | 0.530 | 0.669 | 0.79x |
| 1.1 MP | Bilinear | 0.670 | 0.789 | 0.85x |
| 2.5 MP | Bilinear | 0.822 | 0.921 | 0.89x |
| 5.5 MP | Bilinear | 1.450 | 1.143 | **1.27x** |
| 10.1 MP | Bilinear | 2.080 | 1.317 | **1.58x** |
| 22.1 MP | Bilinear | 3.487 | 2.995 | **1.16x** |
| any | Nearest | 0.165-0.218 | 0.291-0.308 | 0.54-0.73x |

Fork and join cost a roughly fixed 0.10-0.17 ms. That is most of a small
resize and a fraction of a large one, so the crossover sits near **4 MP**.
Nearest never wins: it gathers one source pixel per output pixel and finishes in
0.17-0.22 ms even at 22 MP, less than the cost of distributing it.

**Kept: the feature plus a size and filter gate.** `resize_image_fast` runs
inside a one-thread pool below `RESIZE_THREADING_MIN_PIXELS` (4 MP) or for
Nearest, and on the default pool above it. Re-measured with the gate in place:

| Source | Filter | 1 thread | all cores | speedup |
| --- | --- | ---: | ---: | ---: |
| 0.1-0.2 MP | Bilinear | 0.461-0.489 | 0.456-0.483 | 1.01x |
| 10.1 MP | Bilinear | 2.106 | 1.497 | **1.41x, -0.609 ms** |
| 22.1 MP | Bilinear | 3.526 | 2.915 | **1.21x, -0.610 ms** |
| any | Nearest | 0.170-0.206 | 0.166-0.200 | 1.03x |

So roughly **0.6 ms off every large-image detection** and nothing lost at any
other size or on the `Speed` setting - which matters, because `Speed` exists for
batch throughput and ungated threading would have made it 0.55-0.73x.

The gate is one constant for every machine; the crossover is a function of core
count and memory bandwidth, and the probe re-measures it. Marked `ponytail:` in
the source.

Quality is unchanged: threading splits the separable convolution by rows, and
the new `threaded_and_single_threaded_resize_agree` test asserts byte-identical
output for Triangle, Lanczos3 and Nearest across a 4.3 MP source resized both
ways. Detections cannot depend on the machine's core count.

Experiment 49 is answered in the same pass: the source path was traced looking
for redundant conversions and there are none to remove. `resize_image_fast`
borrows the decoded `RgbImage` when the source is already RGB8 (`as_rgb8`), and
a JPEG decode produces exactly that, so no full-resolution conversion or copy
happens before the resize. The 3.0-3.2 ms full-resolution `to_rgba8` that
`preprocess_cost` reports belongs only to the GPU preprocessing path, which
declines above 2.5 MP and so never pays it on these images.

**Second pass, after the `Resizer` fix (experiment 19) exposed it: stop zeroing
the output buffer.** With the resize's own `memset` gone, the remaining one was
`rgb_to_bgr_chw` allocating `vec![0.0f32; 3 * 640 * 640]` -- 4.9 MB zeroed and
then completely overwritten by the parallel conversion loop, 3.1% of all CPU in
the re-taken profile.

The buffer is now `Vec::with_capacity` plus `spare_capacity_mut`, written
through `MaybeUninit::write`, with one `set_len` afterwards. The traversal was
pulled into `fill_bgr_planes`, generic over the element type, so the initialised
and uninitialised paths cannot drift and "every element is written" stays a
single checkable loop. `f32` has no destructor, so an unwind mid-loop leaves a
length-0 `Vec` with nothing to drop.

In-process A/B on `FCS_ZEROED_CHW`, 10 MP fixture:

| Run | preprocess, uninit | preprocess, zeroed | delta |
| --- | ---: | ---: | ---: |
| 1 | 2.370 | 2.460 | +0.090 |
| 2 | 2.400 | 2.570 | +0.170 |
| 3 | 2.520 | 2.900 | +0.380 |

Plus four `detect_image` runs across 10 MP and 22 MP, all favouring the
uninitialised buffer by 0.07-0.44 ms. Machine state was slow during these runs,
which inflates a bandwidth-bound cost; call it **0.1-0.2 ms typical**,
direction unambiguous at 7/7 runs.

This is the one place in the round where `unsafe` was accepted. The existing
tests spot-checked three of twelve elements, which cannot see a gap in coverage
that `set_len` would then expose, so
`rgb_to_bgr_chw_writes_every_element` now compares **every** element against an
independently computed reference at 37x23 -- a size sharing no factor with any
chunking the implementation might use.

**Where preprocessing stands now.** At 10 MP the resize is 1.60 ms of a 2.33 ms
preprocess; the remaining 0.73 ms is the BGR/CHW conversion and is
size-independent, since it works on the 640x640 result. The next real candidate
is uploading the 640x640 RGB as 1.2 MB of `u8` and doing the BGR/CHW/f32
conversion in a shader, which would remove both that CPU work and three
quarters of the 0.28 ms input upload. Not attempted: it needs a third
preprocessing path alongside the existing CPU and fully-on-device ones.

Note: `preprocess_cost.rs` reports "0.00 ms" for GPU preprocessing whenever
`upload_pays_for_source` declines, and then names GPU the winner. Its
crossover table is wrong for any size above the decline threshold; the numbers
above come from `resize_threading.rs` instead.

### 9/19. Native CPU profile of warm detection - kept (one finding acted on)

`samply` over the `inference_pipeline` bench, 25 s of warm detection on the
10 MP fixture, `--main-thread-only`, release with `strip=none` and
`debug=line-tables-only`. Two profiles: `detect_image/gpu_quality` (GPU
inference, CPU preprocessing - the path a large image actually takes) and
`detect_image/quality` (CPU inference) for contrast.

A caveat worth recording: `quality` and `speed` in that bench are **CPU
inference**; the GPU cases are `gpu`, `gpu_quality` and `gpu_on_device`. The
first profile taken was of the CPU backend by mistake, and its top entry was
`fcs_core::cpu::conv2d`, which is not on the shipped GPU path at all.

Leaf self-time alone was not usable - the top entry was `memset_repmovs` at
8.6% of CPU, which says nothing about who asked for zeroed memory. A small
companion script walks each sample's stack from the leaf to the nearest named
frame and aggregates there. Attributed `memset` on the GPU path:

| Blamed frame | CPU | Share |
| --- | ---: | ---: |
| `fast_image_resize::Resizer::resample_convolution` | 476 ms | **4.1%** |
| `fcs_core::preprocess::cpu_preprocess` | 462 ms | **4.0%** |
| `fcs_core::model::decode_yunet_outputs` | 39 ms | 0.3% |
| `wgpu_hal::dx12::Device::load_shader` | 34 ms | 0.3% |
| `GpuYuNet::run_inference` | 29 ms | 0.2% |

**Per-layer host bookkeeping is not the problem.** `run_inference` accounts for
about 0.9% of CPU in self time, and its `memset` share is 0.2%. Nothing in the
graph traversal, weight lookup or label handling that experiment 19 proposed
attacking shows up. The 0.18-0.28 ms of recording measured in experiment 5 is
real wall time but it is not concentrated anywhere a change could reach; no
bookkeeping change was made, and 20-25 should not be started on the strength of
19 alone.

**What the profile did find: a re-zeroed scratch buffer.** A separable
convolution writes an intermediate image between its horizontal and vertical
passes, and `fast_image_resize` keeps that buffer inside the `Resizer`, growing
it on demand and zeroing only the new part. `resize_image_fast` built a
`Resizer` per call, so the buffer was thrown away and re-zeroed every time - for
a 10 MP source the intermediate is 640x4240x3 = 8.1 MB.

Kept: one `Resizer` per thread in a `thread_local`, alive between calls.
In-process A/B on `FCS_FRESH_RESIZER` (per-call `Resizer` restored), five runs:

| Image | preprocess, cached | preprocess, fresh | delta |
| --- | ---: | ---: | ---: |
| 10.1 MP | 2.830 | 3.100 | +0.270 |
| 10.1 MP | 2.280 | 2.430 | +0.150 |
| 10.1 MP | 1.990 | 2.110 | +0.120 |
| 10.1 MP | 2.180 | 2.220 | +0.040 |
| 22.1 MP | 4.810 | 5.070 | +0.260 |
| 0.17 MP | (GPU preprocess path) | | +0.000 |

Every run favours the cached `Resizer`; the size varies 0.04-0.27 ms because
CPU throughput on this machine wanders far more than the GPU path does (off
blocks spread 4.17-6.99 ms on one 10 MP run). Call it **roughly 0.1-0.15 ms
typical on large images**, direction unambiguous, and exactly zero on small
images, which never reach the CPU resize.

Cost: one retained scratch buffer per thread that has ever resized, sized to
the largest source that thread has seen. Thread-local rather than shared
because `resize` needs `&mut` and a mutex would serialise the batch path.

**Not acted on, but measured:** the other 4.0% of `memset` is inside
`cpu_preprocess` - `rgb_to_bgr_chw` allocating `vec![0.0f32; 3*640*640]`
(4.9 MB) and `FirImage::new` allocating the 1.2 MB destination, both zeroed and
then completely overwritten. Removing that zeroing needs either `unsafe` around
uninitialised memory or a reusable buffer the output tensor cannot take
ownership of, since `chw_tensor_from_vec` consumes the `Vec`. Left alone: it is
a comparable prize to the `Resizer` fix but a materially worse trade in safety
and API churn.

### 55. Output conversion and decode - partly kept

Two candidates, measured separately by in-process A/B on `FCS_OLD_CONVERT`,
0.17 MP fixture (the path where conversion is the largest share of detection).

**Kept: stop copying two buffers that were already owned.** `Tensor::from_vec`
exists precisely to take ownership, and both hot paths were calling
`Tensor::from_shape`, which copies:

- `build_decode_tensors` copied each of the 12 head buffers that
  `reorder_hw_major` had just allocated
- `decode_yunet_outputs` copied the fully written 8400x15 `fused` buffer

Three runs, every one identical to 0.001 ms:

| Phase | from_vec | from_shape (copies) | delta |
| --- | ---: | ---: | ---: |
| gpu_convert | 0.087 | 0.097-0.099 | **-0.011** |
| gpu_decode | 0.166-0.169 | 0.176-0.180 | **-0.010 to -0.013** |
| onnx_inference | 1.140-1.170 | 1.160-1.200 | **-0.020 to -0.030** |
| detect_image | 1.300-1.340 | 1.330-1.370 | **-0.020 to -0.030** |

About **0.02-0.03 ms** for deleting two `to_vec` calls, repeatable to within
0.01 ms across runs and well clear of the +/-0.003 ms control. Raw output is
bit-identical (`0xa116e42f7c2dabdb`).

**Rejected: reordering the reorder loop.** `reorder_hw_major` walks
channel-major, so it reads `data` sequentially and writes `out` with a stride of
`channels` -- for the 10-channel keypoint heads, a fresh cache line per store.
The obvious rewrite walks pixel-major instead: sequential writes, strided reads,
`Vec::with_capacity` and `push` rather than a zeroed vector and indexed stores.

It was **slower**, by 0.004-0.011 ms on `gpu_convert` in four consecutive runs.
The strided stores are not what this loop is limited by, and the `push` bounds
and capacity checks cost more than the store pattern saves. Reverted, with the
measurement recorded in a comment so it is not retried blind.

**Kept, and the largest single win in this group: decode straight from CHW.**
The reorder existed only because `decode_stride_outputs` indexes
`bbox[cell * 4 + c]`, which is cell-major, while the GPU heads are channel-major
planes of raw logits. Rather than transposing twelve buffers so the decoder can
read them the way it likes, the decoder was taught the other layout.

`HeadLayout` has exactly two variants, named for the two producers rather than
as a general matrix, because layout and activation travel together:
`CellMajorActivated` (ONNX Runtime, tract, the CPU graph) and
`ChannelMajorLogits` (the GPU heads). `decode_yunet_outputs` keeps its signature
and its old behaviour; `decode_yunet_outputs_with` takes the layout. The cell
loop is shared and monomorphised over an index closure, so the branch is
resolved once per stride rather than 8400 times. Sigmoid moves into the decoder,
applied to `cls` and `obj` only - the same two heads the reorder used to
activate. `reorder_hw_major` is deleted.

In-process A/B on `FCS_OLD_CONVERT`, three runs, 0.17 MP fixture:

| Phase | CHW decode | transpose first | delta |
| --- | ---: | ---: | ---: |
| gpu_convert | 0.001-0.002 | 0.086-0.098 | **-0.085 to -0.097** |
| gpu_decode | 0.072-0.082 | 0.131-0.140 | **-0.058 to -0.059** |
| onnx_inference | 0.985-1.160 | 1.130-1.330 | **-0.130 to -0.170** |
| detect_image | 1.150-1.380 | 1.290-1.600 | **-0.140 to -0.220** |

About **0.15-0.2 ms off every detection, roughly 11-15% of a small-image
detection**, and the largest saving found anywhere outside preprocessing.

Two things were removed, not one. The transpose itself is now essentially free
(0.001 ms - the guard wraps twelve `Tensor::from_vec` calls and nothing else).
But **decode also got 0.058 ms faster**, which was not the intent: reading the
downloaded buffers directly leaves the twelve freshly written transposed buffers
out of cache entirely, and the decode's gather is no worse for being strided.
That half of the gain is an observation, not a prediction that would have been
made in advance.

Validation: the raw fingerprint is unchanged (`0xa116e42f7c2dabdb` in both
variants), so the decoded rows are bit-identical, not merely within tolerance.
New `channel_major_logits_decode_like_cell_major_activated` builds the same
values in both layouts - transposing and activating by hand, independently of
the implementation - and asserts the two decodes agree, so channel-major
indexing cannot silently invert. The whole workspace passes under
`FCS_STRICT_TESTS=1` with ONNX Runtime 1.24.4, including ONNX raw parity,
GPU/CPU parity and concurrent inference.

**This removes the premise of experiment 38.** That item proposed making the
final layer write HWC so the CPU would not have to reorder. There is no CPU
reorder left to save, and this cost no GPU time at all, so 38 would now have to
justify itself on GPU store coalescing alone. It stays unchecked, but its
stated motivation is gone.

### 50. Upload bytes, not floats, for CPU-resized sources - kept

Instrumenting the two halves of CPU preprocessing separately (`cpu_resize` and
`bgr_chw` guards) corrected an earlier estimate made by subtracting one probe's
number from another's, which is invalid across processes on this machine. The
real split at 10 MP is **resize 1.71 ms, conversion 0.32 ms**, not 1.60 / 0.73.

That still left two costs the GPU could take: the u8-to-f32, RGB-to-BGR,
interleaved-to-planar conversion (0.32-0.40 ms, on the CPU), and the 4.9 MB
float upload that followed it (`gpu_upload`, 0.36-0.47 ms).

Kept: `WgpuPreprocessor::resize_then_convert`, a third route between the two
that existed. Large sources are still resized on the CPU -- uploading a 10 MP
image whole is what `upload_pays_for_source` rejects, and that has not
changed -- but the 640x640 result now goes up as **1.2 MB of bytes** and
`rgb_to_chw.wgsl` writes the tensor directly. `preprocess_into_tensor` no longer
declines for large images, so `detect_on_device` handles every size.

In-process A/B on `FCS_CPU_CHW`, `detect_image` p50, GPU route minus CPU route:

| Image | Runs | Delta |
| --- | --- | ---: |
| 10.1 MP | 6 | -0.03, -0.59, -0.67, -0.69, -0.89, -0.98 |
| 22.1 MP | 5 | -0.45, -0.75, -1.02, -1.07, -1.09 |

**Roughly 0.6-1.0 ms off every large-image detection**, the largest single win
in this round. Phase-level: `gpu_rgb_to_chw` at 0.33-0.47 ms replaces
`bgr_chw` 0.33-0.40 plus `gpu_upload` 0.36-0.47, and the host no longer
allocates a 4.9 MB f32 tensor per image.

The shader does no sampling and no arithmetic -- each output float is exactly
the integer value of one source byte -- so this is bit-exact, not
within-tolerance. `resize_then_convert_matches_cpu_preprocess` asserts every
float equals `cpu_preprocess` at 457x311 to 70x46, chosen so the source row
length is not a multiple of four bytes and the destination is not a multiple of
the 64-wide workgroup.

**That test earned its keep immediately.** `Queue::write_buffer` requires the
*copy length* to respect `COPY_BUFFER_ALIGNMENT`, not merely the buffer size,
and three bytes per pixel is only a multiple of four for some image sizes.
640x640 is one of them, so production would never have hit it; 33x17 panicked.
The aligned prefix is now written separately from a zero-padded tail word.

Not pursued: `queue.write_buffer` still copies 1.2 MB into wgpu's staging belt,
and the source and uniform buffers are created per call. Resizing straight into
a `mapped_at_creation` buffer would remove both, but needs `resize_image` to
write into a caller-provided slice.

**Also assessed and not adopted: a cheaper resize algorithm.** `cpu_resize` is
now the largest cost in the application by a wide margin (1.71 ms at 10 MP,
3.56 ms at 22 MP - 84% of preprocessing). `fast_image_resize` offers
`SuperSampling(filter, multiplicity)`, which is far cheaper for large
downscales. Reading its implementation, the first step is `resample_nearest` -
it *discards* source pixels before convolving, so a 3.7x downscale would
average a handful of the ~14 source pixels per output pixel. That is precisely
the failure `preprocess.wgsl` already documents from the GPU side, where a
fixed small kernel "moved real detections (23px on a landmark)".
`Interpolation(filter)` has the same problem by construction. Either is a Q
candidate needing a recall and landmark-error budget and a corpus, not a
drop-in; neither was measured, and no quality claim is made about them here.

### 51. Cheaper resize algorithms - rejected, and the avenue is closed

After experiment 50, `cpu_resize` is the largest single cost in the application:
1.39 ms at 10 MP, 2.78 ms at 22 MP, against 0.31 ms for the GPU conversion and
1.39 ms for all of inference. Nothing else is close. `fast_image_resize` offers
two cheaper algorithms, and both are quality-changing, so a harness had to come
first.

**The harness.** `fcs-core/examples/resize_quality.rs` runs the whole detector
twice over the fixture corpus -- production's resize, then a candidate selected
by `FCS_RESIZE_ALG` -- and reports faces lost and gained, landmark displacement
in source pixels, box IoU and score deltas. No ground truth is needed or used:
production is the reference, and the question is only whether a candidate moves
detections relative to it. A/A control (an unrecognised algorithm name, which
falls through to production) over 40 images: **0 lost, 0 gained, 0.00 px
landmark shift, IoU exactly 1.0000** -- so any difference below is the
candidate, not the harness.

**Quality**, 120 images, 51 matched faces:

| Candidate | Lost | Gained | Landmark p50 | p95 | max | IoU min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `SuperSampling(Bilinear, 2)` | 0 | 0 | 0.00 | 0.00 | 1.04 px | 0.9970 |
| `SuperSampling(Bilinear, 4)` | 0 | 0 | 0.00 | 0.00 | 0.00 px | 1.0000 |
| `Interpolation(Bilinear)` | 0 | 0 | **1.50** | **7.00** | **35.39 px** | 0.9526 |

`Interpolation` reproduces the failure `preprocess.wgsl` already documents from
the GPU side -- "moved real detections (23px on a landmark)" -- and this time
with a distribution behind it. A 35 px landmark shift places a crop visibly
wrong. Rejected on quality regardless of speed.

`SuperSampling(_, 4)` is identical to production because it never engaged: it
only splits into two steps when `min(scale) / multiplicity > 1.2`, and at a
3.7x downscale that is 0.93. Its row is an accidental second A/A control.

**Speed**, in-process A/B on `phase_timings --ab FCS_RESIZE_ALG=<alg>`,
`cpu_resize` p50 (A/A control on this path: +0.020 ms):

| Candidate | 10 MP | 22 MP |
| --- | ---: | ---: |
| `SuperSampling(Bilinear, 2)` | **+0.17 to +0.22 (slower)** | - |
| `SuperSampling(Bilinear, 3)` | **+1.23 to +1.43 (slower)** | - |
| `Interpolation(Bilinear)` | -0.69 to -0.89 | -1.95 |

**SuperSampling is slower, which settles the avenue.** Its first step is a
nearest-neighbour downscale, and that still reads every source pixel; the
convolution then runs over the intermediate as well. It adds a pass without
removing the dominant one. `Interpolation` is fast for exactly the reason it is
inaccurate: a fixed two-tap kernel reads about four source pixels per output
pixel instead of the ~14 the downscale ratio calls for, so it never touches most
of the image.

The conclusion generalises past these two candidates: **at a fixed source
resolution the resize is bounded below by reading the source once**, and
production's adaptive-kernel convolution already does that and nothing more.
There is no faster *correct* algorithm to find here. The remaining lever is to
make the source smaller before it is read, which is a decode-stage change (see
the new experiment 86), not a resize-stage one.

Nothing adopted. `FCS_RESIZE_ALG` is retained as evaluation scaffolding in
`fir_alg`, since experiments 54, 74, 59 and 79 are all quality-changing and this
harness is what makes them decidable; the per-call `var_os` lookup is on the
order of a microsecond against a 1.3 ms resize and did not separate from noise
in the A/A control.

### Previous work

The compute-pass merge is already shipped in the baseline: roughly 0.40 ms saved in paired
encode/finish/submit/wait measurements, with all 61 profiling records retained.
