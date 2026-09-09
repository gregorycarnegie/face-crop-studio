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
- [x] **6. Build a representative workload matrix.** Cover small/large images,
  portrait/landscape, supported formats, no/one/many faces, difficult small
  faces, warm preview, first detection, webcam and folder export. Record p50/p95
  latency, images/s, CPU use and peak RAM/VRAM where relevant.
- [ ] **7. Quantify timing noise and minimum detectable gains.** Repeat
  identical A/A controls, alternating A/B order and independent runs; record
  clocks, power mode, thermals and competing GPU work. Select sample duration
  from observed variance; treat isolated timestamp ticks as inconclusive.
- [x] **8. Compare profiled and normal execution costs.** Measure one merged
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
- [x] **17. Reuse CPU output storage.** Compare new vectors/channels/temporary
  tensors per detection with appropriately scoped reuse or decoding from a
  mapped view. Measure copies and allocation cost; release mapped memory
  promptly and preserve error handling and concurrent callers. Handed over by
  57. Half kept: the head buffer was copied out of the mapping and then cut into
  branches, and the branches now come straight off the mapping instead
  (-0.011 to -0.021 ms). Decoding from the mapping was measured and **rejected**
  -- reads there are 55% dearer than reads out of a `Vec`, more than the copy
  they would remove.
- [ ] **18. Overlap readback with the next input.** After 11-17, test a bounded
  two/three-slot staging ring if throughput warrants it. Measure frame latency,
  throughput and VRAM, including slow consumers and cancellation. Depends on
  explicit per-request ownership; not a single-image latency claim.

### CPU recording and resource allocation (P1)

- [x] **19. Profile and reduce per-layer host bookkeeping.** Measure shape
  validation, graph traversal, weight-name lookup, temporary collections,
  labels and reference counting in warm inference. Pre-resolve only repeated
  immutable data that is a measured cost; keep public input validation.
- [x] **20. Cache remaining small-operation uniforms.** Measure max-pool,
  add and resize uniform creation after convolution caching. Reuse immutable
  content-keyed values only if the eight dispatches contribute a repeatable
  cost; include cache growth under varied resolutions.
- [ ] **21. Compare per-inference fixed intermediate plans with pooling.**
  Precompute shapes/lifetimes for the fixed YuNet graph and allocate a bounded
  workspace per in-flight request. Measure recording time and peak memory;
  protect current safe reuse and never share writable intermediates concurrently.
- [x] **22. Cache bind groups with stable buffer identities.** Depends on
  evidence from 19/21. Compare construction cost against retained groups per
  workspace, with correct input/output identities and invalidation. The old
  0.107 ms figure is historical, not an expected current saving. Closed once on
  reasoning -- "the pooled intermediates' identities change between passes" --
  which was an assumption and was wrong. Reopened and measured: **-60% of
  recording, -6% of a detection**. See the record.
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

- [x] **26. Sweep pointwise workgroup shapes.** Start with a small explicit
  set such as 8x8, 16x4, 32x2 and 8x4, adjusting dispatch coverage consistently.
  Measure large layers and tiny heads, raw tails and full graph; select per-shape
  kernels only when the gain survives dispatch/selection overhead.
- [x] **27. Sweep pixels and output channels per thread.** Extend the existing
  1/2/4-channel comparison to bounded combinations of spatial width and channel
  tiles, including 8 channels where limits permit. Compare register pressure,
  tail waste and occupancy; do not assume a larger tile wins. 8's occupancy
  reading suggested a *smaller* tile; the sweep covered both ends and the
  prediction was wrong -- see the record.
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
- [x] **34. Specialize the remaining general/stem convolution.** Measure the
  3x3, stride-2, three-input-channel stem separately; try fixed loops and input
  reuse. Its measured roughly 41 us is a small ceiling, so stop if gains do not
  survive whole-graph measurement. The answer was neither loops nor the stem
  specifically -- see the record.

### Graph fusion and intermediate traffic (P2)

- [ ] **35. Fuse depthwise then pointwise.** Prototype one expensive adjacent
  pair, keeping any intervening activation in its original position. Compare
  fewer intermediate reads/writes with recomputation, registers and halo costs;
  require intermediate/raw-head and final-detection parity.
- [ ] **36. Fuse compatible pointwise/add/resize or pool boundaries.** Pick
  one actual graph pattern with measured traffic/dispatch cost. Preserve
  operation order and fan-out consumers; compare with already-merged passes,
  since eliminating a pass is not a new saving here.
- [x] **37. Compute detection head branches together.** Test sharing input
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
  preprocessor contract unless explicitly evaluating a Q variant. **92 tested the
  cheap half of this** -- sharing the compute pass without fusing the shaders --
  and it lost, because the preprocess dispatch currently overlaps the recording of
  inference. A real fusion would remove the dispatch rather than move it, so 39 is
  not settled by 92, but it inherits the warning.
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
- [x] **52. Cache preprocessing for unchanged preview input.** First inspect
  current GUI caches and invalidation. Test reuse across changes that only affect
  crop presentation/enhancement; invalidate for image, orientation, input-size
  or detector changes. Measure repeated-interaction latency and retained memory.
- [~] **53. Keep webcam frames on the GPU where capture permits.** Investigate
  native texture/frame import or fewer colour-conversion copies. Count capture-
  to-result latency, synchronization and format conversion; retain the portable
  fallback and do not assume the capture API exposes compatible device memory.
  **Premise removed by 95:** the loop is capture-bound at 24 fps and does its work
  in a seventh of the frame budget, so removing copies cannot raise the frame
  rate. Re-open for a camera fast enough to saturate the pipeline.
- [x] **54. Evaluate adaptive resize/input routing.** Test a measured CPU/GPU
  cutoff by source size/device and, separately, cheaper preview resize quality.
  Include routing overhead. Quality changes require detection/landmark/crop
  evaluation and distinct settings; smaller input is also covered by 74. Q.

### Detection output conversion and filtering (P1/P2)

- [x] **55. Combine CPU reorder, sigmoid and decode work.** Profile current
  traversals and temporary tensors, then fuse one measured redundant pass.
  Compare exact output ordering and established numerical tolerances; avoid
  reproducing the already-rejected fine-grained loop parallelism.
- [~] **56. Move score/box decoding to the GPU.** Compare GPU decode plus a
  readback against current CPU decoding, including dispatch and map overhead.
  Preserve score calculation, anchors, dimensions and landmark conventions;
  merely moving tiny CPU work to the GPU can lose. **Premise removed by 55:**
  after the CHW decode change `gpu_decode` is 0.072-0.082 ms, about 2.6% of a
  3.1 ms detection, so the whole stage is smaller than this experiment's own
  warning about dispatch overhead -- and 55 already measured moving the adjacent
  conversion to the GPU as *slower*. Reopen only if decode grows: a much larger
  detector input or a model with many more anchors.
- [x] **57. Compact valid candidates before readback.** Use the existing score
  rule and threshold to reduce output bytes; measure sparse and crowded cases,
  counter/scan overhead and overflow handling. Preserve ordering/tie semantics
  where observable; no arbitrary top-K cap to manufacture speed.
- [x] **58. Reprofile NMS on worst-case candidate counts.** Compare current
  spatial-grid NMS with bounded CPU or GPU alternatives only if it is material.
  Validate overlaps, equal-score ties and dense scenes; GPU transfer/dispatch
  cost belongs in the comparison.
- [~] **59. Evaluate approximate candidate pruning separately.** Try top-K,
  alternate thresholds or approximate NMS only against an explicit recall and
  landmark-quality budget. Measure difficult and crowded scenes, not just
  average timing. Q; never substitute these for exact-path optimization.
  **Premise removed with 56:** this trades recall for time in a stage measured at
  0.072-0.082 ms. There is no time here worth any recall. 58 stays open, because
  worst-case candidate counts are a robustness question rather than a throughput
  one, as is 57, which targets readback bytes rather than decode CPU.

### Batch and webcam scheduling (P1/P2)

- [x] **60. Measure bounded GPU concurrency.** Compare 1/2/3/4 in-flight
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
- [x] **63. Tune CPU thread budgets alongside GPU work.** Compare rayon and
  runtime thread counts on single/batch workloads; detect oversubscription,
  driver starvation and memory-bandwidth contention. Record CPU-only results
  as well; avoid a global setting chosen from one developer machine.
- [~] **64. Compare a GPU submission worker with caller-thread submission.**
  Gated on 9/24/60 showing contention or idle gaps. 60 shows neither: throughput
  peaks exactly at one worker per logical processor and falls away above it,
  which is the shape of a CPU-bound schedule, not a contended queue. No premise
  left unless a different machine shows one.
  Only if 9/24/60 show contention or idle gaps, test a bounded dispatcher.
  Include handoff latency and fairness; do not introduce a worker/thread solely
  as an abstraction or serialize independent CPU work unnecessarily.
- [~] **65. Prefer fresh webcam frames under overload.** Compare queued-all
  processing with bounded latest-frame scheduling, skipping stale detections
  when a newer frame supersedes them. Report capture-to-display age, dropped
  frames and detection cadence as well as fps. Q; export must remain complete.
  **Premise removed by 95:** there is no overload to schedule around -- capture
  delivers every 42 ms and the pipeline answers in 3-9. Re-open if a faster
  camera or several at once makes the pipeline the constraint.
- [ ] **66. Share work across identical preview requests.** Inspect existing
  cancellation/caches, then test coalescing duplicate in-flight detections.
  Measure rapid UI edits and mixed images; prevent stale results from replacing
  current ones and preserve errors/cancellation for each requester.

### Decode, CPU execution and export (P1/P2)

- [x] **67. Benchmark alternative full-resolution decoders.** Compare available
  implementations on the actual format corpus, including orientation/colour
  fidelity, cold I/O, warm cache and batch throughput. Preserve full-resolution
  crop pixels; this differs from the rejected reduced-scale decode workflow.
- [x] **68. Tune file I/O and bounded decode prefetch.** Measure cold disk,
  cached files and large folders separately. Compare modest prefetch depth and
  reuse of read buffers; include peak RAM and cancellation. Skip if decode or
  inference dominates and I/O is already hidden.
- [x] **69. Compare CPU inference settings and layout costs.** Benchmark the
  shipped tract and ONNX Runtime paths with bounded thread/optimization settings,
  warm sessions and real batches. Profile tensor conversion and memory copies;
  a faster isolated runtime is not necessarily a faster application.
- [ ] **70. Apply measured CPU vectorization/PGO changes.** Inspect hot-loop
  assembly first; compare compiler flags or representative profile-guided
  optimization with the current x86-64-v3/autovectorized build. Include portable
  fallbacks and startup/binary-size costs; do not re-add `wide` without evidence.
- [x] **71. Keep crop/enhancement intermediates on device.** Trace actual
  filters and crop batches, then remove measured intermediate downloads/uploads
  or fuse compatible filter passes. Include final export readback and verify
  pixel quality plus operation ordering at production image sizes.
- [x] **72. Measure export encoding and concurrency.** Compare encoder settings
  and bounded encode/write parallelism after detection accelerates. Report total
  export time, output size and peak memory. Compression/quality changes are Q;
  identical settings and pixels are the baseline.
- [ ] **73. Evaluate two-stage decoding only for suitable workloads.** For
  mostly no-face or detection-only input, test a cheap screening decode followed
  by full decode only when needed. Count both decodes for positives and all
  missed faces. Q; rejected for normal crop-heavy work unless new evidence
  changes that workload assumption.

### Model and algorithm changes (Q/P2)

- [~] **74. Sweep detector input resolution.** Compare supported smaller/larger
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

- [x] **80. Split cold start into adapter, model and pipeline costs.** Measure
  process launch, device selection, model read/parse/weight upload, shader compile
  and first detection separately from steady state. Compare cached and clean
  runs; prevent lazy initialization from hiding cost in the first user action.
  Answered: `request_adapter` 60%, one shader 22%, model loading 0.15%, nothing
  deferred into the first detection.
- [ ] **81. Reuse device/model/pipelines across real UI lifetimes.** Check
  existing sharing, then eliminate measured accidental recreation across preview,
  webcam and export. **Half-answered by 80:** the GUI already shares eframe's
  device, so it never pays the 546 ms adapter bring-up. What remains is that
  `build_detector` runs synchronously in `App::new`, putting ~235 ms of shader
  compilation ahead of the first frame. Include device loss, switching models and shutting down;
  longer lifetime must not produce unbounded retained GPU memory.
- [x] **82. Evaluate supported pipeline caches or controlled prewarming.**
  **Sized by 80:** the target is one shader, `conv2d.wgsl` at 197 ms of FXC; the
  other four are 20 ms together, so parallel compilation is not the answer.
  `PIPELINE_CACHE` turned out to be Vulkan-only, so the answer was one entry point
  per kernel instead: ~197 ms to ~120, and the grouped fallback is never compiled
  in production.
  Feature-check the active backend, compare cold/warm startup and first-frame
  latency, and invalidate persisted data by compatible device/driver/shader
  identity. Include cache size and total work; moving compilation earlier is
  not the same as making startup cheaper.
- [~] **83. Bound specialization and uniform-cache growth.** Sweep many input
  resolutions/configurations and compare memory, hit rate and eviction cost.
  Test a bounded cache or known-model prepopulation only if growth matters;
  keep content-based correctness and concurrent access safety. **Premise removed
  by 74 and 84:** production supports exactly one detector input (640x640), so
  the conv caches cannot grow from a resolution sweep, and 84 measured the pools
  flat at 44.1 MB over 400 sources up to 23.4 MP. Reopen if a second input size
  ever becomes reachable.
- [x] **84. Tune pool retention under mixed workloads.** Compare current
  retention with bounded high-water marks or idle trimming across large images,
  smaller follow-up runs and concurrent exports. Measure allocation churn,
  p95 latency, VRAM pressure and device failures; memory savings may trade speed.
  Measured: GPU retention is already flat and bounded; the memory that scales is
  host RSS with worker count, ~85 MB each. No change made -- see the record.
- [x] **94. Give the readback poll a timeout.** Every `device.poll` passed
  `PollType::Wait { timeout: None }`, so a submission that never completes blocked
  the calling thread forever instead of returning an error the caller can report
  or retry. Observed once during 82's validation: a test binary sat for ten hours
  on 35 seconds of CPU across a long idle window and had to be killed. Answered,
  and the premise turned out to be wrong. The deadline is 30 seconds and the five
  duplicated wait sites are now one `wait_for_gpu` -- but the hang reproduced
  under `cdb` and no thread was in a wait at all: they were all inside the NVIDIA
  driver creating D3D12 resources, because the test harness built one device per
  test and ran 32 at once. One shared context fixed that and made the `fcs-utils`
  suite 2.6x faster.

- [ ] **85. Validate sustained operation and power efficiency.** Run the best
  candidates through long webcam sessions and large exports, including VRAM
  pressure and background GPU work. Track drift, thermals, memory growth,
  responsiveness and energy/image where measurable; short warm microbenchmarks
  can miss production regressions.
- [x] **96. Letterbox instead of stretching to the model input.** The
  preprocessor scaled x and y independently, so every non-square source reached
  the model distorted and its score fell. Measured over all 1239 images at
  production's threshold: 77 images gain a detection, 19 lose one, and every box
  on a non-square source moves (median IoU 0.76-0.87, landmarks 34-46 px). The
  crops were compared both ways on a real folder, which is the standard 71 was
  held to; **adopted**, and the corpus goes from 1030 detections to 1130.
- [x] **97. Detect on every webcam frame in the GUI.** 95 showed the headroom;
  this spends it, and measures the two things a CLI probe could not: the GUI's
  per-frame texture cost (1.8 us, free) and detection under contention with the
  renderer (6.89 ms against 3.5 standalone). 95% of frames tracked at 15 fps.
- [x] **95. Measure the webcam path.** Where a frame's time goes, and what the
  loop is actually limited by. Answered 53 and 65 and turned up two defects.
- [x] **93. Skip decoding cells that cannot reach the score threshold.** The
  decode runs on all 8400 cells and postprocessing discards nearly all of them.
  Kept: 82% off the decode, exact rather than approximate.
- [x] **92. Record preprocessing into the inference compute pass.** Two submits
  are 18% of a detection (5 continued). Merge them and measure. Rejected: the
  saving is real on the host and is given back by the GPU/CPU overlap it destroys.
- [x] **91. Find what single-image latency is actually made of.** Batch is at its
  floor, so measure the GUI's path instead: decode, detect, preview texture. Take
  whatever the measurement points at rather than starting from the GPU backlog.
- [~] **90. Detect from a scaled JPEG decode.** 86's arithmetic, done properly:
  a 1/2 or 1/4 DCT decode costs less than the full-resolution resize it removes,
  even as a second decode. Rejected on accuracy instead -- see the record.
- [x] **89. Retry 87's rejected quality-metric swap on the new RGBA path.**
  87 measured it neutral and named the reason: no RGBA fast resize existed, so
  the swap converted the region to RGB and gave back what it saved. 88 built one,
  and removed the pool hop that was also in the way. Re-measure.
- [x] **88. Resize crops with SIMD, and stop paying to *avoid* threading.**
  Re-profiling after 71 put `image::imageops::resize` inside
  `crop_face_from_image` at the top of the batch, which 87 had already named as
  the remaining candidate. Measure an RGBA `fast_image_resize` path for it, and
  whatever else the fresh profile hands over.
- [x] **87. Profile the batch path and cut what it is actually spending.**
  Where a folder job's CPU goes, and whether anything in it is a serial
  bottleneck. Distinct from 60/61, which propose scheduling changes; this only
  measures and takes what the measurement hands over.
- [~] **86. Detect from a reduced-scale decode, crop from the full one.**
  Experiment 51 showed the resize is bounded below by reading the source once,
  so the only remaining lever on it is fewer source pixels. A JPEG decoded at
  1/2 via DCT scaling is a proper low-pass, not a dropped-pixel approximation,
  and at 1/2 a 10 MP source is still 4x the 640x640 input. Detection would read
  2.5 MP instead of 10; crops keep the full-resolution decode, so this is not
  the rejected reduced-scale crop workflow. Measure both decodes where a face is
  found, and evaluate recall on small faces with `resize_quality.rs`. Q, and it
  changes the decode stage rather than preprocessing. Distinct from 73, which
  screens for whether faces exist at all. **Premise weak on the real workload:**
  1020 of 1239 images in the reference folder contain a face, and every one of
  those needs the full-resolution decode for its crop anyway, so the reduced
  decode is added work rather than replacement work for 82% of the folder. Only
  worth revisiting for detection-only or mostly-faceless input. **Measured anyway
  in 90**, because "added work" does not settle whether the added work costs less
  than the resize it removes: it does, and the idea still fails, on detection
  accuracy rather than on arithmetic.

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

### 67. Alternative JPEG decoders - measured, not adopted; build issue fixed

Detection is now about 3.1 ms for a 10 MP photo. Decoding that photo is
**24.0 ms** (`stage_breakdown`, warm, from memory), so on a folder of images the
decoder, not the detector, sets what the job costs. That reorders the backlog:
everything left in detection is small against this.

`mozjpeg` (libjpeg-turbo) is already linked into every binary through `nokhwa`,
which uses it for webcam MJPEG frames, so trying it as a still decoder costs no
new native dependency. `fcs-core/examples/decode_bench.rs` compares it with the
shipped `image`/zune-jpeg path over the largest fixtures, decoding from memory
so cold I/O is excluded.

The first run measured **0.51-0.59x -- roughly half the speed** of zune-jpeg,
and that result was void: `mozjpeg-sys` had emitted
`NASM not installed. Mozjpeg's SIMD won't be enabled` and compiled
`jsimd_none.c`, so what was timed was libjpeg-turbo's scalar C fallback. NASM
was in fact installed at `C:\Program Files\NASM` but not on `PATH`, and
`cargo clean -p mozjpeg-sys` does not invalidate a cached build-script run --
the stale output had to be deleted outright before the script would re-run.
With SIMD confirmed present (`nasm-missing=0 jsimd_none=0`):

| Source | `image` (zune-jpeg) | libjpeg-turbo | Speedup |
| --- | ---: | ---: | ---: |
| 8.3 MP (x7) | 27.1-33.8 ms | 23.6-29.1 ms | 1.10-1.27x |
| 11.1-12.2 MP (x5) | 29.2-36.6 ms | 23.2-27.1 ms | 1.24-1.35x |
| 22.1 MP (x3) | 44.1-48.4 ms | 36.2-38.8 ms | 1.22-1.25x |
| **Total, 15 images** | **517 ms** | **420 ms** | **1.23x** |

So libjpeg-turbo is **1.23x faster** -- about 5-10 ms per photo -- and the
scalar fallback is roughly 2.2x slower than the SIMD build, which is what the
first run was really measuring.

**Adopted**, after the outputs were reviewed in the GUI. It shipped first behind
`FCS_JPEG_TURBO` so the difference could be judged rather than argued about; the
flag is gone and JPEG decoding now takes this path by default.
`load_image` takes the turbo path only for `.jpg`/`.jpeg`, and returns `None`
into the ordinary path for anything unusual -- CMYK, 16-bit, truncated, a `.jpg`
that is not a JPEG -- because `image` handles formats `mozjpeg` does not. EXIF
orientation still comes from `image`'s parsing: building its decoder reads
headers, not pixels, so a second EXIF implementation cannot disagree with the
first about which way a photo goes. Verified over 12 fixtures: **0 dimension
mismatches**, so orientation survives.

What actually changes, measured rather than assumed:

| Level | Difference |
| --- | --- |
| Decoded pixels | mean 0.03, worst single channel **5/255** |
| Detection boxes | shift **0.28-1.30 px** across 6 sample images |
| Landmarks | shift **at most 0.37 px** |
| Exported crops | 76-84% of pixels differ, max channel 70-111 |

The crop figure looks alarming and is the least meaningful of the four: the
crop rectangle moves by about a pixel, so the whole crop resamples at a
different offset. It is a shift, not colour corruption -- the decoded pixels
themselves are within 5/255. Reproduce with
`fcs-core/examples/cropdiff.rs`, which compares two folders of exported crops.

`cli_json_output_matches_snapshot` fails with the flag on -- 829.4791 against a
recorded 829.6314, against a 0.001 tolerance. That is the output-stability
question stated exactly: adopting this means accepting a sub-pixel change to
recorded detection coordinates and updating that snapshot. Everything passes
with the flag off, which is the default.

The remaining decision was a product one -- whether ~1.23x on the dominant cost
is worth outputs that are not bit-reproducible against previous releases -- and
it was made by looking at real crops, not by reading this table. Accepted; the
CLI JSON snapshot moved by at most 0.26 px and was updated with the change.

**A regression found while validating this, and not caused by it.** Running the
first real folder through the CLI -- 1239 images, which nothing in this session
had done -- exited 101 with `RefCell already borrowed` and produced 247 crops
instead of 891. The cause was experiment 19's thread-local `Resizer` holding a
borrow across the resize: below the threading gate the resize runs inside
`ThreadPool::install`, and `install` from a rayon worker lets that worker take
other queued work while it waits, re-entering the same function on the same
thread. Fixed by taking the `Resizer` out of the thread-local for the duration.

The lesson is about validation, not about the bug. The whole workspace suite,
every probe in this session and all single-image measurements passed against the
broken code, because none of them nests rayon. The first regression test written
for it also passed against the broken code -- large sources and a wide pool never
force a steal -- and was only found wanting because the fix was temporarily
reverted to check the test could fail. **A batch run belongs in the validation
of anything touching a shared cache or a thread-local**, alongside the test
suite.

**Separate finding, acted on.** `CONTRIBUTING.md` lists NASM as a required build
tool, but a missing NASM only warns and silently degrades -- the build succeeds
and the binary is quietly ~2.2x slower at JPEG work. The Linux leg already
installed `nasm` through `.github/actions/linux-build-deps`; the **Windows and
macOS legs did not**. `nokhwa` decodes webcam MJPEG frames through libjpeg-turbo,
so this reached shipped behaviour on two of three platforms. Both legs now
install it, and the Windows leg **verifies `nasm` is on `PATH` and fails if it
is not** -- a warning that only shows up in a build log is exactly how this got
missed, so the guard matters more than the install.

Recorded regardless of the decoder outcome: **decode is 24 ms against 3.1 ms of
detection**, so experiments 67 and 68 outrank anything remaining in the
detection path for folder work.

### 63. Batch worker budget, refreshed - no change made

First measurement of the thing the application actually does: 1239 real photos
through `fcs-cli --crop`, rather than one image in a loop. Baseline **39.8 s
cold, about 17-19 s warm** -- roughly 65 images/s once the file cache is warm.

A cold first run measured 30.9 s at 32 workers against 20.5 s at 16 and looked
like a 33% win. It was not: the 32-worker run was reading the folder from disk
for the first time. Warm and with the order alternated between pairs:

| Workers | Runs (s) | Median |
| --- | --- | ---: |
| 32 (default) | 18.7, 20.3, 17.0, 21.4, 17.8, 18.4 | 18.55 |
| 16 (physical cores) | 19.0, 18.5, 17.3, 16.5, 16.8, 16.8 | **17.05** |

**About 8%, favouring 16 in four of six pairs**, and 16 is also steadier (spread
2.5 s against 4.4 s). That confirms the direction already recorded in
`fcs-cli/src/main.rs` -- fewer workers than logical processors is better here --
at a smaller magnitude than the 18% recorded there.

**No default changed.** The existing comment gives a reason that this data does
not address: "physical cores" counts P and E cores alike, so a rule tuned on a
symmetric 7950X could be wrong on a hybrid-core machine, and this is one machine.
`RAYON_NUM_THREADS` already overrides it. The refreshed numbers are added to
that comment so the next person sees two measurements rather than one.

**Rejected, and firmly: skipping the inner resize threading when already on a
rayon worker.** Batch parallelises across images and `fast_image_resize` then
splits each resize again, which looks like plain oversubscription. Gating it on
`rayon::current_thread_index().is_some()` is **much worse**, four
order-alternated pairs at `Quality`, winning every pair in both orders:

| Variant | Runs (s) | Median |
| --- | --- | ---: |
| threading throughout (shipped) | 17.1, 15.3, 14.7, 16.2 | **15.75** |
| gated on a rayon worker | 23.0, 19.7, 19.2, 24.4 | 21.35 |

About **27% slower gated**. Rayon's work stealing absorbs the nesting, while the
gate pins each resize to one thread and leaves cores idle on uneven image sizes
and at the tail of the batch. The intuition that nested parallelism must
oversubscribe was simply wrong here.

**Two measurement errors on the way to that, both worth recording.** The first
comparison was unalternated and showed the gate winning three times out of
three by 1.1-2.0 s; the six wall times in execution order were 22.9, 20.9, 20.6,
19.5, 19.5, 18.4, monotonically decreasing, with the gated variant second every
time. Pure drift. **Batch wall time on this machine swings about 10% run to
run.**

The second is worse and invalidated the conclusion outright. `fcs-cli` loads
`config/gui_settings.json` **from the working directory** when no `--config` is
given, and that file sets `resize_quality = speed`. `threading_pays` returns
`false` for `Nearest` before it ever reaches the gate, so every run from the
repository root was comparing the gate against itself. The "measured nothing"
result was structurally guaranteed, not evidence. Redone with an explicit
`--config` selecting `Quality`, the gate is clearly harmful.

**Anything measuring the batch path must state its configuration.** Run from the
repository root the CLI finds that settings file and detects 1020 faces at
`score_threshold` 0.8 with a nearest-neighbour resize; run from anywhere else it
uses the built-in defaults, 0.9 and a filtered resize, and detects 423. Same
binary, same arguments, same images. The worker-count table above was taken from
the repository root and so describes the `speed` configuration.

### 6. The workload matrix, and two ways the earlier numbers were flattering

Everything else in this backlog was measured on one folder of large JPEGs.
`examples/workload_matrix.rs` walks a corpus one image at a time and buckets
decode and detect by resolution, orientation, format and face count. Run over
both the 275-image fixture set and the 1239-image folder:

| Resolution | n | dec p50 | det p50 | detect's share |
| --- | ---: | ---: | ---: | ---: |
| under 1 MP | 117 | 2.6 ms | 4.54 ms | **64%** |
| 1-4 MP | 160 | 8.6 | 6.89 | 44% |
| 4-8 MP | 129 | 18.9 | 7.25 | 28% |
| 8-16 MP | 745 | 21.0 | 7.62 | 27% |
| over 16 MP | 88 | 43.8 | 10.09 | 19% |

**Both scale, and decode scales harder, so which one matters flips around 1-2 MP.**
Above that a photo is mostly decode; below it detection is the larger half.
Orientation makes no difference beyond the sizes in each bucket (landscape 4.32
against portrait 3.57 ms), and neither does face count -- 4+ faces measured
*faster* than 2-3, because those images are smaller, not because counting faces
is free.

**Container format, on identical pixels.** The buckets suggested WebP and PNG
were 4-9x JPEG per megapixel, but each format held different images at different
sizes. `examples/decode_formats.rs` re-encodes one 12.2 MP source into each:

| Format | decode p50 | ms/MP |
| --- | ---: | ---: |
| tiff | 9.3 ms | 0.77 |
| bmp | 23.4 | 1.92 |
| jpeg (`image` crate) | 47.8 | 3.92 |
| png | 71.9 | 5.89 |
| webp | 118.9 | 9.75 |

So the real spread is 2.5x for WebP and 1.5x for PNG, not 4-9x -- and against the
libjpeg-turbo path production actually uses for JPEG (67), the WebP gap is nearer
3x. Not acted on: the corpus is 100% JPEG, the fixtures hold nine WebP files out
of 275, and libwebp would be a new native dependency rather than one already
linked, which is what made turbo free in 67.

**The first run of this had the wrong detector.** `YuNetDetector::new_gpu` pairs
GPU inference with the *CPU* preprocessor; the CLI and GUI both build
`with_gpu_preprocessor`. On the CPU-paired path detection measured 6.00 ms at
under 1 MP against 4.90 at 1-4 MP -- essentially flat across a 40x range, which
reads as "detection is fixed-cost overhead" and is not true. With the production
pairing it scales properly, 2.84 to 10.74 ms. A convenience constructor was
enough to invert the conclusion.

**Repeating one image understates per-image latency by a quarter to a half.**

| Corpus | 24 distinct images, once each | the first image 24 times |
| --- | ---: | ---: |
| fixtures | 7.39 ms | 5.88 ms |
| 1239 photos | **8.40 ms** | **5.48 ms** |

`phase_timings` repeats one image, which is why it reports 3.39 ms where a stream
of distinct images costs 7-8. Its *relative* results are unaffected -- both sides
of an A/B run in the same harness -- but its absolute numbers are a floor, not a
per-image cost, and that distinction was not previously written down.

**This qualifies 91 without overturning it.** That experiment closed the GPU
overhead backlog (22, 25, 26-47) by arguing `gpu_record` is 0.394 ms against a
27 ms decode. That holds for a 12 MP photo. Under 1 MP the balance is different:
detection is 64% of the per-image cost, so for webcam-sized frames those
experiments are worth more than 91's arithmetic suggests. Deciding that needs the
webcam measurement (53, 65), not another photo corpus.

Not measured: peak RAM and VRAM, which this experiment also asks for. That needs
process-level sampling rather than a timer around a call, and is left open.

### 74. Input resolution - only one backend can vary it, and a settings footgun

The premise is "compare supported smaller/larger inputs", and the answer is that
production supports exactly one. `input.width` and `input.height` are settings,
and setting them to anything but 640 produced this:

| Backend | 320 or 960 |
| --- | --- |
| WGSL GPU graph | fails: `stage0 conv` |
| ONNX Runtime | fails: `InvalidArgument: Got invalid dimensions for input` |
| built-in `cpu-graph` | **works** |

The bundled `..._640.onnx` declares a fixed input, and `gpu/graph.rs` writes
`SpatialDims::new(640, 640)` into stage 0 directly, so two of the three are
locked at 640 by construction. Only the pure-Rust graph derives its dimensions
from the configured size, and it is the slowest backend by 4x (69), so this is
not a lever production can pull. Reopening it means a re-exported model and a
GPU graph that is not hardcoded, which is 78's territory.

**The sweep is still worth having**, measured on `cpu-graph` over all 1239
images:

| Input | Faces | Crops | Wall |
| --- | ---: | ---: | ---: |
| 320 | 907 | 808 | 7.6 s |
| 480 | 1006 | 882 | 12.1 s |
| **640** | **1032** | **901** | **18.4 s** |
| 800 | 1041 | 915 | 31.4 s |
| 960 | 1070 | 936 | 59.1 s |

Detections rise monotonically with input size and cost rises faster: 960 finds
3.7% more faces for 3.2x the time, 320 gives up 12% of them for 2.4x the speed.
**Whether the extra detections at 960 are real faces or false positives is not
established here** -- there is no ground truth in this corpus, and "more boxes"
is not "better". 640 is a defensible middle rather than a measured optimum.

A methodological note worth keeping: on a 20-image subset, 960 found *fewer*
faces than 640 (22 against 23) and the curve looked like it peaked at 640. The
full corpus inverted that. Twenty images and single-digit differences were noise
wearing the shape of a result.

**What this experiment actually produced is a fix.** Nothing validated the
configured size, so a bad value loaded the model, read all 1239 files, failed
each one separately and ended with "all detections failed". `probe_input_size`
runs one zeroed tensor through the backend at construction, so the failure
arrives once, before any work, naming the setting to change. It is deliberately
not a hardcoded 640 -- a caller may supply their own model, and the backend is
the thing that knows.

It also made a previously broken configuration work: at 320 the CLI now gets a
clean GPU-init failure, falls back to the CPU path, and `cpu-graph` runs the job.

Cost is one inference at startup, and it does not show: 6.77, 6.74, 6.88 s
against an A/A floor of 6.86-7.17 s measured in 60, with 1032 faces and 901
crops unchanged.

### 69. CPU backends - the default was right, its thread count was not

The backlog calls these "the shipped tract and ONNX Runtime paths", but tract is
gone: the built-in path is `CpuGraph`, a pure-Rust YuNet that needs nothing
installed. `InferenceBackend::Auto` prefers ONNX Runtime whenever a compatible
library is present, and that preference had never been measured.
`examples/cpu_backends.rs` preprocesses once and times inference alone.

| Backend | p50 ms | p95 ms | parallel img/s |
| --- | ---: | ---: | ---: |
| `cpu-graph` | 29.70 | 33.90 | 90.8 |
| `onnxruntime` | 7.03 | 7.70 | 204.7 |

**ONNX Runtime is 4.2x on latency and 2.3x on throughput**, so `Auto` is right.
Raw outputs agree to 0.000427, which is two implementations of the same graph in
f32, not a disagreement.

**The interesting part is `intra_threads`.** It was 1, on the reasoning that
whole images already run concurrently through rayon so the runtime should not
fight it. But the intra-op pool belongs to the *session*, and Face Crop Studio
shares one session across all workers, so the setting is a total rather than a
per-run multiplier. Raising it does not give each concurrent inference its own
threads.

Order-alternated, one inference at a time, winning every pair:

| `intra_threads` | Runs (ms) | Median |
| --- | --- | ---: |
| 1 | 13.03, 6.95, 7.27 | 7.27 |
| 4 | 4.13, 4.17, 2.70 | **4.17** |

And the check this experiment exists to force -- a faster runtime is not
necessarily a faster application. A 1239-image folder export, `--no-gpu`,
alternated:

| `intra_threads` | Runs (s) | Median |
| --- | --- | ---: |
| 1 | 10.32, 10.30, 10.45 | 10.32 |
| 4 | 10.18, 10.08, 10.94 | 10.18 |

**No difference.** Rayon has already filled the cores; the extra threads have
nothing to add. So the setting buys preview latency on a machine with no GPU and
costs nothing in batch, which is the shape of a free change -- but only on a
machine with cores to spare, which is the one I measured on.

`default_intra_threads` therefore divides logical processors by four and clamps
to 1..=4, so a four-core machine keeps exactly today's single thread. That is
deliberately the configuration this change could *not* test.

**The comment I replaced recorded 8.69/1.94/3.74 ms for 1/4/16 threads and
concluded the runtime "oversubscribes itself past about 4".** Neither the size of
the gain nor the 16-thread regression reproduced here (5.64 ms at 16 against 5.10
at 8). Both measurements agree on the direction, and both are now in the comment
rather than one silently replacing the other.

### 58. NMS worst case - the grid was pessimal exactly where it was needed

Typical postprocessing is 0.006 ms, which is why 56 and 59 were closed. This asks
the other question: what happens at worst-case candidate counts? `top_k` defaults
to **5000**, and `apply_nms_in_place` is skipped entirely when `nms_threshold` is
0 -- a value the GUI slider reaches, since it is clamped to `0.0..=1.0`. So both
stages can be handed thousands of detections on the interactive path.

Two shapes, benchmarked in-module (`nms::benchmarks`): **separated**, where
nothing merges and every pair is compared, and **clustered**, where everything
collapses onto one face.

| n = 5000 | Before | After |
| --- | ---: | ---: |
| `apply_nms_in_place`, clustered | **23.138 ms** | **0.248 ms** |
| `apply_nms_in_place`, separated | 0.214 ms | 0.232 ms |
| `dedup_close_centers`, clustered | 6.116 ms | **0.009 ms** |
| `dedup_close_centers`, separated | 14.269 ms | 14.269 ms |

**The spatial grid was 100x slower on clusters than on spread-out boxes**, which
is backwards: a cluster is what the grid is for. The search was never the
problem. `NMS_GRID_SIZE` is a fixed 32x32, so cell size is set by the scene
bounds -- and a tight cluster has *small* bounds. Cells came out around 4 px
against 80 px boxes, so each box was inserted into roughly 640 cells: 5000 boxes,
3.2 million insertions, all of it in `build_spatial_grid`. Spread-out scenes have
large bounds, one cell per box, and were fine, which is why this never showed.

Sizing the grid to the mean box extent instead keeps cells-per-box near 1 either
way. The grid is only an acceleration structure, so this is a pure speed change,
and `grid_and_naive_agree_across_scene_shapes` holds it to that across four
scene shapes -- clustered, spread, mixed scales, and an identical stack where the
extent collapses to a point. Checked against a deliberately under-covering cell
range, where it fails with "mixed scales: kept a different count".

`dedup_close_centers` was separately doing `Vec::remove` inside its inner loop,
O(N) per drop on top of O(N^2) comparisons. Marking a bitmap and compacting once
is exactly what the shifting loop did, and
`dedup_bitmap_matches_the_shifting_implementation` runs the old implementation
alongside the new over pseudo-random scenes at three densities to prove it.
Checked against dropping the outer-loop skip, where it fails at spread 120.

**What is left, honestly.** `dedup_close_centers` on 5000 *separated* detections
is still 14.3 ms, and the bitmap cannot help there because nothing is removed --
it is 12.5 million distance comparisons. Reaching it needs `nms_threshold` at 0
plus thousands of scattered survivors. A spatial index would fix it, but the
merge radius scales with the larger of the two boxes, so the query radius depends
on what it finds, and that is real complexity for a setting that already means
"suppress nothing". Left as a documented bound.

Production output is unchanged: 1032 faces, 901 crops, 901 of 901 byte-identical
against the previous verified run.

### 91. Single-image latency, and a 3.2 s preview stall for large images

Batch is at its floor, so this measures the other workload. `phase_timings` on a
12 MP photo, `detect_image` p50 **3.390 ms**:

| Phase | p50 ms | Share |
| --- | ---: | ---: |
| `cpu_resize` | 1.640 | 48% |
| `readback_wait` (GPU execution) | 0.501 | 15% |
| `gpu_record` (host command recording) | 0.394 | 12% |
| `gpu_rgb_to_chw` | 0.289 | 9% |
| `gpu_submit` | 0.167 | 5% |
| `gpu_decode` | 0.082 | 2% |
| `postprocess` | 0.006 | 0.2% |

**This closes the GPU backlog for this workload, before any of it was attempted.**
`gpu_record` is the interesting entry -- 0.394 ms of pure host work rebuilding
command buffers and bind groups for an unchanging graph, which is what 22 and 25
propose caching. But a GUI image selection costs a **27 ms decode** plus this
3.4 ms detection, so 0.394 ms is 1.3% of the interaction, and the whole 21-22-25
chain is worth about 3% at best. The shader experiments (26-47) target
`readback_wait`: an infinitely fast GPU saves 0.5 ms of 3.4 ms, 1.6% of the
interaction.

**What the measurement found was not on the list.** `clamp_to_texture_limit`
downscales previews past 8192 per side -- its own comment says camera RAWs
"routinely exceed it" -- through `DynamicImage::resize_exact`, the pixel-by-pixel
path replaced everywhere else. `examples/preview_texture_cost.rs`, alternated per
row, 133 MP source:

| Clamp | Texture | MB | `resize_exact` | fir | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| **8192 (production)** | 6144x8192 | 201.3 | **3215.25 ms** | **103.11 ms** | **31.2x** |
| 4096 | 3072x4096 | 50.3 | 1456.55 | 38.81 | 37.5x |
| 2048 | 1536x2048 | 12.6 | 605.89 | 35.77 | 16.9x |

**Over three seconds of preview stall removed** for exactly the images the clamp
exists for. On a 12 MP photo the clamp does not fire and both sides measure 6.8
against 6.8 ms -- an accidental A/A control confirming the change is inert where
it does not apply.

The `ColorImage` build itself is 6.5-6.8 ms at full 12 MP, which is what I went
looking for, and it is not a problem. Lowering the 8192 limit to cut the upload
is *not* done: that would add a resize to every large image to save bandwidth I
have not measured, and the limit exists for a correctness reason (egui panics
above the GPU's maximum texture side), not a performance one.

**One bug avoided on the way in.** Routing everything without an explicit RGBA8
match through `resize_image` returns RGB8, which would silently flatten a LumaA8
or RGBA16 preview. The gate is `color().has_alpha()` instead.
`clamp_keeps_alpha_when_the_source_has_it` was checked against the RGBA8-only
version and fails there with "alpha dropped for La8", so it has teeth.

### 68. File I/O, warm - hidden, and the experiment's own gate says stop

This experiment ends with "skip if decode or inference dominates and I/O is
already hidden", so the whole thing turns on one number.
`examples/io_cost.rs` reads the corpus at the concurrency the batch uses:

| Pass | Bytes | Serial | At 32 threads |
| --- | ---: | --- | --- |
| 1 | 1386.2 MB | 1.29 s (1075 MB/s) | 0.19 s (7470 MB/s) |
| 2 | 1386.2 MB | 0.35 s (4003 MB/s) | 0.16 s (8597 MB/s) |
| 3 | 1386.2 MB | 0.34 s (4099 MB/s) | 0.17 s (8297 MB/s) |

**0.17 s of a 7.1 s run, about 2.4%, already spread across the pool.** Prefetch
depth and read-buffer reuse have nothing to work with: the reads are overlapped
with decode by the same rayon parallelism that runs everything else, and 8.6 GB/s
is page-cache speed, not disk speed. Skipped on the experiment's own terms.

**The cold case is not measured, and I could not measure it honestly.** The file
cache cannot be dropped from inside the process, and every tool that would do it
from outside (RAMMap, EmptyStandbyList) is not on this machine; copying the
folder elsewhere leaves the pages warm. The historical 39.8 s cold against 17 s
warm in 63 is the only figure, it predates most of this backlog, and it was never
attributed -- on a disk that serves 1.3 GB at these rates, a cold read should be
well under a second, so that 22 s gap probably was not the disk. Anyone with a
machine that has not read the corpus can settle it with one run.

### 52. The GUI's caches, inspected first as instructed - all three were dead

This experiment says to inspect the current caches and their invalidation before
proposing anything. There are three, and none of them works:

| Cache | Capacity | What it actually does |
| --- | ---: | --- |
| `cache` (detection) | 50 | `put` and `clear`, **never read** |
| `crop_preview_cache` | 500 | `clear()` in seven places, never written or read |
| `image_cache` | 20 | declared and constructed, never touched at all |

The detection one is not merely dead, it retains. Each entry holds the decoded
`source_image` and an egui `TextureHandle`, so browsing fifty images in the GUI
keeps fifty full-resolution decodes alive -- on this corpus, averaging 8.6 MP,
about **1.3 GB of RAM** plus the textures -- to serve lookups that never happen.

`git log -S` places it: the previous GUI read the cache in
`app_impl.rs::start_detection`, and keyed it properly through
`cache_key_for_path(path, settings)`. The `fcs-gui2` rewrite kept the write and
dropped the read. The key went with it -- both surviving construction sites
hardcode every field (`model_path: None`, `input_width: 640`, `score_bits: 0`,
`top_k: 5000`), so the key is really just the path, and neither `rotation_deg`
nor `auto_orient_exif` appears in it even though both change the result.

**Deleted, all three, with the `lru` dependency behind them** -- 179 lines and one
crate. Nothing observable changes, because nothing read them.

**Restoring the detection cache is a smaller prize than it looks and is not done
here.** A hit would skip a decode and a detection: 20 images single-threaded run
1.97-1.99 s including startup, so roughly 60 ms per re-selection, which is under
the threshold where anyone notices a preview appearing. Against that, a hit has
to reproduce what `DetectionFinished` sets up -- selected faces, edit history,
canvas rotation, quality rules -- and a key that is wrong in the way the current
one is wrong would serve stale detections after a settings or rotation change.
That is a GUI-visible refactor with a correctness failure mode, worth doing only
with someone watching the GUI, and worth about 60 ms.

### 60. Worker count - the previous answer reversed, and the default is now right

The in-flight GPU request count is not configured anywhere: it is however many
rayon workers are inside `detect_image` at once. So the worker sweep *is* the
concurrency experiment, and it needs redoing, because 63's answer was measured
when each image cost 2.4x more CPU than it does now.

**A/A first**, six consecutive identical runs: 7.00, 7.17, 6.87, 7.02, 6.86,
6.92 -- spread **0.31 s**, tighter than at any earlier point in this backlog.

That mattered immediately. A five-way sweep in one batch put the default at 8.74
and 8.42 s while the A/A block, minutes earlier, had it at 6.86-7.17. Interleaving
slow configurations moves the fast one's own numbers by nearly a second, so only
within-pair comparisons from the same batch are used below.

| Workers | Runs (s) | Median |
| --- | --- | ---: |
| 8 | 10.93, 14.72 | ~12.8 |
| 12 | 12.04, 12.11 | 12.08 |
| 16 | 9.04, 9.06, 8.77, 8.51 | 8.90 |
| **32 (default)** | 7.92, 7.78, 7.79, 7.85 | **7.82** |
| 48 | 8.04, 7.46, 8.02 | 8.02 |
| 64 | 8.27, 8.24, 7.93 | 8.24 |

**The default wins all four alternated pairs against 16, by about 12%**, with a
spread of 0.14 s across its four runs. Above 32 it is flat to slightly worse.

**This reverses 63, and 63 was not wrong.** It measured 17.05 s at 16 against
18.55 s at 32 and read it correctly. What changed is the workload: the crop,
resize and quality-metric work came out, per-image CPU fell by about 2.4x, and
the balance went with it. 112 s of CPU across a 7.85 s run on 32 threads is
roughly 45% busy per thread, so more than half of each thread's life is now
blocked on the GPU or on a file read. Capping at 16 leaves cores idle rather
than saving them from contention.

Two prior measurements agreed with each other and both are now stale. The
guidance in `main.rs` and README.md said to cap, and would have made the
application slower; both now carry all three numbers.

**No bounded dispatcher is justified (64).** Throughput peaking exactly at one
worker per logical processor and falling away above it is the shape of a
CPU-bound schedule. A contended or starved GPU queue would keep improving with
more in-flight requests until the queue filled, and it does not.

### 90. Scaled decode for detection - the arithmetic works, the accuracy does not

After 88 and 89 the batch profile is 112 s of CPU, down from 194.7 s, and the
source resize is now the largest single thing in it:

| Cost | Inclusive | Share |
| --- | ---: | ---: |
| `resize_pixels_fast`, all callers | 46.1 s | 40.9% |
| `resize_image` (source -> 640x640) | 36.5 s | **32.3%** |
| `detect_image` | 36.5 s | 32.3% |
| PNG deflate | ~10.3 s | 9.2% |

`detect_image` and the resize inside it are the same number, which is what a GPU
detector should look like: the inference wait costs no CPU, so detection *is* the
resize.

51 closed the resize itself -- at a fixed source resolution it is bounded below
by reading the source once -- leaving only fewer source pixels, which is 86. 86
was dismissed because 82% of the folder has a face and needs the full-resolution
decode anyway, so a reduced decode is *added* work. True, and not the question.
The question is whether the added work costs less than the resize it removes.

**It does.** `examples/scaled_decode.rs`, 1239 JPEGs, 10707 MP, two passes in
opposite orders:

| | pass 1 | pass 2 |
| --- | ---: | ---: |
| resize alone, from full | 1.86 s | 1.39 s |
| scaled decode + its resize | 0.95 s | 0.99 s |

A scaled decode costs about 45% of a full one -- Huffman has to run either way,
only the IDCT shrinks -- and the resize after it is roughly 6x cheaper.

Wired up behind `FCS_SCALED_DETECT`, picking per image the largest reduction
leaving both dimensions at or above the detector input, with detections mapped
back onto the full-resolution image. Order-alternated, winning all four pairs:
**7.68 s to 6.64 s, about 13%.**

**Then it fails the quality bar this project already set.** 51 rejected
`Interpolation(Bilinear)` for a 35 px maximum landmark shift, on the grounds that
it "places a crop visibly wrong". Against production over 1030 matched faces:

| Headroom | Lost | Gained | Landmark p50 | p95 | max | >35 px | IoU min | Wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1x | 2 | 4 | 1.15 | 8.69 | **158.2** | 13 | 0.6647 | 6.64 s |
| 2x | 0 | 0 | 0.00 | 1.94 | 76.1 | 3 | 0.7429 | 6.71 s |
| 3x | 0 | 0 | 0.00 | 0.69 | 21.9 | **0** | 0.9374 | 6.87 s |
| off | -- | -- | -- | -- | -- | -- | -- | 7.14 s |

The worst offenders are single-face images, so these are real displacements and
not the matcher pairing the wrong faces in a crowd. libjpeg's scaled IDCT is a
proper low-pass but not the same low-pass as a convolution downscale, and near
the limit the difference reaches the landmarks.

**Rejected.** Backing off until the accuracy is acceptable takes the saving with
it: at 3x headroom nothing is lost or gained and no face moves past 35 px, but
only the largest images reduce at all and the gain is about 4% -- for a second
decode path, a coordinate remap between two image spaces, and EXIF applied twice.
13% was never available at a quality this project accepts.

`examples/scaled_decode.rs` is kept, since it answers the arithmetic half in
about a minute if the workload assumption changes. Detection-only or
mostly-faceless input, where the full decode is not needed at all, is still the
case 86 identified and this does not close it.

### 72. PNG settings - the current one is right, in both directions

PNG is lossless, so neither the compression level nor the row filter moves a
decoded pixel. Only encode time and file size change, which makes this a rare
clean trade. `examples/png_bench.rs` encodes 901 real crops (944.8 MB raw) with
each combination, twice, in opposite orders:

| Setting | Time (s) | Bytes (MB) | vs default |
| --- | ---: | ---: | ---: |
| fast | 0.10-0.23 | 372.0 | +14.1% |
| fast/paeth | 0.10-0.18 | 371.9 | +14.1% |
| fast/sub | 0.14-0.32 | 441.9 | +35.5% |
| fast/nofilter | 0.39-0.54 | 942.6 | +189.1% |
| **default (current)** | 2.39-2.52 | **326.0** | -- |
| default/up | 2.27-2.29 | 327.6 | +0.5% |
| default/paeth | 2.47-2.52 | 327.4 | +0.4% |
| best | 6.21-6.49 | 319.5 | -2.0% |

The level dominates and the filter barely matters: at `default`, adaptive
filtering costs about 5% of the encode and buys 0.4-0.5% of size against the best
fixed filter. `NoFilter` is a trap -- three times the bytes.

**In isolation `fast` is 16x quicker. End to end it is worth nothing**, and that
is the finding. Order-alternated over the 1239-image folder:

| Setting | Runs (s) | Median | Output |
| --- | --- | ---: | ---: |
| default | 6.90, 7.10, 6.75 | 6.90 | 312 MB |
| fast | 7.00, 7.36, 6.81 | 7.00 | 358 MB |

So export encoding is *not* on the critical path -- except it partly is, because
the trade is asymmetric. Pushing the other way, in its own alternated batch:

| Setting | Runs (s) | Median | Output |
| --- | --- | ---: | ---: |
| default | 8.04, 8.39, 8.14 | 8.14 | 312 MB |
| best | 10.13, 10.36, 10.27 | **10.27** | 306 MB |

**`best` costs 26% of wall time to save 2% of disk.** Taking encode work away
gains nothing while adding it costs plenty, which says the parallel schedule has
enough slack to absorb the encode at `default` and not four times that. The
current default sits at the point where the slack runs out. Nothing to change,
in either direction -- and `fast` would have been a bad trade even if it had been
free, at +14% on disk for output nobody can tell apart, because PNG is lossless.

Note the two batches disagree on `default` itself, 6.90 against 8.14. That is
drift between batches on this machine, and it is why only within-batch
alternated pairs are compared here.

**Concurrency: already there, nothing serialising it.** Export runs inside the
rayon per-image loop, with no mutex on the path and no single writer. 87 already
established the point from the other side -- CPU spread evenly across 50 threads
with no serial bottleneck. `load_png_exif_chunks` looked like a per-crop re-read
of the source, but it checks the extension first and returns empty for the JPEG
sources here without opening anything.

**One thing was worth removing.** `encode_rgba8` called `image.to_rgba8()`, which
*clones* when the image is already RGBA8 -- and every exported crop is, since
`crop_face_from_image` returns `ImageRgba8` and the shape mask keeps it there. So
each export copied a whole 1 MB image to hand the encoder bytes it already had.
Borrowed through a `Cow` now, same for `encode_jpeg`'s `to_rgb8`.

Output is byte-identical, 901 of 901. **The speed is not measurable and is not
claimed:** 7.67 s against 7.33 s median, winning three pairs of four with the
ranges overlapping, against a predicted saving near 0.09 s and a noise band
closer to 1 s. It is kept because it is strictly less work and one less
full-size allocation per concurrent export, which is the peak-memory half of this
experiment, not because the folder got faster.

### 89. The same swap 87 rejected, now worth 9%

Experiment 87 routed `estimate_sharpness`'s downscale through `resize_image` and
measured nothing, in either thread configuration, order-alternated both ways. It
recorded why: callers hand it `ImageRgba8`, `resize_image_fast` took RGB8 only,
so it converted the whole region first and handed back exactly what the faster
kernel saved. It named the fix -- an RGBA-capable path -- and left it.

88 built that path, and removed the cross-registry pool hop that sat on it too.
Retrying the swap on those terms:

| Binary | Runs (s) | Median |
| --- | --- | ---: |
| baseline (88) | 8.15, 7.41, 7.45, 8.08 | 7.77 |
| via `resize_rgba_fast` | 7.06, 7.14, 6.90, 7.44 | **7.10** |

**About 9%, winning all four pairs in both orders.** Exported crops are
byte-identical, 901 of 901 with filenames included.

**The reported scores are not, and I would have missed it.** A snapshot test
caught `quality_score` moving 111.858 to 112.668 on the fixture, which sent me to
measure the whole folder rather than call the change neutral:

| Relative change in `quality_score` | |
| --- | ---: |
| median | 0.000% |
| p95 | 2.03% |
| max | 6.00% |
| quality label flips, 1032 detections | **0** |

The median is zero because most detections are already under `QUALITY_MAX_DIM`
and never resize; the 6% worst case is a score of 5.4 against 5.7, where the
relative figure is large and the absolute one is not.

No label moved here, but **six detections sit within 1% of a 300 or 1000
threshold**, so this is "no flips on this folder", not "cannot flip". What
protects the important output is structural rather than lucky: the crop labels in
filenames and the `auto_select_best_face` ranking both come from
`build_processed_crop`, which scores the finished 512x512 crop and so never
reaches the resize at all. Only the JSON report's per-detection score moves.

The call site that pays is `detection_quality`, which cuts the face region out of
the **full-resolution** source and scores that, so the downscale is real work.
`build_processed_crop` scores the finished 512x512 crop, which never crosses
`QUALITY_MAX_DIM` and never resizes at all.

One thing needed care. `DynamicImage::resize` fits inside the box and keeps the
aspect ratio; `resize_rgba_fast` takes exact dimensions, so the fit is now
computed at the call site and has to match `image`'s `resize_dimensions` exactly
-- the smaller of the two ratios, rounded. A dimension out by one pixel changes
the variance, which changes the label, and `auto_select_best_face` ranks by that
same score. `quality_downscale_matches_image_crate_dimensions` checks eight
shapes against what `image` actually returns rather than against my reading of
its source.

**The general point:** 87's result was correct and its conclusion was correct.
What made it stale was not new information about the measurement but a change to
the thing being measured. A rejected experiment whose record names its blocker is
worth re-running the moment that blocker goes; one that just says "no gain" is
not.

### 88. The gate that cost more than the work it was skipping

Re-recorded the batch profile after 71, rather than reusing its attribution: the
deleted cropper was in most of it. **147.0 s of CPU across 50 threads, down from
194.7 s.** `CicpRgb::cast_pixels_by_layout` had gone entirely, confirming it was
the cropper's `to_rgba8` and not a cost of its own.

Two things moved to the top:

| Cost | Self | Share |
| --- | ---: | ---: |
| `crop_face_from_image` -> `image::imageops::resize` | ~16.0 s | **10.9%** |
| PNG deflate (`zlib_rs::longest_match` + `deflate_medium`) | ~10.7 s | 7.3% |
| `DynamicImage::resize_exact` (`estimate_sharpness`) | 5.5 s | 3.7% |

The first is the RGBA fast-resize candidate 87 left open, now the largest single
cost in the run. `crop_face_from_image` resizes an RGBA canvas -- the fill colour
carries an alpha and rotation needs somewhere to put the corners -- and
`resize_image_fast` accepts RGB8 only, so it was still going pixel-by-pixel
through `GenericImageView`.

**Adding an RGBA path was worth about 2%, and nearly hid the real finding.**
Measured on its own against separate binaries it first came out *slower*, losing
four pairs out of four. The cause was not the resize:

```
26335 ms  17.9%  rayon_core::registry::Registry::in_worker_cross<...
                   ...image_utils::resize_image_fast...>   (registry.rs:561)
```

`threading_pays` returning false does not mean "run it here". It means
`single_thread_pool().install(...)`, and from inside a rayon worker that is a
**cross-registry hop**, not an ordinary join. The batch pays it on every resize
under 4 MP, on both the crop path and the preprocessing path. The gate was
costing more than the threading it existed to avoid.

Four-way matrix in one binary, order-alternated (1239 images, `Quality`,
rectangle):

| Variant | Runs (s) | Median |
| --- | --- | ---: |
| baseline | 9.77, 9.86, 9.50 | 9.77 |
| RGBA fir crop only | 9.27, 9.58, 9.58 | 9.58 |
| no threading gate only | 8.03, 7.86, 8.66 | **8.03** |
| both | 7.55, 7.25, 7.95 | **7.55** |

**The gate is ~18% of batch wall time; the resize is ~2%.** They compose, and
the ranking held in both directions.

**Not fixed by deleting the gate.** The 4 MP threshold was measured, and it was
measured correctly -- one image at a time, threading a sub-4 MP resize really is
0.79-0.89x. Both facts hold; they describe different callers. So the gate now
yields only where the hop is expensive:

```rust
if rayon::current_thread_index().is_some() {
    return true;
}
```

Off a worker -- single image, GUI preview, webcam -- nothing changes.

Final A/B, separate binaries, order-alternated: **9.58 s to 8.17 s median**,
winning all three pairs in both orders, about **15%**.

Output: the threading change is bit-exact, 901 of 901 crops identical. The RGBA
resize changes pixels at the rounding level, which is what swapping one Lanczos3
implementation for another should look like -- max channel difference **23**
across the whole folder against 18-85 for the GPU-crop change in 71, mean 0.13,
and **one** quality label flips of 901 (0.11%) against 18.1% there.

**Three things this cost me.** The first A/B compared separate binaries and said
the change was a loss; the four-way matrix in a single binary, with the probe
switches in the shared helper, is what separated the two effects. I had put the
probe switch in `resize_pixels_fast` rather than in the RGBA wrapper, so it
disabled the gate for the preprocessing path too -- the accident is the only
reason the 18% was visible at all.

And the change broke a test, `eye_line_rotation_matches_an_explicit_rotation`,
which built its reference with `image::imageops::resize` and asserted byte
equality. Its subject is the rotation angle, so it was pinning an implementation
it did not mean to test; it now compares with a tolerance of 4, against a wrong
angle that would move pixels across a 30-degree tilt. Worth noting how nearly
this was missed: the run was backgrounded, and the harness reported "exit code
0" for the *shell*, while `cargo test` had exited 101 four lines up in the
output. Only reading the log caught it. Neither `cargo build --release` nor
`cargo clippy --all-targets` fails on this -- it compiles fine and is wrong at
runtime, the complement of the deletion in 71 that compiled fine and was wrong
at *link* time for tests only.

**Worth revisiting: 87's rejected `estimate_sharpness` swap.** It measured
neutral, but it measured neutral *through the gate that has now gone*. Same for
the `RESIZE_THREADING_MIN_PIXELS` value itself, which was calibrated on the main
thread and now governs only that.

### 87. Batch profile - one finding, measured, and rejected

`samply` over the whole CLI run, all threads, 1239 images, explicit
`--config` selecting `Quality` so the configuration is not ambiguous.
**194.7 s of CPU across 51 threads over about 16 s of wall time** -- roughly
12.5 cores busy on a 16-core machine, with the top ten worker threads within 15%
of each other at 6.2-7.2 s each.

**There is no serial bottleneck.** The even thread distribution rules out the
GPU queue, the buffer-pool mutex or export serialising the batch, which is what
this experiment was opened to look for. An earlier guess that batch achieved
only about 1.7x over serial work was arithmetic on a per-image cost taken from a
different configuration, and is wrong.

Where the CPU goes (self time, summed across the rows each symbol appears in):

| Cost | Share |
| --- | ---: |
| `fast_image_resize` vertical convolution (AVX2) | ~14% |
| `memset` / `memcpy` | ~9% |
| `image::DynamicImage::resize_exact` | 4.4% |
| `image::metadata::cicp::CicpRgb::cast_pixels_by_layout` | ~4.8% |
| `zlib_rs::deflate::longest_match` (PNG export) | ~2.5% |
| libjpeg `decode_mcu_fast` | ~1% |

**Rejected: routing the quality metric's downscale through the SIMD resize.**
All 4.4% of `resize_exact` blames to one caller,
`fcs_utils::quality::estimate_sharpness` (10.7 s, 5.5% of all CPU) -- the same
pixel-by-pixel `image` resize that `resize_image_fast` was written to replace for
preprocessing, still in use here.

Swapping it changed nothing, order-alternated in both directions:

| Configuration | `image` resize | SIMD resize |
| --- | --- | --- |
| 1239 images, 32 workers | 16.0-17.5 s | 16.7-17.2 s |
| 200 large images, 1 worker | 20.6-23.3 s | 20.5-23.8 s |

The reason is in the call site: callers pass `DynamicImage::ImageRgba8`, and
`resize_image_fast` accepts RGB8 only, so it converts the whole region to RGB
first and hands back exactly what the faster kernel saves. Reverted, with the
reason recorded at the call site so the swap is not retried.

Outputs were identical while it was in place -- 901 crops, byte-identical
filenames, so every quality label matched. The change was neutral, not wrong.

**The real candidate is an RGBA-capable fast resize path** (`fir` has
`PixelType::U8x4`), which would remove the conversion instead of moving it. Not
attempted: a 5.5% CPU saving did not move wall time in either configuration
above, so the case for it is energy and small-machine headroom rather than
throughput, and it should be measured as such.

### 71. GPU batch cropping - the largest saving in the backlog is deleting it

Following the `memcpy` in the batch profile (11% of all CPU) rather than
guessing. Two frames own nearly all of it:

| Frame | CPU | Share |
| --- | ---: | ---: |
| `GpuBatchCropper::crop` | 10.3 s | 5.4% |
| `wgpu Queue::write_buffer` | 8.1 s | 4.3% |

Both are the same path. `GpuBatchCropper::crop` calls `to_rgba8()` on the
**full-resolution source**, packs every pixel into `u32`, and uploads the lot --
40 MB for a 10 MP photo -- to produce one 512x512 crop. It is the trade
`upload_pays_for_source` already gates for preprocessing, ungated. The
`cast_pixels_by_layout` cost elsewhere in the profile is the `to_rgba8` half of
it.

The workflow tries the GPU cropper first and falls back to the CPU only when it
declines. Forcing the CPU path, order-alternated, 1239 images:

| Crop path | Runs (s) | Median |
| --- | --- | ---: |
| GPU | 17.5, 16.3, 17.0, 18.3 | 17.25 |
| CPU | 10.4, 9.6, 9.3, 10.0 | **9.8** |

**43% off total batch wall time by not using the GPU**, winning every pair in
both orders against a roughly 1 s noise band, with the same 901 crops produced.
This is the largest measured saving anywhere in this backlog, and it comes from
removing GPU work rather than adding it.

**It changes output, so it is not adopted unilaterally.** `crop.wgsl` samples a
fixed 2x2 neighbourhood -- four source pixels regardless of how far the crop is
being downscaled -- while `crop_face_from_image` uses Lanczos3. Over 171 crops:

- pixels differ by up to 18-85 per channel, across 66-84% of each image
- **18.1% of crops change quality label** (31 of 171): highq to medq 21, medq to
  lowq 7, and 3 the other way
- one image of 171 selects a *different face*, because `auto_select_best_face`
  ranks faces by the same sharpness score

The label shift is almost entirely downward, which is what a fixed 2x2 tap would
predict: undersampling a large downscale aliases, aliasing adds high-frequency
detail, and the sharpness metric is Laplacian variance, which rewards it. On
that reading the GPU path's higher scores are an artefact rather than sharper
crops -- an interpretation of the mechanism, not a measurement, and the reason
the crops need a human eye before this becomes the default.

Available as `FCS_NO_GPU_CROP` while that judgement is made. Comparison crops
from 200 large images are in `~/fcs-crop-comparison/`.

**Checked against the non-rectangular crop shapes**, since the first result was
measured on `rectangle` alone and the shape masks are the app's more interesting
output. The shape mask is a separate step applied to the finished 512x512 crop,
so it neither shares the upload nor changes the resampling, and the measurements
bear that out. Order-alternated over 1239 images:

| Shape | GPU (s) | CPU (s) | Saving |
| --- | --- | --- | ---: |
| `rectangle` | 15.8, 18.3 | 10.3, 10.7 | ~38% |
| `koch_polygon` sides 3, iterations 4 | 19.5, 17.6 | 9.3, 10.3 | **~47%** |
| `star` 8 points | 17.7, 18.2 | 11.6, 10.6 | ~38% |

Complex shapes are, if anything, the better case: the mask cost is the same on
both sides and lands on a 512x512 image, so it dilutes the GPU path's
full-resolution upload rather than the saving.

Quality-label flips are **identical across all three shapes** -- 31 of 171,
18.1%, same directions -- which follows from where the metric sits:
`estimate_sharpness` runs on the crop *before* `apply_shape_mask`, so the shape
cannot reach it.

The visible pixel difference is *smaller* with a mask than without, because the
masked-away area is identical either way: max channel difference 17-51 against
18-85 for rectangles, over 17-36% of the image against 66-84%. A reviewer
comparing shaped crops is therefore looking at a weaker signal than the
rectangles already judged indistinguishable by eye.

**Adopted after review of both crop sets, and `GpuBatchCropper` deleted** rather
than left unreachable: `crop_batch.rs`, `crop.wgsl`, the CLI's cropper plumbing
and the `BatchCropRequest`/`GpuBatchCropper` exports, about **740 lines removed
against 14 added**. The folder now completes in 9.5 s where it took 17.25 s,
with output byte-identical to the flagged CPU path.

`calculate_crop_region` and `pack_rgba_pixels` stay -- the GUI's export and
canvas use the first, and the blur and bilateral effects use the second.

One thing the release build did not catch: the test helper still constructed the
removed `cropper` field, and `cargo build` does not compile test code. Only
`cargo clippy --all-targets` and the test run found it, which is the argument for
running both before believing a deletion is complete.

### 20. The eight uncached uniforms were a third of the recording cost

Convolution has cached its uniform buffers since an earlier round; the four
max-pools, two 2x resizes and two adds have not, and created one 4-32 byte
buffer per dispatch. The experiment asks whether eight dispatches contribute a
repeatable cost, so the first step was to price the call rather than write the
cache and hope.

`examples/encode_cost.rs` times the two remaining per-dispatch host objects
against the cache lookup that would replace them, on the real device, 20000
iterations each:

| Operation | Median |
| --- | ---: |
| `create_buffer_init`, 16-byte uniform | **8.1 us** |
| cached uniform lookup (mutex + hash + `Arc` clone) | below 0.1 us |
| `create_bind_group` | 0.8 us |
| cached bind group lookup | below 0.1 us |

**A uniform buffer costs ten times a bind group**, which was not the expected
ordering: the bind group binds four storage buffers and a uniform, the uniform
buffer holds sixteen bytes. It is an allocation on the device, and that is what
it charges for. Eight of them predicts 0.065 ms per forward pass against a
`gpu_record` measured at 0.186 ms.

Caching them, order-alternated between two binaries built from the same tree,
0.17 MP fixture, 30 runs each:

| Variant | `gpu_record` p50 | `detect_image` wall p50 |
| --- | ---: | ---: |
| baseline | 0.186, 0.183, 0.179, 0.181 | 1.182, 1.153, 1.172, 1.157 |
| cached | **0.103, 0.105, 0.103, 0.109** | **1.041, 1.038, 1.033, 1.065** |

**Recording drops 43%, and small-image detection drops 10%** -- 0.078 ms off
recording, slightly more than the 0.065 ms the probe predicted, the remainder
being the eight buffers' destruction. Every pair wins in both directions with no
overlap between the two distributions.

On the 10 MP fixture the recording saving is the same shape -- 0.653-0.753 ms
down to 0.525-0.619 -- but wall time does not move outside its noise band
(4.61-5.07 against 4.58-4.84), because that path spends 2.65 ms in CPU
preprocessing before it records anything. **This is a small-image win**, which is
the case experiment 6 identified as detection-bound: under 1 MP detection is 64%
of per-image cost, and webcam frames live there.

The cache is the one convolution already had, moved to
`utils::UniformCache<T>` and used by all four pipelines, so conv2d's bespoke copy
was deleted rather than duplicated three more times. Keyed by contents, bounded
by distinct shapes; the small ops' shapes derive from the 640x640 detector input
rather than from the source image, so the source resolution cannot grow it at
all -- a stronger bound than convolution's.

**Bit-exact:** the 126000-float output fingerprint is `0xa116e42f7c2dabdb`
before and after (`readback_parity`), and the whole workspace suite passes under
`FCS_STRICT_TESTS=1` with ONNX Runtime 1.24.4.

**Validated on a batch run**, because 67's regression says a folder is where a
shared cache or thread-local goes wrong and nothing else nests rayon. This adds
three mutexes hit eight times per inference across every concurrent worker, so
contention was the thing to rule out. Order-alternated, 1239 images:

| Variant | Runs (s) |
| --- | --- |
| baseline | 7.58, 6.48, 6.88 |
| cached | 7.44, 6.70, 6.87 |

Neither faster nor slower -- the ranges sit on top of each other, which is the
expected result for a 0.08 ms per-image saving against a decode-bound folder,
and the point was the absence of a contention regression. All 901 crops are
byte-identical with identical filenames.

**Experiment 22 is sized by the same probe and is not worth writing.** Caching
all 61 bind groups would recover 0.049 ms, and unlike the uniforms they depend on
pooled intermediate buffers whose identities change between passes, so the cache
would need invalidation the uniform cache does not. Less than two thirds of this
experiment's saving for materially more machinery, against a recording cost this
experiment has already cut to 0.103 ms. Left unchecked with that as the reason.

### 8. The profiled breakdown is honest, and the dispatches are not compute-bound

Opened because experiment 26's sweep produced a suspicious shape: a 20x20 64->64
pointwise dispatch measured 12.288 us and a 160x160 one measured 27.648, a 2.2x
spread over 64x the arithmetic, and the summed profiled total (0.538 ms) is
larger than the whole `readback_wait` of the unprofiled path (0.457 ms), which
contains the same forward pass *plus* the head copies. Either the per-pass
timestamps were measuring the profiler, or the dispatches were not doing what
their arithmetic suggests.

`examples/pass_overhead.rs` records 26 identical dispatches two ways in one
process: 26 separately timestamped passes, as the profiled runtime does, then all
26 inside one pass under a single timestamp pair. Medians of 30 runs after 10
warm-ups, RTX 4090 / D3D12:

| Case | summed | merged | wall | per pass |
| --- | ---: | ---: | ---: | ---: |
| 160x160 64->64 | 705.5 | 684.0 | 801.2 | 0.83 |
| 80x80 64->64 | 353.3 | 331.8 | 444.3 | 0.83 |
| 20x20 64->64 | 299.0 | 277.5 | 398.3 | 0.83 |
| 1x1 4->4 | 68.6 | 48.1 | 164.9 | 0.79 |

**A pass boundary costs 0.83 us, flat.** So `gpu_pass_breakdown`'s 61 passes carry
about 51 us of profiling, and its 0.538 ms is really 0.487 ms of work -- a 9%
overstatement, worth knowing but not an artefact. The breakdown can be trusted,
and experiment 8 closes without a change.

**The 12 us floor is real, and it is occupancy.** Per dispatch, merged:

| Shape | Arithmetic | Time | Workgroups dispatched |
| --- | ---: | ---: | ---: |
| 160x160 64->64 | 1x | 26.31 us | 1600 |
| 80x80 64->64 | 1/4 | 12.76 us | 480 |
| 20x20 64->64 | 1/64 | 10.67 us | 48 |
| 1x1 4->4 | ~0 | 1.85 us | 1 |

The true fixed cost of a dispatch is 1.85 us, not 12, so the 20x20 layer is not
paying overhead -- it is paying **latency it has no parallelism to hide**. The
production grid is four pixels and four output channels per thread in an 8x8
workgroup, so a 20x20x64 output is `ceil(20/32) * ceil(20/8) * ceil(64/4)` = 48
workgroups on a 128-SM adapter, with two thirds of the machine idle and a serial
64-iteration accumulation loop in each thread. At 3.3 MFLOP in 10.67 us that is
0.3 TFLOPS on a part that does eighty.

**This redirects the shader backlog.** Arithmetic-level tuning (26, 28, 30, 33,
41-46) cannot help a kernel that is 0.4% utilised. The two candidate levers were
more threads doing less each on small layers (27), and fewer, larger dispatches
(35-37).

**The first of those was tested next and does not work.** Experiment 27 cut the
tile to one output channel per thread, which quadruples the 20x20 layer's
workgroups to 192 -- and it measured 8% *slower*, because four channels share
each loaded input vector and dropping to one quadruples the loads per
multiply-add. So the small layers are not idle for want of workgroups, and the
0.4% utilisation figure above describes what the machine is doing rather than
what is holding the kernel back. Read this experiment as "the profiled numbers
are honest and the dispatches are far from peak", not as "occupancy is the
lever". That leaves 35-37.

### 26. Pointwise workgroup shapes - swept, nothing wins

Nine shapes against the production 8x8, over the ten pointwise configurations in
`conv2d_experiment`, coverage adjusted per variant so each covers the same
output: 16x4, 32x2, 4x16, 8x4, 16x8, 8x16, 16x16, 32x4, 64x1.

**The A/A control is exact.** Running the production shader against itself
reports 0.0% on all ten shapes, in every repetition -- the harness alternates A
and B inside one command buffer, so between-run clock drift cancels. Timestamps
quantise to 1.024 us ticks, and a tick is 4-8% of most of these dispatches, so
that control is what makes a one-tick difference readable at all.

Results, in ticks rather than percentages:

- **Nothing beats 8x8 anywhere it matters.** On the 160x160 64->64 layer, which
  is the single most expensive pointwise dispatch, every variant is level or
  worse: 32x2 and 32x4 lose a tick, 8x4 loses 7, 4x16 loses 8.
- **One reproducible win, and it is worth 1 us.** Every variant with 16 or more
  threads in x takes 12.288 us on the 320x320 16->16 layer against 8x8's 13.312,
  reproduced 4 times out of 4 on both 16x4 and 16x8 while A/A stays at 0.0%. That
  is one tick on one dispatch: 0.2% of the graph's GPU time.
- **Narrow-x shapes lose badly.** 4x16 costs +69% at 320x320 and +30% at
  160x160; 8x4 costs +26% at 160x160. Four threads covering 16 pixels leaves too
  little row per workgroup.
- **The tiny head shapes are noise.** 17x5 3->7 is 3-4 ticks in total and flips
  sign between repetitions of the same variant; nothing there is readable.

**Not adopted.** A per-shape kernel selected for one dispatch's single tick
cannot pay for the pipeline, selection and testing it would need, which is the
condition this experiment was written with. The sweep's real value is the
evidence it handed to experiment 8: the dispatches these shapes were being tuned
for are occupancy-bound, so the geometry that matters is threads per output, not
threads per workgroup.

### 27. The 4x4 tile is a local optimum, and the graph's shape mix is what pins it

Experiment 8 predicted the promising direction was a *smaller* tile: the 4-pixel,
4-channel grid leaves a 20x20x64 output with 48 workgroups on 128 SMs, and more
threads doing less each would fill the machine. **That prediction is wrong**, and
the sweep says why.

Channels per thread, against production's 4, over the ten pointwise
configurations (`conv2d_experiment`, A/A control 0.0%):

| Shape | 1 ch | 2 ch | 8 ch |
| --- | ---: | ---: | ---: |
| 320x320 16->16 | +115% | +46% | **-15%** |
| 160x160 16->64 | +125% | +50% | **-17%** |
| 160x160 64->64 | +248% | +96% | **-19%** |
| 80x80 64->64 | +71% | +14% | +21% |
| 40x40 64->64 | +8% | 0% | +31% |
| 20x20 64->64 | +8% | 0% | +31% |
| 80x80 64->1 | 0% | 0% | +25% |
| 40x40 64->10 | +8% | +8% | +31% |

**Occupancy is not what these kernels are short of; reuse is.** Cutting to one
channel per thread quadruples the workgroup count on the 20x20 layer -- 48 to 192,
exactly the fix experiment 8 suggested -- and it gets 8% *slower*. Four output
channels share each loaded input vector, so dropping to one quadruples the loads
per multiply-add, and on the 160x160 layer that costs 248%. The small layers were
never idle for want of workgroups.

**8 channels is a genuine win on the three largest layers and still loses the
graph.** It earns 15-19% where the tile has enough work, so it got the full-graph
trial the acceptance rules require. `POINTWISE_CHANNEL_TILE = 8` with the matching
shader, over the real 26 pointwise dispatches:

| Variant | pointwise total | whole-graph GPU |
| --- | ---: | ---: |
| production, 4 channels | 320.5, 324.6 us | 0.538, 0.537 ms |
| 8 channels | **397.3 us** | **0.611 ms** |

**23% worse on the graph** while being 15-19% better on the shapes the
microbenchmark ranks first, because YuNet's pointwise work is mostly the small
layers where 8 channels costs 21-31%. Reverted. It was bit-exact
(`0xa116e42f7c2dabdb`) while installed, so this is a speed rejection, not a
correctness one -- and it is the clearest example in this backlog of why a
faster microbenchmark earns a full-graph trial rather than adoption.

**Eight pixels per thread loses everywhere**, 15% to 108%, tested with two vec4
accumulators per channel and coverage doubled to match. A 20-wide output covered
by 8 threads at 8 pixels each is 64 pixels of grid for 20 of work, and the tail
waste swamps the extra reuse.

**Not adopted, and the avenue is closed at this altitude.** Both axes are at a
local optimum, and the only variant that beats production on any shape loses the
graph by more than it wins. A per-shape kernel would recover 15-19% of the two or
three largest pointwise dispatches -- single-digit microseconds against 0.487 ms
of GPU compute and a 1.04 ms detection -- which does not pay for a second
pipeline and its selection rule. What remains untested on the shader side is not
geometry but graph structure: fewer, larger dispatches (35-37).

### 37. Four head branches are one convolution, and that is 26% of the GPU graph

Experiment 8 left two levers: fewer, larger dispatches, or nothing. This is the
first of them, and it is the largest GPU saving in the backlog.

**The four branches at one detection level are the same computation with
different weights.** Each is a 1x1 convolution from the shared feature map
followed by a per-channel 3x3, differing only in output channels: cls 1, obj 1,
bbox 4, kps 10. So both halves concatenate along the output-channel axis. A
pointwise output channel depends only on its own row of weights, and a depthwise
channel only on its own 3x3 kernel, so **nothing crosses a branch boundary** and
one convolution over the concatenated weights computes all four.

The concatenation happens once, at weight upload, from four ONNX initializers
into one tensor per part. Eight dispatches per level become two:

| | Dispatches | GPU compute |
| --- | ---: | ---: |
| baseline | 61 | 0.537, 0.538 ms |
| fused heads | **43** | **0.396, 0.399 ms** |

**26% off the graph for removing 18 dispatches**, and the shape of the saving
confirms experiment 8's diagnosis: the head dispatches were 12 pointwise and 12
depthwise on 80x80, 40x40 and 20x20 feature maps with 1 to 10 output channels --
the exact case measured at ~12 us regardless of arithmetic. The fused
convolutions do the same arithmetic in a quarter of the dispatches and take
barely longer than one of the originals.

Wall clock, alternated between two binaries built from the same tree, 0.17 MP
fixture, 30 runs each:

| Phase | baseline | fused |
| --- | --- | --- |
| `readback_wait` (contains GPU execution) | 0.435-0.469 | **0.361-0.375** |
| `gpu_record` | 0.102-0.109 | **0.077-0.088** |
| `detect_image` wall | 1.038-1.073 | 0.915-1.052 |

`readback_wait` drops in **all 11 pairs with no overlap** -- that is the GPU
saving arriving. Recording drops because there are 18 fewer dispatches to
record. Whole-detection wall time moves about 0.1 ms and wins 7 pairs of 8; the
one loss came from a block whose p95 was 2.8 ms, so the wall number is the
weakest of the three and is reported as ~0.1 ms rather than a percentage.

**Bit-exact.** The 126000-float output fingerprint is `0xa116e42f7c2dabdb`
before and after, which is the claim the concatenation argument predicts: same
arithmetic, same order, different grouping.

**The concatenation order has an independent check**, which matters more than the
fingerprint here: the risk in this change is not arithmetic but putting kps where
bbox should be, and a fingerprint over the assembled output would not
necessarily catch a consistent mis-slicing. `cpu/graph.rs` still computes the
four branches separately, and `gpu_cpu_parity` compares GPU detections against
it over the fixture corpus. Swapping any two branches moves boxes and landmarks,
not bits. It passes.

**One test needed updating and it is not a weakened check.**
`profiled_and_merged_inference_match` asserts the per-label dispatch counts, 61
with 26 pointwise and 26 depthwise; it now asserts 43 with 17 and 17. The
output-equality half of that test, which is the parity half, was untouched and
passed throughout.

Two things fell out on the way:

- **Twelve GPU buffers per model stopped being uploaded.** The per-branch
  initializers are inside the fused tensors, so uploading them again left
  buffers nothing binds. `upload_gpu_weights` now skips what it superseded.
- **The readback went from 12 staging buffers to 3**, because each level is one
  channel-major buffer holding cls, obj, bbox and kps in that order. That is worth
  about 0.03 ms on its own: `readback_alloc` falls from 0.047-0.063 to
  0.018-0.026 ms. Experiment 13 measured *packing* the staging buffers as neutral
  and this does not contradict it -- 13 packed twelve copies into one buffer and
  still allocated for twelve, where this allocates three because there are three.

**Splitting the fused buffer was not free, and the first version cost more than
it should have.** `gpu_convert` went from 0.001 ms to 0.058: `Vec::split_off`
copies the tail it returns, so peeling cls, then obj, then bbox moves everything
still ahead of the cut each time -- 39 rows of memory movement for 16 rows of
data. Peeling from the back instead copies each branch once and leaves the first
in place with no copy at all, which halves the stage (0.022-0.024 to 0.009-0.016
ms, winning four pairs of four). The remaining ~0.01 ms is the one unavoidable
copy out of the download buffer; removing it would mean splitting inside
`batch_download` at the point it already copies, which is not worth the parameter
it would need.

The hardcoded `HEAD_BRANCH_CHANNELS` is not a new assumption -- `build_decode_tensors`
already hardcoded the same 1/1/4/10 -- and it is now cross-checked against the
concatenated weight tensor at encode time, which the old code did not do.

**No change to a folder job, as expected.** 1239 images, order-alternated:
6.94/6.50/6.43 s baseline against 7.71/6.43/7.32 s fused, with all 901 crops
byte-identical. A 0.09 ms per-image saving is 0.11 s over the folder against a
spread of nearly 1 s within each variant, so batch cannot see this and the run
was for the shared-weight-map validation rather than the timing. Where it lands
is interactive and webcam-sized work, which is what experiment 6 said detection
dominates.

**Kept.** The remaining fusion candidates are 35 (depthwise into pointwise, 34 of
the 43 surviving dispatches) and 36 (the eight pool/resize/add boundaries, 49 us
between them).

### 34. The stem was reading the source sixteen times

After 37 the single most expensive dispatch in the graph was `conv2d/general` at
42 us -- one dispatch, 10.6% of GPU compute. It is the 640x640 3->16 stride-2
stem, and it was on the fallback path that no specialization had touched.

The problem is not the loops this experiment proposed fixing. It is that the
general path computes **one output channel per thread**, so each of the 16 output
channels gathers the same 27 input values independently: the stem reads its
source sixteen times over. Pointwise has taken four channels per thread since
experiment 3, and an ungrouped general convolution has exactly the same property
that makes that work -- every output channel gathers the same inputs.

Adding the same four-channel tile to the ungrouped general path:

| | `conv2d/general` | whole-graph GPU |
| --- | ---: | ---: |
| before | 42.0 us | 0.396, 0.399 ms |
| after | **18.4, 18.4, 19.5 us** | **0.376, 0.376, 0.378 ms** |

**56% off the dispatch**, 5% off the graph, and `readback_wait` falls from
0.369-0.378 to 0.349-0.366 ms in all five alternated pairs. Whole-detection wall
time does not resolve it: 20 us is well inside a noise band nearer 0.2 ms on this
machine, and it is not claimed.

**Bit-exact.** The arithmetic and its order are unchanged -- bias, then ic, ky,
kx, the same `fma` chain per channel -- so the output fingerprint stays
`0xa116e42f7c2dabdb`. Only how many channels one thread carries changed.

The grouped general path keeps one channel per thread, because there `oc + j` can
cross a group boundary and the four channels would not share a gather. Nothing in
YuNet uses it; it is the public `conv2d` API's fallback, and the conv2d tests are
what cover it.

**The host predicate is the risk in this change, not the shader.** `main` picks
its path from the uniforms and the host sizes dispatch z for whichever it will
pick, so the two must agree exactly. Both now spell out all three conditions --
pointwise, depthwise, ungrouped-general -- next to each other with a comment
saying they are mirrors. A disagreement would silently compute a quarter of the
output channels, which is what the parity fingerprint and the GPU/CPU detection
comparison exist to catch.

Validated the same way as 20 and 37: the whole workspace suite passes under
`FCS_STRICT_TESTS=1` -- including the conv2d tests, which are the only cover the
grouped fallback has -- and a 1239-image folder produces 901 byte-identical
crops (7.57/7.81 s against 8.57/7.98 s, which is noise either way).

**Kept.** 20 us is a small absolute saving, but it is the third-largest single
item found in the GPU graph, it costs one shader function and a predicate, and it
compounds with 20 and 37: the graph is now 0.376 ms against the 0.537 ms this
session started from, **30% less GPU work for the same bits**.

### 92. One compute pass for preprocessing and inference - rejected, and the reason is 5's

The 5 (continued) breakdown showed two `queue.submit` calls costing 0.055 and
0.093 ms, 18% of a detection, on the same queue in a fixed order. `encode_cost.rs`
priced what merging them would actually buy, 2000 iterations each:

| Configuration | Median |
| --- | ---: |
| empty `queue.submit` | 3.5 us |
| record + finish one dispatch, no submit | 30.3 us |
| submit one dispatch | 55.7 us |
| two command buffers, submitted separately | **112.6 us** |
| two command buffers, one submit call | 99.2 us |
| **one pass, two dispatches, one submit** | **57.9 us** |

**The cost is not the submit.** An empty one is 3.5 us; it is the encoder and the
compute pass around the dispatch that cost ~30, and the submit that carries them
another ~26. Merging the two *submits* alone saves 13.4 us. Putting both
dispatches in **one pass** saves 54.7 us, which matched the production numbers
exactly -- so that is what was built.

It works, and it does not help. `PreparedPreprocess` splits preparation (source
conversion, texture upload, bind group) from recording, so the dispatch can go
into the inference pass and the "can the GPU take this source" decision still
happens before any encoder exists. Bit-exact, and the host cost went where it was
supposed to:

| Phase | two passes | merged |
| --- | ---: | ---: |
| gpu_preprocess | 0.162 | **0.102** |
| - encode (incl. its submit) | 0.073 | 0.016 |
| readback_wait | 0.354 | **0.397** |
| detect_image wall p50 | 0.829-0.860 | 0.835-0.889 |

**The 60 us saved on the host is handed straight back by the wait**, and merged
loses four alternated pairs of five. The preprocess dispatch takes about 0.04 ms
of GPU time, and in the two-pass arrangement it is submitted early and runs
*while the host records the 43 inference dispatches*. Merging serialises it
behind them.

This is the same rule experiment 14 established, in the other direction, and it
is already written down in PERFORMANCE.md: host work between the inference submit
and the readback wait is free, so moving work into that window helps and moving
work out of it hurts. Here the thing moved out of the window was GPU work, and
55 us of host saving bought about 43 us of lost overlap.

**Reverted.** `examples/encode_cost.rs` keeps the six submit and pass
measurements, because they are the numbers any future encoding experiment needs
and they are what made this decidable without guessing. What they say for the
backlog: **an encoder plus a compute pass is ~30 us and a submit ~26**, so
merging passes is only worth trying where the merged-away dispatch is not
currently overlapping something.

### 22. Bind group caching - reopened, and this time measured

**Closed once on reasoning, and the reasoning was wrong.** Experiment 20's probe
priced `create_bind_group` at 0.8 us and this item was dismissed on two grounds:
that 61 of them are only 0.049 ms, and that they reference pooled intermediates
"whose identities change between passes". The first was a fair ranking at the
time. The second was an assumption, and it is false.

Two things changed the ranking. Experiment 37 cut the graph to 43 dispatches and
20/34 cut everything around them, so 0.049 ms stopped being small next to what
was left. Then splitting `gpu_submit` showed **`gpu_finish` is 0.059 ms of it**,
so host encoding is `record` 0.073 plus `finish` 0.059 -- **3.1 us per dispatch**,
not the 1.7 that `record` alone suggested.

**The pool does hand the same buffers back.** Keyed on the five buffers a
convolution binds, over 25 inferences: **730 hits, 110 misses**. The misses are
the first few passes -- the pool cycles through about three assignments before it
settles, which is what an execution scope releasing everything at once produces --
and every inference after that is a full hit. That is the fact the earlier
rejection guessed at and got backwards.

Alternated between two binaries built from the same tree, 0.17 MP, 30 runs each:

| Variant | `gpu_record` | `detect_image` wall p50 |
| --- | --- | --- |
| baseline | 0.075, 0.072, 0.075, 0.072, 0.073 | 0.896, 0.856, 0.856, 0.848, 0.859 |
| cached | **0.030, 0.029, 0.028, 0.030, 0.029** | **0.828, 0.804, 0.805, 0.830, 0.799** |

**Recording drops 60% and whole detection 6%**, winning all five pairs on both
measures with no overlap between the distributions. Bit-exact: the raw head
fingerprint is unchanged.

**The cache is only worth its lookup while the pool keeps its ordering**, and
nothing in the pool promises that. A change to release order would turn this into
pure overhead silently, so `conv2d_bind_groups_are_reused_across_inferences`
asserts the hit rate stays above 75% over ten inferences and says in its failure
message what the collapse would mean. The counters exist for that test.

Keyed on the buffers rather than on the layer, so a miss rebuilds a correct bind
group rather than binding a wrong one, and cleared wholesale past 512 entries so a
caller sweeping input sizes cannot pin pooled buffers indefinitely -- the cache
holds `Buffer` handles, so an entry keeps its intermediate alive.

**Not extended to the other nine dispatches.** Max-pool, add and resize also build
a bind group each, but nine of them are about 7 us against the 34 convolutions'
27, and they would need the key generalised over three different layouts. Worth
doing only if recording matters again.

**No measurable change to a folder job**, which is decode-bound: 8.00 and 7.55 s
against 8.90 and 7.89, winning both pairs but well inside the noise for a saving
of 0.056 s over 1239 images. All 901 crops byte-identical.

### 97. Live webcam detection in the GUI, and what it actually costs

95 established the loop is capture-bound, with detection answering in a seventh
of the frame budget. This is the wiring that spends that headroom, and the two
measurements 95 could not make from a CLI probe: the GUI's own per-frame cost,
and what detection costs when it shares a device with the renderer.

**The GUI had scaffolding and no implementation.** `JobMessage::WebcamFrame`
existed with a `detections` field and was never constructed or handled anywhere.
`detect_webcam_faces` was a one-shot button, and its `DetectionFinished` path
uploads a preview texture, rebuilds every face thumbnail, clears the edit
history, resets rotation and the manual box tool and re-selects every face --
right for one deliberate detection, impossible at frame rate. So live detection
needed its own message and its own handling, not a faster call to that one.

**Measured live, in the running application:**

| | CLI probe (95) | in the GUI |
| --- | ---: | ---: |
| texture upload per frame | not measurable | **1.8 us** p50 |
| `detect_image` | 2.45-3.59 ms | **6.89 ms** p50, 7.58 p95 |
| frames displayed | 24 fps @ 640x480 | 15.0 /s @ 1280x720 |
| detections | -- | 14.3 /s |
| **frames tracked** | -- | **95%** |

**The texture upload was the risk and is a non-issue.** `color_image_from_dynamic`
plus `ctx.load_texture` measured 1.8 us because egui queues the upload rather
than performing it; the real transfer is absorbed into the frame it is drawn in.

**Contention roughly doubles detection.** 6.89 ms against 3.5 standalone, because
the GUI shares eframe's device and detection now competes with rendering. That
was the second thing 95 could not measure and it is the larger of the two
effects -- but 6.89 ms against a 69 ms interval is 10%, so it changes the margin
rather than the answer.

**One detection in flight at a time.** A frame arriving while one is running is
counted and dropped rather than queued: an overlay that queues falls further
behind the picture the longer it runs, and the frames are already drained-to-latest
by `poll_webcam_frames`. That drain, which predates this work, is what experiment
65 proposed building; it was already there.

**A measurement mistake worth recording.** The first reading of the session log
said 0.56 detections per frame, which looked like the loop failing to keep up.
It was counting frames from before the toggle was switched on. Over the window
where live detection was actually running the figure is 0.95, and the 15 fps is
capture and repaint bound, not detection bound. The lesson is the same one the
threshold error in 96 taught: a rate is only meaningful over the interval the
thing was running.

Not touched: the crop, quality and thumbnail work stays on the deliberate button,
because none of it is wanted twenty-four times a second. Live results carry
`Quality::Low` and no thumbnail, which nothing reads for an overlay.

Note for 96: the GUI's camera default is 1280x720, the 16:9 case that scores
worst under the stretch-to-square preprocessing. Faces track fine at production's
threshold, which is consistent with 96's corrected numbers rather than its first
ones.

### 96. Squashing to a square costs recall, and the first numbers overstated it

**Correction first.** The version of this record committed with 95 said 16:9
webcam detection "fails almost completely" and that letterboxing finds four times
as many faces at 16:9, with 28 of 77 corpus images finding nothing. Those runs
used `PostprocessConfig::default()`, whose score threshold is **0.9**.
Production reads **0.8** from `config/gui_settings.json`. At the threshold the
application actually uses, 16:9 detection does not fail -- the effect is a
smaller score margin, not a missing detection. The corrected numbers are below;
the earlier ones measured a configuration nobody runs.

**The mechanism is real.** `preprocess.wgsl` computes `ratio = src_size /
dst_size` and samples `(pixel + offset) * ratio`, scaling x and y independently,
so a source is squashed to 640x640 whatever its shape: 1.78:1 for 16:9, 1.50 for
3:2, 1.33 for 4:3. The box mapping back out is consistent, so nothing downstream
is wrong. What is distorted is the face the model is shown, and that costs
confidence.

**One webcam frame, sixteen times, everything derived from the same capture** so
the subject cannot move between candidates, at production's 0.8:

| Detected on | Faces per frame |
| --- | ---: |
| 16:9 2.07 MP (as captured) | 1.00 |
| 16:9 0.92 MP | 0.94 |
| 16:9 0.52 MP | 0.88 |
| 16:9 0.23 MP | 0.69 |
| 4:3 centre crop, 1.56 / 0.48 / 0.31 MP | 1.00 |
| **16:9 letterboxed into 640x640** | **1.00** |

At 0.9 every 16:9 row read 0.00 and every 4:3 and letterboxed row 1.00, which is
what produced the overstated claim. The truthful reading of both is the same
mechanism at different strengths: **distortion pushes scores down**, and whether
that crosses the threshold depends on the threshold and on how much else the
frame has going for it. Small 16:9 frames still degrade at 0.8 -- 0.69 at
0.23 MP -- because they are short of both resolution and shape.

**On the corpus, at production's threshold.** `examples/aspect_recall.rs` detects
each image twice, as production does it and letterboxed, mapping the letterboxed
detections back to source pixels so boxes and landmarks are comparable. All 1239
images:

| Aspect | Images | Faces now | Faces boxed | 0 -> found | found -> 0 | med IoU | med landmark px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| near square (<1.05) | 11 | 10 | 10 | 0 | 0 | 0.998 | 0.33 |
| 1.05-1.25 | 22 | 19 | 20 | 1 | 0 | 0.910 | 10.1 |
| 1.25-1.45 (4:3) | 510 | 345 | 354 | 11 | 6 | 0.868 | 34.2 |
| 1.45-1.70 (3:2) | 301 | 302 | 320 | 12 | 6 | 0.811 | 46.0 |
| over 1.70 (16:9) | 395 | 354 | 425 | **53** | 6 | 0.760 | 34.8 |

**1030 faces become 1129, and 77 images gain a detection against 18 that lose
one** -- a net 59 of 1239, about 5%, concentrated where the mechanism predicts.
Near-square images are untouched, which is the control: 0.998 IoU and 0.33 px of
landmark movement is the noise floor of the comparison itself.

**The geometry moves, and that is the real cost.** Median IoU 0.76-0.87 on
non-square sources and median landmark displacement 34-46 source pixels. For
scale, experiment 90 rejected a 13% speed win because it moved landmarks up to
158 px and even a backed-off version moved 35. This moves them by a comparable
amount on the median image, not the worst one.

**Which set is more correct cannot be settled from these numbers.** The current
path shows the model a distorted face, so its landmarks may well be the wrong
ones and letterboxing may be *correcting* them rather than moving them. Nothing
here distinguishes those, and the crops are what would.

**The crops decided it, which is what the record above asked for.** A temporary
switch went into the GUI so a real folder could be cropped both ways. The verdict
was that the crops were not visibly different and the run produced 57 more of
them, so letterboxing is now what the preprocessor does -- no setting, no branch,
and the switch deleted. Padding is black; 96 measured black against 114 at four
faces in 1239.

**The shape of it.** `fit_input` returns one scale for both axes, the size the
source is drawn at, and a centred origin. All three preprocessing paths take it:
`preprocess.wgsl` samples the drawn region and writes bars outside it,
`rgb_to_chw.wgsl` does the same for the bytes the CPU resize hands it, and
`rgb_to_bgr_chw_letterboxed` writes the bars on the CPU path rather than clearing
a buffer first. Postprocessing does not know about any of it:
`remove_letterbox_offset` subtracts `origin * scale` from the finished detections,
which is the same as subtracting `origin` before the multiply and leaves
`apply_postprocess` and its sixteen call sites alone.

**It is not slower; on a non-square source it is faster.** The resize target is
now the drawn region rather than the full square, which is 44% fewer output pixels
for a 16:9 source, and the byte upload shrinks with it. `phase_timings --mp N`,
medians of three runs on each build, against `bcee920`:

| Source | stretched | letterboxed | delta |
| --- | ---: | ---: | ---: |
| 0.5 MP (whole-source GPU path) | 0.935 | 0.914 | -0.02 |
| 2.0 MP | 1.860 | **1.480** | **-0.38** |
| 6.0 MP | 1.900 | 1.820 | -0.08 |

`cpu_resize` 0.947 -> 0.669 at 2 MP and 0.929 -> 0.863 at 6 MP; `gpu_rgb_to_chw`
0.182 -> 0.149 and 0.211 -> 0.193. The gain shrinks as the source grows because
the resize is bounded below by reading the source once (experiment 51), and only
the writing side got smaller. The 0.5 and 6.0 MP rows are inside the run-to-run
spread, which was 0.834-1.090 ms on the 0.5 MP point; the 2.0 MP row is not.

**The one thing that had to get more expensive did not.** `preprocess.wgsl`'s tap
rule was `ceil(ratio * 0.5)`, a *coverage* criterion -- taps spaced `ratio / n`
apart, each spanning about 2 texels, leave no gaps once `n >= ratio / 2`. Covering
the box is not the same as weighting it the way the CPU resize does, and the
difference had been invisible because the short axis of a stretched non-square
source was an *upscale*, where a bilinear tap and a triangle kernel are the same
thing. Letterboxing makes that axis a downscale too, and the gap appeared as a
CPU/GPU detection mismatch: **0.0010 of score and 0.80 px of box, where the two
paths had previously agreed exactly.** `ceil(ratio)` brings it to 0.0004 and
0.23 px, inside the tolerance the test already had. It is four times the samples
and it did not move `gpu_preprocess` at all (0.269 -> 0.270 ms), because that
phase is host-side RGBA conversion and texture upload, not shader time --
experiment 8's finding that these dispatches are not compute-bound, arriving again.

**The corpus, through the shipped implementation** (`aspect_recall`, which no
longer letterboxes on its own and is 40 lines shorter for it):

| Aspect | Images | Faces before | Faces now | Images with no face |
| --- | ---: | ---: | ---: | ---: |
| near square (<1.05) | 11 | 10 | 10 | 2 |
| 1.05-1.25 | 22 | 19 | 20 | 2 |
| 1.25-1.45 (4:3) | 510 | 345 | 355 | 179 |
| 1.45-1.70 (3:2) | 301 | 302 | 320 | 39 |
| over 1.70 (16:9) | 395 | 354 | 425 | 57 |
| **total** | **1239** | **1030** | **1130** | |

1130 against the 1129 the temporary switch produced -- one 4:3 borderline face,
which is the denser sampling rather than the letterboxing.

**Two parity tests had to change what they claim, and that is worth stating
plainly.** The OpenCV fixtures were produced by a pipeline that stretches. Against
them this detector now differs by up to 0.0175 of score and 144 px of box, with
IoU 0.765-0.994 -- the same 0.76-0.87 band the table above predicted. Coordinate
equality with a differently-preprocessed reference is not a property this code has
any more, so both tests now assert the same count, box IoU >= 0.70, and landmark
movement within 15% of the face (measured worst: 9.6%), with the measurements
written into the constants. They also pair faces by overlap rather than by score
rank, because a 0.0175 score shift is enough to swap two similar faces and report
the wrong pair.

**One fixture legitimately gains a face, and it was looked at rather than waved
through.** `258_o.webp` is a man holding a mask over his nose and mouth with his
eyes clear above it. OpenCV finds nothing there at 0.9; this detector finds the
face at 0.912 with the box and all five landmarks on it. The count is pinned per
fixture in the test, so a *new* divergence fails and has to be inspected instead of
being absorbed by the allowance. The other four negative fixtures gained nothing.

### 95. What a webcam frame costs, and two defects found by measuring it

Experiment 6 said the GPU-overhead backlog could not be ranked without "the
webcam measurement (53, 65)". Nothing had measured it: every number in this
backlog came from files on disk.

`examples/webcam_cost.rs` times capture, MJPEG decode, the buffer wrap and
detection over a warm loop. C920, 120 frames:

| Phase | 1920x1080 | 640x480 |
| --- | ---: | ---: |
| `webcam_grab` | 35.18 ms | 43.69 ms |
| `webcam_decode` | 4.34 | 0.76 |
| `webcam_wrap` | 0.69 | 0.12 |
| `detect_image` | 3.59 | 2.45 |
| whole frame p50 | 46.05 | 47.45 |
| achieved | 24.6 fps | 24.0 fps |

**The loop is capture-bound and nothing else.** `webcam_grab` is not work, it is
blocking until the camera produces the next frame, and it absorbs whatever slack
the rest leaves: cutting the other three stages from 8.6 ms to 3.3 changed the
frame rate by 0.6 fps. At 24 fps the pipeline is idle roughly 90% of each frame.

**That answers 53 and 65 without implementing either.** Keeping frames on the GPU
(53) and dropping stale frames under overload (65) both spend complexity to
finish work sooner, and there is no overload: the work is already done in a
seventh of the frame budget. Neither can raise the frame rate on this hardware.
They become interesting only where capture is fast enough to saturate the
pipeline -- a 60 fps camera at low resolution, or several cameras at once -- and
that is the condition to re-open them under, not fps in general.

**Defect found: `--webcam-width` and `--webcam-height` did nothing.**
`WebcamCapture::with_device_index` opened the camera with
`RequestedFormatType::AbsoluteHighestResolution` and then called
`set_resolution`, which does not take -- a C920 asked for 640x480 delivered
1920x1080. Every webcam user was decoding and detecting 6.75x the pixels they
asked for. Fixed: the open now requests `Closest` to the caller's format and
falls back to the old behaviour if that cannot be satisfied, which is the
portable fallback item 53 asks for.

That fix matters for more than the pixels, because of what 96 found while it was
being isolated: the camera's highest mode is 16:9, and 16:9 is the shape the
detector scores worst. Not the shape it fails on -- an earlier version of this
record said so, using a threshold production does not use. See 96.

### 6 (continued) / 84. Peak memory, which nothing had measured

Experiment 6 asked for peak RAM and VRAM and left both open: "that needs
process-level sampling rather than a timer around a call". 83 and 84 ask whether
the caches and pools grow without bound. Both are answerable without a profiler.

**Nothing on the GPU side grows.** `examples/memory_growth.rs` walks a corpus
largest-source-first -- so anything that only ever enlarges reaches its worst case
immediately and then plateaus visibly -- and reports the pool and the working set
as it goes. 400 images, up to 23.4 MP:

| Images | GPU pool MB | host RSS MB |
| ---: | ---: | ---: |
| 1 | 44.1 | 275.9 |
| 100 | 44.1 | 328.7 |
| 200 | 44.1 | 328.6 |
| 400 | 44.1 | 300.4 |

**Flat at 44.1 MB throughout**, and host RSS plateaus rather than climbing. The
buffer pool already has an idle ceiling (`max_idle_bytes`), the conv uniform and
bind-group caches are keyed by a graph fixed at 640x640 whatever the source is,
and the preprocessor's texture pool is bounded by `MAX_GPU_PREPROCESS_PIXELS`
(1.5 MP, so 6 MB of RGBA) because `upload_pays_for_source` declines anything
larger. **One exception, unmeasured:** that gate returns true unconditionally on
integrated and CPU adapters, so on an iGPU the texture is sized to the largest
source ever seen instead. Worth checking when such hardware is available (10).

**The batch is a different story, and the number is large.** Peak working set
over the 1239-image folder, sampled every 50 ms, `RAYON_NUM_THREADS` swept:

| Workers | Wall s | Peak RSS MB | Crops |
| ---: | ---: | ---: | ---: |
| 8 | 10.61 | 1,057 | 901 |
| 16 | 7.84 | 1,791 | 901 |
| 32 (this machine's default) | **7.36** | **3,181** | 901 |
| 64 | 8.02 | 5,433 | 901 |

**Memory scales almost linearly with workers -- about 85 MB each -- while wall
time flattens after 16.** Going 16 to 32 buys **6% speed for 78% more memory**.
The default is one worker per logical processor, so the bill is set by the core
count and nothing else: this 32-thread machine peaks at 3.2 GB, and the unpinned
default run measured 4.3 GB.

**This does not overturn experiment 60, it adds the axis 60 did not measure.**
60 compared worker counts on speed and concluded the default wins; that still
holds on speed. What it could not see is that the winning margin is 6% and the
price is 1.4 GB.

**No change made, and the 64-worker row is not evidence for a cap.** Those 64
workers ran on 32 logical processors, which is oversubscription, not what a
64-core machine would do -- so it says nothing about the default there. Whether to
cap needs a RAM budget and hardware this session does not have, and `main.rs`
already warns against capping automatically because the number moved as soon as
the work around it changed. What is now on record is the trade: the default is
tuned for throughput on a machine with memory to spare, and a user with 8 GB and
a 32-thread CPU is the case to check before shipping a cap.

### 82. One entry point per kernel, and two wrong turns on the way

Experiment 80 put `conv2d.wgsl` at 197 ms of a ~900 ms cold start: 90% of all
shader compilation and 22% of the whole start. This is what to do about it.

**The obvious answer is unavailable.** Persisting compiled pipelines is
`Features::PIPELINE_CACHE`, and `examples/adapter_cost.rs` asks the adapter for it
opportunistically: **Vulkan yes, D3D12 no**. So on the backend the app actually
ships on Windows there is nothing to cache into.

**The first measurement said splitting was worthless, and it was wrong.**
Compiling the full shader in one process and the split paths in another gave
296.7 ms against 293.7 -- no difference. That comparison is invalid: the first
pipeline built in a process carries about 180 ms of one-off FXC and D3D12
warm-up, and each run was paying it on whichever shader came first. Ordered
inside one process the picture inverts. Two runs, splits first:

| Compiled | run 1 | run 2 |
| --- | ---: | ---: |
| `only_pointwise` | 28.1 | 32.8 |
| `only_depthwise` | 25.8 | 28.1 |
| `only_general` | 63.6 | 61.1 |
| `grouped_only` | 43.8 | 41.4 |
| **`conv2d.wgsl`, all four behind one entry** | **203.2** | **208.5** |

**Compilation is superlinear in what one entry point can reach.** The three
kernels YuNet dispatches cost ~120 ms apart and ~205 ms together.

**The second wrong turn was reaching for four files.** Separate `.wgsl` files
would duplicate `write_output` and the whole header, and WGSL has no include. The
same win is available from **one module with four entry points**, because naga and
FXC only emit what each entry can reach: 25.3 + 46.9 + 52.5 ms for three entry
points of a single module against 183 ms for the branching `main` in the same
process.

So `conv2d.wgsl` gains `main_pointwise`, `main_depthwise` and `main_general`, and
`Conv2dPipeline` builds one pipeline per kernel from one module against one shared
bind group layout -- which is what keeps the bind-group cache from experiment 22
shared rather than split four ways. `main` stays as the grouped fallback and is
built **on first use**, because nothing in YuNet is a grouped convolution, so
production never compiles the expensive branching entry at all.

| | before | after |
| --- | ---: | ---: |
| `compile_conv2d` | 190-197 ms | **99-166 ms**, median ~120 |
| launch to first face | 850-1213 ms | 720-1119 ms |
| GPU compute | 0.376 ms | 0.372, 0.374 ms |
| `detect_image` wall p50 | 0.745-0.875 ms | 0.79 ms |

**Runtime is unchanged**, which was the risk worth checking: specialising the
entry points removes a branch on the uniforms but could have cost register
pressure. It did neither measurably. Bit-exact (`0xa116e42f7c2dabdb`), the
workspace suite passes under `FCS_STRICT_TESTS=1` -- which is what exercises the
lazily built grouped pipeline, since nothing else reaches it -- and 1239 real
images produce 901 byte-identical crops.

**A ten-hour hang during this experiment, which was not this experiment.** The
first validation run of the suite blocked for ten hours on 35 seconds of CPU and
had to be killed. It does not reproduce: the same binary passes 187 tests in 5.3 s
in parallel and 38 s single-threaded, before and after the change. What it points
at is real though, and pre-existing: every `device.poll` in this codebase passes
`PollType::Wait { timeout: None }`, so a submission that never completes -- a lost
or reset device, which a machine idling for hours can produce -- blocks forever
rather than failing. See 94.

### 80. Cold start, and the two things it is not

Every other measurement in this backlog is warm. Nothing had measured the path a
user actually waits for -- process launch to the first face -- so
`examples/cold_start.rs` times each stage once per process, because one process is
one cold start and averaging them would measure something else.

Five runs, RTX 4090 / D3D12, release. Launch to first face **850-1213 ms**,
median around 900. One representative run, fully split:

| Stage | ms | share |
| --- | ---: | ---: |
| adapter + device | 659.8 | **72.9%** |
| - instance creation | 21.3 | 2.4% |
| - **`request_adapter`** | **546.4** | **60.3%** |
| - `request_device` | 92.0 | 10.2% |
| preprocessor pipelines | 16.5 | 1.8% |
| detector construction | 226.9 | 25.1% |
| - ONNX parse | 0.5 | 0.1% |
| - **compile `conv2d.wgsl`** | **197.4** | **21.8%** |
| - compile the other four shaders | 19.9 | 2.2% |
| - weight upload | 0.9 | 0.1% |
| decode the first image | 0.8 | 0.1% |
| first detection | 1.6 | 0.2% |
| = launch to first face | 905.6 | |
| steady-state detection p50 | 0.74 | |

**It is not model loading.** Parsing the ONNX and uploading every weight is
**1.4 ms, 0.15%** of a cold start. `examples/benchmark_model_load.rs` measures
that number and it has never been the startup cost; two shader-and-driver stages
are 94% of it.

**It is not deferred work either.** The first detection is 1.6 ms against a steady
0.74 -- about 0.9 ms of excess on a 900 ms start. Nothing meaningful hides in the
first user action, which is the specific thing this experiment was asked to rule
out.

**Two stages own it, and neither is our code.** `request_adapter` is 546 ms of
D3D12 runtime and driver bring-up inside one wgpu call, and `conv2d.wgsl` is
197 ms of FXC. The other four shaders together are 20 ms, so **compiling them in
parallel would win almost nothing** -- it is one shader, not five.

**The Vulkan number, measured and deliberately not acted on.**
`examples/adapter_cost.rs` times adapter selection per backend set, one process
each, four runs:

| Backends | Adapter init | Selected |
| --- | ---: | --- |
| `PRIMARY` (raw) | 762-1562 ms | Vulkan |
| `DX12` | 578-668 ms | Dx12 |
| `VULKAN` | **278-312 ms** | Vulkan |

Vulkan comes up in **less than half** the time D3D12 does here. Production does
not use it, and should not: `platform_safe_backends` removes Vulkan on Windows
because Intel's ICD (`igvk64.dll` 30.0.101.x) dies with an access violation
during adapter bring-up, which crashed the GUI on two laptops during Store
certification. The comment there already says so. This measures the price of that
decision -- roughly 300 ms of every CLI start -- rather than proposing to reverse
it. Reopen only with evidence about that driver, not about speed.

**The GUI pays a different bill from the CLI**, which the numbers above hide.
`App::new` calls `share_gpu_from_eframe`, so the GUI reuses the device eframe
already built to draw its window and never issues the 546 ms `request_adapter` of
its own. Its marginal cold-start cost is the compilation: about **235 ms**, and
`build_detector` is called synchronously at `app.rs:119`, before the first frame.

So the actionable item is not making anything faster; it is that ~235 ms of
shader compilation sits on the GUI's critical path and does not need to. Moving
it off means the UI can paint while the pipelines build, which is experiment 81's
territory and a change to how the app handles a not-yet-ready detector -- not
something to write blind, and not measurable from here without running the GUI.
Recorded as the finding, left unimplemented.

### 93. Decoding cells that cannot survive - and the reasoning that nearly skipped it

`gpu_decode` was 0.077 ms, 9% of a warm small-image detection, and it decodes all
8400 cells even though `apply_postprocess` immediately discards every one below
the score threshold. The obvious fix is an early-out. The obvious objection is
that it cannot help: the decode moves a megabyte -- 126000 floats gathered
through 14 strided reads per cell, 126000 written -- and a rejected cell still
occupies a row, so the writes stay whichever way.

**That objection is wrong, and only a probe said so.**
`examples/decode_cost.rs`, over the same synthetic heads:

| Variant | Median |
| --- | ---: |
| full decode, no threshold | 0.0980 ms |
| gated at 0.6 | **0.0527 ms** |
| same gathers and writes, no transcendentals | 0.0208 ms |
| writes only | 0.0090 ms |

The traffic is 0.021 ms of it. **The other three quarters are four exponentials
and a square root per cell**, which is exactly what an early-out removes. The
back-of-envelope that said "memory-bound, not worth it" was the third time this
round that arithmetic pointed the wrong way, after 27 and 92.

**The gate is exact, not a heuristic.** `score = sqrt(s(cls) * s(obj))` with both
factors in (0, 1], so `score^2` is at most `min(s(cls), s(obj))`; sigmoid and sqrt
are monotonic, so a cell whose *smaller* logit is below `logit(threshold^2)`
cannot reach the threshold whatever the other one is. One comparison, no
transcendental. A `NaN` fails the comparison and falls through to the full path,
where the existing non-finite handling already lives.

On a real image almost every cell is far below the threshold, so the gate does far
better than the synthetic corpus above suggests:

| Phase | before | after |
| --- | ---: | ---: |
| `gpu_decode` | 0.077 ms | **0.014 ms** |
| `detect_image` wall p50 | 0.875 ms | **0.753 ms** |

**82% off the decode.**

**The parity fingerprint is deliberately untouched.** `readback_parity` covers the
*decoded* output (see the correction below), so gating it would have weakened the
one probe this whole round leaned on. Instead the existing entry points still
decode everything, and only `run_on_device_filtered` gates -- called only by the
detector, because that is the thing that knows the threshold. The fingerprint is
unchanged at `0xa116e42f7c2dabdb` and still checks the full decode arithmetic.

What guards the gate instead is the invariant that matters:
`score_gate_preserves_every_row_above_the_threshold` asserts, for both head
layouts, that no row postprocessing would have kept is changed, and that the gate
never manufactures a keepable row. Whole-tensor equality would have been the wrong
assertion -- zeroing sub-threshold rows is the point.

End to end: the workspace suite passes under `FCS_STRICT_TESTS=1`, and 1239 real
images produce 901 byte-identical crops. A folder job does not move (8.08 s
against 8.09), because it is bound by JPEG decode rather than head decode; this
lands on interactive and webcam work like the rest of the round.

### Correction: what `readback_parity` actually fingerprints

Several records in this round call its 126000 floats "the raw head fingerprint".
They are not the raw heads. `run_on_device` returns the **decoded** fused tensor,
so the fingerprint covers the readback *and* `decode_yunet_outputs_with` --
8400 cells by 15 columns, which is where the 126000 comes from. The probe's own
header says "raw inference output", which is what misled the wording.

This makes every bit-exactness claim in this round *stronger* than stated, not
weaker: 20, 34, 37 and 22 were all checked against the decoded output rather than
the buffers behind it. It also means any future experiment that changes what
decode writes -- compacting sub-threshold rows, for instance -- changes this
fingerprint by design, and cannot be waved through as a readback regression.

### 5 (continued). The last unattributed quarter of the small-image path

Reprofiling after 20, 34 and 37 left `detect_on_device` at 1.060 ms with
`onnx_inference` at 0.842 -- **0.218 ms, a quarter of the detection, with no
label on it.** Experiment 5 split the inference side into eleven phases and never
touched the on-device preprocessor, which has no guards at all; only the
large-image `resize_then_convert` path does.

Seven guards later the path adds up. Warm, 0.17 MP, 30 runs:

| Phase | p50 ms |
| --- | ---: |
| detect_image | 0.841 |
| - allocate_input | 0.000 |
| - gpu_preprocess | 0.162 |
| - - preprocess_acquire | 0.000 |
| - - to_rgba8 | 0.025 |
| - - write_texture | 0.057 |
| - - encode | 0.073 |
| - - - **submit** | **0.055** |
| - onnx_inference | 0.692 |
| - - gpu_encode | 0.172 |
| - - - record | 0.073 |
| - - - **submit** | **0.093** |
| - - gpu_readback | 0.446 |
| - - - wait (contains GPU execution) | 0.354 |
| - - gpu_decode | 0.072 |
| - postprocess | 0.005 |

Two things worth acting on, neither of them the shader:

- **A `queue.submit` costs about 0.05 ms before it costs anything per command.**
  The preprocess submit carries one dispatch and costs 0.055; the inference submit
  carries 43 and costs 0.093. So the fixed part dominates, and the two submits
  together are **18% of a detection**. They are on the same queue in a fixed
  order, so there is no reason for them to be two command buffers.
- **`allocate_input` and `preprocess_acquire` are free**, which retires the idea
  that pooled allocation is a per-detection cost on this path.

`to_rgba8` at 0.025 ms is a full copy of the source per frame, and `write_texture`
at 0.057 ms is 689 KB at roughly 12 GB/s. Both scale with source size, so they
are the webcam-resolution numbers, not a constant.

### 54. The routing cutoff was set against a route that no longer exists

`upload_pays_for_source` sends sources above 1.5 MP down the CPU resize path and
everything below it up to the GPU whole-source path. That number was measured
before experiment 50, when the losing side was "resize on the CPU *and* convert
on the CPU *and* upload 4.9 MB of floats". 50 replaced that with
`resize_then_convert`, which uploads 1.2 MB of bytes -- so the opponent got
cheaper and nobody re-ran the comparison.

**The measurement needs both routes at one source size in one process.** Across
processes `cpu_resize` alone moved 0.83 to 2.23 ms on this machine depending on
what else was running, an order of magnitude more than the difference being
measured. `FCS_MAX_GPU_PREPROCESS_PIXELS` overrides the constant, so
`phase_timings --mp N --ab FCS_MAX_GPU_PREPROCESS_PIXELS=...` alternates the two
routes in 8 blocks each on one image, and `--mp N` rescales one fixture so a
size sweep needs no corpus.

`detect_image` p50 delta, **positive means the GPU whole-source route is slower**:

| Source | delta ms | Source | delta ms |
| --- | ---: | --- | ---: |
| 0.50 MP | -1.370 | 1.75 MP | -0.010, -0.090 |
| 0.80 MP | -0.460 | 1.85 MP | +0.050, +0.000 |
| 1.20 MP | -0.230 | 1.90 MP | +0.130, +0.130 |
| 1.50 MP | -0.130 | 2.00 MP | +0.110, +0.160, +0.060 |
| 1.55 MP | -0.290, -0.250 | 2.50 MP | +0.480, +0.470 |
| 1.65 MP | -0.110, +0.020 | 3.00 MP | +1.010, +0.820, +0.870 |

Three A/A controls, because the override is a no-op wherever both variants
already choose the same route: +0.070 at 0.80 MP and +0.010 at 1.50 MP (forcing
GPU where GPU is already chosen), -0.040 at 2.00 MP (forcing CPU where CPU is
already chosen). **The noise floor on this delta is about +/-0.07 ms**, so
everything from 1.65 to 1.85 MP is a tie and the ends are not.

**The crossover is 1.75-1.85 MP, and the constant is now 1_750_000.** Sources
between 1.5 and 1.75 MP get 0.25-0.29 ms back; nothing else moves.

Two things the phase table says that the wall clock does not. `gpu_preprocess`
is *cheaper* than `cpu_resize + gpu_rgb_to_chw` at every size up to 1.85 MP
(0.99 against 1.19 at 1.85 MP), yet `detect_image` is a tie there -- the GPU
route's full-resolution texture upload costs something outside its own guard,
and only the wall clock sees it. And `cpu_resize` is not monotonic in source
size: 1.13 ms at 0.50 MP against 0.98 ms at 2.00 MP, because a 530x943 source is
*upscaled* to 640 wide. The cutoff is a proxy for a ratio, not for a size.

**Under CPU contention the crossover moves up.** One block of runs taken while
the machine was otherwise busy (`cpu_resize` at 1.73-2.23 ms rather than
1.07-1.29) had the GPU route winning by 0.47-0.61 ms at 1.6-1.9 MP. The resize
is threaded, so a batch job with 32 rayon workers is the contended case; raising
the cutoff is the robust direction as well as the measured one. Routing overhead
itself is `image.dimensions()`, one multiply and one compare.

**The second half -- a cheaper preview resize -- is already a shipped setting,
and it costs faces.** `ResizeQuality::Speed` is `Nearest`. Through 51's harness
(`resize_quality nearest`, 120 fixtures, 49 matched faces):

| | Speed / Nearest |
| --- | ---: |
| faces lost / gained | **2** / 0 |
| landmark shift p50 / p95 / max | 1.87 / 10.68 / **33.55 px** |
| box IoU min | 0.9276 |
| detect_image over the corpus | **1.76x faster** |

That is the failure `Interpolation` was rejected for in 51, at the same
magnitude (35.39 px there, 33.55 px here), and it is reachable from a menu. The
default is `Quality` and stays there; the measurement is now recorded on the
enum variant so the trade is visible where the setting is. No adaptive *quality*
routing was built: switching filters by source size would make detections depend
on image size, which is a worse property than being slow.

**Moving the boundary moves detections, because the two routes were never
identical.** This is a routing change, so it had to go through the same harness
as any other quality candidate. `resize_quality` now takes the variable name and
a `--mp` rescale, so it can put the whole corpus in the band the change actually
affects. 120 fixtures rescaled to 1.6 MP, old cutoff against new:

| At | faces lost / gained | landmark p50 / p95 / max | IoU min |
| --- | ---: | ---: | ---: |
| 1.6 MP, 1.5 MP cutoff vs 1.75 | 0 / 0 | 0.65 / 1.84 / **11.09 px** | 0.9798 |
| 1.4 MP, both routes forced | 0 / **1** | 0.66 / 1.80 / 3.04 px | 0.9807 |
| A/A control (same cutoff twice) | 0 / 0 | 0.00 / 0.00 / **0.00 px** | 1.0000 |

The second row is the point. At 1.4 MP -- a size the *old* cutoff already sent to
the GPU route -- forcing the CPU route instead disagrees by the same p50 and p95,
and gains a face. **The seam is a standing property of shipping two routes, not
something this change introduced**; all it does is move where the seam sits by
0.25 MP. Neither route is a reference: one resizes with `fast_image_resize`
Bilinear on the CPU, the other with the adaptive kernel in `preprocess.wgsl`, and
nothing here says which is closer to the truth.

**Kept, with the trade stated:** sources between 1.5 and 1.75 MP are 0.25-0.29 ms
faster (0.47-0.61 ms when the CPU is busy) and get detections that differ from
their old ones by 0.65 px at p50 and 11 px at worst on one fixture. If that seam
matters more than the milliseconds, it is one constant to put back.

Also kept as evaluation scaffolding, since experiment 10 needs all of it on other
hardware: the `FCS_MAX_GPU_PREPROCESS_PIXELS` override, `phase_timings --mp`,
`resize_quality`'s variable-name and `--mp` arguments, and `nearest` as an
`FCS_RESIZE_ALG` candidate.

### 57. Compacting the heads cannot win more than the whole download costs

The head readback moves 525 KB per detection and the decode discards almost all
of it, so compacting survivors on the GPU first is the obvious next move. In
production the copies hide inside `readback_wait`, which also contains the
forward pass, so the phase table cannot price them. `examples/readback_bytes.rs`
runs the same allocate / copy / map / wait / collect sequence `batch_download`
runs, with no inference in flight, over buffers of decreasing size -- 200 runs
after 10 warm, medians in ms:

| Case | KB | alloc | copy | map | wait | collect | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| production, 3 heads, 8400 cells | 525.0 | 0.024 | 0.061 | 0.001 | 0.086 | 0.036 | **0.208** |
| compacted to 2048 cells | 128.0 | 0.010 | 0.042 | 0.000 | 0.058 | 0.006 | **0.116** |
| compacted to 512 cells | 32.0 | 0.011 | 0.045 | 0.000 | 0.059 | 0.002 | 0.117 |
| compacted to 128 cells | 8.0 | 0.010 | 0.046 | 0.000 | 0.051 | 0.001 | 0.109 |
| compacted to 16 cells | 1.0 | 0.013 | 0.055 | 0.001 | 0.053 | 0.001 | 0.123 |

**The whole ceiling is 0.09 ms, and it is reached at 2048 cells.** From 128 KB
down to 1 KB -- 128x fewer bytes -- nothing changes at all. A download costs
about 0.11 ms before it costs a single byte, so this is not a bandwidth problem,
and compacting tighter than a few thousand rows buys exactly nothing.

**Most of that 0.09 ms is in the window experiments 12, 13 and 14 established is
free.** `alloc` (-0.014) and `copy` (-0.019) happen after the inference submit,
while the GPU is busy, and the wait absorbs them; 13's conclusion was precisely
that. What is left on the critical path is the DMA inside the wait (0.086 to
0.058, so **~0.028 ms**) and `readback_collect` (0.036 to 0.006, **~0.030 ms**),
plus the `gpu_convert` split that compaction would also remove (0.021 ms at
0.8 MP, 0.051 at 10 MP). **Production ceiling: 0.08-0.11 ms** of a 1.3 ms
small-image detection.

What that would have to pay for:

- **Ordering.** An `atomicAdd` append returns rows in nondeterministic order, so
  equal-score NMS ties stop being reproducible -- against a standard of 901
  byte-identical crops over 1239 images. Preserving order needs a prefix scan
  and a scatter: two more dispatches over 8400 cells, not one gate.
- **Overflow.** A fixed capacity needs a full-heads fallback download when it is
  exceeded, which is a second round trip -- five times the ceiling, and it lands
  on the crowded scenes this was meant to help.
- Every compacted row must carry its cell index, since the prior for a box comes
  from level, x and y.
- `readback_parity` fingerprints the decoded output (see the correction above),
  so the probe every result in this round leaned on has to be re-established in
  the same change.

**Rejected.** And the two largest components -- `readback_collect`'s copy out of
the mapped range, and `gpu_convert`'s split into twelve tensors -- are host work
on 525 KB *after* the wait, which needs no GPU pass at all to attack. That is
experiment 17, which already names "decoding from a mapped view". It is a
hypothesis and not a saving: reads from a mapped readback allocation can be far
slower per access than reads from a `Vec`, which is why the copy is there.

### 17. The second copy, and why the first one stays

`readback_collect` copied each level's fused head buffer out of the mapped range
into a `Vec`, and `gpu_convert` then cut the four branches out of that `Vec`
afterwards. 57 handed this over as the last part of the download nothing had
attacked: host work on 525 KB, after the wait, needing no GPU pass to remove.
The branch ranges are contiguous and disjoint, so the second copy only existed
because the first one had happened -- taking each branch straight off the mapped
view does the whole job in one pass.

In-process A/B, `phase_timings --ab FCS_COPY_THEN_SPLIT=1`, 8 blocks of 15 runs
per variant, alternated. Both variants now split inside `readback_collect`, so
that one label holds the entire comparison and `gpu_convert` no longer exists:

| Source | collect, two passes | collect, one pass | delta | `detect_image` block spread |
| --- | ---: | ---: | ---: | --- |
| 0.8 MP | 0.031 | 0.020 | **-0.011** | 0.912-0.954 ms |
| 10.11 MP | 0.053 | 0.032 | **-0.021** | 1.996-2.198 ms |

The wall delta is inside that spread at both sizes, which is what a 0.011-0.021 ms
change on a 0.93-2.10 ms path has to be; the phase is the measurement. The size
dependence is the cache rather than the work -- the same 525 KB is copied either
way, but after a 10 MP resize has walked the caches the surviving copy costs
more, so removing the other one is worth more.

Bit-identical: `readback_parity` returns `0xa116e42f7c2dabdb` over five runs
under both variants, and the whole 1239-image folder was run through the CLI
both ways -- **1129 faces, 959 crops, 959 of 959 byte-identical**, which is the
first record of the post-96 counts and the standard 71 established. **Kept**,
and `split_off` and the comment explaining why it had to peel backwards go with
it. Folder wall time is unchanged and cannot show this: 0.02 ms per detection
over 1129 detections is 0.02 s of a 7.1-7.4 s job.

**Not copying at all loses, which is the other half of the question.** 57 raised
it and called it a hypothesis rather than a saving: reads out of a mapped
allocation can be far slower per access than reads out of a `Vec`. Measured
directly by `readback_bytes --reads`, over the three production head buffers,
200 runs after 10 warm, in-place read taken first so the copy cannot be what
warms the cache for it. The access pattern is the one 93 left behind: both score
channels for every cell, the other fourteen for one cell in six.

| Over 525 KB of production heads | ms |
| --- | ---: |
| copy out of the mapping | 0.009 |
| decode-shaped read of the copy | 0.020 |
| the same read, in the mapping | **0.031** |

Reading the mapped allocation is ~55% more expensive per access, and that
penalty (0.011) is larger than the copy it would remove (0.009): 0.029 for
copy-then-read against 0.031 in place. **The remaining copy is load-bearing and
stays.** What 17 removed was the one that was redundant, and the ceiling 57
priced at 0.05-0.08 ms was never all available: about a third of it was.

### 94. The deadline was the easy half; the hang was not what it looked like

Every blocking wait in the pipeline passed `PollType::Wait { timeout: None }`,
so a submission that never completes parks the calling thread forever with
nothing to report. Observed once during 82's validation -- a test binary sat for
ten hours on 35 seconds of CPU and had to be killed -- and not reproducible on
demand, which is what made it an experiment rather than a fix.

**The deadline, which is the part that was straightforward.** Five copies of the
same twelve lines: the head readback in `gpu/runtime.rs`, `read_buffer` in
`gpu/tensor.rs`, the preprocessing readback in `preprocess.rs`, the timestamp
resolve in `gpu/profiler.rs`, and the `gpu_readback!` macro that every image
operation in `fcs-utils` expands. All five now call one
`fcs_utils::gpu::wait_for_gpu`, and there is exactly one `device.poll` left in
the production tree.

Thirty seconds. Every wait here covers a single submission -- a forward pass, a
preprocessing dispatch, a filter, a timestamp resolve -- and those are
single-digit milliseconds on the slowest hardware the app targets; Windows
already resets a GPU that stops responding for two seconds. Three orders of
magnitude above the largest real wait measured anywhere in this file. The error
names the operation and the deadline, because a missed deadline is the one poll
failure that means "still running" rather than "broken". A wait that expires
leaves the buffer unmapped -- its map callback never fired -- and every site
propagates before `get_mapped_range`; `preprocess.rs` recycles its pooled
buffers only on the success path, so a buffer with an outstanding map request
is dropped rather than handed to the next caller.

**Then the hang reproduced, and it was not a poll.** `cargo test --workspace
--release` wedged for 25 minutes on 4.75 seconds of CPU. Attaching `cdb` to the
`fcs_utils` test binary and walking all 81 threads:

```text
1  Id: ... "enhance::gpu::tests::each_enhancement_changes_the_image..."
   ntdll!RtlpEnterCriticalSectionContended
   nvwgf2umx!NVAPI_DirectMethods+0x621e5
   D3D12Core!CCommandList<...>::VersionedResetCommandList
3  Id: ... "enhance::gpu::tests::negative_exposure_is_active_too"
   win32u!NtGdiDdDDICreateAllocation
   D3D12Core!CResource::FinalConstruct
39 Id: ... "gpu::background_blur::tests::blend_clamps_the_mask_size..."
   win32u!NtGdiDdDDIDestroyAllocation2
```

**Not one thread was in a wait of ours.** Every stuck test was inside the NVIDIA
usermode driver, creating or destroying D3D12 resources, several of them parked
on the driver's own critical section. `wait_for_gpu`'s deadline would never have
fired, because nothing had reached a wait.

The cause is in the test harness, not the pipeline. `test_context()` built a
fresh `GpuContext` -- a fresh D3D12 device -- per test, and six modules had their
own copy of it despite `test_support` existing. `cargo test` runs 32 threads, so
a run opened dozens of devices within a few milliseconds of each other, and the
driver occasionally did not survive it. **One cached context per test binary**,
with the six duplicates deleted:

| | Suite wall time | Hangs |
| --- | ---: | ---: |
| A device per test | 3.8-4.2 s | 1 in the 3 runs after the first sighting; 0 in 15 on master |
| One shared device | **1.5 s** | **0 in 70 consecutive runs** |

The before-rate is not something these numbers pin down -- the hang is rare and
its trigger is timing, so a 15-run block of zeroes proves very little, and 2 of
15 on a candidate that cannot reach the driver's allocator proves less. What is
established is the mechanism, from the stacks, and that 70 consecutive runs
under one device produced nothing. The 2.6x on suite wall time needs no
statistics: it is one `request_adapter` instead of forty, and 80 measured that
at 546 ms.

**Kept, both parts, for different reasons.** The deadline because a wait that
cannot fail is wrong regardless of what caused this particular hang, and the
shared context because it is faster, is less code, and removes the pile-up the
stacks actually blamed. **94's original premise is wrong and is recorded as
wrong:** the ten-hour hang was almost certainly this, and a poll timeout would
not have caught it.

Validated with the full workspace suite under `FCS_STRICT_TESTS=1` (0 failures)
and a 1239-image folder job, which is where 71's lesson says a change to a
shared helper belongs: 959 of 959 crops byte-identical against master.

### Previous work

The compute-pass merge is already shipped in the baseline: roughly 0.40 ms saved in paired
encode/finish/submit/wait measurements, with all 61 profiling records retained.
