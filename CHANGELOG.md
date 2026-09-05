# Changelog

All notable changes to Face Crop Studio are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **GPU compute passes can now be timed on the GPU's own clock**, via wgpu
  timestamp queries. Set `GpuContextOptions::profiling`; `TIMESTAMP_QUERY` is
  then requested opportunistically, so an adapter without it still builds a
  context and simply records nothing. Off by default, allocating nothing.

  Wall-clock benchmarks around a dispatch cannot resolve a shader change here.
  Running an *identical* bench binary twice against the same criterion baseline
  reported "improvements" of 19–47% at p<0.05, because the measurement is
  dominated by upload, readback and GPU clock ramping rather than by the
  shader. Anything smaller than that noise — which is most shader work — was
  invisible, and several plausible-sounding optimisations turned out on
  careful measurement to be regressions.

  All 16 compute passes across both crates are instrumented. Resolving happens
  inside `GpuProfiler::take` with its own encoder rather than in each caller's,
  since timestamps live in the query set until read — which is what keeps
  instrumenting a pass to a single line at the `ComputePassDescriptor` and
  leaves every submit path untouched.

  `examples/gpu_pass_breakdown.rs` prints the per-pass cost of one YuNet
  forward pass. It reports 0.911 ms of GPU compute across 61 passes, 94.6% of
  it convolution — against 2.95 ms wall for the same call, so roughly 70% of
  even the inference call is encode, submit, sync and readback that no shader
  change can reach. `examples/preprocess_cost.rs` does the same for the
  preprocessing paths. The `inference_pipeline` bench also gains
  `gpu_on_device` and `gpu_quality` cases; it previously could not measure the
  fully on-device path at all, because `new_gpu` always pairs GPU inference
  with the CPU preprocessor.

- **Windows releases now ship ONNX Runtime**, so the fastest backend is the one
  users actually get: inference drops from ~15-20 ms on the built-in graph to
  ~7 ms. `onnxruntime.dll` (20 MB, pinned to 1.24.4) and its licence go into the
  dist directory beside the executables, which is where `fcs_ort::locate` looks
  first; the NSIS and WiX installers both harvest that directory, so they pick it
  up without further change.

  A packaging step that quietly does nothing is the failure mode worth guarding,
  because every way this can break — a missing DLL, one placed where `fcs-ort`
  does not look, a version below `REQUIRED_API_VERSION`, an unmet dependency of
  the DLL itself — degrades silently to the built-in graph and ships roughly 3x
  slower with no error. So the workflow runs the *packaged* `fcs-cli.exe` and
  fails the build unless it reports `Detection backend: onnxruntime`.

  Verified locally against a simulated dist layout, including from an unrelated
  working directory: resolution is relative to the executable, not the cwd, and
  the bundled copy is preferred over an older `onnxruntime.dll` on `PATH`.

  Linux and macOS now bundle it too. The AppImage and the `.app` both place the
  library beside the executable — the first place `fcs-ort` looks — rather than
  in a lib directory, which would depend on the loader's search path and could
  be shadowed by a distro-provided onnxruntime. Each platform gets the same
  run-the-packaged-artifact check as Windows, exercising `fcs-ort`'s loader
  against the exact library that was packaged.

  Two platform details were not optional. On macOS a bundled dylib is nested
  code, so it has to be signed before the bundle or `codesign --verify --deep
  --strict` rejects the result; it is signed without entitlements, which apply
  to executables rather than libraries. And for the `.deb`, cargo-deb asset
  lists are static and it fails on a missing file, so bundling unconditionally
  would have forced every local `cargo deb` to download the library first —
  hence a `bundled-ort` variant that `build_linux.sh` selects only when the
  library is present, restating the package name so the artifact is otherwise
  identical.

  `FCS_ORT_LIB` is optional in both installer scripts: unset, they build exactly
  as before and the package falls back to the built-in graph, so a local build
  still needs no network. Only arm64 is published for macOS at 1.24.4, which is
  all that job targets anyway.

### Removed

- **The GPU batch-norm op is gone** — shader, pipeline, the two
  `GpuInferenceOps` methods and its test, 251 lines. BatchNorm is folded into
  the exported weights, so `gpu/graph.rs` never encoded one; nothing outside
  the test suite had called it since.

- **`tract-onnx` is no longer in the shipped binaries.** It stopped being an
  inference backend when the built-in graph landed — it was 4x slower and only
  ever reached as a fallback — but it was still linked into every release, and
  it was large: `fcs-cli` drops from 56.6 MB to **30.2 MB** and `fcs-gui` from
  62.6 MB to **36.3 MB**. That more than pays for the ONNX Runtime library now
  shipped alongside: the total Windows download falls from 119 MB to 87 MB even
  after adding 20 MB of DLL.

  Removing it meant replacing two things it was quietly providing.

  The **tensor type** flowing between preprocessing, inference and decoding was
  `tract_onnx::prelude::Tensor`, which brought arbitrary dtypes, quantisation,
  views and lazy shapes to a pipeline that only ever moves densely packed f32
  through four shapes. `fcs_core::tensor::Tensor` replaces it in about 100
  lines. One test disappeared with it — the "output is not f32" case can no
  longer be constructed, since the type is f32 by definition now.

  The **ONNX protobuf schema** was `tract_onnx::pb`. Reading initializers needs
  three messages and five fields, so `yunet::proto` declares exactly those with
  `prost`, which is already a dependency. Protobuf skips fields it does not
  know, so a subset parses a complete ONNX file. The field tags are the
  load-bearing detail — a wrong tag reads the wrong field rather than failing —
  so they were taken from tract's own generated code and are pinned by a
  round-trip test, with `cpu_parity` reading the real model as the check that
  they match ONNX itself.

  `tract` stays as a **dev-dependency**, which is the part worth keeping. It is
  the only implementation available that interprets the ONNX graph instead of
  re-encoding YuNet's topology by hand, so it remains the oracle both shipped
  backends are checked against — they could otherwise share a mistake about the
  architecture and agree with each other perfectly. `tests/common/mod.rs` runs
  it directly now that `InferenceBackend::Tract` is gone, and `gpu/tests.rs`
  still stops it at a named node to validate individual GPU ops. Nothing it
  provides reaches a released binary.

  The remaining consequence is deliberate: with no general ONNX interpreter
  bundled, a model whose initializers do not match YuNet's topology is now a
  hard error rather than a silent fallback to a slower backend that would have
  coped.

### Fixed

- **GPU preprocessing uploaded the source image at full resolution, which cost
  far more than the transfer it was introduced to avoid.** `detect_on_device`
  fuses preprocessing and inference onto one device so the 640x640 tensor never
  makes a 4.9 MB round trip through host memory. But fusing means the *source*
  goes up instead, and a 10 MP photo is 40.4 MB of RGBA — plus a
  full-resolution `to_rgba8` on the CPU first. End to end on an RTX 4090 the
  fully on-device path measured **19.1 ms against 11.1 ms** for the same
  inference with a CPU resize.

  The shape of it only became visible with GPU timestamps: the preprocess
  shader runs in **0.04 ms** while the path around it takes **6.5 ms**, against
  0.6 ms to resize on the CPU and upload the tensor. The dispatch was never the
  cost; getting the image to it was.

  `WgpuPreprocessor` now declines sources above 1.5 MP and defers to the CPU
  preprocessor, which brings the on-device path to **9.6 ms** — now the fastest
  configuration rather than the slowest. The threshold sits inside a measured
  1.1–2.5 MP crossover and is a tuning constant, not a law: PCIe bandwidth,
  CPU resize speed and decode all move it, so `examples/preprocess_cost.rs`
  prints both sides and finds the crossover on the hardware at hand. Integrated
  GPUs skip the check, since there the upload is a copy inside memory the CPU
  already owns and the penalty does not exist.

  The check lives in the preprocessor rather than in `detect_on_device`, and
  that placement is the whole fix. Declining in the detector sends the caller
  to `WgpuPreprocessor::preprocess` — the *unfused* GPU path, which uploads the
  source whole **and** rounds the result back through host memory. Guarding
  there made the same benchmark slower still, at 23.0 ms. Both entry points
  have to defer for either to help.

- The CLI's JSON snapshot test pinned floats to `1e-5`, which was tighter than
  the difference between backends: it passed on the built-in graph and failed on
  ONNX Runtime, so whether the suite was green depended on what happened to be
  installed. Now that releases ship ONNX Runtime, this would have started
  failing in CI. Widened to `1e-3` — roughly five times the measured
  cross-backend spread, and still far tighter than any real regression, which
  moves boxes by whole pixels rather than by the last digit.

- Loading a file that is not a YuNet export reported only that it "does not
  match YuNet's topology", which could not distinguish a corrupt file from a
  valid ONNX graph of some other model. The error now names the file and keeps
  the underlying decode failure in its chain.

- The NSIS uninstaller removed files by pattern (`*.exe`, `*.ico`, `*.md`,
  `LICENSE-*`) with no `*.dll` among them, so bundling ONNX Runtime would have
  orphaned 20 MB in the install directory on every uninstall.

### Changed

- **YuNet's topology moved out of `gpu` into `fcs-core::yunet`**, along with the
  ONNX initializer reader and the macros that build the block tables. None of it
  was ever GPU-specific — `gpu/graph.rs` hand-encodes the architecture and reads
  only weights from the model file — but it lived under `gpu` because that was
  the only backend at the time. With the pure-Rust graph added, `cpu/graph.rs`
  was reaching across into `gpu` for the network definition, which reads as a
  dependency between backends rather than what it is: both describing the same
  network. `gpu/graph.rs` drops from 487 lines to 338, keeping only the WGSL
  encoding. A pure move; no behaviour changed, and the parity suites cover it.

- Every backend now logs itself at the same level. `tract` announced itself at
  `debug` while the other two used `info`, so the quietest backend was also the
  slowest one and the least likely to be noticed. `YuNetDetector::
  inference_backend()` exposes the winner, and both front ends report it once
  after selection — which backend runs depends on what is installed and what the
  GPU offered, so the settings alone do not say.

### Added

- **An ONNX Runtime CPU inference backend, selected automatically when its
  library is present.** 1.5.3 found that 59-61% of a CPU detection sits in
  tract's `depth_wise::inner_loop_generic`, because tract only unrolls depthwise
  zones with at most 4 taps and YuNet's 3x3 kernels have 9, so every zone takes
  the scalar fallback — and concluded it was "not reachable from this side".
  That held within tract. Swapping runtimes reaches it: ONNX Runtime vectorises
  those convolutions and runs the same graph in 6.7 ms against 65.9 ms.

  Measured per stage on `fixtures/images/006.jpg`, mimalloc enabled as shipped:
  `detect_image` 69.6 ms → 10.7 ms, end-to-end including JPEG decode
  86.9 ms → 31.3 ms. Batch is a different story — 20 images through rayon go
  236 ms → 118 ms, only 2.0x, because batch already parallelised across images
  and becomes memory bound once inference stops being the constraint. The
  single-image figure is the one users feel; the batch figure is the one to
  quote for folder runs.

  This is aimed at machines with no usable GPU. Where one exists the WGSL graph
  already does the whole detection in 8.2 ms, and DirectML was measured and
  deliberately excluded: at ~9.8 ms end to end it merely ties that, for a second
  17.7 MB library. Only the CPU execution provider is used, so one 20.1 MB
  library ships and there is no per-platform provider matrix. (An earlier note
  in `ONNX_RUNTIME_OPTIONS.md` put the cost at ~160 MB; that is the full
  multi-provider package, not what a CPU-only build needs.)

  `YuNetModel::load` prefers ONNX Runtime when a compatible library is found and
  falls back to tract otherwise, matching how GPU acceleration is already
  selected by availability rather than configuration. `load_with` forces a
  specific backend, and `backend_name` reports which is live.

  Two hazards are handled explicitly, both discovered the hard way:

  - **A bad runtime aborts the process rather than returning an error.** `ort`
    has no fallible initialisation: it calls `.expect()` inside a `#[cold]`
    non-unwinding function, the failure poisons a global mutex, and the process
    aborts somewhere `catch_unwind` cannot reach. `dylib_available()` therefore
    duplicates `ort`'s own path resolution *and* its minor-version check before
    any `ort` API is touched. This is not theoretical — a machine with an
    unrelated `onnxruntime.dll` 1.17 on PATH resolved the bare library name to
    that copy, and a probe that only checked the file loaded and exported
    `OrtGetApiBase` passed it straight through to an abort at startup.
  - **`Session::run` takes `&mut self`.** A single shared session behind a lock
    would serialise every detection, which loses outright to tract on batch runs
    — tract is slower per image but spreads across every core. `OrtPool` hands
    out one session per concurrent caller, created on demand so a single-image
    run pays for exactly one graph optimisation.

  `tests/backend_parity.rs` compares final detections, not raw head tensors,
  across 20 fixtures: decode and NMS sit downstream of the runtime and can turn
  a small numeric difference into a different face. Face counts match exactly,
  with scores inside 1e-3 and boxes inside 5 px — the same budget
  `gpu_cpu_parity` allows.

  `ort` is pinned at `=2.0.0-rc.13`, the only non-stable dependency in the
  workspace, and is loaded dynamically rather than linked because the prebuilt
  static library is built against the dynamic CRT and collides with
  `+crt-static`.

- **A pure-Rust CPU inference graph (`fcs-core::cpu`), needing neither an
  external runtime nor a GPU.** Measured at **16 ms** against tract's 66-76 ms
  on the same fixture — roughly 4.5x — with no explicit SIMD anywhere. ONNX Runtime remains ahead at 6-9 ms.

  This is far less code than "write an ONNX runtime" because the model was
  already re-implemented once: `gpu/graph.rs` is not an ONNX interpreter but
  YuNet's topology hand-encoded in Rust, with only the weights read from the
  file. The CPU backend runs that same topology, so the op set is four
  operators — convolution (dense, depthwise or grouped, optionally fusing
  ReLU), 2x2 max pooling, elementwise add, nearest 2x upsample — plus sigmoid
  and a CHW->HWC reorder on the way out. BatchNorm needs no implementation at
  all; it is already folded into the exported weights.

  Convolution takes three paths because YuNet only ever asks for three shapes:
  a pointwise 1x1 where most of the arithmetic lives, a depthwise KxK (the case
  tract lowers to a scalar loop), and a general fallback used once for the
  strided stem. The depthwise path splits interior pixels from the padded frame
  so the common case carries no bounds checks.

  Correctness is the whole risk here — a transposed weight layout, an
  off-by-one pad or a missing ReLU all still produce plausible numbers — so it
  is checked at two levels. Each operator is compared against a naive reference
  written straight from the definition, over shapes chosen to exercise borders
  (a 1x1 image is all border; a 17x5 one is mostly interior), groups, strides
  and batches. Then `tests/cpu_parity.rs` compares final detections against
  tract across 20 real fixtures, because decode and NMS sit downstream and can
  turn a small numeric difference into a different face. Both passed on the
  first run.

  It is wired into `YuNetModel` as `InferenceBackend::CpuGraph`, and `Auto` now
  selects ONNX Runtime, then this, then tract. tract stays as the last resort
  because it is the only backend that interprets an arbitrary ONNX graph: the
  built-in one knows YuNet's topology and nothing else, so a model whose
  initializers do not match falls through to it. It is also the reference the
  other two are checked against, which is why `backend_parity` now compares
  every available backend to tract rather than just ONNX Runtime. With no
  runtime installed, `detect_image` goes from 69.6 ms to 23.2 ms.

  A second optimisation pass took 32 ms to 16 ms, again decided by measurement:

  - **Parallelism, not arithmetic, was the limit.** All three convolution paths
    split work by output channel, so the stem got 16 tasks on a 32-thread
    machine — and the layers with fewest channels are exactly the ones running
    at the largest spatial sizes. Splitting by (channel, row-block) instead cut
    the stem from 4.1 ms to 2.9 ms and the whole model from 19.2 ms to 16.0 ms.
    The block size always divides the output height, so a chunk never straddles
    two channel planes.
  - **Zero-initialisation was measured and dismissed.** Allocating every
    activation buffer costs 0.33 ms of a 16 ms run — 2% — so the `unsafe`
    needed to skip it buys nothing.
  - **Removing the backbone's feature-map clones changed nothing measurable**,
    which the zeroing figure had already predicted. Kept anyway, since it is
    less code.

  One finding is left deliberately unacted on. Single-image inference is 35%
  faster on 16 rayon threads than on 32 (14.3 ms against 22.0 ms) — the machine
  has 16 physical cores, and SMT siblings contend for the same cache and
  execution units on memory-bound work. Batch export wants the opposite, being
  7% faster on 32 (249 ms against 267 ms for 20 images), because parallelism
  across images already saturates the machine. Changing the global rayon pool
  would trade one for the other and would also affect decode, enhancement and
  export, so it is recorded rather than applied.

  What remains is memory traffic in the pointwise path: it reads its whole input
  once per output channel, which is 105 MB for a single 64->64 layer at 80x80.
  Blocking output channels to cut that is the textbook fix and measured slower
  both times it was tried — the second time because blocking and row-splitting
  compete for the same tasks, and with at most 64 channels against 32 threads
  the row split is worth more. Closing the remaining gap to ONNX Runtime needs
  a loop restructure that keeps input rows resident across all output channels,
  which cannot be expressed with `par_chunks_mut` over a channel-major buffer.

  Getting from a first working version (32 ms) to 18.5 ms was decided entirely
  by measurement, and the measurement disagreed with every guess:

  - The **stem** turned out to be the single most expensive layer in the
    network at 6.7 ms — 26% of all convolution time in one layer — despite
    being the one written "for clarity rather than speed" on the assumption it
    did not matter. It runs at the full 640x640 input, which the small channel
    counts hide. Splitting interior rows from the padded edge took it to 3.8 ms.
  - **Depthwise** went from 11.1 ms to 5.3 ms by accumulating a whole output row
    per kernel tap (`out[x] += k * in[x + shift]` over a contiguous span)
    instead of gathering nine taps per pixel. The span form vectorises and also
    removes bounds checking, since the valid range of `x` follows from the shift
    rather than being tested per pixel.
  - **Pointwise blocking was tried and reverted.** Handling four output channels
    per pass to cut input re-reads is the textbook fix, and it measured *slower*
    — 18.5 ms against 17.4 ms. The simple contiguous accumulate already streams
    predictably enough for the prefetcher, and blocking only added
    `split_at_mut` bookkeeping and register pressure. The reasoning sits next to
    the code so it is not re-attempted.

  One real bug came out of that attempt: chunking the output buffer by block
  size let a chunk straddle two batch items whenever the output channel count
  was not a multiple of the block, pairing one image's outputs with another's
  inputs. YuNet always runs with a batch of one, so nothing would have caught it
  in practice; `pointwise_blocks_do_not_straddle_the_batch_boundary` does.

- **`fcs-ort`, a new workspace crate that will take over from the `ort` crate a
  piece at a time.** It starts by owning the whole of ONNX Runtime discovery and
  validation, which `fcs-core` no longer does itself, and `fcs-core` keeps using
  `ort` for sessions and tensors.

  Discovery was the right first piece because it was already the part with no
  `ort` in it — just `libloading` — and the part where getting it wrong aborts
  the process. Moving it made the crate load-bearing immediately rather than
  scaffolding waiting for a second commit.

  The `ort` crate is now gone from the workspace: `fcs-ort` owns the whole
  binding — environment, session options, session, tensors in and out, and
  error handling — over 29 C API functions. Inference speed is unchanged
  (6.3 ms against `ort`'s 6.7 ms, batch 128 ms against 132 ms, both inside this
  machine's noise), but three things improved.

  **The session pool and its lock are gone.** `ort::Session::run` takes
  `&mut self`, which forced a lock, which would have serialised every detection
  — hence the pool built to work around it. ONNX Runtime itself documents
  concurrent `Run` on one session as safe, so the binding takes `&self` and
  batch export shares a single session with no pool, no lock and no
  session-per-thread memory. `concurrent_runs_match_sequential` checks that
  against eight threads rather than trusting the documentation.

  **The `REQUIRED_API_VERSION` / `api-NN` lockstep is gone**, along with the
  release-candidate dependency. Nothing enforced that coupling, and getting it
  wrong meant accepting a library that `ort` then aborted on.

  **A process-wide environment.** ONNX Runtime expects one `OrtEnv` per process;
  `Environment::shared` caches it, including the negative result, so a machine
  with no runtime does not repeat a failed library search on every model load.

  Growing the C API surface is safe incrementally because `OrtApi` is
  append-only: new ONNX Runtime releases add function pointers at the end and
  never reorder existing ones, which is what lets one binary serve every
  `ORT_API_VERSION`. So a declaration covering only the leading fields is
  ABI-correct, and `fcs-ort` can add one function at a time instead of vendoring
  1100 lines of bindings up front. `sys.rs` documents the two rules that come
  with that: declare every field from the start of the struct in header order
  including ones you do not call (a skipped field silently shifts every later
  offset, which is undefined behaviour rather than a compile error), and take
  signatures from the oldest supported release.

  That append-only property is what made the whole thing tractable: everything
  Face Crop Studio needs lives in the first 101 of 424 entries, so 323 fields
  are omitted outright and only 29 of the declared 101 need real signatures.

  The offset hazard is not hypothetical. A first pass at generating the prefix
  from `ort-sys`'s bindings counted `#[cfg]`-duplicated fields twice — it
  declares `CreateSession` once for `wasm32` and once for native — which put the
  struct at 427 fields instead of 424 and shifted three offsets in the region we
  call. That would have compiled and then called the wrong function pointers.
  Two guards now exist: a `const` assertion that the declared struct is exactly
  one pointer per field, which catches a wrong or duplicated *type*, and
  `fcs-ort/tests/end_to_end.rs`, which calls through offsets 3 to 100 against a
  real runtime, which catches a wrong *position*.

  Two fixes came out of the move. The FFI declarations now use `extern "system"`
  rather than `extern "C"`, matching the header's `ORT_API_CALL` — identical on
  x86_64 but wrong on 32-bit Windows. And validation now calls
  `GetApi(REQUIRED_API_VERSION)` and checks for null, the authoritative ABI test
  that the version-string comparison only approximates; a corrupt or mismatched
  build can report a plausible version and still fail to serve the API. The
  library is also held open for the lifetime of the `Runtime`, so the copy that
  was validated is the copy that stays mapped rather than being unloaded and
  re-resolved later against a possibly different file.

- `examples/stage_breakdown.rs`, which prints the per-stage cost of one
  detection plus batch throughput. Written after a detour spent optimising a
  stage that turned out to be 3% of the pipeline.

### Changed

- `docs/PERFORMANCE.md` claimed preprocessing was 33% of the detection pipeline
  and inference 52%. Measured, preprocessing is 3% (2.9 ms) and inference 95% of
  `detect_image`. The wrong figure had already sent one optimisation hunt at a
  stage costing 2.9 ms. The stage table now carries measured per-stage numbers
  for all three backends and names the command that produces them.

- `docs/PERFORMANCE.md` presented INT8 quantisation and adopting the `ort` crate
  as the same decision, pointing both "Future Opportunities" rows at
  `ONNX_RUNTIME_OPTIONS.md` — which never mentions INT8 at all. They are
  independent: `ort` swaps the inference runtime and adds a ~160 MB shared
  library, while INT8 changes only the model file and needs no runtime change,
  since `tract-onnx` already registers `QuantizeLinear`, `DequantizeLinear`,
  `QLinearConv` and `QLinearMatMul` and `tract-linalg` ships x86_64 i8 GEMM
  kernels selected by runtime CPUID. A new section says so, and records what
  would actually decide the idea: whether a quantised export still survives
  `into_optimized()` given this model already needed a fixed-shape re-export,
  whether the i8 GEMM kernels are even on the critical path for a
  depthwise-separable backbone, and what post-training quantisation costs in
  recall. It also notes that gains depend heavily on the CPU — AVX512-VNNI gives
  a 4-way i8 dot per lane, whereas the shipped `x86-64-v3` baseline falls back to
  computing i8 products in i16 lanes, the same lane count as f32 FMA — so
  benchmarking only on a VNNI developer machine overstates what users get.

- The `2–4×` and `2–5×` gain estimates on those two rows were never measured in
  this repository. Both now read "Unmeasured", with a note defining the column,
  so the table is not read as a set of predictions.

- Recorded the `wide` crate under "What Did Not Work". It was removed in
  `616ffa0` because the plain scalar loop benched 18% faster, but the Phase 11
  table still listed "Saturation `wide::f32x4` SIMD" as a shipped optimisation
  with the revert buried in a later row, so the obvious next question — "where
  could SIMD help?" — led straight back to an experiment already run and lost.
  The new entry gives both reasons it lost and why they still hold: the build
  sets `target-cpu=x86-64-v3`, so LLVM autovectorises those loops at AVX2 width,
  and every remaining hot loop is either an interleaved-pixel deinterleave (RGBA
  saturation, RGB→BGR CHW) or a table lookup (tone LUTs, histogram equalisation,
  the bilateral colour LUT), which need `pshufb`/`vgather` — operations `wide`
  does not expose. The Phase 11 rows are merged into one honest entry.

- `docs/ONNX_RUNTIME_OPTIONS.md` gained a scope note saying it covers runtime
  replacement only, so the cross-reference now works in both directions.

- `parquet` 59.2.0 -> 59.3.0.

## [1.6.0] - 2026-08-30

### Changed

- Removed the four `#[allow(clippy::too_many_arguments)]` suppressions rather
  than carrying them. Each hid a parameter list that had a grouping already
  implied by the code: `encode_stage_block` took the four fields of a
  `StageBlock` that every caller was destructuring at the call site and now
  takes the block; the conv2d pipeline's `input`/`weights`/`bias` became
  `Conv2dTensors`, which `execute` benefits from too; the golden-crop test
  helper's `img_w`/`img_h` became one `[u32; 2]`, matching how that file
  already spells pairs. In `fcs-cli`, `process_single_image` and
  `process_crops` each took eleven arguments, six of which were the same per-run
  constants threaded through both; those are now a `BatchContext` built once
  before the parallel loop, taking both functions to six parameters. No
  behaviour change, and the workspace is slightly smaller for it.

### Fixed

- **`fcs-cli` ignored `gpu.preprocessing` entirely.** The setting exists, the
  GUI honours it, and the CLI had no parameter for it at all — it built a
  `WgpuPreprocessor` whenever an adapter was present, whatever the config said.
  `build_cli_detector` now takes the whole `GpuSettings` instead of a single
  `prefer_gpu_inference` flag and derives both preferences in one place, so the
  two front ends cannot drift apart on one flag again. With
  `preprocessing: false` the CLI now reports "Using CPU preprocessing + GPU
  inference" and takes that path.


- **Large batches filled VRAM and then failed, because pooled buffers were
  filed under the wrong size.** `GpuBufferPool::acquire` satisfies a request
  from any idle buffer at least as large, so a 4 MB buffer is routinely handed
  out for a 256 KB request — but `recycle` filed it back under the size the
  caller *asked for*, not the size it actually is. A large buffer relabelled as
  small then stopped matching large requests, so every recurrence of the larger
  size allocated a fresh buffer while the mislabelled one sat idle forever. On a
  folder of mixed image sizes that ratchets upward without limit: a probe
  cycling ten sizes retained 126 MB, then 160 MB, then 194 MB over three
  identical rounds. `recycle` now takes the size from the buffer itself.

  Nothing bounded the growth either. Every pipeline in `fcs-utils` builds its
  pool with `max_memory: None`, which meant the `clear` path that releases idle
  buffers was unreachable, so the pool only ever grew. Such pools now retain at
  most `DEFAULT_MAX_IDLE_BYTES` (512 MiB) of idle buffers, evicting
  smallest-first — smallest because a large buffer can serve a small request but
  never the reverse, so the large ones are the ones worth keeping.

  Pools built *with* a `max_memory` budget are exempt: `acquire` already
  releases idle buffers when that budget is reached, and adding a ceiling on top
  made inference evict at exactly the threshold it works at, freeing buffers the
  next call immediately re-allocated. That showed up as `fcs-core`'s GPU tests
  going from five seconds to minutes. `with_idle_limit` sets the ceiling
  explicitly where a test needs a reachable one.

  Over 968 images the pool now plateaus and stays there rather than climbing.

- **A batch with fewer crops than the one before it failed outright.**
  `gpu_readback!` mapped the entire readback buffer (`slice(..)`) and then
  checked the result length against the expected output. Since the pool returns
  any buffer at least as large as the request, a batch sized for three faces
  left a buffer that a later one-face batch reused — and the check compared one
  face of output against three faces of capacity, failing with "unexpected GPU
  batch crop output size (expected 4096, got 12288)" on a buffer that was
  perfectly valid. No memory pressure was needed; an image with one face after
  an image with three was enough. The macro now maps only the region the
  operation wrote.

  This is the same defect 1.5.4 fixed in `preprocess.rs`; the shared macro was
  missed, so it still affected batch cropping, both blurs, pixel adjust,
  red-eye, shape masking and histogram equalisation — every operation that reads
  results back.

- **Batch detection on the GPU silently lost crops, and sometimes invented
  faces.** Pooled GPU buffers were recycled by `Drop` on the host, which happens
  while a command encoder is still being built — before anything is submitted.
  `GpuYuNet::run` holds the workspace mutex only to take and return the input
  tensor; `run_inference` runs unlocked, and the whole forward pass is
  accumulated into a single command buffer that is submitted at the end. So an
  intermediate released during encoding went straight back to the shared pool,
  and a second thread encoding its own pass could acquire a buffer the first
  pass already referenced. Both submissions then wrote the same memory.

  Both front ends hit this: `fcs-cli` runs batch detection through `par_iter`
  and `fcs-gui`'s export through `into_par_iter`, each over one shared detector,
  and GPU acceleration is on by default. Over the same 20 images the CPU path
  reported a stable 15 faces detected and 15 crops saved on every run, while the
  GPU path gave 7, 11 and 20 detected and 7, 5 and 7 saved — dropping real faces
  and, in the 20-face run, producing detections that are not there. Nothing was
  logged: no warning, no `batch_failures.json`, and the summary line reported the
  reduced count as though it were the answer. Forcing `RAYON_NUM_THREADS=1`
  restored 15/15 exactly, which is what identified concurrency as the cause.

  `GpuBufferPool` now has execution scopes. Buffers released inside a scope are
  parked until it ends rather than returned to the shared pool, so a buffer whose
  work is still in flight cannot reach another thread; `run_inference` holds one
  across its whole encode/submit/readback cycle, and the readback is what
  establishes that the work has finished. Reuse *within* a scope is deliberately
  kept — the release and the reuse land in the same command buffer in program
  order, so the GPU runs them in that order, and without it peak memory would
  grow by one allocation per layer. Five consecutive runs now report 15/15,
  matching the CPU path.

  `concurrent_inference_matches_sequential` covers it: four threads running the
  same input through one detector, compared against the sequential result. It
  fails without the scope and passes with it.

  This also accounts for the intermittent failure in
  `fcs-cli`'s `test_batch_processing_with_multiple_images`, which copies one
  fixture three times and so ran three detections in parallel through a shared
  detector — the corrupted runs produced no output files and tripped its
  assertion. It failed roughly one run in three before, and passed 23
  consecutive runs after. CI never saw it because the nextest profile retries
  once.

- **GPU preprocessing aliased badly on any large downscale, which moved real
  detections.** `preprocess.wgsl` resampled with a single
  `textureSampleLevel(..., 0.0)` — four bilinear taps at mip 0, and the source
  texture is created with `mip_level_count: 1`, so there was nothing else to
  sample. The tap count did not depend on the scale ratio, so downscaling a
  3840x5760 group shot to 640x640 read 4 source pixels out of every ~54. That
  is not a precision difference; the network is handed a different, aliased
  image. Measured against the CPU preprocessor on the same fixtures: a landmark
  moved 23.1 px and a detection score changed by 0.0054, with the error scaling
  exactly with the downscale factor (~54 source pixels per output pixel gave
  23.1 px, ~25 gave 6.5 px, ~9 and ~2 were clean).

  This was never limited to the opt-in GPU inference path. `fcs-cli` selects
  `WgpuPreprocessor` whenever an adapter is available *even when inference stays
  on tract*, so every user with a working GPU has been detecting against an
  aliased image on large photos.

  The shader now derives its tap count from the source/destination ratio and
  averages the box, halving the count because each bilinear tap already spans
  about two texels, and capping it at 16 per axis so the quadratic cost stays
  bounded. At one tap the sample position reduces to `(id + 0.5) / dst_size`,
  algebraically identical to the previous line, so magnification and the
  no-resize case are unchanged — the exact-parity assertions in
  `preprocess.rs` still hold. `gpu_cpu_parity` now passes on all six fixtures
  with its tolerances untouched: score delta 0.0006 (limit 1e-3), landmark
  1.87 px and bbox 4.36 px (limit 5.0). Cost is below this machine's run-to-run
  noise; GPU preprocessing is dominated by upload and readback, not sampling.

  The partial shape of this was already recorded in `preprocess.rs` as a
  parity-test caveat ("differs by up to 51 of 255 per channel" on minification).
  What was missing was the connection to detection output.

- **`gpu_cpu_parity` had been silently skipping, so it validated nothing.** It
  resolved the model with a bare `Path::new("models/...")`, relative to the
  current directory — which under `cargo test -p fcs-core` is the crate
  directory, not the workspace root. It printed a skip notice to stderr and
  passed in 0.00 s. The fixture paths had the same defect. This is the same
  class of bug 1.4.3 fixed for the OpenCV parity tests; this file was missed,
  and it is why the aliasing above went unnoticed. Model resolution is now
  `fcs_utils::model_path`, which searches `YUNET_MODEL_PATH` and then the
  manifest's ancestors, and errors instead of skipping under
  `FCS_STRICT_TESTS`. It replaces three separate hand-rolled copies of the
  same search — in the benchmark, in `gpu/tests.rs`, and the broken one here.

### Changed
- **Detection can now run end to end on the GPU, without a round trip through
  host memory.** Preprocessing and inference each used to build their own
  `wgpu::Device` — a default CLI run initialised the adapter twice — so the
  preprocessed tensor had to be downloaded through a blocking map and uploaded
  straight back before inference could touch it, 4.9 MB each way.

  `GpuYuNet::with_context` now takes an existing device, and `YuNetDetector`
  hands it the preprocessor's, so the two stages share one. The preprocess
  shader already writes exactly the layout a tensor wants (f32, CHW, BGR), so
  `WgpuPreprocessor::preprocess_into_tensor` points it straight at an inference
  tensor's buffer: nothing is copied, and nothing is read back. Ordering comes
  free from the queue, which executes submissions in order, so the two stages
  need no host synchronisation at all. `YuNetDetector::detect_on_device` picks
  this path whenever GPU inference and a GPU preprocessor share a device, and
  returns to the previous route otherwise — CPU inference, a CPU preprocessor,
  mismatched devices, or an image too large for one texture.

  Measured per detection over 20 fixtures, single-threaded: **5.9 ms median,
  against 64.6 ms on the CPU path** — about 11x. Note this is per-image latency,
  not batch throughput: GPU work serialises on one queue, so a whole-folder run
  on a many-core machine is still faster on the CPU, where rayon spreads
  detections across cores (482 ms vs 1301 ms for 20 images here). Restricted to
  one rayon thread the ordering reverses, 1288 ms GPU against 1620 ms CPU. The
  win is therefore real for interactive single-image work and for machines with
  few cores, and the CLI's batch default should still be reconsidered
  separately.

- Removed a redundant row-padding pass from GPU preprocessing. Every row of the
  source was copied into an aligned staging buffer before upload — a second full
  pass over the image, ~14 ms on a 2384x4240 source — to satisfy
  `COPY_BYTES_PER_ROW_ALIGNMENT`. That rule does not apply here:
  wgpu validates `Queue::write_texture` with alignment checks disabled, and
  requires the alignment only for `copy_buffer_to_texture` and
  `copy_texture_to_buffer`. Rows now go up tightly packed. This also retires the
  staging buffer, its high-water-mark shrink logic, and a local `align_to`
  helper duplicating the one in `model.rs`.




- Dependency bumps: `tract-onnx` 0.23.4 → 0.23.5, `libheif-rs` 2.7.0 → 3.0.0,
  `wgpu`/`naga` 30.0.0 → 30.0.1, `imagepipe` 0.5.0 → 0.5.1, `imgref` 1.12.2 →
  1.12.3, `log` 0.4.33 → 0.4.34, `crc32fast` 1.5.0 → 1.5.1, `lru` 0.18.2 →
  0.18.3. `libheif-rs` 3.0.0 is a major release but needed no code changes; the
  `HeifContext`/`LibHeif`/`ColorSpace`/`RgbChroma` surface `load_heic` uses is
  unchanged.
- The `inference_pipeline` benchmark gained a `gpu` case, running `new_gpu`
  against the same `CpuPreprocessor` and resize quality as the existing `speed`
  case so the pair isolates the inference backend. On this hardware the WGSL
  graph runs a 640x640 detection in 8.9 ms against tract's 114 ms — and with
  preprocessing held constant it now reproduces tract's detections exactly
  (0.000000 score delta, 0.000 px landmark delta), having agreed to ~1e-6
  relative on every head output at both 0..1 and 0..255 input magnitudes. It
  skips when no adapter is present, as CI has none.
- The `tract` bump does not move the detection hotspot, which was worth
  confirming rather than assuming: `tract-core`'s `depth_wise.rs` is
  byte-identical between 0.23.4 and 0.23.5, and profiling both binaries back to
  back on `inference_pipeline/detect_image/speed` puts
  `depth_wise::inner_loop_generic` at 60.8% and 61.8% of self time
  respectively — the same scalar fallback described in 1.5.3, still with no
  x86 SIMD variant upstream. What did change is matmul kernel selection:
  0.23.4 ran everything through a single `avx512_mmm_f32_80x2`, while 0.23.5
  picks among `32x6`, `48x4` and `16x12` from the new AVX/FMA kernel set. That
  is a ~9% slice of the run either way.

  Total CPU per run is too noisy on this machine to rank the two versions —
  the same binary measured 15.7 s and 23.6 s over consecutive 25 s windows —
  so only the distribution above is claimed here, not a speedup.

## [1.5.4] - 2026-08-12

### Fixed

- `load_jpeg_exif` underflowed on a malformed segment length. A JPEG segment's
  length field counts its own two bytes, so `length - 2` wrapped for a crafted
  file declaring 0 or 1 — a panic in debug builds and a bogus offset in release
  ones. Found by `cargo mutants`: the surviving mutant pointed straight at an
  unguarded subtraction on data read from an arbitrary user-supplied file.

- The GPU preprocessor mapped the whole pooled readback buffer instead of the
  region the dispatch wrote. `ensure_output_buffers` only ever grows its pooled
  buffers, so once a larger tensor had been processed every smaller one failed
  with "unexpected GPU output size" reporting the pooled capacity rather than
  the requested length. Reachable by lowering the detection input size while the
  preprocessor is reused. `gpu/runtime.rs` already sliced explicitly; this brings
  `preprocess.rs` in line.

### Changed

- Test coverage aimed at the gaps a full mutation run exposed (3579 mutants,
  606 survivors, 81.8% caught):
  - The GPU operation tests were smoke tests. They built inputs with
    `RgbaImage::from_pixel` — a single flat colour — and asserted only that the
    call did not error, so an operation could be replaced wholesale by
    `Ok(Default::default())` and the suite stayed green. A flat image also hides
    every indexing and dispatch bug. New `gpu/test_support.rs` supplies a
    gradient fixture where each pixel differs, plus assertions on dimensions,
    buffer length and non-blankness, and it replaces the eight copies of
    `test_context()`. `pixel_adjust` and `hist_equalize` had no test that ran
    their operation at all and now do, including odd image sizes that are not
    workgroup multiples so a mis-rounded dispatch leaves a detectable tail.
  - EXIF parser tests for truncated and malformed input: every prefix of a PNG
    signature, partial chunk headers, undersized JPEG segment lengths, and a
    chunk ending exactly at EOF. These parse user-selected files, so the bounds
    logic is a trust boundary rather than a coverage statistic.
  - `color.rs`: achromatic input must not divide by a zero delta (the mutant
    turned grey pixels into a NaN hue), and the red-max hue branch must divide
    by delta rather than scale by it.

  Roughly 20 of the `color.rs` survivors turned out to be *equivalent* mutants
  rather than gaps — at every 60-degree boundary `x` is exactly `0` or `c`, so
  adjacent match arms return identical tuples, and `nib << 4` and `nib` occupy
  disjoint bits, so `|` and `^` agree. No test can distinguish those, which is
  worth knowing before treating a survivor count as a to-do list.

## [1.5.3] - 2026-08-10

### Fixed

- The GUI crashed at launch on Intel integrated graphics, which failed
  Microsoft Store certification on two separate laptops. The crash is inside
  Intel's Vulkan driver during wgpu's adapter bring-up, before any window
  appears:

  ```text
  Faulting application name: fcs-gui.exe, version: 1.5.2.0
  Faulting module name: igvk64.dll, version: 30.0.101.1960
  Exception code: 0xc0000005
  ```

  eframe defaults to `Backends::PRIMARY | GL`, and `GpuContextOptions` defaulted
  to `Backends::PRIMARY`; both include Vulkan, so wgpu was free to select the
  Intel ICD. Windows builds now filter Vulkan out of the backend set via
  `platform_safe_backends`, leaving DX12 (with GL as eframe's fallback). The
  fault is in the driver, so avoiding the code path is the only fix available
  from this side — an access violation in native driver code cannot be caught.

  This was never Store-specific: the MSI and ZIP builds crash identically on the
  same hardware, so it affected every Windows user with that driver, not only
  Store installs. `WGPU_BACKEND=vulkan` still forces Vulkan for debugging, and
  non-Windows platforms are untouched, since Vulkan is the correct backend on
  Linux. `fcs-cli` shared the same default and is fixed by the same change.

### Changed

- Single-image detection got roughly 4x faster in Quality mode and 1.5x in
  Speed mode: 305 ms → 74 ms and 106 ms → 69 ms respectively, on the
  `inference_pipeline` benchmark (640x640 YuNet, x86-64-v3). Two independent
  causes, both found by profiling with samply:
  - `resize_image` took the `fast_image_resize` path only for
    `FilterType::Nearest`, so `ResizeQuality::Quality` — which asks for
    `Triangle` — fell through to `image::imageops::resize`. Its per-pixel
    `GenericImageView` sampling was 62% of a quality detection, which is why
    choosing Quality cost three times as much as Speed rather than a little
    more. All five `image` filters now map onto the equivalent SIMD kernel
    (`Triangle` → `Bilinear`, and so on); output is the same up to rounding.
    Quality and Speed now differ by about 5 ms, which is what the filter choice
    should have cost all along.
  - `mimalloc` is the global allocator in `fcs-cli` and `fcs-gui`. tract
    allocates and frees one intermediate tensor per graph node per inference,
    and the Windows system heap decommits blocks that size on free — so every
    run page-faults the same memory back in. 26% of a detection was kernel
    time, split between the page-fault handler and `RtlFreeHeap`; it is now 3%.
    Worth 35% on its own, and it should matter more in batch export, where
    several detections contend for the heap at once.
- What remains is 59% in tract's `depth_wise::inner_loop_generic`, and it is
  not reachable from this side: tract only
  unrolls depthwise zones with at most 4 taps, and YuNet's 3x3 kernels have 9,
  so every zone takes the scalar fallback. tract-linalg's `multithread-mm`
  feature would not help either — it covers matmul, now 9% of the run, and
  would compete with the image-level rayon parallelism batch export already
  uses.
- Dependency bumps: `thiserror` 2.0.19 → 2.0.20, `rusqlite` 0.40.1 → 0.40.2.

## [1.5.2] - 2026-08-07

### Changed

- `parquet` 59.1.0 → 59.2.0.

### Added

- Linux arm64 release artifacts (AppImage and `.deb`), built natively on
  GitHub's `ubuntu-22.04-arm` runner (free for public repositories) — no
  cross-compilation, and the existing from-source dav1d/libde265/libheif action
  is reused unchanged. `windows-release` and `linux-release` became arch
  matrices with `fail-fast: false`, so one architecture failing does not discard
  a good build for the other on a tag that is already public.
- A matching `Linux ARM64` leg in `ci.yml`. Release-only coverage would repeat
  the gap that the macOS and Linux CI legs closed in 1.5.0: an arm64 break would
  not surface until a tag was cut.
- Windows arm64 was implemented and then withdrawn before release: it builds
  everything up to `cargo check` and then fails in `tract-linalg` 0.23.4, the
  current release, which has no `aarch64-pc-windows-msvc` support. Its build
  script special-cases x86_64+Windows to assemble with `ml64.exe`, but the
  aarch64 branch calls `cc::Build` unconditionally, so the GNU-syntax `.S`
  kernels reach `cl.exe`, which ignores them (`D9024`/`D9027`) and leaves
  `lib.exe` to fail with `LNK1181`. No feature or env var disables the assembly.
  The matrix entry is kept commented in `release.yml` so restoring it is a
  matter of uncommenting once upstream lands support. Windows-on-Arm runs the
  x86_64 build under emulation meanwhile.
- Microsoft Store packaging. `installer/windows/msix/` holds an `AppxManifest`
  template, the tile assets, and `build_msix.ps1`, which wraps each
  architecture's dist directory in a `.msix`, bundles them into a
  `.msixbundle`, and zips that into the `.msixupload` that `msstore publish`
  accepts. It packages whichever architectures the release matrix produced, so
  the bundle gains arm64 automatically if that leg is ever restored. Packaged as a full-trust desktop app, so file access is unchanged
  from the MSI/ZIP builds and batch mode does not need the
  `broadFileSystemAccess` restricted capability. The manifest declares the
  `webcam` device capability, without which the packaged build would enumerate
  zero cameras instead of failing visibly.
- Automated Store submission in a `store-submit` job, via
  `microsoft/microsoft-store-apppublisher`. It stays skipped until the
  `MSSTORE_PRODUCT_ID` repository variable is set, so nothing changes for
  anyone who has not done the Partner Center setup. The prerequisites — the
  first submission must be made by hand, free products only — are written up in
  `docs/release_runbook.md`.
- `.github/actions/verify-models`, replacing five copies of the same 40-line
  Python checksum block across the two workflows. Rewritten in
  `sha256sum --check`, because the new arm64 runners are not guaranteed a
  Python the old block could rely on.

### Fixed

- Partner Center rejected the first Store package: shipping the CLI as a second
  `<Application>` with its app-list entry suppressed — intended to keep a
  console tool out of the Start menu — is classified as a "headless app" and
  needs the `HeadlessAppBypass` waiver. An execution alias does not have to
  point at its own application's executable, so the alias now sits inside the
  GUI application and targets `fcs-cli.exe`: one Start menu entry, no waiver,
  and `fcs-cli` still on PATH for Store installs.
- The Windows build could not link from a clean vcpkg tree. `vcpkg install
  libheif` takes the port's default features, whose only member is `hevc` —
  HEVC *encoding* via x265 — and x265's `threadpool.cpp` calls Win32 registry
  APIs while nothing in the graph links `Advapi32`, so the link died with
  `LNK2019: unresolved external symbol __imp_RegQueryValueExA`. It is now
  `libheif[core]`, which takes no default features. HEIC decoding is unaffected:
  the decoder, libde265, is a base dependency of the port, and the app only ever
  decodes HEIC. This matches the Linux build, which already used
  `-DWITH_LIBDE265=ON -DWITH_X265=OFF`. The failure was latent rather than new —
  CI had been restoring a vcpkg cache that predated the port gaining x265, so it
  would have appeared on the next cache miss whenever that came.
- Release jobs all uploaded a checksum file named `SHA256SUMS.txt` to the same
  GitHub release, so whichever job finished last silently overwrote the others
  and most assets shipped unverifiable. They are now
  `SHA256SUMS-<platform>-<arch>.txt`.
- The shipped x86_64 binaries were not actually built with `x86-64-v3`, despite
  the README saying so. Setting `RUSTFLAGS` in the environment makes cargo
  ignore `[target.*.rustflags]` in `.cargo/config.toml` entirely — the two are
  mutually exclusive — and both the Windows release job and the Linux
  `linux-build-deps` action set it, for `+crt-static` and `-l dylib=stdc++`
  respectively. `target-cpu` is now restated wherever `RUSTFLAGS` is set, and
  the Linux action picks it per-architecture from `uname -m`.
- The vcpkg and native-library caches were keyed on `runner.os`, which is
  `Windows`/`Linux` for both architectures. An arm64 job would have restored
  x86_64 static libraries and failed at link time; the keys now include the
  vcpkg triplet and `runner.arch`.

## [1.5.1] - 2026-08-06

### Changed

- Dependency bumps: `clap` 4.6.4 → 4.6.5, `eframe`/`egui`/`egui_extras`/
  `egui_kittest` 0.35 → 0.36, `wgpu`/`naga` 29.0.3 → 30.0.0 (the version
  `egui-wgpu` 0.36 requires), `base64` 0.23.0 → 0.23.1, `lru` 0.18.1 → 0.18.2,
  plus the transitive updates `cargo update` pulled with them.
- Three breaking changes came with those bumps, all mechanical:
  - `BufferSlice::get_mapped_range` now returns `Result`. The four call sites
    (the `gpu_readback!` macro plus `preprocess.rs`, `gpu/runtime.rs`,
    `gpu/tensor.rs` in `fcs-core`) propagate the error rather than unwrapping,
    so a failed map surfaces as a normal GPU error instead of a panic.
  - `RequestAdapterOptions` gained `apply_limit_buckets`. Left at its `false`
    default: bucketing rounds adapter limits down to anti-fingerprinting
    presets, which only matters when wgpu is exposed to untrusted content.
  - egui 0.36 turned `DroppedFile` into a trait whose `path()` returns `&Path`
    rather than `Option<PathBuf>`. Drag-and-drop in `fcs-gui` no longer needs
    the "dropped file without a path" branch, which was unreachable on native
    anyway.

### Fixed

- `fcs-gui/build.rs` tripped `clippy::needless_return` on macOS and Linux. Its
  early `return;` was followed by a `#[cfg(windows)]` block, so on a Windows host
  the return is not trailing and the lint stays silent — but everywhere else the
  `cfg` strips the block and the return becomes the last statement. Both build
  scripts are now a single guarded `if` with no `return`. Caught by the macOS and
  Linux clippy legs added in 1.5.0, on their first run.
- `fcs-cli/build.rs` guarded only on `cfg(windows)` (the host) and not on
  `CARGO_CFG_TARGET_OS`, so a Windows host cross-compiling to Linux would have
  tried to embed Windows resources. It now matches `fcs-gui/build.rs`.

## [1.5.0] - 2026-07-31

### Added

- macOS and Linux legs in CI. Previously only Windows was built and tested on
  push, while `release.yml` built all three — so a macOS or Linux break was not
  discovered until a tag was cut. `ci.yml` is now a matrix over
  `windows-latest`, `macos-latest`, and `ubuntu-22.04` (matching the release
  job's glibc baseline). Clippy runs on every leg, because the `cfg(target_os)`
  blocks in `fcs-utils` and `fcs-gui` are only linted on the platform that
  compiles them; `fmt` and the coverage gate run once. Model generation moved
  into its own `models` job that publishes an artifact, so `onnxsim` runs once
  per workflow instead of once per platform.
- `.github/actions/linux-build-deps`, a composite action holding the
  from-source dav1d/libde265/libheif build that `ci.yml` and `release.yml` both
  need. It was ~60 lines inline in `release.yml`; adding a second caller would
  have duplicated it. Results are now cached under `/usr/local`, keyed on the
  three pinned library versions.
- Tests for `fcs-gui/src/interaction/bbox_drag.rs`, which had none: handle hit
  testing (including corner-over-move precedence and the handle's reach past
  the rect edge) and every `apply_drag` branch, covering the minimum-extent
  clamp that stops a corner inverting the box and the image-bounds clamp. The
  file went from 0% to 100% line coverage, and `cargo mutants` on it now reports
  42 caught, 0 missed, 2 unviable.
- More `egui_kittest` coverage, via a shared `ui/test_support.rs` harness
  extracted from the existing `widgets.rs` tests: the five toolbar button
  helpers (notably that a disabled `icon_btn` runs neither its click nor its
  action, and that `danger_btn` fires its action exactly once per click) and
  `menubar`'s `menu_item` popup routing.
- Non-kittest tests for the GUI's pure helpers: the `shape_variant` /
  `default_for_variant` round trip across all eleven crop-shape variants, the
  agreement between `variant_label` and the dropdown's `ALL_VARIANTS` list,
  the polygon corner/chamfer limits, `metadata_mode_label`, `local_time_str`,
  and `process_ram_mb`.
- Workspace line coverage rose from 62.2% to 68.9%; the CI floor moved from 60
  to 65.

### Changed

- Manifest hygiene across the workspace:
  - `rust-version = "1.96"` in `[workspace.package]`, inherited by all five
    members, matching the toolchain both workflows pin. An older toolchain now
    reports the MSRV instead of failing with a confusing edition-2024 error.
  - `resolver = "3"`, which makes dependency resolution MSRV-aware now that
    `rust-version` exists. (The previous `resolver = "2"` was not a mistake:
    virtual manifests default to resolver 1 regardless of edition.)
  - `[workspace.lints]` with `clippy::all` and `unsafe_op_in_unsafe_fn`, opted
    into by each member via `lints.workspace = true`. Lint policy previously
    lived only in `clippy -- -D warnings` in CI, so a local `cargo clippy`
    disagreed with the pipeline. `missing_docs` was considered and left off: it
    reports 610 items, which under `-D warnings` is a hard failure.
  - `winresource` moved to `[target.'cfg(windows)'.build-dependencies]` in
    `fcs-cli` and `fcs-gui`. Both build scripts only call it behind a Windows
    guard, but it was being compiled on the macOS and Linux release legs.
  - Dropped the redundant `[lib] path = "src/lib.rs"` from four manifests and
    the redundant `[[bin]]` from `fcs-gui`; both are cargo's autodetected
    defaults.
- `imageproc` is now `default-features = false, features = ["rayon"]`. Its own
  `default` feature contains `"image/default"`, which was silently re-enabling
  every image codec and defeating the curated `image` feature list in
  `[workspace.dependencies]` — `image` was resolving with `dds`, `exr`, `ff`,
  `gif`, `hdr`, `pnm`, `qoi` and `tga` on top of the nine wanted ones. None of
  those formats appear in `SUPPORTED_IMAGE_EXTENSIONS`. Only
  `geometric_transformations` and `rect` are used from `imageproc`, so `text`
  (ab_glyph) and `fft` (rustdct) went as well. 513 -> 507 crates in the
  workspace tree.
- `statusbar.rs` now uses the `windows` crate's typed bindings instead of
  hand-declaring `GetLocalTime`, `GetCurrentProcess`, `K32GetProcessMemoryInfo`
  and a `#[repr(C)] struct Pmc`. The crate was already a dependency, so the FFI
  was reimplementing bindings that were being paid for and not used — and the
  hand-rolled `PROCESS_MEMORY_COUNTERS` clone had to stay byte-compatible with
  the Windows SDK by hand. Net 37 lines removed. `Win32_System_Time` was wrong
  (`GetLocalTime` lives in `Win32_System_SystemInformation`) and
  `Win32_Graphics_Dxgi_Common` was unused; the feature list is now one entry per
  module actually imported from, each commented with what it provides.
- **The tabular mapping subsystem is now its own crate, `fcs-mapping`.** CSV,
  Excel, Parquet, and SQLite ingestion lived in `fcs-utils` behind a `mapping`
  feature, which meant every consumer of `fcs-utils` carried `calamine`,
  `parquet`, `rusqlite` and `csv` in its dependency graph resolution. The code
  had no `crate::` references outside its own module tree, so the move was
  mechanical. `fcs-cli` and `fcs-gui` now depend on `fcs-mapping` directly and
  the `mapping` feature is gone; the 43 mapping tests moved with it.
- `cargo-mutants` no longer excludes all of `fcs-gui`. The exclusion now names
  the painting modules (`ui/`, `core/`, `rendering/`, and the crate-root
  files); `fcs-gui/src/interaction/` is pure geometry with no egui frame
  involved and is worth mutating.

### Fixed

- `avif` was missing from `SUPPORTED_IMAGE_EXTENSIONS`. AVIF *output* was fully
  wired (`ImageFormatHint::Avif`, `encode_avif` via ravif), the `image`
  dependency has had `avif`/`avif-native` enabled throughout, and dav1d is
  installed on all three CI and release platforms specifically for it — but the
  extension list drives both the GUI file-dialog filter and CLI/GUI folder
  scanning, so the app could write an `.avif` it then refused to reopen, and
  README's claim of AVIF input was not true in practice. A round-trip test now
  covers save-then-load, which also serves as the check that dav1d is actually
  linked for decode rather than being dead weight.
- The `process_ram_mb` test was gated on `cfg(any(macos, windows))` to match the
  function, but the function has no macOS implementation and always returns
  `None` there, so `.expect()` would have failed on the newly added macOS CI
  leg. Narrowed to `cfg(target_os = "windows")`.
- Removed a dead `let _hs = HANDLE_SIZE / 2.0;` from `hit_test_handle`. It was
  the only line in `interaction/` no test could reach — mutating it survived
  because nothing consumed the value.
- Noted that `toolbar::lighten` assumes an opaque input: `Color32` stores
  premultiplied channels, so a translucent colour gets premultiplied a second
  time by `from_rgba_unmultiplied` and darkens instead of lightening. Every
  caller passes an opaque constant, so this is a documented constraint rather
  than a behaviour change.

## [1.4.5] - 2026-07-29

### Added

- Golden-value tests across the enhancement and shape modules. The existing
  assertions largely checked that an operation had *run* — dimensions
  preserved, a value moving in the right direction, a signed distance having
  the right sign — which an arithmetic operator swap survives unchanged. These
  pin exact outputs for hand-computed inputs instead. Surviving `cargo mutants`
  mutants across `enhance/tone.rs`, `enhance/detail.rs`, `enhance/skin.rs`,
  `shape/outline.rs`, and `shape/mask.rs` fell from 463 to 40.
- Inline test modules for `shape/outline.rs` and `enhance/skin.rs`, neither of
  which had one. The outline geometry helpers (`cubic_bezier`, `koch_fractal`,
  `rounded_rect_points`, `chamfer_polygon`, `rounded_polygon`,
  `bezier_polygon`, `fit_points_to_bounds`) were reachable only through the
  public API and so were untested directly.
- Differential tests for the bilateral skin-smoothing filter and the raster
  mask loop, comparing each against an independent reference implementation
  written from the definition. Hand-computing expected pixels is impractical
  for both: a single smoothed pixel is a ratio of two 25-term sums of products
  of two exponentials, and the raster mask depends on tiny-skia's antialiased
  coverage.
- GUI widget tests built on [`egui_kittest`](https://github.com/emilk/egui/tree/main/crates/egui_kittest)
  (new dev-dependency, pinned to the same 0.35 line as `egui`). Eight tests in
  `fcs-gui/src/ui/widgets.rs` run the custom widgets against a real
  `egui::Ui`, clicking at coordinates the test derives independently from the
  widget's own layout inputs — so wrong hit geometry sends the click to the
  wrong place and fails the assertion. These cover segment selection, toggle
  state, panel-header clicks, and the slider's five value-format arms.

- Golden-value and boundary tests across the remaining pure-logic modules:
  colour-space conversions, Laplacian variance, red-eye correction, the
  enhancement presets, PNG/JPEG metadata builders and parsers, and SQL query
  validation. Together these took 226 surviving mutants down to 61. As before
  the recurring gap was inputs that make different code paths agree: every
  colour case used saturated primaries, where lightness is exactly 0.5 and the
  saturation denominator is always 1; every CMYK case had k at 0 or 1, so the
  general path never ran; and `laplacian_variance` was asserted only as
  `v >= 0.0`.
- `EnhancementSettings` gained a test module. The `natural`, `vivid` and
  `professional` preset values are a product decision that nothing asserted,
  so any of them could have been changed silently.

### Changed

- Updated `base64` to 0.23 and `calamine` to 0.36.1. The `base64` bump is a
  breaking release under Cargo's 0.x rules but needed no code changes; the
  `Engine` trait and `general_purpose::STANDARD` API are unchanged. Note that
  `base64` 0.22 still appears in the tree via `parquet` and `usvg`.
- `cargo mutants` now runs with `all_features = true`. The `mapping`, `webcam`,
  `raw` and `heic` modules are all `#[cfg(feature = ...)]` and none of those
  features is on by default, so mutants there landed in code the build skipped:
  the suite passed and they were recorded as missed regardless of how well
  tested they were. Measured on `mapping/sqlite.rs` with identical tests, 31
  mutants: 0 caught with default features, 27 caught (2 missed, 1 timeout, 1
  unviable) with all of them. This had been misreporting roughly 94 mutants
  across `mapping` (70), `webcam` (17) and the HEIC/RAW loaders.
- `cargo mutants` now reads `.cargo/mutants.toml`, which excludes `fcs-gui`
  along with `fcs-cli/src/webcam.rs` and `fcs-cli/src/gpu.rs`. When the config
  was added no test reached any of them — `fcs-gui` had 2 tests against 1608
  mutants, and the other two need an enumerable camera and a wgpu adapter — so
  mutating them only inflated the missed count and obscured the gaps worth
  closing. This drops the workspace mutant count from 5269 to 3535. `fcs-gui`
  stays excluded
  even though it now has a kittest harness: measured on `ui/widgets.rs`, the
  most logic-heavy file in the crate, interaction tests catch 23 of 116 viable
  mutants (20%). Everything caught is a return value, click route, state
  change, or format string; everything missed is a paint parameter — colours,
  corner radii, stroke widths, text offsets. `gpu_pill`, `ctl_pill`, `tb_sep`,
  and `field_label` return nothing and only paint, so even replacing the whole
  function body with `()` survives. Reaching those needs snapshot testing and
  committed baseline images; the measurement is recorded in
  `.cargo/mutants.toml` so the experiment need not be repeated.
- Removed two redundant clamps in `shape/outline.rs`, both behaviour
  preserving: `outline_points` re-clamped corner percentages that
  `CropShape::sanitized()` has already capped at 0.5, and `rounded_polygon`
  halved each adjacent edge length separately when `min(a*0.5, b*0.5)` is just
  `min(a, b)*0.5`. The equivalent clamp in `shape/mask.rs` is retained —
  `apply_shape_mask` accepts an unsanitized shape, so there it still binds.

## [1.4.4] - 2026-07-26

### Changed

- CI now runs the `ci` nextest profile (no fail-fast, one retry for flaky GPU
  adapter acquisition, five-minute hang timeout) with `FCS_STRICT_TESTS=1`, so a
  missing model or fixture fails the run instead of silently skipping tests.
- CI reports line and region coverage via `cargo llvm-cov nextest`, writing the
  totals to the job summary and failing below 60% line coverage. The same single
  test run is instrumented, so this does not add a second pass.

### Added

- Excel mapping reader tests covering headers, header-less sheets, blank-row
  skipping, preview truncation, explicit sheet selection, and open failures,
  built on an in-test minimal `.xlsx` writer. Coverage of
  `fcs-utils/src/mapping/excel.rs` went from 0% to 97% of lines.
- Regression test pinning `CropRegion::requires_padding` for a face fully inside
  the source image; only the padded case had been asserted.
- A "Test tooling" section in the README covering nextest, coverage, strict
  fixture mode, and the `cargo mutants` commands, including why mutation testing
  stays out of CI.

### Fixed

- `--naming-template` can no longer write outside the chosen output directory;
  path separators, `..` segments, and Windows drive prefixes in a template or
  source filename are reduced to a single plain filename.

## [1.4.3] - 2026-07-22

### Changed

- Strengthened CI with cargo-nextest, separate doctest coverage, and
  property-based verification that optimized NMS matches its reference
  implementation.
- Updated Rust dependencies, including `anyhow`, `thiserror`, `clap`,
  `fast_image_resize`, `bytemuck`, `serde`, and `serde_json`.
- Documented how to curate local face fixtures and generate OpenCV golden
  detections.

### Fixed

- CLI JSON snapshot regressions now fail when the command exits unsuccessfully
  instead of being reported as passing.
- Local OpenCV parity tests now resolve models and fixtures from the workspace
  instead of silently skipping them from the crate directory.

## [1.4.2] - 2026-07-17

### Fixed

- Custom position offsets X/Y are now drag-value fields covering the full
  -1.00 to 1.00 range; the previous text boxes reformatted to whole numbers on
  every frame, making values like 0.5 or -1 impossible to enter. The fields
  are greyed out unless positioning mode is Custom. ([#4])
- Chinese (and other CJK) file names now display in the GUI: a system font
  (Microsoft YaHei, PingFang, or Noto Sans CJK) is loaded as a glyph fallback
  when available. ([#4])

## [1.4.1] - 2026-07-13

### Changed

- Targeted release binaries at `x86-64-v3` and replaced slower `libm`-style
  calls in hot loops with native floating-point methods.
- Updated dependencies, removed three unused dependencies and dead code, and
  documented supported architectures and RAW input.

## [1.4.0] - 2026-07-05

### Added

- Undo and redo for manual face-box edits.
- Animated GUI widgets, hover cursors, and an empty-canvas backdrop.

### Changed

- Parallelized unsharp masking with Rayon.
- Improved Linux build and release compatibility.

## [1.3.0-beta] - 2026-06-26

### Added

- Camera RAW input behind the `raw` Cargo feature (enabled in CLI and GUI):
  DNG, CR2, CR3, NEF, ARW, RW2, ORF, RAF, SRW, and PEF decode via the pure-Rust
  `imagepipe`/`rawloader` stack, routed through the existing image-load path so
  batch and single-image flows both accept RAW. Note: `rawloader` does not
  support every DNG variant; unsupported files are skipped rather than crashing.
- HEIC and HEIF input behind the `heic` Cargo feature.
- Architecture and detection-pipeline diagrams.

### Changed

- Updated `egui`/`eframe`/`egui_extras` to 0.35 and bumped `anyhow`,
  `env_logger`, `log`, and `tract-onnx`.
- Release builds now use `panic = "unwind"`, allowing per-file panic recovery
  to skip an undecodable batch item instead of terminating the process.

### Fixed

- GPU preprocessing falls back to CPU for images larger than the device's
  maximum 2D texture dimension.
- GUI preview and thumbnail textures are downscaled to the texture-side limit;
  detection and cropping still use the full-resolution source.

## [1.2.7-beta] - 2026-06-13

### Changed

- Split large modules into focused files and removed redundant copies and image
  passes from hot paths.
- Updated Rust dependencies, including `imageproc` 0.27 and `tract-onnx` 0.23.

### Fixed

- Corrected the macOS release build and the `nokhwa` dependency resolution.

## [1.2.4-beta] - 2026-06-03

### Added

- A redesigned gallery and updated workflow documentation.
- Batch export failure log now records skipped-but-detected images, not just
  hard failures. Items with `BatchFileStatus::Failed`, and items marked
  `Completed` with `faces_exported == 0`, are both logged.
- A `path` field/column in the `batch_failures.json` / `batch_failures.csv`
  output so each entry maps back to its source file.

#### Log format

`batch_failures.json`:

```json
[
  {
    "index": 3,
    "path": "C:\\images\\vacation\\img_003.jpg",
    "error": "No faces detected",
    "faces_detected": 0
  },
  {
    "index": 5,
    "path": "C:\\images\\vacation\\img_005.jpg",
    "error": "Faces detected but skipped (quality checks)",
    "faces_detected": 2
  }
]
```

`batch_failures.csv`:

```csv
index,path,error,faces_detected
3,"C:\images\vacation\img_003.jpg","No faces detected",0
5,"C:\images\vacation\img_005.jpg","Faces detected but skipped (quality checks)",2
```

### Changed

- Centralized workspace dependencies and bundled GUI GPU state into one
  pipeline.

## [1.2.3-beta] - 2026-05-25

### Added

- Cross-platform copy and paste support for Windows, macOS, and Linux.

### Changed

- Improved documentation screenshots and release workflow portability.

## [1.2.2-beta] - 2026-05-17

### Added

- Linux and macOS release binaries.
- Multithreaded batch export.

### Changed

- Replaced custom window chrome with native platform chrome.
- Pooled GPU storage and readback buffers and reduced preview-cache memory use.
- Consolidated supported-image extension handling across CLI and GUI.

### Fixed

- Custom output dimensions now take precedence over stale preset labels.
- Crop mode configuration parsing and redundant image reads.

## [1.2.0-beta] - 2026-05-13

### Added

- Live webcam streaming, manual face-box drawing, free rotation, and 90-degree
  rotation controls.
- Face thumbnails, detection timing, and GPU status in the GUI.
- Enhancement settings in both single-image and batch export paths.

### Changed

- Restricted red-eye correction to detected eye landmarks when available.
- Limited scroll-to-zoom input to the image canvas.

## [1.1.0-beta] - 2026-05-09

### Added

- A redesigned GUI with menus, presets, aspect-ratio controls, shape selection,
  mapping drop zones, and persistent batch actions.
- Windows MSI and NSIS installers with optional `PATH` integration.
- AVIF decoding support.

### Changed

- Applied EXIF orientation during image loading and normalized JPEG orientation
  metadata.
- Renamed packages to the `fcs-*` names and aligned release branding.

### Fixed

- Transparent fill compositing, aspect-ratio selection, narrow-layout clipping,
  and installer/release workflow failures.

## [1.0.0] - 2026-02-15

First public release. Windows release binaries (`fcs-cli.exe`, `fcs-gui.exe`).
See [docs/releases/v1.0.0.md](docs/releases/v1.0.0.md) for the full release notes.

### Added

- End-to-end face crop pipeline across CLI and GUI, powered by YuNet:
  preset or custom output dimensions, face-height targeting, positioning modes
  (Center, Rule of Thirds, Custom offsets), out-of-bounds fill color, and
  shaped/vignette masking.
- Quality scoring and automation: Laplacian-variance classification
  (Low/Medium/High), auto-select best face, skip low-quality outputs, and
  quality-suffix naming.
- Enhancement pipeline with CPU and GPU (WGSL) variants: auto-color, exposure,
  brightness, contrast, saturation, sharpening, skin smoothing, red-eye
  removal, and portrait background blur.
- Mapping-driven batch workflows: CSV/TSV, Excel, Parquet, and SQLite imports
  for source/output mapping.
- Clipboard and drag-and-drop support in the GUI: single-image preview,
  folder/path ingestion for the batch queue, and data-table ingestion for
  mapping.
- Custom GPU YuNet inference graph (WGSL Conv2D/BatchNorm/activation), with
  GPU/CPU parity validated in `fcs-core/tests/gpu_cpu_parity.rs`.
- Release automation: tag-driven Windows artifact workflow with checksum
  publishing, plus SHA256 model-integrity checks in CI.

### Fixed

- CSV batch log writes propagate I/O errors instead of unwrapping.
- GPU workspace mutex poisoning returns descriptive errors instead of
  panicking.
- GUI export composites masked transparency against the selected fill color,
  matching preview behavior (with regression tests for opaque and
  semi-transparent compositing).

[#4]: https://github.com/gregorycarnegie/face-crop-studio/issues/4

[Unreleased]: https://github.com/gregorycarnegie/face-crop-studio/compare/v1.6.0...HEAD
[1.6.0]: https://github.com/gregorycarnegie/face-crop-studio/compare/v1.5.4...v1.6.0
[1.5.4]: https://github.com/gregorycarnegie/face-crop-studio/compare/v1.5.3...v1.5.4
[1.5.3]: https://github.com/gregorycarnegie/face-crop-studio/compare/v1.5.2...v1.5.3
[1.5.2]: https://github.com/gregorycarnegie/face-crop-studio/compare/v1.5.1...v1.5.2
[1.5.1]: https://github.com/gregorycarnegie/face-crop-studio/compare/v1.5.0...v1.5.1
[1.5.0]: https://github.com/gregorycarnegie/face-crop-studio/compare/v1.4.5...v1.5.0
[1.4.5]: https://github.com/gregorycarnegie/face-crop-studio/compare/v1.4.4...v1.4.5
[1.4.4]: https://github.com/gregorycarnegie/face-crop-studio/compare/v1.4.3...v1.4.4
[1.4.3]: https://github.com/gregorycarnegie/face-crop-studio/compare/v1.4.2...v1.4.3
[1.4.2]: https://github.com/gregorycarnegie/face-crop-studio/compare/v1.4.1...v1.4.2
[1.4.1]: https://github.com/gregorycarnegie/face-crop-studio/compare/v1.4.0...v1.4.1
[1.4.0]: https://github.com/gregorycarnegie/face-crop-studio/compare/v1.3.0-beta...v1.4.0
[1.3.0-beta]: https://github.com/gregorycarnegie/face-crop-studio/compare/v1.2.7-beta...v1.3.0-beta
[1.2.7-beta]: https://github.com/gregorycarnegie/face-crop-studio/compare/v1.2.4-beta...v1.2.7-beta
[1.2.4-beta]: https://github.com/gregorycarnegie/face-crop-studio/compare/v1.2.3-beta...v1.2.4-beta
[1.2.3-beta]: https://github.com/gregorycarnegie/face-crop-studio/compare/v1.2.2-beta...v1.2.3-beta
[1.2.2-beta]: https://github.com/gregorycarnegie/face-crop-studio/compare/v1.2.0-beta...v1.2.2-beta
[1.2.0-beta]: https://github.com/gregorycarnegie/face-crop-studio/compare/v1.1.0-beta...v1.2.0-beta
[1.1.0-beta]: https://github.com/gregorycarnegie/face-crop-studio/compare/v1.0.0...v1.1.0-beta
[1.0.0]: https://github.com/gregorycarnegie/face-crop-studio/releases/tag/v1.0.0
