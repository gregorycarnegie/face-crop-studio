# Changelog

All notable changes to Face Crop Studio are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.1] - 2026-09-24

Fixes, one security hardening and a speed-up; nothing to do before upgrading. The macOS build that
2.0.0 never shipped now builds. Custom SQL mapping queries are held read-only by SQLite itself
rather than by a keyword list. The built-in GPU engine -- what runs on machines without ONNX
Runtime -- detects 1.6 to 1.8 ms faster. Most of the rest came out of mutation testing: tests that
had been passing without running, a CPU convolution panic in a public function, and a GPU pipeline
nothing called.

One public method is gone, `fcs_core::gpu::GpuInferenceOps::encode_resize2x_add_tensors`, which
would make this a major by the letter of semver. It is a patch because nothing could have called
it: its only caller went with YuNet in 2.0.0, and the crates are not published.

### Fixed

- **The macOS release build exited immediately.** Removing YuNet from
  `installer/macos/build_macos.sh` left an orphaned `exit 1` and `fi` behind the binary-presence
  check, so the script exited 1 in 0.1 seconds with no output. It only shows up in a release
  build, and by then Linux and Windows had already uploaded their artifacts -- a half-built
  v2.0.0 with seven of eight platforms.
- **`fcs_core::cpu::conv2d::conv2d` panicked when a kernel was wider than the padded input** on
  its general path (dense or grouped, not depthwise). The bound on columns that could skip the
  per-tap bounds check used `saturating_sub`, which turned "no column fits" into "column 0 fits",
  so that column read past the end of the input plane. The depthwise path and the reference
  implementation both handled the shape; the detector's own layers never produce it, so the app
  was unaffected. Found while working out why two mutants in that bound survived.

### Security

- **Custom SQL mapping queries are now guarded by SQLite, not by a keyword denylist.** A mapping
  database opens read-only (`SQLITE_OPEN_READ_ONLY`, and without rusqlite's default
  `SQLITE_OPEN_URI`, under which a path spelled `file:x.db?mode=rwc` reopens writable), and a
  `Connection::authorizer` allowlist permits only reads, functions and recursion while SQLite
  compiles the statement. Everything else is refused by default, including actions a future
  SQLite may add. The twelve-word denylist it replaces was guarding a different thing than it
  looked like: `DROP TABLE t` reaches the authorizer as a delete from `sqlite_master`, and
  `CREATE INDEX` and `CREATE VIEW` both arrive as inserts into it. Both guards are kept because
  neither covers the other -- `VACUUM` raises no authorization callback at all and is stopped
  only by the read-only flags.
- **A missing mapping database is now an error** rather than being created as an empty file, a
  side effect of the read-only open flags.

### Added

- **CI parses every installer script** (`bash -n`). These run only inside a release build, so a
  typo in one is invisible until a tag is pushed. The new check takes a second, fails if it finds
  no scripts to check, and was verified against the exact bug above.
- **The Linux CI legs run the GPU tests**, on Mesa's lavapipe, a software Vulkan adapter. Until
  now they skipped all of them: the hosted runners have no graphics card. They are free to run
  on the CPU, as they already were without anyone noting it on Windows (WARP) and macOS (Metal)
  -- so every WGSL shader and dispatch is now checked on all three backends. Software adapters
  say nothing about speed or vendor drivers; they do say whether the numbers are right. Proven
  in WSL first: with no driver 63 GPU tests skip, with lavapipe none do and none fail.
- **`FCS_REQUIRE_GPU`**, set on every CI leg, fails a test when no adapter is found. GPU tests
  skip on "no adapter" by design, so a leg that lost its adapter would otherwise report the same
  green as one that ran them all -- which is how the Linux legs looked the whole time.

### Changed

- **The built-in GPU engine submits each SCRFD forward pass once instead of 66 times**, recording
  every step into one command buffer. It is **-1.6 to -1.8 ms a detection**: 3.59 -> 2.02 ms on an
  RTX 4090 (-44%) and 24.0 -> 22.2 ms on a Radeon iGPU (-7%), paired over 40 blocks of 50 runs
  with every block faster, and bit-identical on every head output of all 1,239 images in the
  reference folder on both adapters. YuNet's runtime had made exactly this change; SCRFD's WGSL
  port, written when YuNet was replaced, went back to one submission per op. Only machines
  without ONNX Runtime use this engine.
- **Mutation testing has a 120-second floor on the test timeout** (`minimum_test_timeout` in
  `.cargo/mutants.toml`). The automatic timeout is five times the baseline, but the baseline runs
  alone while four mutant jobs contend for one GPU: `fcs-utils` baselined at 5.3 s for a 27 s
  timeout while its slowest *passing* mutant took 17.4 s. A timeout reads as a missed mutant, so
  one caused by contention is a silent loss. Of three that looked like contention, one cleared at
  120 s; the other two were genuine -- a test had printed FAILED while a different test under the
  same mutant never ended, and `cargo test` waits for the whole binary. All six timeouts left
  after the full run of 2026-09-24 are genuine hangs or grinds, each identified by name.
- **`rounded_polygon` normalises its arc sweep with an `if` instead of a `while`.** Both angles
  come from `atan2`, so one addition of TAU always suffices; the loop was identical for every
  finite input but turned any mutation of its condition or step into a hang.

### Fixed (testing)

- **`output_dim` is asserted directly** in `fcs-core`'s CPU convolution. The shape-agreement test
  compares against a reference that calls `output_dim` too, so both sides moved together and
  replacing its `/ stride` with `* stride` left every convolution test passing -- surfacing only
  as a mutation-testing timeout, when the inflated dimensions made `scrfd_parity` grind.
- **A rounded-polygon arc point is checked at an index that is not a multiple of `steps / 2`.**
  The three points the test did check were all invariant under an arc swept backwards the long
  way round, so a mutant that shifted the sweep by a whole turn agreed with every assertion.
- **The last ten surviving convolution mutants are killed**, each by asserting a value that no
  output comparison could see, because every wrong value it could take still gave the right
  pixels:
  - the plane count each CPU path passes to `rows_per_task` (six mutants) -- any value yields
    a divisor of the height, so it only moves work between tasks. `rows_per_task` now takes
    batch and channels and multiplies them itself, under its own test;
  - the interior-column bound in the general path (two) -- too narrow only sends columns
    through the bounds-checked path. Extracted as `interior_columns` and checked against its
    definition over every small shape, which is also what found the panic above;
  - the GPU dispatch width (two) -- the shader returns early past the output edge, so a
    sixteen-fold over-dispatch changed no pixel. Extracted as `workgroups`, like `kernel_for`.
- **The default model location is tested.** Every test loaded SCRFD and the eye refiner by
  explicit path, so `ScrfdDetector::load` and `EyeRefiner::load` -- the calls the app actually
  makes at startup -- could be replaced with `None` and nothing failed. Both had been accepted as
  untestable, because `load` resolves `models/` against the working directory and changing that
  is process-global. A test file of its own holding a single test has no other test to race
  with, so `tests/default_model_location.rs` moves to the workspace root and requires both to
  load, under `FCS_STRICT_TESTS`.
- **Three `FaceDetector` tests had never run.** They called `FaceDetector::load`, which finds
  nothing from the crate root, so they skipped on every machine -- and did so without
  consulting `FCS_STRICT_TESTS`, so strict CI counted them as passes. One of them is the guard
  for "a setting read into the struct but never passed to the model". They now load the model
  by path from the workspace root and fail under strict when it is missing; all three pass.
- **Every reachable surviving mutant is killed; what is left is equivalent or unmeasurable here.**
  The full run of 2026-09-24 missed 75. 33 now have tests, 8 went with dead code (below), and
  the rest are listed with their reasons in `.cargo/mutants.toml`, so a future run is checked
  against that list rather than re-triaged. Among the tests: a zero blur radius switches
  sharpening and background blur off on both engines -- that is a contract, not a fast path,
  because `fast_blur` at radius 0 still changes pixels (the first draft of this list called
  those mutants equivalent until a probe showed otherwise); telemetry output is now asserted
  through a per-thread log capture, and the tests that flip the global telemetry switch share a
  lock, closing a latent race between them; the GPU engine's accessors, the pooled-dimension
  formula, the eye refiner's output-length check and `FaceDetector::load` are asserted directly.
  A third, redundant zero-radius guard in the GPU enhancer is removed -- `try_gpu_blur` and the
  CPU fallback already carried it, which is why its mutants could not be killed.
- **ONNX Runtime handles are released, and a test proves it.** `Environment` and `Session` free
  their native handles only in `Drop`, and no functional test can see a leak, so replacing either
  `drop` with `()` survived. The test runs the real runtime through a copy of its API table whose
  three release entries count their calls before forwarding, then requires exactly one of each.

### Removed

- **`GpuInferenceOps::encode_resize2x_add_tensors` and its fused upsample-and-add pipeline.** It
  was experiment 36's optimisation for YuNet's neck, and its only caller went with YuNet in 2.0,
  leaving a shader compiled at every GPU start and never dispatched. SCRFD's neck has the same
  upsample-then-add shape twice, so the fusion was brought back and measured before this entry was
  final: bit-identical on all 1,239 images of the reference folder on both a 4090 and a Radeon
  iGPU, and not measurably faster -- +0.025 +/- 0.037 ms and -0.049 +/- 0.160 ms paired, with the
  pass batched as above. Two dispatches out of 66 are not where the time goes, so it stays deleted.

## [2.0.0] - 2026-09-21

The licence question is closed: nothing shipped is trained on non-commercial data. YuNet is
gone, and with it the fallback detector whose WIDER FACE weights made a commercial application
depend on a model released for "non-commercial academic research only". SCRFD -- trained here on
80,000 CC BY 2.0 Open Images photographs -- now runs on all three engines, so there is no
machine left that needs a second detector.

The major version is for the API and output changes that came with clearing that out, listed
below. Two other things are worth knowing before upgrading: **a failed export used to be
reported as a success, and a failed write used to destroy the file it was replacing** -- both
fixed, across all six write paths. And **six settings turned out to do nothing**; the ones that
could be made meaningful were wired up, the rest are deleted, and a test now requires every
remaining setting to change something observable.

### Added

- **A test that every setting changes something** (`fcs-core/tests/settings_have_effect.rs`).
  Six settings in a row had turned out to do nothing, each found by accident while deleting
  adjacent code. That is one missing invariant, not six bugs, and this is the invariant: flip one
  field, require an observable difference from the pipeline it feeds.

  Reachability is deliberately *not* what it checks. A grep finds a reader for all 53 fields, and
  found one for `nms_threshold` while it did nothing -- read into `FaceDetector`'s own field and
  stopped there. Verified to fail, too: dropping `brightness` on its way into the enhancement
  mapping is reported as `enhance.brightness` wired to nothing.

  It found one on its first run. See below.

- **An independent oracle, back in CI.** Until now every test compared this project against
  itself: three engines agreeing to 1e-05, golden crop regions, a CLI snapshot. That proves the
  parts are consistent, not that they are right. `tract` used to interpret the ONNX file rather
  than re-encode it, which is what let it catch a mistake both of our own engines shared, and it
  left with YuNet in 1.9.0.

  `fcs-core/tests/python_parity.rs` restores it by comparing against
  `tools/dataset/scrfd_detect.py` -- the same checkpoint run through torch in WSL, decoded by the
  numpy reference. Agreement covers the whole chain at once: the top-left letterbox, RGB channel
  order, normalised padding, the export and the decode, every one of which was wrong at some
  point during the port. On the committed fixtures the two paths agree on **9 of 9 faces**, with
  the worst box edge 1.53 px and a median error of 0.44% of box width.

  **Verified to fail, not just to pass.** Flipping the channel order to BGR is caught four ways
  at once (a confident face lost, the worst box edge at 27 px, max error 30.7%, score drift over
  its limit); centring the letterbox the way YuNet did drops matches from 9 to 1. Neither
  mutation is detectable by any amount of self-comparison -- all three engines would agree
  happily on the wrong convention.

  It also pins the shipped model to `epoch_100.pth`: exporting from a different epoch moves
  every score far more than the tolerances allow.

- **`fixtures/oracle/`: the first and only images committed to this repository.** Eight Open
  Images photographs, every one CC BY 2.0, attributed in `fixtures/oracle/ATTRIBUTION.md`. The
  rest of `fixtures/` stays git-ignored, deliberately, because it is real faces that were never
  licensed for redistribution -- and the earlier plan to commit some of *those* for this test
  would have quietly undone that policy. These come from the same licence-clean dataset the
  detector was trained on, and the licence was verified rather than assumed: all 167,056 rows of
  the val+test splits carry exactly one licence.

  Chosen for spread rather than size, because a wrong preprocessing convention shows up as a
  systematic box shift and it is the letterbox axes and strides that shift: aspect ratios 0.57 to
  2.60, face heights 4% to 79%, face counts one to five, and one image where both sides must find
  nothing. 227 KB in total.

- `FCS_PARITY_REFERENCE` points that test at a larger corpus instead, which is how it began life
  as `examples/scrfd_parity.rs`. The example is gone rather than duplicated: two copies of a
  comparison is how the two sides drift apart.

- **`docs/ENGINE_SPEED.md` and `examples/engine_speed.rs`:** what a detection costs, per
  engine, measured honestly. The GPU figure had never been: wgpu queues dispatches and
  returns, so the old timings stopped the clock before the work happened. Recording alone
  reads as a 22% GPU win; waiting for the readback that `detect` must do anyway makes it 3%
  slower. Also: preprocessing (5.18 ms) is larger than the network (3.67 ms) on a 36 MP RAW,
  and the built-in CPU graph is 6.3x slower than ONNX Runtime -- a usable floor, not a fast
  path.
- `scrfd::preprocess` and `scrfd::Letterbox` are public, so the letterbox can be measured and
  driven without going through `detect`.

### Changed

- **Batch throughput measured, and the worker-count guidance corrected.** Every figure in
  `docs/ENGINE_SPEED.md` was one image at a time; the real workload is a folder through rayon.
  On the 1,239-image reference folder (9.8 MP average), detect + crop + write: **114 images/s,
  8.8 ms each** at the default worker count, against 44.25 ms serial. Cropping, masking and
  writing account for about 0.9 ms of that; the rest is detection.

  Two findings worth acting on:

  - **16 workers now beat 32, inverting experiment 60.** That measurement made rayon's default
    the deliberate choice (32 at 7.85 s against 16 at 8.9 s) while warning the number "moved as
    soon as the work around it changed". It moved again: 16 wins both alternated pairs by 5-8%.
    It is **not** the atomic write added in 2.0 -- with nothing written at all, 16 takes 8.79 s
    against 32 at 9.73 s -- so the change is in the detection path, the part that was replaced.
    The default is left alone (`RAYON_NUM_THREADS` overrides it) because 5-8% on one machine is
    thin and this ranking has now reversed twice, but the stale numbers in `README.md` and
    `fcs-cli/src/main.rs` are corrected.
  - **Parallel efficiency is only 34%** -- 5.4x on 16 physical cores. Ruled out as the cause:
    file writes, and ONNX Runtime's intra-op threads (experiment 69 measured no difference).
    Not ruled out and recorded as open: memory bandwidth in decode and resize, internal locking
    in the shared `fcs_ort::Session`, and page-cache misses on ~1.2 GB of sources.

- **Landmarks are `Option<Landmark>` rather than an all-zero sentinel.** *(breaking)* The
  shipped detector predicts only the two eyes -- nose and mouth were weighted to zero in
  training -- and the other three were reported as `(0, 0)`. Every consumer then had to
  re-derive what that meant, which they did in three different spellings across four call
  sites, and the one that got it wrong was the one that mattered: `face_cropper` treated a
  landmark at exactly the origin as a coordinate, so a missing eye beside a real one rotated
  the crop by the angle from `(0, 0)` to the other eye -- 45 degrees, from a value that meant
  "nothing was predicted". A test asserted that behaviour as correct ("one populated landmark
  is enough to align"); it now asserts the opposite, with the reasoning written down.

  `Option` makes the question unavoidable rather than optional, and the four hand-rolled checks
  collapse into `iter().flatten()` or a two-`Some` pattern. It also makes a real landmark at
  the origin expressible, which the sentinel could not.

- **`--json` writes `null` for an absent landmark, not `[0, 0]`.** *(breaking output format)*
  A consumer had no way to tell the sentinel from a real prediction at the origin. Now:

  ```json
  "landmarks": [[957.67, 719.57], [1227.01, 721.86], null, null, null]
  ```

- **Detector parity is now checked against a weaker oracle, and this is a real
  loss.** `tract` interpreted the ONNX file rather than re-encoding the topology
  by hand, which is what made it able to catch a mistake both of our own engines
  shared. It was dev-only and it left with YuNet. `fcs-core/tests/scrfd_parity.rs`
  replaces the five deleted parity tests: it checks the built-in CPU graph and the
  WGSL engine against ONNX Runtime on the model that ships (1.24e-05 and 1.10e-05),
  and re-adds the concurrency guard that caught pooled GPU buffers escaping between
  rayon workers -- comparing 252,000 head values across 8 concurrent runs, because
  a synthetic input finds no faces and comparing detections would compare two empty
  lists. Everything now compares this project against itself.
- CI and the release workflow fetch both models as release assets and verify their
  digests; `verify-models` now fails when given no digests at all, rather than
  passing having checked nothing. CI gains the detector model, which it never had,
  so the tests that need it stop skipping under `FCS_STRICT_TESTS=1`.
- `FCS_FIXTURE_ROOT` and `FCS_MODEL_PATH` replace `YUNET_FIXTURE_ROOT` and
  `YUNET_MODEL_PATH`.
- `docs/PERFORMANCE.md`, `docs/gpu_research.md`, `docs/parity_report.md` and
  `docs/ONNX_RUNTIME_OPTIONS.md` measure or design around YuNet and are marked
  historical rather than rewritten. The deleted implementation and experiments are
  preserved in the `face-crop-studio-yunet-archive` fork.

### Removed

- **`enhance.enabled`.** *(breaking settings field)* Found by the test above. The GUI enhanced
  unconditionally and the CLI gated on its own `--enhance` flag, so the settings field -- default
  `false`, and present in every `gui_settings.json` -- was read by **nobody**. Deleted rather
  than honoured: the per-feature controls already express "no enhancement" by sitting at their
  neutral values, and a master switch with no UI, reachable only by editing JSON, would silently
  disable everything.
- **`crop.webp_quality` and `--webp-quality`.** *(breaking)* It reached `OutputOptions` and died
  there: `encode_webp` calls `WebPEncoder::new_lossless`, and `image` offers no lossy WebP at
  all. `writer.rs` had documented this as "currently has no effect" for long enough that the
  caveat outlived the setting. Honouring it needs a different encoder, not a config field.

  Existing settings files keep loading -- serde ignores both removed keys.

- **The preprocessing pipeline (`fcs-core::preprocess`), and the settings that fed it.**
  Nothing on the detection path had used it since SCRFD landed: the detector letterboxes
  internally, on conventions the old code did not share (RGB in the top-left, not centred BGR).
  Measured before deleting rather than after (`docs/ENGINE_SPEED.md`):
  - GPU preprocessing cannot beat the CPU above about 2 MP, which the repo had already
    established (experiment 10) and which `WgpuPreprocessor` encoded as a pixel cutoff. On a
    36 MP RAW the cutoff means the CPU path is what runs, so `--benchmark-preprocess` was
    reporting CPU times under a `gpu:` label.
  - The CPU resize is already the SIMD path, about 7 GP/s, so there was no win left to chase.

  Gone with it: `--benchmark-preprocess`, `fcs-cli/src/benchmark.rs`, the `preprocessing`
  Criterion bench, `preprocess.wgsl` and `rgb_to_chw.wgsl`, and `CliGpuRuntime::context`
  (the benchmark was its only reader).
- **`gpu.inference` / `--gpu-inference`.** Inert since SCRFD landed -- written by the CLI and
  the GUI checkbox, read by nobody, because the detector picks its own engine. Deleted rather
  than wired up, because the measurement says there is nothing to expose: the WGSL engine and
  ONNX Runtime are a tie (3.78 ms against 3.67 ms for the network, readback included).
- **`gpu.preprocessing`** and its GUI checkbox, which after the untangle gated nothing.
- **The `input` settings section** (`input.width`, `input.height`, `input.resize_quality`) and
  the `--width`, `--height` and `--resize-quality` flags. The export fixes the input at
  640x640, so the dimensions were never choosable. `resize_quality` is a real trade in
  principle -- preprocessing is 42% of a detection on a 36 MP RAW -- but its quality cost was
  measured against YuNet's preprocessing (experiment 54), is unmeasured against SCRFD, and the
  GUI radio triggered a full detector rebuild for a value nothing read. Re-adding it needs the
  measurement first. Existing settings files keep loading: serde ignores the removed keys, and
  a test pins that.

- **YuNet is gone, and with it the licence question this project set out to
  remove.** It was the fallback detector for machines without ONNX Runtime, so
  every package shipped its weights -- and those weights are trained on WIDER
  FACE, released for "non-commercial academic research only"
  (`tools/dataset/DATA_CARD.md`). Keeping it meant shipping a commercial app
  around a non-commercial model. It is no longer needed as a floor: SCRFD runs on
  the WGSL kernels and the built-in CPU graph as well as ONNX Runtime, agreeing
  to about 1e-05 and producing identical detections, so there is nothing a machine
  can be missing that leaves it without a detector. Deleted: the two `.onnx`
  files, the `opencv_zoo` download and the `onnxsim` re-export from CI and the
  release workflow (and with them Python from both model jobs), `crate::yunet`'s
  compiled-in topology and macros, `YuNetModel`, `YuNetDetector`, `CpuYuNet`,
  `GpuYuNet`, the YuNet halves of the CPU and WGSL graphs, `apply_postprocess`
  and its `[N, 15]` decode, and 22 examples and benches that measured them.
  `tract-onnx` leaves the dependency graph entirely.
- **`nms::dedup_close_centers`.** A second suppression pass after IoU NMS,
  dropping detections whose centres sat within 50% of the larger box's longest
  edge. It only ever ran on YuNet's decode path. **SCRFD has never run with it**,
  and every measurement -- `SCRFD_80K.md`, the 1,239-image batch, the shipped
  1.8.0 -- was made without it, so nothing changes; whether SCRFD produces
  centre-duplicated boxes at a lowered confidence floor is an open question,
  recorded in `ARCHITECTURE.md`.
- **`PostprocessConfig` and `DetectionOutput::scale_x`/`scale_y`.** The config was
  YuNet's three knobs on YuNet's scale and had no callers left outside the crate;
  the two scale fields were always `1.0` and documented as "do not apply this
  again", which is a trap rather than an API.

### Fixed

- **The GUI never used the WGSL enhancement shaders, and the CLI never aimed red-eye removal.**
  Both front-ends enhance crops and each had grown half the feature:

  | | GPU shaders | Eye positions for red-eye |
  |---|---|---|
  | CLI (1.x) | yes | **no** -- hard-coded `None` |
  | GUI (1.x) | **no** -- called the CPU pipeline | yes |

  Neither was a decision. The GUI held a `GpuContext` the whole time and used it for a status
  label; the CLI could not pass eye positions because the mapping from landmarks to crop
  coordinates lived inside the GUI. `fcs_utils::EnhancementRuntime` is now the one answer for
  both, and `fcs_core::eye_positions` is reachable from either. A test asserts that passing the
  eye positions changes the result, so the argument cannot quietly become decoration again.

  This is the divergence predicted by "five copies of one pipeline": the shared geometry was in
  `fcs-core` while the step after it was reimplemented per front-end, so the two drifted in
  opposite directions without either looking wrong on its own.

- **A failed export was reported as a success.** `write_bytes` finished with
  `writer.flush().ok()`, discarding the result. `BufWriter::write_all` only fills a buffer, so
  for anything past the buffer size a full disk or a failing drive reports itself at `flush` --
  and that error was thrown away. The CLI printed `crops_saved=1` and the GUI showed success for
  a crop that had not been written.
- **A failed write destroyed the file it was replacing.** `File::create` and `fs::write`
  truncate the destination before the new bytes arrive, so a write that failed part-way left a
  truncated or empty file where good data had been -- the previous export lost to the attempt to
  replace it. Writes now go to a temporary file beside the destination and `fs::rename` over it
  only once every byte is written, through one shared `fcs_utils::write_atomically`. This was
  not one call site but **six**: crops, the detection JSON, annotated images, the GUI's queue
  list and batch report, and the settings file -- where a half-written config then fails to
  parse and silently reverts every preference to its default. The replacement is atomic;
  durability across a power loss is not promised, and the docs say so rather than implying it.
- **`--annotate` wrote a 0-byte file for every `.jpg` input.** Found while fixing the above, and
  demonstrated with the old code: annotation is drawn in RGBA, JPEG has no alpha channel, so
  encoding refused and `image::save` -- having already created the file -- left nothing but a
  truncated stub that looked like a successful annotation. Alpha is now dropped for JPEG, and a
  source extension with no encoder (a RAW, say) fails with a reason instead of silently. The
  existing tests missed this by only ever using `.png`; both cases are now covered.
- **`--watch` with `--output-dir` pointed at the watched directory fed on itself.** Every crop
  landed as a new filesystem event, was detected, and produced another crop, forever. It is now
  refused before the watcher starts, naming the offending flag and the path as the user typed it.
  An output directory *inside* the watched one is still allowed, and deliberately: the watcher is
  non-recursive, so a subdirectory produces no events and cannot loop.

- **Turning off "GPU preprocessing" disabled GPU *enhancement* in the GUI.** The context was
  obtained as a side effect of building a `WgpuPreprocessor`, and both the "disabled" branch
  and a preprocessor that failed to build returned `None` for the context along with itself.
  The GUI now takes the context directly and reports the adapter from it.
- **The GUI compiled two preprocessing shader pipelines at every startup and dropped the
  result.** On a machine with ONNX Runtime -- the common case, and every released package --
  those were the only compute pipelines built during startup, so this removes shader
  compilation from launch entirely. The saving is not measured; experiment 81 put all five
  pipelines of the old `build_detector` at 150 ms of an 890 ms launch.

- **The NMS threshold and top-K settings do something again.** Since the detector
  changed, `FaceDetector` read `confidence` from settings but held the other two
  at compile-time constants, so the GUI's NMS slider and Top-K control -- and
  `--nms-threshold` and `--top-k` -- moved nothing at all. This is the same class
  of bug as the `score_threshold` rename in 1.8.0 and was found by deleting the
  code that used to consume them. `FaceDetector::with_settings` now takes the
  whole `DetectionSettings` and applies all three, and a test asserts each field
  reaches the detector.
- **The Linux `.deb` shipped the wrong models.** Its asset list carried YuNet and
  neither `scrfd80k_500m_640.onnx` nor `eye_refiner.onnx`, so the package that
  users installed detected with the fallback and levelled crops with the
  detector's own eye points. Both installers' AppImage and macOS paths also
  hard-failed on the missing YuNet file, which would have broken the next release
  build.
- **`fcs-ort`'s end-to-end test pointed at a deleted model.** It loaded YuNet and
  asserted 12 outputs; it now loads the shipped detector and asserts 9. This is
  the test that catches a wrong offset in the hand-maintained `OrtApi` table, so
  it silently skipping would have been the worst of the three.

## [1.8.0] - 2026-09-20

Two models trained for this project ship in this release: a detector that finds
considerably more faces than the one before it, and a small model that fixes where the
eyes are so levelled crops are actually level. Both are optional at runtime -- without
ONNX Runtime the app keeps the detector and landmarks it always had.

### Added

- **Levelled crops use better eye points.** Eye-line alignment rotates each crop
  by the angle between the two eyes, and YuNet's eye landmarks are 4.05° out at
  the median against hand-clicked truth, with only 57.6% of faces within 5°. A
  new 1.5 M-parameter model, `models/eye_refiner.onnx`, reads the face box and
  replaces those two points: 1.18° median, 93.3% within 5°, measured on 1,382
  held-out faces (`tools/dataset/CURVE_RESULTS.md`). Checked by eye as well as by
  number: on the 40 faces where the two disagree most, the refined crops were the
  upright ones every time. It refines in the CLI batch and webcam paths and in the
  GUI load, webcam and batch-export paths; the GUI's live webcam overlay is left
  alone because it draws boxes and never levels anything. It needs ONNX Runtime,
  which every package bundles; a build without it logs one line and keeps the
  detector's own points.
- **`--eye-line-align`** on the CLI. Eye-line alignment was a GUI toggle with no
  command-line equivalent, reachable from `fcs-cli` only by hand-writing a JSON
  config. Documented in the README and `docs/cli_recipes.md`, where it had not
  been mentioned at all.

- **A better detector, trained on this project's own licence-clean data.**
  `models/scrfd80k_500m_640.onnx` is an SCRFD-500M trained here on 80,000 CC BY 2.0
  Open Images photographs (244,683 boxed faces). Scored against complete test
  boxes at matched false-positive rates it finds **86.0%** of faces at 0.11 false
  positives per image, where YuNet 2023 finds 71.6% at 0.14. On a 1,239-image
  corpus neither model had seen, judged crop by crop, it finds 135 faces YuNet
  misses while missing 13 that YuNet finds. It costs 6.4 ms/image against YuNet's
  5.8 on ONNX Runtime, and 2.5 MB on disk. `tools/dataset/SCRFD_80K.md` records
  the measurements, including what they do not establish.

  It is **preferred, not a replacement**: it runs only under ONNX Runtime, while
  YuNet's architecture is compiled into the built-in CPU graph and the WGSL
  kernels. A machine without a runtime keeps YuNet rather than losing detection,
  the same way the eye refiner degrades. Which one is running appears in the log
  as `Detector: SCRFD-80k on onnxruntime`.

### Changed

- **Faster test builds.** Dependencies in the dev profile now carry line tables
  instead of full debuginfo. They are built at opt-level 3, so their locals were
  mostly optimized out anyway, but the debuginfo doubled every PDB -- and a test
  build links around 60 binaries, 34 of them fcs-core examples. `fcs_gui.pdb`
  drops from 515 MB to 288 MB, and `cargo test --workspace --no-run` after
  touching fcs-core from 86-92 s to 65-80 s. Backtraces keep file and line, the
  workspace crates keep full debuginfo, and release builds are untouched.
  Swapping MSVC `link.exe` for `rust-lld` was measured too and made no difference.
- Removed dependencies nothing used: `thiserror` from fcs-core, `predicates` from
  fcs-cli's tests and `sha2` from fcs-gui's. Test builds compile eight fewer
  crates (`sha2`, `digest`, `block-buffer`, `crypto-common`, `const-oid`,
  `hybrid-array`, `float-cmp` 0.10, `normalize-line-endings`).
- Dependency bumps: `clap` 4.6.6 → 4.6.7, `crc32fast` 1.5.1 → 1.5.2.
- **`cargo nextest-fast` and `cargo test-fast`.** Aliases that run the suite with
  `--lib --bins --tests`, leaving out fcs-core's 35 examples. A test build after
  touching fcs-core drops from 39-45 s to 22 s. CI still compiles the examples.
- **Shaders are translated for Vulkan and Metal on every CI leg.** The WGSL
  validation test now also lowers each shader to SPIR-V and to MSL 2.3. The macOS
  and Linux runners have no GPU adapter, so until now nothing there reached the
  translation step that fails for a shader those backends cannot take.

- **The YuNet fallback is no longer built when SCRFD is going to replace it.** On
  the GPU path that was five compiled WGSL pipelines and the VRAM they hold, for a
  detector that was then never called.
- **The GUI status line names the detector**: "SCRFD-80k ready on onnxruntime", or
  "YuNet 2023 ready on wgsl-gpu" without a runtime. Which one you get depends on
  what is installed, and until now only the log said so.

### Fixed

- **A CMYK JPEG no longer kills the whole batch.** `decode_jpeg_turbo` asked
  libjpeg-turbo to convert four-component (CMYK/YCCK) JPEGs to RGB, which libjpeg
  treats as a fatal error. mozjpeg installs that error handler as `extern "C-unwind"`,
  so it unwound through C rather than returning an error, escaping the `Option` the
  decoder returns for exactly this kind of fallback. The process died with no message
  and no output file: a 12,416-image folder processed 12,387 images and then wrote
  nothing, because 7 of them were CMYK. Those files now fall through to the `image`
  crate, which decodes them.

- **`Export batch report…` is visible again.** The queue's action bar sat in a
  fixed 96 px reserve at the bottom of the sidebar, and once the bar grew past it
  the report button was drawn below the window edge. The bar is now a bottom panel
  that sizes itself to its buttons, with the file list scrolling in the rest.

- **The label files behind the earlier training runs boxed only the faces whose
  eyes had been clicked** -- 2,500 boxes in images where Open Images drew 12,070 --
  so every run learned ~9,300 real faces as background, and every false-positive
  figure counted real, unlabelled faces as false positives. Nothing shipped was
  affected; the correction is at the top of `tools/dataset/CURVE_RESULTS.md`.

- **SCRFD reported three landmarks it had never learned.** The detection JSON
  carried five points per face, of which the nose and mouth corners sat on top of
  each other above the face box. The model's landmark head was trained on eye
  pairs with nose and mouth weighted to zero, so those outputs never received a
  gradient and decoded to the anchor centre. Cropping was unaffected -- it reads
  the two eyes, which the refiner replaces -- but the JSON published them and both
  the CLI's `--annotate` and the GUI's preview drew all five. They now read as the
  all-zero "absent" point, which the drawing code skips.
- **The packaged-build check in the release workflow matched a log line that had
  been renamed**, so a correct Windows package failed its own verification. Only a
  tagged build runs that workflow, which is why it survived every local check.
- **One unreachable host could fail the whole Linux release.** dav1d was fetched
  from a single mirror, which timed out from both Linux runners twice in a row. It
  now retries and falls back to the project's GitHub mirror.

- **The CLI announced a model it may never open.** `Loading YuNet model from ...`
  was logged before the detector was chosen, so a run that selected SCRFD claimed
  to be loading YuNet two lines above the line saying SCRFD had loaded. It now
  reads `YuNet fallback configured: ...`, and the detector that won is still
  logged after the choice is made.

- **The GUI named the wrong detector.** The header badge hard-coded
  `YuNet 640 · ready` and the status bar took its name from the configured model
  path, which only ever points at the YuNet fallback -- so both read YuNet while
  SCRFD was detecting. Both now report the detector that actually loaded, as does
  the status line.

## [1.7.0] - 2026-09-13

### Added

- **Batch reports.** `Export batch report…` under the queue writes one row per
  image -- path, outcome (`succeeded`, `failed`, `no_faces`, `filtered`,
  `pending`), faces found, faces exported and the error -- as CSV, or as JSON
  with success and failure totals, chosen by the file's extension.

- **`Export every face` beside `Run batch`.** The batch already exported every
  detected face whenever the `Auto-select best face` quality rule was off, but
  that switch lived in the Settings menu and the shipped `config/gui_settings.json`
  turns it on. The checkbox is the same setting, so the two stay in step;
  `Skip if no high-quality face` still applies.

- **Watch-folder mode in the CLI.** `fcs-cli --watch <dir>` monitors a directory
  and runs the ordinary batch crop/export path on images as they arrive, so a
  scanner, camera import or shared drop folder can feed it continuously. Files
  already in the directory are left alone — pass `--input <dir>` for those — so
  pointing it at a processed folder does not rewrite every crop. A file is only
  picked up once its size has stopped changing for 400 ms, because a create event
  arrives long before whatever is writing the file has finished, and an
  undecodable file is logged and skipped rather than ending the session. Cannot
  be combined with `--json`: the mode runs until interrupted, so there is no final
  document to write.

- **Live face detection on the webcam preview.** A `Live` toggle next to the
  camera's `Detect faces` button runs detection on every frame instead of only on
  demand, drawing boxes that follow the subject. Measured in the running
  application: **95% of frames tracked at 15 fps**, detection 6.9 ms against a
  69 ms frame interval. One detection runs at a time and frames arriving during
  one are dropped rather than queued, so the overlay tracks the picture rather
  than trailing it. The sidebar shows detection latency and the dropped-frame
  count while it is on.

  Live results carry boxes only. Crop thumbnails, quality scores and the edit
  history stay on the deliberate `Detect faces` button, which is the only thing
  that wants them.

### Changed

- **Queue rows say what happened to each image.** A finished row read
  `Done · N exported` whatever the result, so `0 exported` covered an image with
  no faces, one whose faces the quality rules all held back, and one whose crops
  failed to save. Rows now read `Exported 2 of 3 face(s)` in green,
  `No faces found` or `N found, none passed quality rules` in orange, and
  `Failed` in red with the reason on hover.

- **Charcoal and orange GUI theme.** The palette now matches the Face Crop Studio
  website -- charcoal surfaces with orange accents -- in place of the navy
  mockup palette.

- **Cropping no longer copies the source region twice.** Extracting the crop went
  through `crop_imm(..).to_image()`, which allocates a second copy of the region
  and fills it one pixel at a time through an accessor that re-checks the image's
  pixel format on every pixel; a second loop then copied that buffer into the
  padded canvas, also pixel by pixel. It now copies a row at a time. **About 4% of
  the CPU a 1239-image folder job spends**, with all 959 crops byte-identical.
  Wall time is unchanged on a 16-core machine, so this is efficiency and headroom
  on smaller ones rather than a faster batch.

- **The window opens before the detector is built, not after.** Building the
  detector compiles five compute pipelines, and it ran on the way to the first
  frame, so the window sat empty for it. It is built on a background thread
  instead: **launch to first painted frame goes from 894 ms to 737** over 24
  alternated launches, and the detector is still ready at the same moment
  (896 ms against 894), because the build now overlaps the first frames rather
  than preceding them. Anything the app is asked to open in that window waits for
  the detector rather than being told the model is not configured.

- **Non-square images are letterboxed into the detector rather than squashed.**
  The preprocessor scaled x and y independently, so a 16:9 photo reached the model
  stretched to a square and the face it was shown was distorted in proportion to
  how far the source was from 1:1. It now scales both axes by the same factor,
  centres the result and pads the rest. Over a 1239-image folder that is **1130
  faces found instead of 1030** -- 77 images gain a detection and 19 lose one --
  concentrated on the widest sources, and detections on non-square images move
  accordingly (median box IoU 0.76-0.87, landmarks 34-46 source pixels). Square
  images are unaffected.

  It is not slower: the resize target is now the drawn region rather than the full
  square, which is 44% fewer output pixels for a 16:9 source, and a 2 MP detection
  went from 1.86 ms to 1.48 ms.

- **Sources up to 1.75 MP are preprocessed on the GPU, up from 1.5 MP.** The old
  cutoff was measured against a route that no longer exists: preprocessing used to
  fall back to a CPU resize *and* a CPU conversion *and* a 4.9 MB float upload, and
  that fallback is now a CPU resize plus a 1.2 MB byte upload. Alternating the two
  routes at a fixed source size in one process puts the crossover at 1.75-1.85 MP,
  so sources in the band this opens are **0.25-0.29 ms faster** -- 0.47-0.61 ms when
  the CPU is busy, which is what a folder export with 32 workers looks like. The two
  routes do not agree exactly (landmarks move 0.65 px at p50, 11 px at worst over
  120 fixtures, no faces lost or gained), but they disagree by the same amount at
  sizes either cutoff routes the same way: the seam comes from having two routes,
  not from where the boundary sits.

### Fixed

- **Mapping files matched image names case-sensitively.** Windows and macOS treat
  `Photo.JPG` and `photo.jpg` as one file, and spreadsheet exports recase names,
  but the queue compared mapping rows byte for byte. A 1239-row mapping over the
  reference folder matched 497 images, and the other 547 crops kept their
  original names; by the new rule 1238 rows match (the last names no file). Rows
  match by file name and then stem, each trying the exact spelling before
  ignoring case, so a file name also now wins over an earlier row that only
  shares its stem. Applying a mapping clears names an earlier mapping left on
  files it no longer matches.

- **A crop that failed to save still counted as done.** The error was logged and
  the image reported `Completed` with a lower count. It now fails with the paths
  and errors, so the queue, the batch report and the batch totals all show it.

- **The GPU pill said `GPU · wgpu` with the GPU off.** The label came from the
  preprocessing status alone and fell back to that fixed text whenever it carried
  no adapter, including when the GPU was disabled; inference on the GPU with
  preprocessing off meanwhile read as CPU in the status bar. Both now show where
  detection runs: `GPU · <adapter>`, `GPU · inference only`, or `CPU`.

- **Every wait for the GPU now has a deadline.** All five blocking `device.poll`
  calls asked to wait indefinitely, so a submission that never completed would
  park the calling thread forever with nothing to report. They are now one shared
  `wait_for_gpu` with a 30-second limit -- three orders of magnitude above any
  real wait, and past the two seconds at which Windows resets an unresponsive GPU
  by itself -- which reports the operation and the deadline instead of hanging.
  An expired wait leaves nothing mapped and returns no buffer to a pool.

- **A fresh install detected with the nearest-neighbour resize.**
  `InputDimensions::default()` said `Speed` while `ResizeQuality::default()` and
  the shipped `config/gui_settings.json` both said `Quality`, so anyone starting
  without a settings file got the fast resize without choosing it. Measured over
  120 fixtures, that setting loses 2 of 51 faces and moves landmarks by 33.6 px at
  worst. The default is now `Quality` everywhere; an explicit `Speed` in a config
  file or on the command line is unchanged.

- **`--webcam-width` and `--webcam-height` had no effect.** The camera was opened
  with `AbsoluteHighestResolution` and then asked to change resolution, which
  cameras ignore, so a request for 640x480 delivered 1920x1080 on a C920 and
  every frame carried 6.75x the pixels asked for -- 4.3 ms of MJPEG decode per
  frame instead of 0.8. The camera is now opened with the format closest to what
  the caller asked for, falling back to the previous behaviour if that cannot be
  satisfied.

### Removed

- **GPU batch cropping.** `GpuBatchCropper` converted the full-resolution source
  to RGBA, packed every pixel into a `u32` and uploaded all of it -- 40 MB for a
  10 MP photo -- to produce one 512x512 crop. Cropping on the CPU instead takes a
  1239-image folder from a median **17.25 s to 9.8 s, 43% faster**,
  order-alternated and winning every pair; the saving holds for the shaped crops
  too (38% rectangle, 47% koch snowflake, 38% star), where the mask costs the
  same either way and lands on the finished 512x512 image.

  It was also the lower-quality path. `crop.wgsl` sampled a fixed 2x2
  neighbourhood -- four source pixels however far the crop was being downscaled
  -- where `crop_face_from_image` uses Lanczos3. **Crops therefore change**: up
  to 85 per channel on a rectangle, less under a mask, and 18% of them shift
  quality label. The shift is almost entirely downward, which is what
  undersampling predicts, since aliasing adds high-frequency detail and the
  sharpness metric is Laplacian variance. One image in 171 selects a different
  face, because `auto_select_best_face` ranks by that same score. Crops from both
  paths were compared before this was removed.

  This deletes about 740 lines: `crop_batch.rs`, `crop.wgsl`, the CLI's cropper
  plumbing and the `BatchCropRequest`/`GpuBatchCropper` exports.

### Changed

- **Batch resizes no longer pay to avoid being threaded.** `threading_pays`
  holds a small resize to one core by running it inside a one-thread rayon pool,
  but from *inside* a rayon worker `install()` is a cross-registry hop rather
  than an ordinary join, and a batch profile put that hop at **17.9% of all
  CPU**. The gate now yields when a worker is already running the resize;
  off-worker callers -- single image, GUI preview, webcam -- are unchanged, which
  is where the 4 MP threshold was measured and where it still holds. Pixel output
  is identical.

- **A bad detector input size now fails immediately and says what to change.**
  `input.width` and `input.height` accepted any value, and anything other than
  640x640 failed every image separately -- the bundled model and the GPU graph
  are both fixed at that size -- before ending with "all detections failed" on a
  whole folder. The backend is now probed once at construction, so the error
  arrives before any files are read and names the setting. The built-in CPU graph
  does support other sizes, and a run configured that way now falls back to it
  instead of failing outright.

- **CPU inference gets more of the machine when nothing else is using it.**
  ONNX Runtime's intra-op threads were pinned to 1 so the runtime would not fight
  rayon, but that pool belongs to the shared session and is a total rather than a
  per-inference multiplier. One inference at a time now takes **4.17 ms against
  7.27 ms**; a folder export is unchanged at 10.2 s, because rayon has already
  filled the cores there. The new default scales with logical processors and
  leaves machines with four or fewer exactly as they were.

- **NMS no longer degrades on crowded scenes.** The spatial grid used a fixed
  32x32 resolution, so its cells were sized by the scene bounds -- and a tight
  cluster of faces has small bounds, making each cell far smaller than the boxes
  and putting every box into hundreds of them. On 5000 clustered candidates that
  cost **23.1 ms; it is now 0.25 ms**. The post-NMS dedup pass also removed from
  the middle of a vector inside its inner loop, which on the same input went from
  6.1 ms to 0.009 ms. Both are pure speed changes; detections are unchanged, and
  a folder run produced byte-identical crops.

- **Large previews no longer stall for seconds.** Images past 8192 pixels a side
  -- camera RAWs and panoramas -- are downscaled for the preview texture, and that
  downscale went through `DynamicImage::resize_exact`, which samples pixel by
  pixel. On a 133 MP source it took **3215 ms; it now takes 103 ms, 31x faster**.
  Images under the limit are untouched, and alpha is preserved for every format
  that carries it.

- **Three dead GUI caches removed, and the `lru` dependency with them.** The
  detection cache was written and cleared but never read, and each entry held a
  full decoded source image plus a texture, so browsing fifty images retained
  around 1.3 GB for lookups that never happened. The crop-preview cache was only
  ever cleared, and the image cache was never touched. The read was lost in the
  GUI rewrite; its key had also decayed to a bare path, with every other field
  hardcoded. Nothing observable changes.

- **Exports stop copying the image to encode it.** `encode_rgba8` called
  `to_rgba8()`, which clones when the image is already RGBA8, and every exported
  crop is. Borrowed instead, along with `encode_jpeg`'s `to_rgb8()`. Output is
  byte-identical; this is one fewer full-size allocation per concurrent export
  rather than a measurable speed-up.

- **The quality metric's downscale resizes through `fast_image_resize`.**
  `estimate_sharpness` scores a face region cut from the full-resolution source,
  and that downscale was still `DynamicImage::resize`. Worth about **9%** of a
  folder job on top of the two changes below. Exported crops are byte-identical,
  filenames included, and face selection is unchanged -- both come from the score
  of the finished 512x512 crop, which is under the downscale threshold.

  **The JSON report's per-detection `quality_score` does change**, by a median of
  0.000% and at most 6% over a 1239-image folder, since the two resamplers round
  differently. No `quality` label moved across 1032 detections, though six sat
  within 1% of a threshold.

- **Crops resize through `fast_image_resize`.** `crop_face_from_image` works on
  an RGBA canvas and so could not use the existing RGB fast path, leaving it on
  `image::imageops::resize` -- pixel-by-pixel through `GenericImageView`, and
  10.9% of all CPU in a folder job once GPU cropping had gone.

  Together these take a 1239-image folder from a median **9.58 s to 8.17 s**,
  order-alternated and winning every pair. **Crops change very slightly**: the
  same Lanczos3 kernel from a different implementation, so differences are
  rounding -- at most 23 per channel across the folder, mean 0.13, and one crop
  in 901 shifts quality label. For comparison, removing GPU cropping above moved
  pixels by up to 85 and shifted 18% of labels.

- **The bundled GUI settings now use the filtered resize.**
  `config/gui_settings.json` had `input.resize_quality` set to `speed`, which
  selects nearest-neighbour sampling when scaling a source down to the detector's
  640x640 input. On a 1239-image folder that cost **12 detections and 9 crops**
  (1020 faces against 1032, 892 crops against 901) for no measurable time:
  16.3 s against 16.0-17.5 s for the same folder on `speed`, within the roughly
  10% swing batch wall time shows run to run on this machine.

  Nearest sampling reads about four source pixels per output pixel at a large
  downscale rather than averaging the area it covers, which is the same aliasing
  that experiment 51 measured moving landmarks by up to 35 px. `speed` remains
  available for anyone who wants it.

- **JPEG files now decode with libjpeg-turbo.** Decoding is the largest single
  cost in processing a folder -- about 24 ms for a 10 MP photo against 3.1 ms of
  detection -- and libjpeg-turbo decodes the fixture corpus **1.23x faster**
  (517 ms to 420 ms over 15 images of 8-22 MP). It was already linked into every
  binary through `nokhwa`, which uses it for webcam frames, so this adds no new
  native dependency. Only `.jpg`/`.jpeg` take the new path, and anything unusual
  -- CMYK, 16-bit, truncated, or a `.jpg` that is not a JPEG -- falls back to the
  previous decoder, which handles more formats. EXIF orientation is unchanged.

  **Output moves slightly.** The JPEG standard leaves IDCT precision open, so the
  two decoders disagree by up to 5/255 on a channel. That shifts detection boxes
  by 0.28-1.30 px and landmarks by at most 0.37 px, which moves an exported crop
  by about a pixel; on one 1239-image folder it changed the number of detected
  faces from 891 to 892. Crops from both decoders were compared before this
  became the default. The CLI JSON snapshot moved by at most 0.26 px and has been
  updated.

  Requires NASM at build time. Without it `mozjpeg-sys` silently compiles a
  scalar fallback that is *slower* than the decoder this replaces; the Windows
  and macOS release jobs now install NASM and the Windows job fails if it is
  missing. See [CONTRIBUTING.md](CONTRIBUTING.md).

### Fixed

- **Batch processing panicked on most images after the resize cache landed.**
  A thread-local `fast_image_resize::Resizer`, added to stop an 8 MB scratch
  buffer being reallocated per image, held a `RefCell` borrow across the resize.
  Below the threading threshold a resize runs inside `ThreadPool::install`, and
  `install` called from a rayon worker lets that worker take other queued work
  while it waits -- another image's resize, re-entering the same function on the
  same thread. A 1239-image folder produced 247 crops instead of 891 and exited
  with a panic. The `Resizer` is now taken out of the thread-local for the
  duration and put back afterwards, so no borrow spans the call. Only nested
  parallelism reaches this, so single-image use was never affected.

- **GPU convolutions now specialize pointwise and depthwise layers.** Eligible
  1x1 convolutions bypass general spatial loops and compute four output channels
  per thread; depthwise 3x3 convolutions reuse six input values per row for four
  adjacent outputs. The general fallback and fused activations remain available.
  Paired full-graph profiling on RTX 4090 / D3D12 reduced GPU compute from
  **about 0.910 ms to 0.536 ms (41%)** across three separately measured changes.
  Final whole-detection timings overlapped around 3.5 ms, so this is not an
  established detection-latency or batch-throughputput improvement.

  [docs/PERFORMANCE.md](docs/PERFORMANCE.md#experiment-index) records every attempt and decision.
  FP16 storage was slower; subgroup reduction helped small layers but heavily
  regressed large ones. Both remain standalone probes, with no new production
  feature requirements. Convolution timestamps now distinguish pointwise,
  depthwise and general families, and `conv2d_experiment` compares actual shader
  variants with alternating GPU timestamp pairs and raw-output checks. The old
  standard/vec4 benchmark labels, which called the same pipeline, were replaced
  by one explicitly named operation-with-readback benchmark. Validation: 823
  strict workspace tests and two doctests passed, plus Clippy and formatting.

- **GPU inference now records its 61 dispatches in one compute pass when
  profiling is off.** The experiment was kept: on an RTX 4090 / D3D12, three
  alternating, same-process comparisons measured `encoder.finish()` at
  **0.35-0.36 ms with separate passes versus 0.07-0.08 ms merged**. Total
  encode/finish/submit/wait time fell from **1.58-1.66 ms to 1.18-1.25 ms**,
  consistently saving about 0.40 ms. This is smaller than the previously
  suggested ~0.9 ms; that was a hypothesis based on an earlier timing, not a
  saving the experiment established.

  Whole `detect_image` timings also improved: an old/new/new/old sequence of
  release benchmark binaries measured **4.16-4.41 ms separate versus
  3.30-3.63 ms merged**, using `inference_pipeline/detect_image/gpu` (CPU
  speed resize plus GPU inference). These are ranges of run estimates, not
  confidence intervals, and their drift is why the smaller paired measurement
  is recorded alongside them. No batch-throughputput gain is claimed.

  Per-operation GPU profiling is retained. With a profiler present, inference
  uses separate passes and still reports all 53 convolutions, four pools, two
  adds and two resizes. Both modes share the graph, validation, resource
  preparation and dispatch code; only the pass lifetime differs. Each compute
  dispatch has its own [WebGPU usage scope](https://www.w3.org/TR/webgpu/#programming-model-synchronization), so dependencies between layers
  and reuse of pooled intermediates do not require separate passes. The
  execution scope still isolates concurrent inferences through completion.

  `cargo run --release -p fcs-core --example gpu_encode_comparison` reproduces
  the paired measurement with profiling off, reverses the order on each pair,
  and verifies exact equality of all 12 raw detection heads. A regression test
  also compares normal and profiled inference and checks the 61 timestamp
  records; the existing concurrency and ONNX parity checks cover the merged
  runtime path. Validation: 822 workspace tests and two doctests passed, with
  strict model/fixture/runtime checks enabled for the workspace suite.

- **Convolution uniform buffers are cached by contents instead of rebuilt for
  every dispatch.** On the RTX 4090, `detect_image` fell from **4.79 ms to
  3.73 ms (22%)**, with 823 tests passing, including concurrent inference and
  the ONNX parity chain.

  An encode probe put creation of the 53 tiny uniforms at 0.445 ms (26% of
  encode/finish/submit time), versus 0.107 ms (6%) for bind groups. The cache
  shares immutable buffers across concurrent encodes; distinct configurations
  cannot collide because the key contains the complete uniform contents.
  YuNet uses fewer than twenty distinct configurations, but callers sweeping
  arbitrary shapes through one pipeline can grow the cache without bound.

  Bind groups still refer to pooled buffers that change between runs, so
  caching them would require a larger change for a smaller measured cost.
  The eight pool/add/resize dispatches retain their existing uniforms.
  `encoder.finish()` was the largest measured component at 0.981 ms, which
  prompted the compute-pass experiment recorded above. These latency figures
  do not establish a batch-throughputput gain.

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

[Unreleased]: https://github.com/gregorycarnegie/face-crop-studio/compare/v2.0.1...HEAD
[2.0.1]: https://github.com/gregorycarnegie/face-crop-studio/compare/v2.0.0...v2.0.1
[2.0.0]: https://github.com/gregorycarnegie/face-crop-studio/compare/v1.8.0...v2.0.0
[1.8.0]: https://github.com/gregorycarnegie/face-crop-studio/compare/v1.7.0...v1.8.0
[1.7.0]: https://github.com/gregorycarnegie/face-crop-studio/compare/v1.6.0...v1.7.0
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
