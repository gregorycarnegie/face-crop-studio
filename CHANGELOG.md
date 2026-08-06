# Changelog

All notable changes to Face Crop Studio are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/gregorycarnegie/face-crop-studio/compare/v1.5.1...HEAD
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
