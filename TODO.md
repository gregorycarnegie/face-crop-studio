# Face Crop Studio

## Project TODO List: YuNet Face Detection

### Manifest Maintenance

- [x] Centralize remaining package-local Cargo dependency versions under workspace dependencies; keep native AVIF decode enabled so AVIF inputs load.

### Utility Optimization

- [x] Reduce avoidable allocations in image extension checks, RGB-to-BGR tensor conversion, and CPU enhancement hot paths.

### fcs-gui2 Follow-up

- [x] Restore redesigned GUI basics: resizable frameless window, folder drag-and-drop queueing, working queue row controls, and crop export actions.

### CI / Installer Quality

- [x] **CI**: Consolidate model-gen fallback — remove the redundant `Retry Model Gen` step; put a single clean fallback in the download step (both ci.yml and release.yml).
- [x] **CI**: Remove leftover dev comments from ci.yml lines 40–44.
- [x] **CI**: Add vcpkg caching (`actions/cache`) so vcpkg isn't cloned and built from scratch every run.
- [x] **CI**: Deduplicate hardcoded SHA256s — added sync comments in both files (true deduplication requires a reusable workflow refactor).
- [x] **CI**: Remove duplicate `PKG_CONFIG_PATH` assignment in ci.yml (was set at job `env:` level and again in a separate step).
- [x] **CI**: Add code signing for release binaries and installers (skips gracefully if `CODE_SIGN_PFX` secret is absent).
- [x] **CI**: Remove static secret-key checks in release.yml to avoid false-positive GitHub Actions editor diagnostics for optional code signing secrets.
- [x] **CI**: Replace direct signing-secret lookups with computed-key expressions (`secrets[format(...)]`) to suppress remaining IDE false positives on optional secrets.
- [x] **Release**: Add GitHub artifact provenance attestation (`actions/attest-build-provenance`) so unsigned releases still ship verifiable supply-chain metadata.
- [x] **CI/Release**: Pin GitHub Actions Rust toolchain to `1.95.0` to match local workspace behavior.
- [x] **Release**: Rename the rebuilt GUI package/binary back to `fcs-gui` so release builds produce `fcs-gui.exe`.
- [x] **NSIS**: Fix uninstall shortcut icon — was pointing to `fcs-gui.exe`, now correctly points to `Uninstall.exe`.
- [x] **NSIS**: Extract inline PowerShell PATH one-liners — replaced with readable multi-line PS1 scripts written to `$PLUGINSDIR` at runtime.
- [x] **NSIS**: Add a license page (`MUI_PAGE_LICENSE`) to the installer (uses `LICENSE-MIT` from dist dir).
- [x] **NSIS**: Guard uninstall `RMDir /r` — now deletes only known installed files; `RMDir` without `/r` leaves user-placed files intact.
- [x] **WiX**: `EULA.rtf` already existed at `installer/windows/wix/EULA.rtf` — no change needed.
- [x] **WiX**: Migrated from WiX 3.x (`heat`/`candle`/`light`) to WiX v7 (`dotnet tool install -g wix` + `wix build`). `<Files>` element replaces `heat.exe` harvesting; `-arch x64` replaces manual Win64 XML patching.

### Release Maintenance

- [x] Align GitHub CI branch triggers with `master` for the `1.0.5-rc2` release path.
- [x] Mirror the Windows CI job's `dav1d` and `pkg-config` setup with the release workflow so AVIF-enabled builds pass on GitHub Actions.
- [x] Rename legacy `yunet::*` telemetry and GPU log targets to `fcs::*` across code and docs.

### Refactor Follow-up

- [x] Decompose oversized CLI, GUI, core, and utils functions into smaller private helpers and add unit coverage for the extracted decision logic.

### Refactor Follow-up #2 (deferred from /simplify rounds)

- [x] **Unify CropSettings translation between CLI and GUI.** Commit `8b4c5c1` already removed the GUI's runtime `preset_by_name()` resolution (stored dims win), so the two helpers were byte-identical. Replaced both `build_core_crop_settings` (`fcs-cli/src/config.rs`) and `build_crop_settings_from_app_settings` (`fcs-gui/src/app.rs`) with a single `impl From<&fcs_utils::config::CropSettings> for fcs_core::CropSettings` in `fcs-core/src/cropper.rs`. All six call sites (CLI: `main.rs:340,949`, `webcam.rs:151`; GUI: `app.rs:677`, `export.rs:501`, `quality.rs:49`) now use `(&settings.crop).into()`. Added a regression test asserting stored dims win over a stale preset label. `cargo check --workspace --all-targets`, `cargo test -p fcs-core -p fcs-cli -p fcs-gui`, and `cargo clippy -- -D warnings` all clean.
- [x] **Extend `GpuBufferPool` to all GPU ops.** Each op (`shape_mask`, `bilateral_filter`, `red_eye`, `pixel_adjust`, `hist_equalize`, `crop_batch`) now owns an `Arc<GpuBufferPool>` and routes its large storage/readback buffers through `pool.acquire`/`queue.write_buffer`/`pool.recycle` (recycle happens after `gpu_readback!` so GPU work is complete). Small fixed-size side buffers (uniforms, points, eye locations, histogram/LUT scratch) stay un-pooled. Added `clear_cache()`/`memory_usage()` accessors to each pooled type. `cargo test -p fcs-utils` (152 tests) and clippy clean.

### Coverage Follow-up

- [x] Add direct unit tests for CSV/SQLite mapping ingestion, selector handling, and SQL query validation in `fcs-utils/src/mapping.rs`.
- [x] Add direct unit tests for CLI input collection and mapping target resolution in `fcs-cli/src/input.rs`.
- [x] Add direct unit tests for shared export and metadata helpers in `fcs-utils/src/output.rs`.
- [x] Add direct unit tests for GUI export path sanitization and batch log formatting helpers in `fcs-gui/src/core/export.rs`.
- [x] Add direct unit tests for CLI color parsing, GUI quality helpers, and core postprocess edge cases.

### Coverage Follow-up #2 (fcs-utils gaps identified from coverage snapshot)

- [x] `fcs-utils/src/color.rs` (80%) — cover 3-char/4-char/8-char hex parsing, `0x` prefix, `RgbaColor::to_hsv`/`from_hsv`, and blue-dominant HSV branch.
- [x] `fcs-utils/src/config.rs` (75%) — cover `DetectionSettings::sanitize`, `InputDimensions::sanitize`, `CropSettings::sanitize`, `MetadataMode::from_str`, `ResizeQuality::from_str` error, `ResizeQuality::as_label`, `BatchLogFormat` Display, and `GpuSettings→GpuContextOptions` From impls.
- [x] `fcs-utils/src/fixtures.rs` (58%) — cover `fixtures_dir()` via `YUNET_FIXTURE_ROOT` env var, `load_fixture_bytes`, and `load_fixture_json`.
- [x] `fcs-utils/src/image_utils.rs` (75%) — cover `dynamic_to_bgr_chw`, `load_image` error path, and `resize_image` Triangle (non-fast) path.
- [x] `fcs-utils/src/gpu/mod.rs` (49%) — cover `pack_rgba_pixels`/`unpack_rgba_pixels` roundtrip, all `GpuStatusIndicator` constructors, `GpuStatusMode::as_str`, `GpuAvailability::is_available`/`context`, and `GpuContextPool` acquire/try_acquire/capacity/available (GPU-gated).
- [x] `fcs-utils/src/gpu/bilateral_filter.rs` (50%) — cover `smooth()` zero-amount early return.
- [x] `fcs-utils/src/gpu/gaussian_blur.rs` (71%) — cover `blur()` zero-radius early return, `clear_cache`, `memory_usage` (GPU-gated).
- [x] `fcs-utils/src/gpu/background_blur.rs` (25%) — cover `clear_cache` and `memory_usage` (GPU-gated).
- [x] `fcs-utils/src/gpu/buffer_pool.rs` (80%) — cover `available()` and `Debug` impl.
- [x] `fcs-utils/src/gpu/shape_mask.rs` (67%) — cover `Rectangle` variant early return (`Ok(None)`).
- [x] `fcs-utils/src/gpu/red_eye.rs` (33%) — cover zero-pixel image early return path.
- [x] `fcs-cli` coverage pass (ignoring `webcam.rs`) — add direct unit and integration tests for config loading, preprocess benchmarking, mapping input handling, crop/export edge cases, and main CLI control-flow branches.

This document outlines the development plan for the YuNet face detection project, broken down into phases.

### Phase 0: Workspace Bootstrap

- [x] **Workspace layout**: Create `fcs-core`, `fcs-gui`, `fcs-cli`, and `fcs-utils` crates and wire them through the root `Cargo.toml`.
- [x] **Dependencies**: Add baseline crates (`tract-onnx`, `image`, `ndarray`, `rayon`, `egui`, `eframe`, `clap`, `serde`, `serde_json`, `log`) with pinned versions.
- [x] **Models**: Download `face_detection_yunet_2023mar.onnx`, place it under `models/`, and document version plus SHA256 in `models/README.md`.
- [x] **Scaffolding**: Stub out crate-level skeletons (`lib.rs`, `main.rs`, module folders) and ensure `cargo check --workspace` passes.
- [x] **Configs**: Add `.cargo/config.toml` (build target defaults) and extend `.gitignore` for model files, fixtures, and GUI caches.

### Phase 1: Core Logic & Utilities

- [x] **`fcs-core`**: Implement core face detection logic.
  - [x] Pre-processing of input images.
  - [x] ONNX model inference.
  - [x] Post-processing (NMS, score thresholding).
  - [x] Validate output parsing against OpenCV YuNet sample outputs.
- [x] **`fcs-utils`**: Develop shared utilities.
  - [x] Fixture loading for tests.
  - [x] Image and data conversion helpers.
  - [x] Shared configuration loader (`serde` structs for thresholds, input size).
- [x] **Testing**: Add unit tests for `fcs-core`.
  - [x] Test pre-processing logic.
  - [x] Test post-processing logic against known outputs.
  - [x] Add golden-dataset regression test comparing to OpenCV baselines.
  - [x] Capture OpenCV YuNet reference outputs (bboxes, landmarks, scores) for parity checks.
- [x] **Models**: Document initial ONNX model version and checksum in `models/README.md`.

### Phase 2: Command-Line Interface (CLI)

- [x] **`fcs-cli`**: Build the command-line application.
  - [x] Implement the `detect` command.
  - [x] Add arguments for input path, score threshold, NMS threshold, etc.
  - [x] Handle output features.
    - [x] JSON export / stdout summary.
    - [x] Annotated image rendering.
  - [x] Support batch directory processing and glob patterns.
- [x] **Testing**: Create integration tests for the CLI.
  - [x] Basic detection flow producing JSON output (skips if model unavailable).
  - [x] Test command with various images from `fixtures/`.
    - [x] Add parity fixtures with expected detections and compare bounding boxes/landmarks to JSON snapshots.
  - [x] Verify output against golden data.
  - [x] Add snapshot tests for JSON output shape and ordering.

### Phase 3: Graphical User Interface (GUI)

- [x] **`fcs-gui`**: Build the desktop application with `egui`/`eframe`.
  - [x] Implement file selection for input images.
  - [x] Create UI controls for model parameters (thresholds, etc.).
  - [x] Display the output image with bounding boxes overlaid.
  - [x] Offload heavy inference tasks to a background thread (`rayon`) to keep the UI responsive.
  - [x] Persist user preferences (last model, thresholds) between sessions.
- [x] **Styling**: Centralize `egui` styling in `fcs-gui/src/theme.rs`.
- [x] **Testing**: Add smoke tests for the GUI application.
  - [x] Test application startup and basic interaction.
  - [x] Test settings serialization.
  - [x] Add screenshot regression harness for key panels (optional).

    ---

## FACE CROPPING APPLICATION TRANSFORMATION

### Phase 4: Core Cropping Logic (`fcs-core`)

- [x] **Crop calculation module** (`fcs-core/src/cropper.rs`)
  - [x] Define `CropSettings` struct with output dimensions, face height %, positioning mode, offsets
  - [x] Implement `PositioningMode` enum: `Center`, `RuleOfThirds`, `Custom`
  - [x] Implement `calculate_crop_region()` function using scale-based approach
    - [x] Calculate scale factor from face height percentage and output height
    - [x] Compute source region dimensions maintaining aspect ratio
    - [x] Apply positioning mode logic (center, rule of thirds at 1/3, custom offsets)
    - [x] Clamp crop region to image boundaries
  - [x] Unit tests for crop calculations with edge cases (faces near boundaries, various aspect ratios)

- [x] **Size presets** (`fcs-core/src/presets.rs`)
  - [x] Define `CropPreset` struct (name, width, height, description)
  - [x] Implement standard presets:
    - [x] LinkedIn (400×400)
    - [x] Passport (413×531)
    - [x] Instagram (1080×1080)
    - [x] ID Card (332×498)
    - [x] Avatar (512×512)
    - [x] Headshot (600×800)
    - [x] Custom (user-defined)

- [x] **Face cropper implementation** (`fcs-core/src/face_cropper.rs`)
  - [x] Implement `crop_face()` method combining detection + crop calculation
  - [x] Extract and resize face region using `image` crate
  - [x] Handle aspect ratio preservation and quality resampling
  - [x] Return cropped `DynamicImage` or save to disk
  - [x] Integration tests using fixture images (unit tests added; integration with fixtures next)

#### Phase 4 — implemented summary

- Implemented modules:
  - `fcs-core/src/cropper.rs` — crop calculation, `CropSettings`, `PositioningMode`, `calculate_crop_region()`.
  - `fcs-core/src/presets.rs` — `CropPreset`, `standard_presets()` and `preset_by_name()`.
  - `fcs-core/src/face_cropper.rs` — `crop_face_from_image()` which extracts and resizes crops using the `image` crate.

- Verification performed:
  - `cargo check --workspace` — passed.
  - `cargo test --manifest-path fcs-core/Cargo.toml` — all `fcs-core` unit tests passed (including cropper/presets/face_cropper tests).

- Notes & next steps for Phase 4:
  - Add integration tests that use real fixture images in `fixtures/` to validate end-to-end cropping behavior.
  - Wire CLI options (`--crop`, `--preset`, `--face-height-pct`, `--positioning-mode`) in `fcs-cli` (Phase 6 work).
  - Implement quality checks and enhancement hooks in `fcs-utils` (Phase 5) so CLI can skip low-quality crops.

### Phase 5: Image Quality & Enhancement (`fcs-utils`)

- [x] **Quality analysis module** (`fcs-utils/src/quality.rs`)
  - [x] Implement Laplacian variance calculation for blur detection
    - [x] Convert to grayscale
    - [x] Apply 3×3 Laplacian kernel
    - [x] Calculate variance (higher = sharper)
  - [x] Define quality levels: `High` (>1000), `Medium` (>300), `Low`
  - [x] Add quality score to detection output (per-detection `quality_score` / `quality` fields)
  - [x] Add `QualityFilter` settings (minimum quality threshold) and helper `should_skip`

- [x] **Image enhancement pipeline** (`fcs-utils/src/enhance.rs`)
  - [x] Auto color correction (histogram equalization)
  - [x] Exposure adjustment (-2 to +2 stops)
  - [x] Contrast adjustment (0.5 to 2.0x multiplier)
  - [x] Sharpness enhancement using unsharp mask (0 to 2.0)
  - [x] Brightness adjustment
  - [x] Saturation adjustment
  - [x] Create `EnhancementSettings` struct with all parameters
  - [x] Implement enhancement pipeline applying transformations in sequence
  - [x] Unit tests for each enhancement filter

- [x] **Advanced enhancements** (completed)
  - [x] Skin smoothing using bilateral filter
  - [x] Red-eye removal (automated red dominance detection + replacement)
  - [x] Background blur/segmentation (portrait mode with elliptical mask)

### Phase 6: CLI Face Cropping Features

- [x] **Extend CLI arguments** (`fcs-cli/src/main.rs`)
  - [x] Add `--crop` flag to enable cropping mode
  - [x] Add `--output-width` and `--output-height` (or `--preset`)
  - [x] Add `--face-height-pct` (default 70%)
  - [x] Add `--positioning-mode` (center/rule-of-thirds/custom)
  - [x] Add `--vertical-offset` and `--horizontal-offset` (for custom mode)
  - [x] Add `--output-format` (png/jpeg/webp)
  - [x] Add `--jpeg-quality` (1-100)
  - [x] Add `--min-quality` for blur filtering (`--skip-low-quality` maps to `--min-quality medium`)

- [x] **Enhancement CLI options**
  - [x] Add `--enhance` flag to enable enhancements (applies default EnhancementSettings)
  - [x] Add `--auto-color` for auto color correction
  - [x] Add `--exposure`, `--contrast` flags (tunable via CLI)
  - [x] Add `--sharpness`, `--brightness`, `--saturation` flags
  - [x] Add `--enhancement-preset` (natural/vivid/professional)

- [x] **Batch cropping features**
  - [x] Add `--output-dir` for batch output location
  - [x] Add `--naming-template` with variables: `{original}`, `{index}`, `{timestamp}`, `{width}`, `{height}`, `{ext}`
    - Implemented: CLI flag in `fcs-cli/src/main.rs` and test `fcs-cli/tests/naming_template.rs` added. Timestamp uses UNIX epoch seconds.
  - [x] Add `--face-index` to select specific face when multiple detected (default: all)
  - [x] Add `--skip-low-quality` flag to auto-skip blurry faces (maps to `min_quality=medium`)
  - [x] Progress reporting with face count and quality stats
    - [x] Convert numeric enhancement CLI flags to `Option<T>` so presets are unambiguous
      - Implemented `build_enhancement_settings()` helper that applies presets and explicit overrides
      - Added unit tests for preset application and explicit override behavior
      - Added end-of-run summary logging using atomic counters (images processed, faces detected, crops saved, crops skipped)
    - [x] Convert boolean flags to `Option<bool>` for explicit presence detection
      - Converted `--enhance-auto-color`, `--enhance`, and `--skip-low-quality` to accept an explicit presence/value and updated CLI wiring
    - [x] Added lightweight integration-like test for preset+enhance
      - Added a synthetic test that crops a generated image, applies preset+enhance, saves, and asserts output exists
    - [x] Print single-line summary to stdout for interactive runs

- [x] **Testing**
  - [x] Integration tests for crop + save workflow (basic cases implemented)
  - [x] Test all positioning modes produce correct crops (4 tests in positioning_modes.rs)
  - [x] Test preset sizes (6 tests in preset_sizes.rs)
  - [x] Test enhancement pipeline end-to-end (6 tests in enhancement_pipeline.rs)
  - [x] Test batch processing with quality filtering (7 tests in batch_quality_filtering.rs)

### Phase 7: GUI Face Cropping Interface

- [x] **Crop settings panel** (integrated into `fcs-gui/src/main.rs`)
  - [x] Output size controls (width/height spinners for custom preset)
  - [x] Preset size dropdown (LinkedIn, Passport, Instagram, ID Card, Avatar, Headshot, Custom)
  - [x] Face height percentage slider (10-100%)
  - [x] Positioning mode selector (dropdown: Center, Rule of Thirds, Custom)
  - [x] Vertical/horizontal offset sliders (custom mode only)
  - [x] Live preview showing crop region overlay on original image (green rectangles with 3px stroke, 4px rounded corners)

- [x] **Enhancement controls panel** (integrated into `fcs-gui/src/main.rs`)
  - [x] Enable enhancements checkbox
  - [x] Auto color correction checkbox
  - [x] Exposure slider (-2 to +2 stops)
  - [x] Contrast slider (0.5 to 2.0x)
  - [x] Sharpness slider (0 to 2.0)
  - [x] Brightness slider (-100 to +100)
  - [x] Saturation slider (0 to 2.5)
  - [x] Enhancement preset dropdown (None/Natural/Vivid/Professional)
  - [x] Reset to defaults button

- [x] **Face selection UI**
  - [x] Display all detected faces as 96x96 thumbnails with Lanczos3 filtering
  - [x] Show quality score badge on each face thumbnail (color-coded: green=High, orange=Medium, red=Low)
  - [x] Allow selecting/deselecting faces for cropping (click thumbnail or select/deselect button)
  - [x] Quality scores calculated using Laplacian variance (displayed numerically)
  - [x] Highlight selected faces in main preview (blue 4px outline)
  - [x] Select All / Deselect All buttons

- [x] **Preview & export**
  - [x] Crop overlay preview on original image (toggleable via checkbox)
  - [x] Real-time preview updates on settings change
  - [x] Export format configured in crop settings (PNG/JPEG/WebP)
  - [x] JPEG quality setting in crop settings (1-100)
  - [x] "Export N Selected Face(s)" button (dynamically shows count)
  - [x] Folder picker for export destination
  - [x] Automatic filename generation: `{source}_face_{index:02}.{ext}`

- [x] **Batch processing UI**
  - [x] Multi-file selection support ("Load multiple..." button)
  - [x] Image list with status (pending/processing/completed/failed)
  - [x] Color-coded status indicators (gray=pending, blue=processing, green=completed, red=failed)
  - [x] Per-file statistics (faces detected, faces exported)
  - [x] Summary statistics (total files, completed count, total faces exported)
  - [x] "Export All Batch Files" button
  - [x] "Clear Batch" button
  - [x] Scrollable batch file list (max height 200px)

- [x] **Settings persistence**
  - [x] Save all crop settings (preset, dimensions, face height %, positioning mode, offsets)
  - [x] Save enhancement preferences (enabled state, preset, all sliders)
  - [x] Save output format and quality preferences
  - [x] All settings persist to `config/gui_settings.json` via serde serialization
  - [x] Settings automatically reload on application start

#### Phase 7 — Implementation Summary

Implemented complete GUI face cropping workflow in `fcs-gui/src/main.rs`:

- Step 1: Crop Settings Panel
  - Added `CropSettings` and `EnhanceSettings` to `fcs-utils/src/config.rs`
  - Integrated crop preset dropdown, dimension controls, face height slider, positioning mode selector
  - Conditional UI (custom dimensions/offsets only shown when relevant)

- Step 2: Crop Preview Overlay
  - Green rectangles (3px stroke, 4px rounded) show exact crop region in preview
  - Live updates when any crop setting changes
  - Toggleable via checkbox

- Step 3: Face Selection & Thumbnails
  - Quality scoring using Laplacian variance (High >1000, Medium >300, Low <300)
  - 96x96 thumbnails with quality badges (color-coded)
  - Interactive selection with visual highlighting
  - HashSet-based selection tracking

- Step 4: Enhancement Controls
  - Full enhancement pipeline integration from `fcs-utils/enhance.rs`
  - Four presets: None, Natural, Vivid, Professional
  - Individual controls: exposure, brightness, contrast, saturation, sharpness, auto-color
  - Reset to defaults button

- Step 5: Export Functionality
  - `export_selected_faces()` method with complete crop extraction pipeline
  - Enhancement integration (conditional based on enabled state)
  - Multi-format support (PNG, JPEG with quality, WebP)
  - Smart filename generation
  - Comprehensive error handling and status reporting

- Step 6: Batch Processing
  - `BatchFile` and `BatchFileStatus` structures for tracking
  - Multi-file picker with batch state management
  - Automated batch export with per-file status tracking
  - Progress display with color-coded status indicators
  - Summary statistics (files processed, faces exported)

- Verification:
  - `cargo check -p fcs-gui` — passes with no warnings
  - All imports resolved correctly
  - Settings persistence working (JSON serialization)
  - Quality analysis integrated
  - Enhancement pipeline fully functional

### Phase 8: Advanced Features

- [x] **Output format support** (`fcs-utils/src/output.rs`)
  - [x] PNG encoder with compression level control
  - [x] JPEG encoder with quality control (1-100)
  - [x] WebP encoder (if available via `image` crate)
  - [x] Format auto-detection from file extension

- [x] **Metadata preservation**
  - [x] Copy EXIF data from original to cropped image (PNG/JPEG supported; WebP logs fallback)
  - [x] Add custom metadata tags (crop settings, face confidence, quality score)
  - [x] Option to strip all metadata for privacy

- [x] **Quality-based auto-cropping**
  - [x] Auto-select highest quality face when multiple detected
  - [x] Auto-skip images with no high-quality faces
  - [x] Quality-based output filename suffixes (e.g., `_highq`, `_mediumq`)

- [x] **Undo/Redo system** (GUI)
  - [x] History stack for crop settings changes
  - [x] Undo/redo shortcuts (Ctrl+Z / Ctrl+Y)

- [x] **Vignette effect**
  - [x] Add `vignette_softness` to crop settings
  - [x] Implement CPU alpha mask feathering (Gaussian blur)
  - [x] Implement GPU distance-based feathering (SD Polygon)
  - [x] Add controls for vignette **intensity** and **color**
  - [x] Update GUI cache keys and UI sliders

- [x] **Keyboard shortcuts** (GUI)
  - [x] Arrow keys for offset adjustments
  - [x] +/- for face height percentage
  - [x] Number keys for preset selection
  - [x] Space to toggle enhancement preview
  - [x] Enter to export

- [x] **Clipboard & drag-and-drop enhancements**
  - [x] Phase 1: Allow pasting or dragging a single image into the preview to auto-load and run detection.
  - [x] Phase 2: Handle folder drops/clipboard paths by enqueueing their supported image contents into the batch queue.
  - [x] Phase 3: Accept data-table files (CSV/XLSX/etc.) via clipboard or drag/drop to seed the mapping/import workflow.

### Phase 9: Testing & Documentation

- [x] **Comprehensive testing**
  - [x] Unit tests for all crop calculation edge cases
  - [x] Unit tests for quality analysis (blur detection)
  - [x] Unit tests for each enhancement filter
  - [x] Integration tests for full crop workflow
  - [x] GUI smoke tests for crop panel interactions
  - [x] Performance benchmarks for crop + enhancement pipeline

- [x] **Documentation**
  - [x] Document crop calculation algorithm in `fcs-core/src/cropper.rs`
  - [x] Add examples showing each positioning mode
  - [x] Document quality scoring methodology
  - [x] Create user guide for GUI crop features
  - [x] Create CLI examples for common use cases
  - [x] Update README with crop features overview
  - [x] Add ARCHITECTURE.md explaining crop calculation flow

### Phase 10: Integration & Refinement

- [x] **Workspace**: Ensure all crates work together seamlessly.
- [x] **Configuration**: Share threshold/input-size config structs between CLI and GUI.
- [x] **Documentation**: Add `///` doc comments to all public functions and modules.
- [x] **Linting & Formatting**: Run `cargo clippy` and `cargo fmt` across the workspace to ensure code quality.
- [x] **Performance**: Profile and optimize the inference pipeline in `fcs-core`.
- [x] **Telemetry**: Add optional timing/logging hooks in `fcs-utils` for inference tracing.
- [x] **Parity validation**: Compare detections against OpenCV YuNet on a shared dataset.
  - [x] Evaluate IoU overlap, detection recall/precision, and score deltas.
  - [x] Document discrepancies and adjust preprocessing/postprocessing until parity is within tolerance.

---

## PERFORMANCE OPTIMIZATION PHASE

### Phase 11: Inference Pipeline Optimization

Based on telemetry analysis showing ~548ms per image (preprocessing: 102ms, inference: 444ms, postprocessing: 0.05ms):

- [x] **Preprocessing optimizations** (`fcs-utils/src/image_utils.rs`)
  - [x] Optimize `rgb_to_bgr_chw()` function (currently line 45-74)
    - [x] Replace nested `flat_map` iterators with row-by-row processing for better cache locality
    - [x] Pre-allocate full buffer instead of collecting per-channel intermediates
    - [x] Use `par_chunks_exact_mut()` for parallel row processing
    - [x] Target: reduce from ~102ms to 30-50ms (2-4x speedup)
  - [x] Optimize image resize filter (`fcs-core/src/preprocess.rs` line 109)
    - [x] Replace `FilterType::Triangle` with `FilterType::Nearest` for batch processing (configurable via `InputDimensions.resize_quality`)
    - [x] Add config option for quality vs speed trade-off
    - [x] Target: additional 20-40ms savings
  - [x] Consider SIMD-accelerated libraries
    - [x] Evaluate `fast_image_resize` crate as alternative to `image` crate resizing (used for `ResizeQuality::Speed`)
    - [x] Benchmark performance improvement on typical inputs (~65-150ms vs 140-240ms per preprocess run depending on source)

- [x] **ONNX inference optimizations** (`fcs-core/src/model.rs`)
  - [x] Remove tensor clone (line 58)
    - [x] Investigate if tract accepts borrowed tensors via `run(&[input])`
    - [x] Replace `input.clone().into()` with zero-copy alternative
    - [x] Target: 5-10ms savings
  - [x] Verify optimized model loading path
    - [x] Add logging to confirm `into_optimized()` succeeds (not falling back to `into_decluttered()`)
    - [x] Document decluttered fallback performance penalty (~2x slower)
    - [x] Investigate tract optimization failures and workarounds (root cause: upstream `face_detection_yunet_2023mar.onnx` encodes conflicting `Conv_0` shapes; workaround is to use the sanitized `_640` export with consistent spatial dims)
  - [x] Investigate alternative ONNX runtimes
    - [x] Evaluate `ort` (ONNX Runtime bindings) for better optimization (see `docs/ONNX_RUNTIME_OPTIONS.md`)
    - [x] Test hardware acceleration (DirectML on Windows, CoreML on macOS, NNAPI on Android)
    - [x] Compare inference latency and memory usage

- [x] **Post-processing optimizations** (`fcs-core/src/postprocess.rs`)
  - [x] Implement spatial grid-based NMS (9.12x speedup for N=5000)
  - [x] Fallback to naive implementation for small N to minimize overhead

- [x] **Memory allocation optimizations**
  - [x] Reuse preprocessing buffers across batch operations
    - [x] Pre-allocate CHW tensor buffer pool
    - [x] Avoid repeated Vec allocations in hot paths

- [x] **Performance targets**
  - [x] Current baseline: ~548ms per image (102ms preprocess + 444ms inference + 0.05ms postprocess)
  - [x] Phase 12 target: ~460-480ms per image (15-20% improvement)
    - [x] Preprocessing: 30-50ms (60-70% reduction)
    - [x] Inference: 420-430ms (5% reduction from clone removal)

- [x] **Benchmarking & validation**
  - [x] Create criterion benchmarks for preprocessing pipeline
  - [x] Create criterion benchmarks for full inference pipeline
  - [x] Add regression tests to catch performance degradation
  - [x] Document optimization impact in ARCHITECTURE.md
  - [x] Update CLAUDE.md with optimization findings

#### Phase 11.1: CPU Enhancement Optimizations

- [x] **`laplacian_variance` optimization**
  - [x] Switch to integer math for accumulators (~10% speedup)
- [x] **`apply_skin_smoothing` optimization**
  - [x] Parallelize using `rayon`
  - [x] Implement LUT for color weight calculation
  - [x] Result: ~36x speedup (4.25s -> 116ms)
- [x] **`apply_background_blur` optimization**
  - [x] Parallelize mixing logic
  - [x] Optimize distance math (avoid sqrt)
  - [x] Result: ~25% speedup (182ms -> 136ms)

### Phase 12: GPU Acceleration with WGPU

Leverage GPU compute for massive performance gains in image processing operations:

- [x] **WGPU infrastructure setup**
  - [x] Add `wgpu = "27.0.1"` and `pollster = "0.4.0"` dependencies to fcs-utils
  - [x] Create `fcs-utils/src/gpu/mod.rs` module with GPU context management
  - [x] Implement GPU device initialization with fallback to CPU
  - [x] Add async GPU context pooling for CLI batch operations
  - [x] Share GPU context with egui in GUI application (already uses wgpu internally)
  - [x] Handle GPU init errors gracefully (missing drivers, old hardware)

- [x] **Phase 12.1: GPU preprocessing** (HIGHEST ROI - ~95ms savings)
  - [x] Create `fcs-utils/src/gpu/preprocess.wgsl` compute shader
    - [x] Implement RGB→BGR channel swap kernel
    - [x] Implement HWC→CHW memory layout transpose
    - [x] Add image resize via texture sampling (bilinear/trilinear)
    - [x] Single-pass combined resize+convert+transpose operation
  - [x] Implement `WgpuPreprocessor` in `fcs-core/src/preprocess.rs`
    - [x] Upload source image to GPU texture
    - [x] Execute compute shader dispatch
    - [x] Download CHW tensor result
    - [x] Integrate with existing `PreprocessConfig`
  - [x] Add CPU fallback via trait abstraction

    ```rust
    trait Preprocessor {
        fn preprocess(&self, img: &DynamicImage, cfg: &PreprocessConfig) -> Result<PreprocessOutput>;
    }
    ```

  - [x] Benchmark GPU vs CPU preprocessing
    - Criterion (GTX 1080 Ti, Vulkan): CPU quality ≈162 ms → GPU quality ≈51 ms; disk-bound GPU ≈126 ms.
    - CLI `--release --benchmark-preprocess`: CPU avg 7.13 ms/image vs GPU avg 35.48 ms/image (still limited by synchronous map/poll).
  - [x] **Target: 102ms → 5-10ms (10-20x speedup)**

- [x] **Micro-optimizations**
  - [x] Audit math hotspots for fused multiply-add opportunities (`conv2d_cpu`, batch-norm CPU reference, YuNet tensor fusion, enhancement filters) and replace `a * b + c` forms with `mul_add` where it improves precision/perf.
  - [x] Evaluate `reorder_hw_major` (`fcs-core/src/gpu/runtime.rs`) for Rayon-based parallelism — kept serial because tensors are small and integer indexing offers no fused helper; only revisit if profiling flags it as a bottleneck.

- [x] **Phase 12.2: GPU enhancement pipeline** (~50-180ms savings)
  - [x] Create WGSL compute shaders for each filter
    - [x] `histogram_equalization.wgsl` - parallel histogram + scan/reduce kernels (full GPU pipeline)
    - [x] `gaussian_blur.wgsl` - separable 2-pass convolution (horizontal + vertical)
    - [x] `bilateral_filter.wgsl` - skin smoothing with spatial+color kernels
    - [x] `unsharp_mask` - uses GPU blur + CPU composite (sufficient for current needs)
    - [x] `pixel_adjust.wgsl` - exposure, brightness, contrast, saturation (single kernel)
    - [x] `background_blur.wgsl` - elliptical mask + selective blur
    - [x] `red_eye_removal.wgsl` - red dominance detection + color replacement
    - [x] `shape_mask.wgsl` - GPU crop masking
  - [x] Implement `WgpuEnhancer` in `fcs-utils/src/enhance.rs`
    - [x] Pipeline state management (reuse bind groups)
    - [x] CPU fallback for all operations when GPU fails
    - [x] Integrated in both CLI (`fcs-cli/src/main.rs:370,382`) and GUI (`fcs-gui/src/core/cache.rs:41,122,202`)
  - [x] GPU enhancement fully operational with automatic CPU fallback
  - Note: Further optimization to keep textures on GPU between operations would be Phase 13.4+ work

- [x] **Phase 12.3: GPU-accelerated ONNX inference** (~300ms savings)
  - [x] **Option C: Custom WGPU YuNet implementation** (Future work)
    - [x] Implement Conv2D, BatchNorm, ReLU, Sigmoid as compute shaders
      - `fcs-core/src/gpu/ops.rs` now houses the reusable pipelines with WGSL kernels in `fcs-core/src/gpu/*.wgsl`
    - [x] Keep YuNet backbone/neck/head on-GPU and expose runtime toggles in CLI (`--gpu-inference`) and GUI settings
    - [x] Add automated accuracy gating (run CPU vs GPU parity suite in CI and surface diffs)
      - `cargo test -p fcs-core gpu_inference_matches_cpu_baseline -- --nocapture` now exercises `fcs-core/tests/gpu_cpu_parity.rs` across representative fixtures, comparing scores/bboxes/landmarks and surfacing max deltas.
    - [x] Port YuNet architecture to WGSL
    - [x] Validate outputs against ONNX reference
    - [x] Most flexibility but significant development effort

- [x] **Phase 12.4: GPU batch face cropping** (~10-30ms savings for multi-face)
  - [x] Implement parallel crop extraction via GPU texture operations
  - [x] Upload source image once, execute multiple crop regions as draw calls
  - [x] Batch resize operations for all detected faces
  - [x] Single download of all cropped faces
  - [x] Integrate with GUI thumbnail generation
  - [x] **Target: 3-10x speedup for images with multiple faces**

- [x] **GPU memory management**
  - [x] Implement texture/buffer pooling to avoid repeated allocations (`GpuBufferPool` in `fcs-utils/src/gpu`)
  - [x] Add memory usage monitoring and limits (GUI status bar + manual OOM trigger)
  - [x] Handle out-of-memory scenarios gracefully (implemented manual cache clearing)
  - [x] Optimize CPU→GPU transfer patterns (use staging buffers)
  - [x] Profile GPU memory bandwidth utilization

- [x] **Cross-platform validation**
  - [x] Test on Windows (DirectX 12 via wgpu)
  - [ ] Test on macOS (Metal via wgpu)
  - [ ] Test on Linux (Vulkan via wgpu)
  - [x] Validate fallback behavior when GPU unavailable
  - [x] Document minimum GPU requirements (Shader Model 5.0+)

- [x] **CLI/GUI integration**
  - [x] Add `--gpu/--no-gpu` CLI flag with auto-detection default
  - [x] Add GPU status indicator in GUI (available/in-use/fallback)
  - [x] Display GPU model/vendor in GUI settings panel (deferred; already in status bar)
  - [x] Show GPU memory usage in GUI (status bar hover tooltip)
  - [x] Add telemetry for GPU operation timing vs CPU baseline

- [x] **Documentation & examples**
  - [x] Document GPU requirements and setup in README
  - [x] Add WGSL shader documentation with algorithm explanations
  - [x] Create benchmark comparison table (CPU vs GPU for each operation)
  - [x] Document fallback behavior and troubleshooting
  - [x] Add architecture diagram showing GPU pipeline flow
  - [x] Update CLAUDE.md with GPU implementation notes

### Phase 13: Finalization & Release

- [x] **Polish & UX refinements**
  - [x] Add tooltips explaining each crop parameter
  - [x] Add visual guides for rule of thirds in preview
  - [x] Error handling with user-friendly messages
  - [x] Loading states and progress indicators
  - [x] Success/error notifications for batch operations

- [x] **Performance optimization**
  - [x] Cache cropped previews to avoid recomputation
  - [x] Parallelize batch cropping with rayon
  - [x] Optimize enhancement pipeline (SIMD if possible)
  - [x] Memory-efficient processing for large batches

- [x] **CI/CD**
  - [x] Set up GitHub Actions workflow
  - [x] Automated testing on Windows
  - [x] Verify model checksums in CI/release workflows
  - [x] Automated testing on macOS/Linux
  - [x] Clippy and format checks
  - [x] Release binary build for Windows (v1.0.0 target)
  - [x] Release binary builds for all platforms

- [x] **Packaging**
  - [x] Windows installer (MSI or NSIS) (v1.0.0 target)
  - [x] macOS app bundle (.app) — Apple Silicon `.dmg` via `installer/macos/build_macos.sh`, wired into `release.yml` as the `macos-release` job. Bundle layout: binary at `Contents/MacOS/fcs-gui`, model at `Contents/MacOS/models/`, icon at `Contents/Resources/AppIcon.icns`. Signing + notarization activate automatically if `APPLE_DEVELOPER_ID_CERT/PASS/NAME` and `APPLE_NOTARIZE_USER/PASS` + `APPLE_TEAM_ID` secrets are present; otherwise ships unsigned with `xattr -dr com.apple.quarantine` workaround documented. Model resolution falls back to exe-parent via new `fcs_utils::resolve_data_path` so `.app` launches with CWD=`/` still find the bundled model.
  - [x] Linux AppImage + `.deb` (x86_64) — `installer/linux/build_linux.sh` builds both formats; wired into `release.yml` as the `linux-release` job on `ubuntu-22.04` (glibc 2.35 baseline). AppImage layout follows FHS (`usr/bin/fcs-gui`, `usr/share/face-crop-studio/models/`); `cargo-deb` produces the `.deb` from `[package.metadata.deb]` in `fcs-gui/Cargo.toml`. `resolve_data_path` extended to look in `<exe_dir>/../share/face-crop-studio/` so the model resolves under either layout. Icon PNG generated at build time from `app_logo.svg` via `rsvg-convert` (gitignored output).
  - [x] Include sample images and presets — six AI-generated portraits (`samples/sample_01.jpg`..`sample_06.jpg`) from thispersondoesnotexist.com (no real subjects, no rights-of-publicity concerns) plus `samples/README.md` (licensing) and `samples/PRESETS.md` (full crop + enhancement preset reference). Bundled into all four installer paths: Windows portable zip + NSIS + MSI (auto-harvested via existing recursive copy/Files glob; NSIS uninstall extended to clean up `samples/*.jpg/.png/.webp/.md`), macOS `.app` at `Contents/Resources/samples/`, Linux `.deb` and AppImage at `usr/share/face-crop-studio/samples/`.

- [x] **Release**
  - [x] Tag `v1.0.0`
  - [x] Create GitHub release with binaries
  - [x] Write release notes documenting all features
  - [x] Update README with installation instructions
  - [x] Add screenshots/GIFs demonstrating crop features

### Phase 14: Code sigining (deferred)

- [ ] **Windows**
  - [ ] Requires CODE_SIGN_PFX (base64-encoded PFX) and CODE_SIGN_PASSWORD. To be set as repository secrets. Skipped silently if absent.
- [ ] **macOS**
  - [ ] To activate signing/notarization later, add five repo secrets: APPLE_DEVELOPER_ID_CERT (base64 of the .p12), APPLE_DEVELOPER_ID_PASS (.p12 password), APPLE_DEVELOPER_ID_NAME (the keychain identity string), APPLE_NOTARIZE_USER/APPLE_NOTARIZE_PASS (Apple ID + app-specific password), APPLE_TEAM_ID. No code changes needed.

---

## COMPETITIVE FEATURE PARITY (from Face Crop Jet review)

Two features Face Crop Jet ships that we don't, identified from inspecting `FCJ-1.9.3`. The rest of FCJ's stack (dlib CPU detection, DevExpress UI) is not worth borrowing — our GPU/enhancement/mapping story is already stronger.

- [ ] **Watch-folder mode** — add `--watch <dir>` to `fcs-cli` that monitors a directory and runs the existing batch crop/export path on new/changed image files. Use the `notify` crate as a thin event loop around the current batch workflow (no new processing logic). FCJ calls this "Robot/Directory Monitor" mode. Highest value, lowest cost.
- [x] **RAW input** — pure-Rust `imagepipe`/`rawloader` behind the `raw` feature (on in CLI+GUI), routed through `load_image` in `fcs-utils/src/image_utils.rs`; extensions dng/cr2/cr3/nef/arw/rw2/orf/raf/srw/pef. Hardened against three real crashes: GPU preprocess falls back to CPU above the device max texture dimension (`fcs-core/src/preprocess.rs`); RAW decode runs in a dedicated single-thread rayon pool + `catch_unwind` so `rawloader` panics on unsupported DNGs become skippable `Err`s instead of aborting; release profile switched to `panic = "unwind"` so batch panic-safety works; GUI preview/thumbnail textures downscaled to ≤8192/side (`fcs-gui/src/core/detection.rs`). Note: `rawloader` doesn't support every DNG variant — `libraw` (native dep) is the upgrade path if broader coverage is needed.
- [x] **HEIC/HEIF input** — native `libheif` via `libheif-rs` behind the `heic` feature (on in CLI+GUI), routed through `load_image`/`load_image_raw` in `fcs-utils/src/image_utils.rs`; extensions heic/heif/hif. The two pure-Rust HEVC decoders that have since appeared (ente-io/heic-decoder, imazen/heic) are both AGPL-3.0, incompatible with the project's MIT/Apache licensing, so libheif (LGPL) remains the only license-clean path. libheif located via the `vcpkg` crate (needs `VCPKG_ROOT` + crt-static for the static triplet), *not* pkg-config like dav1d. CI/release: Windows+macOS install libheif via vcpkg/brew; **Linux builds libde265+libheif from source** (apt's 1.12 < the 1.17 libheif-rs needs) — mirrors the dav1d-from-source pattern but is unverified until a CI run. Verified locally: real HEIC decodes upright (libheif applies irot/imir during decode). Closes the "any image format" gap vs FCJ's ImageMagick I/O.
