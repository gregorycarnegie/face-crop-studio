# Changelog

All notable changes to Face Crop Studio are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/gregorycarnegie/face-crop-studio/compare/v1.4.4...HEAD
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
