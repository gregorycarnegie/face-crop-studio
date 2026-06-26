# Changelog

All notable changes to Face Crop Studio are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Camera RAW input behind the `raw` Cargo feature (enabled in CLI and GUI):
  DNG, CR2, CR3, NEF, ARW, RW2, ORF, RAF, SRW, and PEF decode via the pure-Rust
  `imagepipe`/`rawloader` stack, routed through the existing image-load path so
  batch and single-image flows both accept RAW. Note: `rawloader` does not
  support every DNG variant; unsupported files are skipped rather than crashing.
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

- Updated `egui`/`eframe`/`egui_extras` to 0.35 (panel `show_inside` → `show`)
  and bumped `anyhow`, `env_logger`, `log`, and `tract-onnx` to their latest
  patch releases.
- Release builds now use `panic = "unwind"` instead of `"abort"`, so the
  per-file `catch_unwind` panic-safety in batch processing actually takes
  effect in shipped binaries (one undecodable file is skipped, not fatal).

### Fixed

- GPU preprocessing no longer crashes on images whose width or height exceeds
  the device's maximum 2D texture dimension (e.g. full-resolution RAWs); it
  falls back to CPU preprocessing above the limit.
- GUI preview/thumbnail textures are downscaled to the texture-side limit, so
  loading a very large image no longer panics on upload. Detection overlays and
  crops still use the full-resolution source.

## [1.0.0]

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

[Unreleased]: https://github.com/gregorycarnegie/face-crop-studio/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/gregorycarnegie/face-crop-studio/releases/tag/v1.0.0
