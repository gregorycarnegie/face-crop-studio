# Architecture Overview

The Face Crop Studio workspace is split into four crates that collaborate to deliver face detection, cropping, and post-processing across CLI and GUI front-ends.

```text
Root Cargo.toml
├── fcs-core     # Detection, cropping, presets, ONNX integration
├── fcs-utils    # Shared config, quality scoring, enhancement & export helpers
├── fcs-cli      # Command-line entry point and batch automation
└── fcs-gui      # eframe/egui desktop application
```

## How the Crates Talk

Dependencies point downward: the two front-ends depend on `fcs-core` and
`fcs-utils`, `fcs-core` depends on `fcs-utils`, and `fcs-utils` is the
dependency-free foundation. Nothing depends back up toward the front-ends, so
the detection/crop/enhance logic stays UI-agnostic and is exercised by both
surfaces (and the test suite) identically.

```mermaid
graph TD
    cli["fcs-cli<br/><i>batch automation, CLI flags</i>"]
    gui["fcs-gui<br/><i>eframe/egui desktop app</i>"]
    core["fcs-core<br/><i>YuNet detection, crop geometry,<br/>GPU inference graph</i>"]
    utils["fcs-utils<br/><i>config, quality scoring,<br/>enhancement, export</i>"]
    mapping["fcs-mapping<br/><i>CSV/XLSX/Parquet/SQLite ingestion</i>"]

    cli --> core
    cli --> mapping
    cli -->|features: webcam| utils
    gui --> core
    gui --> mapping
    gui -->|features: webcam| utils
    core --> utils
```

| Crate | Depends on | Role |
|-------|------------|------|
| `fcs-utils` | — | Foundation: config structs, Laplacian quality scoring, CPU+GPU enhancement, and output encoders. |
| `fcs-mapping` | — | Tabular ingestion: reads CSV, Excel, Parquet, and SQLite tables into source/output path pairs. Depends on nothing else in the workspace. |
| `fcs-core` | `fcs-utils` | Detection and geometry: YuNet ONNX loading, preprocessing/postprocessing, `calculate_crop_region`, and the custom WGSL inference graph. |
| `fcs-cli` | `fcs-core`, `fcs-utils`, `fcs-mapping` | Synchronous batch front-end with GPU auto-detection and context pooling. |
| `fcs-gui` | `fcs-core`, `fcs-utils`, `fcs-mapping` | Desktop front-end; pushes detection/enhancement onto background Rayon tasks and shares wgpu context with eframe. |

This document focuses on the crop calculation pipeline introduced in Phase 4 and extended through Phase 9.

## Data Flow

1. **Detection** – `YuNetDetector` (in `fcs-core`) runs the ONNX model via `tract-onnx`, returning `DetectionOutput` records that contain bounding boxes, landmarks, and confidence scores.
2. **Crop Derivation** – `calculate_crop_region` in `fcs-core/src/cropper.rs` transforms a detection into a bounded `CropRegion`. The algorithm:
   - clamps the configured face-height percentage to `[1, 100]`,
   - derives a source height that will yield the requested face coverage once resized,
   - mirrors the output aspect ratio to compute the source width,
   - applies positioning logic (center, rule of thirds, custom offsets),
   - and clamps the resulting rectangle to the original image dimensions.
3. **Crop Extraction** – `crop_face_from_image` uses `image::imageops::crop_imm` followed by a Lanczos3 resize to obtain the final output dimensions. This abstraction is reused by both the CLI and GUI.
4. **Enhancement & Quality** – `fcs-utils` kicks in next:
   - `apply_enhancements` runs the optional enhancement pipeline (auto color, exposure, contrast, saturation, unsharp mask, skin smoothing, red-eye removal, and background blur).
   - `estimate_sharpness` computes Laplacian variance to classify the crop as Low/Medium/High quality. These scores inform automation like `QualityFilter::select_best_index` and filename suffixing.
5. **Export** – `save_dynamic_image` encodes the image (PNG/JPEG/WebP), optionally injects metadata (original EXIF, crop configuration, quality metrics), and writes to disk.

The CLI stitches these steps together in synchronous code paths. The GUI pushes detection and enhancement workloads onto background Rayon tasks to keep the egui frame loop responsive, caching results in `DetectionCacheEntry`s keyed by model configuration.

## YuNet Model Loading

`YuNetModel::load` first attempts to run `tract-onnx`'s `into_optimized()` pipeline, which performs operator fusion and constant folding. When tract cannot optimize the graph, the loader logs a warning and falls back to the decluttered graph via `into_decluttered()`. This mode keeps inference functional but roughly doubles end-to-end inference latency because key optimizations are skipped. Watch for the warning in CLI/GUI logs to diagnose unexpected slowdowns or incompatible ONNX exports. The upstream `face_detection_yunet_2023mar.onnx` file, for example, encodes contradictory spatial hints (Conv_0 claims its output is `1×16×160×160` even though the input tensor is `1×3×320×320`), so tract refuses to type-check the network. We ship the sanitized `face_detection_yunet_2023mar_640.onnx` export—which locks the input to `640×640`—as the default model to keep tract on the fast path.

## Crop History & Undo/Redo

The GUI maintains a circular history buffer (max 100 entries) of `CropSettings` snapshots. Interactions that affect framing—preset switches, slider changes, keyboard nudges—call `push_crop_history`, making undo/redo operations deterministic. This behaviour is covered by the smoke tests in `fcs-gui/src/main.rs`.

## Benchmarks

`fcs-core/benches/crop_enhance.rs` provides a Criterion micro-benchmark for the combined crop + enhancement pipeline. It generates a synthetic high-frequency region, runs `crop_face_from_image`, and immediately applies the enhancement stack. Use `cargo bench -p fcs-core crop_enhance` to track latency when tuning algorithms.

### GPU Pipeline Architecture

```mermaid
graph TD
    HostImage[Host Image (RAM)] -->|Upload Texture| GPU[GPU Memory]
    
    subgraph "Preprocessing (WGSL)"
        GPU -->|Texture| PreprocShader[Preprocess Shader]
        PreprocShader -->|Resize + RGB->BGR| PreprocOutput[Tensor (CHW)]
    end
    
    subgraph "Inference (Custom WGSL)"
        PreprocOutput -->|Input| Conv2D[Conv2D Layer]
        Conv2D -->|Feature Map| BatchNorm[BatchNorm]
        BatchNorm -->|Activation| ReLU[ReLU/Sigmoid]
        ReLU -->|Next Layer| Conv2D
        ReLU -->|Final| OutputTensors[Output Tensors]
    end
    
    subgraph "Postprocessing"
        OutputTensors -->|Download| CPU_Post[CPU Postprocess]
        CPU_Post -->|NMS & Decode| Detections[Detections list]
    end
    
    subgraph "Enhancement (WGSL)"
        Detections -->|Crop Region| EnhanceShader[Enhancement Shaders]
        EnhanceShader -->|Apply Filters| EnhancedTex[Enhanced Texture]
    end
    
    EnhancedTex -->|Display| GUI[GUI Preview]
    EnhancedTex -->|Download| Disk[Export to Disk]
```

### GPU vs CPU Performance

| Operation                      | CPU (Ryzen 5950X) | GPU (GTX 1080 Ti) | Speedup |
|--------------------------------|-------------------|-------------------|---------|
| Preprocessing (Quality Resize) | ~162 ms           | ~51 ms            | ~3.2x   |
| Inference (640x640)            | ~20 ms            | ~15 ms            | ~1.3x   |
| Enhancement (Full Pipeline)    | ~180 ms           | ~12 ms            | ~15x    |

*Note: GPU preprocessing includes PCIe transfer overhead which dominates for single images. Batch throughput sees higher gains.*

### GPU Inference Determinism

The custom WGSL YuNet inference path (`gpu.inference: true`) is deterministic on a given adapter and driver: two rayon batch runs over the 1239-image reference folder (RTX 4090, DX12) produce bit-identical detections, every score and box. Each shader invocation evaluates its loops in a fixed order, so there is no run-to-run float wobble to absorb.

Builds before 2026-08-29 varied on about 1% of images. The cause was not float ordering but a race in `GpuBufferPool`: buffers released while a command encoder was still being built went straight back to the shared pool, so another worker could encode into memory the first submission still referenced. `GpuBufferPool::execution_scope` fixed it, and `concurrent_inference_matches_sequential` in `fcs-core/src/gpu/tests.rs` guards it.

GPU and CPU (`tract`) inference are not bit-identical to each other. On the same folder, 791 of 1239 images differ in the low bits of a score or box, and one borderline detection appears on only one path (1130 vs 1129 detections).

#### Duplicate suppression

`apply_postprocess` runs IoU NMS, then `fcs-core::nms::dedup_close_centers`, which drops detections whose centers sit within 50% of the larger box's longest edge (8 px floor). YuNet's stride-8/16/32 anchors can place boxes around one face that overlap too little for NMS to merge. Measured on the reference folder (GPU, 2026-09-15):

| Score threshold   | Detections | Removed by dedup | NMS 0.2 vs 0.3 |
|-------------------|------------|------------------|----------------|
| 0.8 (GUI default) | 1130       | 0                | no change      |
| 0.5               | 1342       | 3                | no change      |
| 0.3               | 1593       | 34               | no change      |

Dedup matters once the confidence floor is lowered: 30 of the 34 boxes it removes at 0.3 are centred on a box that is kept. The GUI settings' `nms_threshold: 0.2` (the library default is 0.3) changed nothing at any of these thresholds, because dedup already removes what the lower threshold would.

## Testing Matrix

| Area                 | Location                               | Purpose                                         |
|----------------------|----------------------------------------|-------------------------------------------------|
| Crop edge cases      | `fcs-core/src/cropper.rs`              | Property + unit tests for clamping, aspect ratio, offsets |
| Golden crop regions  | `fcs-core/tests/golden_crop_regions.rs` | Locks exact `CropRegion` output for representative scenarios |
| Face extraction      | `fcs-core/src/face_cropper.rs`         | Ensures resize dimensions match configuration   |
| Quality scoring      | `fcs-utils/src/quality.rs`             | Threshold bucketing, filters, suffix logic      |
| Enhancement pipeline | `fcs-utils/src/enhance.rs`             | Unit + pipeline parity tests                    |
| Full crop workflow   | `fcs-core/tests/full_crop_workflow.rs` | Integration test from detection to export       |
| CLI scenarios        | `fcs-cli/tests/*.rs`                   | Snapshot, batch, naming, enhancement workflows  |
| GUI smoke tests      | `fcs-gui/src/main.rs` (test module)    | Crop adjustments, undo/redo, preset application |
| GUI visuals          | `fcs-gui/tests/screenshot.rs`          | Snapshot-based overlay verification             |

These layers give confidence that the crop calculation flow remains stable across both user interfaces while providing hooks to measure and iterate on performance.
