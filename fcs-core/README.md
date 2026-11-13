# fcs-core

Face detection and cropping core for Face Crop Studio.

- SCRFD face detection via `FaceDetector` (WGSL GPU kernels when available, otherwise a built-in CPU graph; no external inference runtime needed).
- Pre/post-processing, NMS, eye refinement, and cropping presets (`crop_face_from_image`, `CropSettings`).
- The ONNX model file is supplied by the caller.

## Example

```rust
use fcs_core::FaceDetector;
let detector = FaceDetector::load_from("models/scrfd80k_500m_640.onnx");
```

See the crate-level docs (`cargo doc -p fcs-core --open`) for a full detect-and-crop example.

## Testing

```sh
cargo test -p fcs-core
```