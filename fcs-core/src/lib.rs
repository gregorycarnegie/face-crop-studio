//! Face detection and cropping with YuNet.
//!
//! Start with [`YuNetDetector`] to load a model once and detect faces in images.
//! Its default path uses CPU preprocessing and selects ONNX Runtime when a
//! compatible shared library is available, falling back to the built-in Rust
//! graph. [`YuNetDetector::new_gpu`] selects the WGSL GPU graph and returns an
//! error if GPU initialization fails. The ONNX model file is supplied by the caller.
//!
//! # Detect and crop
//!
//! Detection coordinates are already in the original, EXIF-oriented image's
//! pixel space. Pass them directly to [`crop_face_from_image`] without rescaling.
//! The default input size is 640 by 640; custom sizes must match the model/backend.
//!
//! ```no_run
//! use fcs_core::{CropSettings, PostprocessConfig, PreprocessConfig, YuNetDetector,
//!                crop_face_from_image};
//!
//! # fn main() -> anyhow::Result<()> {
//! let detector = YuNetDetector::new(
//!     "models/face_detection_yunet_2023mar_640.onnx",
//!     PreprocessConfig::default(),
//!     PostprocessConfig::default(),
//! )?;
//! let image = fcs_utils::load_image("portrait.jpg")?;
//! let output = detector.detect_image(&image)?;
//! for (index, face) in output.detections.iter().enumerate() {
//!     let crop = crop_face_from_image(&image, face, &CropSettings::default());
//!     crop.save(format!("face-{index}.png"))?;
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Lower-level APIs
//!
//! [`preprocess`] produces letterboxed BGR tensors in `[1, 3, height, width]`
//! order with channel values in 0..=255. [`YuNetModel::run`] decodes predictions
//! into `[N, 15]` rows in model-input coordinates. [`apply_postprocess`] filters
//! and scales these rows; manual callers must also remove the letterbox offset
//! carried by [`PreprocessOutput::fit`]. [`YuNetDetector`] handles all these steps.
//!
//! [`CropSettings`] controls crop geometry. The serializable
//! [`fcs_utils::config::CropSettings`] additionally includes export and UI options;
//! convert it with `CropSettings::from(&settings)` when sharing configuration.

#![warn(missing_docs)]

/// YuNet's architecture, shared by every backend.
#[macro_use]
pub mod yunet;
/// Pure-Rust CPU inference graph.
pub mod cpu;
/// Face cropping geometry and padding.
pub mod cropper;
/// High-level face detection runner.
pub mod detector;
/// Optional refinement of a detector's two eye points, for eye-line alignment.
pub mod eye_refiner;
/// Utilities to extract and resize face crops from images.
pub mod face_cropper;
/// GPU inference building blocks and the complete YuNet runtime.
pub mod gpu;
/// ONNX model loading and execution.
pub mod model;
/// YuNet model-specific constants shared across core modules.
pub mod model_config;
/// Non-maximum suppression implementation (spatial grid + naive fallback).
mod nms;
/// ONNX Runtime inference backend (dynamically loaded, optional at runtime).
mod ort_backend;
/// Detection post-processing (NMS, score filtering).
pub mod postprocess;
/// Image pre-processing (resizing, tensor conversion).
pub mod preprocess;
/// Standard crop size presets for face crops.
pub mod presets;
/// SCRFD, the detector trained on this project's own data (optional, ONNX Runtime only).
pub mod scrfd;
/// The f32 tensor shared by every stage.
pub mod tensor;

pub use crate::{
    cropper::{CropRegion, CropSettings, FillColor, PositioningMode, calculate_crop_region},
    face_cropper::crop_face_from_image,
    presets::{CropPreset, preset_by_name, standard_presets},
};

pub use detector::{DetectionOutput, YuNetDetector};
pub use eye_refiner::EyeRefiner;
pub use model::{InferenceBackend, YuNetModel, decode_yunet_outputs};
pub use postprocess::{BoundingBox, Detection, Landmark, PostprocessConfig, apply_postprocess};
pub use preprocess::{
    CpuPreprocessor, InputSize, PreprocessConfig, PreprocessOutput, Preprocessor, WgpuPreprocessor,
    preprocess_dynamic_image, preprocess_image, preprocess_image_with,
};
pub use scrfd::ScrfdDetector;

/// Returns the crate version for diagnostics.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_returns_non_empty_string() {
        let v = version();
        assert!(!v.is_empty(), "version should not be empty");
        // Should look like semver: start with a digit
        assert!(v.chars().next().is_some_and(|c| c.is_ascii_digit()));
    }
}
