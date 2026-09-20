//! Face detection and cropping with SCRFD.
//!
//! Start with [`FaceDetector`] to load the model once and detect faces in images. It picks its
//! own engine: ONNX Runtime when a compatible shared library is available, then the WGSL compute
//! kernels, then the built-in CPU graph, so detection works with no external runtime and no GPU.
//! The ONNX model file is supplied by the caller.
//!
//! # Detect and crop
//!
//! Detection coordinates are already in the original, EXIF-oriented image's
//! pixel space. Pass them directly to [`crop_face_from_image`] without rescaling.
//!
//! ```no_run
//! use fcs_core::{CropSettings, FaceDetector, crop_face_from_image};
//!
//! # fn main() -> anyhow::Result<()> {
//! // `None` when the model file is missing: there is no second detector to fall back to.
//! let detector = FaceDetector::load_from("models/scrfd80k_500m_640.onnx")
//!     .ok_or_else(|| anyhow::anyhow!("no detector model"))?;
//! let image = fcs_utils::load_image("portrait.jpg")?;
//! for (index, face) in detector.detect_image(&image)?.detections.iter().enumerate() {
//!     let crop = crop_face_from_image(&image, face, &CropSettings::default());
//!     crop.save(format!("face-{index}.png"))?;
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Lower-level APIs
//!
//! [`ScrfdDetector`] is the detector itself, without the settings plumbing; it letterboxes,
//! runs, and decodes in one call. [`scrfd::plan`] and [`scrfd::gpu`] run the network on the
//! built-in CPU graph and the WGSL kernels respectively, given weights read by name from the
//! export. [`preprocess`] letterboxes into `[1, 3, height, width]` tensors for callers driving
//! a graph themselves.
//!
//! [`CropSettings`] controls crop geometry. The serializable
//! [`fcs_utils::config::CropSettings`] additionally includes export and UI options;
//! convert it with `CropSettings::from(&settings)` when sharing configuration.

#![warn(missing_docs)]

/// Pure-Rust CPU inference building blocks.
pub mod cpu;
/// Face cropping geometry and padding.
pub mod cropper;
/// Optional refinement of a detector's two eye points, for eye-line alignment.
pub mod eye_refiner;
/// Utilities to extract and resize face crops from images.
pub mod face_cropper;
/// The detector the application uses, with its settings.
pub mod face_detector;
/// GPU inference building blocks, in WGSL compute shaders.
pub mod gpu;
/// Non-maximum suppression implementation (spatial grid + naive fallback).
mod nms;
/// Reading named float initializers out of an ONNX file.
pub mod onnx;
/// Detection types, score filtering and NMS settings.
pub mod postprocess;
/// Image pre-processing (resizing, tensor conversion).
pub mod preprocess;
/// Standard crop size presets for face crops.
pub mod presets;
/// SCRFD, the detector trained on this project's own data.
pub mod scrfd;
/// The f32 tensor shared by every stage.
pub mod tensor;

pub use crate::{
    cropper::{CropRegion, CropSettings, FillColor, PositioningMode, calculate_crop_region},
    face_cropper::crop_face_from_image,
    presets::{CropPreset, preset_by_name, standard_presets},
};

pub use eye_refiner::EyeRefiner;
pub use face_detector::{DetectionOutput, FaceDetector};
pub use postprocess::{BoundingBox, Detection, Landmark};
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
