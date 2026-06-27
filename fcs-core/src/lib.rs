//! Core YuNet inference primitives.
//!
//! This crate loads the YuNet ONNX model, runs inference with `tract-onnx`,
//! and provides preprocessing and postprocessing helpers.

/// Face cropping utilities (Phase 4)
pub mod cropper;
/// High-level face detection runner.
pub mod detector;
/// Utilities to extract and resize face crops from images.
pub mod face_cropper;
/// GPU inference building blocks (Phase 13.3 Option C).
pub mod gpu;
/// ONNX model loading and execution.
pub mod model;
/// YuNet model-specific constants shared across core modules.
pub mod model_config;
/// Non-maximum suppression implementation (spatial grid + naive fallback).
mod nms;
/// Detection post-processing (NMS, score filtering).
pub mod postprocess;
/// Image pre-processing (resizing, tensor conversion).
pub mod preprocess;
/// Standard crop size presets for face crops.
pub mod presets;

pub use crate::{
    cropper::{CropRegion, CropSettings, FillColor, PositioningMode, calculate_crop_region},
    face_cropper::crop_face_from_image,
    presets::{CropPreset, preset_by_name, standard_presets},
};

pub use detector::{DetectionOutput, YuNetDetector};
pub use model::YuNetModel;
pub use postprocess::{BoundingBox, Detection, Landmark, PostprocessConfig, apply_postprocess};
pub use preprocess::{
    CpuPreprocessor, InputSize, PreprocessConfig, PreprocessOutput, Preprocessor, WgpuPreprocessor,
    preprocess_dynamic_image, preprocess_image, preprocess_image_with,
};

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
