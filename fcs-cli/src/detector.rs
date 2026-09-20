//! Detector construction for the CLI.

use anyhow::{Context, Result};
use fcs_core::FaceDetector;
use fcs_utils::config::DetectionSettings;
use log::info;

/// Build the detector the CLI will use.
///
/// There is one detector now, and it brings its own engine selection -- ONNX Runtime, the WGSL
/// kernels, or the built-in CPU graph -- so the GPU settings that used to choose YuNet's
/// backend no longer have anything to choose. The whole of `settings` is passed through:
/// `confidence` is on this detector's own scale (see `fcs_utils::config::DEFAULT_CONFIDENCE`),
/// and `--nms-threshold` and `--top-k` reach it too.
pub fn build_cli_detector(
    model_path: &std::path::Path,
    settings: &DetectionSettings,
) -> Result<FaceDetector> {
    let detector = FaceDetector::load_from(model_path)
        .map(|detector| detector.with_settings(settings))
        .with_context(|| format!("no usable detector model at {}", model_path.display()))?;
    // Reported once, after selection: the hardware alone does not say which engine won.
    info!(
        "Detector: {} on {}",
        detector.model_name(),
        detector.inference_backend()
    );
    Ok(detector)
}
