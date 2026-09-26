//! Detector construction for the CLI.

use anyhow::{Context, Result};
use fcs_core::FaceDetector;
use fcs_utils::{GpuContextOptions, config::DetectionSettings};
use log::info;

/// Build the detector the CLI will use.
///
/// There is one detector now, and it brings its own engine selection -- the WGSL
/// kernels or the built-in CPU graph -- so the GPU settings that used to choose YuNet's
/// backend no longer have anything to choose. The whole of `settings` is passed through:
/// `confidence` is on this detector's own scale (see `fcs_utils::config::DEFAULT_CONFIDENCE`),
/// and `--nms-threshold` and `--top-k` reach it too.
///
/// `gpu` is the same policy the enhancement runtime gets, so `--no-gpu` keeps detection on the
/// CPU graph as well. It used to open its own context with default options.
pub fn build_cli_detector(
    model_path: &std::path::Path,
    settings: &DetectionSettings,
    gpu: &GpuContextOptions,
) -> Result<FaceDetector> {
    let detector = FaceDetector::load_from_with_gpu(model_path, gpu)
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
