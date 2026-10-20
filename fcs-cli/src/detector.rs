//! Detector construction for the CLI.

use anyhow::{Context, Result};
use fcs_core::FaceDetector;
use fcs_utils::{GpuContext, config::DetectionSettings};
use log::info;
use std::sync::Arc;

/// Build the detector the CLI will use.
///
/// There is one detector now, and it brings its own engine selection -- the WGSL
/// kernels or the built-in CPU graph -- so the GPU settings that used to choose YuNet's
/// backend no longer have anything to choose. The whole of `settings` is passed through:
/// `confidence` is on this detector's own scale (see `fcs_utils::config::DEFAULT_CONFIDENCE`),
/// and `--nms-threshold` and `--top-k` reach it too.
///
/// `gpu` is the CLI runtime's own context, so `--no-gpu` (no context) keeps detection on the
/// CPU graph as well. It used to open a second device on the same adapter (experiment 104).
pub fn build_cli_detector(
    model_path: &std::path::Path,
    settings: &DetectionSettings,
    gpu: Option<Arc<GpuContext>>,
    gpu_settings: &fcs_utils::config::GpuSettings,
) -> Result<FaceDetector> {
    let detector = if gpu_settings.experimental_cuda {
        FaceDetector::load_from_cuda(model_path, gpu_settings)
            .with_context(|| format!("loading the CUDA detector at {}", model_path.display()))
    } else {
        FaceDetector::load_from_with_context(model_path, gpu)
            .with_context(|| format!("no usable detector model at {}", model_path.display()))
    }?
    .with_settings(settings);
    // Reported once, after selection: the hardware alone does not say which engine won.
    info!(
        "Detector: {} on {}",
        detector.model_name(),
        detector.inference_backend()
    );
    Ok(detector)
}
