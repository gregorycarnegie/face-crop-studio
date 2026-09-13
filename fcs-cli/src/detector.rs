//! YuNet detector construction and configuration.

use std::{path::Path, sync::Arc};

use anyhow::Result;
use fcs_core::{
    CpuPreprocessor, PostprocessConfig, PreprocessConfig, Preprocessor, WgpuPreprocessor,
    YuNetDetector,
};
use fcs_utils::{config::GpuSettings, gpu::GpuStatusIndicator};
use log::{info, warn};

use crate::gpu::CliGpuRuntime;

/// Build the detector the CLI will use, honouring every GPU setting.
///
/// Takes the whole [`GpuSettings`] rather than a per-feature flag: `preprocessing` used to have
/// no parameter at all, so the CLI ran GPU preprocessing whenever an adapter existed no matter
/// what the setting said, while the GUI honoured it. Deriving both preferences here keeps the
/// two front ends from drifting apart again.
pub fn build_cli_detector(
    model_path: &Path,
    preprocess: &PreprocessConfig,
    postprocess: &PostprocessConfig,
    gpu_runtime: &CliGpuRuntime,
    gpu: &GpuSettings,
) -> Result<YuNetDetector> {
    let detector = select_detector(model_path, preprocess, postprocess, gpu_runtime, gpu)?;
    // Reported once, after selection: the backend depends on what is installed
    // and what the GPU offered, so the settings alone do not say which one won.
    info!("Detection backend: {}", detector.inference_backend());
    Ok(detector)
}

/// The selection itself, which returns from several places depending on what
/// initialises successfully.
fn select_detector(
    model_path: &Path,
    preprocess: &PreprocessConfig,
    postprocess: &PostprocessConfig,
    gpu_runtime: &CliGpuRuntime,
    gpu: &GpuSettings,
) -> Result<YuNetDetector> {
    // A disabled GPU leaves the runtime without a context, which the branch below already checks.
    let use_gpu_inference = gpu.inference;
    let use_gpu_preprocessing = gpu.preprocessing;

    if use_gpu_inference {
        if let Some(gpu_ctx) = gpu_runtime.context() {
            let preprocessor: Arc<dyn Preprocessor> = if use_gpu_preprocessing {
                match WgpuPreprocessor::new(gpu_ctx.clone()) {
                    Ok(pre) => {
                        info!(
                            "Using GPU preprocessing + inference on {} ({:?})",
                            gpu_ctx.adapter_info().name,
                            gpu_ctx.adapter_info().backend
                        );
                        Arc::new(pre)
                    }
                    Err(err) => {
                        warn!(
                            "GPU preprocessor initialization failed ({err}); using CPU preprocessing for GPU inference."
                        );
                        Arc::new(CpuPreprocessor)
                    }
                }
            } else {
                info!("Using CPU preprocessing + GPU inference (GPU preprocessing disabled).");
                Arc::new(CpuPreprocessor)
            };

            match YuNetDetector::with_gpu_preprocessor(
                model_path,
                preprocess.clone(),
                postprocess.clone(),
                preprocessor,
            ) {
                Ok(detector) => {
                    info!("GPU inference enabled for CLI detector.");
                    return Ok(detector);
                }
                Err(err) => {
                    warn!("GPU inference initialization failed: {err}; falling back to CPU path.");
                }
            }
        } else {
            warn!(
                "GPU inference requested but no GPU context available; falling back to CPU path."
            );
        }
    }

    if let Some(gpu_ctx) = gpu_runtime.context().filter(|_| use_gpu_preprocessing) {
        match WgpuPreprocessor::new(gpu_ctx.clone()) {
            Ok(pre) => {
                info!(
                    "Using GPU preprocessing on {} ({:?})",
                    gpu_ctx.adapter_info().name,
                    gpu_ctx.adapter_info().backend
                );
                let preprocessor: Arc<dyn Preprocessor> = Arc::new(pre);
                YuNetDetector::with_preprocessor(
                    model_path,
                    preprocess.clone(),
                    postprocess.clone(),
                    preprocessor,
                )
            }
            Err(err) => {
                let info = gpu_ctx.adapter_info();
                let status = GpuStatusIndicator::fallback(
                    format!("Failed to initialize GPU preprocessor: {err}"),
                    Some(info.name.clone()),
                    Some(format!("{:?}", info.backend)),
                );

                // Log the fallback status
                warn!("{:?}", status);
                YuNetDetector::new(model_path, preprocess.clone(), postprocess.clone())
            }
        }
    } else {
        YuNetDetector::new(model_path, preprocess.clone(), postprocess.clone())
    }
}
