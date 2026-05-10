//! Face detection workflow — ported from fcs-gui.

use crate::types::*;
use fcs_utils::gpu::GpuStatusIndicator;
use image::DynamicImage;
use imageproc::geometric_transformations::{Interpolation, rotate_about_center};

use anyhow::{Context as AnyhowContext, Result};
use fcs_core::{
    CpuPreprocessor, PostprocessConfig, PreprocessConfig, Preprocessor, WgpuPreprocessor,
    YuNetDetector,
};
use fcs_utils::{
    GpuAvailability, GpuContext, GpuContextOptions, config::AppSettings, load_image,
    quality::estimate_sharpness,
};
use log::{error, info, warn};
use std::{
    path::PathBuf,
    sync::{Arc, mpsc},
};

pub fn build_detector(
    settings: &AppSettings,
    shared_gpu_context: Option<Arc<GpuContext>>,
) -> (
    GpuStatusIndicator,
    Option<Arc<GpuContext>>,
    Result<YuNetDetector>,
) {
    let (preprocessor, gpu_context, gpu_status) = if let Some(shared_ctx) = shared_gpu_context {
        info!("Using shared GPU context from egui renderer");
        build_preprocessor_from_context(shared_ctx, settings)
    } else {
        maybe_build_gpu_preprocessor(settings)
    };

    let Some(model_path) = settings.model_path.as_deref() else {
        return (
            gpu_status,
            gpu_context,
            Err(anyhow::anyhow!("no model path configured")),
        );
    };

    let preprocess: PreprocessConfig = settings.input.into();
    let postprocess: PostprocessConfig = (&settings.detection).into();
    let prefer_gpu_inference = settings.gpu.enabled && settings.gpu.inference;

    let build_cpu = || -> Result<YuNetDetector> {
        if let Some(pre) = &preprocessor {
            YuNetDetector::with_preprocessor(
                model_path,
                preprocess.clone(),
                postprocess.clone(),
                Arc::clone(pre),
            )
            .with_context(|| {
                format!("failed to load YuNet model with GPU preprocessing from {model_path}")
            })
        } else {
            YuNetDetector::new(model_path, preprocess.clone(), postprocess.clone())
                .with_context(|| format!("failed to load YuNet model from {model_path}"))
        }
    };

    let detector_result = if prefer_gpu_inference {
        let pre: Arc<dyn Preprocessor> = preprocessor
            .as_ref()
            .map(Arc::clone)
            .unwrap_or_else(|| Arc::new(CpuPreprocessor));
        match YuNetDetector::with_gpu_preprocessor(
            model_path,
            preprocess.clone(),
            postprocess.clone(),
            pre,
        )
        .with_context(|| format!("failed GPU YuNet from {model_path}"))
        {
            Ok(d) => {
                info!("Using GPU inference");
                Ok(d)
            }
            Err(err) => {
                warn!("GPU inference failed ({err}); falling back");
                build_cpu()
            }
        }
    } else {
        build_cpu()
    };

    (gpu_status, gpu_context, detector_result)
}

fn maybe_build_gpu_preprocessor(
    settings: &AppSettings,
) -> (
    Option<Arc<dyn Preprocessor>>,
    Option<Arc<GpuContext>>,
    GpuStatusIndicator,
) {
    if !settings.gpu.preprocessing {
        let status = GpuStatusIndicator::disabled("GPU preprocessing disabled".to_string());
        return (None, None, status);
    }
    let options: GpuContextOptions = (&settings.gpu).into();
    match GpuContext::init_with_fallback(&options) {
        GpuAvailability::Available(ctx) => build_preprocessor_from_context(ctx, settings),
        GpuAvailability::Disabled { reason } => (None, None, GpuStatusIndicator::disabled(reason)),
        GpuAvailability::Unavailable { error } => {
            (None, None, GpuStatusIndicator::error(error.to_string()))
        }
    }
}

fn build_preprocessor_from_context(
    context: Arc<GpuContext>,
    settings: &AppSettings,
) -> (
    Option<Arc<dyn Preprocessor>>,
    Option<Arc<GpuContext>>,
    GpuStatusIndicator,
) {
    if !settings.gpu.enabled || !settings.gpu.preprocessing {
        return (
            None,
            None,
            GpuStatusIndicator::disabled("Disabled by config".to_string()),
        );
    }
    let info = context.adapter_info();
    match WgpuPreprocessor::new(context.clone()) {
        Ok(pre) => {
            let status = GpuStatusIndicator::available(
                info.name.clone(),
                format!("{:?}", info.backend),
                Some(info.driver.clone()),
                Some(info.vendor),
                Some(info.device),
            );
            (Some(Arc::new(pre)), Some(context), status)
        }
        Err(err) => {
            warn!("GPU preprocessor failed: {err}");
            let status = GpuStatusIndicator::fallback(
                format!("{err}"),
                Some(info.name.clone()),
                Some(format!("{:?}", info.backend)),
            );
            (None, Some(context), status)
        }
    }
}

fn rotate_image(image: Arc<DynamicImage>, rotation_deg: f32) -> Arc<DynamicImage> {
    if rotation_deg.abs() < 0.01 {
        return image;
    }
    let rgba = image.to_rgba8();
    let rotated = rotate_about_center(
        &rgba,
        rotation_deg.to_radians(),
        Interpolation::Bilinear,
        image::Rgba([0, 0, 0, 255]),
    );
    Arc::new(DynamicImage::ImageRgba8(rotated))
}

pub fn perform_detection(
    detector: Arc<YuNetDetector>,
    path: PathBuf,
    rotation_deg: f32,
) -> Result<DetectionJobSuccess> {
    let raw =
        Arc::new(load_image(&path).with_context(|| format!("failed to load {}", path.display()))?);
    let image = rotate_image(raw, rotation_deg);
    let detection_output = detector
        .detect_image(&image)
        .with_context(|| format!("detection failed for {}", path.display()))?;

    let detections: Vec<DetectionWithQuality> = detection_output
        .detections
        .into_iter()
        .map(|det| {
            let bbox = det.bbox;
            let x = bbox.x.max(0.0) as u32;
            let y = bbox.y.max(0.0) as u32;
            let w = bbox.width.max(1.0) as u32;
            let h = bbox.height.max(1.0) as u32;
            let x = x.min(image.width().saturating_sub(1));
            let y = y.min(image.height().saturating_sub(1));
            let w = w.min(image.width().saturating_sub(x));
            let h = h.min(image.height().saturating_sub(y));
            let face = image.crop_imm(x, y, w, h);
            let (quality_score, quality) = estimate_sharpness(&face);
            DetectionWithQuality {
                detection: det,
                quality_score,
                quality,
                thumbnail: None,
                current_bbox: bbox,
                original_bbox: bbox,
                origin: DetectionOrigin::Detector,
            }
        })
        .collect();

    let rgba = image.to_rgba8();
    let size = [rgba.width() as usize, rgba.height() as usize];
    let color_image = egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_raw());

    Ok(DetectionJobSuccess {
        path,
        color_image,
        detections,
        original_size: detection_output.original_size,
        original_image: image,
    })
}

pub fn spawn_detection_job(
    job_id: u64,
    path: PathBuf,
    detector: Option<Arc<YuNetDetector>>,
    settings: AppSettings,
    rotation_deg: f32,
    job_tx: mpsc::Sender<JobMessage>,
) {
    let Some(detector) = detector else {
        let _ = job_tx.send(JobMessage::DetectionFailed {
            job_id,
            error: "No detector loaded. Configure model path in settings.".to_owned(),
        });
        return;
    };

    rayon::spawn(move || {
        let _ = settings; // used for future cache-key construction
        let payload = match perform_detection(detector, path.clone(), rotation_deg) {
            Ok(data) => {
                let cache_key = CacheKey {
                    path: path.clone(),
                    model_path: None,
                    input_width: 640,
                    input_height: 640,
                    resize_quality: Default::default(),
                    score_bits: 0,
                    nms_bits: 0,
                    top_k: 5000,
                };
                JobMessage::DetectionFinished {
                    job_id,
                    cache_key,
                    data,
                }
            }
            Err(err) => JobMessage::DetectionFailed {
                job_id,
                error: format!("{err:#}"),
            },
        };
        if job_tx.send(payload).is_err() {
            error!("GUI dropped detection result for {}", path.display());
        }
    });
}
