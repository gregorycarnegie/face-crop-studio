//! Face detection workflow — ported from fcs-gui.

use crate::types::*;
use fcs_utils::gpu::GpuStatusIndicator;
use image::DynamicImage;
use imageproc::geometric_transformations::{Border, Interpolation, rotate_about_center};

use anyhow::{Context as AnyhowContext, Result};
use fcs_core::{
    CpuPreprocessor, PostprocessConfig, PreprocessConfig, Preprocessor, WgpuPreprocessor,
    YuNetDetector,
};
use fcs_utils::{
    GpuAvailability, GpuContext, GpuContextOptions, config::AppSettings, load_image,
    load_image_raw, quality::estimate_sharpness, resolve_data_path,
};
use log::{error, info, warn};
use std::{
    path::PathBuf,
    sync::{Arc, mpsc},
    time::Instant,
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

    let Some(configured_model_path) = settings.model_path.as_deref() else {
        return (
            gpu_status,
            gpu_context,
            Err(anyhow::anyhow!("no model path configured")),
        );
    };
    let model_path = resolve_data_path(configured_model_path);
    let model_path_display = model_path.display().to_string();

    let preprocess: PreprocessConfig = settings.input.into();
    let postprocess: PostprocessConfig = (&settings.detection).into();
    let prefer_gpu_inference = settings.gpu.enabled && settings.gpu.inference;

    let build_cpu = || -> Result<YuNetDetector> {
        if let Some(pre) = &preprocessor {
            YuNetDetector::with_preprocessor(
                &model_path,
                preprocess.clone(),
                postprocess.clone(),
                Arc::clone(pre),
            )
            .with_context(|| {
                format!(
                    "failed to load YuNet model with GPU preprocessing from {model_path_display}"
                )
            })
        } else {
            YuNetDetector::new(&model_path, preprocess.clone(), postprocess.clone())
                .with_context(|| format!("failed to load YuNet model from {model_path_display}"))
        }
    };

    let detector_result = if prefer_gpu_inference {
        let pre: Arc<dyn Preprocessor> = preprocessor
            .as_ref()
            .map(Arc::clone)
            .unwrap_or_else(|| Arc::new(CpuPreprocessor));
        match YuNetDetector::with_gpu_preprocessor(
            &model_path,
            preprocess.clone(),
            postprocess.clone(),
            pre,
        )
        .with_context(|| format!("failed GPU YuNet from {model_path_display}"))
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
        Border::Constant(image::Rgba([0, 0, 0, 255])),
    );
    Arc::new(DynamicImage::ImageRgba8(rotated))
}

pub fn perform_detection(
    detector: Arc<YuNetDetector>,
    path: PathBuf,
    rotation_deg: f32,
    auto_orient_exif: bool,
) -> Result<DetectionJobSuccess> {
    let raw = if auto_orient_exif {
        load_image(&path)
    } else {
        load_image_raw(&path)
    }
    .with_context(|| format!("failed to load {}", path.display()))?;
    let image = rotate_image(Arc::new(raw), rotation_deg);
    let t0 = Instant::now();
    let detection_output = detector
        .detect_image(&image)
        .with_context(|| format!("detection failed for {}", path.display()))?;
    let detect_ms = t0.elapsed().as_millis() as u64;

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
        detect_ms,
    })
}

pub fn perform_detection_from_image(
    detector: Arc<YuNetDetector>,
    image: Arc<DynamicImage>,
    synthetic_path: PathBuf,
) -> Result<DetectionJobSuccess> {
    let t0 = Instant::now();
    let detection_output = detector
        .detect_image(&image)
        .context("webcam detection failed")?;
    let detect_ms = t0.elapsed().as_millis() as u64;

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
        path: synthetic_path,
        color_image,
        detections,
        original_size: detection_output.original_size,
        original_image: image,
        detect_ms,
    })
}

/// Spawns a background thread that continuously captures webcam frames and sends
/// `(egui::ColorImage, Arc<DynamicImage>)` pairs over `frame_tx` until `stop_flag` is set.
pub fn spawn_webcam_stream(
    device_index: u32,
    width: u32,
    height: u32,
    fps: u32,
    stop_flag: Arc<std::sync::atomic::AtomicBool>,
    frame_tx: mpsc::Sender<(egui::ColorImage, Arc<DynamicImage>)>,
) {
    std::thread::spawn(move || {
        let mut cam =
            match fcs_utils::WebcamCapture::with_device_index(device_index, width, height, fps) {
                Ok(c) => c,
                Err(e) => {
                    warn!("Failed to open webcam device {device_index}: {e}");
                    return;
                }
            };
        while !stop_flag.load(std::sync::atomic::Ordering::Relaxed) {
            match cam.capture_frame() {
                Ok(frame) => {
                    let rgba = frame.to_rgba8();
                    let size = [rgba.width() as usize, rgba.height() as usize];
                    let color_image = egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_raw());
                    let arc_frame = Arc::new(frame);
                    if frame_tx.send((color_image, arc_frame)).is_err() {
                        break;
                    }
                }
                Err(e) => {
                    warn!("Webcam frame capture error: {e}");
                    break;
                }
            }
        }
    });
}

/// Runs face detection on an already-captured `DynamicImage` in a rayon thread,
/// sending the result back through `job_tx`.
pub fn spawn_detection_job_from_image(
    job_id: u64,
    image: Arc<DynamicImage>,
    synthetic_path: PathBuf,
    detector: Option<Arc<YuNetDetector>>,
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
        let payload = match perform_detection_from_image(detector, image, synthetic_path.clone()) {
            Ok(data) => {
                let cache_key = CacheKey {
                    path: synthetic_path,
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
            error!("GUI dropped webcam detection result");
        }
    });
}

pub fn spawn_detection_job(
    job_id: u64,
    path: PathBuf,
    detector: Option<Arc<YuNetDetector>>,
    rotation_deg: f32,
    auto_orient_exif: bool,
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
        let payload =
            match perform_detection(detector, path.clone(), rotation_deg, auto_orient_exif) {
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
