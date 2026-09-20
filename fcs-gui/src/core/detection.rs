//! Face detection workflow — ported from fcs-gui.

use crate::types::*;
use fcs_utils::gpu::GpuStatusIndicator;
use image::DynamicImage;
use imageproc::geometric_transformations::{Border, Interpolation, rotate_about_center};

use anyhow::{Context as AnyhowContext, Result};
use fcs_core::{FaceDetector, Preprocessor, WgpuPreprocessor};
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
    Result<FaceDetector>,
    Option<fcs_core::EyeRefiner>,
) {
    // The preprocessor itself is no longer used for detection -- the detector does its own,
    // on its own conventions -- but building it is still how the GPU context and its status are
    // obtained, and the enhancement pipeline runs on that context.
    let (_preprocessor, gpu_context, gpu_status) = if let Some(shared_ctx) = shared_gpu_context {
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
            None,
        );
    };
    let model_path = resolve_data_path(configured_model_path);
    let model_path_display = model_path.display().to_string();

    // The detector, on whichever engine is available: ONNX Runtime, the WGSL kernels, or the
    // built-in CPU graph. Loaded here so it lands on the same background thread the GPU
    // preprocessor was built on (experiment 81).
    let detector_result = FaceDetector::load_from(&model_path)
        .map(|detector| detector.with_score_threshold(settings.detection.confidence))
        .with_context(|| format!("no detector model at {model_path_display}"));
    if let Ok(detector) = &detector_result {
        info!(
            "Detector: {} on {}",
            detector.model_name(),
            detector.inference_backend()
        );
    }

    // Built here so it lands on the same background thread as the detector (experiment 81):
    // it opens an ONNX session, which has no business on the UI thread. Skipped when the
    // detector failed, since nothing will produce boxes for it to refine.
    let eye_refiner = detector_result
        .is_ok()
        .then(fcs_core::EyeRefiner::load)
        .flatten();

    (gpu_status, gpu_context, detector_result, eye_refiner)
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

/// Upper bound for a preview texture side. egui rejects (panics on) any texture
/// whose width or height exceeds the GPU's max 2D texture dimension; full-resolution
/// camera RAWs routinely exceed it. The app requires a Vulkan/DX12/Metal adapter, all
/// of which guarantee at least 8192, and a preview never needs more on screen.
// ponytail: fixed 8192 floor; thread the real `max_texture_side` through if a target
// device ever reports a lower limit.
const MAX_PREVIEW_TEXTURE_SIDE: u32 = 8192;

/// Downscale `image` so neither side exceeds [`MAX_PREVIEW_TEXTURE_SIDE`], preserving
/// aspect ratio. Returns the original untouched when it already fits.
fn clamp_to_texture_limit(image: &DynamicImage) -> std::borrow::Cow<'_, DynamicImage> {
    let (w, h) = (image.width(), image.height());
    if w <= MAX_PREVIEW_TEXTURE_SIDE && h <= MAX_PREVIEW_TEXTURE_SIDE {
        return std::borrow::Cow::Borrowed(image);
    }
    let scale = MAX_PREVIEW_TEXTURE_SIDE as f32 / w.max(h) as f32;
    let nw = ((w as f32 * scale).floor() as u32).max(1);
    let nh = ((h as f32 * scale).floor() as u32).max(1);
    // Through the SIMD resize, not `DynamicImage::resize_exact`, which samples pixel by pixel
    // through `GenericImageView`. This branch only runs for images past 8192 -- camera RAWs and
    // panoramas -- so it is exactly the case where the slow path hurts most: measured at 169 ms
    // to reach 2304x3072 from 12 MP, against a few ms here (experiment 91).
    //
    // Anything carrying alpha goes through `resize_rgba_fast` so it keeps it -- `resize_image`
    // returns RGB8, which would silently flatten a LumaA8 or RGBA16 preview. Everything else
    // goes out as RGB8, which `color_image_from_dynamic` reads in place anyway. `None` means
    // fir declined the request, and the original path still has to answer for it.
    let filter = image::imageops::FilterType::Triangle;
    let resized = if image.color().has_alpha() {
        let rgba = match image {
            DynamicImage::ImageRgba8(rgba) => std::borrow::Cow::Borrowed(rgba),
            other => std::borrow::Cow::Owned(other.to_rgba8()),
        };
        fcs_utils::resize_rgba_fast(&rgba, nw, nh, filter).map(DynamicImage::ImageRgba8)
    } else {
        Some(DynamicImage::ImageRgb8(fcs_utils::resize_image(
            image, nw, nh, filter,
        )))
    };
    std::borrow::Cow::Owned(resized.unwrap_or_else(|| image.resize_exact(nw, nh, filter)))
}

/// Convert a decoded image into an egui texture image without the
/// intermediate full-image RGBA copy that `to_rgba8()` incurs: RGB8 and RGBA8
/// buffers (the common decode/webcam formats) are read in place.
///
/// Oversized images are downscaled to the texture-side limit first; detection overlays
/// stay correct because they are positioned against the original image size, not the
/// preview texture's pixel dimensions.
pub(crate) fn color_image_from_dynamic(image: &DynamicImage) -> egui::ColorImage {
    let clamped = clamp_to_texture_limit(image);
    let image = clamped.as_ref();
    let size = [image.width() as usize, image.height() as usize];
    match image {
        DynamicImage::ImageRgb8(rgb) => egui::ColorImage::from_rgb(size, rgb.as_raw()),
        DynamicImage::ImageRgba8(rgba) => {
            egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_raw())
        }
        other => {
            let rgba = other.to_rgba8();
            egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_raw())
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
    detector: Arc<FaceDetector>,
    eye_refiner: Option<&fcs_core::EyeRefiner>,
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
    let mut detection_output = detector
        .detect_image(&image)
        .with_context(|| format!("detection failed for {}", path.display()))?;
    let detect_ms = t0.elapsed().as_millis() as u64;
    // Refine here rather than at crop time: these landmarks are drawn on the canvas, carried
    // in undo snapshots and read by export, so refining once at the source keeps all three
    // showing the same eyes.
    if let Some(refiner) = eye_refiner {
        refiner.refine(&image, &mut detection_output.detections);
    }

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

    let color_image = color_image_from_dynamic(&image);

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
    detector: Arc<FaceDetector>,
    eye_refiner: Option<&fcs_core::EyeRefiner>,
    image: Arc<DynamicImage>,
    synthetic_path: PathBuf,
) -> Result<DetectionJobSuccess> {
    let t0 = Instant::now();
    let mut detection_output = detector
        .detect_image(&image)
        .context("webcam detection failed")?;
    let detect_ms = t0.elapsed().as_millis() as u64;
    // Same reasoning as `perform_detection`: one refinement at the source, shared by the
    // canvas, the snapshots and export.
    if let Some(refiner) = eye_refiner {
        refiner.refine(&image, &mut detection_output.detections);
    }

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

    let color_image = color_image_from_dynamic(&image);

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
                    let color_image = color_image_from_dynamic(&frame);
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
/// Detect on a live webcam frame, cheaply enough to do it on every frame.
///
/// Everything the one-shot path does around the detection -- preview texture, per-face
/// thumbnails, quality scoring, edit history -- is skipped. What comes back is boxes to draw
/// over a texture the frame poller has already uploaded. Quality is left at its default
/// because nothing reads it for a live overlay, and computing it would mean cropping and
/// scoring every face at frame rate.
///
/// The eye refiner is skipped for the same reason, and this is deliberate rather than an
/// oversight: it costs a forward pass per face, and nothing downstream of a live overlay reads
/// the eye line. Landmarks here are drawn, never used to level a crop. The moment the user
/// freezes a frame, `detect_webcam_faces` runs the one-shot path through
/// `perform_detection_from_image`, which does refine -- so anything that gets exported carries
/// refined eyes. A sweep for `detect_image` call sites finds six; this is the one without a
/// refiner, on purpose.
pub fn spawn_webcam_detection(
    frame_number: u32,
    image: Arc<DynamicImage>,
    detector: Arc<FaceDetector>,
    job_tx: mpsc::Sender<JobMessage>,
) {
    rayon::spawn(move || {
        let started = std::time::Instant::now();
        let detections = match detector.detect_image(&image) {
            Ok(output) => output.detections,
            Err(err) => {
                // A live overlay reports failures once, not at frame rate; the next frame
                // will try again on its own.
                warn!("Live webcam detection failed: {err:#}");
                return;
            }
        };
        let detect_ms = started.elapsed().as_secs_f64() * 1e3;
        let detections = detections
            .into_iter()
            .map(|detection| DetectionWithQuality {
                current_bbox: detection.bbox,
                original_bbox: detection.bbox,
                detection,
                quality_score: 0.0,
                quality: fcs_utils::Quality::Low,
                thumbnail: None,
                origin: DetectionOrigin::Detector,
            })
            .collect();
        let _ = job_tx.send(JobMessage::WebcamDetections {
            frame_number,
            detections,
            detect_ms,
        });
    });
}

pub fn spawn_detection_job_from_image(
    job_id: u64,
    image: Arc<DynamicImage>,
    synthetic_path: PathBuf,
    detector: Option<Arc<FaceDetector>>,
    eye_refiner: Option<Arc<fcs_core::EyeRefiner>>,
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
        let payload = match perform_detection_from_image(
            detector,
            eye_refiner.as_deref(),
            image,
            synthetic_path.clone(),
        ) {
            Ok(data) => JobMessage::DetectionFinished { job_id, data },
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
    detector: Option<Arc<FaceDetector>>,
    eye_refiner: Option<Arc<fcs_core::EyeRefiner>>,
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
        let payload = match perform_detection(
            detector,
            eye_refiner.as_deref(),
            path.clone(),
            rotation_deg,
            auto_orient_exif,
        ) {
            Ok(data) => JobMessage::DetectionFinished { job_id, data },
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

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, RgbImage};

    #[test]
    fn clamp_keeps_alpha_when_the_source_has_it() {
        // The fast path returns RGB8 for anything without alpha, so an alpha-carrying source
        // has to be routed the other way or the preview is silently flattened. LumaA8 is the
        // awkward case: it has alpha but is not RGBA8 (experiment 91).
        for src in [
            DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
                9000,
                600,
                image::Rgba([10, 20, 30, 40]),
            )),
            DynamicImage::ImageLumaA8(image::GrayAlphaImage::from_pixel(
                9000,
                600,
                image::LumaA([90, 40]),
            )),
        ] {
            let out = clamp_to_texture_limit(&src);
            assert!(
                out.color().has_alpha(),
                "alpha dropped for {:?}",
                src.color()
            );
            assert!(out.width() <= MAX_PREVIEW_TEXTURE_SIDE);
            // A uniform source stays uniform through a downscale, so the value is checkable.
            let px = out.to_rgba8();
            assert_eq!(px.get_pixel(0, 0)[3], 40, "alpha value changed");
        }
    }

    #[test]
    fn clamp_without_alpha_still_downscales() {
        let src =
            DynamicImage::ImageRgb8(RgbImage::from_pixel(9000, 600, image::Rgb([10, 20, 30])));
        let out = clamp_to_texture_limit(&src);
        assert_eq!(out.width(), MAX_PREVIEW_TEXTURE_SIDE);
        assert!(!out.color().has_alpha());
    }

    #[test]
    fn clamp_leaves_small_images_untouched() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(1024, 768));
        // Borrowed (no copy) means it fit and was returned as-is.
        assert!(matches!(
            clamp_to_texture_limit(&img),
            std::borrow::Cow::Borrowed(_)
        ));
    }

    #[test]
    fn clamp_downscales_oversized_within_limit_keeping_aspect() {
        // 6192x8256 RAW (the crashing case) — taller side drives the scale.
        let img = DynamicImage::ImageRgb8(RgbImage::new(6192, 8256));
        let clamped = clamp_to_texture_limit(&img);
        let (w, h) = (clamped.width(), clamped.height());
        assert!(w <= MAX_PREVIEW_TEXTURE_SIDE && h <= MAX_PREVIEW_TEXTURE_SIDE);
        assert_eq!(h, MAX_PREVIEW_TEXTURE_SIDE); // longest side pinned to the limit
        // Aspect ratio preserved within rounding.
        let orig = 6192.0_f32 / 8256.0;
        let got = w as f32 / h as f32;
        assert!((orig - got).abs() < 0.01);
    }
}
