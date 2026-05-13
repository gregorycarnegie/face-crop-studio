//! Webcam capture mode functionality.

use std::{fs, io::Write, sync::Arc};

use anyhow::{Context, Result};
use fcs_core::{YuNetDetector, crop_face_from_image};
use fcs_utils::{
    MetadataContext, OutputOptions, QualityFilter, WebcamCapture, append_suffix_to_filename,
    config::AppSettings, estimate_sharpness, list_webcam_devices, normalize_path,
    save_dynamic_image,
};
use log::{debug, info, warn};

use crate::{
    annotate::annotate_image, args::DetectArgs, config::build_core_crop_settings,
    gpu::CliGpuRuntime,
};

/// Process frames from webcam in real-time.
pub fn run_webcam_mode(
    args: &DetectArgs,
    detector: Arc<YuNetDetector>,
    settings: Arc<AppSettings>,
    gpu_runtime: Arc<CliGpuRuntime>,
    quality_filter: Arc<QualityFilter>,
    enhancement_settings: Option<Arc<fcs_utils::EnhancementSettings>>,
) -> Result<()> {
    info!(
        "Opening webcam device {} at {}x{} @ {} fps",
        args.webcam_device, args.webcam_width, args.webcam_height, args.webcam_fps
    );

    // List available devices
    match list_webcam_devices() {
        Ok(devices) => {
            info!("Available webcam devices:");
            for (idx, name) in devices {
                info!("  [{}] {}", idx, name);
            }
        }
        Err(e) => warn!("Could not enumerate webcam devices: {}", e),
    }

    let mut webcam = WebcamCapture::with_device_index(
        args.webcam_device,
        args.webcam_width,
        args.webcam_height,
        args.webcam_fps,
    )
    .context("Failed to open webcam")?;

    let (actual_width, actual_height) = webcam.resolution();
    info!(
        "Webcam opened successfully: {}x{} @ {} fps",
        actual_width,
        actual_height,
        webcam.frame_rate()
    );

    let annotate_dir = if let Some(dir) = args.annotate.as_ref() {
        fs::create_dir_all(dir)
            .with_context(|| format!("failed to create annotation directory {}", dir.display()))?;
        Some(normalize_path(dir)?)
    } else {
        None
    };

    let crop_enabled = args.crop;
    let crop_output_dir = if crop_enabled {
        if let Some(dir) = args.output_dir.as_ref() {
            fs::create_dir_all(dir)
                .with_context(|| format!("failed to create output dir {}", dir.display()))?;
            Some(normalize_path(dir)?)
        } else {
            anyhow::bail!("--crop requires --output-dir to be specified");
        }
    } else {
        None
    };

    let max_frames = args.webcam_frames;
    let continuous_mode = max_frames == 0;
    let mut frame_count = 0u32;
    let mut total_faces = 0usize;

    if continuous_mode {
        info!("Starting webcam detection loop (continuous mode - press Ctrl+C to stop)");
    } else {
        info!(
            "Starting webcam detection loop (capturing {} frames)",
            max_frames
        );
    }

    loop {
        if !continuous_mode && frame_count >= max_frames {
            break;
        }

        let frame = match webcam.capture_frame() {
            Ok(f) => f,
            Err(e) => {
                warn!("Failed to capture frame: {}", e);
                continue;
            }
        };

        frame_count += 1;

        // Run detection on the frame
        let output = match detector.detect_image(&frame) {
            Ok(out) => out,
            Err(e) => {
                warn!("Detection failed on frame {}: {}", frame_count, e);
                continue;
            }
        };

        let num_faces = output.detections.len();
        total_faces += num_faces;

        info!("Frame {}: detected {} face(s)", frame_count, num_faces);

        // Print detection results
        for (idx, det) in output.detections.iter().enumerate() {
            debug!(
                "  Face {}: score={:.3}, bbox=[{:.1}, {:.1}, {:.1}, {:.1}]",
                idx + 1,
                det.score,
                det.bbox.x,
                det.bbox.y,
                det.bbox.width,
                det.bbox.height
            );
        }

        // Save annotated frame if requested
        if let Some(dir) = annotate_dir.as_ref() {
            match tempfile::Builder::new()
                .prefix("frame_")
                .suffix(".png")
                .tempfile()
            {
                Ok(mut tmp) => {
                    let frame_path = tmp.path().to_path_buf();
                    // Encode as PNG and write to the secure temp file.
                    let png_data = {
                        let mut buf = std::io::Cursor::new(Vec::new());
                        if let Err(e) = frame.write_to(&mut buf, image::ImageFormat::Png) {
                            warn!("Failed to encode temporary frame: {}", e);
                            continue;
                        }
                        buf.into_inner()
                    };
                    if let Err(e) = tmp.write_all(&png_data) {
                        warn!("Failed to write temporary frame: {}", e);
                        continue;
                    }
                    match annotate_image(&frame_path, &output.detections, dir) {
                        Ok(path) => debug!("Saved annotated frame to {}", path.display()),
                        Err(e) => warn!("Failed to annotate frame {}: {}", frame_count, e),
                    }
                    // tmp is dropped here, which deletes the file automatically.
                }
                Err(e) => {
                    warn!("Failed to create temporary frame file: {}", e);
                }
            }
        }

        // Crop faces if requested
        if crop_enabled
            && !output.detections.is_empty()
            && let Some(out_dir) = crop_output_dir.as_ref()
        {
            let core_settings = build_core_crop_settings(&settings.crop);
            let output_options = OutputOptions::from_crop_settings(&settings.crop);

            for (idx, det) in output.detections.iter().enumerate() {
                let mut crop_img = crop_face_from_image(&frame, det, &core_settings);

                if let Some(enh) = enhancement_settings.as_ref() {
                    crop_img = gpu_runtime.enhance(&crop_img, enh);
                }

                let (quality_score, quality) = estimate_sharpness(&crop_img);

                if quality_filter.should_skip(quality) {
                    debug!(
                        "Skipping frame {} face {} due to {:?} quality",
                        frame_count,
                        idx + 1,
                        quality
                    );
                    continue;
                }

                crop_img = gpu_runtime.apply_shape_mask(
                    &crop_img,
                    &settings.crop.shape,
                    settings.crop.vignette_softness,
                    settings.crop.vignette_intensity,
                    settings.crop.vignette_color,
                );

                let mut ext = settings.crop.output_format.clone();
                if ext.is_empty() {
                    ext = "png".to_string();
                }

                let mut out_name =
                    format!("webcam_frame{:06}_face{}.{}", frame_count, idx + 1, ext);
                if let Some(suffix) = quality_filter.suffix_for(quality) {
                    out_name = append_suffix_to_filename(&out_name, suffix);
                }

                let out_path = out_dir.join(&out_name);

                let metadata_ctx = MetadataContext {
                    source_path: None,
                    crop_settings: Some(&settings.crop),
                    detection_score: Some(det.score),
                    quality: Some(quality),
                    quality_score: Some(quality_score),
                };

                match save_dynamic_image(&crop_img, &out_path, &output_options, &metadata_ctx) {
                    Ok(_) => info!("Saved crop to {}", out_path.display()),
                    Err(e) => warn!("Failed to save crop: {}", e),
                }
            }
        }
    }

    info!(
        "Webcam capture complete: {} frames processed, {} total faces detected",
        frame_count, total_faces
    );

    webcam.stop().context("Failed to stop webcam")?;
    Ok(())
}
