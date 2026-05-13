//! Export / batch crop logic.

use crate::{
    app::build_crop_settings_from_app_settings,
    types::{App2, BatchFileStatus, JobMessage},
};

use fcs_core::{Detection, YuNetDetector, crop_face_from_image};
use fcs_utils::{
    MetadataContext, OutputOptions, append_suffix_to_filename, apply_enhancements,
    apply_shape_mask, estimate_sharpness, load_image, quality::Quality, save_dynamic_image,
};
use image::{DynamicImage, Rgba};
use log::{info, warn};
use rfd::FileDialog;
use std::{
    cmp::Ordering,
    path::{Path, PathBuf},
    sync::Arc,
};

/// Apply shape mask then composite transparent pixels onto `fill`.
/// When `fill.alpha == 0` the output keeps its alpha channel (transparent PNG/WEBP).
/// When `fill.alpha == 255` the output is fully opaque (safe for JPEG).
fn apply_shape_and_fill(
    crop: DynamicImage,
    settings: &fcs_utils::config::CropSettings,
) -> DynamicImage {
    let mut rgba = crop.to_rgba8();
    apply_shape_mask(
        &mut rgba,
        &settings.shape,
        settings.vignette_softness,
        settings.vignette_intensity,
        settings.vignette_color,
    );

    let fill = settings.fill_color;
    let bg_r = fill.red as f32;
    let bg_g = fill.green as f32;
    let bg_b = fill.blue as f32;
    let bg_a = fill.alpha as f32 / 255.0;

    for px in rgba.pixels_mut() {
        let src_a = px[3] as f32 / 255.0;
        if src_a >= 1.0 {
            continue;
        }
        let inv = 1.0 - src_a;
        let r = (px[0] as f32 * src_a + bg_r * bg_a * inv) as u8;
        let g = (px[1] as f32 * src_a + bg_g * bg_a * inv) as u8;
        let b = (px[2] as f32 * src_a + bg_b * bg_a * inv) as u8;
        let a = ((src_a + bg_a * inv) * 255.0).min(255.0) as u8;
        *px = Rgba([r, g, b, a]);
    }
    DynamicImage::ImageRgba8(rgba)
}

struct ExportCandidate {
    face_index: usize,
    quality: Quality,
    quality_score: f64,
    detection_score: f32,
}

/// Exports the currently selected preview faces after prompting for a folder.
pub fn export_selected_faces(app: &mut App2) {
    let mut selected: Vec<_> = app.selected_faces.iter().copied().collect();
    selected.sort_unstable();
    export_preview_faces(app, selected, "Export failed");
}

/// Exports a single preview face after prompting for a folder.
pub fn export_one_face(app: &mut App2, face_index: usize) {
    export_preview_faces(app, vec![face_index], "Export failed");
}

fn export_preview_faces(app: &mut App2, selected: Vec<usize>, error_title: &str) {
    if selected.is_empty() {
        app.show_error(error_title, "No faces selected for export");
        return;
    }

    let Some(source_image) = app.preview.source_image.as_ref() else {
        app.show_error(error_title, "No image loaded");
        return;
    };

    let Some(output_dir) = FileDialog::new().set_title("Export crops").pick_folder() else {
        return;
    };

    if let Err(err) = std::fs::create_dir_all(&output_dir) {
        app.show_error(
            error_title,
            format!("Failed to create output directory: {err}"),
        );
        return;
    }

    let detections = &app.preview.detections;
    let source_path = app.preview.image_path.as_deref();
    let settings = &app.settings;
    let crop_settings = app.build_crop_settings();
    let output_options = OutputOptions::from_crop_settings(&settings.crop);
    let ext = output_extension(&settings.crop.output_format);
    let source_stem = source_path
        .and_then(Path::file_stem)
        .and_then(|s| s.to_str())
        .unwrap_or("face");

    let mut exported = 0usize;
    let mut failed = 0usize;

    for face_index in selected {
        let Some(det) = detections.get(face_index) else {
            failed += 1;
            continue;
        };

        let detection_for_crop = Detection {
            bbox: det.active_bbox(),
            landmarks: det.detection.landmarks,
            score: det.detection.score,
        };
        let raw_crop =
            crop_face_from_image(source_image.as_ref(), &detection_for_crop, &crop_settings);
        let shaped = apply_shape_and_fill(raw_crop, &settings.crop);
        let crop = apply_enhancements(&shaped, &settings.enhance.to_enhancement_settings(), None);

        let mut filename = format!("{source_stem}_face_{:02}.{ext}", face_index + 1);
        if let Some(suffix) = quality_suffix(settings, det.quality) {
            filename = append_suffix_to_filename(&filename, suffix);
        }
        let output_path = output_dir.join(filename);

        let metadata_ctx = MetadataContext {
            source_path,
            crop_settings: Some(&settings.crop),
            detection_score: Some(det.detection.score),
            quality: Some(det.quality),
            quality_score: Some(det.quality_score),
        };

        match save_dynamic_image(&crop, &output_path, &output_options, &metadata_ctx) {
            Ok(()) => {
                exported += 1;
                info!(
                    "Exported face {} to {}",
                    face_index + 1,
                    output_path.display()
                );
            }
            Err(err) => {
                failed += 1;
                warn!(
                    "Failed to export face {} to {}: {err}",
                    face_index + 1,
                    output_path.display()
                );
            }
        }
    }

    if failed == 0 {
        app.show_success(format!("Exported {exported} crop(s)"));
    } else {
        app.show_error(
            format!("Exported {exported} crop(s)"),
            format!("{failed} failed"),
        );
    }
}

/// Starts batch export processing for every queued image.
pub fn start_batch_export(app: &mut App2) {
    if app.batch_files.is_empty() {
        app.show_error("Batch export failed", "No queued images");
        return;
    }

    let Some(detector) = app.detector.clone() else {
        app.show_error("Batch export failed", "No detector loaded");
        return;
    };

    let Some(output_dir) = FileDialog::new()
        .set_title("Export batch crops")
        .pick_folder()
    else {
        return;
    };

    if let Err(err) = std::fs::create_dir_all(&output_dir) {
        app.show_error(
            "Batch export failed",
            format!("Failed to create output directory: {err}"),
        );
        return;
    }

    let tasks: Vec<_> = app
        .batch_files
        .iter()
        .enumerate()
        .map(|(index, file)| (index, file.path.clone(), file.output_override.clone()))
        .collect();

    for file in &mut app.batch_files {
        file.status = BatchFileStatus::Pending;
    }

    let settings = Arc::new(app.settings.clone());
    let tx = app.job_tx.clone();
    app.is_busy = true;
    app.show_success(format!("Starting batch export of {} image(s)", tasks.len()));

    std::thread::spawn(move || {
        let mut completed = 0usize;
        let mut failed = 0usize;

        for (index, path, output_override) in tasks {
            let _ = tx.send(JobMessage::BatchProgress {
                index,
                status: BatchFileStatus::Processing,
            });

            let status = run_batch_job(
                detector.as_ref(),
                path,
                output_dir.as_path(),
                settings.as_ref(),
                output_override,
            );

            if matches!(status, BatchFileStatus::Failed { .. }) {
                failed += 1;
            } else {
                completed += 1;
            }

            let _ = tx.send(JobMessage::BatchProgress { index, status });
        }

        let _ = tx.send(JobMessage::BatchComplete { completed, failed });
    });
}

fn run_batch_job(
    detector: &YuNetDetector,
    path: PathBuf,
    output_dir: &Path,
    settings: &fcs_utils::config::AppSettings,
    output_override: Option<PathBuf>,
) -> BatchFileStatus {
    let source_image = match load_image(&path) {
        Ok(image) => image,
        Err(err) => {
            return BatchFileStatus::Failed {
                error: format!("Failed to load: {err}"),
            };
        }
    };

    let detections = match detector.detect_image(&source_image) {
        Ok(output) => output.detections,
        Err(err) => {
            return BatchFileStatus::Failed {
                error: format!("Detection failed: {err}"),
            };
        }
    };

    let faces_detected = detections.len();
    if detections.is_empty() {
        return BatchFileStatus::Completed {
            faces_detected,
            faces_exported: 0,
        };
    }

    let crop_settings = build_crop_settings_from_app_settings(settings);
    let output_options = OutputOptions::from_crop_settings(&settings.crop);
    let mut crops = Vec::with_capacity(detections.len());
    let mut candidates = Vec::with_capacity(detections.len());

    for (face_index, detection) in detections.iter().enumerate() {
        let raw = crop_face_from_image(&source_image, detection, &crop_settings);
        let shaped = apply_shape_and_fill(raw, &settings.crop);
        let crop = apply_enhancements(&shaped, &settings.enhance.to_enhancement_settings(), None);
        let (quality_score, quality) = estimate_sharpness(&crop);
        crops.push(crop);
        candidates.push(ExportCandidate {
            face_index,
            quality,
            quality_score,
            detection_score: detection.score,
        });
    }

    let Some(selected) = select_candidates(&candidates, &settings.crop.quality_rules) else {
        return BatchFileStatus::Completed {
            faces_detected,
            faces_exported: 0,
        };
    };

    let multi_face = selected.len() > 1;
    let mut faces_exported = 0usize;
    for candidate_index in selected {
        let candidate = &candidates[candidate_index];
        if settings
            .crop
            .quality_rules
            .min_quality
            .is_some_and(|min| candidate.quality < min)
        {
            continue;
        }

        let Some(crop) = crops.get(candidate.face_index) else {
            continue;
        };

        let output_path = build_output_path(
            output_dir,
            &path,
            candidate,
            settings,
            output_override.as_ref(),
            multi_face,
        );
        let metadata_ctx = MetadataContext {
            source_path: Some(path.as_path()),
            crop_settings: Some(&settings.crop),
            detection_score: Some(candidate.detection_score),
            quality: Some(candidate.quality),
            quality_score: Some(candidate.quality_score),
        };

        match save_dynamic_image(crop, &output_path, &output_options, &metadata_ctx) {
            Ok(()) => {
                faces_exported += 1;
            }
            Err(err) => {
                warn!(
                    "Failed to save face {} from {}: {err}",
                    candidate.face_index + 1,
                    path.display()
                );
            }
        }
    }

    BatchFileStatus::Completed {
        faces_detected,
        faces_exported,
    }
}

fn select_candidates(
    candidates: &[ExportCandidate],
    quality_rules: &fcs_utils::config::QualityAutomationSettings,
) -> Option<Vec<usize>> {
    let best_quality = candidates.iter().map(|c| c.quality).max();
    if quality_rules.auto_skip_no_high_quality && best_quality != Some(Quality::High) {
        return None;
    }

    let mut selected: Vec<usize> = (0..candidates.len()).collect();
    if quality_rules.auto_select_best_face && selected.len() > 1 {
        let best = selected
            .iter()
            .copied()
            .max_by(|a, b| compare_candidates(&candidates[*a], &candidates[*b]))
            .unwrap_or(0);
        selected.retain(|idx| *idx == best);
    }
    Some(selected)
}

fn compare_candidates(a: &ExportCandidate, b: &ExportCandidate) -> Ordering {
    a.quality
        .cmp(&b.quality)
        .then_with(|| {
            a.quality_score
                .partial_cmp(&b.quality_score)
                .unwrap_or(Ordering::Equal)
        })
        .then_with(|| {
            a.detection_score
                .partial_cmp(&b.detection_score)
                .unwrap_or(Ordering::Equal)
        })
}

fn build_output_path(
    output_dir: &Path,
    source_path: &Path,
    candidate: &ExportCandidate,
    settings: &fcs_utils::config::AppSettings,
    output_override: Option<&PathBuf>,
    multi_face: bool,
) -> PathBuf {
    let ext = output_extension(&settings.crop.output_format);

    if let Some(custom) = output_override {
        return resolve_override_path(output_dir, custom, &ext, candidate.face_index, multi_face);
    }

    let source_stem = source_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("face");
    let mut filename = format!("{}_face_{:02}.{ext}", source_stem, candidate.face_index + 1);
    if let Some(suffix) = quality_suffix(settings, candidate.quality) {
        filename = append_suffix_to_filename(&filename, suffix);
    }
    output_dir.join(filename)
}

fn resolve_override_path(
    output_dir: &Path,
    override_target: &Path,
    ext: &str,
    face_index: usize,
    multi_face: bool,
) -> PathBuf {
    let safe_target = sanitize_relative_path(override_target);
    let parent = safe_target.parent().unwrap_or_else(|| Path::new(""));
    let stem = safe_target
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("output");
    let stem = if multi_face {
        format!("{stem}_face{}", face_index + 1)
    } else {
        stem.to_string()
    };
    output_dir.join(parent).join(format!("{stem}.{ext}"))
}

fn sanitize_relative_path(path: &Path) -> PathBuf {
    use std::path::Component;
    path.components()
        .filter(|component| matches!(component, Component::Normal(_)))
        .collect()
}

fn quality_suffix(
    settings: &fcs_utils::config::AppSettings,
    quality: Quality,
) -> Option<&'static str> {
    if !settings.crop.quality_rules.quality_suffix {
        return None;
    }
    match quality {
        Quality::High => Some("_highq"),
        Quality::Medium => Some("_medq"),
        Quality::Low => Some("_lowq"),
    }
}

fn output_extension(output_format: &str) -> String {
    match output_format.trim().to_ascii_lowercase().as_str() {
        "" => "png".to_string(),
        "jpeg" => "jpg".to_string(),
        other => other.to_string(),
    }
}
