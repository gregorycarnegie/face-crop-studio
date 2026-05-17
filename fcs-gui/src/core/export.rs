//! Export / batch crop logic.

use crate::{
    app::build_crop_settings_from_app_settings,
    types::{App2, BatchFileStatus, JobMessage},
};

use fcs_core::{
    CropSettings as CoreCropSettings, Detection, YuNetDetector, calculate_crop_region,
    crop_face_from_image,
};
use fcs_utils::{
    ImageFormatHint, MetadataContext, OutputOptions, RedEye, append_suffix_to_filename,
    apply_enhancements, apply_shape_mask, estimate_sharpness, load_image, quality::Quality,
    save_dynamic_image,
};
use image::{DynamicImage, GenericImageView, Rgba};
use log::{error, info, warn};
use rayon::prelude::*;
use rfd::FileDialog;
use std::{
    cmp::Ordering,
    panic::{AssertUnwindSafe, catch_unwind},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering as AtomicOrdering},
    },
};

/// Map the two eye landmarks into output-crop coordinates for targeted red-eye removal.
fn eye_positions(
    detection: &Detection,
    img_w: u32,
    img_h: u32,
    crop: &CoreCropSettings,
) -> Vec<RedEye> {
    let re = &detection.landmarks[0];
    let le = &detection.landmarks[1];
    if re.x == 0.0 && re.y == 0.0 && le.x == 0.0 && le.y == 0.0 {
        return vec![];
    }
    let region = calculate_crop_region(img_w, img_h, detection.bbox, crop);
    let sx = crop.output_width as f32 / region.width.max(1) as f32;
    let sy = crop.output_height as f32 / region.height.max(1) as f32;
    let face_h_out =
        detection.bbox.height / region.height.max(1) as f32 * crop.output_height as f32;
    let radius = (face_h_out * 0.12).max(4.0);
    [re, le]
        .iter()
        .map(|lm| RedEye {
            x: (lm.x - region.x as f32) * sx,
            y: (lm.y - region.y as f32) * sy,
            radius,
            _pad: 0.0,
        })
        .collect()
}

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
    let ext = output_extension(settings.crop.output_format);
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
        let eyes = eye_positions(
            &detection_for_crop,
            source_image.width(),
            source_image.height(),
            &crop_settings,
        );
        let crop = apply_enhancements(
            &shaped,
            &settings.enhance.to_enhancement_settings(),
            if eyes.is_empty() { None } else { Some(&eyes) },
        );

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
    let parallelism = settings.resolved_batch_parallelism();
    let tx = app.job_tx.clone();
    app.is_busy = true;
    app.show_success(format!(
        "Starting batch export of {} image(s) using {} worker(s)",
        tasks.len(),
        parallelism
    ));

    std::thread::spawn(move || {
        let completed = Arc::new(AtomicUsize::new(0));
        let failed = Arc::new(AtomicUsize::new(0));

        // Dedicated pool so this batch can't starve other rayon work and so the
        // worker count stays bounded regardless of the global pool size.
        let pool = match rayon::ThreadPoolBuilder::new()
            .num_threads(parallelism)
            .thread_name(|i| format!("fcs-batch-{i}"))
            .build()
        {
            Ok(p) => p,
            Err(err) => {
                error!("Failed to build batch thread pool: {err}; running sequentially");
                run_batch_sequential(
                    tasks,
                    detector.as_ref(),
                    output_dir.as_path(),
                    settings.as_ref(),
                    &tx,
                    &completed,
                    &failed,
                );
                let _ = tx.send(JobMessage::BatchComplete {
                    completed: completed.load(AtomicOrdering::Relaxed),
                    failed: failed.load(AtomicOrdering::Relaxed),
                });
                return;
            }
        };

        // Clone everything the install closure needs. `mpsc::Sender` is `!Sync`,
        // so it must be moved (not borrowed) into the closure; we clone here so
        // the outer `tx` survives for the final `BatchComplete` send.
        let inner_tx = tx.clone();
        let inner_detector = detector.clone();
        let inner_output_dir = output_dir.clone();
        let inner_settings = settings.clone();
        let inner_completed = completed.clone();
        let inner_failed = failed.clone();

        pool.install(move || {
            // `for_each_with` clones the init once per worker thread, so the
            // sender refcount bumps once per worker rather than once per task.
            tasks.into_par_iter().for_each_with(
                inner_tx,
                |tx, (index, path, output_override)| {
                    let _ = tx.send(JobMessage::BatchProgress {
                        index,
                        status: BatchFileStatus::Processing,
                    });

                    let status = run_batch_job_panic_safe(
                        inner_detector.as_ref(),
                        path,
                        inner_output_dir.as_path(),
                        inner_settings.as_ref(),
                        output_override,
                    );

                    if matches!(status, BatchFileStatus::Failed { .. }) {
                        inner_failed.fetch_add(1, AtomicOrdering::Relaxed);
                    } else {
                        inner_completed.fetch_add(1, AtomicOrdering::Relaxed);
                    }

                    let _ = tx.send(JobMessage::BatchProgress { index, status });
                },
            );
        });

        let _ = tx.send(JobMessage::BatchComplete {
            completed: completed.load(AtomicOrdering::Relaxed),
            failed: failed.load(AtomicOrdering::Relaxed),
        });
    });
}

// Per-image detection diagnostic — commented out for normal use. Uncomment
// this function and the call site in `run_batch_job` to capture per-image
// detection counts and sorted score lists at `debug` level for diffing two
// batch runs. See ARCHITECTURE.md "GPU Inference Determinism" for usage.
//
// Output lines are prefixed `[batch-diag]` for easy grepping; format:
//   `[batch-diag] det=<n> border=<n> min=<f.4> max=<f.4> path=<...> scores=[<f.4>,...]`
//
// `compare_batch_diag.py` in the repo root parses two such files and reports
// per-image mismatches.
//
// fn log_detection_diag(path: &Path, detections: &[Detection], score_threshold: f32) {
//     if !log::log_enabled!(log::Level::Debug) {
//         return;
//     }
//
//     const BORDERLINE_MARGIN: f32 = 0.02;
//     let borderline = detections
//         .iter()
//         .filter(|d| d.score < score_threshold + BORDERLINE_MARGIN)
//         .count();
//
//     let (min_score, max_score) = detections.iter().fold((f32::MAX, f32::MIN), |(lo, hi), d| {
//         (lo.min(d.score), hi.max(d.score))
//     });
//
//     // Sort scores descending so cross-run diffs line up by rank rather than by
//     // detection order (which can shuffle even without count change).
//     let mut sorted_scores: Vec<f32> = detections.iter().map(|d| d.score).collect();
//     sorted_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
//     let scores_str = sorted_scores
//         .iter()
//         .map(|s| format!("{:.4}", s))
//         .collect::<Vec<_>>()
//         .join(",");
//
//     let (min_display, max_display) = if detections.is_empty() {
//         ("-".to_string(), "-".to_string())
//     } else {
//         (format!("{:.4}", min_score), format!("{:.4}", max_score))
//     };
//
//     debug!(
//         "[batch-diag] det={} border={} min={} max={} path={} scores=[{}]",
//         detections.len(),
//         borderline,
//         min_display,
//         max_display,
//         path.display(),
//         scores_str
//     );
// }

/// Run one batch job and convert any panic into a [`BatchFileStatus::Failed`].
/// One corrupt/edge-case image must not abort the rest of the batch.
fn run_batch_job_panic_safe(
    detector: &YuNetDetector,
    path: PathBuf,
    output_dir: &Path,
    settings: &fcs_utils::config::AppSettings,
    output_override: Option<PathBuf>,
) -> BatchFileStatus {
    let path_display = path.display().to_string();
    let result = catch_unwind(AssertUnwindSafe(move || {
        run_batch_job(detector, path, output_dir, settings, output_override)
    }));
    match result {
        Ok(status) => status,
        Err(payload) => {
            let msg = panic_payload_message(payload);
            error!("Panic processing batch image {path_display}: {msg}");
            BatchFileStatus::Failed {
                error: format!("panic: {msg}"),
            }
        }
    }
}

/// Best-effort extraction of the message from a panic payload (`&str` or `String`).
fn panic_payload_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        return (*s).to_string();
    }
    if let Some(s) = payload.downcast_ref::<String>() {
        return s.clone();
    }
    "panic with non-string payload".to_string()
}

/// Sequential fallback used only if building the rayon pool fails.
fn run_batch_sequential(
    tasks: Vec<(usize, PathBuf, Option<PathBuf>)>,
    detector: &YuNetDetector,
    output_dir: &Path,
    settings: &fcs_utils::config::AppSettings,
    tx: &std::sync::mpsc::Sender<JobMessage>,
    completed: &AtomicUsize,
    failed: &AtomicUsize,
) {
    for (index, path, output_override) in tasks {
        let _ = tx.send(JobMessage::BatchProgress {
            index,
            status: BatchFileStatus::Processing,
        });
        let status = run_batch_job_panic_safe(detector, path, output_dir, settings, output_override);
        if matches!(status, BatchFileStatus::Failed { .. }) {
            failed.fetch_add(1, AtomicOrdering::Relaxed);
        } else {
            completed.fetch_add(1, AtomicOrdering::Relaxed);
        }
        let _ = tx.send(JobMessage::BatchProgress { index, status });
    }
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

    // Per-batch detection diagnostic — disabled. Re-enable by uncommenting this
    // line and the `log_detection_diag` function below to investigate GPU inference
    // run-to-run variance. See ARCHITECTURE.md "GPU Inference Determinism".
    // log_detection_diag(&path, &detections, settings.detection.score_threshold);

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

    let (src_w, src_h) = source_image.dimensions();
    for (face_index, detection) in detections.iter().enumerate() {
        let raw = crop_face_from_image(&source_image, detection, &crop_settings);
        let shaped = apply_shape_and_fill(raw, &settings.crop);
        let eyes = eye_positions(detection, src_w, src_h, &crop_settings);
        let crop = apply_enhancements(
            &shaped,
            &settings.enhance.to_enhancement_settings(),
            if eyes.is_empty() { None } else { Some(&eyes) },
        );
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
    let ext = output_extension(settings.crop.output_format);

    if let Some(custom) = output_override {
        return resolve_override_path(output_dir, custom, ext, candidate.face_index, multi_face);
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

fn output_extension(format: ImageFormatHint) -> &'static str {
    match format {
        ImageFormatHint::Png => "png",
        ImageFormatHint::Jpeg => "jpg",
        ImageFormatHint::Webp => "webp",
        ImageFormatHint::Tiff => "tif",
        ImageFormatHint::Bmp => "bmp",
        ImageFormatHint::Avif => "avif",
    }
}
