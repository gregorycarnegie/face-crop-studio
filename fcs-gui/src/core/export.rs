//! Export / batch crop logic.

use crate::types::{App2, BatchFile, BatchFileStatus, JobMessage};

use fcs_core::{CropSettings as CoreCropSettings, Detection, FaceDetector, crop_face_from_image};
use fcs_utils::{
    ImageFormatHint, MetadataContext, OutputClaims, OutputOptions, OverwritePolicy,
    append_suffix_to_filename, load_image, quality::Quality, save_dynamic_image,
};
use image::{DynamicImage, GenericImageView};
use log::{error, info, warn};
use rayon::prelude::*;
use rfd::FileDialog;
use std::{
    cmp::Ordering,
    collections::HashMap,
    panic::{AssertUnwindSafe, catch_unwind},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering as AtomicOrdering},
    },
};

struct ExportCandidate {
    face_index: usize,
    quality: Quality,
    quality_score: f64,
    detection_score: f32,
}

/// One face of the loaded image as it will be exported: cropped, enhanced, scored, shaped and
/// filled. The canvas's result preview calls this too, so what it shows is what gets saved.
pub fn finish_face(
    source: &DynamicImage,
    detection: &Detection,
    crop: &fcs_utils::config::CropSettings,
    enhance: &fcs_utils::config::EnhanceSettings,
    enhancement: &fcs_utils::EnhancementRuntime,
) -> fcs_utils::FinishedCrop {
    let core: CoreCropSettings = crop.into();
    let raw = crop_face_from_image(source, detection, &core);
    let eyes = fcs_core::eye_positions(detection, source.width(), source.height(), &core);
    enhancement.finish_crop(
        raw,
        crop,
        Some(&enhance.to_enhancement_settings()),
        (!eyes.is_empty()).then_some(&eyes[..]),
    )
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

/// An export waiting on the user's answer about files it would replace.
pub struct PendingExport {
    pub output_dir: PathBuf,
    pub kind: PendingKind,
    /// Files already in the folder that this export would (or, for a batch, could) replace.
    pub conflicts: Vec<PathBuf>,
}

pub enum PendingKind {
    Faces(Vec<usize>),
    Batch,
}

/// The overwrite dialog's answer: `None` cancels.
pub fn resolve_pending_export(app: &mut App2, choice: Option<OverwritePolicy>) {
    let Some(pending) = app.pending_export.take() else {
        return;
    };
    let Some(policy) = choice else {
        app.show_success("Export cancelled");
        return;
    };
    match pending.kind {
        PendingKind::Faces(selected) => {
            write_preview_faces(app, selected, &pending.output_dir, policy)
        }
        PendingKind::Batch => run_batch_export(app, pending.output_dir, policy),
    }
}

/// Where a preview face is saved. Known before cropping, so clashes can be asked about first.
fn preview_face_path(app: &App2, face_index: usize, output_dir: &Path) -> Option<PathBuf> {
    let det = app.preview.detections.get(face_index)?;
    let stem = app
        .preview
        .image_path
        .as_deref()
        .and_then(Path::file_stem)
        .and_then(|s| s.to_str())
        .unwrap_or("face");
    let ext = output_extension(app.settings.crop.output_format);
    let mut filename = format!("{stem}_face_{:02}.{ext}", face_index + 1);
    if let Some(suffix) = quality_suffix(&app.settings, det.quality) {
        filename = append_suffix_to_filename(&filename, suffix);
    }
    Some(output_dir.join(filename))
}

fn export_preview_faces(app: &mut App2, selected: Vec<usize>, error_title: &str) {
    if selected.is_empty() {
        app.show_error(error_title, "No faces selected for export");
        return;
    }
    if app.preview.source_image.is_none() {
        app.show_error(error_title, "No image loaded");
        return;
    }

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

    let conflicts: Vec<PathBuf> = selected
        .iter()
        .filter_map(|&i| preview_face_path(app, i, &output_dir))
        .filter(|path| path.exists())
        .collect();
    if conflicts.is_empty() {
        write_preview_faces(app, selected, &output_dir, OverwritePolicy::Overwrite);
    } else {
        app.pending_export = Some(PendingExport {
            output_dir,
            kind: PendingKind::Faces(selected),
            conflicts,
        });
    }
}

fn write_preview_faces(
    app: &mut App2,
    selected: Vec<usize>,
    output_dir: &Path,
    policy: OverwritePolicy,
) {
    let Some(source_image) = app.preview.source_image.clone() else {
        app.show_error("Export failed", "No image loaded");
        return;
    };
    let source_path = app.preview.image_path.clone();
    let output_options = OutputOptions::from_crop_settings(&app.settings.crop);
    let claims = OutputClaims::new(policy);

    let mut exported = 0usize;
    let mut failed = 0usize;

    for face_index in selected {
        let (Some(det), Some(planned)) = (
            app.preview.detections.get(face_index),
            preview_face_path(app, face_index, output_dir),
        ) else {
            failed += 1;
            continue;
        };
        let output_path = claims.claim(planned, source_path.as_deref().unwrap_or(Path::new("")));

        // The same runtime the CLI and the batch path use, so the three agree about whether
        // enhancement runs on the shaders.
        let crop = finish_face(
            &source_image,
            &det.edited_detection(),
            &app.settings.crop,
            &app.settings.enhance,
            &app.gpu.enhancement,
        )
        .image;

        let metadata_ctx = MetadataContext {
            source_path: source_path.as_deref(),
            crop_settings: Some(&app.settings.crop),
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
    // The buttons that start a batch are disabled while one runs, but the toolbar and menu
    // reach this too, and two batches writing one folder would race for the same names.
    if app.batch_running {
        app.show_error("Batch export failed", "A batch is already running");
        return;
    }
    if app.batch_files.is_empty() {
        app.show_error("Batch export failed", "No queued images");
        return;
    }

    if app.detector.is_none() {
        app.show_error("Batch export failed", "No detector loaded");
        return;
    }

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

    let ext = output_extension(app.settings.crop.output_format);
    let conflicts = batch_conflicts(&app.batch_files, &output_dir, ext);
    if conflicts.is_empty() {
        run_batch_export(app, output_dir, OverwritePolicy::Overwrite);
    } else {
        app.pending_export = Some(PendingExport {
            output_dir,
            kind: PendingKind::Batch,
            conflicts,
        });
    }
}

/// Existing files in the output folder that a batch could replace.
///
/// A batch's names depend on how many faces each image turns out to have, so this matches the
/// names a queued image can produce rather than exact ones: `{stem}_face_NN[_highq|_medq|_lowq]`
/// by default, and `{name}` or `{name}_faceN` for a mapped output name.
fn batch_conflicts(files: &[BatchFile], output_dir: &Path, ext: &str) -> Vec<PathBuf> {
    let mut listings: HashMap<PathBuf, Vec<PathBuf>> = HashMap::new();
    let mut conflicts = Vec::new();
    for file in files {
        let (dir, stem, mapped) = match &file.output_override {
            Some(target) => {
                let safe = sanitize_relative_path(target);
                let stem = safe
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .filter(|s| !s.is_empty())
                    .unwrap_or("output")
                    .to_owned();
                let parent = safe.parent().unwrap_or_else(|| Path::new(""));
                (output_dir.join(parent), stem, true)
            }
            None => {
                let stem = file
                    .path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("face")
                    .to_owned();
                (output_dir.to_path_buf(), stem, false)
            }
        };
        let listing = listings.entry(dir.clone()).or_insert_with(|| {
            std::fs::read_dir(&dir)
                .map(|entries| {
                    entries
                        .filter_map(Result::ok)
                        .map(|e| e.path())
                        .filter(|p| p.is_file())
                        .collect()
                })
                .unwrap_or_default()
        });
        conflicts.extend(
            listing
                .iter()
                .filter(|existing| {
                    existing
                        .file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|name| could_write(name, &stem, ext, mapped))
                })
                .cloned(),
        );
    }
    conflicts.sort();
    conflicts.dedup();
    conflicts
}

/// Whether `name` is one a batch could write for an image whose output stem is `stem`.
fn could_write(name: &str, stem: &str, ext: &str, mapped: bool) -> bool {
    let (name, stem) = (name.to_lowercase(), stem.to_lowercase());
    let Some(rest) = name
        .strip_prefix(&stem)
        .and_then(|rest| rest.strip_suffix(&format!(".{ext}")))
    else {
        return false;
    };
    let digits = |s: &str| s.len() - s.trim_start_matches(|c: char| c.is_ascii_digit()).len();
    if mapped {
        return rest.is_empty()
            || rest
                .strip_prefix("_face")
                .is_some_and(|n| !n.is_empty() && digits(n) == n.len());
    }
    let Some(rest) = rest.strip_prefix("_face_") else {
        return false;
    };
    let n = digits(rest);
    n >= 2 && ["", "_highq", "_medq", "_lowq"].contains(&&rest[n..])
}

/// Run a batch into `output_dir`, the overwrite question already settled.
fn run_batch_export(app: &mut App2, output_dir: PathBuf, policy: OverwritePolicy) {
    // Checked again: the dialog may have been open while something else started a batch.
    if app.batch_running {
        app.show_error("Batch export failed", "A batch is already running");
        return;
    }
    let Some(detector) = app.detector.clone() else {
        app.show_error("Batch export failed", "No detector loaded");
        return;
    };
    // Batch export detects for itself rather than reusing the canvas's detections, so it needs
    // the refiner too. Without this, exported crops would be levelled by YuNet's eye points
    // while the preview beside them shows refined ones.
    let eye_refiner = app.eye_refiner.clone();
    // Cheap to clone: the enhancer sits behind an Arc, so every worker shares one.
    let enhancement = app.gpu.enhancement.clone();

    // Keyed by path, not queue position: the queue can still be edited while this runs, and a
    // removed row shifted every later index onto the wrong file. Queue paths are unique.
    let tasks: Vec<_> = app
        .batch_files
        .iter()
        .map(|file| (file.path.clone(), file.output_override.clone()))
        .collect();

    for file in &mut app.batch_files {
        file.status = BatchFileStatus::Pending;
    }

    let settings = Arc::new(app.settings.clone());
    let parallelism = settings.resolved_batch_parallelism();
    let tx = app.job_tx.clone();
    app.batch_running = true;
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
                    eye_refiner.as_deref(),
                    &enhancement,
                    output_dir.as_path(),
                    settings.as_ref(),
                    &tx,
                    &completed,
                    &failed,
                    policy,
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
        let inner_eye_refiner = eye_refiner.clone();
        let inner_enhancement = enhancement.clone();
        let inner_output_dir = output_dir.clone();
        let inner_settings = settings.clone();
        let inner_completed = completed.clone();
        let inner_failed = failed.clone();
        let claims = OutputClaims::new(policy);

        pool.install(move || {
            // `for_each_with` clones the init once per worker thread, so the
            // sender refcount bumps once per worker rather than once per task.
            tasks
                .into_par_iter()
                .for_each_with(inner_tx, |tx, (path, output_override)| {
                    let _ = tx.send(JobMessage::BatchProgress {
                        path: path.clone(),
                        status: BatchFileStatus::Processing,
                    });

                    let status = run_batch_job_panic_safe(
                        inner_detector.as_ref(),
                        inner_eye_refiner.as_deref(),
                        &inner_enhancement,
                        path.clone(),
                        inner_output_dir.as_path(),
                        &claims,
                        inner_settings.as_ref(),
                        output_override,
                    );

                    if matches!(status, BatchFileStatus::Failed { .. }) {
                        inner_failed.fetch_add(1, AtomicOrdering::Relaxed);
                    } else {
                        inner_completed.fetch_add(1, AtomicOrdering::Relaxed);
                    }

                    let _ = tx.send(JobMessage::BatchProgress { path, status });
                });
        });

        let _ = tx.send(JobMessage::BatchComplete {
            completed: completed.load(AtomicOrdering::Relaxed),
            failed: failed.load(AtomicOrdering::Relaxed),
        });
    });
}

/// Run one batch job and convert any panic into a [`BatchFileStatus::Failed`].
/// One corrupt/edge-case image must not abort the rest of the batch.
#[allow(clippy::too_many_arguments)]
fn run_batch_job_panic_safe(
    detector: &FaceDetector,
    eye_refiner: Option<&fcs_core::EyeRefiner>,
    enhancement: &fcs_utils::EnhancementRuntime,
    path: PathBuf,
    output_dir: &Path,
    claims: &OutputClaims,
    settings: &fcs_utils::config::AppSettings,
    output_override: Option<PathBuf>,
) -> BatchFileStatus {
    let path_display = path.display().to_string();
    let result = catch_unwind(AssertUnwindSafe(move || {
        run_batch_job(
            detector,
            eye_refiner,
            enhancement,
            path,
            output_dir,
            claims,
            settings,
            output_override,
        )
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
// Eight arguments, one over the lint. The codebase's usual answer is a context struct (see
// `BatchContext` in fcs-cli), but this is the only caller and the only callee, so that struct
// would exist to satisfy a counter rather than to remove repetition.
#[allow(clippy::too_many_arguments)]
fn run_batch_sequential(
    tasks: Vec<(PathBuf, Option<PathBuf>)>,
    detector: &FaceDetector,
    eye_refiner: Option<&fcs_core::EyeRefiner>,
    enhancement: &fcs_utils::EnhancementRuntime,
    output_dir: &Path,
    settings: &fcs_utils::config::AppSettings,
    tx: &std::sync::mpsc::Sender<JobMessage>,
    completed: &AtomicUsize,
    failed: &AtomicUsize,
    policy: OverwritePolicy,
) {
    let claims = OutputClaims::new(policy);
    for (path, output_override) in tasks {
        let _ = tx.send(JobMessage::BatchProgress {
            path: path.clone(),
            status: BatchFileStatus::Processing,
        });
        let status = run_batch_job_panic_safe(
            detector,
            eye_refiner,
            enhancement,
            path.clone(),
            output_dir,
            &claims,
            settings,
            output_override,
        );
        if matches!(status, BatchFileStatus::Failed { .. }) {
            failed.fetch_add(1, AtomicOrdering::Relaxed);
        } else {
            completed.fetch_add(1, AtomicOrdering::Relaxed);
        }
        let _ = tx.send(JobMessage::BatchProgress { path, status });
    }
}

#[allow(clippy::too_many_arguments)]
fn run_batch_job(
    detector: &FaceDetector,
    eye_refiner: Option<&fcs_core::EyeRefiner>,
    enhancement: &fcs_utils::EnhancementRuntime,
    path: PathBuf,
    output_dir: &Path,
    claims: &OutputClaims,
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

    let mut detections = match detector.detect_image(&source_image) {
        Ok(output) => output.detections,
        Err(err) => {
            return BatchFileStatus::Failed {
                error: format!("Detection failed: {err}"),
            };
        }
    };
    // Before cropping, so batch output is levelled by the same eye points the canvas shows.
    if let Some(refiner) = eye_refiner {
        refiner.refine(&source_image, &mut detections);
    }

    let faces_detected = detections.len();
    if detections.is_empty() {
        return BatchFileStatus::Completed {
            faces_detected,
            faces_exported: 0,
        };
    }

    let crop_settings: CoreCropSettings = (&settings.crop).into();
    let output_options = OutputOptions::from_crop_settings(&settings.crop);
    let mut crops = Vec::with_capacity(detections.len());
    let mut candidates = Vec::with_capacity(detections.len());

    let (src_w, src_h) = source_image.dimensions();
    for (face_index, detection) in detections.iter().enumerate() {
        let raw = crop_face_from_image(&source_image, detection, &crop_settings);
        let eyes = fcs_core::eye_positions(detection, src_w, src_h, &crop_settings);
        // The CLI's order, from the same function: the GUI used to shape first and score the
        // shaped result, so the fill's edge could decide which faces passed the quality rules.
        let finished = enhancement.finish_crop(
            raw,
            &settings.crop,
            Some(&settings.enhance.to_enhancement_settings()),
            (!eyes.is_empty()).then_some(&eyes[..]),
        );
        crops.push(finished.image);
        candidates.push(ExportCandidate {
            face_index,
            quality: finished.quality,
            quality_score: finished.quality_score,
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
    let mut save_errors = Vec::new();
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

        let output_path = claims.claim(
            build_output_path(
                output_dir,
                &path,
                candidate,
                settings,
                output_override.as_ref(),
                multi_face,
            ),
            &path,
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
                save_errors.push(format!("{}: {err}", output_path.display()));
            }
        }
    }

    // A crop that was chosen but never written is lost output, so it fails the image
    // rather than hiding inside a lower "exported" count.
    if !save_errors.is_empty() {
        return BatchFileStatus::Failed {
            error: format!(
                "Saved {faces_exported} of {} crop(s): {}",
                faces_exported + save_errors.len(),
                save_errors.join("; ")
            ),
        };
    }

    BatchFileStatus::Completed {
        faces_detected,
        faces_exported,
    }
}

#[derive(serde::Serialize)]
struct ReportRow<'a> {
    image: String,
    outcome: &'static str,
    faces_detected: Option<usize>,
    faces_exported: Option<usize>,
    error: Option<&'a str>,
}

/// One row per queued image with its outcome, as CSV or as JSON with success/failure totals.
pub fn batch_report(files: &[BatchFile], csv: bool) -> anyhow::Result<Vec<u8>> {
    let rows: Vec<ReportRow> = files
        .iter()
        .map(|file| {
            let (faces_detected, faces_exported) = match file.status {
                BatchFileStatus::Completed {
                    faces_detected,
                    faces_exported,
                } => (Some(faces_detected), Some(faces_exported)),
                _ => (None, None),
            };
            ReportRow {
                image: file.path.display().to_string(),
                outcome: file.status.outcome(),
                faces_detected,
                faces_exported,
                error: match &file.status {
                    BatchFileStatus::Failed { error } => Some(error.as_str()),
                    _ => None,
                },
            }
        })
        .collect();

    if csv {
        let mut writer = csv::Writer::from_writer(Vec::new());
        for row in &rows {
            writer.serialize(row)?;
        }
        return Ok(writer.into_inner().map_err(|e| e.into_error())?);
    }
    let count = |outcome: &str| rows.iter().filter(|row| row.outcome == outcome).count();
    Ok(serde_json::to_vec_pretty(&serde_json::json!({
        "succeeded": count("succeeded"),
        "failed": count("failed"),
        "images": rows,
    }))?)
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

#[cfg(test)]
mod tests {
    use super::*;
    use fcs_utils::config::QualityAutomationSettings;

    #[test]
    fn select_candidates_keeps_every_face_unless_auto_select_is_on() {
        // Multi-face mode is this rule switched off.
        let candidate = |face_index, quality, quality_score| ExportCandidate {
            face_index,
            quality,
            quality_score,
            detection_score: 0.9,
        };
        let candidates = [
            candidate(0, Quality::Medium, 5.0),
            candidate(1, Quality::High, 9.0),
            candidate(2, Quality::High, 7.0),
        ];
        let mut rules = QualityAutomationSettings::default();
        assert_eq!(select_candidates(&candidates, &rules), Some(vec![0, 1, 2]));

        rules.auto_select_best_face = true;
        assert_eq!(select_candidates(&candidates, &rules), Some(vec![1]));
    }

    #[test]
    fn batch_report_lists_each_image_with_its_outcome() {
        let file = |name: &str, status: BatchFileStatus| BatchFile {
            path: PathBuf::from(name),
            status,
            output_override: None,
        };
        let completed = |faces_detected, faces_exported| BatchFileStatus::Completed {
            faces_detected,
            faces_exported,
        };
        let files = [
            file("ok.jpg", completed(3, 3)),
            file("empty.jpg", completed(0, 0)),
            file("filtered.jpg", completed(2, 0)),
            file(
                "bad, \"name\".jpg",
                BatchFileStatus::Failed {
                    error: "Failed to load: truncated".into(),
                },
            ),
            file("later.jpg", BatchFileStatus::Pending),
        ];

        let csv = String::from_utf8(batch_report(&files, true).unwrap()).unwrap();
        assert_eq!(
            csv,
            "image,outcome,faces_detected,faces_exported,error\n\
             ok.jpg,succeeded,3,3,\n\
             empty.jpg,no_faces,0,0,\n\
             filtered.jpg,filtered,2,0,\n\
             \"bad, \"\"name\"\".jpg\",failed,,,Failed to load: truncated\n\
             later.jpg,pending,,,\n"
        );

        let json: serde_json::Value =
            serde_json::from_slice(&batch_report(&files, false).unwrap()).unwrap();
        assert_eq!(json["succeeded"], 1);
        assert_eq!(json["failed"], 1);
        assert_eq!(json["images"].as_array().map(Vec::len), Some(5));
        assert_eq!(json["images"][3]["outcome"], "failed");
        assert_eq!(json["images"][3]["error"], "Failed to load: truncated");
    }

    /// `a/portrait.jpg` and `b/portrait.jpg` both default to `portrait_face_01`. Under one
    /// batch's claims the second gets its own file instead of replacing the first.
    #[test]
    fn same_named_sources_in_one_batch_keep_both_crops() {
        let Some(model) = fcs_utils::model_path("models/scrfd80k_500m_640.onnx").unwrap() else {
            eprintln!("skipped: no model present");
            return;
        };
        let detector = FaceDetector::load_from(model).expect("the shipped model loads");
        let sample = fcs_utils::model_path("samples/sample_01.jpg")
            .unwrap()
            .expect("sample image");
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out");
        let settings = fcs_utils::config::AppSettings::default();
        let enhancement = fcs_utils::EnhancementRuntime::cpu_only();
        let claims = OutputClaims::default();

        let mut exported = 0;
        for folder in ["a", "b"] {
            let source = dir.path().join(folder).join("portrait.jpg");
            std::fs::create_dir_all(source.parent().unwrap()).unwrap();
            std::fs::copy(&sample, &source).unwrap();
            match run_batch_job(
                &detector,
                None,
                &enhancement,
                source,
                &out,
                &claims,
                &settings,
                None,
            ) {
                BatchFileStatus::Completed { faces_exported, .. } => exported += faces_exported,
                other => panic!("batch job did not complete: {other:?}"),
            }
        }
        assert!(exported >= 2, "each copy exports at least one face");
        assert_eq!(std::fs::read_dir(&out).unwrap().count(), exported);
    }

    /// Only names a batch could actually write count as clashes: not a kept `(2)` copy, not
    /// another extension, not a longer stem that merely starts the same way.
    #[test]
    fn could_write_matches_only_names_a_batch_produces() {
        assert!(could_write(
            "portrait_face_01.png",
            "portrait",
            "png",
            false
        ));
        assert!(could_write(
            "Portrait_Face_12_highq.png",
            "portrait",
            "png",
            false
        ));
        assert!(!could_write(
            "portrait_face_01(2).png",
            "portrait",
            "png",
            false
        ));
        assert!(!could_write(
            "portrait_face_01.jpg",
            "portrait",
            "png",
            false
        ));
        assert!(!could_write(
            "portrait2_face_01.png",
            "portrait",
            "png",
            false
        ));
        assert!(!could_write(
            "portrait_face_1.png",
            "portrait",
            "png",
            false
        ));

        assert!(could_write("jane.png", "jane", "png", true));
        assert!(could_write("jane_face2.png", "jane", "png", true));
        assert!(!could_write("jane_face.png", "jane", "png", true));
        assert!(!could_write("janet.png", "jane", "png", true));
    }

    #[test]
    fn batch_conflicts_lists_existing_files_the_queue_could_replace() {
        let dir = tempfile::tempdir().unwrap();
        for name in ["a_face_01.png", "a_face_01(2).png", "b.png", "other.png"] {
            std::fs::write(dir.path().join(name), b"x").unwrap();
        }
        std::fs::create_dir(dir.path().join("mapped")).unwrap();
        std::fs::write(dir.path().join("mapped").join("jane.png"), b"x").unwrap();
        let file = |path: &str, output_override: Option<&str>| BatchFile {
            path: PathBuf::from(path),
            status: BatchFileStatus::Pending,
            output_override: output_override.map(PathBuf::from),
        };
        let queue = [
            file("in/a.jpg", None),
            file("in/b.jpg", None),
            file("in/c.jpg", Some("mapped/jane.jpg")),
        ];

        assert_eq!(
            batch_conflicts(&queue, dir.path(), "png"),
            vec![
                dir.path().join("a_face_01.png"),
                dir.path().join("mapped").join("jane.png"),
            ]
        );
    }
}
