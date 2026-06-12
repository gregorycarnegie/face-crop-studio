//! Command-line interface for running YuNet face detection.

use std::{
    fs::{self, File},
    path::{Path, PathBuf},
    sync::{Arc, atomic::Ordering},
    time::SystemTime,
};

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use fcs_core::{
    BoundingBox, Detection, DetectionOutput, PostprocessConfig, PreprocessConfig,
    crop_face_from_image,
};
use fcs_utils::{
    OutputOptions, Quality, config::AppSettings, configure_telemetry, estimate_sharpness,
    init_logging, load_image, normalize_path, resolve_data_path,
};
use image::{DynamicImage, GenericImageView};
use log::{debug, info, warn};
use rayon::prelude::*;

mod annotate;
mod args;
mod benchmark;
mod color;
mod config;
mod detector;
mod enhancement;
mod gpu;
mod input;
mod output_path;
mod quality;
mod types;
mod webcam;

use output_path::{build_crop_output_path, normalized_output_extension, save_processed_crop};

use args::DetectArgs;
use config::{apply_cli_overrides, load_settings};
use detector::build_cli_detector;
use enhancement::build_enhancement_settings;
use gpu::init_cli_gpu_runtime;
use input::{collect_mapping_targets, collect_standard_targets, resolve_override_output_path};
use quality::build_quality_filter;
use types::{DetectionRecord, ImageDetections, ProgressCounters};

#[derive(Clone, Debug)]
pub(crate) struct ProcessedCrop {
    index: usize,
    image: DynamicImage,
    quality: Quality,
    quality_score: f64,
    score: f32,
}

fn main() -> Result<()> {
    let args = DetectArgs::parse();

    let mut settings = load_settings(args.config.as_ref())?;
    apply_cli_overrides(&mut settings, &args);

    configure_telemetry(
        settings.telemetry.enabled,
        settings.telemetry.level_filter(),
    );
    init_logging(log::LevelFilter::Info)?;

    if settings.telemetry.enabled {
        info!(
            "Telemetry logging enabled (level={:?})",
            settings.telemetry.level_filter()
        );
    }

    let model_path = normalize_path(resolve_data_path(&args.model))?;
    let annotate_dir = if let Some(dir) = args.annotate.as_ref() {
        fs::create_dir_all(dir)
            .with_context(|| format!("failed to create annotation directory {}", dir.display()))?;
        Some(normalize_path(dir)?)
    } else {
        None
    };

    // Build a centralized quality filter using resolved automation settings so the same
    // policy is used for cropping, batch export, and future GUI wiring.
    let quality_filter = build_quality_filter(&settings.crop.quality_rules);
    let gpu_runtime = Arc::new(init_cli_gpu_runtime(&settings)?);

    let preprocess_config: PreprocessConfig = settings.input.into();
    let postprocess_config: PostprocessConfig = (&settings.detection).into();
    let input_size = preprocess_config.input_size;

    // Check if webcam mode is enabled
    if args.webcam {
        info!(
            "Loading YuNet model from {} at resolution {}x{}",
            model_path.display(),
            input_size.width,
            input_size.height
        );
        let prefer_gpu_inference = settings.gpu.enabled && settings.gpu.inference;
        let detector = build_cli_detector(
            &model_path,
            &preprocess_config,
            &postprocess_config,
            gpu_runtime.as_ref(),
            prefer_gpu_inference,
        )?;
        let detector = Arc::new(detector);
        let settings = Arc::new(settings);
        let quality_filter = Arc::new(quality_filter);
        let enhancement_settings = build_enhancement_settings(&args).map(Arc::new);

        return webcam::run_webcam_mode(
            &args,
            detector,
            settings,
            gpu_runtime,
            quality_filter,
            enhancement_settings,
        );
    }

    let processing_items = if let Some(mapping_file) = args.mapping_file.as_ref() {
        collect_mapping_targets(mapping_file, &args)?
    } else {
        let input_arg = args
            .input
            .as_ref()
            .ok_or_else(|| anyhow!("--input is required when --mapping-file is not provided"))?;
        let input_path = normalize_path(input_arg)?;
        collect_standard_targets(&input_path)?
    };
    if processing_items.is_empty() {
        anyhow::bail!("no images were queued for processing");
    }

    if args.benchmark_preprocess {
        benchmark::run_preprocess_benchmark(
            &processing_items,
            &preprocess_config,
            gpu_runtime.context(),
        )?;
        return Ok(());
    }

    info!(
        "Loading YuNet model from {} at resolution {}x{}",
        model_path.display(),
        input_size.width,
        input_size.height
    );
    let prefer_gpu_inference = settings.gpu.enabled && settings.gpu.inference;
    let detector = build_cli_detector(
        &model_path,
        &preprocess_config,
        &postprocess_config,
        gpu_runtime.as_ref(),
        prefer_gpu_inference,
    )?;

    if args.mapping_file.is_some() && !args.crop {
        info!(
            "Mapping loaded without --crop; output overrides will be applied when cropping is executed."
        );
    }

    info!("Processing {} target(s)...", processing_items.len());

    // Wrap detector in Arc for thread-safe shared access
    let detector = Arc::new(detector);
    let annotate_dir = Arc::new(annotate_dir);

    // Prepare crop output directory if requested
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
    let crop_output_dir = Arc::new(crop_output_dir);
    let shared_settings = Arc::new(settings);
    let quality_filter = Arc::new(quality_filter);
    let enhancement_settings = build_enhancement_settings(&args).map(Arc::new);

    let counters = ProgressCounters::default();

    let results: Vec<ImageDetections> = processing_items
        .par_iter()
        .filter_map(|target| {
            process_single_image(
                target,
                &detector,
                &annotate_dir,
                &shared_settings,
                &quality_filter,
                &enhancement_settings,
                &gpu_runtime,
                &args,
                crop_enabled,
                &crop_output_dir,
                &counters,
            )
        })
        .collect();

    if results.is_empty() {
        anyhow::bail!("all detections failed; cannot produce output");
    }

    if let Some(json_path) = args.json.as_ref() {
        let parent = json_path.parent();
        if let Some(dir) = parent {
            fs::create_dir_all(dir)
                .with_context(|| format!("failed to create directory {}", dir.display()))?;
        }
        let file = File::create(json_path)
            .with_context(|| format!("failed to create {}", json_path.display()))?;
        serde_json::to_writer_pretty(file, &results).with_context(|| {
            format!("failed to write detection JSON to {}", json_path.display())
        })?;
        info!("Wrote detections to {}", json_path.display());
    } else {
        let json =
            serde_json::to_string_pretty(&results).context("failed to serialize detections")?;
        println!("{json}");
    }

    let summary = counters.snapshot();
    let summary_line = format!(
        "images_processed={} faces_detected={} crops_saved={} crops_skipped_quality={}",
        summary.images_processed,
        summary.faces_detected,
        summary.crops_saved,
        summary.crops_skipped_quality
    );
    info!("Summary: {summary_line}");
    println!("{summary_line}");

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn process_single_image(
    target: &input::ProcessingItem,
    detector: &Arc<fcs_core::YuNetDetector>,
    annotate_dir: &Arc<Option<std::path::PathBuf>>,
    settings: &Arc<AppSettings>,
    quality_filter: &Arc<fcs_utils::QualityFilter>,
    enhancement_settings: &Option<Arc<fcs_utils::EnhancementSettings>>,
    runtime: &Arc<gpu::CliGpuRuntime>,
    args: &DetectArgs,
    crop_enabled: bool,
    crop_output_dir: &Arc<Option<std::path::PathBuf>>,
    counters: &ProgressCounters,
) -> Option<ImageDetections> {
    counters.images_processed.fetch_add(1, Ordering::Relaxed);
    let image_path = &target.source;
    let override_target = target.output_override.as_ref();
    if let Some(row) = target.mapping_row {
        debug!("Mapping row {} -> {}", row, image_path.display());
    }

    // Decode once and reuse for detection, annotation, and cropping. Detection
    // must see the same EXIF-oriented pixels the crops are taken from.
    let img_opt = load_source_image(image_path);
    let output = detect_image(detector, img_opt.as_ref()?, image_path)?;
    counters
        .faces_detected
        .fetch_add(output.detections.len(), Ordering::Relaxed);

    info!(
        "{} -> {} detection(s)",
        image_path.display(),
        output.detections.len()
    );
    let annotated_path = annotate_image_if_requested(
        img_opt.as_ref(),
        image_path,
        &output.detections,
        annotate_dir,
    );

    match (crop_enabled, crop_output_dir.as_ref(), img_opt.as_ref()) {
        (true, Some(out_dir), Some(img)) => {
            process_crops(
                img,
                image_path,
                &output.detections,
                settings,
                quality_filter,
                enhancement_settings,
                runtime,
                args,
                out_dir,
                override_target,
                counters,
            );
        }
        (true, Some(_), None) => {
            warn!(
                "Cannot crop {} because the source image failed to load",
                image_path.display()
            );
        }
        _ => {}
    }

    let detection_records = build_detection_records(&output.detections, img_opt.as_ref());

    Some(ImageDetections {
        image: image_path.display().to_string(),
        detections: detection_records,
        annotated: annotated_path,
    })
}

#[allow(clippy::too_many_arguments)]
fn process_crops(
    img: &image::DynamicImage,
    image_path: &Path,
    detections: &[fcs_core::Detection],
    settings: &Arc<AppSettings>,
    quality_filter: &Arc<fcs_utils::QualityFilter>,
    enhancement_settings: &Option<Arc<fcs_utils::EnhancementSettings>>,
    runtime: &Arc<gpu::CliGpuRuntime>,
    args: &DetectArgs,
    out_dir: &Path,
    override_target: Option<&std::path::PathBuf>,
    counters: &ProgressCounters,
) {
    let output_options = OutputOptions::from_crop_settings(&settings.crop);
    let core_settings: fcs_core::CropSettings = (&settings.crop).into();
    let processed = generate_processed_crops(
        img,
        detections,
        settings,
        &core_settings,
        enhancement_settings.as_ref(),
        runtime.as_ref(),
    );

    if processed.is_empty() {
        debug!(
            "No crops generated for {} (no detections)",
            image_path.display()
        );
        return;
    }

    let best_quality = processed.iter().map(|c| c.quality).max();

    if quality_filter.should_skip_image(best_quality) {
        info!(
            "Skipping exports for {} because no face reached high quality",
            image_path.display()
        );
        counters
            .crops_skipped_quality
            .fetch_add(processed.len().max(1), Ordering::Relaxed);
        return;
    }

    let exports = filter_faces_for_export(processed, args, quality_filter, image_path);

    let multi_face = exports.len() > 1;
    for crop in exports.into_iter() {
        if quality_filter.should_skip(crop.quality) {
            info!(
                "Skipping crop for {} face {} due to {:?} quality (score {:.1})",
                image_path.display(),
                crop.index + 1,
                crop.quality,
                crop.quality_score
            );
            counters
                .crops_skipped_quality
                .fetch_add(1, Ordering::Relaxed);
            continue;
        }

        let ext = normalized_output_extension(settings.crop.output_format);
        let timestamp = if args.naming_template.is_some() {
            SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0)
        } else {
            0
        };

        let out_path = if let Some(custom) = override_target {
            resolve_override_output_path(out_dir, custom, ext, crop.index, multi_face)
        } else {
            build_crop_output_path(
                out_dir,
                image_path,
                output_path::CropFilenameSpec {
                    source_stem: "",
                    crop_index: crop.index,
                    output_width: settings.crop.output_width,
                    output_height: settings.crop.output_height,
                    ext,
                    naming_template: args.naming_template.as_deref(),
                    quality_suffix: quality_filter.suffix_for(crop.quality),
                    timestamp,
                },
            )
        };

        if let Some(parent) = out_path.parent()
            && let Err(err) = fs::create_dir_all(parent)
        {
            warn!("Failed to create directory {}: {err}", parent.display());
        }

        match save_processed_crop(&crop, &out_path, image_path, settings, &output_options) {
            Ok(_) => {
                info!("Saved crop to {}", out_path.display());
                counters.crops_saved.fetch_add(1, Ordering::Relaxed);
            }
            Err(e) => {
                warn!("Failed to export crop {}: {e:?}", out_path.display());
            }
        }
    }
}

fn detect_image(
    detector: &Arc<fcs_core::YuNetDetector>,
    image: &DynamicImage,
    image_path: &Path,
) -> Option<DetectionOutput> {
    match detector.detect_image(image) {
        Ok(out) => Some(out),
        Err(err) => {
            warn!("Failed to process {}: {err}", image_path.display());
            None
        }
    }
}

fn load_source_image(image_path: &Path) -> Option<DynamicImage> {
    // load_image applies EXIF orientation, matching what the detector sees.
    match load_image(image_path) {
        Ok(img) => Some(img),
        Err(e) => {
            warn!("Failed to open image {}: {}", image_path.display(), e);
            None
        }
    }
}

fn annotate_image_if_requested(
    image: Option<&DynamicImage>,
    image_path: &Path,
    detections: &[Detection],
    annotate_dir: &Arc<Option<PathBuf>>,
) -> Option<String> {
    let dir = annotate_dir.as_ref().as_ref()?;
    let image = image?;

    match annotate::annotate_image(image, image_path, detections, dir) {
        Ok(path) => {
            info!("Annotated image saved to {}", path.display());
            Some(path.display().to_string())
        }
        Err(err) => {
            warn!("Failed to annotate {}: {err}", image_path.display());
            None
        }
    }
}

fn detection_crop_rect(
    bbox: &BoundingBox,
    image_width: u32,
    image_height: u32,
) -> Option<(u32, u32, u32, u32)> {
    let x1 = bbox.x.clamp(0.0, image_width as f32);
    let y1 = bbox.y.clamp(0.0, image_height as f32);
    let x2 = bbox.right().clamp(0.0, image_width as f32);
    let y2 = bbox.bottom().clamp(0.0, image_height as f32);
    let w = (x2 - x1).max(1.0).round() as u32;
    let h = (y2 - y1).max(1.0).round() as u32;
    let left = x1.round() as u32;
    let top = y1.round() as u32;

    if left + w <= image_width && top + h <= image_height {
        Some((left, top, w, h))
    } else {
        None
    }
}

fn detection_quality(image: &DynamicImage, bbox: &BoundingBox) -> Option<(f64, String)> {
    let (image_width, image_height) = image.dimensions();
    let (left, top, width, height) = detection_crop_rect(bbox, image_width, image_height)?;
    let sub = image::imageops::crop_imm(image, left, top, width, height).to_image();
    let dynsub = image::DynamicImage::ImageRgba8(sub);
    let (score, quality) = estimate_sharpness(&dynsub);
    Some((score, format!("{:?}", quality)))
}

fn build_detection_record(det: &Detection, image: Option<&DynamicImage>) -> DetectionRecord {
    let mut record = DetectionRecord::from(det);
    if let Some(img) = image
        && let Some((score, quality)) = detection_quality(img, &det.bbox)
    {
        record.quality_score = Some(score);
        record.quality = Some(quality);
    }
    record
}

fn build_detection_records(
    detections: &[Detection],
    image: Option<&DynamicImage>,
) -> Vec<DetectionRecord> {
    detections
        .iter()
        .map(|det| build_detection_record(det, image))
        .collect()
}

fn build_processed_crop(
    index: usize,
    detection: &Detection,
    mut crop_img: DynamicImage,
    settings: &AppSettings,
    enhancement_settings: Option<&Arc<fcs_utils::EnhancementSettings>>,
    runtime: &gpu::CliGpuRuntime,
) -> ProcessedCrop {
    if let Some(enh) = enhancement_settings {
        crop_img = runtime.enhance(&crop_img, enh);
    }

    let (quality_score, quality) = estimate_sharpness(&crop_img);
    crop_img = runtime.apply_shape_mask(
        &crop_img,
        &settings.crop.shape,
        settings.crop.vignette_softness,
        settings.crop.vignette_intensity,
        settings.crop.vignette_color,
    );

    ProcessedCrop {
        index,
        image: crop_img,
        quality,
        quality_score,
        score: detection.score,
    }
}

fn generate_processed_crops(
    img: &DynamicImage,
    detections: &[Detection],
    settings: &AppSettings,
    core_settings: &fcs_core::CropSettings,
    enhancement_settings: Option<&Arc<fcs_utils::EnhancementSettings>>,
    runtime: &gpu::CliGpuRuntime,
) -> Vec<ProcessedCrop> {
    let mut processed = Vec::with_capacity(detections.len());

    if let Some(gpu_images) = runtime.crop_faces_gpu(img, detections, core_settings) {
        for ((idx, det), crop_img) in detections.iter().enumerate().zip(gpu_images) {
            processed.push(build_processed_crop(
                idx,
                det,
                crop_img,
                settings,
                enhancement_settings,
                runtime,
            ));
        }
    }

    if processed.is_empty() {
        for (idx, det) in detections.iter().enumerate() {
            let crop_img = crop_face_from_image(img, det, core_settings);
            processed.push(build_processed_crop(
                idx,
                det,
                crop_img,
                settings,
                enhancement_settings,
                runtime,
            ));
        }
    }

    processed
}

fn retain_requested_face(processed: Vec<ProcessedCrop>, face_index: usize) -> Vec<ProcessedCrop> {
    processed
        .into_iter()
        .filter(|crop| crop.index == face_index)
        .collect()
}

/// Decide which faces survive into the export step based on `--face-index` and
/// quality-driven auto-selection. Logs the rationale at the corresponding level.
fn filter_faces_for_export(
    mut exports: Vec<ProcessedCrop>,
    args: &DetectArgs,
    quality_filter: &fcs_utils::QualityFilter,
    image_path: &Path,
) -> Vec<ProcessedCrop> {
    if let Some(fidx) = args.face_index {
        if fidx == 0 {
            warn!(
                "--face-index is 1-based; ignoring 0 for {}",
                image_path.display()
            );
            return Vec::new();
        }
        let target = fidx - 1;
        let available = exports.len();
        let filtered = retain_requested_face(exports, target);
        if filtered.is_empty() {
            warn!(
                "Requested face index {} not found for {} ({} detections)",
                fidx,
                image_path.display(),
                available
            );
        }
        return filtered;
    }

    if quality_filter.auto_select
        && exports.len() > 1
        && let Some(best_rel) = select_best_quality_index(&exports, quality_filter)
    {
        let best_idx = exports[best_rel].index;
        exports = retain_requested_face(exports, best_idx);
        debug!(
            "Auto-selected face {} for {} based on quality {:?}",
            best_idx + 1,
            image_path.display(),
            exports.first().map(|c| c.quality)
        );
    }

    exports
}

fn select_best_quality_index(
    processed: &[ProcessedCrop],
    quality_filter: &fcs_utils::QualityFilter,
) -> Option<usize> {
    let qualities: Vec<(Quality, f64)> = processed
        .iter()
        .map(|crop| (crop.quality, crop.quality_score))
        .collect();
    quality_filter.select_best_index(&qualities)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use fcs_core::{FillColor, InputSize, Landmark};
    use fcs_utils::{QualityFilter, config::ResizeQuality};
    use image::{Rgba, RgbaImage};
    use std::{fs, sync::Arc};
    use tempfile::tempdir;

    fn sample_image(width: u32, height: u32) -> DynamicImage {
        let mut image = RgbaImage::new(width, height);
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            let value = ((x + y) % 255) as u8;
            *pixel = Rgba([value, value.wrapping_add(1), value.wrapping_add(2), 255]);
        }
        DynamicImage::ImageRgba8(image)
    }

    fn sample_detection(x: f32, y: f32, width: f32, height: f32, score: f32) -> Detection {
        Detection {
            bbox: BoundingBox {
                x,
                y,
                width,
                height,
            },
            landmarks: [
                Landmark { x: 1.0, y: 1.0 },
                Landmark { x: 2.0, y: 1.0 },
                Landmark { x: 1.5, y: 1.5 },
                Landmark { x: 1.25, y: 2.0 },
                Landmark { x: 1.75, y: 2.0 },
            ],
            score,
        }
    }

    fn sample_processed_crop(index: usize, quality: Quality, quality_score: f64) -> ProcessedCrop {
        ProcessedCrop {
            index,
            image: sample_image(2, 2),
            quality,
            quality_score,
            score: 0.5,
        }
    }

    fn solid_image(width: u32, height: u32, value: u8) -> DynamicImage {
        DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            width,
            height,
            Rgba([value, value, value, 255]),
        ))
    }

    fn parse_args(args: &[&str]) -> DetectArgs {
        let mut full = vec!["fcs-cli"];
        full.extend_from_slice(args);
        DetectArgs::try_parse_from(full).expect("test CLI arguments should parse")
    }

    fn no_gpu_runtime() -> Arc<gpu::CliGpuRuntime> {
        let mut settings = AppSettings::default();
        settings.gpu.enabled = false;
        Arc::new(init_cli_gpu_runtime(&settings).expect("disabled GPU runtime should initialize"))
    }

    fn crop_settings_app() -> Arc<AppSettings> {
        let mut settings = AppSettings::default();
        settings.gpu.enabled = false;
        settings.crop.output_width = 16;
        settings.crop.output_height = 16;
        settings.crop.output_format = fcs_utils::ImageFormatHint::Png;
        settings.crop.fill_color = FillColor::default();
        Arc::new(settings)
    }

    fn build_test_detector() -> Option<Arc<fcs_core::YuNetDetector>> {
        let model_path = Path::new("models/face_detection_yunet_2023mar_640.onnx");
        if !model_path.exists() {
            return None;
        }
        let preprocess = PreprocessConfig {
            input_size: InputSize::new(32, 32),
            resize_quality: ResizeQuality::Quality,
        };
        let postprocess = PostprocessConfig::default();
        let detector = build_cli_detector(
            model_path,
            &preprocess,
            &postprocess,
            no_gpu_runtime().as_ref(),
            false,
        )
        .ok()?;
        Some(Arc::new(detector))
    }

    fn write_sample_png(path: &Path) {
        sample_image(32, 32)
            .save(path)
            .expect("sample PNG should be written");
    }

    #[test]
    fn detection_crop_rect_clamps_to_image_bounds() {
        let rect = detection_crop_rect(
            &BoundingBox {
                x: -3.4,
                y: 2.4,
                width: 8.2,
                height: 12.0,
            },
            10,
            10,
        );
        assert_eq!(rect, Some((0, 2, 5, 8)));
    }

    #[test]
    fn detection_crop_rect_returns_none_when_rounded_rect_exceeds_bounds() {
        let rect = detection_crop_rect(
            &BoundingBox {
                x: 9.6,
                y: 9.6,
                width: 1.0,
                height: 1.0,
            },
            10,
            10,
        );
        assert_eq!(rect, None);
    }

    #[test]
    fn build_detection_record_populates_quality_for_valid_crop() {
        let image = sample_image(10, 10);
        let detection = sample_detection(1.0, 2.0, 4.0, 4.0, 0.95);

        let record = build_detection_record(&detection, Some(&image));
        assert_eq!(record.score, 0.95);
        assert!(record.quality_score.is_some());
        assert!(record.quality.is_some());
    }

    #[test]
    fn build_detection_record_leaves_quality_empty_without_image() {
        let detection = sample_detection(1.0, 2.0, 4.0, 4.0, 0.95);

        let record = build_detection_record(&detection, None);
        assert_eq!(record.score, 0.95);
        assert_eq!(record.quality_score, None);
        assert_eq!(record.quality, None);
    }

    #[test]
    fn retain_requested_face_keeps_only_matching_index() {
        let crops = vec![
            sample_processed_crop(0, Quality::Low, 0.2),
            sample_processed_crop(1, Quality::Medium, 0.5),
            sample_processed_crop(2, Quality::High, 0.8),
        ];

        let filtered = retain_requested_face(crops, 1);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].index, 1);
        assert_eq!(filtered[0].quality, Quality::Medium);
    }

    #[test]
    fn select_best_quality_index_prefers_quality_then_score() {
        let crops = vec![
            sample_processed_crop(0, Quality::High, 0.1),
            sample_processed_crop(1, Quality::Medium, 1.0),
            sample_processed_crop(2, Quality::High, 0.9),
        ];
        let filter = QualityFilter {
            min_quality: None,
            auto_select: true,
            fallback: None,
            auto_skip_no_high: false,
            suffix_enabled: false,
        };

        let selected = select_best_quality_index(&crops, &filter);
        assert_eq!(selected, Some(2));
    }

    #[test]
    fn detect_image_returns_output_for_valid_image() {
        let Some(detector) = build_test_detector() else {
            return;
        };
        let dir = tempdir().expect("temp directory should be created");
        let image_path = dir.path().join("sample.png");
        write_sample_png(&image_path);

        let image = load_source_image(&image_path).expect("sample image should load");
        assert!(detect_image(&detector, &image, &image_path).is_some());
    }

    #[test]
    fn load_source_image_handles_success_and_failure() {
        let dir = tempdir().expect("temp directory should be created");
        let valid = dir.path().join("ok.png");
        write_sample_png(&valid);
        let missing = dir.path().join("missing.png");

        assert!(load_source_image(&valid).is_some());
        assert!(load_source_image(&missing).is_none());
    }

    #[test]
    fn annotate_image_if_requested_covers_none_success_and_failure() {
        let dir = tempdir().expect("temp directory should be created");
        let image_path = dir.path().join("annotate.png");
        write_sample_png(&image_path);
        let detections = vec![sample_detection(4.0, 4.0, 12.0, 12.0, 0.9)];

        let image = sample_image(32, 32);

        let none_dir = Arc::new(None);
        assert_eq!(
            annotate_image_if_requested(None, &image_path, &detections, &none_dir),
            None
        );

        let annotate_dir = Arc::new(Some(dir.path().join("annotated")));
        let success =
            annotate_image_if_requested(Some(&image), &image_path, &detections, &annotate_dir);
        assert!(
            success
                .as_ref()
                .is_some_and(|p| p.ends_with("annotate.png"))
        );

        let blocked_file = dir.path().join("blocked");
        fs::write(&blocked_file, b"not a dir").expect("blocked path file should be written");
        let blocked_dir = Arc::new(Some(blocked_file));
        assert_eq!(
            annotate_image_if_requested(Some(&image), &image_path, &detections, &blocked_dir),
            None
        );
    }

    #[test]
    fn detection_quality_returns_none_when_bbox_cannot_form_valid_crop() {
        let image = sample_image(10, 10);
        let bbox = BoundingBox {
            x: 9.6,
            y: 9.6,
            width: 1.0,
            height: 1.0,
        };

        assert_eq!(detection_quality(&image, &bbox), None);
    }

    #[test]
    fn build_detection_records_preserves_detection_count() {
        let image = sample_image(20, 20);
        let detections = vec![
            sample_detection(1.0, 1.0, 5.0, 5.0, 0.8),
            sample_detection(8.0, 8.0, 5.0, 5.0, 0.9),
        ];

        let records = build_detection_records(&detections, Some(&image));
        assert_eq!(records.len(), 2);
        assert!(records.iter().all(|record| record.quality.is_some()));
    }

    #[test]
    fn generate_processed_crops_falls_back_to_cpu_when_gpu_runtime_has_no_cropper() {
        let settings = crop_settings_app();
        let runtime = no_gpu_runtime();
        let core_settings: fcs_core::CropSettings = (&settings.crop).into();
        let detections = vec![sample_detection(2.0, 2.0, 10.0, 10.0, 0.95)];

        let crops = generate_processed_crops(
            &sample_image(24, 24),
            &detections,
            settings.as_ref(),
            &core_settings,
            None,
            runtime.as_ref(),
        );

        assert_eq!(crops.len(), 1);
        assert_eq!(crops[0].index, 0);
        assert_eq!(crops[0].score, 0.95);
    }

    #[test]
    fn process_crops_returns_early_when_there_are_no_detections() {
        let settings = crop_settings_app();
        let runtime = no_gpu_runtime();
        let filter = Arc::new(QualityFilter::new(None));
        let args = parse_args(&["--input", "x.jpg"]);
        let dir = tempdir().expect("temp directory should be created");
        let counters = ProgressCounters::default();

        process_crops(
            &sample_image(24, 24),
            Path::new("input.png"),
            &[],
            &settings,
            &filter,
            &None,
            &runtime,
            &args,
            dir.path(),
            None,
            &counters,
        );

        assert_eq!(counters.crops_saved.load(Ordering::Relaxed), 0);
        assert_eq!(counters.crops_skipped_quality.load(Ordering::Relaxed), 0);
        assert_eq!(
            fs::read_dir(dir.path())
                .expect("temp output directory should be readable")
                .count(),
            0
        );
    }

    #[test]
    fn process_crops_skips_image_when_no_high_quality_face_exists() {
        let settings = crop_settings_app();
        let runtime = no_gpu_runtime();
        let filter = Arc::new(QualityFilter {
            min_quality: None,
            auto_select: false,
            fallback: None,
            auto_skip_no_high: true,
            suffix_enabled: false,
        });
        let args = parse_args(&["--input", "x.jpg"]);
        let dir = tempdir().expect("temp directory should be created");
        let counters = ProgressCounters::default();
        let detections = vec![sample_detection(2.0, 2.0, 10.0, 10.0, 0.9)];

        process_crops(
            &solid_image(24, 24, 80),
            Path::new("input.png"),
            &detections,
            &settings,
            &filter,
            &None,
            &runtime,
            &args,
            dir.path(),
            None,
            &counters,
        );

        assert_eq!(counters.crops_saved.load(Ordering::Relaxed), 0);
        assert_eq!(counters.crops_skipped_quality.load(Ordering::Relaxed), 1);
        assert_eq!(
            fs::read_dir(dir.path())
                .expect("temp output directory should be readable")
                .count(),
            0
        );
    }

    #[test]
    fn process_crops_ignores_face_index_zero() {
        let settings = crop_settings_app();
        let runtime = no_gpu_runtime();
        let filter = Arc::new(QualityFilter::new(None));
        let args = parse_args(&["--input", "x.jpg", "--face-index", "0"]);
        let dir = tempdir().expect("temp directory should be created");
        let counters = ProgressCounters::default();
        let detections = vec![
            sample_detection(1.0, 1.0, 8.0, 8.0, 0.9),
            sample_detection(10.0, 10.0, 8.0, 8.0, 0.95),
        ];

        process_crops(
            &sample_image(24, 24),
            Path::new("input.png"),
            &detections,
            &settings,
            &filter,
            &None,
            &runtime,
            &args,
            dir.path(),
            None,
            &counters,
        );

        assert_eq!(counters.crops_saved.load(Ordering::Relaxed), 0);
        assert_eq!(counters.crops_skipped_quality.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn process_crops_skips_low_quality_export_when_threshold_is_high() {
        let settings = crop_settings_app();
        let runtime = no_gpu_runtime();
        let filter = Arc::new(QualityFilter {
            min_quality: Some(Quality::High),
            auto_select: false,
            fallback: None,
            auto_skip_no_high: false,
            suffix_enabled: false,
        });
        let args = parse_args(&["--input", "x.jpg"]);
        let dir = tempdir().expect("temp directory should be created");
        let counters = ProgressCounters::default();
        let detections = vec![sample_detection(2.0, 2.0, 10.0, 10.0, 0.9)];

        process_crops(
            &solid_image(24, 24, 64),
            Path::new("input.png"),
            &detections,
            &settings,
            &filter,
            &None,
            &runtime,
            &args,
            dir.path(),
            None,
            &counters,
        );

        assert_eq!(counters.crops_saved.load(Ordering::Relaxed), 0);
        assert_eq!(counters.crops_skipped_quality.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn process_crops_warns_when_requested_face_is_missing() {
        let settings = crop_settings_app();
        let runtime = no_gpu_runtime();
        let filter = Arc::new(QualityFilter::new(None));
        let args = parse_args(&["--input", "x.jpg", "--face-index", "3"]);
        let dir = tempdir().expect("temp directory should be created");
        let counters = ProgressCounters::default();
        let detections = vec![sample_detection(2.0, 2.0, 10.0, 10.0, 0.9)];

        process_crops(
            &sample_image(24, 24),
            Path::new("input.png"),
            &detections,
            &settings,
            &filter,
            &None,
            &runtime,
            &args,
            dir.path(),
            None,
            &counters,
        );

        assert_eq!(counters.crops_saved.load(Ordering::Relaxed), 0);
        assert_eq!(
            fs::read_dir(dir.path())
                .expect("temp output directory should be readable")
                .count(),
            0
        );
    }

    #[test]
    fn process_crops_auto_selects_the_best_face() {
        let settings = crop_settings_app();
        let runtime = no_gpu_runtime();
        let filter = Arc::new(QualityFilter {
            min_quality: None,
            auto_select: true,
            fallback: None,
            auto_skip_no_high: false,
            suffix_enabled: false,
        });
        let args = parse_args(&["--input", "x.jpg"]);
        let dir = tempdir().expect("temp directory should be created");
        let counters = ProgressCounters::default();

        let mut image = RgbaImage::from_pixel(40, 20, Rgba([80, 80, 80, 255]));
        for y in 0..20 {
            for x in 20..40 {
                let value = ((x + y) % 255) as u8;
                image.put_pixel(x, y, Rgba([value, value.wrapping_add(20), value, 255]));
            }
        }
        let detections = vec![
            sample_detection(0.0, 0.0, 18.0, 18.0, 0.8),
            sample_detection(20.0, 0.0, 18.0, 18.0, 0.9),
        ];

        process_crops(
            &DynamicImage::ImageRgba8(image),
            Path::new("input.png"),
            &detections,
            &settings,
            &filter,
            &None,
            &runtime,
            &args,
            dir.path(),
            None,
            &counters,
        );

        assert_eq!(counters.crops_saved.load(Ordering::Relaxed), 1);
        assert_eq!(
            fs::read_dir(dir.path())
                .expect("temp output directory should be readable")
                .count(),
            1
        );
    }

    #[test]
    fn process_crops_saves_to_override_target_and_updates_counter() {
        let settings = crop_settings_app();
        let runtime = no_gpu_runtime();
        let filter = Arc::new(QualityFilter::new(None));
        let args = parse_args(&["--input", "x.jpg"]);
        let dir = tempdir().expect("temp directory should be created");
        let counters = ProgressCounters::default();
        let detections = vec![sample_detection(2.0, 2.0, 10.0, 10.0, 0.9)];
        let override_target = PathBuf::from("nested/custom-name.jpg");

        process_crops(
            &sample_image(24, 24),
            Path::new("input.png"),
            &detections,
            &settings,
            &filter,
            &None,
            &runtime,
            &args,
            dir.path(),
            Some(&override_target),
            &counters,
        );

        assert_eq!(counters.crops_saved.load(Ordering::Relaxed), 1);
        assert_eq!(counters.crops_skipped_quality.load(Ordering::Relaxed), 0);
        assert!(dir.path().join("nested").join("custom-name.png").exists());
    }

    #[test]
    fn process_crops_handles_parent_directory_creation_errors() {
        let settings = crop_settings_app();
        let runtime = no_gpu_runtime();
        let filter = Arc::new(QualityFilter::new(None));
        let args = parse_args(&["--input", "x.jpg"]);
        let dir = tempdir().expect("temp directory should be created");
        let blocked = dir.path().join("blocked");
        fs::write(&blocked, b"file").expect("blocked output path should be written");
        let counters = ProgressCounters::default();
        let detections = vec![sample_detection(2.0, 2.0, 10.0, 10.0, 0.9)];
        let override_target = PathBuf::from("nested/output.png");

        process_crops(
            &sample_image(24, 24),
            Path::new("input.png"),
            &detections,
            &settings,
            &filter,
            &None,
            &runtime,
            &args,
            &blocked,
            Some(&override_target),
            &counters,
        );

        assert_eq!(counters.crops_saved.load(Ordering::Relaxed), 0);
        assert_eq!(counters.crops_skipped_quality.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn process_single_image_handles_mapping_rows_and_annotation_output() {
        let Some(detector) = build_test_detector() else {
            return;
        };
        let dir = tempdir().expect("temp directory should be created");
        let image_path = dir.path().join("single.png");
        write_sample_png(&image_path);
        let target = input::ProcessingItem {
            source: image_path.clone(),
            output_override: None,
            mapping_row: Some(7),
        };
        let annotate_dir = Arc::new(Some(dir.path().join("annotated")));
        let settings = crop_settings_app();
        let quality_filter = Arc::new(QualityFilter::new(None));
        let runtime = no_gpu_runtime();
        let args = parse_args(&[
            "--input",
            image_path
                .to_str()
                .expect("temp image path should be valid UTF-8"),
        ]);
        let crop_output_dir = Arc::new(None);
        let counters = ProgressCounters::default();

        let result = process_single_image(
            &target,
            &detector,
            &annotate_dir,
            &settings,
            &quality_filter,
            &None,
            &runtime,
            &args,
            false,
            &crop_output_dir,
            &counters,
        )
        .expect("processing one test image should succeed");

        assert_eq!(counters.images_processed.load(Ordering::Relaxed), 1);
        assert_eq!(result.image, image_path.display().to_string());
        assert!(
            result
                .annotated
                .as_ref()
                .is_some_and(|p| p.ends_with("single.png"))
        );
    }
}
