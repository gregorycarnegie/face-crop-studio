//! Command-line interface for running YuNet face detection.

use std::{
    fs::{self, File},
    sync::Arc,
};

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use fcs_core::{PostprocessConfig, PreprocessConfig};
use fcs_utils::{configure_telemetry, init_logging, normalize_path, resolve_data_path};
use log::info;
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
mod workflow;

pub(crate) use workflow::ProcessedCrop;

use args::DetectArgs;
use config::{apply_cli_overrides, load_settings};
use detector::build_cli_detector;
use enhancement::build_enhancement_settings;
use gpu::init_cli_gpu_runtime;
use input::{collect_mapping_targets, collect_standard_targets};
use quality::build_quality_filter;
use types::{ImageDetections, ProgressCounters};
use workflow::process_single_image;

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
