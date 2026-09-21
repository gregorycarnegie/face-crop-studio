//! Command-line interface for running face detection.

/// See the `mimalloc` note in the workspace Cargo.toml: tract's per-node tensor churn is
/// pathological on the Windows system heap, and swapping the allocator is worth ~35% of a
/// single detection.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::{fs, sync::Arc};

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use fcs_utils::{configure_telemetry, init_logging, normalize_path, resolve_data_path};
use log::info;
use rayon::prelude::*;

mod annotate;
mod args;
mod color;
mod config;
mod detector;
mod enhancement;
mod gpu;
mod input;
mod output_path;
mod quality;
mod types;
mod watch;
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

    // Check if webcam mode is enabled
    if args.webcam {
        let detector = build_cli_detector(&model_path, &settings.detection)?;
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

    // Watch mode has no list up front: the directory supplies one as files arrive.
    let processing_items = if args.watch.is_some() {
        Vec::new()
    } else if let Some(mapping_file) = args.mapping_file.as_ref() {
        collect_mapping_targets(mapping_file, &args)?
    } else {
        let input_arg = args
            .input
            .as_ref()
            .ok_or_else(|| anyhow!("--input is required when --mapping-file is not provided"))?;
        let input_path = normalize_path(input_arg)?;
        collect_standard_targets(&input_path)?
    };
    if args.watch.is_none() && processing_items.is_empty() {
        anyhow::bail!("no images were queued for processing");
    }

    let detector = build_cli_detector(&model_path, &settings.detection)?;

    if args.mapping_file.is_some() && !args.crop {
        info!(
            "Mapping loaded without --crop; output overrides will be applied when cropping is executed."
        );
    }

    if args.watch.is_none() {
        info!("Processing {} target(s)...", processing_items.len());
    }

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

    // Loaded only when it would be used: the refiner's single job is the eye line, and nothing
    // reads that unless crops are being levelled. `None` here is ordinary rather than an error
    // -- see `fcs_core::EyeRefiner::load`.
    let eye_refiner = shared_settings
        .crop
        .eye_line_align
        .then(fcs_core::EyeRefiner::load)
        .flatten();

    let counters = ProgressCounters::default();

    let batch_ctx = workflow::BatchContext {
        settings: &shared_settings,
        quality_filter: &quality_filter,
        enhancement_settings: &enhancement_settings,
        runtime: &gpu_runtime,
        args: &args,
        counters: &counters,
        eye_refiner: &eye_refiner,
    };

    // ponytail: Rayon's default pool, one worker per logical processor. No pool is built here,
    // so `RAYON_NUM_THREADS` overrides it.
    //
    // **The ranking has inverted since this was last measured, and the default is no longer the
    // fastest setting on this machine.** Experiment 60 found 32 workers at 7.85 s against 16 at
    // 8.9 s over the 1239-image folder, and that is why the default was kept. Re-measured on
    // 2026-09-21 with SCRFD in place of YuNet, alternated: 16 workers 10.14 / 10.55 s against
    // 32 at 10.99 / 11.05 s, so 16 now wins by 5-8%. It is not the atomic write added in 2.0 --
    // with nothing written at all, 16 takes 8.79 s against 32 at 9.73 s -- so the change is in
    // the detection path, which is the part that was replaced.
    //
    // Still not capped automatically, for the reason that held before: 5-8% on one machine is
    // thin, "physical cores" counts P and E cores alike, and this number has now moved twice as
    // soon as the work around it changed. See docs/ENGINE_SPEED.md and "Batch worker threads" in
    // README.md, both of which carry the current figures.
    if let Some(watch_dir) = args.watch.as_ref() {
        return watch::run(
            watch_dir,
            &batch_ctx,
            &detector,
            &annotate_dir,
            crop_enabled,
            &crop_output_dir,
        );
    }

    let results: Vec<ImageDetections> = processing_items
        .par_iter()
        .filter_map(|target| {
            process_single_image(
                &batch_ctx,
                target,
                &detector,
                &annotate_dir,
                crop_enabled,
                &crop_output_dir,
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
        // Serialised to memory and then replaced atomically: writing straight into a
        // `File::create` truncates the previous run's JSON before the new one is written, so a
        // failure part-way left neither.
        let json = serde_json::to_vec_pretty(&results).context("failed to serialize detections")?;
        fcs_utils::write_atomically(json_path, &json).with_context(|| {
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
