//! What a cold start costs, stage by stage, and how much of it the first detection hides.
//!
//! Everything else in this backlog measures a warm pipeline. Nothing measured the path from
//! process launch to the first face, which is what a user actually waits for when the app
//! opens or the GUI switches to the GPU. Experiment 80 asks for that split: adapter and
//! device, shader compilation, model parse, weight upload, then the first detection against
//! the steady state -- so that lazy initialisation cannot hide in "the first user action".
//!
//! One process is one cold start, so this measures each stage once and does not average.
//! Run it twice to see what the OS file cache and any driver shader cache are worth: the
//! second run of the same binary is the "warm disk, cold process" case.
//!
//! Run with: cargo run --release -p fcs-core --example cold_start

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use fcs_core::{
    InputSize, PostprocessConfig, PreprocessConfig, Preprocessor, WgpuPreprocessor, YuNetDetector,
};
use fcs_utils::{
    config::ResizeQuality,
    gpu::{GpuAvailability, GpuContext, GpuContextOptions},
};
use std::sync::{Mutex, OnceLock};

#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

const MODEL: &str = "models/face_detection_yunet_2023mar_640.onnx";
const INPUT: InputSize = InputSize::new(640, 640);
const STEADY_RUNS: usize = 20;

/// Captures the runtime's own `timing_guard` lines so the detector stage can be split into
/// parse, compile and upload without printing a screen of log output.
struct Collector;

fn captured() -> &'static Mutex<Vec<(String, f64)>> {
    static S: OnceLock<Mutex<Vec<(String, f64)>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(Vec::new()))
}

impl log::Log for Collector {
    fn enabled(&self, m: &log::Metadata) -> bool {
        m.target() == "fcs::telemetry"
    }
    fn log(&self, record: &log::Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let text = record.args().to_string();
        if let Some((label, rest)) = text.split_once(" completed in ")
            && let Some(ms) = parse_ms(rest)
        {
            captured().lock().unwrap().push((label.to_string(), ms));
        }
    }
    fn flush(&self) {}
}

/// `{:.2?}` on a `Duration` picks its own unit, so the unit has to be read back off.
fn parse_ms(text: &str) -> Option<f64> {
    let text = text.trim();
    let (value, scale) = if let Some(v) = text.strip_suffix("ms") {
        (v, 1.0)
    } else if let Some(v) = text.strip_suffix("\u{b5}s") {
        (v, 1e-3)
    } else if let Some(v) = text.strip_suffix("ns") {
        (v, 1e-6)
    } else {
        (text.strip_suffix('s')?, 1e3)
    };
    value.trim().parse::<f64>().ok().map(|v| v * scale)
}

fn main() -> Result<()> {
    let process_start = Instant::now();
    log::set_boxed_logger(Box::new(Collector))?;
    log::set_max_level(log::LevelFilter::Trace);
    fcs_utils::telemetry::configure(true, log::LevelFilter::Trace);
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "fixtures/images/249_o.jpg".into());

    let mut stages: Vec<(&str, f64)> = Vec::new();
    let mark = |stages: &mut Vec<(&'static str, f64)>, name, at: &mut Instant| {
        stages.push((name, at.elapsed().as_secs_f64() * 1e3));
        *at = Instant::now();
    };
    let mut at = Instant::now();

    let context = match GpuContext::init_with_fallback(&GpuContextOptions::default()) {
        GpuAvailability::Available(ctx) => ctx,
        GpuAvailability::Disabled { reason } => anyhow::bail!("GPU disabled: {reason}"),
        GpuAvailability::Unavailable { error } => anyhow::bail!("GPU unavailable: {error}"),
    };
    mark(&mut stages, "adapter + device", &mut at);

    let preprocessor: Arc<dyn Preprocessor> =
        Arc::new(WgpuPreprocessor::new(context.clone()).context("build GPU preprocessor")?);
    mark(&mut stages, "preprocessor pipelines", &mut at);

    let model_path = fcs_utils::model_path(MODEL)
        .context("resolve YuNet model")?
        .with_context(|| format!("model {MODEL} not found"))?;
    let cfg = PreprocessConfig {
        input_size: INPUT,
        resize_quality: ResizeQuality::Quality,
    };
    let detector = YuNetDetector::with_gpu_preprocessor(
        &model_path,
        cfg,
        PostprocessConfig::default(),
        preprocessor,
    )
    .context("build GPU detector")?;
    mark(&mut stages, "detector (parse + compile + upload)", &mut at);

    let image = image::open(&path).with_context(|| format!("open {path}"))?;
    mark(&mut stages, "decode the first image", &mut at);

    let faces = detector.detect_image(&image).context("first detection")?;
    mark(&mut stages, "FIRST detection", &mut at);

    let mut steady = Vec::with_capacity(STEADY_RUNS);
    for _ in 0..STEADY_RUNS {
        let started = Instant::now();
        detector.detect_image(&image).context("steady detection")?;
        steady.push(started.elapsed().as_secs_f64() * 1e3);
    }
    steady.sort_by(f64::total_cmp);
    let steady_p50 = steady[steady.len() / 2];

    let to_first_face: f64 = stages.iter().map(|(_, ms)| ms).sum();
    println!(
        "adapter: {} ({:?})",
        context.adapter_info().name,
        context.adapter_info().backend
    );
    println!(
        "image:   {path} {}x{}, {} faces\n",
        image.width(),
        image.height(),
        faces.detections.len()
    );

    println!("{:<38} {:>9} {:>8}", "stage", "ms", "share");
    let inner_stages = captured().lock().unwrap().clone();
    for (name, ms) in &stages {
        println!("{name:<38} {ms:>9.2} {:>7.1}%", 100.0 * ms / to_first_face);
        if name.starts_with("adapter") {
            for label in [
                "fcs_utils::gpu_instance",
                "fcs_utils::gpu_request_adapter",
                "fcs_utils::gpu_request_device",
            ] {
                if let Some((_, inner)) = inner_stages.iter().find(|(l, _)| l == label) {
                    println!(
                        "  - {:<34} {inner:>9.2} {:>7.1}%",
                        label.trim_start_matches("fcs_utils::"),
                        100.0 * inner / to_first_face
                    );
                }
            }
        }
        // The detector constructor is the one stage with guards of its own inside.
        if name.starts_with("detector") {
            for label in [
                "fcs_core::load_onnx_weights",
                "fcs_core::compile_pipelines",
                "fcs_core::compile_conv2d",
                "fcs_core::compile_activation",
                "fcs_core::compile_max_pool",
                "fcs_core::compile_add",
                "fcs_core::compile_upsample2x",
                "fcs_core::upload_weights",
            ] {
                if let Some((_, inner)) = inner_stages.iter().find(|(l, _)| l == label) {
                    println!(
                        "  - {:<34} {inner:>9.2} {:>7.1}%",
                        label.trim_start_matches("fcs_core::"),
                        100.0 * inner / to_first_face
                    );
                }
            }
        }
    }
    println!("{:<38} {:>9.2}", "= launch to first face", to_first_face);
    println!("{:<38} {:>9.2}", "steady-state detection p50", steady_p50);
    println!(
        "{:<38} {:>9.2}",
        "process start to first face",
        process_start.elapsed().as_secs_f64() * 1e3 - steady.iter().sum::<f64>()
    );

    println!(
        "\nThe first detection costs {:.2} ms against a steady {:.2}: {:.0}x, and that excess\n\
         is whatever the constructors deferred rather than did.",
        stages
            .iter()
            .find(|(n, _)| *n == "FIRST detection")
            .map(|(_, ms)| *ms)
            .unwrap_or_default(),
        steady_p50,
        stages
            .iter()
            .find(|(n, _)| *n == "FIRST detection")
            .map(|(_, ms)| *ms)
            .unwrap_or_default()
            / steady_p50,
    );
    Ok(())
}
