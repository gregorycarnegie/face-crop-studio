//! Wall-clock breakdown of the normal GPU detection path, phase by phase.
//!
//! `gpu_pass_breakdown` answers "where does the GPU time go"; this answers "where does the
//! wall time go", which is a different question — the profiled forward pass is ~0.5 ms
//! against several ms of `detect_image`. Every phase here comes from the `timing_guard`
//! lines the runtime already emits, captured instead of printed so 30 runs collapse to a
//! median per label rather than 30 screens of log output.
//!
//! Waits are not independent costs: `readback_wait` contains the forward pass itself, so it
//! is GPU execution plus the head copies, and adding it to a GPU timestamp total double
//! counts. Children are indented under the phase that contains them.
//!
//! Run with: cargo run --release -p fcs-core --example phase_timings [image]

use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use anyhow::{Context, Result};
use fcs_core::{
    InputSize, PostprocessConfig, PreprocessConfig, Preprocessor, WgpuPreprocessor, YuNetDetector,
};
use fcs_utils::{
    config::ResizeQuality,
    gpu::{GpuAvailability, GpuContext, GpuContextOptions},
};

#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

const MODEL: &str = "models/face_detection_yunet_2023mar_640.onnx";
const INPUT: InputSize = InputSize::new(640, 640);
const RUNS: usize = 30;

/// Phases in containment order; the number is the indent depth.
const DISPLAY: &[(&str, usize)] = &[
    ("fcs_core::detect_image", 0),
    ("fcs_core::preprocess_dynamic_image", 1),
    ("fcs_core::cpu_resize", 2),
    ("fcs_core::bgr_chw", 2),
    ("fcs_core::gpu_rgb_to_chw", 2),
    ("fcs_core::detect_on_device", 1),
    ("fcs_core::allocate_input", 2),
    ("fcs_core::gpu_preprocess", 2),
    ("fcs_core::preprocess_acquire", 3),
    ("fcs_core::preprocess_to_rgba", 3),
    ("fcs_core::preprocess_upload", 3),
    ("fcs_core::preprocess_encode", 3),
    ("fcs_core::preprocess_submit", 4),
    ("fcs_core::onnx_inference", 2),
    ("fcs_core::gpu_encode", 3),
    ("fcs_core::gpu_record", 4),
    ("fcs_core::gpu_submit", 4),
    ("fcs_core::gpu_finish", 5),
    ("fcs_core::gpu_readback", 3),
    ("fcs_core::readback_alloc", 4),
    ("fcs_core::readback_copy", 4),
    ("fcs_core::readback_map", 4),
    ("fcs_core::readback_wait", 4),
    ("fcs_core::readback_collect", 4),
    ("fcs_core::gpu_convert", 4),
    ("fcs_core::gpu_decode", 3),
    ("fcs_core::postprocess", 2),
    ("fcs_core::gpu_upload", 2),
    ("fcs_core::run_preprocessed", 1),
];

type Samples = Mutex<Vec<(String, f64)>>;

fn samples() -> &'static Samples {
    static S: OnceLock<Samples> = OnceLock::new();
    S.get_or_init(|| Mutex::new(Vec::new()))
}

/// Collects the guard lines rather than printing them.
struct Collector;

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
            samples().lock().unwrap().push((label.to_string(), ms));
        }
    }
    fn flush(&self) {}
}

/// `{:.2?}` on a `Duration` picks its own unit, so the unit has to be read back off.
fn parse_ms(text: &str) -> Option<f64> {
    let text = text.trim();
    let (value, scale) = if let Some(v) = text.strip_suffix("ms") {
        (v, 1.0)
    } else if let Some(v) = text.strip_suffix("µs") {
        (v, 1e-3)
    } else if let Some(v) = text.strip_suffix("ns") {
        (v, 1e-6)
    } else {
        (text.strip_suffix('s')?, 1e3)
    };
    value.trim().parse::<f64>().ok().map(|v| v * scale)
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx]
}

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .filter(|a| !a.starts_with("--"))
        .unwrap_or_else(|| "fixtures/images/006.jpg".into());

    log::set_boxed_logger(Box::new(Collector))?;
    log::set_max_level(log::LevelFilter::Trace);
    fcs_utils::telemetry::configure(true, log::LevelFilter::Trace);

    let context = match GpuContext::init_with_fallback(&GpuContextOptions::default()) {
        GpuAvailability::Available(ctx) => ctx,
        GpuAvailability::Disabled { reason } => anyhow::bail!("GPU disabled: {reason}"),
        GpuAvailability::Unavailable { error } => anyhow::bail!("GPU unavailable: {error}"),
    };
    println!(
        "adapter: {} ({:?})",
        context.adapter_info().name,
        context.adapter_info().backend
    );

    let model_path = fcs_utils::model_path(MODEL)
        .context("resolve YuNet model")?
        .with_context(|| format!("model {MODEL} not found"))?;
    let cfg = PreprocessConfig {
        input_size: INPUT,
        resize_quality: ResizeQuality::Quality,
    };
    let preprocessor: Arc<dyn Preprocessor> =
        Arc::new(WgpuPreprocessor::new(context.clone()).context("build GPU preprocessor")?);
    let detector = YuNetDetector::with_gpu_preprocessor(
        &model_path,
        cfg,
        PostprocessConfig::default(),
        preprocessor,
    )
    .context("build GPU detector")?;
    println!("backend: {}", detector.inference_backend());

    let mut image = image::open(&path).with_context(|| format!("open {path}"))?;
    // `--mp N` rescales the source before timing, so one fixture covers a size sweep. The
    // routing cutoff in `upload_pays_for_source` is a source-size decision, and finding
    // where it belongs needs sources on both sides of it (experiment 54).
    if let Some(mp) = flag_value("--mp").and_then(|v| v.parse::<f64>().ok()) {
        let scale = (mp * 1e6 / (f64::from(image.width()) * f64::from(image.height()))).sqrt();
        let (w, h) = (
            (f64::from(image.width()) * scale).round().max(1.0) as u32,
            (f64::from(image.height()) * scale).round().max(1.0) as u32,
        );
        image = image.resize_exact(w, h, image::imageops::FilterType::Lanczos3);
    }
    println!(
        "image:   {path} {}x{} ({:.2} MP)",
        image.width(),
        image.height(),
        f64::from(image.width()) * f64::from(image.height()) / 1e6
    );

    for _ in 0..5 {
        detector.detect_image(&image).context("warm-up")?;
    }
    samples().lock().unwrap().clear();

    // `--ab VAR` alternates that environment flag between short blocks inside one
    // process. Comparing across processes cannot resolve these candidates: repeated
    // identical runs drift by up to 0.08 ms on `gpu_readback` as the GPU clocks ramp,
    // which is larger than anything the readback experiments change.
    if let Some((var, value)) = ab_variable() {
        return run_ab(&detector, &image, &var, &value);
    }

    let mut wall = Vec::with_capacity(RUNS);
    let mut faces = 0;
    for _ in 0..RUNS {
        let started = Instant::now();
        let out = detector.detect_image(&image).context("timed run")?;
        wall.push(started.elapsed().as_secs_f64() * 1e3);
        faces = out.detections.len();
    }

    let collected = samples().lock().unwrap().clone();
    wall.sort_by(f64::total_cmp);
    println!(
        "\n{RUNS} runs, {faces} faces.  detect_image wall p50 {:.3} ms  p95 {:.3} ms\n",
        quantile(&wall, 0.5),
        quantile(&wall, 0.95)
    );

    println!("{:<38} {:>7} {:>9} {:>9}", "phase", "n", "p50 ms", "p95 ms");
    for (label, depth) in DISPLAY {
        let mut values: Vec<f64> = collected
            .iter()
            .filter(|(l, _)| l == label)
            .map(|(_, v)| *v)
            .collect();
        if values.is_empty() {
            continue;
        }
        values.sort_by(f64::total_cmp);
        let name = format!(
            "{}{}",
            "  ".repeat(*depth),
            label.trim_start_matches("fcs_core::")
        );
        println!(
            "{name:<38} {:>7} {:>9.3} {:>9.3}",
            values.len(),
            quantile(&values, 0.5),
            quantile(&values, 0.95)
        );
    }

    // Anything instrumented but not in the display table, so a new guard is never lost.
    let mut extra: Vec<&str> = collected
        .iter()
        .map(|(l, _)| l.as_str())
        .filter(|l| !DISPLAY.iter().any(|(d, _)| d == l))
        .collect();
    extra.sort_unstable();
    extra.dedup();
    if !extra.is_empty() {
        println!("\nnot placed in the tree above:");
        for label in extra {
            let values: Vec<f64> = collected
                .iter()
                .filter(|(l, _)| l == label)
                .map(|(_, v)| *v)
                .collect();
            println!("  {label:<36} {:>7} {:>9.3}", values.len(), median(values));
        }
    }
    Ok(())
}

/// `--ab VAR` toggles a flag on and off; `--ab VAR=VALUE` sets it to that value, for
/// candidates selected by content rather than by presence.
fn ab_variable() -> Option<(String, String)> {
    let spec = flag_value("--ab")?;
    Some(match spec.split_once('=') {
        Some((name, value)) => (name.to_string(), value.to_string()),
        None => (spec.clone(), "1".to_string()),
    })
}

/// The argument after `name`, if it is present.
fn flag_value(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    let index = args.iter().position(|a| a == name)?;
    args.get(index + 1).cloned()
}

/// Blocks per variant, alternated so a drifting clock lands on both equally.
const AB_BLOCKS: usize = 8;
const AB_BLOCK_RUNS: usize = 15;

fn run_ab(
    detector: &YuNetDetector,
    image: &image::DynamicImage,
    var: &str,
    value: &str,
) -> Result<()> {
    println!(
        "
in-process A/B on {var}={value}: {AB_BLOCKS} blocks per variant, {AB_BLOCK_RUNS} runs each"
    );

    // label -> (off samples, on samples)
    let mut off: Vec<(String, f64)> = Vec::new();
    let mut on: Vec<(String, f64)> = Vec::new();
    let mut off_wall: Vec<f64> = Vec::new();
    let mut on_wall: Vec<f64> = Vec::new();

    for block in 0..AB_BLOCKS * 2 {
        let enabled = block % 2 == 1;
        // SAFETY: single-threaded probe; nothing else reads the environment concurrently.
        unsafe {
            if enabled {
                std::env::set_var(var, value);
            } else {
                std::env::remove_var(var);
            }
        }
        samples().lock().unwrap().clear();
        let mut wall = Vec::with_capacity(AB_BLOCK_RUNS);
        for _ in 0..AB_BLOCK_RUNS {
            let started = Instant::now();
            detector.detect_image(image).context("A/B run")?;
            wall.push(started.elapsed().as_secs_f64() * 1e3);
        }
        let drained = samples().lock().unwrap().clone();
        if enabled {
            on.extend(drained);
            on_wall.push(median(wall));
        } else {
            off.extend(drained);
            off_wall.push(median(wall));
        }
    }
    // SAFETY: as above; leave the environment as it was found.
    unsafe { std::env::remove_var(var) };

    println!(
        "
detect_image block-median p50: off {:.3} ms   on {:.3} ms",
        median(off_wall.clone()),
        median(on_wall.clone())
    );
    println!(
        "off block spread {:.3}-{:.3} ms (this is the noise floor)",
        off_wall.iter().copied().fold(f64::INFINITY, f64::min),
        off_wall.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    );

    println!(
        "
{:<32} {:>9} {:>9} {:>9}",
        "phase", "off p50", "on p50", "delta"
    );
    for (label, _) in DISPLAY {
        let a: Vec<f64> = off
            .iter()
            .filter(|(l, _)| l == label)
            .map(|(_, v)| *v)
            .collect();
        let b: Vec<f64> = on
            .iter()
            .filter(|(l, _)| l == label)
            .map(|(_, v)| *v)
            .collect();
        if a.is_empty() && b.is_empty() {
            continue;
        }
        let name = label.trim_start_matches("fcs_core::");
        // A phase can exist in only one variant -- a candidate that removes it entirely.
        match (a.is_empty(), b.is_empty()) {
            (false, false) => {
                let (pa, pb) = (median(a), median(b));
                println!("{name:<32} {pa:>9.3} {pb:>9.3} {:>+9.3}", pb - pa);
            }
            (false, true) => println!("{name:<32} {:>9.3} {:>9} {:>9}", median(a), "-", "gone"),
            (true, false) => println!("{name:<32} {:>9} {:>9.3} {:>9}", "-", median(b), "new"),
            (true, true) => {}
        }
    }
    Ok(())
}
