//! Per-compute-pass GPU cost of one YuNet forward pass, read from the GPU's own clock.
//!
//! Wall-clock benchmarks around these dispatches are dominated by upload, readback and
//! driver overhead, and on this hardware they swing far enough run to run that identical
//! code can measure 40% apart. These numbers come from timestamp queries at pass
//! boundaries, so a shader change shows up in the pass that contains it.
//!
//! Run with: cargo run --release --example gpu_pass_breakdown

use std::{sync::Arc, time::Instant};

use anyhow::{Context, Result};
use fcs_core::{
    InputSize, PostprocessConfig, PreprocessConfig, Preprocessor, WgpuPreprocessor, YuNetDetector,
    gpu::GpuYuNet,
};
use fcs_utils::{
    config::ResizeQuality,
    gpu::{GpuAvailability, GpuContext, GpuContextOptions, PassTiming, total_by_label},
};

#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

const MODEL: &str = "models/face_detection_yunet_2023mar_640.onnx";
const FIXTURE: &str = "images/006.jpg";
const INPUT: InputSize = InputSize::new(640, 640);
/// Passes are timed per run; more runs average out clock ramping.
const RUNS: usize = 20;

fn main() -> Result<()> {
    let options = GpuContextOptions {
        profiling: true,
        ..GpuContextOptions::default()
    };
    let context = match GpuContext::init_with_fallback(&options) {
        GpuAvailability::Available(ctx) => ctx,
        GpuAvailability::Disabled { reason } => anyhow::bail!("GPU disabled: {reason}"),
        GpuAvailability::Unavailable { error } => anyhow::bail!("GPU unavailable: {error}"),
    };
    if context.profiler().is_none() {
        anyhow::bail!(
            "adapter '{}' has no TIMESTAMP_QUERY support, so passes cannot be timed",
            context.adapter_info().name
        );
    }
    println!("adapter: {}", context.adapter_info().name);

    let model_path = fcs_utils::model_path(MODEL)
        .context("resolve YuNet model")?
        .with_context(|| format!("model {MODEL} not found"))?;
    let model =
        GpuYuNet::with_context(context.clone(), &model_path, INPUT).context("build GPU YuNet")?;

    let input = model.allocate_input(INPUT).context("allocate input")?;
    let pixels = INPUT.width as usize * INPUT.height as usize;
    let data: Vec<f32> = (0..pixels * 3).map(|i| (i % 257) as f32 / 256.0).collect();
    input.write(&data).context("upload input")?;

    // Warm up: first run pays shader compilation and clock ramp, and would otherwise
    // dominate the totals.
    for _ in 0..3 {
        model.run_on_device(&input).context("warm-up run")?;
    }
    context.take_pass_timings()?;

    let mut runs: Vec<Vec<PassTiming>> = Vec::with_capacity(RUNS);
    let mut wall_ms: Vec<f64> = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let started = std::time::Instant::now();
        model.run_on_device(&input).context("profiled run")?;
        wall_ms.push(started.elapsed().as_secs_f64() * 1e3);
        runs.push(context.take_pass_timings()?);
    }

    let per_run_total: Vec<f64> = runs
        .iter()
        .map(|r| r.iter().map(|t| t.duration_ns).sum())
        .collect();
    let passes = runs.first().map_or(0, Vec::len);
    println!(
        "\n{RUNS} runs, {passes} timed passes each, median total {:.3} ms\n",
        median(per_run_total.clone()) / 1e6
    );

    // Report the median run rather than the mean, so one scheduling hiccup does not
    // shift every row.
    let median_run = runs
        .iter()
        .zip(&per_run_total)
        .min_by(|a, b| {
            let m = median(per_run_total.clone());
            (a.1 - m).abs().total_cmp(&(b.1 - m).abs())
        })
        .map(|(run, _)| run.as_slice())
        .unwrap_or(&[]);

    let rows = total_by_label(median_run);
    let run_total: f64 = rows.iter().map(|(_, _, ns)| ns).sum();
    println!(
        "{:<20} {:>6} {:>12} {:>8}",
        "pass", "count", "total (µs)", "share"
    );
    println!("{}", "-".repeat(50));
    for (label, count, ns) in &rows {
        println!(
            "{label:<20} {count:>6} {:>12.1} {:>7.1}%",
            ns / 1e3,
            if run_total > 0.0 {
                ns / run_total * 100.0
            } else {
                0.0
            }
        );
    }
    println!("{}", "-".repeat(50));
    println!("{:<20} {:>6} {:>12.1}", "total", passes, run_total / 1e3);

    // The gap between these two is encode, submit, sync and output readback — none of it
    // reachable by a shader change. Worth printing next to the shader numbers so nobody
    // optimises a pass that is already a rounding error on the call.
    let wall = median(wall_ms);
    let gpu = median(per_run_total) / 1e6;
    println!(
        "\nrun_on_device wall {wall:.3} ms, of it {gpu:.3} ms GPU compute ({:.1}%)",
        if wall > 0.0 { gpu / wall * 100.0 } else { 0.0 }
    );
    println!(
        "{:.3} ms ({:.1}%) is encode + submit + sync + readback, which no shader change touches",
        wall - gpu,
        if wall > 0.0 {
            (wall - gpu) / wall * 100.0
        } else {
            0.0
        }
    );

    detect_phase_split(&context, &model_path, gpu)?;
    Ok(())
}

/// The same split one level up: what a whole `detect_image` costs, and how much of it is
/// the inference above. Run single-threaded on one image, which is how the criterion
/// bench measures it -- the CLI fans out over rayon, so its per-image times are contended
/// and several times larger.
fn detect_phase_split(
    context: &Arc<GpuContext>,
    model_path: &std::path::Path,
    gpu: f64,
) -> Result<()> {
    let image = fcs_utils::load_fixture_image(FIXTURE).context("load fixture")?;
    let preprocess = PreprocessConfig {
        input_size: INPUT,
        resize_quality: ResizeQuality::Speed,
    };
    let preprocessor = Arc::new(WgpuPreprocessor::new(context.clone())?);
    let detector = YuNetDetector::with_gpu_preprocessor(
        model_path,
        preprocess.clone(),
        PostprocessConfig::default(),
        preprocessor.clone(),
    )?;

    let mut whole = Vec::with_capacity(RUNS);
    let mut prep = Vec::with_capacity(RUNS);
    for _ in 0..RUNS + 3 {
        let started = Instant::now();
        let out = detector.detect_image(&image)?;
        let elapsed = started.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(out.detections.len());

        let started = Instant::now();
        std::hint::black_box(preprocessor.preprocess(&image, &preprocess)?);
        let prep_ms = started.elapsed().as_secs_f64() * 1e3;

        whole.push(elapsed);
        prep.push(prep_ms);
    }
    // Drop the warm-up runs from the front.
    let whole = median(whole.split_off(3));
    let prep = median(prep.split_off(3));

    println!("\ndetect_image        {whole:7.3} ms");
    println!(
        "  preprocess        {prep:7.3} ms  ({:.1}%)",
        prep / whole * 100.0
    );
    println!(
        "  inference + rest  {:7.3} ms  ({:.1}%), of which {gpu:.3} ms is GPU compute",
        whole - prep,
        (whole - prep) / whole * 100.0,
    );
    Ok(())
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}
