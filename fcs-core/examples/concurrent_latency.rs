//! What one detection costs while N others are in flight on the same device.
//!
//! Experiment 16 needs this and nothing else could supply it. `phase_timings` measures one
//! detection at a time, so it cannot see a wait that is long only because other work is
//! queued; a CLI folder job has the concurrency but measuring it needs telemetry, and
//! `env_logger` takes a lock on stderr, so 32 workers writing a line per phase serialise on
//! that lock and the tail being measured becomes the tail of the logger.
//!
//! So: images are decoded once up front, N threads then loop over them calling
//! `detect_image`, and every call is timed in-process with nothing written until the end.
//! `--ab VAR` alternates an environment flag between blocks, the way `phase_timings` does,
//! because run-to-run drift on a shared GPU is larger than what these candidates change.
//!
//! `--rayon` dispatches through a rayon pool instead of plain threads, which is how both
//! the CLI and the GUI create their concurrency -- and it is not cosmetic: `threading_pays`
//! takes a different branch on a rayon worker, so the two shapes exercise different resize
//! paths.
//!
//!   cargo run --release -p fcs-core --example concurrent_latency -- <dir> [--threads N]
//!       [--images N] [--rounds N] [--rayon] [--ab VAR]

use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result};
use fcs_core::{
    InputSize, PostprocessConfig, PreprocessConfig, Preprocessor, WgpuPreprocessor, YuNetDetector,
};
use fcs_utils::{
    config::ResizeQuality,
    gpu::{GpuAvailability, GpuContext, GpuContextOptions},
};
use image::DynamicImage;
use rayon::prelude::*;

#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

const MODEL: &str = "models/face_detection_yunet_2023mar_640.onnx";
const INPUT: InputSize = InputSize::new(640, 640);

/// Blocks per variant when alternating, and detections per thread per block.
const AB_BLOCKS: usize = 6;

fn flag_value(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    let index = args.iter().position(|a| a == name)?;
    args.get(index + 1).cloned()
}

fn flag_usize(name: &str, default: usize) -> usize {
    flag_value(name)
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
        .max(1)
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

/// One round: `threads` threads each detect every image `rounds` times.
///
/// Returns every individual latency in milliseconds, plus the wall time of the round, so
/// latency and throughput can be reported separately -- they do not move together here.
///
/// `rayon` picks how the concurrency is created, and it is not a detail. `threading_pays`
/// in `image_utils` short-circuits to true when it is already on a rayon worker, so a
/// detection dispatched through rayon takes a different resize path from one on a plain
/// thread -- the latter hops into a process-wide one-thread pool instead. Both the CLI
/// (`par_iter` over the folder) and the GUI (`rayon::spawn`) are the rayon case, so that
/// is the one that mirrors production; plain threads are kept because the difference
/// between them is the measurement (experiment 24).
fn run_round(
    detector: &Arc<YuNetDetector>,
    images: &Arc<Vec<DynamicImage>>,
    threads: usize,
    rounds: usize,
    rayon_dispatch: bool,
) -> Result<(Vec<f64>, f64)> {
    let collected: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::new()));
    let one = |image: &DynamicImage| {
        let t = Instant::now();
        // A failure here is a real one: the inputs are already decoded.
        detector.detect_image(image).expect("detect");
        t.elapsed().as_secs_f64() * 1e3
    };

    let started = Instant::now();
    if rayon_dispatch {
        // One task per detection over a pool of `threads` workers, which is the shape of
        // the CLI's `processing_items.par_iter()`.
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .context("build the dispatch pool")?;
        let work: Vec<&DynamicImage> = (0..rounds).flat_map(|_| images.iter()).collect();
        let collected = Arc::clone(&collected);
        pool.install(|| {
            let values: Vec<f64> = work.par_iter().map(|image| one(image)).collect();
            collected.lock().expect("collect").extend(values);
        });
    } else {
        std::thread::scope(|scope| {
            for _ in 0..threads {
                let images = Arc::clone(images);
                let collected = Arc::clone(&collected);
                let one = &one;
                scope.spawn(move || {
                    let mut mine = Vec::with_capacity(images.len() * rounds);
                    for _ in 0..rounds {
                        for image in images.iter() {
                            mine.push(one(image));
                        }
                    }
                    collected.lock().expect("collect").extend(mine);
                });
            }
        });
    }
    let wall = started.elapsed().as_secs_f64() * 1e3;
    let values = Arc::try_unwrap(collected)
        .expect("threads joined")
        .into_inner()
        .expect("collect");
    Ok((values, wall))
}

fn report(label: &str, mut values: Vec<f64>, wall_ms: f64) {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    let mean = values.iter().sum::<f64>() / n as f64;
    println!(
        "{label:<12} n={n:<6} p50={:>7.3} p90={:>7.3} p95={:>7.3} p99={:>8.3} max={:>8.3} mean={:>7.3}  {:>7.1} det/s",
        quantile(&values, 0.50),
        quantile(&values, 0.90),
        quantile(&values, 0.95),
        quantile(&values, 0.99),
        values[n - 1],
        mean,
        n as f64 / (wall_ms / 1e3),
    );
}

fn main() -> Result<()> {
    let dir = std::env::args()
        .nth(1)
        .filter(|a| !a.starts_with("--"))
        .unwrap_or_else(|| "fixtures/images".into());
    let threads = flag_usize("--threads", 32);
    let image_count = flag_usize("--images", 24);
    let rounds = flag_usize("--rounds", 4);
    // Plain threads by default so the two dispatch shapes stay comparable across runs.
    let rayon_dispatch = std::env::args().any(|a| a == "--rayon");

    let context = match GpuContext::init_with_fallback(&GpuContextOptions::default()) {
        GpuAvailability::Available(ctx) => ctx,
        other => anyhow::bail!("GPU unavailable: {other:?}"),
    };
    println!("adapter: {}", context.adapter_info().name);

    let preprocessor: Arc<dyn Preprocessor> =
        Arc::new(WgpuPreprocessor::new(context.clone()).context("build GPU preprocessor")?);
    let model_path = fcs_utils::model_path(MODEL)
        .context("resolve model")?
        .context("model not found")?;
    let detector = Arc::new(YuNetDetector::with_gpu_preprocessor(
        &model_path,
        PreprocessConfig {
            input_size: INPUT,
            resize_quality: ResizeQuality::Quality,
        },
        PostprocessConfig::default(),
        preprocessor,
    )?);

    // Decoded once, before any timing: decode is not what this measures, and 32 threads
    // decoding would contend for memory bandwidth and hide the thing that is.
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .with_context(|| format!("read {dir}"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png"))
        })
        .collect();
    files.sort();
    files.truncate(image_count);
    anyhow::ensure!(!files.is_empty(), "no images in {dir}");
    let images: Arc<Vec<DynamicImage>> = Arc::new(
        files
            .iter()
            .filter_map(|p| image::open(p).ok())
            .collect::<Vec<_>>(),
    );
    anyhow::ensure!(!images.is_empty(), "nothing decoded from {dir}");
    println!(
        "{} images, {threads} threads, {rounds} rounds per block, dispatch: {}",
        images.len(),
        if rayon_dispatch {
            "rayon"
        } else {
            "plain threads"
        }
    );

    // Warm: pipelines, pools and the file cache all settle on the first pass.
    run_round(&detector, &images, threads, 1, rayon_dispatch)?;

    let Some(var) = flag_value("--ab") else {
        let (values, wall) = run_round(&detector, &images, threads, rounds, rayon_dispatch)?;
        println!();
        report("latency", values, wall);
        // Each in-flight inference parks its intermediates in its own execution scope, so the
        // pool's high water is a function of concurrency, which a serial probe cannot see
        // (experiments 21 and 40).
        println!(
            "GPU pool after the round: {:.1} MB",
            detector.gpu_memory_usage().unwrap_or(0) as f64 / (1024.0 * 1024.0)
        );
        return Ok(());
    };

    println!("\nin-process A/B on {var}: {AB_BLOCKS} blocks per variant, alternated\n");
    let (mut off, mut on) = (Vec::new(), Vec::new());
    let (mut off_wall, mut on_wall) = (0.0, 0.0);
    for block in 0..AB_BLOCKS * 2 {
        let enabled = block % 2 == 1;
        // SAFETY: single-threaded at this point; the worker threads of the previous block
        // have all been joined by `run_round` before this runs.
        unsafe {
            if enabled {
                std::env::set_var(&var, "1");
            } else {
                std::env::remove_var(&var);
            }
        }
        let (values, wall) = run_round(&detector, &images, threads, rounds, rayon_dispatch)?;
        if enabled {
            on.extend(values);
            on_wall += wall;
        } else {
            off.extend(values);
            off_wall += wall;
        }
    }
    // SAFETY: as above; leave the environment as it was found.
    unsafe { std::env::remove_var(&var) };

    report("off", off, off_wall);
    report("on", on, on_wall);
    Ok(())
}
