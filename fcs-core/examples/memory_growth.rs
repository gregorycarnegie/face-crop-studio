//! What the caches and pools retain as a session goes on.
//!
//! Experiment 6 asked for peak RAM and VRAM and left them unmeasured; experiments 83 and 84
//! ask whether the caches and pools grow without bound. This walks a real corpus in an order
//! designed to be unkind -- largest source first, so every pool sizes itself to the worst
//! case and then never sees it again -- and reports what is still held afterwards.
//!
//! Three things could grow: the GPU buffer pool (bounded by `max_idle_bytes`), the
//! preprocessor's work buffers (a texture that `ensure_texture` only ever enlarges), and the
//! convolution uniform and bind-group caches (keyed by content and by buffer identity).
//! The detection graph is fixed at 640x640 whatever the source is, so the last of those can
//! only grow through the buffer pool handing out new buffers.
//!
//! `--passes N` walks the corpus N times in **one process**, which is the question
//! experiment 85 asks and a single pass cannot answer. Every batch job the CLI runs is its
//! own process, so a per-run measurement can never see a leak; the GUI is the long-lived
//! one, and this is the stand-in for it. Per-pass wall time, retained memory and detection
//! count together show drift, growth, and any change in what the detector finds.
//!
//!   cargo run --release -p fcs-core --example memory_growth -- <dir> [limit] [--passes N]

use std::sync::Arc;

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

/// Windows reports the process working set; this is enough to see growth, not a heap profile.
#[cfg(windows)]
fn working_set_mb() -> f64 {
    use std::mem::MaybeUninit;
    #[repr(C)]
    struct Counters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }
    unsafe extern "system" {
        fn GetCurrentProcess() -> isize;
        fn K32GetProcessMemoryInfo(process: isize, counters: *mut Counters, cb: u32) -> i32;
    }
    let mut c = MaybeUninit::<Counters>::zeroed();
    unsafe {
        let size = std::mem::size_of::<Counters>() as u32;
        if K32GetProcessMemoryInfo(GetCurrentProcess(), c.as_mut_ptr(), size) == 0 {
            return f64::NAN;
        }
        c.assume_init().working_set_size as f64 / (1024.0 * 1024.0)
    }
}

#[cfg(not(windows))]
fn working_set_mb() -> f64 {
    f64::NAN
}

/// The value after `name` on the command line, if it is there.
fn flag_value(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    let index = args.iter().position(|a| a == name)?;
    args.get(index + 1).cloned()
}

fn main() -> Result<()> {
    let positional: Vec<String> = std::env::args()
        .skip(1)
        .take_while(|a| !a.starts_with("--"))
        .collect();
    let dir = positional
        .first()
        .cloned()
        .unwrap_or_else(|| "fixtures/images".into());
    let limit: usize = positional
        .get(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(400);
    let passes: usize = flag_value("--passes")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1)
        .max(1);

    let context = match GpuContext::init_with_fallback(&GpuContextOptions::default()) {
        GpuAvailability::Available(ctx) => ctx,
        other => anyhow::bail!("GPU unavailable: {other:?}"),
    };
    let preprocessor: Arc<dyn Preprocessor> =
        Arc::new(WgpuPreprocessor::new(context.clone()).context("build GPU preprocessor")?);
    let model_path = fcs_utils::model_path(MODEL)
        .context("resolve model")?
        .context("model not found")?;
    let detector = YuNetDetector::with_gpu_preprocessor(
        &model_path,
        PreprocessConfig {
            input_size: INPUT,
            resize_quality: ResizeQuality::Quality,
        },
        PostprocessConfig::default(),
        preprocessor,
    )?;

    // Largest first: every pool sizes itself to the worst case immediately, so anything that
    // only grows shows up as a plateau rather than a slow climb, and a pool that never trims
    // is visible as memory held long after the big images are gone.
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .with_context(|| format!("read {dir}"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png"))
        })
        .collect();
    files.sort_by_key(|p| std::cmp::Reverse(std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)));
    files.truncate(limit);
    anyhow::ensure!(!files.is_empty(), "no images in {dir}");

    println!("{} images from {dir}, largest first\n", files.len());
    let baseline_rss = working_set_mb();
    let mut biggest_mp = 0.0f64;

    if passes > 1 {
        // Sustained operation (experiment 85). One row per pass over the same corpus, so
        // wall time, retained memory and detections are all comparable pass to pass and
        // any drift is a slope rather than a single number. Every CLI batch job is its own
        // process, so no per-run measurement can see a leak; the GUI is the long-lived one
        // and this stands in for it.
        println!(
            "{passes} passes over the same {} images, in one process",
            files.len()
        );
        println!(
            "{:>6} {:>10} {:>12} {:>12} {:>12}",
            "pass", "wall s", "detections", "GPU pool MB", "host RSS MB"
        );
        let mut first_detections: Option<usize> = None;
        for pass in 1..=passes {
            let started = std::time::Instant::now();
            let mut detections = 0usize;
            for path in &files {
                let Ok(image) = image::open(path) else {
                    continue;
                };
                let mp = (image.width() as f64 * image.height() as f64) / 1e6;
                biggest_mp = biggest_mp.max(mp);
                detections += detector
                    .detect_image(&image)
                    .context("detect")?
                    .detections
                    .len();
            }
            let first = *first_detections.get_or_insert(detections);
            anyhow::ensure!(
                detections == first,
                "pass {pass} found {detections} faces, pass 1 found {first}: the same images stopped producing the same answer"
            );
            println!(
                "{pass:>6} {:>10.2} {detections:>12} {:>12.1} {:>12.1}",
                started.elapsed().as_secs_f64(),
                detector.gpu_memory_usage().unwrap_or(0) as f64 / (1024.0 * 1024.0),
                working_set_mb()
            );
        }
    } else {
        println!(
            "{:>7} {:>12} {:>12} {:>12}",
            "images", "source MP", "GPU pool MB", "host RSS MB"
        );
        for (index, path) in files.iter().enumerate() {
            let Ok(image) = image::open(path) else {
                continue;
            };
            let mp = (image.width() as f64 * image.height() as f64) / 1e6;
            biggest_mp = biggest_mp.max(mp);
            detector.detect_image(&image).context("detect")?;
            let n = index + 1;
            if n == 1 || n == 10 || n == 50 || n % 100 == 0 || n == files.len() {
                println!(
                    "{n:>7} {mp:>12.2} {:>12.1} {:>12.1}",
                    detector.gpu_memory_usage().unwrap_or(0) as f64 / (1024.0 * 1024.0),
                    working_set_mb()
                );
            }
        }
    }

    println!(
        "\nlargest source seen {biggest_mp:.2} MP; host RSS grew {:.1} MB from {:.1} at start.",
        working_set_mb() - baseline_rss,
        baseline_rss
    );
    Ok(())
}
