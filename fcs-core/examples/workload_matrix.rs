//! Where does the time go across the shapes of input this application actually sees?
//!
//! Experiment 6. Nearly everything else in this backlog was measured on one folder of large
//! JPEGs, so its conclusions are only known to hold there. This walks a corpus one image at a
//! time -- decode, then detect -- and buckets the results by resolution, orientation, container
//! format and face count, so a bucket that behaves differently has somewhere to show up.
//!
//! Single-threaded on purpose. These are latencies, and a contended measurement is not one;
//! throughput is reported separately at the end, from the same corpus.
//!
//! Not measured here: peak RAM and VRAM, which experiment 6 also asks for. Doing that honestly
//! needs process-level sampling rather than a timer around a call.
//!
//!   cargo run --release -p fcs-core --example workload_matrix -- <dir> [more dirs...]

use std::collections::BTreeMap;
use std::time::Instant;

use anyhow::{Context, Result};
use fcs_core::{
    InputSize, PostprocessConfig, PreprocessConfig, Preprocessor, WgpuPreprocessor, YuNetDetector,
};
use fcs_utils::{
    gpu::{GpuAvailability, GpuContext, GpuContextOptions},
    is_supported_image_path, load_image,
};
use rayon::prelude::*;
use std::sync::Arc;

const MODEL: &str = "models/face_detection_yunet_2023mar_640.onnx";

struct Sample {
    decode_ms: f64,
    detect_ms: f64,
    megapixels: f64,
    portrait: bool,
    format: String,
    faces: usize,
}

#[derive(Default)]
struct Bucket {
    decode: Vec<f64>,
    detect: Vec<f64>,
    megapixels: f64,
}

impl Bucket {
    fn push(&mut self, s: &Sample) {
        self.decode.push(s.decode_ms);
        self.detect.push(s.detect_ms);
        self.megapixels += s.megapixels;
    }
}

fn pct(sorted: &[f64], f: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[((sorted.len() as f64 * f) as usize).min(sorted.len() - 1)]
}

fn report(title: &str, buckets: &BTreeMap<String, Bucket>) {
    println!("\n{title}");
    println!(
        "{:<22} {:>6} {:>8} {:>10} {:>10} {:>10} {:>10}",
        "bucket", "n", "mean MP", "dec p50", "dec p95", "det p50", "det p95"
    );
    for (name, b) in buckets {
        let mut dec = b.decode.clone();
        let mut det = b.detect.clone();
        dec.sort_by(f64::total_cmp);
        det.sort_by(f64::total_cmp);
        let n = dec.len();
        println!(
            "{name:<22} {n:>6} {:>8.1} {:>10.1} {:>10.1} {:>10.2} {:>10.2}",
            b.megapixels / n as f64,
            pct(&dec, 0.5),
            pct(&dec, 0.95),
            pct(&det, 0.5),
            pct(&det, 0.95),
        );
    }
}

fn main() -> Result<()> {
    let dirs: Vec<String> = std::env::args().skip(1).collect();
    anyhow::ensure!(!dirs.is_empty(), "usage: workload_matrix <dir> [more dirs]");

    let mut paths: Vec<std::path::PathBuf> = Vec::new();
    for dir in &dirs {
        for entry in std::fs::read_dir(dir)? {
            let path = entry?.path();
            if path.is_file() && is_supported_image_path(&path) {
                paths.push(path);
            }
        }
    }
    paths.sort();
    println!("{} images across {} director(ies)", paths.len(), dirs.len());

    let config = PreprocessConfig {
        input_size: InputSize::new(640, 640),
        resize_quality: fcs_utils::config::ResizeQuality::Quality,
    };
    // The same pairing the CLI and GUI build: GPU inference behind a GPU preprocessor.
    // `new_gpu` pairs GPU inference with the *CPU* preprocessor instead, which production does
    // not do and which measured about 3.5 ms slower per image here.
    let context = match GpuContext::init_with_fallback(&GpuContextOptions::default()) {
        GpuAvailability::Available(ctx) => ctx,
        GpuAvailability::Disabled { reason } => anyhow::bail!("GPU disabled: {reason}"),
        GpuAvailability::Unavailable { error } => anyhow::bail!("GPU unavailable: {error}"),
    };
    let preprocessor: Arc<dyn Preprocessor> =
        Arc::new(WgpuPreprocessor::new(context).context("build GPU preprocessor")?);
    let detector = YuNetDetector::with_gpu_preprocessor(
        MODEL,
        config.clone(),
        PostprocessConfig::default(),
        preprocessor,
    )?;
    println!("backend: {}\n", detector.inference_backend());

    // Warm: the first detection pays for pipeline creation and pool growth.
    if let Some(first) = paths.first()
        && let Ok(image) = load_image(first)
    {
        for _ in 0..3 {
            let _ = detector.detect_image(&image);
        }
    }

    let mut samples = Vec::with_capacity(paths.len());
    for path in &paths {
        let start = Instant::now();
        let Ok(image) = load_image(path) else {
            continue;
        };
        let decode_ms = start.elapsed().as_secs_f64() * 1e3;

        let start = Instant::now();
        let Ok(output) = detector.detect_image(&image) else {
            continue;
        };
        let detect_ms = start.elapsed().as_secs_f64() * 1e3;

        let (w, h) = (image.width(), image.height());
        samples.push(Sample {
            decode_ms,
            detect_ms,
            megapixels: f64::from(w) * f64::from(h) / 1e6,
            portrait: h > w,
            format: path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("?")
                .to_ascii_lowercase(),
            faces: output.detections.len(),
        });
    }
    println!("{} measured\n", samples.len());

    let mut by_size: BTreeMap<String, Bucket> = BTreeMap::new();
    let mut by_shape: BTreeMap<String, Bucket> = BTreeMap::new();
    let mut by_format: BTreeMap<String, Bucket> = BTreeMap::new();
    let mut by_faces: BTreeMap<String, Bucket> = BTreeMap::new();

    for s in &samples {
        let size = match s.megapixels {
            mp if mp < 1.0 => "1. under 1 MP",
            mp if mp < 4.0 => "2. 1-4 MP",
            mp if mp < 8.0 => "3. 4-8 MP",
            mp if mp < 16.0 => "4. 8-16 MP",
            _ => "5. over 16 MP",
        };
        by_size.entry(size.into()).or_default().push(s);
        by_shape
            .entry(if s.portrait { "portrait" } else { "landscape" }.into())
            .or_default()
            .push(s);
        by_format.entry(s.format.clone()).or_default().push(s);
        let faces = match s.faces {
            0 => "0 faces",
            1 => "1 face",
            2..=3 => "2-3 faces",
            _ => "4+ faces",
        };
        by_faces.entry(faces.into()).or_default().push(s);
    }

    report("By resolution", &by_size);
    report("By orientation", &by_shape);
    report("By container format", &by_format);
    report("By face count", &by_faces);

    // Repeating one image is the cheap way to benchmark a detector, and it is not the same
    // measurement: pools, caches and the allocator are all warm for that exact shape by the
    // second iteration. A stream of distinct images is what the application actually does.
    let large: Vec<_> = samples
        .iter()
        .zip(&paths)
        .filter(|(s, _)| s.megapixels >= 4.0)
        .map(|(_, p)| p.clone())
        .take(24)
        .collect();
    if large.len() >= 8 {
        let images: Vec<_> = large.iter().filter_map(|p| load_image(p).ok()).collect();

        let mut fresh = Vec::new();
        for image in &images {
            let start = Instant::now();
            let _ = detector.detect_image(image);
            fresh.push(start.elapsed().as_secs_f64() * 1e3);
        }

        let mut repeat = Vec::new();
        for _ in 0..images.len() {
            let start = Instant::now();
            let _ = detector.detect_image(&images[0]);
            repeat.push(start.elapsed().as_secs_f64() * 1e3);
        }

        fresh.sort_by(f64::total_cmp);
        repeat.sort_by(f64::total_cmp);
        println!(
            "
Distinct images vs one repeated, {} samples each, 4 MP and up:",
            images.len()
        );
        println!(
            "  {} distinct images once each : p50 {:.2} ms  p95 {:.2} ms",
            images.len(),
            pct(&fresh, 0.5),
            pct(&fresh, 0.95)
        );
        println!(
            "  the first image {} times     : p50 {:.2} ms  p95 {:.2} ms",
            images.len(),
            pct(&repeat, 0.5),
            pct(&repeat, 0.95)
        );
    }

    // Throughput from the same corpus, at the concurrency a folder export uses.
    let start = Instant::now();
    let done: usize = paths
        .par_iter()
        .filter(|p| {
            load_image(p)
                .ok()
                .and_then(|img| detector.detect_image(&img).ok())
                .is_some()
        })
        .count();
    let secs = start.elapsed().as_secs_f64();
    println!(
        "\nThroughput: {done} images in {secs:.2} s = {:.1} images/s across {} threads",
        done as f64 / secs,
        rayon::current_num_threads()
    );
    Ok(())
}
