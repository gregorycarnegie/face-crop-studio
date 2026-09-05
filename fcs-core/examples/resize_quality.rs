//! What a cheaper resize costs in detection quality (experiments 51, 54, 74).
//!
//! The source resize is the largest single cost in detecting a large image, and the cheap
//! alternatives -- `SuperSampling`, `Interpolation` -- all work by looking at fewer source
//! pixels. This repository has already been bitten by exactly that on the GPU side, where a
//! fixed small kernel "moved real detections (23px on a landmark)", so no resize change can
//! be adopted on a speed measurement alone.
//!
//! This runs the whole detector twice over a corpus -- once with production's resize, once
//! with a candidate -- and reports what changed. There is no ground truth here and none is
//! needed: production *is* the reference, and the question is whether the candidate moves
//! detections relative to it.
//!
//! What to look at, in order:
//!   * faces lost / gained -- a face that stops being detected is not a rounding error
//!   * landmark p95 and max displacement, in source pixels -- crops are placed from these
//!   * box IoU minimum -- a low minimum means one face moved a long way
//!
//! Timing is reported too, but only so the trade is visible; it is measured over the same
//! corpus rather than in the alternating harness `phase_timings --ab` uses, so treat it as
//! an order of magnitude rather than a precise delta.
//!
//! Run with:
//!   cargo run --release -p fcs-core --example resize_quality -- super2
//!   cargo run --release -p fcs-core --example resize_quality -- interp [image-count]

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use fcs_core::{
    Detection, InputSize, PostprocessConfig, PreprocessConfig, Preprocessor, WgpuPreprocessor,
    YuNetDetector,
};
use fcs_utils::{
    config::ResizeQuality,
    gpu::{GpuAvailability, GpuContext, GpuContextOptions},
};

#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

const MODEL: &str = "models/face_detection_yunet_2023mar_640.onnx";
const INPUT: InputSize = InputSize::new(640, 640);
const DEFAULT_IMAGES: usize = 60;
/// Below this IoU two boxes are different faces, not the same face moved.
const MATCH_IOU: f32 = 0.3;

fn iou(a: &Detection, b: &Detection) -> f32 {
    let (ab, bb) = (&a.bbox, &b.bbox);
    let x0 = ab.x.max(bb.x);
    let y0 = ab.y.max(bb.y);
    let x1 = (ab.x + ab.width).min(bb.x + bb.width);
    let y1 = (ab.y + ab.height).min(bb.y + bb.height);
    let inter = (x1 - x0).max(0.0) * (y1 - y0).max(0.0);
    let union = ab.width * ab.height + bb.width * bb.height - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    sorted[(((sorted.len() - 1) as f64) * q).round() as usize]
}

fn build(preprocessor: Arc<dyn Preprocessor>, model: &std::path::Path) -> Result<YuNetDetector> {
    let cfg = PreprocessConfig {
        input_size: INPUT,
        resize_quality: ResizeQuality::Quality,
    };
    YuNetDetector::with_gpu_preprocessor(model, cfg, PostprocessConfig::default(), preprocessor)
        .context("build detector")
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let candidate = args.next().unwrap_or_else(|| "super2".into());
    let limit: usize = args
        .next()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_IMAGES);

    let context = match GpuContext::init_with_fallback(&GpuContextOptions::default()) {
        GpuAvailability::Available(ctx) => ctx,
        GpuAvailability::Disabled { reason } => anyhow::bail!("GPU disabled: {reason}"),
        GpuAvailability::Unavailable { error } => anyhow::bail!("GPU unavailable: {error}"),
    };
    let model = fcs_utils::model_path(MODEL)
        .context("resolve model")?
        .context("model missing")?;
    let preprocessor: Arc<dyn Preprocessor> =
        Arc::new(WgpuPreprocessor::new(context).context("gpu preprocessor")?);
    let detector = build(preprocessor, &model)?;

    let mut paths: Vec<_> = std::fs::read_dir("fixtures/images")
        .context("read fixtures/images")?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("jpg")))
        .collect();
    paths.sort();
    paths.truncate(limit);
    anyhow::ensure!(!paths.is_empty(), "no .jpg fixtures under fixtures/images");

    println!(
        "candidate FCS_RESIZE_ALG={candidate}, {} images\n",
        paths.len()
    );

    let mut lost = 0usize;
    let mut gained = 0usize;
    let mut matched = 0usize;
    let mut ious: Vec<f64> = Vec::new();
    let mut landmark_px: Vec<f64> = Vec::new();
    let mut score_delta: Vec<f64> = Vec::new();
    let mut worst: Option<(f64, String)> = None;
    let (mut base_ms, mut cand_ms) = (0.0f64, 0.0f64);

    for path in &paths {
        let image = match image::open(path) {
            Ok(image) => image,
            Err(err) => {
                eprintln!("skipping {}: {err}", path.display());
                continue;
            }
        };

        // SAFETY of the env writes: single-threaded probe, and the detector reads the
        // variable inside the call below, not across threads.
        unsafe { std::env::remove_var("FCS_RESIZE_ALG") };
        detector.detect_image(&image).context("warm-up")?;
        let started = Instant::now();
        let base = detector.detect_image(&image).context("baseline detect")?;
        base_ms += started.elapsed().as_secs_f64() * 1e3;

        unsafe { std::env::set_var("FCS_RESIZE_ALG", &candidate) };
        detector.detect_image(&image).context("warm-up")?;
        let started = Instant::now();
        let cand = detector.detect_image(&image).context("candidate detect")?;
        cand_ms += started.elapsed().as_secs_f64() * 1e3;
        unsafe { std::env::remove_var("FCS_RESIZE_ALG") };

        let mut taken = vec![false; cand.detections.len()];
        for a in &base.detections {
            // Greedy nearest match; detections are few per image and already NMS'd.
            let best = cand
                .detections
                .iter()
                .enumerate()
                .filter(|(i, _)| !taken[*i])
                .map(|(i, b)| (i, iou(a, b)))
                .max_by(|x, y| x.1.total_cmp(&y.1));
            match best {
                Some((i, overlap)) if overlap >= MATCH_IOU => {
                    taken[i] = true;
                    matched += 1;
                    let b = &cand.detections[i];
                    ious.push(f64::from(overlap));
                    score_delta.push(f64::from(b.score - a.score));
                    for (la, lb) in a.landmarks.iter().zip(b.landmarks.iter()) {
                        let d = f64::from((lb.x - la.x).hypot(lb.y - la.y));
                        landmark_px.push(d);
                        if worst.as_ref().is_none_or(|(w, _)| d > *w) {
                            worst = Some((d, path.display().to_string()));
                        }
                    }
                }
                _ => lost += 1,
            }
        }
        gained += taken.iter().filter(|t| !**t).count();
    }

    ious.sort_by(f64::total_cmp);
    landmark_px.sort_by(f64::total_cmp);
    score_delta.sort_by(|a, b| a.abs().total_cmp(&b.abs()));

    println!("faces matched      {matched}");
    println!("faces lost         {lost}   (detected by production, not by the candidate)");
    println!("faces gained       {gained}   (detected by the candidate only)");
    println!();
    println!(
        "landmark shift px  p50 {:.2}   p95 {:.2}   max {:.2}",
        quantile(&landmark_px, 0.5),
        quantile(&landmark_px, 0.95),
        landmark_px.last().copied().unwrap_or(f64::NAN)
    );
    println!(
        "box IoU            min {:.4}   p05 {:.4}   p50 {:.4}",
        ious.first().copied().unwrap_or(f64::NAN),
        quantile(&ious, 0.05),
        quantile(&ious, 0.5)
    );
    println!(
        "score delta        p50 {:+.4}  max |delta| {:.4}",
        quantile(&score_delta, 0.5),
        score_delta.last().copied().unwrap_or(f64::NAN).abs()
    );
    if let Some((d, path)) = worst {
        println!("worst landmark     {d:.2} px in {path}");
    }
    println!(
        "\ndetect_image total production {base_ms:.0} ms, candidate {cand_ms:.0} ms \
         ({:.2}x)",
        base_ms / cand_ms
    );
    Ok(())
}
