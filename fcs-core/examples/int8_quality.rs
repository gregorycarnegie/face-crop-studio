//! An INT8 YuNet against the bundled f32 model on the CPU inference path (experiment 77).
//!
//! The GPU graph is written in f32 WGSL and gains nothing from a quantised model file, so the
//! only place INT8 can pay is the CPU path: `--no-gpu`, or a machine with no usable adapter.
//! This runs both models over a corpus with production's settings and reports what moved,
//! the way `resize_quality` does, plus the time each model spent detecting.
//!
//! Evaluate on images the model was not calibrated on -- `quantize_yunet.py` calibrates on
//! `fixtures/images`, so point this at a different folder.
//!
//!   ORT_DYLIB_PATH=... cargo run --release -p fcs-core --example int8_quality -- \
//!       <int8.onnx> <image dir> [limit]

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use fcs_core::{Detection, InputSize, PostprocessConfig, PreprocessConfig, YuNetDetector};
use fcs_utils::config::{AppSettings, ResizeQuality};

#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

const MODEL: &str = "models/face_detection_yunet_2023mar_640.onnx";
const INPUT: InputSize = InputSize::new(640, 640);
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

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    anyhow::ensure!(
        args.len() >= 2,
        "usage: int8_quality <int8.onnx> <image dir> [limit]"
    );
    let int8 = PathBuf::from(&args[0]);
    let dir = &args[1];
    let limit: usize = args
        .get(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);

    // Production's thresholds, not the library defaults: 96 was bitten by measuring at 0.9
    // when the application runs at 0.8.
    let settings = AppSettings::load_from_path(Path::new("config/gui_settings.json"))
        .context("load config/gui_settings.json")?;
    let post: PostprocessConfig = (&settings.detection).into();
    let pre = PreprocessConfig {
        input_size: INPUT,
        resize_quality: ResizeQuality::Quality,
    };
    let f32_model = fcs_utils::model_path(MODEL)
        .context("resolve model")?
        .context("model missing")?;
    let base = YuNetDetector::new(&f32_model, pre.clone(), post.clone()).context("f32 model")?;
    let cand = YuNetDetector::new(&int8, pre, post).context("int8 model")?;
    println!(
        "backends: f32 {}, int8 {}",
        base.inference_backend(),
        cand.inference_backend()
    );

    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .with_context(|| format!("read {dir}"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "jpg" | "jpeg"))
        })
        .collect();
    paths.sort();
    paths.truncate(limit);
    anyhow::ensure!(!paths.is_empty(), "no images under {dir}");

    let (mut lost, mut gained, mut matched, mut base_faces) = (0usize, 0usize, 0usize, 0usize);
    let (mut changed_images, mut images) = (0usize, 0usize);
    let mut ious: Vec<f64> = Vec::new();
    let mut landmark_px: Vec<f64> = Vec::new();
    let mut score_delta: Vec<f64> = Vec::new();
    let mut worst: Option<(f64, String)> = None;
    let (mut base_ms, mut cand_ms) = (0.0f64, 0.0f64);

    for (n, path) in paths.iter().enumerate() {
        let Ok(image) = image::open(path) else {
            eprintln!("skipping {}", path.display());
            continue;
        };
        if n == 0 {
            base.detect_image(&image).context("warm-up")?;
            cand.detect_image(&image).context("warm-up")?;
        }
        images += 1;
        // Alternate which model goes first, so neither always meets a warm cache.
        let time = |d: &YuNetDetector, acc: &mut f64| -> Result<Vec<Detection>> {
            let started = Instant::now();
            let out = d.detect_image(&image)?;
            *acc += started.elapsed().as_secs_f64() * 1e3;
            Ok(out.detections)
        };
        let (a, b) = if n % 2 == 0 {
            let a = time(&base, &mut base_ms)?;
            (a, time(&cand, &mut cand_ms)?)
        } else {
            let b = time(&cand, &mut cand_ms)?;
            (time(&base, &mut base_ms)?, b)
        };
        base_faces += a.len();

        let (lost_before, gained_before) = (lost, gained);
        let mut taken = vec![false; b.len()];
        let mut moved = false;
        for da in &a {
            let best = b
                .iter()
                .enumerate()
                .filter(|(i, _)| !taken[*i])
                .map(|(i, db)| (i, iou(da, db)))
                .max_by(|x, y| x.1.total_cmp(&y.1));
            match best {
                Some((i, overlap)) if overlap >= MATCH_IOU => {
                    taken[i] = true;
                    matched += 1;
                    let db = &b[i];
                    ious.push(f64::from(overlap));
                    score_delta.push(f64::from(db.score - da.score));
                    for (la, lb) in da.landmarks.iter().zip(db.landmarks.iter()) {
                        let d = f64::from((lb.x - la.x).hypot(lb.y - la.y));
                        moved |= d > 0.0;
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
        if moved || lost != lost_before || gained != gained_before {
            changed_images += 1;
        }
    }

    ious.sort_by(f64::total_cmp);
    landmark_px.sort_by(f64::total_cmp);
    score_delta.sort_by(|a, b| a.abs().total_cmp(&b.abs()));
    let over = |px: f64| landmark_px.iter().filter(|d| **d > px).count();

    println!("{images} images, {base_faces} faces from the f32 model\n");
    println!("faces matched      {matched}");
    println!("faces lost         {lost}   (f32 only)");
    println!("faces gained       {gained}   (int8 only)");
    println!("images changed     {changed_images}");
    println!(
        "landmark shift px  p50 {:.2}   p95 {:.2}   max {:.2}   >10 px {}   >35 px {}",
        quantile(&landmark_px, 0.5),
        quantile(&landmark_px, 0.95),
        landmark_px.last().copied().unwrap_or(f64::NAN),
        over(10.0),
        over(35.0)
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
        "\ndetect_image total  f32 {base_ms:.0} ms, int8 {cand_ms:.0} ms ({:.2}x)",
        base_ms / cand_ms
    );
    Ok(())
}
