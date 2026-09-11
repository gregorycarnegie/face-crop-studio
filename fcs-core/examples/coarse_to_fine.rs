//! Coarse-to-fine and cascaded detection against production's single 640 pass (experiments 75
//! and 79).
//!
//! Each image is detected three ways, in an order rotated per image so no arm always meets a
//! warm cache:
//!
//! - `640`: production -- one pass at 640, score 0.8. Everything else is matched against it.
//! - `320`: one pass at 320, score 0.8.
//! - `320 low`: one pass at 320, score 0.3 -- the candidates for both strategies below. A higher
//!   threshold's output is a subset of it, because NMS keeps the higher score and so a box under
//!   the threshold never suppresses one over it. `320` is run directly anyway, and every image
//!   where the subset and the direct run disagree is counted.
//!
//! Then every `320 low` candidate is re-detected at 320 on a square crop around it. The
//! strategies are assembled per image from those measured calls:
//!
//! - cascade (79): `320 low`. A candidate scoring in [t, 0.8) makes the image unsure, and an
//!   unsure image also runs 640 and takes its result; a sure one keeps 320's detections.
//!   Cost: `320 low`, plus `640` when escalated.
//! - screen (79): `320 low` as a no-face screen. Any candidate >= t runs 640 and takes its
//!   result, so every face kept is production's own; only an image with no candidate skips 640.
//!   Cost: `320 low`, plus `640` when escalated.
//! - roi (75): `320 low`, then every candidate scoring >= t refined on its crop; a refinement
//!   that finds nothing overlapping its candidate drops it. Cost: `320 low` plus the crops.
//!
//! Lost faces are bucketed by production's box in model-input pixels -- its longest edge scaled
//! by 640 over the source's longest side -- because that is what a coarse pass cannot see.
//!
//! A huge ROI scale makes every crop the whole image, so roi must then reproduce `320`'s
//! landmarks exactly: that is the check on the crop's coordinate mapping.
//!
//!   cargo run --release -p fcs-core --example coarse_to_fine -- <dir> [limit] [roi scale]
//!   WGPU_POWER_PREF=low cargo run ...   (the integrated GPU)

use std::{sync::Arc, time::Instant};

use anyhow::{Context, Result};
use fcs_core::{
    Detection, InputSize, PostprocessConfig, PreprocessConfig, Preprocessor, WgpuPreprocessor,
    YuNetDetector,
};
use fcs_utils::{
    config::ResizeQuality,
    gpu::{GpuAvailability, GpuContext, GpuContextOptions},
};
use image::DynamicImage;

#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Production's score threshold (`config/gui_settings.json`).
const SCORE: f32 = 0.8;
/// The candidate threshold `320 low` runs at; the higher ones are subsets of its output.
const LOW: f32 = 0.3;
const CANDIDATE_THRESHOLDS: [f32; 3] = [0.3, 0.5, 0.7];
/// Below this IoU two boxes are different faces, not the same face moved (as in `int8_quality`).
const MATCH_IOU: f32 = 0.3;
/// A refinement crop is `roi scale` candidate longest-edges wide (default 2), but never narrower
/// than the 320 input, so a small face is not upscaled.
const ROI_MIN_SIDE: f32 = 320.0;
/// A face whose box comes within this fraction of the source's width or height of an edge.
const EDGE_MARGIN: f32 = 0.05;
/// Production faces in one image for it to count as crowded.
const CROWDED: usize = 4;

struct Timed {
    ms: f64,
    detections: Vec<Detection>,
}

fn timed(detector: &YuNetDetector, image: &DynamicImage) -> Result<Timed> {
    let started = Instant::now();
    let detections = detector.detect_image(image)?.detections;
    Ok(Timed {
        ms: started.elapsed().as_secs_f64() * 1e3,
        detections,
    })
}

/// A `320 low` candidate re-detected on the crop around it.
struct Refined {
    candidate_score: f32,
    ms: f64,
    detection: Option<Detection>,
}

fn refine(
    detector: &YuNetDetector,
    image: &DynamicImage,
    candidate: &Detection,
    roi_scale: f32,
) -> Result<Refined> {
    let (w, h) = (image.width() as f32, image.height() as f32);
    let side = (candidate.bbox.longest_edge() * roi_scale)
        .max(ROI_MIN_SIDE)
        .min(w.max(h));
    let centre = candidate.bbox.center();
    let x0 = (centre.x - side / 2.0)
        .clamp(0.0, (w - side).max(0.0))
        .floor();
    let y0 = (centre.y - side / 2.0)
        .clamp(0.0, (h - side).max(0.0))
        .floor();

    let started = Instant::now();
    let crop = image.crop_imm(
        x0 as u32,
        y0 as u32,
        side.min(w - x0) as u32,
        side.min(h - y0) as u32,
    );
    let found = detector.detect_image(&crop)?.detections;
    let ms = started.elapsed().as_secs_f64() * 1e3;

    let detection = found
        .into_iter()
        .map(|mut d| {
            d.bbox.x += x0;
            d.bbox.y += y0;
            for landmark in &mut d.landmarks {
                landmark.x += x0;
                landmark.y += y0;
            }
            (candidate.bbox.iou(&d.bbox), d)
        })
        .filter(|(overlap, _)| *overlap >= MATCH_IOU)
        .max_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, d)| d);
    Ok(Refined {
        candidate_score: candidate.score,
        ms,
        detection,
    })
}

#[derive(Default)]
struct Stats {
    name: String,
    ms: f64,
    /// By production box size in model-input pixels: <16, 16-32, 32-64, 64+.
    lost: [usize; 4],
    lost_edge: usize,
    lost_crowded: usize,
    gained: usize,
    landmark_px: Vec<f64>,
    escalated: usize,
}

impl Stats {
    fn named(name: String) -> Self {
        Self {
            name,
            ..Default::default()
        }
    }

    fn add(&mut self, reference: &[Detection], candidate: &[Detection], size: (f32, f32), ms: f64) {
        self.ms += ms;
        let mut taken = vec![false; candidate.len()];
        for r in reference {
            let best = candidate
                .iter()
                .enumerate()
                .filter(|(i, _)| !taken[*i])
                .map(|(i, c)| (i, r.bbox.iou(&c.bbox)))
                .max_by(|a, b| a.1.total_cmp(&b.1));
            match best {
                Some((i, overlap)) if overlap >= MATCH_IOU => {
                    taken[i] = true;
                    for (a, b) in r.landmarks.iter().zip(&candidate[i].landmarks) {
                        self.landmark_px
                            .push(f64::from((b.x - a.x).hypot(b.y - a.y)));
                    }
                }
                _ => {
                    self.lost[size_bucket(r, size)] += 1;
                    self.lost_edge += usize::from(at_edge(r, size));
                    self.lost_crowded += usize::from(reference.len() >= CROWDED);
                }
            }
        }
        self.gained += taken.iter().filter(|t| !**t).count();
    }
}

/// The box in model-input pixels -- longest edge scaled by 640 over the source's longest side --
/// bucketed <16, 16-32, 32-64, 64+.
fn size_bucket(d: &Detection, (w, h): (f32, f32)) -> usize {
    let px = d.bbox.longest_edge() * 640.0 / w.max(h);
    [16.0, 32.0, 64.0]
        .iter()
        .position(|edge| px < *edge)
        .unwrap_or(3)
}

fn at_edge(d: &Detection, (w, h): (f32, f32)) -> bool {
    let b = &d.bbox;
    b.x < EDGE_MARGIN * w
        || b.y < EDGE_MARGIN * h
        || b.right() > (1.0 - EDGE_MARGIN) * w
        || b.bottom() > (1.0 - EDGE_MARGIN) * h
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    sorted[(((sorted.len() - 1) as f64) * q).round() as usize]
}

fn detector(
    preprocessor: &Arc<dyn Preprocessor>,
    model: &str,
    side: u32,
    score_threshold: f32,
) -> Result<YuNetDetector> {
    let model = fcs_utils::model_path(model)?.with_context(|| format!("{model} not found"))?;
    YuNetDetector::with_gpu_preprocessor(
        &model,
        PreprocessConfig {
            input_size: InputSize::new(side, side),
            resize_quality: ResizeQuality::Quality,
        },
        PostprocessConfig {
            score_threshold,
            nms_threshold: 0.2,
            top_k: 5000,
        },
        Arc::clone(preprocessor),
    )
}

fn main() -> Result<()> {
    let dir = std::env::args()
        .nth(1)
        .context("usage: coarse_to_fine <dir> [limit] [roi scale]")?;
    let limit: usize = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);
    let roi_scale: f32 = std::env::args()
        .nth(3)
        .and_then(|v| v.parse().ok())
        .unwrap_or(2.0);

    let context = match GpuContext::init_with_fallback(&GpuContextOptions::default()) {
        GpuAvailability::Available(ctx) => ctx,
        other => anyhow::bail!("GPU unavailable: {other:?}"),
    };
    let preprocessor: Arc<dyn Preprocessor> = Arc::new(WgpuPreprocessor::new(context.clone())?);
    let d640 = detector(
        &preprocessor,
        "models/face_detection_yunet_2023mar_640.onnx",
        640,
        SCORE,
    )?;
    let d320 = detector(
        &preprocessor,
        "models/face_detection_yunet_2023mar_320.onnx",
        320,
        SCORE,
    )?;
    let d320_low = d320.with_postprocess(PostprocessConfig {
        score_threshold: LOW,
        ..d320.postprocess_config().clone()
    });
    let arms = [&d640, &d320, &d320_low];

    let mut paths: Vec<_> = std::fs::read_dir(&dir)
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

    let mut stats = vec![
        Stats::named("640 (production)".into()),
        Stats::named("320".into()),
    ];
    for t in CANDIDATE_THRESHOLDS {
        stats.push(Stats::named(format!("cascade t={t}")));
        stats.push(Stats::named(format!("screen t={t}")));
        stats.push(Stats::named(format!("roi t={t}")));
    }
    let (mut images, mut faces, mut subset_mismatch) = (0usize, 0usize, 0usize);
    let mut empty_images = 0usize;
    // Denominators for the loss columns.
    let mut sizes = [0usize; 4];
    let (mut edge_faces, mut crowded_faces, mut crowded_images) = (0usize, 0usize, 0usize);

    for (n, path) in paths.iter().enumerate() {
        let Ok(image) = image::open(path) else {
            eprintln!("skipping {}", path.display());
            continue;
        };
        if n == 0 {
            for arm in arms {
                arm.detect_image(&image).context("warm-up")?;
            }
        }
        let size = (image.width() as f32, image.height() as f32);

        let mut runs: [Option<Timed>; 3] = [None, None, None];
        for k in 0..3 {
            let arm = (n + k) % 3;
            runs[arm] = Some(timed(arms[arm], &image)?);
        }
        let [production, coarse, low] = runs.map(|r| r.expect("every arm ran"));
        let refined = low
            .detections
            .iter()
            .map(|c| refine(&d320, &image, c, roi_scale))
            .collect::<Result<Vec<_>>>()?;

        images += 1;
        faces += production.detections.len();
        empty_images += usize::from(production.detections.is_empty());
        for d in &production.detections {
            sizes[size_bucket(d, size)] += 1;
            edge_faces += usize::from(at_edge(d, size));
        }
        if production.detections.len() >= CROWDED {
            crowded_images += 1;
            crowded_faces += production.detections.len();
        }
        let confident: Vec<Detection> = low
            .detections
            .iter()
            .filter(|d| d.score >= SCORE)
            .cloned()
            .collect();
        if confident.len() != coarse.detections.len() {
            subset_mismatch += 1;
        }

        let reference = &production.detections;
        stats[0].add(reference, reference, size, production.ms);
        stats[1].add(reference, &coarse.detections, size, coarse.ms);
        for (j, t) in CANDIDATE_THRESHOLDS.into_iter().enumerate() {
            let unsure = low
                .detections
                .iter()
                .any(|d| d.score >= t && d.score < SCORE);
            let any_candidate = low.detections.iter().any(|d| d.score >= t);
            // Not escalating under either gate means `confident` is the answer; under the screen
            // it is always empty there, since t <= 0.8.
            for (gate, escalate) in [unsure, any_candidate].into_iter().enumerate() {
                let s = &mut stats[2 + j * 3 + gate];
                if escalate {
                    s.escalated += 1;
                    s.add(reference, reference, size, low.ms + production.ms);
                } else {
                    s.add(reference, &confident, size, low.ms);
                }
            }

            let mut kept: Vec<Detection> = Vec::new();
            let mut ms = low.ms;
            for r in refined.iter().filter(|r| r.candidate_score >= t) {
                ms += r.ms;
                if let Some(d) = &r.detection
                    && kept.iter().all(|k| k.bbox.iou(&d.bbox) < MATCH_IOU)
                {
                    kept.push(d.clone());
                }
            }
            stats[2 + j * 3 + 2].add(reference, &kept, size, ms);
        }
    }

    println!(
        "{} ({:?}), {images} images from {dir}, {faces} production faces, {empty_images} images \
         with none, roi scale {roi_scale}",
        context.adapter_info().name,
        context.adapter_info().backend
    );
    println!(
        "production faces by size <16/16/32/64+: {}/{}/{}/{}; {edge_faces} at an edge; \
         {crowded_faces} in {crowded_images} images with {CROWDED}+",
        sizes[0], sizes[1], sizes[2], sizes[3]
    );
    println!(
        "320 low's >=0.8 subset differed from the direct 320 run on {subset_mismatch} images\n"
    );
    println!(
        "{:<24} {:>8} {:>6}  {:>21} {:>5} {:>7} {:>6} {:>7} {:>10} {:>9}",
        "strategy",
        "detect s",
        "speed",
        "lost <16/16/32/64+",
        "edge",
        "crowded",
        "gained",
        "lm p95",
        ">35px",
        "escalated"
    );
    let base_ms = stats[0].ms;
    for s in &mut stats {
        s.landmark_px.sort_by(f64::total_cmp);
        let over = s.landmark_px.iter().filter(|d| **d > 35.0).count();
        println!(
            "{:<24} {:>8.2} {:>5.2}x  {:>21} {:>5} {:>7} {:>6} {:>7.2} {:>10} {:>9}",
            s.name,
            s.ms / 1e3,
            base_ms / s.ms,
            format!("{}/{}/{}/{}", s.lost[0], s.lost[1], s.lost[2], s.lost[3]),
            s.lost_edge,
            s.lost_crowded,
            s.gained,
            quantile(&s.landmark_px, 0.95),
            format!("{over}/{}", s.landmark_px.len()),
            s.escalated
        );
    }
    Ok(())
}
