//! Does squashing a non-square image to 640x640 cost detections, and what would fixing it
//! move?
//!
//! The webcam work found that a 16:9 frame stretched to a square 640x640 loses the face
//! entirely while the same frame letterboxed into the same square keeps it (experiment 95).
//! `preprocess.wgsl` scales x and y independently, so every non-square source is distorted in
//! proportion to how far its aspect is from 1:1 -- a claim about the whole corpus, not about
//! webcams.
//!
//! Each image is detected twice: as production does it, and letterboxed into a square canvas
//! first, which makes the preprocessor's resize a no-op so the model sees an undistorted
//! face. Letterboxed detections come back in canvas coordinates and are mapped back to source
//! pixels, so boxes and landmarks are comparable rather than merely countable.
//!
//! Reports recall by aspect bucket, and for images where both agree, how far the geometry
//! moved -- because a change that finds more faces but moves every box is not free.
//!
//!   cargo run --release -p fcs-core --example aspect_recall -- <dir> [limit] [pad]
//!
//! `pad` is the fill byte for the letterbox bars, default 0. 114 is what YOLO uses.

use std::sync::Arc;

use anyhow::{Context, Result};
use fcs_core::{
    BoundingBox, Detection, InputSize, PostprocessConfig, PreprocessConfig, Preprocessor,
    WgpuPreprocessor, YuNetDetector,
};
use fcs_utils::{
    config::ResizeQuality,
    gpu::{GpuAvailability, GpuContext, GpuContextOptions},
};
use image::{DynamicImage, Rgb, RgbImage, imageops::FilterType};

#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

const MODEL: &str = "models/face_detection_yunet_2023mar_640.onnx";
const INPUT: InputSize = InputSize::new(640, 640);
const SIDE: u32 = 640;

/// Where the source landed inside the square canvas: uniform scale, then centred.
struct Fit {
    scale: f64,
    x0: f64,
    y0: f64,
}

fn fit_of(w: u32, h: u32) -> Fit {
    let scale = (SIDE as f64 / w as f64).min(SIDE as f64 / h as f64);
    Fit {
        scale,
        x0: (SIDE as f64 - w as f64 * scale) / 2.0,
        y0: (SIDE as f64 - h as f64 * scale) / 2.0,
    }
}

fn letterbox(image: &DynamicImage, pad: u8) -> DynamicImage {
    let scaled = image.resize(SIDE, SIDE, FilterType::Triangle).to_rgb8();
    let mut canvas = RgbImage::from_pixel(SIDE, SIDE, Rgb([pad, pad, pad]));
    let x0 = (SIDE - scaled.width()) / 2;
    let y0 = (SIDE - scaled.height()) / 2;
    image::imageops::replace(&mut canvas, &scaled, x0 as i64, y0 as i64);
    DynamicImage::ImageRgb8(canvas)
}

/// Canvas coordinates back to source pixels, so the two runs can be compared directly.
fn unfit(d: &Detection, fit: &Fit) -> Detection {
    let map = |x: f32, y: f32| {
        (
            ((x as f64 - fit.x0) / fit.scale) as f32,
            ((y as f64 - fit.y0) / fit.scale) as f32,
        )
    };
    let (x, y) = map(d.bbox.x, d.bbox.y);
    let mut out = d.clone();
    out.bbox = BoundingBox {
        x,
        y,
        width: (d.bbox.width as f64 / fit.scale) as f32,
        height: (d.bbox.height as f64 / fit.scale) as f32,
    };
    for (slot, lm) in d.landmarks.iter().enumerate() {
        let (lx, ly) = map(lm.x, lm.y);
        out.landmarks[slot].x = lx;
        out.landmarks[slot].y = ly;
    }
    out
}

fn iou(a: &BoundingBox, b: &BoundingBox) -> f64 {
    let ax2 = a.x + a.width;
    let ay2 = a.y + a.height;
    let bx2 = b.x + b.width;
    let by2 = b.y + b.height;
    let ix = (ax2.min(bx2) - a.x.max(b.x)).max(0.0) as f64;
    let iy = (ay2.min(by2) - a.y.max(b.y)).max(0.0) as f64;
    let inter = ix * iy;
    let union = (a.width * a.height) as f64 + (b.width * b.height) as f64 - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

fn best(detections: &[Detection]) -> Option<&Detection> {
    detections.iter().max_by(|a, b| a.score.total_cmp(&b.score))
}

fn anisotropy(w: u32, h: u32) -> f64 {
    let (w, h) = (w as f64, h as f64);
    if w > h { w / h } else { h / w }
}

#[derive(Default)]
struct Bucket {
    images: usize,
    stretched_faces: usize,
    boxed_faces: usize,
    gained: usize,
    lost: usize,
    ious: Vec<f64>,
    landmark_px: Vec<f64>,
}

fn main() -> Result<()> {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "fixtures/images".into());
    let limit: usize = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);
    let pad: u8 = std::env::args()
        .nth(3)
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let context = match GpuContext::init_with_fallback(&GpuContextOptions::default()) {
        GpuAvailability::Available(ctx) => ctx,
        other => anyhow::bail!("GPU unavailable: {other:?}"),
    };
    let preprocessor: Arc<dyn Preprocessor> = Arc::new(WgpuPreprocessor::new(context.clone())?);
    let model_path = fcs_utils::model_path(MODEL)?.context("model not found")?;
    let detector = YuNetDetector::with_gpu_preprocessor(
        &model_path,
        PreprocessConfig {
            input_size: INPUT,
            resize_quality: ResizeQuality::Quality,
        },
        // The app's configured thresholds, not the library defaults: this experiment is about
        // how many crops production would make, and `config/gui_settings.json` uses a 0.8
        // score threshold where the default is lower.
        PostprocessConfig {
            score_threshold: 0.8,
            nms_threshold: 0.2,
            top_k: 5000,
        },
        preprocessor,
    )?;

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
    files.truncate(limit);

    let bounds = [1.05, 1.25, 1.45, 1.70, f64::INFINITY];
    let labels = [
        "near square (<1.05)",
        "1.05-1.25",
        "1.25-1.45 (4:3)",
        "1.45-1.70 (3:2)",
        "over 1.70 (16:9)",
    ];
    let mut buckets: Vec<Bucket> = (0..5).map(|_| Bucket::default()).collect();

    for path in &files {
        let Ok(image) = image::open(path) else {
            continue;
        };
        let (w, h) = (image.width(), image.height());
        let slot = bounds
            .iter()
            .position(|b| anisotropy(w, h) < *b)
            .unwrap_or(4);

        let a = detector.detect_image(&image)?.detections;
        let b = detector.detect_image(&letterbox(&image, pad))?.detections;
        let fit = fit_of(w, h);

        let bucket = &mut buckets[slot];
        bucket.images += 1;
        bucket.stretched_faces += a.len();
        bucket.boxed_faces += b.len();
        if a.is_empty() && !b.is_empty() {
            bucket.gained += 1;
        }
        if b.is_empty() && !a.is_empty() {
            bucket.lost += 1;
        }
        if let (Some(pa), Some(pb)) = (best(&a), best(&b)) {
            let pb = unfit(pb, &fit);
            bucket.ious.push(iou(&pa.bbox, &pb.bbox));
            let worst = pa
                .landmarks
                .iter()
                .zip(pb.landmarks.iter())
                .map(|(l, r)| (((l.x - r.x) as f64).powi(2) + ((l.y - r.y) as f64).powi(2)).sqrt())
                .fold(0.0f64, f64::max);
            bucket.landmark_px.push(worst);
        }
    }

    let median = |v: &mut Vec<f64>| {
        if v.is_empty() {
            return f64::NAN;
        }
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };

    println!("pad byte {pad}, {} images from {dir}\n", files.len());
    println!(
        "{:<20} {:>6} {:>9} {:>7} {:>7} {:>6} {:>9} {:>12}",
        "aspect", "images", "stretched", "boxed", "gained", "lost", "med IoU", "med lm px"
    );
    for (slot, label) in labels.iter().enumerate() {
        let b = &mut buckets[slot];
        if b.images == 0 {
            continue;
        }
        println!(
            "{label:<20} {:>6} {:>9} {:>7} {:>7} {:>6} {:>9.4} {:>12.2}",
            b.images,
            b.stretched_faces,
            b.boxed_faces,
            b.gained,
            b.lost,
            median(&mut b.ious),
            median(&mut b.landmark_px),
        );
    }
    println!(
        "\n`gained` is images where production finds nothing and letterboxing finds a face; \
         `lost` the reverse.\nIoU and landmark distance compare the highest-scoring detection \
         where both found one, in source pixels."
    );
    Ok(())
}
