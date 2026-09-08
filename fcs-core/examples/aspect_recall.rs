//! Does squashing a non-square image to 640x640 cost detections on real photos too?
//!
//! The webcam work found that a 16:9 frame stretched to a square 640x640 loses the face
//! entirely, while the same frame letterboxed into the same square keeps it. The
//! preprocessor scales x and y independently, so every non-square source is distorted in
//! proportion to how far its aspect is from 1:1 -- which is a claim about the whole corpus,
//! not about webcams.
//!
//! This detects each image twice: as production does it, and letterboxed into a square
//! canvas first so the preprocessor's resize becomes a no-op and the model sees an
//! undistorted face. Buckets by aspect ratio, because the prediction is that the loss grows
//! with distance from square.
//!
//!   cargo run --release -p fcs-core --example aspect_recall -- <dir> [limit]

use std::sync::Arc;

use anyhow::{Context, Result};
use fcs_core::{
    InputSize, PostprocessConfig, PreprocessConfig, Preprocessor, WgpuPreprocessor, YuNetDetector,
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

/// Fit into a square canvas with the aspect preserved, padding the remainder.
///
/// The detector then resizes 640x640 to 640x640, so this is the undistorted comparison.
/// Padding is black, which is what a letterboxing preprocessor would most simply do; a real
/// implementation would want to check edge replication too.
fn letterbox(image: &DynamicImage) -> DynamicImage {
    let scaled = image.resize(640, 640, FilterType::Triangle).to_rgb8();
    let mut canvas = RgbImage::from_pixel(640, 640, Rgb([0, 0, 0]));
    let x0 = (640 - scaled.width()) / 2;
    let y0 = (640 - scaled.height()) / 2;
    image::imageops::replace(&mut canvas, &scaled, x0 as i64, y0 as i64);
    DynamicImage::ImageRgb8(canvas)
}

/// How far from square, always >= 1.0 whichever side is longer.
fn anisotropy(w: u32, h: u32) -> f64 {
    let (w, h) = (w as f64, h as f64);
    if w > h { w / h } else { h / w }
}

fn main() -> Result<()> {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "fixtures/images".into());
    let limit: usize = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);

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
        PostprocessConfig::default(),
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

    // Bucketed by how far from square the source is, which is what the stretch scales with.
    let bounds = [1.05, 1.25, 1.45, 1.70, f64::INFINITY];
    let labels = [
        "near square (<1.05)",
        "1.05-1.25",
        "1.25-1.45 (4:3 is 1.33)",
        "1.45-1.70 (3:2 is 1.50)",
        "over 1.70 (16:9 is 1.78)",
    ];
    let mut images = [0usize; 5];
    let mut stretched = [0usize; 5];
    let mut boxed = [0usize; 5];
    let mut only_boxed = [0usize; 5];
    let mut only_stretched = [0usize; 5];

    for path in &files {
        let Ok(image) = image::open(path) else {
            continue;
        };
        let ratio = anisotropy(image.width(), image.height());
        let slot = bounds.iter().position(|b| ratio < *b).unwrap_or(4);

        let a = detector.detect_image(&image)?.detections.len();
        let b = detector.detect_image(&letterbox(&image))?.detections.len();
        images[slot] += 1;
        stretched[slot] += a;
        boxed[slot] += b;
        if a == 0 && b > 0 {
            only_boxed[slot] += 1;
        }
        if b == 0 && a > 0 {
            only_stretched[slot] += 1;
        }
    }

    println!(
        "{:<26} {:>7} {:>10} {:>10} {:>12} {:>12}",
        "aspect bucket", "images", "stretched", "boxed", "boxed only", "stretch only"
    );
    for slot in 0..5 {
        if images[slot] == 0 {
            continue;
        }
        println!(
            "{:<26} {:>7} {:>10} {:>10} {:>12} {:>12}",
            labels[slot],
            images[slot],
            stretched[slot],
            boxed[slot],
            only_boxed[slot],
            only_stretched[slot]
        );
    }
    println!(
        "\n`boxed only` counts images where production finds nothing and letterboxing finds a \
         face;\n`stretch only` is the reverse. Face counts are totals, not per image."
    );
    Ok(())
}
