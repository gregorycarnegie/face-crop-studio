//! Detections per aspect-ratio bucket, over a real corpus.
//!
//! This began as the probe that decided experiment 96: it detected every image twice, once
//! squashed to the model input and once letterboxed, and reported what changed. Letterboxing
//! won and shipped, so there is no second variant left to compare against -- what is useful
//! now is the other half, a recall figure per bucket that can be held against the numbers the
//! experiment recorded. A change to `preprocess.wgsl`, `rgb_to_chw.wgsl` or `fit_input` that
//! quietly costs detections on non-square sources shows up here and nowhere else, because
//! every parity test in the suite compares the paths to *each other* rather than to a corpus.
//!
//! Expected on the 1239-image reference corpus (experiment 96): 11/22/510/301/395 images per
//! bucket, and 10/20/354/320/425 faces -- 1129 in total.
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

#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

const MODEL: &str = "models/face_detection_yunet_2023mar_640.onnx";
const INPUT: InputSize = InputSize::new(640, 640);

fn anisotropy(w: u32, h: u32) -> f64 {
    let (w, h) = (f64::from(w), f64::from(h));
    if w > h { w / h } else { h / w }
}

#[derive(Default)]
struct Bucket {
    images: usize,
    faces: usize,
    /// Images where nothing was found at all, which is the failure this probe exists to see.
    empty: usize,
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
        // The app's configured thresholds, not the library defaults: this is about how many
        // crops production would make, and `config/gui_settings.json` uses a 0.8 score
        // threshold where the default is higher.
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

        let found = detector.detect_image(&image)?.detections.len();
        let bucket = &mut buckets[slot];
        bucket.images += 1;
        bucket.faces += found;
        if found == 0 {
            bucket.empty += 1;
        }
    }

    println!("{} images from {dir}\n", files.len());
    println!(
        "{:<20} {:>6} {:>7} {:>8}",
        "aspect", "images", "faces", "no face"
    );
    let (mut images, mut faces) = (0usize, 0usize);
    for (slot, label) in labels.iter().enumerate() {
        let b = &buckets[slot];
        if b.images == 0 {
            continue;
        }
        images += b.images;
        faces += b.faces;
        println!("{label:<20} {:>6} {:>7} {:>8}", b.images, b.faces, b.empty);
    }
    println!("{:<20} {images:>6} {faces:>7}", "total");
    Ok(())
}
