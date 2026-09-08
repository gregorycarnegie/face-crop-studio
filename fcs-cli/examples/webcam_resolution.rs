//! Does capture resolution change what the detector finds, or was that the scene moving?
//!
//! Running the webcam loop at different resolutions gave 90/90 faces at 640x480 and 0/90 at
//! 1280x720, but face counts also moved between identical runs, so the subject moving is a
//! live alternative explanation. This removes it: capture **one** frame and detect on that
//! same frame at several sizes. Same instant, same scene, same lighting -- the only variable
//! left is what the detector was handed.
//!
//! Saves the frames it disagrees on, so the answer can be looked at rather than argued about.
//!
//!   cargo run --release -p fcs-cli --example webcam_resolution -- [frames] [outdir]

use std::sync::Arc;

use anyhow::{Context, Result};
use fcs_core::{
    InputSize, PostprocessConfig, PreprocessConfig, Preprocessor, WgpuPreprocessor, YuNetDetector,
};
use fcs_utils::{
    WebcamCapture,
    config::ResizeQuality,
    gpu::{GpuAvailability, GpuContext, GpuContextOptions},
};
use image::imageops::FilterType;

const MODEL: &str = "models/face_detection_yunet_2023mar_640.onnx";
const INPUT: InputSize = InputSize::new(640, 640);

/// Sizes to re-detect each captured frame at. The 4:3 entries share the capture's aspect
/// only if the capture is 4:3, which is the point: if 16:9 is what hurts, the 4:3 rows win
/// regardless of pixel count, and if size is what hurts, the small rows win regardless of
/// shape.
const SIZES: &[(u32, u32, &str)] = &[
    (1920, 1080, "16:9 2.07 MP (as captured)"),
    (1280, 720, "16:9 0.92 MP"),
    (960, 540, "16:9 0.52 MP"),
    (640, 360, "16:9 0.23 MP"),
    (1440, 1080, "4:3  1.56 MP (centre crop)"),
    (800, 600, "4:3  0.48 MP (centre crop)"),
    (640, 480, "4:3  0.31 MP (centre crop)"),
];

fn main() -> Result<()> {
    let frames: usize = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(12);
    let outdir = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "webcam_frames".into());

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
        // The app's configured threshold, not the library default of 0.9: production reads
        // 0.8 from config/gui_settings.json, and the gap between them changes what this
        // measures.
        PostprocessConfig {
            score_threshold: 0.8,
            nms_threshold: 0.2,
            top_k: 5000,
        },
        preprocessor,
    )?;

    // Capture at the camera's widest mode, then derive every other size from the same frame.
    let mut webcam = WebcamCapture::new(1920, 1080, 30).context("open webcam")?;
    let (cw, ch) = webcam.resolution();
    println!("captured at {cw}x{ch}\n");
    for _ in 0..10 {
        webcam.capture_frame()?;
    }

    let mut counts = vec![0usize; SIZES.len()];
    let mut letterboxed = 0usize;
    let mut saved = 0usize;
    std::fs::create_dir_all(&outdir).ok();

    for frame_index in 0..frames {
        let frame = webcam.capture_frame()?;
        let mut per_frame = Vec::with_capacity(SIZES.len());
        for (slot, (w, h, _)) in SIZES.iter().enumerate() {
            // A 4:3 target from a 16:9 source is a centre crop, not a squash: cropping is
            // what a camera does when it offers a 4:3 mode, so this compares like with like.
            let candidate = if (*w as f64 / *h as f64 - cw as f64 / ch as f64).abs() > 0.01 {
                let crop_w = (ch as f64 * (*w as f64 / *h as f64)).round() as u32;
                let crop_w = crop_w.min(cw);
                let x0 = (cw - crop_w) / 2;
                frame
                    .crop_imm(x0, 0, crop_w, ch)
                    .resize_exact(*w, *h, FilterType::Triangle)
            } else {
                frame.resize_exact(*w, *h, FilterType::Triangle)
            };
            let found = detector.detect_image(&candidate)?.detections.len();
            counts[slot] += found;
            per_frame.push(found);
        }
        // Letterbox: the same 16:9 content fitted into a 640x640 canvas with its aspect
        // preserved and the remainder padded. The detector then "resizes" 640x640 to
        // 640x640, so this is the only row where the model sees an undistorted face. It is
        // also the candidate fix, which is why it is measured rather than argued.
        let mut canvas = image::RgbImage::from_pixel(640, 640, image::Rgb([0, 0, 0]));
        let scaled = frame.resize(640, 640, FilterType::Triangle).to_rgb8();
        let y0 = (640 - scaled.height()) / 2;
        image::imageops::replace(&mut canvas, &scaled, 0, y0 as i64);
        letterboxed += detector
            .detect_image(&image::DynamicImage::ImageRgb8(canvas))?
            .detections
            .len();

        // Save the frames where the sizes disagree; those are the ones worth looking at.
        let agree = per_frame.iter().all(|n| *n == per_frame[0]);
        if !agree && saved < 4 {
            let path = format!("{outdir}/disagree_{frame_index:02}.png");
            frame.save(&path).ok();
            println!("saved {path}  per-size counts {per_frame:?}");
            saved += 1;
        }
    }

    println!(
        "\n{:<34} {:>12} {:>10}",
        "detected on", "faces", "per frame"
    );
    for (slot, (_, _, label)) in SIZES.iter().enumerate() {
        println!(
            "{label:<34} {:>12} {:>10.2}",
            counts[slot],
            counts[slot] as f64 / frames as f64
        );
    }
    println!(
        "{:<34} {:>12} {:>10.2}",
        "16:9 letterboxed into 640x640",
        letterboxed,
        letterboxed as f64 / frames as f64
    );
    println!("\n{frames} frames, all sizes derived from the same capture.");
    Ok(())
}
