//! What a webcam frame costs, stage by stage.
//!
//! Experiment 6 found that under 1 MP detection is 64% of per-image cost, and said the
//! decision about the GPU-overhead experiments "needs the webcam measurement (53, 65)".
//! Nothing had measured it: every number in this backlog comes from files on disk.
//!
//! A webcam frame is not a small JPEG from disk. It arrives as MJPEG over USB, is decoded by
//! libjpeg-turbo inside nokhwa, copied into an `image` buffer, and only then detected. This
//! times each of those against detection, and reports the frame budget the loop actually has.
//!
//!   cargo run --release -p fcs-cli --example webcam_cost -- [frames] [width] [height] [fps]

use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use anyhow::{Context, Result};
use fcs_core::{
    InputSize, PostprocessConfig, PreprocessConfig, Preprocessor, WgpuPreprocessor, YuNetDetector,
};
use fcs_utils::{
    WebcamCapture,
    config::ResizeQuality,
    gpu::{GpuAvailability, GpuContext, GpuContextOptions},
    list_webcam_devices,
};

const MODEL: &str = "models/face_detection_yunet_2023mar_640.onnx";
const INPUT: InputSize = InputSize::new(640, 640);

/// Phases in containment order; the number is the indent depth.
const DISPLAY: &[(&str, usize)] = &[
    ("fcs_utils::webcam_grab", 0),
    ("fcs_utils::webcam_decode", 0),
    ("fcs_utils::webcam_wrap", 0),
    ("fcs_core::detect_image", 0),
    ("fcs_core::detect_on_device", 1),
    ("fcs_core::gpu_preprocess", 2),
    ("fcs_core::preprocess_to_rgba", 3),
    ("fcs_core::preprocess_upload", 3),
    ("fcs_core::preprocess_encode", 3),
    ("fcs_core::onnx_inference", 2),
    ("fcs_core::gpu_encode", 3),
    ("fcs_core::gpu_readback", 3),
    ("fcs_core::readback_wait", 4),
    ("fcs_core::gpu_decode", 3),
    ("fcs_core::preprocess_dynamic_image", 1),
];

struct Collector;

fn captured() -> &'static Mutex<Vec<(String, f64)>> {
    static S: OnceLock<Mutex<Vec<(String, f64)>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(Vec::new()))
}

impl log::Log for Collector {
    fn enabled(&self, m: &log::Metadata) -> bool {
        m.target() == "fcs::telemetry"
    }
    fn log(&self, record: &log::Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let text = record.args().to_string();
        if let Some((label, rest)) = text.split_once(" completed in ")
            && let Some(ms) = parse_ms(rest)
        {
            captured().lock().unwrap().push((label.to_string(), ms));
        }
    }
    fn flush(&self) {}
}

fn parse_ms(text: &str) -> Option<f64> {
    let text = text.trim();
    let (value, scale) = if let Some(v) = text.strip_suffix("ms") {
        (v, 1.0)
    } else if let Some(v) = text.strip_suffix("\u{b5}s") {
        (v, 1e-3)
    } else if let Some(v) = text.strip_suffix("ns") {
        (v, 1e-6)
    } else {
        (text.strip_suffix('s')?, 1e3)
    };
    value.trim().parse::<f64>().ok().map(|v| v * scale)
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

fn arg<T: std::str::FromStr>(index: usize, default: T) -> T {
    std::env::args()
        .nth(index)
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() -> Result<()> {
    let frames: usize = arg(1, 120);
    let width: u32 = arg(2, 640);
    let height: u32 = arg(3, 480);
    let fps: u32 = arg(4, 30);

    log::set_boxed_logger(Box::new(Collector))?;
    log::set_max_level(log::LevelFilter::Trace);
    fcs_utils::telemetry::configure(true, log::LevelFilter::Trace);

    match list_webcam_devices() {
        Ok(devices) if !devices.is_empty() => {
            for (idx, name) in &devices {
                println!("device [{idx}] {name}");
            }
        }
        Ok(_) => anyhow::bail!("no webcam devices found"),
        Err(e) => anyhow::bail!("could not enumerate webcam devices: {e}"),
    }

    let context = match GpuContext::init_with_fallback(&GpuContextOptions::default()) {
        GpuAvailability::Available(ctx) => ctx,
        other => anyhow::bail!("GPU unavailable: {other:?}"),
    };
    let preprocessor: std::sync::Arc<dyn Preprocessor> =
        std::sync::Arc::new(WgpuPreprocessor::new(context.clone())?);
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

    let mut webcam = WebcamCapture::new(width, height, fps).context("open webcam")?;
    let (aw, ah) = webcam.resolution();
    println!(
        "capture {aw}x{ah} @ {} fps requested {fps}; {} MP per frame\n",
        webcam.frame_rate(),
        (aw as f64 * ah as f64) / 1e6
    );

    // The first frames include stream start-up and the pipeline warming, which is not what a
    // steady loop pays.
    for _ in 0..15 {
        let frame = webcam.capture_frame()?;
        detector.detect_image(&frame)?;
    }
    captured().lock().unwrap().clear();

    let mut wall = Vec::with_capacity(frames);
    let mut faces_seen = 0usize;
    let loop_started = Instant::now();
    for _ in 0..frames {
        let started = Instant::now();
        let frame = webcam.capture_frame()?;
        let out = detector.detect_image(&frame)?;
        wall.push(started.elapsed().as_secs_f64() * 1e3);
        faces_seen += out.detections.len();
    }
    let loop_secs = loop_started.elapsed().as_secs_f64();

    wall.sort_by(f64::total_cmp);
    println!(
        "{frames} frames in {loop_secs:.2} s = {:.1} fps; per-frame p50 {:.2} ms p95 {:.2} ms; \
         {faces_seen} faces total\n",
        frames as f64 / loop_secs,
        quantile(&wall, 0.5),
        quantile(&wall, 0.95)
    );

    let collected = captured().lock().unwrap().clone();
    println!("{:<40} {:>6} {:>9} {:>9}", "phase", "n", "p50 ms", "p95 ms");
    for (label, depth) in DISPLAY {
        let mut values: Vec<f64> = collected
            .iter()
            .filter(|(l, _)| l == label)
            .map(|(_, v)| *v)
            .collect();
        if values.is_empty() {
            continue;
        }
        values.sort_by(f64::total_cmp);
        let short = label
            .trim_start_matches("fcs_core::")
            .trim_start_matches("fcs_utils::");
        println!(
            "{:<40} {:>6} {:>9.3} {:>9.3}",
            format!("{}{}", "  ".repeat(*depth), short),
            values.len(),
            quantile(&values, 0.5),
            quantile(&values, 0.95)
        );
    }
    Ok(())
}
