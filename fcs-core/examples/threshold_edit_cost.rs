//! What a detection-threshold edit costs the GUI, before and after experiment 66.
//!
//! Before, NMS and top-k edits went through `rebuild_detector`: a new `YuNetDetector` on the
//! already-open device (model load plus every compute pipeline) on the UI thread, then
//! `load_image_path`, which decodes the file again and detects. After, the detector is swapped
//! for one sharing the compiled model and the in-memory image is detected again. This times the
//! three pieces on the device the GUI would already have.
//!
//!   cargo run --release -p fcs-core --example threshold_edit_cost -- [image] [reps]

use std::sync::Arc;
use std::time::Instant;

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

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args
        .first()
        .cloned()
        .unwrap_or_else(|| "fixtures/images/006.jpg".into());
    let reps: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(10);

    let context = match GpuContext::init_with_fallback(&GpuContextOptions::default()) {
        GpuAvailability::Available(ctx) => ctx,
        other => anyhow::bail!("GPU unavailable: {other:?}"),
    };
    println!("adapter: {}", context.adapter_info().name);
    let model = fcs_utils::model_path(MODEL)
        .context("resolve model")?
        .context("model missing")?;
    let pre = PreprocessConfig {
        input_size: InputSize::new(640, 640),
        resize_quality: ResizeQuality::Quality,
    };
    let build = || -> Result<YuNetDetector> {
        let preprocessor: Arc<dyn Preprocessor> = Arc::new(WgpuPreprocessor::new(context.clone())?);
        YuNetDetector::with_gpu_preprocessor(
            &model,
            pre.clone(),
            PostprocessConfig::default(),
            preprocessor,
        )
    };

    // The first pipeline in a process carries one-off FXC and driver warm-up (82); a GUI
    // rebuild happens long after that has been paid, so it is excluded.
    let detector = build().context("warm-up build")?;
    let image = fcs_utils::load_image(std::path::Path::new(&path)).context("decode")?;
    detector.detect_image(&image)?;

    let (mut rebuild, mut swap, mut decode, mut detect) = (vec![], vec![], vec![], vec![]);
    for i in 0..reps {
        let started = Instant::now();
        let rebuilt = build()?;
        rebuild.push(started.elapsed().as_secs_f64() * 1e3);
        drop(rebuilt);

        let post = PostprocessConfig {
            score_threshold: if i % 2 == 0 { 0.7 } else { 0.8 },
            ..PostprocessConfig::default()
        };
        let started = Instant::now();
        let swapped = detector.with_postprocess(post);
        swap.push(started.elapsed().as_secs_f64() * 1e3);

        let started = Instant::now();
        let decoded = fcs_utils::load_image(std::path::Path::new(&path))?;
        decode.push(started.elapsed().as_secs_f64() * 1e3);

        let started = Instant::now();
        swapped.detect_image(&decoded)?;
        detect.push(started.elapsed().as_secs_f64() * 1e3);
    }

    println!(
        "{} {}x{}, {reps} reps, medians",
        path,
        image.width(),
        image.height()
    );
    let (rebuild, swap, decode, detect) = (
        median(rebuild),
        median(swap),
        median(decode),
        median(detect),
    );
    println!("  rebuild detector on the open device  {rebuild:>9.3} ms");
    println!("  swap postprocessing only             {swap:>9.3} ms");
    println!("  decode the file again                {decode:>9.3} ms");
    println!("  detect                               {detect:>9.3} ms");
    println!(
        "  threshold edit, before: {:.1} ms (rebuild on the UI thread) + {:.1} ms of work",
        rebuild,
        decode + detect
    );
    println!(
        "  threshold edit, after:  {:.3} ms on the UI thread + {:.1} ms of work",
        swap, detect
    );
    Ok(())
}
