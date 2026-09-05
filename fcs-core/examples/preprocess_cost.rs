//! Where preprocessing time goes on each path, for one real image.
//!
//! `detect_on_device` fuses preprocessing and inference onto one device to avoid a
//! 4.9 MB download and re-upload between them. That trade is only worth it if getting
//! the source image onto the GPU is cheap — and the source goes up at full resolution,
//! while the CPU path resizes first and uploads the 640x640 tensor. This prints both
//! sides so the trade can be judged per image size rather than assumed.
//!
//! Run with: cargo run --release --example preprocess_cost

use std::time::Instant;

use anyhow::{Context, Result};
use fcs_core::{
    CpuPreprocessor, InputSize, PreprocessConfig, Preprocessor, WgpuPreprocessor, gpu::GpuYuNet,
};
use fcs_utils::{
    config::ResizeQuality,
    gpu::{GpuAvailability, GpuContext, GpuContextOptions},
    load_fixture_image,
};

#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

const MODEL: &str = "models/face_detection_yunet_2023mar_640.onnx";
const FIXTURE: &str = "images/006.jpg";
const INPUT: InputSize = InputSize::new(640, 640);
const RUNS: usize = 30;

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn time<T>(runs: usize, mut f: impl FnMut() -> T) -> f64 {
    for _ in 0..3 {
        std::hint::black_box(f());
    }
    median(
        (0..runs)
            .map(|_| {
                let t = Instant::now();
                std::hint::black_box(f());
                t.elapsed().as_secs_f64() * 1e3
            })
            .collect(),
    )
}

fn main() -> Result<()> {
    let options = GpuContextOptions {
        profiling: true,
        ..GpuContextOptions::default()
    };
    let context = match GpuContext::init_with_fallback(&options) {
        GpuAvailability::Available(ctx) => ctx,
        GpuAvailability::Disabled { reason } => anyhow::bail!("GPU disabled: {reason}"),
        GpuAvailability::Unavailable { error } => anyhow::bail!("GPU unavailable: {error}"),
    };
    let image = load_fixture_image(FIXTURE).context("load fixture")?;
    let (w, h) = (image.width(), image.height());
    println!(
        "adapter: {}\nimage:   {w}x{h} ({:.1} MP)\n",
        context.adapter_info().name,
        f64::from(w) * f64::from(h) / 1e6
    );

    let model_path = fcs_utils::model_path(MODEL)
        .context("resolve model")?
        .context("model missing")?;
    let model = GpuYuNet::with_context(context.clone(), &model_path, INPUT)?;
    let tensor = model.allocate_input(INPUT)?;

    // The RGBA conversion the GPU path needs before it can touch the texture. It is a
    // full-resolution allocation and copy, and it happens per image.
    let rgba_ms = time(RUNS, || image.to_rgba8());
    println!(
        "to_rgba8 (full res)           {rgba_ms:7.2} ms   allocates {:.1} MB",
        f64::from(w) * f64::from(h) * 4.0 / 1e6
    );

    let gpu_pre = WgpuPreprocessor::new(context.clone())?;
    let cfg = PreprocessConfig {
        input_size: INPUT,
        resize_quality: ResizeQuality::Speed,
    };
    let gpu_ms = time(RUNS, || {
        gpu_pre
            .preprocess_into_tensor(&image, &cfg, &tensor)
            .expect("gpu preprocess")
    });
    context.take_pass_timings()?;
    gpu_pre.preprocess_into_tensor(&image, &cfg, &tensor)?;
    let pass_ns: f64 = context
        .take_pass_timings()?
        .iter()
        .filter(|t| t.label == "preprocess")
        .map(|t| t.duration_ns)
        .sum();
    println!(
        "GPU preprocess (total)        {gpu_ms:7.2} ms   of it {:.2} ms is the shader",
        pass_ns / 1e6
    );
    println!(
        "  -> {:.2} ms is to_rgba8 + {:.1} MB texture upload + encode/submit",
        gpu_ms - pass_ns / 1e6,
        f64::from(w) * f64::from(h) * 4.0 / 1e6
    );

    let cpu = CpuPreprocessor;
    for (label, quality) in [
        ("CPU preprocess (nearest)  ", ResizeQuality::Speed),
        ("CPU preprocess (triangle) ", ResizeQuality::Quality),
    ] {
        let cfg = PreprocessConfig {
            input_size: INPUT,
            resize_quality: quality,
        };
        let ms = time(RUNS, || {
            cpu.preprocess(&image, &cfg).expect("cpu preprocess")
        });
        println!(
            "{label}    {ms:7.2} ms   uploads {:.1} MB",
            640.0 * 640.0 * 3.0 * 4.0 / 1e6
        );
    }

    // Both costs scale with source pixels, but at very different rates, so there is a
    // size below which uploading the full image is genuinely cheaper. Find it by
    // measurement rather than picking a round number.
    println!(
        "\n{:>11}  {:>10}  {:>10}  {:>8}",
        "source", "GPU pre", "CPU pre", "winner"
    );
    println!("{}", "-".repeat(46));
    for scale in [16u32, 12, 8, 6, 4, 3, 2] {
        let (sw, sh) = (w / scale, h / scale);
        if sw < INPUT.width || sh < INPUT.height {
            continue;
        }
        let scaled = image.resize_exact(sw, sh, image::imageops::FilterType::Triangle);
        let gpu = time(10, || {
            gpu_pre
                .preprocess_into_tensor(&scaled, &cfg, &tensor)
                .expect("gpu preprocess")
        });
        let cpu_cfg = PreprocessConfig {
            input_size: INPUT,
            resize_quality: ResizeQuality::Speed,
        };
        let cpu_ms = time(10, || {
            cpu.preprocess(&scaled, &cpu_cfg).expect("cpu preprocess")
        });
        println!(
            "{:>5}x{:<5}  {gpu:>7.2} ms  {cpu_ms:>7.2} ms  {:>8}",
            sw,
            sh,
            if gpu < cpu_ms { "GPU" } else { "CPU" }
        );
    }

    Ok(())
}
