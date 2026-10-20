//! Compare complete CUDA/WESL detection and enhancement, including transfers and CPU stages.
//! cargo run --release -p fcs-core --example cuda_comparison -- photo.jpg
use std::time::Instant;

use anyhow::{Context, Result};
use fcs_core::FaceDetector;
use fcs_utils::{
    EnhancementRuntime, EnhancementSettings,
    config::{CropSettings, GpuSettings},
    gpu::{GpuAvailability, GpuContext, GpuContextOptions},
};

fn median_ms(mut work: impl FnMut() -> Result<()>) -> Result<f64> {
    for _ in 0..3 {
        work()?;
    }
    let mut times = Vec::new();
    for _ in 0..15 {
        let start = Instant::now();
        work()?;
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(f64::total_cmp);
    Ok(times[times.len() / 2])
}

fn main() -> Result<()> {
    env_logger::init();
    let path = std::env::args().nth(1).context("pass a photograph path")?;
    let image = fcs_utils::load_image(path)?;
    let model = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("models/scrfd80k_500m_640.onnx");
    let context = match GpuContext::init_with_fallback(&GpuContextOptions::default()) {
        GpuAvailability::Available(context) => context,
        other => anyhow::bail!("WESL GPU required: {other:?}"),
    };
    println!("Adapter: {}", context.adapter_info().name);
    anyhow::ensure!(
        context.adapter_info().device_type != wgpu::DeviceType::Cpu,
        "a hardware WESL adapter is required for a GPU timing comparison"
    );
    let wesl = FaceDetector::load_from_with_context(&model, Some(context.clone()))
        .context("WESL detector model required")?;
    anyhow::ensure!(
        wesl.inference_backend() == "wgsl-gpu",
        "WESL detector required, got {}",
        wesl.inference_backend()
    );
    let cuda = FaceDetector::load_from_cuda(
        &model,
        &GpuSettings {
            experimental_cuda: true,
            ..Default::default()
        },
    )?;
    for (label, detector) in [("WESL", wesl), ("CUDA", cuda)] {
        let ms = median_ms(|| {
            detector.detect_image(&image)?;
            Ok(())
        })?;
        println!("{label}: complete detection {ms:.2} ms");
    }
    let crop = image.resize_exact(512, 512, image::imageops::FilterType::Triangle);
    let enhancement = EnhancementSettings {
        auto_color: true,
        skin_smooth_amount: 0.4,
        background_blur: true,
        ..EnhancementSettings::natural()
    };
    let settings = CropSettings {
        shape: fcs_utils::CropShape::Ellipse,
        vignette_softness: 0.3,
        ..Default::default()
    };
    for (label, runtime) in [
        ("WESL", EnhancementRuntime::new(Some(context))),
        ("CUDA", EnhancementRuntime::experimental_cuda()?),
    ] {
        let ms = median_ms(|| {
            std::hint::black_box(runtime.finish_crop(
                crop.clone(),
                &settings,
                Some(&enhancement),
                None,
            ));
            Ok(())
        })?;
        println!("{label}: 512x512 enhancement, quality, shape and fill {ms:.2} ms");
    }
    Ok(())
}
