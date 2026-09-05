#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::{hint::black_box, path::Path, sync::Arc};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fcs_core::{InputSize, PostprocessConfig, PreprocessConfig, WgpuPreprocessor, YuNetDetector};
use fcs_utils::{
    config::ResizeQuality,
    gpu::{GpuAvailability, GpuContext, GpuContextOptions},
    load_fixture_image, model_path,
};

const MODEL_PATH: &str = "models/face_detection_yunet_2023mar_640.onnx";
const FIXTURE_IMAGE: &str = "images/006.jpg";
const INPUT_SIZE: InputSize = InputSize::new(640, 640);

fn build_detectors(model_path: &Path) -> Vec<(&'static str, YuNetDetector)> {
    let mut detectors = Vec::new();
    for (label, resize_quality) in [
        ("quality", ResizeQuality::Quality),
        ("speed", ResizeQuality::Speed),
    ] {
        let preprocess = PreprocessConfig {
            input_size: INPUT_SIZE,
            resize_quality,
        };
        match YuNetDetector::new(model_path, preprocess, PostprocessConfig::default()) {
            Ok(detector) => detectors.push((label, detector)),
            Err(err) => {
                eprintln!("skipping inference benchmarks; failed to build {label} detector: {err}");
                return Vec::new();
            }
        }
    }

    // GPU inference against the same CPU preprocessor and the same resize quality as the
    // "speed" case above, so the pair isolates the inference backend and nothing else.
    // Skipped rather than fatal: there is no GPU adapter in CI.
    let gpu_preprocess = PreprocessConfig {
        input_size: INPUT_SIZE,
        resize_quality: ResizeQuality::Speed,
    };
    match YuNetDetector::new_gpu(model_path, gpu_preprocess, PostprocessConfig::default()) {
        Ok(detector) => detectors.push(("gpu", detector)),
        Err(err) => eprintln!("skipping the gpu inference benchmark; detector init failed: {err}"),
    }

    // Same GPU inference, but a filtered CPU resize rather than nearest. This is the
    // quality-comparable partner to `gpu_on_device`: nearest aliases badly on a 6x
    // downscale, so timing the on-device path against `gpu` alone flatters it.
    let gpu_quality_preprocess = PreprocessConfig {
        input_size: INPUT_SIZE,
        resize_quality: ResizeQuality::Quality,
    };
    match YuNetDetector::new_gpu(
        model_path,
        gpu_quality_preprocess,
        PostprocessConfig::default(),
    ) {
        Ok(detector) => detectors.push(("gpu_quality", detector)),
        Err(err) => eprintln!("skipping the gpu_quality benchmark; detector init failed: {err}"),
    }

    // Preprocess on the GPU as well, which is what lets `detect_on_device` fuse the two
    // stages onto one device and skip the CPU resize entirely. `new_gpu` above always
    // pairs GPU inference with the CPU preprocessor, so without this case the bench
    // cannot see the fully on-device path the app actually prefers.
    match build_on_device_detector(model_path) {
        Ok(Some(detector)) => detectors.push(("gpu_on_device", detector)),
        Ok(None) => eprintln!("skipping the on-device benchmark; no GPU adapter"),
        Err(err) => eprintln!("skipping the on-device benchmark; init failed: {err}"),
    }

    detectors
}

fn build_on_device_detector(model_path: &Path) -> anyhow::Result<Option<YuNetDetector>> {
    let context = match GpuContext::init_with_fallback(&GpuContextOptions::default()) {
        GpuAvailability::Available(ctx) => ctx,
        _ => return Ok(None),
    };
    let preprocessor = Arc::new(WgpuPreprocessor::new(context)?);
    let preprocess = PreprocessConfig {
        input_size: INPUT_SIZE,
        resize_quality: ResizeQuality::Speed,
    };
    YuNetDetector::with_gpu_preprocessor(
        model_path,
        preprocess,
        PostprocessConfig::default(),
        preprocessor,
    )
    .map(Some)
}

fn inference_pipeline_benchmark(c: &mut Criterion) {
    let model_path = match model_path(MODEL_PATH).expect("resolve YuNet model") {
        Some(path) => path,
        None => {
            eprintln!(
                "skipping inference benchmarks; model missing at {} (set YUNET_MODEL_PATH to override)",
                MODEL_PATH
            );
            return;
        }
    };

    let image = match load_fixture_image(FIXTURE_IMAGE) {
        Ok(image) => image,
        Err(err) => {
            eprintln!(
                "skipping inference benchmarks; failed to load fixture {FIXTURE_IMAGE}: {err}"
            );
            return;
        }
    };

    let detectors = build_detectors(model_path.as_path());
    if detectors.is_empty() {
        return;
    }

    let mut group = c.benchmark_group("inference_pipeline");
    for (label, detector) in detectors.iter() {
        group.bench_with_input(
            BenchmarkId::new("detect_image", label),
            detector,
            |b, det| {
                b.iter(|| {
                    let output = det
                        .detect_image(black_box(&image))
                        .expect("detection should succeed");
                    black_box(output.detections.len());
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, inference_pipeline_benchmark);
criterion_main!(benches);
