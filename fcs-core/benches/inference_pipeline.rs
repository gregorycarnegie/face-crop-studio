#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::{
    env,
    hint::black_box,
    path::{Path, PathBuf},
};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fcs_core::{InputSize, PostprocessConfig, PreprocessConfig, YuNetDetector};
use fcs_utils::{config::ResizeQuality, load_fixture_image};

const MODEL_PATH: &str = "models/face_detection_yunet_2023mar_640.onnx";
const FIXTURE_IMAGE: &str = "images/006.jpg";
const INPUT_SIZE: InputSize = InputSize::new(640, 640);

fn resolve_model_path() -> Option<PathBuf> {
    if let Ok(value) = env::var("YUNET_MODEL_PATH") {
        let candidate = PathBuf::from(value);
        if candidate.exists() {
            return Some(candidate);
        }
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for ancestor in manifest_dir.ancestors() {
        let candidate = ancestor.join(MODEL_PATH);
        if candidate.exists() {
            return Some(candidate);
        }
    }

    None
}

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
    detectors
}

fn inference_pipeline_benchmark(c: &mut Criterion) {
    let model_path = match resolve_model_path() {
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
