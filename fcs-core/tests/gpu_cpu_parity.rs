use fcs_core::{
    DetectionOutput, PostprocessConfig, YuNetDetector,
    preprocess::{InputSize, PreprocessConfig, Preprocessor, WgpuPreprocessor},
};
use fcs_utils::{
    fixture_path,
    gpu::{GpuAvailability, GpuContext, GpuContextOptions},
    model_path,
};
use std::{path::Path, sync::Arc};

const MODEL_PATH: &str = "models/face_detection_yunet_2023mar_640.onnx";

struct FixtureSpec {
    path: &'static str,
    label: &'static str,
}

const FIXTURES: &[FixtureSpec] = &[
    FixtureSpec {
        path: "images/001.jpg",
        label: "single-face portrait",
    },
    FixtureSpec {
        path: "images/006.jpg",
        label: "crowded street",
    },
    FixtureSpec {
        path: "images/068.jpg",
        label: "profile view",
    },
    FixtureSpec {
        path: "images/168_o.jpg",
        label: "partially occluded subject",
    },
    FixtureSpec {
        path: "images/190_g.jpg",
        label: "group shot",
    },
    FixtureSpec {
        path: "images/014_n.jpg",
        label: "no-face negative",
    },
];

const SCORE_TOLERANCE: f32 = 1e-3;
const BBOX_TOLERANCE: f32 = 5.0;
const LANDMARK_TOLERANCE: f32 = 5.0;

#[derive(Default)]
struct ParityStats {
    max_score_delta: f32,
    max_bbox_delta: f32,
    max_landmark_delta: f32,
}

struct Detectors {
    cpu: YuNetDetector,
    gpu: YuNetDetector,
}

#[test]
fn gpu_inference_matches_cpu_baseline() {
    let Some(model_path) = model_path(MODEL_PATH).expect("resolve YuNet model") else {
        eprintln!("skipping GPU parity test (model {MODEL_PATH} missing)");
        return;
    };
    let model_path = model_path.as_path();

    let preprocess = PreprocessConfig {
        input_size: InputSize {
            width: 640,
            height: 640,
        },
        ..Default::default()
    };
    let postprocess = PostprocessConfig::default();

    let Some(detectors) = build_detectors(model_path, &preprocess, &postprocess) else {
        return;
    };

    let mut stats = ParityStats::default();
    for fixture in FIXTURES {
        let Ok(image_path) = fixture_path(fixture.path) else {
            eprintln!(
                "skipping GPU parity test (fixture {} missing)",
                fixture.path
            );
            return;
        };
        let image_path = image_path.as_path();

        let cpu = detectors
            .cpu
            .detect_path(image_path)
            .unwrap_or_else(|e| panic!("CPU detection failed on {}: {e}", fixture.path));
        let gpu = detectors
            .gpu
            .detect_path(image_path)
            .unwrap_or_else(|e| panic!("GPU detection failed on {}: {e}", fixture.path));

        compare_detections(&cpu, &gpu, fixture, &mut stats);
    }

    println!(
        "GPU parity suite passed on {} fixtures (max score Δ={:.4}, bbox Δ={:.2}px, landmark Δ={:.2}px)",
        FIXTURES.len(),
        stats.max_score_delta,
        stats.max_bbox_delta,
        stats.max_landmark_delta
    );
}

fn build_detectors(
    model_path: &Path,
    preprocess: &PreprocessConfig,
    postprocess: &PostprocessConfig,
) -> Option<Detectors> {
    let cpu = match YuNetDetector::new(model_path, preprocess.clone(), postprocess.clone()) {
        Ok(detector) => detector,
        Err(err) => {
            eprintln!("skipping GPU parity test (failed to build CPU detector: {err})");
            return None;
        }
    };

    let gpu = build_gpu_detector(model_path, preprocess, postprocess)?;

    Some(Detectors { cpu, gpu })
}

fn build_gpu_detector(
    model_path: &Path,
    preprocess: &PreprocessConfig,
    postprocess: &PostprocessConfig,
) -> Option<YuNetDetector> {
    let availability = GpuContext::init_with_fallback(&GpuContextOptions::default());
    let context = match availability {
        GpuAvailability::Available(ctx) => ctx,
        GpuAvailability::Disabled { reason } => {
            eprintln!("skipping GPU parity test (GPU disabled: {reason})");
            return None;
        }
        GpuAvailability::Unavailable { error } => {
            eprintln!("skipping GPU parity test (GPU unavailable: {error})");
            return None;
        }
    };

    let preprocessor: Arc<dyn Preprocessor> = match WgpuPreprocessor::new(context) {
        Ok(pre) => Arc::new(pre),
        Err(err) => {
            eprintln!("skipping GPU parity test (failed to build GPU preprocessor: {err})");
            return None;
        }
    };

    match YuNetDetector::with_gpu_preprocessor(
        model_path,
        preprocess.clone(),
        postprocess.clone(),
        preprocessor,
    ) {
        Ok(detector) => Some(detector),
        Err(err) => {
            eprintln!("skipping GPU parity test (GPU detector init failed: {err})");
            None
        }
    }
}

fn compare_detections(
    cpu: &DetectionOutput,
    gpu: &DetectionOutput,
    fixture: &FixtureSpec,
    stats: &mut ParityStats,
) {
    assert_eq!(
        cpu.detections.len(),
        gpu.detections.len(),
        "detection count mismatch for {} ({})",
        fixture.path,
        fixture.label
    );

    let mut cpu_sorted = cpu.detections.clone();
    cpu_sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    let mut gpu_sorted = gpu.detections.clone();
    gpu_sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    for (cpu_det, gpu_det) in cpu_sorted.iter().zip(gpu_sorted.iter()) {
        let score_delta = (cpu_det.score - gpu_det.score).abs();
        stats.max_score_delta = stats.max_score_delta.max(score_delta);
        assert!(
            score_delta <= SCORE_TOLERANCE,
            "score mismatch for {} ({}): {} vs {}",
            fixture.path,
            fixture.label,
            cpu_det.score,
            gpu_det.score
        );

        let cpu_bbox = [
            cpu_det.bbox.x,
            cpu_det.bbox.y,
            cpu_det.bbox.width,
            cpu_det.bbox.height,
        ];
        let gpu_bbox = [
            gpu_det.bbox.x,
            gpu_det.bbox.y,
            gpu_det.bbox.width,
            gpu_det.bbox.height,
        ];
        for i in 0..4 {
            let delta = (cpu_bbox[i] - gpu_bbox[i]).abs();
            stats.max_bbox_delta = stats.max_bbox_delta.max(delta);
            assert!(
                delta <= BBOX_TOLERANCE,
                "bbox[{i}] mismatch for {} ({}): {} vs {}",
                fixture.path,
                fixture.label,
                cpu_bbox[i],
                gpu_bbox[i]
            );
        }

        for (idx, (cpu_lm, gpu_lm)) in cpu_det
            .landmarks
            .iter()
            .zip(gpu_det.landmarks.iter())
            .enumerate()
        {
            let delta_x = (cpu_lm.x - gpu_lm.x).abs();
            let delta_y = (cpu_lm.y - gpu_lm.y).abs();
            stats.max_landmark_delta = stats.max_landmark_delta.max(delta_x.max(delta_y));
            assert!(
                delta_x <= LANDMARK_TOLERANCE,
                "landmark {idx} x mismatch for {} ({}): {} vs {}",
                fixture.path,
                fixture.label,
                cpu_lm.x,
                gpu_lm.x
            );
            assert!(
                delta_y <= LANDMARK_TOLERANCE,
                "landmark {idx} y mismatch for {} ({}): {} vs {}",
                fixture.path,
                fixture.label,
                cpu_lm.y,
                gpu_lm.y
            );
        }
    }
}
