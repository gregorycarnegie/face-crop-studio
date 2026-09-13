//! A configured input size the model cannot run must fail at construction, not per image.
//!
//! `input.width` and `input.height` are settings, and nothing rejected values the bundled
//! model does not accept. A folder run then loaded the model, read every file, and failed each
//! one separately -- "Got invalid dimensions for input" from ONNX Runtime, "stage0 conv" from
//! the WGSL graph, whose stem was then compiled in at 640 -- before ending with "all
//! detections failed" (experiment 74).

use fcs_core::{
    InferenceBackend, InputSize, PostprocessConfig, PreprocessConfig, YuNetDetector, YuNetModel,
};
use fcs_utils::model_path;

const MODEL: &str = "models/face_detection_yunet_2023mar_640.onnx";

fn configs(size: InputSize) -> (PreprocessConfig, PostprocessConfig) {
    (
        PreprocessConfig {
            input_size: size,
            resize_quality: fcs_utils::config::ResizeQuality::Quality,
        },
        PostprocessConfig::default(),
    )
}

#[test]
fn the_bundled_input_size_still_constructs() {
    let strict = std::env::var("FCS_STRICT_TESTS").is_ok();
    let Some(model) = model_path(MODEL).expect("resolve model") else {
        assert!(!strict, "model missing under FCS_STRICT_TESTS");
        eprintln!("skipping: model not found");
        return;
    };

    let (pre, post) = configs(InputSize::new(640, 640));
    YuNetDetector::new(&model, pre, post).expect("640x640 is what the bundled model expects");
}

#[test]
fn a_detector_reports_the_configuration_and_backend_it_was_built_with() {
    let strict = std::env::var("FCS_STRICT_TESTS").is_ok();
    let Some(model) = model_path(MODEL).expect("resolve model") else {
        assert!(!strict, "model missing under FCS_STRICT_TESTS");
        return;
    };
    // Distinctive values, so a defaulted config cannot pass for the one supplied. The size must
    // stay 640 for the bundled model, so the preprocess difference is the resize quality.
    let pre = PreprocessConfig {
        input_size: InputSize::new(640, 640),
        resize_quality: fcs_utils::config::ResizeQuality::Speed,
    };
    let post = PostprocessConfig {
        score_threshold: 0.73,
        nms_threshold: 0.41,
        top_k: 37,
    };

    let cpu = YuNetDetector::new(&model, pre.clone(), post.clone()).expect("cpu detector");
    let got = cpu.postprocess_config();
    assert_eq!(
        (got.score_threshold, got.nms_threshold, got.top_k),
        (0.73, 0.41, 37)
    );
    assert_eq!(cpu.preprocess_config().resize_quality, pre.resize_quality);
    assert!(["onnxruntime", "cpu-graph"].contains(&cpu.inference_backend()));
    assert_eq!(cpu.gpu_memory_usage(), None);

    // A GPU detector reports its pooled memory once it has run; skipped where there is no adapter.
    if let Ok(gpu) = YuNetDetector::new_gpu(&model, pre, post) {
        assert_eq!(gpu.inference_backend(), "wgsl-gpu");
        let image = image::DynamicImage::ImageRgb8(image::RgbImage::new(64, 48));
        gpu.detect_image(&image).expect("gpu detection");
        assert!(
            gpu.gpu_memory_usage().is_some_and(|bytes| bytes > 1),
            "pooled buffers after an inference"
        );
    }
}

#[test]
fn the_cpu_graph_reports_the_input_size_it_was_loaded_at() {
    let strict = std::env::var("FCS_STRICT_TESTS").is_ok();
    let Some(model) = model_path(MODEL).expect("resolve model") else {
        assert!(!strict, "model missing under FCS_STRICT_TESTS");
        return;
    };
    // 320, not the 640 default, so a defaulted size cannot pass. Only the built-in graph takes it.
    let size = InputSize::new(320, 320);
    let loaded =
        YuNetModel::load_with(&model, size, InferenceBackend::CpuGraph).expect("cpu graph at 320");
    assert_eq!(loaded.input_size(), size);
    assert_eq!(loaded.backend_name(), "cpu-graph");

    let graph = fcs_core::cpu::runtime::CpuYuNet::load(&model, size).expect("cpu graph at 320");
    assert_eq!(graph.input_size(), size);
}

#[test]
fn a_size_the_backend_cannot_run_fails_at_construction() {
    let strict = std::env::var("FCS_STRICT_TESTS").is_ok();
    let Some(model) = model_path(MODEL).expect("resolve model") else {
        assert!(!strict, "model missing under FCS_STRICT_TESTS");
        eprintln!("skipping: model not found");
        return;
    };

    // The built-in CPU graph derives its dimensions from the configured size and so accepts
    // these; ONNX Runtime and the WGSL graph do not. `YuNetDetector::new` takes whichever CPU
    // backend is available, so this asserts on the outcome rather than on which one ran: if it
    // constructs, the backend genuinely supports the size and detection must work.
    for size in [InputSize::new(320, 320), InputSize::new(960, 960)] {
        let (pre, post) = configs(size);
        match YuNetDetector::new(&model, pre, post) {
            Err(e) => {
                let text = format!("{e:#}");
                assert!(
                    text.contains("rejected a") && text.contains("input.width"),
                    "error should name the setting to change, got: {text}"
                );
            }
            Ok(detector) => {
                let image = image::DynamicImage::ImageRgb8(image::RgbImage::new(200, 150));
                detector.detect_image(&image).unwrap_or_else(|e| {
                    panic!(
                        "{}x{} constructed, so the probe passed, yet detection failed: {e:#}",
                        size.width, size.height
                    )
                });
            }
        }
    }
}
