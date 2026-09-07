//! A configured input size the model cannot run must fail at construction, not per image.
//!
//! `input.width` and `input.height` are settings, and nothing rejected values the bundled
//! model does not accept. A folder run then loaded the model, read every file, and failed each
//! one separately -- "Got invalid dimensions for input" from ONNX Runtime, "stage0 conv" from
//! the WGSL graph, whose stage dimensions are compiled in as 640 -- before ending with "all
//! detections failed" (experiment 74).

use fcs_core::{InputSize, PostprocessConfig, PreprocessConfig, YuNetDetector};
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
