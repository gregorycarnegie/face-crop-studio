//! Do the alternative backends agree with tract on detections?
//!
//! Timing was never the risk — a runtime swap that quietly moves boxes is.
//! Comparing raw head tensors is not enough: decode and NMS sit downstream and
//! can amplify a small numeric difference into a different face, so this
//! compares final detections over real fixtures.
//!
//! Skips when no compatible ONNX Runtime is present, unless FCS_STRICT_TESTS is
//! set, matching how the GPU parity tests treat a missing adapter.

mod common;

use common::TractOracle;
use fcs_core::{InferenceBackend, InputSize, PostprocessConfig, PreprocessConfig, YuNetModel};
use fcs_utils::model_path;

const MODEL: &str = "models/face_detection_yunet_2023mar_640.onnx";
// The GPU parity suite allows 1e-3 on score and 5 px on geometry; the same
// budget applies here, since both are "a different runtime computed this".
const SCORE_TOL: f32 = 1e-3;
const COORD_TOL: f32 = 5.0;

/// Every backend must agree with tract, which is the only one that interprets
/// the ONNX graph rather than re-implementing it.
#[test]
fn every_backend_matches_tract_detections() {
    let strict = std::env::var("FCS_STRICT_TESTS").is_ok();
    let Some(model) = model_path(MODEL).expect("resolve model") else {
        assert!(!strict, "model missing under FCS_STRICT_TESTS");
        eprintln!("skipping: model not found");
        return;
    };

    let size = InputSize::new(640, 640);
    let tract = TractOracle::load(&model, 640, 640).expect("tract oracle should load");

    // The built-in graph needs nothing installed, so it is always compared.
    let cpu_graph =
        YuNetModel::load_with(&model, size, InferenceBackend::CpuGraph).expect("cpu graph load");
    assert_eq!(cpu_graph.backend_name(), "cpu-graph");

    let mut alternatives = vec![("cpu-graph", cpu_graph)];
    match YuNetModel::load_with(&model, size, InferenceBackend::OnnxRuntime) {
        Ok(ort) => {
            assert_eq!(ort.backend_name(), "onnxruntime");
            alternatives.push(("onnxruntime", ort));
        }
        Err(err) => {
            assert!(
                !strict,
                "ONNX Runtime unavailable under FCS_STRICT_TESTS: {err}"
            );
            eprintln!("note: skipping the ONNX Runtime comparison ({err})");
        }
    }

    let cfg = PreprocessConfig {
        input_size: size,
        resize_quality: fcs_utils::config::ResizeQuality::Quality,
    };
    let post = PostprocessConfig::default();
    let mut compared = 0usize;

    // Committed synthetic portraits are available on a clean checkout too.
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../samples");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .expect("read samples")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("jpg")))
        .collect();
    paths.sort();
    let step = (paths.len() / 20).max(1);

    for path in paths.into_iter().step_by(step).take(20) {
        let Ok(image) = image::open(&path) else {
            continue;
        };
        let pre = fcs_core::preprocess_dynamic_image(&image, &cfg).expect("preprocess");

        let a = tract
            .detect(pre.tensor.as_slice(), pre.scale_x, pre.scale_y, &post)
            .expect("tract oracle run");

        for (name, model) in &alternatives {
            let b = fcs_core::apply_postprocess(
                &model.run(pre.tensor.clone()).expect("backend run"),
                pre.scale_x,
                pre.scale_y,
                &post,
            )
            .expect("backend postprocess");

            assert_eq!(
                a.len(),
                b.len(),
                "{}: tract found {} faces, {name} found {}",
                path.display(),
                a.len(),
                b.len()
            );
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                assert!(
                    (x.score - y.score).abs() <= SCORE_TOL,
                    "{} [{name}] face {i}: score {} vs {}",
                    path.display(),
                    x.score,
                    y.score
                );
                for (label, l, r) in [
                    ("x", x.bbox.x, y.bbox.x),
                    ("y", x.bbox.y, y.bbox.y),
                    ("w", x.bbox.width, y.bbox.width),
                    ("h", x.bbox.height, y.bbox.height),
                ] {
                    assert!(
                        (l - r).abs() <= COORD_TOL,
                        "{} [{name}] face {i}: bbox {label} {l} vs {r}",
                        path.display()
                    );
                }
            }
        }
        compared += 1;
    }
    assert!(compared > 0, "no fixtures compared");
    eprintln!(
        "parity OK across {compared} fixtures for: {}",
        alternatives
            .iter()
            .map(|(n, _)| *n)
            .collect::<Vec<_>>()
            .join(", ")
    );
}
