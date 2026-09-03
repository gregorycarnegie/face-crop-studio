//! Does the pure-Rust CPU graph agree with `tract` on real images?
//!
//! The CPU backend re-implements YuNet from a hand-encoded topology rather than
//! interpreting the ONNX graph, so nothing structural forces it to agree. A
//! transposed weight layout, an off-by-one pad, or a missing ReLU all still
//! produce plausible-looking numbers. This compares final detections over
//! fixtures, because decode and NMS sit downstream and can turn a small numeric
//! difference into a different face.

use fcs_core::{
    InferenceBackend, InputSize, PostprocessConfig, PreprocessConfig, YuNetModel,
    cpu::runtime::CpuYuNet, preprocess_dynamic_image,
};
use fcs_utils::{fixtures_dir, model_path};

const MODEL: &str = "models/face_detection_yunet_2023mar_640.onnx";
// Same budget the GPU parity suite allows: both are "a different implementation
// computed this".
const SCORE_TOL: f32 = 1e-3;
const COORD_TOL: f32 = 5.0;

#[test]
fn cpu_graph_matches_tract_detections() {
    let strict = std::env::var("FCS_STRICT_TESTS").is_ok();
    let Some(model) = model_path(MODEL).expect("resolve model") else {
        assert!(!strict, "model missing under FCS_STRICT_TESTS");
        eprintln!("skipping: model not found");
        return;
    };

    let size = InputSize::new(640, 640);
    let cpu = CpuYuNet::load(&model, size).expect("cpu graph should load");
    let tract = YuNetModel::load_with(&model, size, InferenceBackend::Tract).expect("tract");

    let cfg = PreprocessConfig {
        input_size: size,
        resize_quality: fcs_utils::config::ResizeQuality::Quality,
    };
    let post = PostprocessConfig::default();

    let dir = fixtures_dir().expect("fixtures dir").join("images");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .expect("read fixtures/images")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("jpg")))
        .collect();
    paths.sort();
    let step = (paths.len() / 20).max(1);

    let mut compared = 0usize;
    for path in paths.into_iter().step_by(step).take(20) {
        let Ok(image) = image::open(&path) else {
            continue;
        };
        let pre = preprocess_dynamic_image(&image, &cfg).expect("preprocess");

        let expected = fcs_core::apply_postprocess(
            &tract.run(pre.tensor.clone()).expect("tract run"),
            pre.scale_x,
            pre.scale_y,
            &post,
        )
        .expect("tract postprocess");
        let actual = fcs_core::apply_postprocess(
            &cpu.run(pre.tensor.clone()).expect("cpu run"),
            pre.scale_x,
            pre.scale_y,
            &post,
        )
        .expect("cpu postprocess");

        assert_eq!(
            expected.len(),
            actual.len(),
            "{}: tract found {} faces, cpu found {}",
            path.display(),
            expected.len(),
            actual.len()
        );
        for (i, (want, got)) in expected.iter().zip(actual.iter()).enumerate() {
            assert!(
                (want.score - got.score).abs() <= SCORE_TOL,
                "{} face {i}: score {} vs {}",
                path.display(),
                want.score,
                got.score
            );
            for (name, l, r) in [
                ("x", want.bbox.x, got.bbox.x),
                ("y", want.bbox.y, got.bbox.y),
                ("w", want.bbox.width, got.bbox.width),
                ("h", want.bbox.height, got.bbox.height),
            ] {
                assert!(
                    (l - r).abs() <= COORD_TOL,
                    "{} face {i}: bbox {name} {l} vs {r}",
                    path.display()
                );
            }
        }
        compared += 1;
    }
    assert!(compared > 0, "no fixtures compared");
    eprintln!("cpu/tract parity OK across {compared} fixtures");
}
