use fcs_core::{DetectionOutput, InputSize, PostprocessConfig, PreprocessConfig, YuNetDetector};
use fcs_utils::load_fixture_json;
use serde::Deserialize;
use std::path::Path;

const MODEL_PATH: &str = "models/face_detection_yunet_2023mar.onnx";

#[test]
fn yunet_core_matches_opencv_parity() -> anyhow::Result<()> {
    let model_path = Path::new(MODEL_PATH);
    if !model_path.exists() {
        eprintln!(
            "skipping parity test; model missing at {}",
            model_path.display()
        );
        return Ok(());
    }

    let cases = collect_parity_cases()?;
    let input_size = InputSize::new(320, 320);

    for (image_path, fixture) in cases {
        let preprocess = PreprocessConfig {
            input_size,
            ..Default::default()
        };
        let postprocess = PostprocessConfig {
            score_threshold: fixture
                .score_threshold
                .unwrap_or(PostprocessConfig::default().score_threshold),
            nms_threshold: fixture
                .nms_threshold
                .unwrap_or(PostprocessConfig::default().nms_threshold),
            top_k: fixture.top_k.unwrap_or(PostprocessConfig::default().top_k),
        };

        let detector = YuNetDetector::new(model_path, preprocess, postprocess)?;
        let output = detector.detect_path(&image_path)?;

        assert_detections_close(&output, &fixture, 1e-2);
    }

    Ok(())
}

const MAX_CASES_PER_CATEGORY: usize = 3;

fn collect_parity_cases() -> anyhow::Result<Vec<(std::path::PathBuf, FixtureFile)>> {
    use std::{collections::HashMap, fs};

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum Category {
        Single,
        Group,
        Occluded,
        Negative,
    }

    fn classify(name: &str) -> Category {
        if name.ends_with("_g") {
            Category::Group
        } else if name.ends_with("_o") {
            Category::Occluded
        } else if name.ends_with("_n") {
            Category::Negative
        } else {
            Category::Single
        }
    }

    let mut counts: HashMap<Category, usize> = HashMap::new();
    let mut cases = Vec::new();

    if !Path::new("fixtures/images").is_dir() {
        eprintln!("skipping: fixtures/images not available in this environment");
        return Ok(vec![]);
    }

    let mut entries: Vec<_> = fs::read_dir("fixtures/images")?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().is_file())
        .collect();
    entries.sort_by_key(|entry| entry.file_name());

    for entry in entries {
        let image_path = entry.path();
        let stem = match image_path.file_stem().and_then(|s| s.to_str()) {
            Some(stem) => stem,
            None => continue,
        };
        let category = classify(stem);
        let count = counts.entry(category).or_insert(0);
        if *count >= MAX_CASES_PER_CATEGORY {
            continue;
        }
        let fixture_path = Path::new("fixtures/opencv").join(format!("{}.json", stem));
        if !fixture_path.exists() {
            continue;
        }
        let fixture: FixtureFile = load_fixture_json(&fixture_path)?;
        if category == Category::Negative {
            if !fixture.detections.is_empty() {
                continue;
            }
        } else if fixture.detections.is_empty() {
            continue;
        }
        cases.push((image_path, fixture));
        *count += 1;
    }

    Ok(cases)
}

fn assert_detections_close(actual: &DetectionOutput, expected: &FixtureFile, tol: f32) {
    assert_eq!(
        actual.detections.len(),
        expected.detections.len(),
        "detection count mismatch"
    );

    let mut actual_sorted = actual.detections.clone();
    actual_sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    let mut expected_sorted = expected.detections.clone();
    expected_sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    for (a, e) in actual_sorted.iter().zip(expected_sorted.iter()) {
        assert!((a.score - e.score).abs() <= tol, "score mismatch");
        let actual_bbox = [a.bbox.x, a.bbox.y, a.bbox.width, a.bbox.height];
        for (idx, ev) in e.bbox.iter().enumerate() {
            assert!((actual_bbox[idx] - ev).abs() <= tol, "bbox mismatch");
        }
        for (al, el) in a.landmarks.iter().zip(e.landmarks.iter()) {
            let actual_lm = [al.x, al.y];
            for coord_idx in 0..2 {
                assert!(
                    (actual_lm[coord_idx] - el[coord_idx]).abs() <= tol,
                    "landmark mismatch"
                );
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
struct FixtureDetection {
    score: f32,
    bbox: [f32; 4],
    landmarks: [[f32; 2]; 5],
}

#[derive(Debug, Deserialize)]
struct FixtureFile {
    #[serde(default)]
    score_threshold: Option<f32>,
    #[serde(default)]
    nms_threshold: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    detections: Vec<FixtureDetection>,
}
