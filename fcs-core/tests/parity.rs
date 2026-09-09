use fcs_core::{DetectionOutput, InputSize, PostprocessConfig, PreprocessConfig, YuNetDetector};
use fcs_utils::{fixtures_dir, load_fixture_json};
use serde::Deserialize;
use std::path::Path;

const MODEL_PATH: &str = "models/face_detection_yunet_2023mar_640.onnx";

/// The same faces OpenCV finds, in the same places, from the same weights.
///
/// Not the same numbers: the fixtures come from OpenCV's YuNet, which stretches a source to
/// the model input, and this detector letterboxes it instead (experiment 96). See
/// [`MAX_SCORE_DELTA`] for what that costs and why the limits are shaped the way they are.
#[test]
fn yunet_core_matches_opencv_parity() -> anyhow::Result<()> {
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join(MODEL_PATH);
    if !model_path.exists() {
        eprintln!(
            "skipping parity test; model missing at {}",
            model_path.display()
        );
        return Ok(());
    }

    let cases = collect_parity_cases(&fixtures_dir()?)?;
    let input_size = InputSize::new(640, 640);

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

        let detector = YuNetDetector::new(&model_path, preprocess, postprocess)?;
        let output = detector.detect_path(&image_path)?;

        assert_detections_close(&output, &fixture);
    }

    Ok(())
}

const MAX_CASES_PER_CATEGORY: usize = 3;

fn collect_parity_cases(
    fixture_root: &Path,
) -> anyhow::Result<Vec<(std::path::PathBuf, FixtureFile)>> {
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

    let images_dir = fixture_root.join("images");
    if !images_dir.is_dir() {
        eprintln!("skipping: fixtures/images not available in this environment");
        return Ok(vec![]);
    }

    let mut entries: Vec<_> = fs::read_dir(images_dir)?
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
        let fixture_path = fixture_root.join("opencv").join(format!("{}.json", stem));
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

/// Largest score difference from the OpenCV fixtures, and the smallest box overlap.
///
/// These are not rounding tolerances. The fixtures were produced by OpenCV's YuNet, which
/// stretches a source to the model input; this detector letterboxes it instead, so it shows
/// the model an undistorted face and gets a different -- usually better -- answer
/// (experiment 96). Over the corpus that shift measured a median IoU of 0.76-0.87 on
/// non-square sources, and these fixtures land in the same place: IoU 0.765 to 0.994, score
/// within 0.0175, landmarks within 9.6% of the box.
///
/// So what this test still checks is real and worth having -- the same faces, in the same
/// places, from the same weights and decode -- and what it deliberately no longer checks is
/// coordinate equality with a reference that preprocesses differently. Tightening these back
/// down would mean reproducing the distortion, which is the thing that was removed.
const MAX_SCORE_DELTA: f32 = 0.03;
const MIN_BOX_IOU: f32 = 0.70;
/// Landmark movement is bounded relative to the face, not in absolute pixels: 74 px on a
/// 974 px face is the same error as 7 px on a 95 px one, and only the ratio is comparable
/// across a corpus whose faces span an order of magnitude.
const MAX_LANDMARK_FRACTION: f32 = 0.15;

fn box_iou(a: &fcs_core::BoundingBox, b: &[f32; 4]) -> f32 {
    let ix = (a.x + a.width).min(b[0] + b[2]) - a.x.max(b[0]);
    let iy = (a.y + a.height).min(b[1] + b[3]) - a.y.max(b[1]);
    let inter = ix.max(0.0) * iy.max(0.0);
    let union = a.width * a.height + b[2] * b[3] - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

fn assert_detections_close(actual: &DetectionOutput, expected: &FixtureFile) {
    assert_eq!(
        actual.detections.len(),
        expected.detections.len(),
        "detection count mismatch"
    );

    // Pair each fixture face with the actual face that overlaps it most, rather than by
    // score rank: the two pipelines' scores differ by up to 0.0175, which is enough to swap
    // the order of two similar faces and produce a confusing failure about the wrong pair.
    let mut taken = vec![false; actual.detections.len()];
    for e in &expected.detections {
        let best = actual
            .detections
            .iter()
            .enumerate()
            .filter(|(i, _)| !taken[*i])
            .map(|(i, a)| (i, box_iou(&a.bbox, &e.bbox)))
            .max_by(|x, y| x.1.total_cmp(&y.1));
        let Some((idx, iou)) = best else {
            panic!("no detection left to pair with fixture face {:?}", e.bbox);
        };
        assert!(
            iou >= MIN_BOX_IOU,
            "box overlap {iou:.4} below {MIN_BOX_IOU}: {:?} against fixture {:?}",
            actual.detections[idx].bbox,
            e.bbox
        );
        taken[idx] = true;
        let a = &actual.detections[idx];

        // Failure messages carry the numbers: a bare "score mismatch" says nothing about
        // whether a change moved a detection by a rounding error or across the image.
        let score_delta = (a.score - e.score).abs();
        assert!(
            score_delta <= MAX_SCORE_DELTA,
            "score {} against fixture {} (delta {score_delta}, limit {MAX_SCORE_DELTA})",
            a.score,
            e.score
        );

        let face = a.bbox.width.max(a.bbox.height);
        for (idx, (al, el)) in a.landmarks.iter().zip(e.landmarks.iter()).enumerate() {
            let delta = (al.x - el[0]).abs().max((al.y - el[1]).abs());
            assert!(
                delta <= face * MAX_LANDMARK_FRACTION,
                "landmark {idx} moved {delta:.2} px on a {face:.0} px face \
                 ({:.1}%, limit {:.0}%)",
                100.0 * delta / face,
                100.0 * MAX_LANDMARK_FRACTION
            );
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
