//! Does the shipped detector still find faces as well as it did?
//!
//! The parity tests prove the engines agree with each other and with the Python reference. None
//! of them notice a detector that agrees with itself while getting worse: a new export, a changed
//! default threshold, or a preprocessing change made in both places at once. This scores the
//! detector against human-drawn boxes instead.
//!
//! The fixtures are 60 Open Images photographs (145 faces) sampled from the held-out set behind
//! `docs/MODEL_COMPARISON.md`, all CC BY 2.0 and attributed in `fixtures/images/ATTRIBUTION.md`;
//! `tools/dataset/make_accuracy_fixtures.py` rebuilds them. Scoring follows that benchmark:
//! one-to-one, score-ordered matching at IoU >= 0.5, and an unmatched detection that overlaps a
//! group-of or depiction region at IoU > 0.3 is not counted against the detector.
//!
//! It runs at the application's default confidence and NMS, on the CPU graph, which gives the
//! same answer on every CI leg. The bounds are counts measured on that graph with a little room
//! for float differences between CPUs. A detection score near the threshold is the only thing that
//! can move, and only by a face or two. A real regression moves many.
//!
//! ```text
//! cargo test -p fcs-core --test detection_accuracy -- --nocapture
//! ```

use std::path::PathBuf;

use fcs_core::{Detection, ScrfdDetector};
use fcs_utils::config::{DEFAULT_CONFIDENCE, DEFAULT_NMS_THRESHOLD};

/// Measured 107 of 145 (73.8%) on the development machine, less a margin of three. Five
/// detections score within 0.02 of the 0.5 threshold, one at 0.5012; the same graph on another
/// CPU moves scores by around 1e-5, so three faces is generous for float noise and still far
/// below what any real regression costs.
const MIN_MATCHED: usize = 104;
/// Measured 2 (a few unannotated faces count here too), plus the same margin.
const MAX_UNMATCHED: usize = 5;

fn iou(a: &Detection, b: &[f32]) -> f32 {
    let (ax2, ay2) = (a.bbox.x + a.bbox.width, a.bbox.y + a.bbox.height);
    let ix = (ax2.min(b[2]) - a.bbox.x.max(b[0])).max(0.0);
    let iy = (ay2.min(b[3]) - a.bbox.y.max(b[1])).max(0.0);
    let inter = ix * iy;
    let union = a.bbox.width * a.bbox.height + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

fn boxes(value: &serde_json::Value) -> Vec<Vec<f32>> {
    serde_json::from_value(value.clone()).expect("boxes are [x1, y1, x2, y2] arrays")
}

/// (matched faces, unmatched detections) for one image.
fn score(
    mut detections: Vec<Detection>,
    faces: &[Vec<f32>],
    ignore: &[Vec<f32>],
) -> (usize, usize) {
    detections.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut taken = vec![false; faces.len()];
    let (mut matched, mut unmatched) = (0, 0);
    for detection in &detections {
        let best = (0..faces.len())
            .filter(|&i| !taken[i])
            .map(|i| (i, iou(detection, &faces[i])))
            .filter(|&(_, overlap)| overlap >= 0.5)
            .max_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((i, _)) = best {
            taken[i] = true;
            matched += 1;
        } else if !ignore.iter().any(|region| iou(detection, region) > 0.3) {
            unmatched += 1;
        }
    }
    (matched, unmatched)
}

#[test]
fn the_detector_finds_the_annotated_faces() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let model = root.join("models/scrfd80k_500m_640.onnx");
    let Some(detector) =
        ScrfdDetector::load_from_with_gpu(&model, &fcs_utils::GpuContextOptions::disabled())
    else {
        assert!(
            std::env::var_os("FCS_STRICT_TESTS").is_none(),
            "FCS_STRICT_TESTS: SCRFD did not load from {}",
            model.display()
        );
        eprintln!("skipped: SCRFD did not load from {}", model.display());
        return;
    };

    let images = root.join("fixtures/images");
    let truth: serde_json::Value = serde_json::from_slice(
        &std::fs::read(images.join("ground_truth.json")).expect("fixtures/images is committed"),
    )
    .expect("ground_truth.json is JSON");
    let truth = truth["images"].as_object().expect("an images object");

    let (mut faces, mut matched, mut unmatched) = (0, 0, 0);
    for (name, entry) in truth {
        let image = image::open(images.join(name)).expect("fixture image decodes");
        let detections = detector
            .detect(&image, DEFAULT_CONFIDENCE, DEFAULT_NMS_THRESHOLD)
            .expect("detection runs");
        let expected = boxes(&entry["faces"]);
        let (found, extra) = score(detections, &expected, &boxes(&entry["ignore"]));
        faces += expected.len();
        matched += found;
        unmatched += extra;
    }

    let recall = matched as f32 / faces as f32;
    eprintln!(
        "{} images: matched {matched}/{faces} faces ({:.1}% recall), {unmatched} unmatched",
        truth.len(),
        100.0 * recall
    );
    assert!(
        matched >= MIN_MATCHED,
        "recall fell: {matched} of {faces} faces found, at least {MIN_MATCHED} expected"
    );
    assert!(
        unmatched <= MAX_UNMATCHED,
        "{unmatched} unmatched detections, at most {MAX_UNMATCHED} expected"
    );
}
