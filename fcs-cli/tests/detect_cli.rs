use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
};

use assert_cmd::cargo::cargo_bin_cmd;
use fcs_utils::{fixture_path, load_fixture_json, normalize_path};
use image::{ImageBuffer, Rgb};
use serde::Deserialize;
use serde_json::Value;
use tempfile::tempdir;

const MODEL_REL_PATH: &str = "../models/face_detection_yunet_2023mar_640.onnx";
/// How far a float in the CLI's JSON may drift from the snapshot.
///
/// This has to be loose enough to span backends. Detection now runs on ONNX
/// Runtime when the library is present and the built-in graph otherwise, and
/// the two agree to about 2e-4 px on box coordinates and exactly on scores —
/// so at the previous 1e-5 the snapshot passed on one backend and failed on the
/// other, making a green suite depend on what happened to be installed.
///
/// 1e-3 keeps roughly five times the observed cross-backend spread while still
/// catching any real regression, which moves boxes by whole pixels rather than
/// by the last digit.
const SNAPSHOT_FLOAT_TOLERANCE: f64 = 1.0e-3;

#[test]
fn detect_single_image_produces_json_output() -> Result<(), Box<dyn Error>> {
    let Some(model) = ensure_model_path() else {
        return Ok(());
    };

    let work_dir = tempdir()?;
    let image_path = work_dir.path().join("sample.png");
    let json_path = work_dir.path().join("out.json");

    // Generate a simple RGB test image.
    let img = ImageBuffer::from_fn(32, 32, |x, y| {
        let r = ((x + y) % 255) as u8;
        Rgb([r, 128, 255u8.saturating_sub(r)])
    });
    img.save(&image_path)?;

    let empty: Vec<String> = Vec::new();
    let detections = run_cli_detection(&image_path, &json_path, &model, &empty)?;
    assert_eq!(detections.len(), 1, "expected exactly one CLI output entry");
    let expected_image_path = image_path.canonicalize()?.display().to_string();
    assert_eq!(
        detections[0].image, expected_image_path,
        "CLI should echo the image path"
    );

    Ok(())
}

#[test]
fn detect_annotate_matches_fixture_when_no_detections() -> Result<(), Box<dyn Error>> {
    let Some(model) = ensure_model_path() else {
        return Ok(());
    };
    if !fixtures_available() {
        eprintln!("skipping: fixture images not available in this environment");
        return Ok(());
    }
    let fixture_image = fixture_path("images/test_pattern.png")?;

    let work_dir = tempdir()?;
    let input_path = work_dir.path().join("pattern.png");
    fs::copy(&fixture_image, &input_path)?;
    let json_path = work_dir.path().join("out.json");
    let annotate_dir = work_dir.path().join("annotated");

    let extra = vec![
        "--annotate".to_string(),
        annotate_dir.to_str().unwrap().to_string(),
        "--score-threshold".to_string(),
        "2.0".to_string(),
    ];
    let detections = run_cli_detection(&input_path, &json_path, &model, &extra)?;
    assert_eq!(detections.len(), 1);
    assert!(detections[0].detections.is_empty());

    let annotated_path = annotate_dir.join("pattern.png");
    assert!(
        annotated_path.exists(),
        "annotated image missing at {}",
        annotated_path.display()
    );

    let original = image::open(fixture_image)?.into_rgba8();
    let annotated = image::open(&annotated_path)?.into_rgba8();
    assert_eq!(annotated.dimensions(), original.dimensions());
    assert_eq!(annotated.as_raw(), original.as_raw());

    let expected: FixtureFile = load_fixture_json("opencv/pattern_no_faces.json")?;
    assert!(
        expected.detections.is_empty(),
        "fixture should have no detections"
    );

    Ok(())
}

#[test]
fn cli_detections_match_opencv_parity_samples() -> Result<(), Box<dyn Error>> {
    let Some(model) = ensure_model_path() else {
        return Ok(());
    };
    if !fixtures_available() {
        eprintln!("skipping: fixture images not available in this environment");
        return Ok(());
    }
    // The third field is how many faces this detector finds that the OpenCV fixture does
    // not. It is not slack: every entry is a face that was looked at. Letterboxing shows the
    // model an undistorted face and so finds some that a squashed one hid (experiment 96
    // counted 77 such images in 1239), and the count is pinned per fixture so that a *new*
    // divergence fails here and has to be looked at rather than absorbed.
    let cases = [
        ("images/006.jpg", "opencv/006.json", 0),
        ("images/190_g.jpg", "opencv/190_g.json", 0),
        ("images/002_n.jpg", "opencv/002_n.json", 0),
        ("images/168_o.jpg", "opencv/168_o.json", 0),
        ("images/169_o.jpg", "opencv/169_o.json", 0),
        ("images/250_o.jpg", "opencv/250_o.json", 0),
        ("images/253_o.jpg", "opencv/253_o.json", 0),
        ("images/255_o.jpg", "opencv/255_o.json", 0),
        // A man holding a mask over his nose and mouth, eyes clear above it. OpenCV finds
        // nothing here at 0.9; this detector finds the face at 0.912, and the annotated
        // output puts the box and all five landmarks on it. A true positive the distortion
        // was costing, not a hallucination.
        ("images/258_o.webp", "opencv/258_o.json", 1),
    ];

    for (image_rel, fixture_rel, extra_faces) in cases {
        let image_path = fixture_path(image_rel)?;
        let fixture: FixtureFile = load_fixture_json(fixture_rel)?;
        let work_dir = tempdir()?;
        let json_path = work_dir.path().join("out.json");
        let mut extra = Vec::new();
        if let Some(score) = fixture.score_threshold
            && (score - 0.9).abs() > f64::EPSILON
        {
            extra.push("--score-threshold".to_string());
            extra.push(score.to_string());
        }
        if let Some(nms) = fixture.nms_threshold
            && (nms - 0.3).abs() > f64::EPSILON
        {
            extra.push("--nms-threshold".to_string());
            extra.push(nms.to_string());
        }
        if let Some(top_k) = fixture.top_k
            && top_k != 5000
        {
            extra.push("--top-k".to_string());
            extra.push(top_k.to_string());
        }
        let detections = run_cli_detection(&image_path, &json_path, &model, &extra)?;
        assert_eq!(
            detections.len(),
            1,
            "expected single CLI output entry for {}",
            image_rel
        );

        let actual_list = &detections[0].detections;
        assert_detections_close(actual_list, &fixture.detections, extra_faces, image_rel);
    }

    Ok(())
}

fn fixtures_available() -> bool {
    fcs_utils::fixture_path("images/006.jpg").is_ok()
}

fn ensure_model_path() -> Option<PathBuf> {
    let path = Path::new(MODEL_REL_PATH);
    if !path.exists() {
        eprintln!(
            "skipping test because YuNet model is missing at {}",
            path.display()
        );
        return None;
    }
    Some(normalize_path(path).expect("normalize_path should succeed"))
}

fn run_cli_detection(
    image_path: &Path,
    json_path: &Path,
    model_path: &Path,
    extra_args: &[String],
) -> Result<Vec<CliDetectionRecord>, Box<dyn Error>> {
    let mut cmd = cargo_bin_cmd!("fcs-cli");
    cmd.arg("--input")
        .arg(image_path)
        .arg("--model")
        .arg(model_path)
        .arg("--no-gpu")
        .arg("--json")
        .arg(json_path);
    for arg in extra_args {
        cmd.arg(arg);
    }

    cmd.assert().success();
    let payload = fs::read_to_string(json_path)?;
    let parsed: Vec<CliDetectionRecord> = serde_json::from_str(&payload)?;
    Ok(parsed)
}

/// Limits against the OpenCV fixtures, matching `fcs-core/tests/parity.rs`.
///
/// The fixtures come from OpenCV's YuNet, which stretches a source to the model input; this
/// detector letterboxes it instead (experiment 96), so the answers differ by more than
/// rounding and coordinate equality is no longer the property to assert. The core parity
/// test carries the measurements these numbers come from. Note that the old form passed the
/// same `40.0` for the score as for pixels, so scores were not really being checked at all.
const MAX_SCORE_DELTA: f64 = 0.03;
const MIN_BOX_IOU: f64 = 0.70;
const MAX_LANDMARK_FRACTION: f64 = 0.15;

/// `[x, y, width, height]` overlap, as both sides store boxes.
fn box_iou(a: &[f64], b: &[f64]) -> f64 {
    let ix = (a[0] + a[2]).min(b[0] + b[2]) - a[0].max(b[0]);
    let iy = (a[1] + a[3]).min(b[1] + b[3]) - a[1].max(b[1]);
    let inter = ix.max(0.0) * iy.max(0.0);
    let union = a[2] * a[3] + b[2] * b[3] - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

fn assert_detections_close(
    actual: &[Detection],
    expected: &[Detection],
    extra_faces: usize,
    image: &str,
) {
    assert_eq!(
        actual.len(),
        expected.len() + extra_faces,
        "detection count mismatch for {image} (actual={}, fixture={} plus {extra_faces} known extra)",
        actual.len(),
        expected.len()
    );

    // Pair each fixture face with the actual face that overlaps it most, rather than by
    // score rank: the two pipelines' scores differ by up to 0.0175, which is enough to swap
    // the order of two similar faces and produce a confusing failure about the wrong pair.
    let mut taken = vec![false; actual.len()];
    for e in expected {
        let best = actual
            .iter()
            .enumerate()
            .filter(|(i, _)| !taken[*i])
            .map(|(i, a)| (i, box_iou(&a.bbox, &e.bbox)))
            .max_by(|x, y| x.1.total_cmp(&y.1));
        let Some((idx, iou)) = best else {
            panic!(
                "no detection left to pair with fixture face {:?} in {image}",
                e.bbox
            );
        };
        assert!(
            iou >= MIN_BOX_IOU,
            "box overlap {iou:.4} below {MIN_BOX_IOU} for {image}: {:?} against fixture {:?}",
            actual[idx].bbox,
            e.bbox
        );
        taken[idx] = true;
        let a = &actual[idx];
        let score_delta = (a.score - e.score).abs();
        assert!(
            score_delta <= MAX_SCORE_DELTA,
            "score {} against fixture {} for {image} (delta {score_delta}, limit {MAX_SCORE_DELTA})",
            a.score,
            e.score
        );

        let face = a.bbox[2].max(a.bbox[3]);
        for (landmark_idx, (al, el)) in a.landmarks.iter().zip(e.landmarks.iter()).enumerate() {
            let delta = (al[0] - el[0]).abs().max((al[1] - el[1]).abs());
            assert!(
                delta <= face * MAX_LANDMARK_FRACTION,
                "landmark {landmark_idx} moved {delta:.2} px on a {face:.0} px face in \
                 {image} ({:.1}%, limit {:.0}%)",
                100.0 * delta / face,
                100.0 * MAX_LANDMARK_FRACTION
            );
        }
    }
}

#[derive(Debug, Deserialize)]
struct CliDetectionRecord {
    image: String,
    detections: Vec<Detection>,
}

#[derive(Clone, Debug, Deserialize)]
struct Detection {
    score: f64,
    bbox: [f64; 4],
    landmarks: [[f64; 2]; 5],
}

#[derive(Debug, Deserialize)]
struct FixtureFile {
    #[serde(default)]
    score_threshold: Option<f64>,
    #[serde(default)]
    nms_threshold: Option<f64>,
    #[serde(default)]
    top_k: Option<usize>,
    detections: Vec<Detection>,
}

#[test]
fn cli_json_output_matches_snapshot() -> Result<(), Box<dyn Error>> {
    let Some(model) = ensure_model_path() else {
        return Ok(());
    };
    if !fixtures_available() {
        eprintln!("skipping: fixture images not available in this environment");
        return Ok(());
    }

    let fixture_image = fixture_path("images/006.jpg")?;

    let work_dir = tempdir()?;
    let json_path = work_dir.path().join("out.json");

    let mut cmd = cargo_bin_cmd!("fcs-cli");
    cmd.arg("--input")
        .arg(fixture_image)
        .arg("--model")
        .arg(&model)
        .arg("--no-gpu")
        .arg("--json")
        .arg(&json_path);

    cmd.assert().success();

    let raw = fs::read_to_string(&json_path)?;
    let sanitized = sanitize_cli_json(&raw)?;
    let expected = load_snapshot("cli_single_image.json")?;
    let sanitized_value: Value = serde_json::from_str(&sanitized)?;
    let expected_value: Value = serde_json::from_str(&expected)?;

    assert_json_close(
        &sanitized_value,
        &expected_value,
        SNAPSHOT_FLOAT_TOLERANCE,
        "$",
    );

    Ok(())
}

fn assert_json_close(actual: &Value, expected: &Value, tol: f64, path: &str) {
    match (actual, expected) {
        (Value::Number(actual), Value::Number(expected)) => {
            let actual = actual
                .as_f64()
                .unwrap_or_else(|| panic!("actual JSON number at {path} is not finite"));
            let expected = expected
                .as_f64()
                .unwrap_or_else(|| panic!("expected JSON number at {path} is not finite"));
            assert!(
                (actual - expected).abs() <= tol,
                "CLI JSON output changed at {path}: actual={actual}, expected={expected}, tolerance={tol}.\nUpdate tests/snapshots/cli_single_image.json if this is expected."
            );
        }
        (Value::Array(actual), Value::Array(expected)) => {
            assert_eq!(
                actual.len(),
                expected.len(),
                "CLI JSON output changed at {path}: array length differs.\nUpdate tests/snapshots/cli_single_image.json if this is expected."
            );
            for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
                let child_path = format!("{path}[{idx}]");
                assert_json_close(actual, expected, tol, &child_path);
            }
        }
        (Value::Object(actual), Value::Object(expected)) => {
            assert_eq!(
                actual.len(),
                expected.len(),
                "CLI JSON output changed at {path}: object keys differ (actual={:?}, expected={:?}).\nUpdate tests/snapshots/cli_single_image.json if this is expected.",
                actual.keys().collect::<Vec<_>>(),
                expected.keys().collect::<Vec<_>>()
            );
            for (key, expected_value) in expected {
                let actual_value = actual.get(key).unwrap_or_else(|| {
                    panic!("CLI JSON output changed at {path}: missing key {key:?}")
                });
                let child_path = format!("{path}.{key}");
                assert_json_close(actual_value, expected_value, tol, &child_path);
            }
        }
        _ => assert_eq!(
            actual, expected,
            "CLI JSON output changed at {path}.\nUpdate tests/snapshots/cli_single_image.json if this is expected."
        ),
    }
}

fn sanitize_cli_json(raw: &str) -> Result<String, Box<dyn Error>> {
    let mut value: Value = serde_json::from_str(raw)?;
    if let Some(entries) = value.as_array_mut() {
        for entry in entries {
            if let Some(obj) = entry.as_object_mut() {
                if let Some(image) = obj.get_mut("image")
                    && let Some(path_str) = image.as_str()
                    && let Some(file_name) =
                        Path::new(path_str).file_name().and_then(|n| n.to_str())
                {
                    *image = Value::String(file_name.to_string());
                }
                if let Some(annotated) = obj.get_mut("annotated")
                    && let Some(path_str) = annotated.as_str()
                    && let Some(file_name) =
                        Path::new(path_str).file_name().and_then(|n| n.to_str())
                {
                    *annotated = Value::String(file_name.to_string());
                }
            }
        }
    }
    Ok(serde_json::to_string_pretty(&value)?)
}

fn load_snapshot(name: &str) -> Result<String, Box<dyn Error>> {
    let snapshot_path = Path::new("tests").join("snapshots").join(name);
    if !snapshot_path.exists() {
        return Err(format!("snapshot file missing: {}", snapshot_path.display()).into());
    }
    let contents = fs::read_to_string(&snapshot_path)?;
    Ok(contents)
}
