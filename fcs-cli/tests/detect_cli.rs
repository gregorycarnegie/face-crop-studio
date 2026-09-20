use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
};

use assert_cmd::cargo::cargo_bin_cmd;
use fcs_utils::{fixture_path, normalize_path};
use image::{ImageBuffer, Rgb};
use serde::Deserialize;
use serde_json::Value;
use tempfile::tempdir;

const MODEL_REL_PATH: &str = "../models/scrfd80k_500m_640.onnx";
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

    Ok(())
}

// `cli_detections_match_opencv_parity_samples` lived here. It compared this CLI against
// OpenCV's YuNet output on nine fixtures, at YuNet's 0.9 threshold, with a pinned count of the
// extra faces letterboxing recovered. The CLI does not run YuNet any more, and the comparison
// cannot be carried over: the scores are on a different scale, and the current detector both
// finds faces YuNet misses and misses some it found (`tools/dataset/SCRFD_80K.md`), so neither
// equality nor "never worse" would be true.
//
// That is a real loss of coverage -- it was the only check against an implementation nobody
// here wrote. What replaces it is the parity between this project's own three engines
// (`fcs-core/tests/scrfd_parity.rs`) and the snapshot below, both of which only compare this
// project against itself. The `fixtures/opencv/` goldens went with the test: 276 files of
// YuNet's opinions at its own threshold, which this detector disagrees with by design.
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

/// One image's entry in the CLI's JSON, as much of it as the remaining tests read.
#[derive(Debug, Deserialize)]
struct CliDetectionRecord {
    image: String,
    detections: Vec<serde_json::Value>,
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
