//! Does the Rust SCRFD path agree with the Python one it was ported from?
//!
//! **This is the only test in the workspace that checks this project against something it did
//! not write.** Everything else — the three engines agreeing to 1e-05, the golden crop regions,
//! the CLI snapshot — compares the project against itself, which proves the parts are
//! consistent, not that they are right. `tract` used to fill this role by interpreting the ONNX
//! file, and it left with YuNet.
//!
//! The reference is `tools/dataset/scrfd_detect.py`: the same checkpoint
//! (`work_dirs/oi80k/epoch_100.pth`), run through torch on CUDA in WSL, decoded by the numpy
//! reference in `eval_eye_error.py`. This side runs the exported ONNX through `fcs_ort` and
//! `fcs_core::scrfd`. Agreement therefore covers the whole chain at once: the top-left
//! letterbox, RGB channel order, normalised padding, the export, and the decode — every one of
//! which was wrong at some point during the port, and none of which a self-comparison catches.
//!
//! It also pins the shipped model to that checkpoint. Exporting from a different epoch would
//! move every score far more than the tolerances below allow.
//!
//! The fixtures are committed, which the rest of `fixtures/` is not: they are eight Open Images
//! photographs, all CC BY 2.0, attributed in `fixtures/oracle/ATTRIBUTION.md`. See that file for
//! why these eight and how to regenerate the reference.
//!
//! ```text
//! ORT_DYLIB_PATH=.../onnxruntime.dll cargo test -p fcs-core --test python_parity
//! ```
//!
//! `FCS_PARITY_REFERENCE=<path/to/detections.json>` points it at a bigger corpus instead — the
//! 1,239-image reference folder, say — which is how this began life as an example.

use std::{collections::HashMap, path::PathBuf};

use fcs_core::{Detection, ScrfdDetector};

/// The threshold `scrfd_detect.py` was run at, so both sides must use it.
const THRESHOLD: f32 = 0.4;

/// Upstream's `test_cfg.nms.iou_threshold`, and what the numpy reference uses.
const NMS: f32 = 0.4;

/// Above this score, the two sides must agree that a face exists.
///
/// Not [`THRESHOLD`], and the gap matters. A detection scored right at the threshold is one
/// rounding difference away from falling below it, so requiring agreement there would make this
/// test fail on a different CPU rather than on a real defect — and the reference genuinely
/// contains faces at 0.424 and 0.472. Score drift between the two paths measures 0.0152 on the
/// development machine, so 0.1 of headroom is about six times the observed noise.
///
/// Below it, a face found by only one side is reported and not failed. That costs little: a
/// ported convention that is wrong does not nudge one borderline box, it moves every box and
/// every score at once, which the checks below would catch many times over.
const REQUIRED_SCORE: f32 = 0.5;

/// Thresholds for "the same detector, two runtimes" rather than "a convention ported wrong".
///
/// Absolute pixels are the wrong unit: the source is letterboxed into 640, so on an image
/// downscaled 6.9x every difference inside the model comes back multiplied by 6.9. Box error is
/// therefore judged as a fraction of the box's own width, and the tail is reported rather than
/// gated, because a low-confidence box big enough to fill a third of the frame is genuinely
/// unstable under any change to the input — while a wrong convention moves *every* box.
const MAX_SCORE_DRIFT: f32 = 0.02;
const MAX_MEDIAN_RELATIVE: f32 = 0.01;
const MAX_ANY_RELATIVE: f32 = 0.15;

/// A face in the reference: pixel corners plus the score it was found at.
struct Expected {
    corners: [f32; 4],
    score: f32,
}

fn iou(a: &Detection, b: &[f32; 4]) -> f32 {
    let (ax2, ay2) = (a.bbox.x + a.bbox.width, a.bbox.y + a.bbox.height);
    let ix = (ax2.min(b[2]) - a.bbox.x.max(b[0])).max(0.0);
    let iy = (ay2.min(b[3]) - a.bbox.y.max(b[1])).max(0.0);
    let inter = ix * iy;
    let union = a.bbox.width * a.bbox.height + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("fcs-core sits in the workspace root")
        .to_path_buf()
}

/// Resolve a reference record's image path.
///
/// The committed fixture stores repo-relative paths so it works on every platform and in CI. A
/// corpus JSON supplied through `FCS_PARITY_REFERENCE` was written under WSL and carries
/// `/mnt/c/...`, which has to be mapped to a Windows path or every image silently fails to load
/// and the comparison has nothing to compare — which is why zero comparisons is an error below.
fn resolve_image(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("/mnt/") {
        let (drive, tail) = rest.split_at(1);
        return PathBuf::from(format!("{}:{}", drive.to_uppercase(), tail));
    }
    let candidate = PathBuf::from(path);
    if candidate.is_absolute() {
        candidate
    } else {
        repo_root().join(candidate)
    }
}

/// Fail instead of skipping, for CI.
fn strict() -> bool {
    std::env::var("FCS_STRICT_TESTS").is_ok_and(|v| v != "0" && !v.is_empty())
}

#[test]
fn the_rust_path_agrees_with_the_python_reference() {
    let reference = std::env::var("FCS_PARITY_REFERENCE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| repo_root().join("fixtures/oracle/torch_epoch100.json"));
    if !reference.exists() {
        panic!("no reference at {}", reference.display());
    }

    // An explicit path, not `ScrfdDetector::load()`: that resolves relative to the working
    // directory, which for a test is the crate root rather than the workspace root.
    let model = repo_root().join("models/scrfd80k_500m_640.onnx");
    let Some(detector) = ScrfdDetector::load_from(&model) else {
        // The model and the runtime are both things CI provides, so their absence is a
        // misconfiguration rather than a fact about the machine.
        if strict() {
            panic!(
                "FCS_STRICT_TESTS: SCRFD did not load from {}; set ORT_DYLIB_PATH and fetch the model",
                model.display()
            );
        }
        eprintln!(
            "skipped: SCRFD did not load from {} (set ORT_DYLIB_PATH and fetch the model)",
            model.display()
        );
        return;
    };

    let document: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&reference).expect("read the reference"))
            .expect("the reference is JSON");
    let records = document.as_array().expect("expected a JSON array");

    let mut compared = 0usize;
    let mut matched = 0usize;
    let mut missing_required = Vec::new();
    let mut extra_required = Vec::new();
    let mut borderline = 0usize;
    let mut worst_score = 0.0f32;
    let mut worst_case = String::new();
    let mut worst_box = 0.0f32;
    let mut offenders: Vec<(String, f32, f32)> = Vec::new();

    for record in records.iter() {
        let path = record["image"]
            .as_str()
            .expect("record without an image path");
        let expected: Vec<Expected> = record["detections"]
            .as_array()
            .map(|dets| {
                dets.iter()
                    .filter_map(|d| {
                        let score = d["score"].as_f64()? as f32;
                        if score < THRESHOLD {
                            return None;
                        }
                        let b = d["bbox"].as_array()?;
                        let v: Vec<f32> =
                            b.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect();
                        Some(Expected {
                            corners: [v[0], v[1], v[0] + v[2], v[1] + v[3]],
                            score,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        let local = resolve_image(path);
        let image = match fcs_utils::load_image(&local) {
            Ok(image) => image,
            Err(err) => {
                // A committed fixture that will not load is a failure, not something to skip.
                panic!("failed to load {}: {err:#}", local.display());
            }
        };
        let found = detector
            .detect(&image, THRESHOLD, NMS)
            .expect("detection runs");
        compared += 1;
        let name = path.rsplit(['/', '\\']).next().unwrap_or(path).to_string();

        let mut taken: HashMap<usize, ()> = HashMap::new();
        for detection in &found {
            let best = expected
                .iter()
                .enumerate()
                .filter(|(i, _)| !taken.contains_key(i))
                .map(|(i, want)| (i, iou(detection, &want.corners)))
                .max_by(|a, b| a.1.total_cmp(&b.1));
            match best {
                Some((index, overlap)) if overlap >= 0.5 => {
                    taken.insert(index, ());
                    matched += 1;
                    let want = &expected[index];
                    let gap = (detection.bbox.x - want.corners[0])
                        .abs()
                        .max((detection.bbox.y - want.corners[1]).abs())
                        .max((detection.bbox.x + detection.bbox.width - want.corners[2]).abs())
                        .max((detection.bbox.y + detection.bbox.height - want.corners[3]).abs());
                    offenders.push((name.clone(), gap, gap / detection.bbox.width.max(1.0)));
                    if gap > worst_box {
                        worst_box = gap;
                        worst_case = format!(
                            "{name}: rust x{:.1} y{:.1} w{:.1} h{:.1} s{:.4} | python x{:.1} y{:.1} w{:.1} h{:.1} s{:.4}",
                            detection.bbox.x,
                            detection.bbox.y,
                            detection.bbox.width,
                            detection.bbox.height,
                            detection.score,
                            want.corners[0],
                            want.corners[1],
                            want.corners[2] - want.corners[0],
                            want.corners[3] - want.corners[1],
                            want.score
                        );
                    }
                    worst_score = worst_score.max((detection.score - want.score).abs());
                }
                // Found here and not there. Only a confident one is a defect.
                _ if detection.score >= REQUIRED_SCORE => {
                    extra_required.push(format!("{name} @ {:.3}", detection.score));
                }
                _ => borderline += 1,
            }
        }
        for (index, want) in expected.iter().enumerate() {
            if !taken.contains_key(&index) {
                if want.score >= REQUIRED_SCORE {
                    missing_required.push(format!("{name} @ {:.3}", want.score));
                } else {
                    borderline += 1;
                }
            }
        }
    }

    offenders.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!("compared {compared} images at threshold {THRESHOLD}");
    println!("  matched                  {matched}");
    println!("  borderline, not required {borderline}");
    println!("  worst box edge           {worst_box:.2} px  ({worst_case})");
    println!("  worst score drift        {worst_score:.4}");
    for (name, gap, relative) in offenders.iter().take(4) {
        println!(
            "    {gap:7.2} px  {:>5.1}% of box  {name}",
            relative * 100.0
        );
    }

    let mut relative: Vec<f32> = offenders.iter().map(|o| o.2).collect();
    relative.sort_by(f32::total_cmp);
    let median = relative.get(relative.len() / 2).copied().unwrap_or(0.0);
    let max = relative.last().copied().unwrap_or(0.0);
    println!(
        "  box error                median {:.2}%, max {:.2}% of box width",
        median * 100.0,
        max * 100.0
    );

    assert!(
        compared > 0,
        "no images were readable, so nothing was compared"
    );
    assert!(
        matched > 0,
        "nothing matched, so the tolerances below checked nothing"
    );
    assert!(
        missing_required.is_empty() && extra_required.is_empty(),
        "the two paths disagree about confident faces (>= {REQUIRED_SCORE}): \
         missing here {missing_required:?}, absent from the reference {extra_required:?}"
    );
    assert!(
        worst_score <= MAX_SCORE_DRIFT,
        "scores differ by {worst_score:.4}, more than {MAX_SCORE_DRIFT} -- \
         either the export drifted from the checkpoint or a convention is wrong"
    );
    assert!(
        median <= MAX_MEDIAN_RELATIVE,
        "the typical box is {:.2}% out, more than {:.2}% -- that is systematic, not rounding",
        median * 100.0,
        MAX_MEDIAN_RELATIVE * 100.0
    );
    assert!(
        max <= MAX_ANY_RELATIVE,
        "a box is {:.1}% out, past the {:.0}% tail allowance",
        max * 100.0,
        MAX_ANY_RELATIVE * 100.0
    );
}
