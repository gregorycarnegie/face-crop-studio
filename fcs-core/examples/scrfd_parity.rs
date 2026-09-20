//! Does the Rust SCRFD path agree with the Python one it was ported from?
//!
//! ```text
//! ORT_DYLIB_PATH=.../onnxruntime.dll cargo run -p fcs-core --example scrfd_parity -- \
//!     C:/Users/grego/Downloads/face-data/vinaskyy_scrfd80k_0.4.json [images]
//! ```
//!
//! The JSON is `scrfd_detect.py`'s output: the same checkpoint, run through torch on the GPU,
//! decoded by the numpy reference in `eval_eye_error.py`. This runs the exported ONNX through
//! `fcs_ort` and `fcs_core::scrfd` instead, so agreement covers the whole chain at once --
//! preprocessing (top-left letterbox, RGB, normalised padding), the export, and the decode.
//!
//! Exact equality is not the claim: one side is torch on CUDA, the other ONNX Runtime on CPU,
//! and the resize filters differ. A box out by a pixel is that. A box out by tens of pixels, or
//! a face found by one side only, is a ported convention that is wrong -- which is what the
//! layout traps during the export showed this code is prone to.

use std::{collections::HashMap, path::PathBuf};

use fcs_core::{Detection, ScrfdDetector};

/// The threshold `scrfd_detect.py` was run at.
const THRESHOLD: f32 = 0.4;
/// Upstream's `test_cfg.nms.iou_threshold`, and what the numpy reference uses.
const NMS: f32 = 0.4;
/// Thresholds for "the same detector, two runtimes" rather than "a convention ported wrong".
///
/// Absolute pixels are the wrong unit: the source is letterboxed into 640, so on an image
/// downscaled 6.9x every difference inside the model comes back multiplied by 6.9. Box error is
/// therefore judged as a fraction of the box's own width, and the tail is reported rather than
/// gated, because a low-confidence box big enough to fill a third of the frame is genuinely
/// unstable under any change to the input -- while a wrong convention moves *every* box.
const MAX_SCORE_DRIFT: f32 = 0.02;
const MAX_MEDIAN_RELATIVE: f32 = 0.01;
const MAX_ANY_RELATIVE: f32 = 0.15;

fn iou(a: &Detection, b: &[f32; 4]) -> f32 {
    let (ax2, ay2) = (a.bbox.x + a.bbox.width, a.bbox.y + a.bbox.height);
    let ix = (ax2.min(b[2]) - a.bbox.x.max(b[0])).max(0.0);
    let iy = (ay2.min(b[3]) - a.bbox.y.max(b[1])).max(0.0);
    let inter = ix * iy;
    let union = a.bbox.width * a.bbox.height + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let reference = PathBuf::from(
        args.next()
            .ok_or("usage: scrfd_parity <detections.json> [n]")?,
    );
    let limit: usize = args.next().map(|v| v.parse()).transpose()?.unwrap_or(40);

    let Some(detector) = ScrfdDetector::load() else {
        return Err("SCRFD did not load: set ORT_DYLIB_PATH and put the model in models/".into());
    };

    let document: serde_json::Value = serde_json::from_slice(&std::fs::read(&reference)?)?;
    let records = document.as_array().ok_or("expected a JSON array")?;

    let mut compared = 0usize;
    let mut matched = 0usize;
    let mut python_only = 0usize;
    let mut rust_only = 0usize;
    let mut worst_box = 0.0f32;
    let mut worst_score = 0.0f32;
    let mut worst_case = String::new();
    let mut offenders: Vec<(String, f32, f32, f32)> = Vec::new();

    for record in records.iter().take(limit) {
        let path = record["image"]
            .as_str()
            .ok_or("record without an image path")?;
        let expected: Vec<[f32; 4]> = record["detections"]
            .as_array()
            .map(|dets| {
                dets.iter()
                    .filter(|d| d["score"].as_f64().unwrap_or(0.0) as f32 >= THRESHOLD)
                    .map(|d| {
                        let b = d["bbox"].as_array().unwrap();
                        let v: Vec<f32> = b.iter().map(|x| x.as_f64().unwrap() as f32).collect();
                        [v[0], v[1], v[0] + v[2], v[1] + v[3]]
                    })
                    .collect()
            })
            .unwrap_or_default();
        let scores: Vec<f32> = record["detections"]
            .as_array()
            .map(|dets| {
                dets.iter()
                    .map(|d| d["score"].as_f64().unwrap_or(0.0) as f32)
                    .filter(|s| *s >= THRESHOLD)
                    .collect()
            })
            .unwrap_or_default();

        // The reference JSON is written under WSL, so its paths are `/mnt/c/...`; this runs on
        // Windows. Without the mapping every image fails to load and the comparison silently
        // has nothing to compare, which is why zero comparisons is an error below.
        let local = if let Some(rest) = path.strip_prefix("/mnt/") {
            let (drive, tail) = rest.split_at(1);
            format!("{}:{}", drive.to_uppercase(), tail)
        } else {
            path.to_string()
        };
        let image = match fcs_utils::load_image(&local) {
            Ok(image) => image,
            Err(_) => continue,
        };
        let found = detector.detect(&image, THRESHOLD, NMS)?;
        compared += 1;

        let mut taken: HashMap<usize, ()> = HashMap::new();
        for detection in &found {
            let best = expected
                .iter()
                .enumerate()
                .filter(|(i, _)| !taken.contains_key(i))
                .map(|(i, want)| (i, iou(detection, want)))
                .max_by(|a, b| a.1.total_cmp(&b.1));
            match best {
                Some((index, overlap)) if overlap >= 0.5 => {
                    taken.insert(index, ());
                    matched += 1;
                    let want = expected[index];
                    let gap = (detection.bbox.x - want[0])
                        .abs()
                        .max((detection.bbox.y - want[1]).abs())
                        .max((detection.bbox.x + detection.bbox.width - want[2]).abs())
                        .max((detection.bbox.y + detection.bbox.height - want[3]).abs());
                    let name = path.rsplit(['/', '\\']).next().unwrap_or(path).to_string();
                    let shrink = image.height() as f32 / 640.0;
                    offenders.push((
                        name.clone(),
                        gap,
                        gap / detection.bbox.width.max(1.0),
                        shrink,
                    ));
                    if gap > worst_box {
                        worst_box = gap;
                        worst_case = format!(
                            "{name}\n    rust   x{:.1} y{:.1} w{:.1} h{:.1} score {:.4}\
                             \n    python x{:.1} y{:.1} w{:.1} h{:.1} score {:.4}",
                            detection.bbox.x,
                            detection.bbox.y,
                            detection.bbox.width,
                            detection.bbox.height,
                            detection.score,
                            want[0],
                            want[1],
                            want[2] - want[0],
                            want[3] - want[1],
                            scores[index]
                        );
                    }
                    worst_score = worst_score.max((detection.score - scores[index]).abs());
                }
                _ => rust_only += 1,
            }
        }
        python_only += expected.len() - taken.len();
    }

    offenders.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!("worst disagreements, with how far the source was downscaled:");
    for (name, gap, relative, shrink) in offenders.iter().take(6) {
        println!(
            "  {gap:7.2} px  {:>5.1}% of box  shrunk {shrink:.1}x  {name}",
            relative * 100.0
        );
    }
    println!("compared {compared} images at threshold {THRESHOLD}");
    println!("  matched         {matched}");
    println!("  python only     {python_only}");
    println!("  rust only       {rust_only}");
    println!("  worst box edge  {worst_box:.2} px ({worst_case})");
    println!("  worst score     {worst_score:.4}");

    let mut relative: Vec<f32> = offenders.iter().map(|o| o.2).collect();
    relative.sort_by(f32::total_cmp);
    let median = relative.get(relative.len() / 2).copied().unwrap_or(0.0);
    let p90 = relative
        .get(relative.len() * 9 / 10)
        .copied()
        .unwrap_or(0.0);
    let max = relative.last().copied().unwrap_or(0.0);
    println!(
        "  box error       median {:.2}%, 90th {:.2}%, max {:.2}% of box width",
        median * 100.0,
        p90 * 100.0,
        max * 100.0
    );

    if compared == 0 {
        return Err("no images were readable, so nothing was compared".into());
    }
    if python_only > 0 || rust_only > 0 {
        return Err(format!(
            "the two paths disagree on which faces exist: {python_only} only in Python, \
             {rust_only} only in Rust"
        )
        .into());
    }
    if worst_score > MAX_SCORE_DRIFT {
        return Err(
            format!("scores differ by {worst_score:.4}, more than {MAX_SCORE_DRIFT}").into(),
        );
    }
    if median > MAX_MEDIAN_RELATIVE {
        return Err(format!(
            "the typical box is {:.2}% out, more than {:.2}% -- that is systematic",
            median * 100.0,
            MAX_MEDIAN_RELATIVE * 100.0
        )
        .into());
    }
    if max > MAX_ANY_RELATIVE {
        return Err(format!(
            "a box is {:.1}% out, past the {:.0}% tail allowance",
            max * 100.0,
            MAX_ANY_RELATIVE * 100.0
        )
        .into());
    }
    println!("PARITY OK");
    Ok(())
}
