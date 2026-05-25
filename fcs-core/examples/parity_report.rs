//! Generate parity metrics against the OpenCV YuNet baseline fixtures.
//!
//! This example compares detections from `YuNetDetector` to the reference JSON fixtures captured
//! from the OpenCV implementation. It reports recall, precision, IoU overlap, and score deltas
//! aggregated across the dataset. Use this to monitor regressions when tweaking preprocessing or
//! postprocessing logic.

use std::{
    collections::HashMap,
    fmt,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context, Result};
use fcs_core::{
    BoundingBox, Detection, InputSize, PostprocessConfig, PreprocessConfig, YuNetDetector,
};
use fcs_utils::{fixture_path, fixtures_dir, load_fixture_json};
use serde::Deserialize;

const MODEL_PATH: &str = "models/face_detection_yunet_2023mar_640.onnx";
const IOU_THRESHOLD: f32 = 0.5;

fn main() -> Result<()> {
    let model_path = PathBuf::from(MODEL_PATH);
    if !model_path.exists() {
        anyhow::bail!(
            "model missing at {} - download YuNet and place it under models/",
            model_path.display()
        );
    }

    let mut report = DatasetReport::default();
    let mut detectors = HashMap::new();

    let fixtures = load_fixtures()?;
    if fixtures.is_empty() {
        anyhow::bail!("no OpenCV parity fixtures found under fixtures/opencv");
    }

    println!("YuNet ↔ OpenCV parity report (IoU threshold = {IOU_THRESHOLD})");
    println!(
        "{:<18} {:>4} {:>4} {:>4} {:>6} {:>6} {:>8} {:>9} {:>9}",
        "Image", "Exp", "Our", "Hit", "Recall", "Prec", "MeanIoU", "Δscore", "|Δscore|"
    );

    for fixture in fixtures {
        let key = DetectorKey::from(&fixture);
        let detector = get_detector(&mut detectors, &model_path, key)?;

        let detections = detector.detect_path(&fixture.image_path).with_context(|| {
            format!(
                "failed to run detection on {}",
                fixture.image_path.display()
            )
        })?;

        let analysis = analyse_case(&detections.detections, &fixture)?;
        println!("{}", analysis);
        report.add(&analysis);
    }

    println!("{}", report);
    Ok(())
}

#[derive(Debug)]
struct CaseAnalysis {
    image_name: String,
    expected: usize,
    ours: usize,
    matched: usize,
    recall: f32,
    precision: f32,
    mean_iou: f32,
    mean_score_delta: f32,
    mean_abs_score_delta: f32,
    worst_iou: f32,
    worst_abs_score_delta: f32,
}

impl fmt::Display for CaseAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:<18} {:>4} {:>4} {:>4} {:>6.3} {:>6.3} {:>8.3} {:>+9.4} {:>9.4}",
            self.image_name,
            self.expected,
            self.ours,
            self.matched,
            self.recall,
            self.precision,
            self.mean_iou,
            self.mean_score_delta,
            self.mean_abs_score_delta
        )
    }
}

#[derive(Default)]
struct DatasetReport {
    total_expected: usize,
    total_ours: usize,
    total_matched: usize,
    sum_recall: f32,
    sum_precision: f32,
    sum_iou: f32,
    sum_score_delta: f32,
    sum_abs_score_delta: f32,
    cases: usize,
    worst_iou: f32,
    worst_abs_score_delta: f32,
    recall_gaps: Vec<CaseGap>,
    precision_gaps: Vec<CaseGap>,
}

impl DatasetReport {
    fn add(&mut self, case: &CaseAnalysis) {
        self.total_expected += case.expected;
        self.total_ours += case.ours;
        self.total_matched += case.matched;
        self.sum_recall += case.recall;
        self.sum_precision += case.precision;
        self.sum_iou += case.mean_iou * case.matched as f32;
        self.sum_score_delta += case.mean_score_delta * case.matched as f32;
        self.sum_abs_score_delta += case.mean_abs_score_delta * case.matched as f32;
        self.worst_iou = if self.cases == 0 {
            case.worst_iou
        } else {
            self.worst_iou.min(case.worst_iou)
        };
        self.worst_abs_score_delta = self.worst_abs_score_delta.max(case.worst_abs_score_delta);
        self.cases += 1;

        if case.recall < 0.999 {
            self.recall_gaps.push(CaseGap {
                image: case.image_name.clone(),
                expected: case.expected,
                ours: case.ours,
                matched: case.matched,
                value: case.recall,
            });
        }
        if case.precision < 0.999 {
            self.precision_gaps.push(CaseGap {
                image: case.image_name.clone(),
                expected: case.expected,
                ours: case.ours,
                matched: case.matched,
                value: case.precision,
            });
        }
    }
}

impl fmt::Display for DatasetReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.cases == 0 {
            return write!(f, "No fixtures analysed.");
        }

        let avg_recall = self.sum_recall / self.cases as f32;
        let avg_precision = self.sum_precision / self.cases as f32;
        let matched = self.total_matched.max(1);
        let mean_iou = self.sum_iou / matched as f32;
        let mean_score_delta = self.sum_score_delta / matched as f32;
        let mean_abs_delta = self.sum_abs_score_delta / matched as f32;

        writeln!(
            f,
            "---------------------------------------------------------------------"
        )?;
        writeln!(
            f,
            "Dataset summary: expected {} | ours {} | matched {}",
            self.total_expected, self.total_ours, self.total_matched
        )?;
        writeln!(
            f,
            "Average recall {:.3} | precision {:.3} | mean IoU {:.3}",
            avg_recall, avg_precision, mean_iou
        )?;
        writeln!(
            f,
            "Mean score delta {:+.4} | mean |delta| {:.4} | worst IoU {:.3} | worst |delta| {:.4}",
            mean_score_delta, mean_abs_delta, self.worst_iou, self.worst_abs_score_delta
        )?;

        if !self.recall_gaps.is_empty() {
            writeln!(
                f,
                "Recall gaps ({} cases with missed detections):",
                self.recall_gaps.len()
            )?;
            for gap in self.recall_gaps.iter().take(5) {
                writeln!(
                    f,
                    "  {:<18} recall {:.3} (expected {}, ours {}, matched {})",
                    gap.image, gap.value, gap.expected, gap.ours, gap.matched
                )?;
            }
            if self.recall_gaps.len() > 5 {
                writeln!(
                    f,
                    "  … and {} more cases below 100% recall",
                    self.recall_gaps.len() - 5
                )?;
            }
        }

        if !self.precision_gaps.is_empty() {
            writeln!(
                f,
                "Precision gaps ({} cases with extra detections):",
                self.precision_gaps.len()
            )?;
            for gap in self.precision_gaps.iter().take(5) {
                writeln!(
                    f,
                    "  {:<18} precision {:.3} (expected {}, ours {}, matched {})",
                    gap.image, gap.value, gap.expected, gap.ours, gap.matched
                )?;
            }
            if self.precision_gaps.len() > 5 {
                writeln!(
                    f,
                    "  … and {} more cases below 100% precision",
                    self.precision_gaps.len() - 5
                )?;
            }
        }

        Ok(())
    }
}

#[derive(Debug)]
struct CaseGap {
    image: String,
    expected: usize,
    ours: usize,
    matched: usize,
    value: f32,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
struct FixtureDetection {
    score: f32,
    bbox: [f32; 4],
    landmarks: [[f32; 2]; 5],
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
struct FixtureFile {
    #[serde(default)]
    image: Option<String>,
    #[serde(default)]
    input_size: Option<[u32; 2]>,
    #[serde(default)]
    score_threshold: Option<f32>,
    #[serde(default)]
    nms_threshold: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    detections: Vec<FixtureDetection>,
}

#[derive(Debug)]
struct FixtureCase {
    image_name: String,
    image_path: PathBuf,
    fixture: FixtureFile,
}

fn load_fixtures() -> Result<Vec<FixtureCase>> {
    let root = fixtures_dir()?;
    let opencv_dir = root.join("opencv");
    anyhow::ensure!(
        opencv_dir.is_dir(),
        "expected fixtures/opencv directory under {}",
        root.display()
    );

    let mut cases = Vec::new();
    for entry in std::fs::read_dir(&opencv_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let fixture: FixtureFile = load_fixture_json(path.strip_prefix(&root)?)?;
        let image_ref = match &fixture.image {
            Some(img) => img.clone(),
            None => continue,
        };
        let image_path = resolve_image_path(&image_ref)?;
        let image_name = Path::new(&image_ref)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        cases.push(FixtureCase {
            image_name,
            image_path,
            fixture,
        });
    }

    cases.sort_by(|a, b| a.image_name.cmp(&b.image_name));
    Ok(cases)
}

fn resolve_image_path(image_ref: &str) -> Result<PathBuf> {
    let filename = Path::new(image_ref)
        .file_name()
        .with_context(|| format!("invalid image filename in fixture: {image_ref}"))?;
    let rel = Path::new("images").join(filename);
    fixture_path(rel)
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct DetectorKey {
    input_w: u32,
    input_h: u32,
    score_threshold: u32,
    nms_threshold: u32,
    top_k: usize,
}

impl DetectorKey {
    fn from(case: &FixtureCase) -> Self {
        let score = case
            .fixture
            .score_threshold
            .unwrap_or(PostprocessConfig::default().score_threshold);
        let nms = case
            .fixture
            .nms_threshold
            .unwrap_or(PostprocessConfig::default().nms_threshold);
        let top_k = case
            .fixture
            .top_k
            .unwrap_or(PostprocessConfig::default().top_k);
        let [input_w, input_h] = case.fixture.input_size.unwrap_or([640, 640]);
        DetectorKey {
            input_w,
            input_h,
            score_threshold: score.to_bits(),
            nms_threshold: nms.to_bits(),
            top_k,
        }
    }
}

fn get_detector(
    cache: &mut HashMap<DetectorKey, Arc<YuNetDetector>>,
    model_path: &Path,
    key: DetectorKey,
) -> Result<Arc<YuNetDetector>> {
    if let Some(det) = cache.get(&key) {
        return Ok(det.clone());
    }

    let preprocess = PreprocessConfig {
        input_size: InputSize::new(key.input_w, key.input_h),
        ..Default::default()
    };
    let postprocess = PostprocessConfig {
        score_threshold: f32::from_bits(key.score_threshold),
        nms_threshold: f32::from_bits(key.nms_threshold),
        top_k: key.top_k,
    };
    let detector = Arc::new(YuNetDetector::new(model_path, preprocess, postprocess)?);
    cache.insert(key, detector.clone());
    Ok(detector)
}

fn analyse_case(detections: &[Detection], case: &FixtureCase) -> Result<CaseAnalysis> {
    let matches = match_detections(detections, &case.fixture.detections, IOU_THRESHOLD);

    let matched = matches.records.len();
    let expected = case.fixture.detections.len();
    let ours = detections.len();

    let recall = if expected == 0 {
        if ours == 0 { 1.0 } else { 0.0 }
    } else {
        matched as f32 / expected as f32
    };
    let precision = if ours == 0 {
        if expected == 0 { 1.0 } else { 0.0 }
    } else {
        matched as f32 / ours as f32
    };

    let (mut iou_sum, mut score_sum, mut abs_score_sum) = (0.0f32, 0.0f32, 0.0f32);
    let mut worst_iou = 1.0f32;
    let mut worst_abs_score_delta = 0.0f32;

    for record in &matches.records {
        iou_sum += record.iou;
        score_sum += record.score_delta;
        abs_score_sum += record.abs_score_delta;
        worst_iou = worst_iou.min(record.iou);
        worst_abs_score_delta = worst_abs_score_delta.max(record.abs_score_delta);
    }

    if matched == 0 {
        worst_iou = 1.0;
    }

    let mean_iou = if matched == 0 {
        0.0
    } else {
        iou_sum / matched as f32
    };
    let mean_score_delta = if matched == 0 {
        0.0
    } else {
        score_sum / matched as f32
    };
    let mean_abs_score_delta = if matched == 0 {
        0.0
    } else {
        abs_score_sum / matched as f32
    };

    Ok(CaseAnalysis {
        image_name: case.image_name.clone(),
        expected,
        ours,
        matched,
        recall,
        precision,
        mean_iou,
        mean_score_delta,
        mean_abs_score_delta,
        worst_iou,
        worst_abs_score_delta,
    })
}

struct MatchResults {
    records: Vec<MatchRecord>,
}

struct MatchRecord {
    iou: f32,
    score_delta: f32,
    abs_score_delta: f32,
}

fn match_detections(
    ours: &[Detection],
    expected: &[FixtureDetection],
    threshold: f32,
) -> MatchResults {
    let mut records = Vec::new();
    let mut ours_used = vec![false; ours.len()];

    for exp in expected {
        let mut best = None;
        let mut best_iou = 0.0;
        for (idx, det) in ours.iter().enumerate() {
            if ours_used[idx] {
                continue;
            }
            let iou = bbox_iou(
                &det.bbox,
                exp.bbox[0],
                exp.bbox[1],
                exp.bbox[2],
                exp.bbox[3],
            );
            if iou >= threshold && iou > best_iou {
                best = Some((idx, det));
                best_iou = iou;
            }
        }

        if let Some((idx, det)) = best {
            ours_used[idx] = true;
            let score_delta = det.score - exp.score;
            records.push(MatchRecord {
                iou: best_iou,
                score_delta,
                abs_score_delta: score_delta.abs(),
            });
        }
    }

    MatchResults { records }
}

fn bbox_iou(bbox: &BoundingBox, x: f32, y: f32, w: f32, h: f32) -> f32 {
    bbox.iou(&BoundingBox {
        x,
        y,
        width: w,
        height: h,
    })
}
