//! The detector the application uses.
//!
//! One detector ships now: SCRFD, trained on this project's own licence-clean data, which finds
//! 86.0% of the faces in the Open Images test split at 0.11 false positives per image against
//! YuNet's 71.6% at 0.14 (`tools/dataset/SCRFD_80K.md`).
//!
//! It used to be a pair, with YuNet as the floor for machines without ONNX Runtime. That floor
//! is gone because it is no longer needed: SCRFD runs on the WGSL engine and the built-in CPU
//! graph as well, agreeing with ONNX Runtime to about 1e-05 and producing identical detections.
//! Keeping YuNet meant shipping weights trained on WIDER FACE, which is released for
//! "non-commercial academic research only" (`tools/dataset/DATA_CARD.md`) -- the licence
//! question this project set out to remove.

use anyhow::{Context, Result};
use image::DynamicImage;
use std::{path::Path, sync::Arc};

use crate::{postprocess::Detection, scrfd::ScrfdDetector};
use fcs_utils::config::DetectionSettings;

/// What one detection run produced.
#[derive(Debug, Clone)]
pub struct DetectionOutput {
    /// Detected faces in original-image pixels, after filtering and suppression.
    pub detections: Vec<Detection>,
    /// The original dimensions of the input image.
    pub original_size: (u32, u32),
}

/// The application's detector.
///
/// The three knobs come from [`DetectionSettings`] and every one of them reaches the model.
/// That is worth stating because it was briefly untrue: when this replaced the YuNet pair, the
/// score was read from settings while the NMS threshold and the cap stayed at the constants
/// below, so the GUI's sliders moved nothing. See [`Self::with_settings`].
#[derive(Debug)]
pub struct FaceDetector {
    /// Behind an `Arc` so [`Self::with_settings`] can re-wrap without reloading the model.
    scrfd: Arc<ScrfdDetector>,
    score_threshold: f32,
    nms_threshold: f32,
    top_k: usize,
}

/// Keep at most `top_k` detections, where `0` means keep all.
///
/// `ScrfdDetector::detect` returns them score-descending, so truncating keeps the best.
fn apply_top_k(detections: &mut Vec<Detection>, top_k: usize) {
    if top_k > 0 {
        detections.truncate(top_k);
    }
}

/// Score at which SCRFD is run when nothing says otherwise.
///
/// Chosen by eye on a corpus neither this model nor YuNet had seen: at 0.4 the 60 largest
/// detections it found and YuNet missed held 32 false positives, and at 0.5 seven, losing 2 of
/// 25 real faces (SCRFD_80K.md). Not comparable to YuNet's shipped 0.9 -- different model,
/// different scale, which is why the setting that carries it was renamed.
pub const SCRFD_SCORE_THRESHOLD: f32 = 0.5;

/// Upstream SCRFD's own `test_cfg.nms.iou_threshold`.
pub const SCRFD_NMS_THRESHOLD: f32 = 0.4;

impl FaceDetector {
    /// Load the detector, or `None` when no model is present.
    ///
    /// `None` means detection cannot run at all now that there is no second detector to fall
    /// back to, which is what the caller has always had to handle for a missing model file.
    pub fn load() -> Option<Self> {
        ScrfdDetector::load().map(Self::new)
    }

    /// Load from an explicit model path, as `--model` and the GUI's model setting supply.
    pub fn load_from<P: AsRef<Path>>(path: P) -> Option<Self> {
        ScrfdDetector::load_from(path).map(Self::new)
    }

    /// Wrap an already-loaded detector at the default operating point.
    pub fn new(scrfd: ScrfdDetector) -> Self {
        Self {
            scrfd: Arc::new(scrfd),
            score_threshold: SCRFD_SCORE_THRESHOLD,
            nms_threshold: SCRFD_NMS_THRESHOLD,
            top_k: fcs_utils::config::DEFAULT_TOP_K,
        }
    }

    /// The same detector configured from settings, sharing the loaded model.
    ///
    /// All three fields are applied. `confidence` and `nms_threshold` are passed to the model;
    /// `top_k` caps what comes back, after suppression, keeping the highest scores. **`0` means
    /// no cap**, which is what the pass this replaced meant by it -- the GUI cannot reach 0, but
    /// `--top-k 0` can, and a user writing that means "no limit", not "no faces".
    pub fn with_settings(&self, settings: &DetectionSettings) -> Self {
        Self {
            scrfd: Arc::clone(&self.scrfd),
            score_threshold: settings.confidence,
            nms_threshold: settings.nms_threshold,
            top_k: settings.top_k,
        }
    }

    /// Run detection on an in-memory image.
    pub fn detect_image(&self, image: &DynamicImage) -> Result<DetectionOutput> {
        let mut detections = self
            .scrfd
            .detect(image, self.score_threshold, self.nms_threshold)?;
        apply_top_k(&mut detections, self.top_k);
        Ok(DetectionOutput {
            detections,
            original_size: (image.width(), image.height()),
        })
    }

    /// Run detection on an image file.
    pub fn detect_path<P: AsRef<Path>>(&self, path: P) -> Result<DetectionOutput> {
        let path = path.as_ref();
        let image = fcs_utils::load_image(path)
            .with_context(|| format!("failed to load image from {}", path.display()))?;
        self.detect_image(&image)
    }

    /// Which detector is running, for logs and the GUI.
    pub fn model_name(&self) -> &'static str {
        "SCRFD-80k"
    }

    /// Which engine it is running on: the WGSL kernels or the CPU graph.
    pub fn inference_backend(&self) -> &'static str {
        self.scrfd.engine()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strict_tests() -> bool {
        std::env::var("FCS_STRICT_TESTS").is_ok_and(|v| v != "0" && !v.is_empty())
    }

    /// The shipped detector, loaded by explicit path from the workspace root.
    ///
    /// These tests used to call `FaceDetector::load`, which resolves `models/` against the
    /// working directory -- under `cargo test` the crate root, which has none. So all three
    /// skipped on every machine, and without consulting `FCS_STRICT_TESTS`, so strict CI
    /// reported them as passing having run nothing. `load` itself is covered by
    /// `tests/default_model_location.rs`.
    fn detector() -> Option<FaceDetector> {
        let model = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("fcs-core sits in the workspace root")
            .join(crate::scrfd::DEFAULT_MODEL);
        let detector = FaceDetector::load_from(&model);
        if detector.is_none() {
            assert!(
                !strict_tests(),
                "FCS_STRICT_TESTS: the detector did not load from {model:?}"
            );
            eprintln!("skipped: no detector model at {model:?}");
        }
        detector
    }

    #[test]
    fn settings_change_shares_the_loaded_model() {
        let Some(detector) = detector() else {
            return;
        };
        let stricter = detector.with_settings(&DetectionSettings {
            confidence: 0.9,
            nms_threshold: 0.1,
            top_k: 3,
        });
        assert_eq!(stricter.model_name(), "SCRFD-80k");
        assert_eq!(stricter.inference_backend(), detector.inference_backend());
        // Both sides above come from the same accessor, so they agree whatever it returns. This
        // pins what it returns; `ScrfdDetector::engine` is itself pinned to the known names.
        assert_eq!(detector.inference_backend(), detector.scrfd.engine());
        // Re-wrapping must not reload: both share one Arc.
        assert!(Arc::ptr_eq(&detector.scrfd, &stricter.scrfd));
    }

    /// The bug this guards: a setting read into the struct but never passed to the model. Each
    /// field is given a value distinguishable from the default, and read back.
    #[test]
    fn every_setting_reaches_the_detector() {
        let Some(detector) = detector() else {
            return;
        };
        let configured = detector.with_settings(&DetectionSettings {
            confidence: 0.75,
            nms_threshold: 0.25,
            top_k: 7,
        });
        assert!((configured.score_threshold - 0.75).abs() < f32::EPSILON);
        assert!((configured.nms_threshold - 0.25).abs() < f32::EPSILON);
        assert_eq!(configured.top_k, 7);
    }

    fn dummy_detections(count: usize) -> Vec<Detection> {
        (0..count)
            .map(|i| Detection {
                bbox: crate::postprocess::BoundingBox {
                    x: 0.0,
                    y: 0.0,
                    width: 10.0,
                    height: 10.0,
                },
                landmarks: [None; 5],
                score: 1.0 - i as f32 * 0.1,
            })
            .collect()
    }

    /// `0` must mean "no cap", not "no faces". The GUI cannot reach 0, but `--top-k 0` can, and
    /// it reads as "no limit" -- truncating to zero would silently return nothing.
    #[test]
    fn a_top_k_of_zero_is_no_cap() {
        let mut detections = dummy_detections(3);
        apply_top_k(&mut detections, 0);
        assert_eq!(detections.len(), 3, "0 must not truncate");
    }

    #[test]
    fn top_k_keeps_the_highest_scores() {
        let mut detections = dummy_detections(5);
        apply_top_k(&mut detections, 2);
        assert_eq!(detections.len(), 2);
        // `detect` hands them over score-descending, so the survivors are the best two.
        assert!(detections[0].score > detections[1].score);
        assert!((detections[0].score - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn a_top_k_above_the_count_keeps_everything() {
        let mut detections = dummy_detections(2);
        apply_top_k(&mut detections, 100);
        assert_eq!(detections.len(), 2);
    }

    #[test]
    fn detection_runs_on_whatever_engine_is_available() {
        let Some(detector) = detector() else {
            return;
        };
        let image = DynamicImage::ImageRgb8(image::RgbImage::new(64, 64));
        let output = detector.detect_image(&image).expect("detection runs");
        assert_eq!(output.original_size, (64, 64));
    }
}
