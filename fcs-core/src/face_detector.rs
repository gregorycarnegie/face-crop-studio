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

use crate::{detector::DetectionOutput, scrfd::ScrfdDetector};

/// Score at which SCRFD is run.
///
/// Not the configured YuNet threshold, which is on a different scale entirely -- YuNet ships at
/// 0.9, and the two models' scores are not comparable. 0.5 was chosen by eye on a corpus neither
/// model had seen: at 0.4 the 60 largest detections YuNet missed held 32 false positives, and at
/// 0.5 seven, losing 2 of 25 real faces (SCRFD_80K.md). The configured threshold still governs
/// YuNet whenever YuNet is the one running.
pub const SCRFD_SCORE_THRESHOLD: f32 = 0.5;

/// Upstream SCRFD's own `test_cfg.nms.iou_threshold`.
pub const SCRFD_NMS_THRESHOLD: f32 = 0.4;

/// The application's detector.
#[derive(Debug)]
pub struct FaceDetector {
    /// Behind an `Arc` so [`Self::with_postprocess`] can re-wrap without reloading the model.
    scrfd: Arc<ScrfdDetector>,
    score_threshold: f32,
    nms_threshold: f32,
}

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

    /// Wrap an already-loaded detector.
    pub fn new(scrfd: ScrfdDetector) -> Self {
        Self {
            scrfd: Arc::new(scrfd),
            score_threshold: SCRFD_SCORE_THRESHOLD,
            nms_threshold: SCRFD_NMS_THRESHOLD,
        }
    }

    /// Run detection on an in-memory image.
    pub fn detect_image(&self, image: &DynamicImage) -> Result<DetectionOutput> {
        let detections = self
            .scrfd
            .detect(image, self.score_threshold, self.nms_threshold)?;
        Ok(DetectionOutput {
            detections,
            // The decoding has already undone the letterbox, so these are the identity: a
            // caller that rescaled by them would move every box a second time.
            scale_x: 1.0,
            scale_y: 1.0,
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

    /// The same detector at a different score threshold, sharing the loaded model.
    ///
    /// Takes the threshold directly rather than a `PostprocessConfig`: that config's fields --
    /// score, NMS, top-k -- were YuNet's, on YuNet's scale, and this model's scores are not
    /// comparable to them.
    pub fn with_score_threshold(&self, score_threshold: f32) -> Self {
        Self {
            scrfd: Arc::clone(&self.scrfd),
            score_threshold,
            nms_threshold: self.nms_threshold,
        }
    }

    /// Which detector is running, for logs and the GUI.
    pub fn model_name(&self) -> &'static str {
        "SCRFD-80k"
    }

    /// Which engine it is running on: ONNX Runtime, the WGSL kernels, or the CPU graph.
    pub fn inference_backend(&self) -> &'static str {
        self.scrfd.engine()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_threshold_change_shares_the_loaded_model() {
        let Some(detector) = FaceDetector::load() else {
            eprintln!("skipped: no detector model present");
            return;
        };
        let stricter = detector.with_score_threshold(0.9);
        assert_eq!(stricter.model_name(), "SCRFD-80k");
        assert_eq!(stricter.inference_backend(), detector.inference_backend());
        // Re-wrapping must not reload: both share one Arc.
        assert!(Arc::ptr_eq(&detector.scrfd, &stricter.scrfd));
    }

    #[test]
    fn detection_runs_on_whatever_engine_is_available() {
        let Some(detector) = FaceDetector::load() else {
            eprintln!("skipped: no detector model present");
            return;
        };
        let image = DynamicImage::ImageRgb8(image::RgbImage::new(64, 64));
        let output = detector.detect_image(&image).expect("detection runs");
        assert_eq!(output.original_size, (64, 64));
    }
}
