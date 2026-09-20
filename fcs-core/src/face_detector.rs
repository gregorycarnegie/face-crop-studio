//! Picks the best detector available and hides the choice from every caller.
//!
//! Two detectors ship. SCRFD, trained on this project's own licence-clean data, finds 86.0% of
//! the faces in the Open Images test split at 0.11 false positives per image against YuNet's
//! 71.6% at 0.14 (`tools/dataset/SCRFD_80K.md`), but runs only under ONNX Runtime. YuNet is
//! compiled into the built-in CPU graph and the WGSL kernels, so it runs anywhere.
//!
//! So SCRFD is preferred and YuNet is the floor: a machine without a runtime gets a slower,
//! older detector rather than none. Callers construct one of these and call [`Self::detect_image`]
//! exactly as they called `YuNetDetector`'s.

use anyhow::{Context, Result};
use image::DynamicImage;
use std::{path::Path, sync::Arc};

use crate::{
    detector::{DetectionOutput, YuNetDetector},
    postprocess::PostprocessConfig,
    scrfd::ScrfdDetector,
};

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

/// A detector that is SCRFD where possible and YuNet otherwise.
#[derive(Debug)]
pub struct FaceDetector {
    /// `None` when SCRFD loaded, because then nothing would ever call it. Building it anyway
    /// costs what experiment 81 measured for the GPU path -- five compiled WGSL pipelines and
    /// the VRAM they hold -- to produce a detector that is immediately shadowed.
    yunet: Option<YuNetDetector>,
    /// Behind an `Arc` so [`Self::with_postprocess`] can re-wrap without reloading a session.
    scrfd: Option<Arc<ScrfdDetector>>,
    score_threshold: f32,
    nms_threshold: f32,
}

impl FaceDetector {
    /// Load SCRFD if it can run, and only build YuNet when it cannot.
    ///
    /// The builder is a closure rather than a value so the fallback is never constructed when
    /// it would not be used: on the GPU path `YuNetDetector` compiles five WGSL pipelines and
    /// holds VRAM for them.
    pub fn load_or_build<F>(build_yunet: F) -> Result<Self>
    where
        F: FnOnce() -> Result<YuNetDetector>,
    {
        match ScrfdDetector::load() {
            Some(scrfd) => Ok(Self::with_parts(None, Some(Arc::new(scrfd)))),
            None => Ok(Self::with_parts(Some(build_yunet()?), None)),
        }
    }

    /// Wrap a YuNet detector, preferring SCRFD when its model and a runtime are both present.
    pub fn new(yunet: YuNetDetector) -> Self {
        Self::with_scrfd(yunet, ScrfdDetector::load().map(Arc::new))
    }

    /// Wrap a YuNet detector with an explicit SCRFD, or `None` to force YuNet.
    pub fn with_scrfd(yunet: YuNetDetector, scrfd: Option<Arc<ScrfdDetector>>) -> Self {
        Self::with_parts(Some(yunet), scrfd)
    }

    fn with_parts(yunet: Option<YuNetDetector>, scrfd: Option<Arc<ScrfdDetector>>) -> Self {
        Self {
            yunet,
            scrfd,
            score_threshold: SCRFD_SCORE_THRESHOLD,
            nms_threshold: SCRFD_NMS_THRESHOLD,
        }
    }

    /// Run detection on an in-memory image.
    pub fn detect_image(&self, image: &DynamicImage) -> Result<DetectionOutput> {
        let Some(scrfd) = self.scrfd.as_ref() else {
            return self.yunet_or_error()?.detect_image(image);
        };
        let detections = scrfd.detect(image, self.score_threshold, self.nms_threshold)?;
        Ok(DetectionOutput {
            detections,
            // SCRFD's decoding has already undone its own letterbox, so these are the identity:
            // a caller that rescaled by them would move every box a second time.
            scale_x: 1.0,
            scale_y: 1.0,
            original_size: (image.width(), image.height()),
        })
    }

    /// Run detection on an image file.
    pub fn detect_path<P: AsRef<Path>>(&self, path: P) -> Result<DetectionOutput> {
        let path = path.as_ref();
        if self.scrfd.is_none() {
            return self.yunet_or_error()?.detect_path(path);
        }
        let image = fcs_utils::load_image(path)
            .with_context(|| format!("failed to load image from {}", path.display()))?;
        self.detect_image(&image)
    }

    /// The same detector with different YuNet post-processing, sharing this SCRFD session.
    ///
    /// Only YuNet's settings change: SCRFD filters with its own threshold, for the reason given
    /// on [`SCRFD_SCORE_THRESHOLD`].
    pub fn with_postprocess(&self, postprocess: PostprocessConfig) -> Self {
        Self {
            yunet: self.yunet.as_ref().map(|y| y.with_postprocess(postprocess)),
            scrfd: self.scrfd.clone(),
            score_threshold: self.score_threshold,
            nms_threshold: self.nms_threshold,
        }
    }

    /// Which detector is actually running, for logs and the GUI's status line.
    pub fn model_name(&self) -> &'static str {
        if self.scrfd.is_some() {
            "SCRFD-80k"
        } else {
            "YuNet 2023"
        }
    }

    /// The inference backend underneath, for diagnostics.
    pub fn inference_backend(&self) -> &'static str {
        match (&self.scrfd, &self.yunet) {
            (Some(_), _) => "onnxruntime",
            (None, Some(yunet)) => yunet.inference_backend(),
            (None, None) => "none",
        }
    }

    /// GPU memory held by the YuNet backend, or `None` on CPU.
    pub fn gpu_memory_usage(&self) -> Option<u64> {
        self.yunet
            .as_ref()
            .and_then(YuNetDetector::gpu_memory_usage)
    }

    /// The YuNet fallback, which is absent when SCRFD loaded and nothing would have used it.
    fn yunet_or_error(&self) -> Result<&YuNetDetector> {
        self.yunet
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no detector: SCRFD is absent and YuNet was not built"))
    }

    /// The YuNet detector underneath, absent when SCRFD is doing the work.
    pub fn yunet(&self) -> Option<&YuNetDetector> {
        self.yunet.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PostprocessConfig, PreprocessConfig};

    fn yunet() -> Option<YuNetDetector> {
        let path = fcs_utils::model_path("models/face_detection_yunet_2023mar_640.onnx").ok()??;
        YuNetDetector::new(
            path,
            PreprocessConfig::default(),
            PostprocessConfig::default(),
        )
        .ok()
    }

    #[test]
    fn the_fallback_is_not_built_when_scrfd_is_there() {
        // The point of the closure: on the GPU path building YuNet compiles five WGSL pipelines
        // and holds VRAM for them, and when SCRFD loads nothing would ever call it.
        let mut built = false;
        let detector = FaceDetector::load_or_build(|| {
            built = true;
            anyhow::bail!("the fallback should not have been built")
        });
        match detector {
            // With a runtime and the model present, SCRFD wins and the closure never runs.
            Ok(detector) => {
                assert_eq!(detector.model_name(), "SCRFD-80k");
                assert!(!built, "the fallback was built despite SCRFD loading");
                assert!(detector.yunet().is_none());
            }
            // Without them the closure is the only way to get a detector, so it must have run.
            Err(_) => assert!(built, "neither SCRFD nor the fallback was attempted"),
        }
    }

    #[test]
    fn without_scrfd_it_is_yunet_and_says_so() {
        let Some(yunet) = yunet() else {
            eprintln!("skipped: the bundled YuNet model is not present");
            return;
        };
        let detector = FaceDetector::with_scrfd(yunet, None);
        assert_eq!(detector.model_name(), "YuNet 2023");
        // The floor is the point: no runtime must still mean a working detector.
        let image = DynamicImage::ImageRgb8(image::RgbImage::new(64, 64));
        let output = detector
            .detect_image(&image)
            .expect("YuNet runs without SCRFD");
        assert_eq!(output.original_size, (64, 64));
    }
}
