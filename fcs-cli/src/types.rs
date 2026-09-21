//! Shared types and conversions for fcs-cli.

use fcs_core::Detection;
use serde::Serialize;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

/// Atomic per-run counters shared across rayon worker threads. Cloning is cheap —
/// each field is an `Arc<AtomicUsize>`, so cloned instances refer to the same counter.
#[derive(Clone, Default)]
pub(crate) struct ProgressCounters {
    pub images_processed: Arc<AtomicUsize>,
    pub faces_detected: Arc<AtomicUsize>,
    pub crops_saved: Arc<AtomicUsize>,
    pub crops_skipped_quality: Arc<AtomicUsize>,
}

impl ProgressCounters {
    pub fn snapshot(&self) -> ProgressSnapshot {
        ProgressSnapshot {
            images_processed: self.images_processed.load(Ordering::Relaxed),
            faces_detected: self.faces_detected.load(Ordering::Relaxed),
            crops_saved: self.crops_saved.load(Ordering::Relaxed),
            crops_skipped_quality: self.crops_skipped_quality.load(Ordering::Relaxed),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ProgressSnapshot {
    pub images_processed: usize,
    pub faces_detected: usize,
    pub crops_saved: usize,
    pub crops_skipped_quality: usize,
}

/// A serializable representation of a single detection.
#[derive(Debug, Serialize)]
pub struct DetectionRecord {
    pub score: f32,
    pub bbox: [f32; 4],
    /// Five landmark slots, `null` where the detector predicted nothing.
    ///
    /// Emitted as `null` rather than `[0, 0]`, which is what 1.x wrote for the three the
    /// detector was never trained on. A consumer had no way to tell that apart from a real
    /// landmark at the origin, and this project's own code got it wrong more than once.
    pub landmarks: [Option<[f32; 2]>; 5],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<String>,
}

/// A serializable representation of all detections for a single image.
#[derive(Debug, Serialize)]
pub struct ImageDetections {
    pub image: String,
    pub detections: Vec<DetectionRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotated: Option<String>,
}

impl From<&Detection> for DetectionRecord {
    fn from(detection: &Detection) -> Self {
        Self {
            score: detection.score,
            bbox: [
                detection.bbox.x,
                detection.bbox.y,
                detection.bbox.width,
                detection.bbox.height,
            ],
            landmarks: detection.landmarks.map(|lm| lm.map(|p| [p.x, p.y])),
            quality_score: None,
            quality: None,
        }
    }
}
