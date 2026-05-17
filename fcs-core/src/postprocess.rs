use anyhow::Result;
use fcs_utils::{config::DetectionSettings, point::Point};
use std::cmp::Ordering;
use tract_onnx::prelude::{Tensor, tract_ndarray::ArrayView2};

use crate::{
    model_config::{DETECTION_OUTPUT_COLS, DETECTION_SCORE_INDEX},
    nms::{apply_nms_in_place, dedup_close_centers},
};

/// Centroid-distance threshold for the post-NMS dedup pass, expressed as a
/// fraction of the *larger* bbox's longest edge (see `dedup_close_centers`).
/// `0.5` means "if two box centers are within half the bigger face's longest
/// edge, treat them as the same face." Sized to span YuNet's multi-stride
/// anchors while leaving genuinely-separate faces alone (their centers
/// typically sit a full face-width or more apart).
const POST_NMS_RELATIVE_DEDUP_DISTANCE: f32 = 0.5;

/// Hard floor for the post-NMS dedup pass in pixels, applied alongside the
/// relative-distance check. Prevents pathologically small bboxes from
/// degrading the relative metric to nothing.
const POST_NMS_MIN_ABSOLUTE_DEDUP_DISTANCE_PX: f32 = 8.0;

pub use crate::model_config::{DEFAULT_NMS_THRESHOLD, DEFAULT_SCORE_THRESHOLD, DEFAULT_TOP_K};

/// Canonical YuNet detection configuration.
///
/// These parameters control how raw model outputs are filtered and refined.
#[derive(Debug, Clone)]
pub struct PostprocessConfig {
    /// Minimum confidence score for a detection to be considered valid.
    pub score_threshold: f32,
    /// Threshold for non-maximum suppression to merge overlapping bounding boxes.
    pub nms_threshold: f32,
    /// The maximum number of detections to return after sorting by score.
    pub top_k: usize,
}

impl Default for PostprocessConfig {
    fn default() -> Self {
        Self {
            score_threshold: DEFAULT_SCORE_THRESHOLD,
            nms_threshold: DEFAULT_NMS_THRESHOLD,
            top_k: DEFAULT_TOP_K,
        }
    }
}

/// Axis-aligned bounding box in image coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    /// The x-coordinate of the top-left corner.
    pub x: f32,
    /// The y-coordinate of the top-left corner.
    pub y: f32,
    /// The width of the box.
    pub width: f32,
    /// The height of the box.
    pub height: f32,
}

impl BoundingBox {
    /// Calculates the area of the bounding box.
    #[inline]
    pub fn area(&self) -> f32 {
        (self.width.max(0.0)) * (self.height.max(0.0))
    }

    /// Calculates the Intersection over Union (IoU) with another bounding box.
    #[inline]
    pub fn iou(&self, other: &Self) -> f32 {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = (self.x + self.width).min(other.x + other.width);
        let y2 = (self.y + self.height).min(other.y + other.height);

        if x2 <= x1 || y2 <= y1 {
            return 0.0;
        }

        let intersection = (x2 - x1) * (y2 - y1);

        let union = self.area() + other.area() - intersection;
        if union > 0.0 {
            intersection / union
        } else {
            0.0
        }
    }
}

/// Facial landmark coordinate (x, y) in image space.
pub type Landmark = Point;

/// A single YuNet detection result, including a bounding box, landmarks, and confidence score.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    /// The bounding box of the detected face.
    pub bbox: BoundingBox,
    /// An array of 5 facial landmarks (right eye, left eye, nose tip, right mouth corner, left mouth corner).
    pub landmarks: [Landmark; 5],
    /// The confidence score of the detection.
    pub score: f32,
}

/// Decode YuNet outputs into filtered detections.
///
/// This function takes the raw tensor output from the model and applies:
/// 1. Score filtering.
/// 2. Coordinate scaling to match the original image dimensions.
/// 3. Non-maximum suppression (NMS).
///
/// # Arguments
///
/// * `output` - The raw output tensor from the YuNet model.
/// * `scale_x` - The horizontal scale factor to map coordinates to the original image.
/// * `scale_y` - The vertical scale factor to map coordinates to the original image.
/// * `config` - The post-processing parameters.
pub fn apply_postprocess(
    output: &Tensor,
    scale_x: f32,
    scale_y: f32,
    config: &PostprocessConfig,
) -> Result<Vec<Detection>> {
    let rows = detection_rows(output)?;
    anyhow::ensure!(
        rows.shape()[1] == DETECTION_OUTPUT_COLS,
        "YuNet output must have {} columns per detection",
        DETECTION_OUTPUT_COLS
    );

    let mut detections = Vec::with_capacity(rows.nrows());
    for row in rows.rows() {
        let score = row[DETECTION_SCORE_INDEX];
        if !score.is_finite() || score < config.score_threshold {
            continue;
        }

        let bbox = BoundingBox {
            x: row[0] * scale_x,
            y: row[1] * scale_y,
            width: row[2] * scale_x,
            height: row[3] * scale_y,
        };
        if bbox.width <= 0.0 || bbox.height <= 0.0 {
            continue;
        }

        let landmarks = [
            Landmark {
                x: row[4] * scale_x,
                y: row[5] * scale_y,
            },
            Landmark {
                x: row[6] * scale_x,
                y: row[7] * scale_y,
            },
            Landmark {
                x: row[8] * scale_x,
                y: row[9] * scale_y,
            },
            Landmark {
                x: row[10] * scale_x,
                y: row[11] * scale_y,
            },
            Landmark {
                x: row[12] * scale_x,
                y: row[13] * scale_y,
            },
        ];

        detections.push(Detection {
            bbox,
            landmarks,
            score,
        });
    }

    sort_by_score_desc(&mut detections, config.top_k);

    if config.nms_threshold > 0.0 && detections.len() > 1 {
        apply_nms_in_place(&mut detections, config.nms_threshold);
    }

    dedup_close_centers(
        &mut detections,
        POST_NMS_RELATIVE_DEDUP_DISTANCE,
        POST_NMS_MIN_ABSOLUTE_DEDUP_DISTANCE_PX,
    );

    Ok(detections)
}

/// Extract the detection rows from the model's output tensor.
fn detection_rows<'a>(output: &'a Tensor) -> Result<ArrayView2<'a, f32>> {
    let shape = output.shape();
    let rows = match shape {
        [rows, DETECTION_OUTPUT_COLS] => *rows,
        [1, rows, DETECTION_OUTPUT_COLS] => *rows,
        other => anyhow::bail!(
            "YuNet output must have shape [N, {}] or [1, N, {}] (got {:?})",
            DETECTION_OUTPUT_COLS,
            DETECTION_OUTPUT_COLS,
            other
        ),
    };

    let slice = output
        .as_slice::<f32>()
        .map_err(|e| anyhow::anyhow!("YuNet output is not f32: {e}"))?;

    ArrayView2::from_shape((rows, DETECTION_OUTPUT_COLS), slice)
        .map_err(|_| anyhow::anyhow!("YuNet output data is not contiguous"))
}

/// Apply non-maximum suppression to a list of detections.
fn sort_by_score_desc(detections: &mut Vec<Detection>, top_k: usize) {
    fn cmp(a: &Detection, b: &Detection) -> Ordering {
        b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal)
    }

    if top_k > 0 && detections.len() > top_k {
        let nth = top_k - 1;
        detections.select_nth_unstable_by(nth, cmp);
        let (head, _) = detections.split_at_mut(top_k);
        head.sort_unstable_by(cmp);
        detections.truncate(top_k);
    } else {
        detections.sort_unstable_by(cmp);
    }
}

impl From<DetectionSettings> for PostprocessConfig {
    fn from(settings: DetectionSettings) -> Self {
        PostprocessConfig {
            score_threshold: settings.score_threshold,
            nms_threshold: settings.nms_threshold,
            top_k: settings.top_k,
        }
    }
}

impl From<&DetectionSettings> for PostprocessConfig {
    fn from(settings: &DetectionSettings) -> Self {
        PostprocessConfig {
            score_threshold: settings.score_threshold,
            nms_threshold: settings.nms_threshold,
            top_k: settings.top_k,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fcs_utils::config::DetectionSettings;

    fn detection_with_score(score: f32, bbox: BoundingBox) -> Detection {
        Detection {
            bbox,
            landmarks: [Landmark::new(0.0, 0.0); 5],
            score,
        }
    }

    fn tensor_from_rows(rows: &[[f32; DETECTION_OUTPUT_COLS]]) -> Tensor {
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        Tensor::from_shape(&[rows.len(), DETECTION_OUTPUT_COLS], &flat)
            .expect("build test tensor from rows")
    }

    #[test]
    fn filters_by_score_and_scales_coordinates() {
        let tensor = tensor_from_rows(&[
            [
                10.0, 20.0, 30.0, 40.0, // bbox
                12.0, 22.0, // landmarks...
                18.0, 25.0, //
                25.0, 30.0, //
                15.0, 35.0, //
                20.0, 28.0, //
                0.95, // score
            ],
            [
                5.0, 5.0, 10.0, 10.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 9.0, 10.0, 10.0, 0.2,
            ],
        ]);

        let detections = apply_postprocess(
            &tensor,
            2.0,
            0.5,
            &PostprocessConfig {
                score_threshold: 0.3,
                ..Default::default()
            },
        )
        .expect("postprocess should succeed");

        assert_eq!(detections.len(), 1);
        let det = &detections[0];
        assert_eq!(det.score, 0.95);
        assert!((det.bbox.x - 20.0).abs() < f32::EPSILON);
        assert!((det.bbox.y - 10.0).abs() < f32::EPSILON);
        assert!((det.bbox.width - 60.0).abs() < f32::EPSILON);
        assert!((det.bbox.height - 20.0).abs() < f32::EPSILON);
        assert!((det.landmarks[0].x - 24.0).abs() < f32::EPSILON);
        assert!((det.landmarks[0].y - 11.0).abs() < f32::EPSILON);
    }

    #[test]
    fn applies_non_max_suppression() {
        let tensor = tensor_from_rows(&[
            [
                0.0, 0.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.99,
            ],
            [
                1.0, 1.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.95,
            ],
        ]);

        let detections = apply_postprocess(
            &tensor,
            1.0,
            1.0,
            &PostprocessConfig {
                score_threshold: 0.3,
                nms_threshold: 0.3,
                top_k: 10,
            },
        )
        .expect("postprocess should succeed");

        assert_eq!(detections.len(), 1);
        assert!((detections[0].score - 0.99).abs() < f32::EPSILON);
    }

    #[test]
    fn handles_batched_output_shape() {
        let tensor = Tensor::from_shape(
            &[1, 1, DETECTION_OUTPUT_COLS],
            &[
                0.0f32, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.95,
            ],
        )
        .expect("build batched test tensor");

        let detections = apply_postprocess(&tensor, 1.0, 1.0, &PostprocessConfig::default())
            .expect("postprocess should succeed");
        assert_eq!(detections.len(), 1);
    }

    #[test]
    fn converts_detection_settings_into_config() {
        let settings = DetectionSettings {
            score_threshold: 0.75,
            nms_threshold: 0.25,
            top_k: 123,
        };

        let config: PostprocessConfig = settings.clone().into();
        assert_eq!(config.score_threshold, settings.score_threshold);
        assert_eq!(config.nms_threshold, settings.nms_threshold);
        assert_eq!(config.top_k, settings.top_k);

        let config_from_ref: PostprocessConfig = (&settings).into();
        assert_eq!(config_from_ref.score_threshold, settings.score_threshold);
        assert_eq!(config_from_ref.nms_threshold, settings.nms_threshold);
        assert_eq!(config_from_ref.top_k, settings.top_k);
    }

    #[test]
    fn bounding_box_area_and_iou_handle_overlap_and_degenerate_boxes() {
        let a = BoundingBox {
            x: 0.0,
            y: 0.0,
            width: 10.0,
            height: 10.0,
        };
        let b = BoundingBox {
            x: 5.0,
            y: 5.0,
            width: 10.0,
            height: 10.0,
        };
        let degenerate = BoundingBox {
            x: 0.0,
            y: 0.0,
            width: -5.0,
            height: 2.0,
        };

        assert_eq!(a.area(), 100.0);
        assert_eq!(degenerate.area(), 0.0);
        assert!((a.iou(&b) - (25.0 / 175.0)).abs() < 1e-6);
        assert_eq!(a.iou(&degenerate), 0.0);
    }

    #[test]
    fn apply_postprocess_rejects_invalid_tensor_shape_and_type() {
        let wrong_shape =
            Tensor::from_shape(&[DETECTION_OUTPUT_COLS], &[0.0f32; DETECTION_OUTPUT_COLS])
                .expect("build wrong-shape test tensor");
        let err = apply_postprocess(&wrong_shape, 1.0, 1.0, &PostprocessConfig::default())
            .unwrap_err()
            .to_string();
        assert!(err.contains("YuNet output must have shape"));

        let wrong_type =
            Tensor::from_shape(&[1, DETECTION_OUTPUT_COLS], &[1i32; DETECTION_OUTPUT_COLS])
                .expect("build wrong-type test tensor");
        let err = apply_postprocess(&wrong_type, 1.0, 1.0, &PostprocessConfig::default())
            .unwrap_err()
            .to_string();
        assert!(err.contains("YuNet output is not f32"));
    }

    #[test]
    fn sort_by_score_desc_honors_top_k_and_zero_as_unlimited() {
        let mut detections = vec![
            detection_with_score(
                0.1,
                BoundingBox {
                    x: 0.0,
                    y: 0.0,
                    width: 1.0,
                    height: 1.0,
                },
            ),
            detection_with_score(
                0.9,
                BoundingBox {
                    x: 2.0,
                    y: 0.0,
                    width: 1.0,
                    height: 1.0,
                },
            ),
            detection_with_score(
                0.5,
                BoundingBox {
                    x: 4.0,
                    y: 0.0,
                    width: 1.0,
                    height: 1.0,
                },
            ),
        ];

        sort_by_score_desc(&mut detections, 2);
        assert_eq!(detections.len(), 2);
        assert!((detections[0].score - 0.9).abs() < f32::EPSILON);
        assert!((detections[1].score - 0.5).abs() < f32::EPSILON);

        let mut detections = vec![
            detection_with_score(
                0.1,
                BoundingBox {
                    x: 0.0,
                    y: 0.0,
                    width: 1.0,
                    height: 1.0,
                },
            ),
            detection_with_score(
                0.9,
                BoundingBox {
                    x: 2.0,
                    y: 0.0,
                    width: 1.0,
                    height: 1.0,
                },
            ),
            detection_with_score(
                0.5,
                BoundingBox {
                    x: 4.0,
                    y: 0.0,
                    width: 1.0,
                    height: 1.0,
                },
            ),
        ];
        sort_by_score_desc(&mut detections, 0);
        assert_eq!(detections.len(), 3);
        assert!((detections[0].score - 0.9).abs() < f32::EPSILON);
        assert!((detections[1].score - 0.5).abs() < f32::EPSILON);
        assert!((detections[2].score - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn apply_postprocess_filters_non_finite_scores() {
        // Rows with NaN/Inf scores should be dropped regardless of threshold.
        let tensor = tensor_from_rows(&[
            [
                0.0,
                0.0,
                5.0,
                5.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                f32::NAN,
            ],
            [
                0.0,
                0.0,
                5.0,
                5.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                f32::INFINITY,
            ],
            [
                0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.95,
            ],
        ]);
        let detections = apply_postprocess(
            &tensor,
            1.0,
            1.0,
            &PostprocessConfig {
                score_threshold: 0.0,
                ..Default::default()
            },
        )
        .expect("postprocess should succeed");
        assert_eq!(detections.len(), 1);
        assert!((detections[0].score - 0.95).abs() < f32::EPSILON);
    }

    #[test]
    fn apply_postprocess_drops_zero_area_boxes() {
        let tensor = tensor_from_rows(&[
            // zero width
            [
                0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.95,
            ],
            // zero height
            [
                0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.95,
            ],
            // valid box
            [
                1.0, 1.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9,
            ],
        ]);
        let detections = apply_postprocess(
            &tensor,
            1.0,
            1.0,
            &PostprocessConfig {
                score_threshold: 0.0,
                ..Default::default()
            },
        )
        .expect("postprocess should succeed");
        assert_eq!(detections.len(), 1);
        assert!((detections[0].bbox.x - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn iou_returns_zero_when_union_underflows_to_zero() {
        // Both boxes have positive but subnormal dimensions: area() underflows to 0.0 in f32,
        // so union = 0 + 0 - 0 = 0 → covers the `union > 0` else branch.
        let tiny = f32::MIN_POSITIVE;
        let a = BoundingBox {
            x: 0.0,
            y: 0.0,
            width: tiny,
            height: tiny,
        };
        let b = BoundingBox {
            x: 0.0,
            y: 0.0,
            width: tiny,
            height: tiny,
        };
        assert_eq!(a.iou(&b), 0.0);
    }
}
