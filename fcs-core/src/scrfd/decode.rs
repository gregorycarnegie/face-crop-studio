//! Turning one head row into a detection: plain `f32` geometry, kept apart so Kani can check it.

use crate::postprocess::{BoundingBox, Detection, Landmark};

/// Anchors per cell, from the config's two scales at one ratio.
pub(super) const ANCHORS_PER_CELL: usize = 2;

/// Landmarks the head predicts: eyes, nose, then the two mouth corners.
pub(super) const LANDMARKS: usize = 5;

/// How many of those five this model was actually taught: the two eyes, and nothing else.
/// See the note in `decode_row` on why the rest are reported absent rather than passed through.
pub(super) const TRAINED_LANDMARKS: usize = 2;

/// How a source image was laid onto the square input, and what undoes it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Letterbox {
    /// Factor the source was multiplied by; detections are divided by it to come back.
    pub scale: f32,
}

/// One row's detection: `boxes` holds its four distances and `points` starts at its landmark
/// offsets, of which only the two trained eyes are read.
pub(super) fn decode_row(
    score: f32,
    boxes: &[f32],
    points: &[f32],
    row: usize,
    cells: usize,
    stride: f32,
    letterbox: Letterbox,
) -> Detection {
    // Rows run spatially with the anchors interleaved, so a cell's anchors are adjacent.
    let cell = row / ANCHORS_PER_CELL;
    let centre_x = (cell % cells) as f32 * stride;
    let centre_y = (cell / cells) as f32 * stride;

    // distance2bbox: left, top, right, bottom distances outward from the centre.
    let x1 = centre_x - boxes[0] * stride;
    let y1 = centre_y - boxes[1] * stride;
    let x2 = centre_x + boxes[2] * stride;
    let y2 = centre_y + boxes[3] * stride;

    // Only the two eyes are real. This model's landmark head was trained on eye pairs
    // clicked for this project, with nose and mouth corners written `-1` and weighted
    // to zero (`tools/dataset/to_labelv2.zig`), so those three outputs received no
    // gradient at all: they emit near-zero distances that decode to the anchor centre,
    // which lands *above* the face and is identical for all three. Left in, they look
    // like landmarks and are drawn as landmarks, so they are reported absent.
    let mut landmarks = [None; LANDMARKS];
    for (index, landmark) in landmarks.iter_mut().enumerate().take(TRAINED_LANDMARKS) {
        let dx = points[index * 2];
        let dy = points[index * 2 + 1];
        *landmark = Some(Landmark::new(
            (centre_x + dx * stride) / letterbox.scale,
            (centre_y + dy * stride) / letterbox.scale,
        ));
    }

    Detection {
        bbox: BoundingBox {
            x: x1 / letterbox.scale,
            y: y1 / letterbox.scale,
            width: (x2 - x1) / letterbox.scale,
            height: (y2 - y1) / letterbox.scale,
        },
        landmarks,
        score,
    }
}

/// The score head's activation. One function, so both decode paths round identically.
pub(super) fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The arms `tools/kani` leaves to a test: `exp` overflowing, and infinite logits.
    #[test]
    fn sigmoid_saturates_without_nan() {
        for (logit, expected) in [
            (-100.0, 0.0),
            (f32::NEG_INFINITY, 0.0),
            (100.0, 1.0),
            (f32::INFINITY, 1.0),
        ] {
            assert_eq!(sigmoid(logit), expected, "sigmoid({logit})");
        }
    }
}
