//! Detection types: what a detected face is, and the settings that filter them.
//!
//! The decode itself belongs to the detector. This used to hold `apply_postprocess`, which
//! turned YuNet's `[N, 15]` rows into detections; SCRFD decodes its own nine head tensors in
//! `crate::scrfd` because the layouts are nothing alike, so only the shared types remain here.

use fcs_utils::point::Point;

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
    /// Right edge of the bounding box.
    #[inline]
    pub fn right(&self) -> f32 {
        self.x + self.width
    }

    /// Bottom edge of the bounding box.
    #[inline]
    pub fn bottom(&self) -> f32 {
        self.y + self.height
    }

    /// Center point of the bounding box.
    #[inline]
    pub fn center(&self) -> Point {
        Point::new(
            self.width.mul_add(0.5, self.x),
            self.height.mul_add(0.5, self.y),
        )
    }

    /// Longest side of the bounding box.
    #[inline]
    pub fn longest_edge(&self) -> f32 {
        self.width.max(self.height)
    }

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
        let x2 = self.right().min(other.right());
        let y2 = self.bottom().min(other.bottom());

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

/// One detected face: a bounding box, five landmark slots, and a confidence score.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    /// The bounding box of the detected face.
    pub bbox: BoundingBox,
    /// Five landmark slots: right eye, left eye, nose tip, right mouth corner, left mouth corner.
    ///
    /// **Only the two eyes are populated.** The shipped detector was trained with the nose and
    /// mouth keypoints weighted to zero, so it never learned them; they are reported as
    /// `(0, 0)` rather than passed through, because the untrained head decodes to the anchor
    /// centre, which looks like a plausible facial point and is not one. Consumers that draw or
    /// measure landmarks must skip the zeros.
    pub landmarks: [Landmark; 5],
    /// The confidence score of the detection.
    pub score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

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

        assert_eq!(a.right(), 10.0);
        assert_eq!(a.bottom(), 10.0);
        assert_eq!(a.center(), Landmark::new(5.0, 5.0));
        assert_eq!(a.longest_edge(), 10.0);
        assert_eq!(a.area(), 100.0);
        assert_eq!(degenerate.area(), 0.0);
        assert!((a.iou(&b) - (25.0 / 175.0)).abs() < 1e-6);
        assert_eq!(a.iou(&degenerate), 0.0);
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
