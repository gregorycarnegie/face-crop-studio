//! YuNet model-specific constants shared by inference and post-processing.

/// Default minimum confidence score for a detection to be considered valid.
pub const DEFAULT_SCORE_THRESHOLD: f32 = 0.9;
/// Default threshold for non-maximum suppression to merge overlapping bounding boxes.
pub const DEFAULT_NMS_THRESHOLD: f32 = 0.3;
/// Default maximum number of detections to return after sorting by score.
pub const DEFAULT_TOP_K: usize = 5_000;

/// Strides emitted by the YuNet head, in output tensor order.
pub const STRIDES: [usize; 3] = [8, 16, 32];
/// Number of output tensors produced for each stride: cls, obj, bbox, kps.
pub const OUTPUTS_PER_STRIDE: usize = 4;
/// Number of columns in YuNet detection output (bbox + landmarks + score).
pub const DETECTION_OUTPUT_COLS: usize = 15;
/// Index of the confidence score in a detection row.
pub const DETECTION_SCORE_INDEX: usize = 14;
/// Alignment used when deriving stride grid dimensions from model input size.
pub const STRIDE_ALIGNMENT: usize = 32;
/// Spatial grid resolution used by the optimized NMS path.
pub const NMS_GRID_SIZE: usize = 32;

/// "Same" padding for an odd square kernel: half of it, rounded down.
pub(crate) fn same_padding(kernel: usize) -> usize {
    kernel / 2
}

/// An environment override that names a positive count; anything else means "use the default".
pub(crate) fn positive_count(value: Option<&str>) -> Option<usize> {
    value
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n > 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_padding_is_half_the_kernel() {
        // 5 % 2 and 7 % 2 are both 1, so a remainder cannot pass for the half.
        assert_eq!([1, 3, 5, 7].map(same_padding), [0, 1, 2, 3]);
    }

    #[test]
    fn positive_count_accepts_only_a_positive_number() {
        assert_eq!(positive_count(Some("3")), Some(3));
        assert_eq!(positive_count(Some("0")), None);
        assert_eq!(positive_count(Some("-2")), None);
        assert_eq!(positive_count(Some("x")), None);
        assert_eq!(positive_count(None), None);
    }
}
