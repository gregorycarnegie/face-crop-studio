//! Checked NCHW plane sizes and ranges.

// shortcut: Aeneas lacks Option's ? model, remove this exception when supported.
#![allow(clippy::question_mark)]

use super::size::checked_elements;

pub(crate) fn checked_nchw(
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> Option<(usize, usize)> {
    let Some(plane) = checked_elements(&[height, width]) else {
        return None;
    };
    let Some(elements) = checked_elements(&[batch, channels, plane]) else {
        return None;
    };
    Some((plane, elements))
}

pub(crate) fn plane_range(
    batch: usize,
    channel: usize,
    batches: usize,
    channels: usize,
    plane: usize,
    elements: usize,
) -> Option<(usize, usize)> {
    if batch >= batches || channel >= channels {
        return None;
    }
    let Some(index) = batch.checked_mul(channels) else {
        return None;
    };
    let Some(index) = index.checked_add(channel) else {
        return None;
    };
    let Some(start) = index.checked_mul(plane) else {
        return None;
    };
    let Some(end) = start.checked_add(plane) else {
        return None;
    };
    if end > elements {
        None
    } else {
        Some((start, end))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_empty_planes_overflow_and_indices() {
        assert_eq!(checked_nchw(2, 3, 4, 5), Some((20, 120)));
        assert_eq!(checked_nchw(0, 1, usize::MAX, 2), None);
        assert_eq!(checked_nchw(usize::MAX, 2, 1, 1), None);
        assert_eq!(checked_nchw(2, 3, 0, 5), Some((0, 0)));
        assert_eq!(plane_range(1, 2, 2, 3, 20, 120), Some((100, 120)));
        assert_eq!(plane_range(0, 3, 2, 3, 20, 120), None);
        assert_eq!(plane_range(2, 0, 2, 3, 20, 120), None);
        assert_eq!(plane_range(1, 2, 2, 3, 20, 119), None);
        assert_eq!(plane_range(1, 2, 2, 3, 0, 0), Some((0, 0)));
        assert_eq!(plane_range(1, 1, 2, usize::MAX, 1, usize::MAX), None);
    }
}
