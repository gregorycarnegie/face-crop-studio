//! Integer addresses for blocked-channel extraction and nearest-neighbour resizing.

pub(crate) fn double_dim(value: usize) -> Option<usize> {
    value.checked_mul(2)
}

// shortcut: Aeneas lacks div_ceil/is_multiple_of models, use them when supported.
#[allow(clippy::manual_div_ceil, clippy::manual_is_multiple_of)]
pub(crate) fn channel_blocks(channels: usize) -> usize {
    channels / 8 + if channels % 8 == 0 { 0 } else { 1 }
}

pub(crate) fn channel_end(first: usize, count: usize, channels: usize) -> Option<usize> {
    match first.checked_add(count) {
        Some(end) if end <= channels => Some(end),
        _ => None,
    }
}

#[inline]
pub(crate) fn channel_parts(channel: usize) -> (usize, usize) {
    (channel / 8, channel % 8)
}

#[inline]
pub(crate) fn plane_base(batch: usize, block: usize, blocks: usize, plane: usize) -> usize {
    (batch * blocks + block) * plane
}

/// Real lanes in a valid channel block.
#[inline]
pub(crate) fn active_lanes(block: usize, channels: usize) -> usize {
    8.min(channels - block * 8)
}

/// Source row of flattened output row `row` of a 2x nearest-neighbour resize.
#[inline]
pub(crate) fn upsample_row(row: usize, width: usize) -> (usize, usize) {
    let start = (row / 2) * width;
    (start, start + width)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_ranges_and_partial_blocks() {
        assert_eq!(channel_end(5, 3, 8), Some(8));
        assert_eq!(channel_end(usize::MAX, 1, 8), None);
        assert_eq!(channel_end(9, 0, 8), None);
        for channels in 1..40usize {
            for c in 0..channels {
                let (b, lane) = channel_parts(c);
                assert_eq!(b * 8 + lane, c);
                assert!(lane < active_lanes(b, channels));
            }
        }
        for row in 0..24 {
            let (start, end) = upsample_row(row, 7);
            assert_eq!((start, end), ((row / 2) * 7, (row / 2 + 1) * 7));
        }
    }
}
