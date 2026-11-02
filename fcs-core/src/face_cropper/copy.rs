//! Integer clipping and row addresses for the crop blitter.

pub(super) fn clip_extent(requested: u32, offset: u32, canvas: u32) -> u32 {
    requested.min(canvas.saturating_sub(offset))
}

/// A row of pixels in a valid, packed image; the caller supplies valid coordinates.
#[inline]
pub(super) fn pixel_row(
    row: usize,
    column: usize,
    width: usize,
    count: usize,
    lanes: usize,
) -> (usize, usize) {
    let start = (row * width + column) * lanes;
    (start, start + count * lanes)
}

#[inline]
pub(super) fn opaque_rgba(rgb: [u8; 3]) -> [u8; 4] {
    [rgb[0], rgb[1], rgb[2], 255]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clips_copy_extents_and_addresses_rows_exactly() {
        assert_eq!(clip_extent(10, 7, 12), 5);
        assert_eq!(clip_extent(10, u32::MAX, 12), 0);
        assert_eq!(pixel_row(2, 3, 10, 4, 3), (69, 81));
        assert_eq!(opaque_rgba([2, 17, 255]), [2, 17, 255, 255]);
    }
}
