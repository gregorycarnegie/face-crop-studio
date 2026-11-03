//! Checked copying of RGB rows from a padded decoder buffer.

// shortcut: Aeneas lacks Option's ? model, remove this exception when supported.
#![allow(clippy::question_mark)]

pub(crate) fn row_range(
    row: usize,
    row_bytes: usize,
    stride: usize,
    data_len: usize,
) -> Option<(usize, usize)> {
    let Some(start) = row.checked_mul(stride) else {
        return None;
    };
    let Some(end) = start.checked_add(row_bytes) else {
        return None;
    };
    if end > data_len {
        None
    } else {
        Some((start, end))
    }
}

/// Copy RGB pixels while discarding row padding; reject invalid strides, overflowing sizes and
/// truncated buffers before allocating or copying. The output is never longer than `data`,
/// so it always fits an allocation.
pub(super) fn pack_rows(
    data: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Option<Vec<u8>> {
    let Some(row_bytes) = width.checked_mul(3) else {
        return None;
    };
    if stride < row_bytes {
        return None;
    }
    let Some(length) = row_bytes.checked_mul(height) else {
        return None;
    };
    if height == 0 || row_bytes == 0 {
        return Some(Vec::new());
    }
    let Some(_) = row_range(height - 1, row_bytes, stride, data.len()) else {
        return None;
    };
    let mut output = Vec::with_capacity(length);
    let mut row = 0;
    while row < height {
        let Some((start, end)) = row_range(row, row_bytes, stride, data.len()) else {
            return None;
        };
        output.extend_from_slice(&data[start..end]);
        row += 1;
    }
    Some(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copies_pixels_without_padding_and_rejects_invalid_claims() {
        let data: Vec<u8> = (0..21).collect();
        assert_eq!(
            pack_rows(&data, 2, 3, 7),
            Some([&data[0..6], &data[7..13], &data[14..20]].concat())
        );
        assert_eq!(pack_rows(&data, 2, 4, 7), None);
        // The last row may end exactly at the buffer's end, with no trailing padding.
        assert_eq!(
            pack_rows(&data[..20], 2, 3, 7),
            Some([&data[0..6], &data[7..13], &data[14..20]].concat())
        );
        assert_eq!(pack_rows(&data[..19], 2, 3, 7), None);
        assert_eq!(pack_rows(&data, 2, 3, 5), None);
        assert_eq!(pack_rows(&[], usize::MAX, 0, 0), None);
        assert_eq!(pack_rows(&[], 1, usize::MAX, 3), None);
        assert_eq!(pack_rows(&[], 0, usize::MAX, 0), Some(vec![]));
        assert_eq!(pack_rows(&[], 2, 0, 6), Some(vec![]));
        assert_eq!(row_range(usize::MAX, 1, 2, usize::MAX), None);
        assert_eq!(row_range(usize::MAX, 1, 1, usize::MAX), None);
    }
}
