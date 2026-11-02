//! Integer geometry and read addresses shared with the Lean verifier.

// shortcut: Aeneas lacks Option's ? model, remove this exception when supported.
#![allow(clippy::question_mark)]

/// Preserve the kernels' saturating small-input convention; reject invalid or overflowing sizes.
pub(crate) fn output_dim(input: usize, kernel: usize, stride: usize, pad: usize) -> Option<usize> {
    if kernel == 0 || stride == 0 {
        return None;
    }
    let Some(padding) = pad.checked_mul(2) else {
        return None;
    };
    let Some(padded) = input.checked_add(padding) else {
        return None;
    };
    (padded.saturating_sub(kernel) / stride).checked_add(1)
}

/// Columns whose complete kernel window is inside the input. Sizes must pass output_dim.
// shortcut: Aeneas lacks div_ceil/is_multiple_of models, use them when supported.
#[allow(clippy::manual_is_multiple_of)]
pub(crate) fn interior(
    width: usize,
    kernel: usize,
    stride: usize,
    pad: usize,
    out: usize,
) -> (usize, usize) {
    let hi = if width + pad >= kernel {
        (width + pad - kernel) / stride + 1
    } else {
        0
    }
    .min(out);
    let lo = pad / stride + if pad % stride == 0 { 0 } else { 1 };
    (lo.min(hi), hi)
}

/// Valid kernel rows and their first input row; height + pad must fit.
/// Empty windows have a bounded empty slice.
pub(crate) fn tap_rows(
    origin: usize,
    height: usize,
    kernel: usize,
    pad: usize,
) -> (usize, usize, usize) {
    let lo = pad.saturating_sub(origin).min(kernel);
    let hi = (height + pad).saturating_sub(origin).min(kernel).max(lo);
    let first = origin.saturating_sub(pad).min(height);
    (lo, hi, first)
}

/// Whether a padded coordinate names an input column; width + pad must fit.
#[inline]
pub(crate) fn column_inside(padded: usize, pad: usize, width: usize) -> bool {
    padded >= pad && padded - pad < width
}

/// Unpadded column for a tap known to be inside; origin + offset must fit.
#[inline]
pub(crate) fn column_index(origin: usize, offset: usize, pad: usize) -> usize {
    origin + offset - pad
}

#[inline]
pub(crate) fn input_index(row: usize, column: usize, block: usize, lane: usize) -> usize {
    (row + column) * block + lane
}

#[inline]
pub(crate) fn tap_offset(input_block: usize, ky: usize, kernel: usize, block: usize) -> usize {
    (input_block * kernel + ky) * kernel * block
}

#[inline]
pub(crate) fn weight_index(
    filter: usize,
    per_block: usize,
    tap: usize,
    kx: usize,
    block: usize,
    lane: usize,
) -> usize {
    filter * per_block + tap + kx * block + lane
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_matches_integer_windows_and_rejects_overflow() {
        for width in 0..12usize {
            for kernel in [1usize, 3] {
                for stride in [1usize, 2] {
                    for pad in 0..6usize {
                        let out = output_dim(width, kernel, stride, pad).unwrap();
                        assert_eq!(out, (width + 2 * pad).saturating_sub(kernel) / stride + 1);
                        let (lo, hi) = interior(width, kernel, stride, pad, out);
                        for x in 0..out {
                            assert_eq!(
                                lo <= x && x < hi,
                                (0..kernel).all(|k| {
                                    let i = (x * stride + k) as i64 - pad as i64;
                                    0 <= i && i < width as i64
                                })
                            );
                            let (a, b, first) = tap_rows(x * stride, width, kernel, pad);
                            assert!(first + b - a <= width);
                            for k in 0..kernel {
                                let i = (x * stride + k) as i64 - pad as i64;
                                assert_eq!(a <= k && k < b, 0 <= i && i < width as i64);
                                if a <= k && k < b {
                                    assert_eq!(first + k - a, i as usize);
                                }
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(output_dim(1, 1, 1, usize::MAX), None);
        assert_eq!(output_dim(usize::MAX, 1, 1, 1), None);
        assert_eq!(output_dim(1, 0, 1, 0), None);
        assert_eq!(output_dim(1, 1, 0, 0), None);
        assert_eq!(tap_rows(0, 1, 3, 4), (3, 3, 0));
        assert_eq!(tap_rows(8, 1, 3, 4), (0, 0, 1));
    }
}
