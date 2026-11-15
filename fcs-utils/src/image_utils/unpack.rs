//! Copying tightly packed rows into a buffer with padded rows, checked by tools/verus.

use vstd::prelude::*;

verus! {

/// Copy `height` rows of `row_bytes` from tightly packed `src` into `dst`, whose rows start
/// every `stride` bytes. Bytes between rows are left alone. Returns false, copying nothing,
/// when a row is longer than the stride or either buffer is too short; any extra bytes at the
/// end of `src` are ignored.
pub fn unpack_rows(src: &[u8], row_bytes: usize, height: usize, dst: &mut [u8], stride: usize) -> (copied: bool)
    ensures
        final(dst).len() == old(dst).len(),
        copied == (row_bytes <= stride && row_bytes * height <= src.len() && (height == 0
            || (height - 1) * stride + row_bytes <= old(dst).len())),
        copied ==> forall|y: int, i: int| 0 <= y < height && 0 <= i < row_bytes ==>
            #[trigger] final(dst)[y * stride + i] == src[y * row_bytes + i],
        !copied ==> final(dst)@ == old(dst)@,
{
    if stride < row_bytes {
        return false;
    }
    let Some(needed) = row_bytes.checked_mul(height) else {
        return false;
    };
    if needed > src.len() {
        return false;
    }
    if height == 0 {
        return true;
    }
    let Some(last) = (height - 1).checked_mul(stride) else {
        return false;
    };
    if last > dst.len() || row_bytes > dst.len() - last {
        return false;
    }
    let mut y = 0;
    while y < height
        invariant
            dst.len() == old(dst).len(),
            row_bytes <= stride,
            row_bytes * height <= src.len(),
            (height - 1) * stride + row_bytes <= dst.len(),
            0 < height,
            y <= height,
            forall|r: int, i: int| 0 <= r < y && 0 <= i < row_bytes ==>
                #[trigger] dst[r * stride + i] == src[r * row_bytes + i],
        decreases height - y,
    {
        proof {
            assert(y * stride <= (height - 1) * stride) by (nonlinear_arith)
                requires y < height;
            assert((y + 1) * row_bytes <= height * row_bytes) by (nonlinear_arith)
                requires y < height;
            assert((y + 1) * row_bytes == y * row_bytes + row_bytes) by (nonlinear_arith);
            assert(row_bytes * height == height * row_bytes) by (nonlinear_arith);
        }
        let at = y * stride;
        let from = y * row_bytes;
        let ghost before = dst@;
        dst[at..at + row_bytes].copy_from_slice(&src[from..from + row_bytes]);
        proof {
            // Earlier rows end before this one starts, because a row fits in a stride.
            assert forall|r: int, i: int| 0 <= r < y && 0 <= i < row_bytes implies
                #[trigger] dst[r * stride + i] == src[r * row_bytes + i] by {
                assert(r * stride + i < at) by (nonlinear_arith)
                    requires r < y, i < row_bytes, row_bytes <= stride, at == y * stride;
                assert(dst[r * stride + i] == before[r * stride + i]);
            }
            assert forall|i: int| 0 <= i < row_bytes implies
                #[trigger] dst[y * stride + i] == src[y * row_bytes + i] by {
                assert(dst[at + i] == src[from + i]);
            }
        }
        y += 1;
    }
    true
}

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copies_rows_into_padding_and_rejects_short_buffers() {
        let src: Vec<u8> = (1..=6).collect();
        let mut dst = vec![0; 8];
        assert!(unpack_rows(&src, 3, 2, &mut dst, 4));
        assert_eq!(dst, [1, 2, 3, 0, 4, 5, 6, 0]);
        // The last row may end at the buffer's end, with no trailing padding.
        let mut dst = vec![0; 7];
        assert!(unpack_rows(&src, 3, 2, &mut dst, 4));
        assert_eq!(dst, [1, 2, 3, 0, 4, 5, 6]);
        // Extra source bytes past `height` rows are never copied.
        let mut dst = vec![0; 4];
        assert!(unpack_rows(&src, 3, 1, &mut dst, 4));
        assert_eq!(dst, [1, 2, 3, 0]);

        let mut dst = vec![9; 8];
        assert!(!unpack_rows(&src, 3, 2, &mut dst[..6], 4), "dst too short");
        assert!(!unpack_rows(&src, 3, 3, &mut dst, 4), "src too short");
        assert!(
            !unpack_rows(&src, 3, 2, &mut dst, 2),
            "stride shorter than a row"
        );
        assert!(
            !unpack_rows(&src, usize::MAX, 2, &mut dst, usize::MAX),
            "overflow"
        );
        assert_eq!(dst, [9; 8], "a refused copy writes nothing");
        // Nothing to copy: zero-width or zero-height images.
        assert!(unpack_rows(&[], 0, 5, &mut [], 0));
        assert!(unpack_rows(&src, 3, 0, &mut [], 4));
    }
}
