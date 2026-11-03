/// Multiply dimensions, rejecting the first prefix whose product exceeds usize.
// shortcut: Aeneas lacks Option's ? model, remove this lint exception when supported.
#[allow(clippy::question_mark)]
pub(crate) fn checked_elements(dims: &[usize]) -> Option<usize> {
    let mut elements = 1usize;
    let mut index = 0;
    while index < dims.len() {
        elements = match elements.checked_mul(dims[index]) {
            Some(count) => count,
            None => return None,
        };
        index += 1;
    }
    Some(elements)
}

/// Size of a densely packed f32 buffer, or None if its byte count overflows usize.
pub(crate) fn checked_f32_bytes(elements: usize) -> Option<usize> {
    elements.checked_mul(4)
}

/// Rust vectors and slices require a byte size no greater than isize::MAX.
pub(crate) fn checked_allocation(elements: usize, element_bytes: usize) -> Option<usize> {
    match elements.checked_mul(element_bytes) {
        Some(bytes) if bytes <= isize::MAX as usize => Some(bytes),
        _ => None,
    }
}

/// Convert an ONNX/ORT dimension without accepting negative values or narrowing wrap.
pub(crate) fn signed_dim(value: i64) -> Option<usize> {
    if value < 0 || value as u64 > usize::MAX as u64 {
        None
    } else {
        Some(value as usize)
    }
}

/// ORT's input shape uses signed 64-bit dimensions.
#[allow(dead_code)] // Used by the development-only ORT oracle.
pub(crate) fn ort_dim(value: usize) -> Option<i64> {
    if value as u128 > i64::MAX as u128 {
        None
    } else {
        Some(value as i64)
    }
}

// shortcut: Aeneas lacks is_multiple_of's model, use it when supported.
#[allow(clippy::manual_is_multiple_of)]
pub(crate) fn raw_f32_elements(bytes: usize) -> Option<usize> {
    if bytes % 4 == 0 {
        Some(bytes / 4)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn element_counts_handle_scalars_empty_tensors_and_overflow() {
        assert_eq!(checked_elements(&[]), Some(1));
        assert_eq!(checked_elements(&[2, 3, 4]), Some(24));
        assert_eq!(checked_elements(&[0, usize::MAX, usize::MAX]), Some(0));
        assert_eq!(checked_elements(&[usize::MAX, 0, usize::MAX]), Some(0));
        assert_eq!(checked_elements(&[usize::MAX, usize::MAX]), None);
        assert_eq!(checked_elements(&[usize::MAX, 2, 0]), None);
    }

    #[test]
    fn byte_counts_reject_the_first_unrepresentable_size() {
        let largest = usize::MAX / 4;
        assert_eq!(checked_f32_bytes(0), Some(0));
        assert_eq!(checked_f32_bytes(largest), Some(largest * 4));
        assert_eq!(checked_f32_bytes(largest + 1), None);
        assert_eq!(checked_f32_bytes(usize::MAX), None);
        assert_eq!(
            checked_allocation(isize::MAX as usize / 4, 4),
            Some(isize::MAX as usize / 4 * 4)
        );
        assert_eq!(checked_allocation(isize::MAX as usize / 4 + 1, 4), None);
        assert_eq!(checked_allocation(usize::MAX, 4), None);
        assert_eq!(signed_dim(-1), None);
        assert_eq!(signed_dim(0), Some(0));
        assert_eq!(ort_dim(17), Some(17));
        assert_eq!(raw_f32_elements(12), Some(3));
        assert_eq!(raw_f32_elements(13), None);
    }

    /// A 64-bit `usize` reaches past i64, so ORT's signed dimension is the binding limit: the
    /// largest one converts, one more does not.
    #[cfg(target_pointer_width = "64")]
    #[test]
    fn ort_dimensions_stop_at_i64_max() {
        assert_eq!(ort_dim(i64::MAX as usize), Some(i64::MAX));
        assert_eq!(ort_dim(i64::MAX as usize + 1), None);
    }

    /// A 32-bit `usize` is narrower than an ONNX dimension, so the narrowing guard binds: the
    /// largest `usize` converts, one more does not.
    #[cfg(target_pointer_width = "32")]
    #[test]
    fn signed_dimensions_stop_at_the_largest_usize() {
        assert_eq!(signed_dim(usize::MAX as i64), Some(usize::MAX));
        assert_eq!(signed_dim(usize::MAX as i64 + 1), None);
    }
}
