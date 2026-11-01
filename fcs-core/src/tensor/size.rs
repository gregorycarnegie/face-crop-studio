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
    }
}
