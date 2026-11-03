//! Checked host-side dimensions for shaders with u32 indices.

// shortcut: Aeneas lacks Option's ? and integer rounding models, remove when supported.
#![allow(
    clippy::question_mark,
    clippy::manual_is_multiple_of,
    clippy::manual_div_ceil
)]

pub(crate) fn output_dim(size: u32, pad: u32, kernel: u32, stride: u32) -> Option<u32> {
    if kernel == 0 || stride == 0 {
        return None;
    }
    let Some(padding) = pad.checked_mul(2) else {
        return None;
    };
    let Some(padded) = size.checked_add(padding) else {
        return None;
    };
    let Some(numerator) = padded.checked_sub(kernel) else {
        return None;
    };
    (numerator / stride).checked_add(1)
}

/// Number of workgroups, with divisor > 0.
pub(crate) fn ceil_div(value: u32, divisor: u32) -> u32 {
    value / divisor + if value % divisor == 0 { 0 } else { 1 }
}

pub(crate) fn shader_dim(value: usize) -> Option<u32> {
    if value > u32::MAX as usize {
        None
    } else {
        Some(value as u32)
    }
}

pub(crate) fn grouped_channels(input: u32, output: u32, groups: u32) -> Option<(u32, u32)> {
    if groups == 0 {
        return None;
    }
    if input % groups != 0 || output % groups != 0 {
        return None;
    }
    Some((input / groups, output / groups))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_and_dispatch_reject_overflow_and_cover_tails() {
        assert_eq!(output_dim(1, u32::MAX, 1, 1), None);
        assert_eq!(output_dim(u32::MAX, 1, 1, 1), None);
        assert_eq!(output_dim(1, 0, 3, 1), None);
        // Either zero alone is rejected, not only both together.
        assert_eq!(output_dim(4, 0, 0, 1), None);
        assert_eq!(output_dim(4, 0, 1, 0), None);
        assert_eq!(shader_dim(u32::MAX as usize), Some(u32::MAX));
        // Only a 64-bit usize can name a size past u32::MAX.
        #[cfg(target_pointer_width = "64")]
        assert_eq!(shader_dim(u32::MAX as usize + 1), None);
        assert_eq!(output_dim(9, 1, 3, 2), Some(5));
        assert_eq!(grouped_channels(12, 8, 4), Some((3, 2)));
        assert_eq!(grouped_channels(12, 8, 0), None);
        assert_eq!(grouped_channels(12, 8, 3), None);
        for value in [0u32, 1, 7, 8, 9, 31, u32::MAX] {
            for divisor in [1u32, 2, 8, 32, u32::MAX] {
                let n = ceil_div(value, divisor);
                assert!(u64::from(value) <= u64::from(n) * u64::from(divisor));
                assert!(n == 0 || u64::from(n - 1) * u64::from(divisor) < u64::from(value));
            }
        }
    }
}
