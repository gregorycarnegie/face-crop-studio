//! Writes into uninitialized CPU output rows, checked by tools/verus.

use std::mem::MaybeUninit;

/// Duplicate each input value into two output slots.
/// The caller supplies an output row exactly twice the input length.
pub(super) fn write_upsampled_row<T: Copy>(input: &[T], output: &mut [MaybeUninit<T>]) {
    let mut x = 0;
    while x < output.len() {
        output[x].write(input[x / 2]);
        x += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn miri_cpu_buffers_upsample_rows_are_initialized() {
        for len in [0, 1, 3, 49] {
            let input: Vec<_> = (0..len).collect();
            let mut output = vec![MaybeUninit::uninit(); 2 * len];
            write_upsampled_row(&input, &mut output);
            for (x, value) in output.iter().enumerate() {
                // SAFETY: the row writer initializes every output slot.
                assert_eq!(unsafe { value.assume_init() }, input[x / 2]);
            }
        }
    }
}
