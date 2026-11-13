//! Writes into uninitialized CPU output rows, checked by tools/verus.

use vstd::prelude::*;
#[cfg(verus_only)]
use vstd::std_specs::maybe_uninit::MaybeUninitAdditionalSpecFns;

verus! {

use std::mem::MaybeUninit;

/// Duplicate each input value into two output slots.
/// The caller supplies an output row exactly twice the input length.
pub(super) fn write_upsampled_row<T: Copy>(input: &[T], output: &mut [MaybeUninit<T>])
    requires output.len() == 2 * input.len(),
    ensures
        final(output).len() == old(output).len(),
        forall|i: int| 0 <= i < final(output).len() ==>
            #[trigger] final(output)[i].as_option() == Some(input[i / 2]),
{
    let mut x = 0;
    while x < output.len()
        invariant
            output.len() == 2 * input.len(),
            0 <= x <= output.len(),
            forall|i: int| 0 <= i < x ==>
                #[trigger] output[i].as_option() == Some(input[i / 2]),
        decreases output.len() - x,
    {
        output[x].write(input[x / 2]);
        x += 1;
    }
}

/// Write `value(i)` into every slot `i`, whatever lengths the caller's inputs have.
pub(super) fn write_each<T, F: Fn(usize) -> T>(output: &mut [MaybeUninit<T>], value: F)
    requires forall|i: usize| i < output.len() ==> value.requires((i,)),
    ensures
        final(output).len() == old(output).len(),
        forall|i: int| 0 <= i < final(output).len() ==> {
            &&& #[trigger] final(output)[i].as_option() is Some
            &&& value.ensures((i as usize,), final(output)[i].as_option()->Some_0)
        },
{
    let mut e = 0;
    while e < output.len()
        invariant
            output.len() == old(output).len(),
            forall|i: usize| i < output.len() ==> value.requires((i,)),
            0 <= e <= output.len(),
            forall|i: int| 0 <= i < e ==> {
                &&& #[trigger] output[i].as_option() is Some
                &&& value.ensures((i as usize,), output[i].as_option()->Some_0)
            },
        decreases output.len() - e,
    {
        output[e].write(value(e));
        e += 1;
    }
}

/// Write one depthwise output row: single pixels before `lo`, 4-pixel tiles from `lo` while
/// a whole tile fits before `hi`, then single pixels to the end. `quad` is only called with
/// its whole tile inside `lo..hi`, the interior whose taps need no bounds checks.
pub(super) fn write_tiled_row<T: Copy, S: Fn(usize) -> T, Q: Fn(usize) -> [T; 4]>(
    output: &mut [MaybeUninit<T>],
    lo: usize,
    hi: usize,
    single: S,
    quad: Q,
)
    requires
        lo <= hi <= output.len(),
        forall|x: usize| x < output.len() ==> single.requires((x,)),
        forall|x: usize| lo <= x && x + 4 <= hi ==> quad.requires((x,)),
    ensures
        final(output).len() == old(output).len(),
        forall|i: int| 0 <= i < final(output).len() ==>
            #[trigger] final(output)[i].as_option() is Some,
{
    let mut ox = 0;
    while ox < lo
        invariant
            output.len() == old(output).len(),
            lo <= hi <= output.len(),
            forall|x: usize| x < output.len() ==> single.requires((x,)),
            ox <= lo,
            forall|i: int| 0 <= i < ox ==> #[trigger] output[i].as_option() is Some,
        decreases lo - ox,
    {
        output[ox].write(single(ox));
        ox += 1;
    }
    while 4 <= hi - ox
        invariant
            output.len() == old(output).len(),
            lo <= ox <= hi <= output.len(),
            forall|x: usize| x < output.len() ==> single.requires((x,)),
            forall|x: usize| lo <= x && x + 4 <= hi ==> quad.requires((x,)),
            forall|i: int| 0 <= i < ox ==> #[trigger] output[i].as_option() is Some,
        decreases hi - ox,
    {
        let tile = quad(ox);
        output[ox].write(tile[0]);
        output[ox + 1].write(tile[1]);
        output[ox + 2].write(tile[2]);
        output[ox + 3].write(tile[3]);
        ox += 4;
    }
    while ox < output.len()
        invariant
            output.len() == old(output).len(),
            ox <= output.len(),
            forall|x: usize| x < output.len() ==> single.requires((x,)),
            forall|i: int| 0 <= i < ox ==> #[trigger] output[i].as_option() is Some,
        decreases output.len() - ox,
    {
        output[ox].write(single(ox));
        ox += 1;
    }
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

    #[test]
    fn miri_cpu_buffers_indexed_and_tiled_rows_are_initialized() {
        for (len, lo, hi) in [(0, 0, 0), (1, 0, 1), (7, 1, 6), (9, 2, 9), (12, 0, 12)] {
            let mut output = vec![MaybeUninit::uninit(); len];
            write_each(&mut output, |i| 3 * i);
            // SAFETY: the indexed writer initializes every output slot.
            assert!((0..len).all(|i| unsafe { output[i].assume_init() } == 3 * i));

            let mut output = vec![MaybeUninit::uninit(); len];
            write_tiled_row(
                &mut output,
                lo,
                hi,
                |x| x,
                |x| {
                    assert!(lo <= x && x + 4 <= hi, "quad called outside the interior");
                    [x, x + 1, x + 2, x + 3]
                },
            );
            // SAFETY: the tiled writer initializes every output slot.
            assert!((0..len).all(|i| unsafe { output[i].assume_init() } == i));
        }
    }
}
