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


/// Write `values[f][p]` into column `x + p` of row `f`, for every row and every `p < P`.
#[inline(always)]
fn write_tile<T: Copy, const F: usize, const P: usize>(
    rows: &mut [&mut [MaybeUninit<T>]],
    x: usize,
    values: [[T; P]; F],
)
    requires
        old(rows).len() == F,
        forall|f: int| 0 <= f < F ==> #[trigger] old(rows)[f].len() == old(rows)[0].len(),
        x + P <= old(rows)[0].len(),
    ensures
        final(rows).len() == F,
        forall|f: int| 0 <= f < F ==> #[trigger] final(rows)[f].len() == old(rows)[0].len(),
        forall|f: int, i: int| 0 <= f < F && 0 <= i < old(rows)[0].len()
            && (x <= i < x + P || old(rows)[f][i].as_option() is Some) ==>
            #[trigger] final(rows)[f][i].as_option() is Some,
{
    let ghost len = old(rows)[0].len();
    let mut f = 0;
    while f < F
        invariant
            rows.len() == F,
            forall|g: int| 0 <= g < F ==> #[trigger] rows[g].len() == len,
            len == old(rows)[0].len(),
            x + P <= len,
            f <= F,
            forall|g: int, i: int| 0 <= g < F && 0 <= i < len
                && ((g < f && x <= i < x + P) || old(rows)[g][i].as_option() is Some) ==>
                #[trigger] rows[g][i].as_option() is Some,
        decreases F - f,
    {
        let mut p = 0;
        while p < P
            invariant
                rows.len() == F,
                forall|g: int| 0 <= g < F ==> #[trigger] rows[g].len() == len,
                len == old(rows)[0].len(),
                x + P <= len,
                f < F,
                p <= P,
                forall|g: int, i: int| 0 <= g < F && 0 <= i < len
                    && ((g < f && x <= i < x + P) || (g == f && x <= i < x + p)
                        || old(rows)[g][i].as_option() is Some) ==>
                    #[trigger] rows[g][i].as_option() is Some,
            decreases P - p,
        {
            assert(rows[f as int].len() == len);
            rows[f][x + p].write(values[f][p]);
            p += 1;
        }
        f += 1;
    }
}

/// Write `F` dense convolution output row segments of one length together: `border(x)` single columns before `lo` and
/// from `hi`, `P`-column tiles from `lo` while a whole tile fits before `hi`, then `inner(x)`
/// single columns up to `hi`. `tile` and `inner` are only called inside `lo..hi`, the interior
/// whose taps need no bounds checks. Tile value `[f][p]` goes to row `f`, column `x + p`.
#[inline(always)]
pub(super) fn write_tiled_rows<
    T: Copy,
    const F: usize,
    const P: usize,
    B: Fn(usize) -> [[T; 1]; F],
    I: Fn(usize) -> [[T; 1]; F],
    Q: Fn(usize) -> [[T; P]; F],
>(rows: &mut [&mut [MaybeUninit<T>]], lo: usize, hi: usize, border: B, inner: I, tile: Q)
    requires
        0 < F,
        0 < P,
        old(rows).len() == F,
        forall|f: int| 0 <= f < F ==> #[trigger] old(rows)[f].len() == old(rows)[0].len(),
        lo <= hi <= old(rows)[0].len(),
        forall|x: usize| x < old(rows)[0].len() ==> border.requires((x,)),
        forall|x: usize| lo <= x < hi ==> inner.requires((x,)),
        forall|x: usize| lo <= x && x + P <= hi ==> tile.requires((x,)),
    ensures
        final(rows).len() == F,
        forall|f: int| 0 <= f < F ==> #[trigger] final(rows)[f].len() == old(rows)[0].len(),
        forall|f: int, i: int| 0 <= f < F && 0 <= i < old(rows)[0].len() ==>
            #[trigger] final(rows)[f][i].as_option() is Some,
{
    let len = rows[0].len();
    let mut ox = 0;
    while ox < lo
        invariant
            0 < F,
            rows.len() == F,
            forall|g: int| 0 <= g < F ==> #[trigger] rows[g].len() == len,
            len == old(rows)[0].len(),
            lo <= hi <= len,
            forall|x: usize| x < len ==> border.requires((x,)),
            ox <= lo,
            forall|g: int, i: int| 0 <= g < F && 0 <= i < ox ==>
                #[trigger] rows[g][i].as_option() is Some,
        decreases lo - ox,
    {
        write_tile(rows, ox, border(ox));
        ox += 1;
    }
    while P <= hi - ox
        invariant
            0 < F,
            rows.len() == F,
            forall|g: int| 0 <= g < F ==> #[trigger] rows[g].len() == len,
            len == old(rows)[0].len(),
            lo <= ox <= hi <= len,
            0 < P,
            forall|x: usize| lo <= x && x + P <= hi ==> tile.requires((x,)),
            forall|g: int, i: int| 0 <= g < F && 0 <= i < ox ==>
                #[trigger] rows[g][i].as_option() is Some,
        decreases hi - ox,
    {
        write_tile(rows, ox, tile(ox));
        ox += P;
    }
    while ox < hi
        invariant
            0 < F,
            rows.len() == F,
            forall|g: int| 0 <= g < F ==> #[trigger] rows[g].len() == len,
            len == old(rows)[0].len(),
            lo <= ox <= hi <= len,
            forall|x: usize| lo <= x < hi ==> inner.requires((x,)),
            forall|g: int, i: int| 0 <= g < F && 0 <= i < ox ==>
                #[trigger] rows[g][i].as_option() is Some,
        decreases hi - ox,
    {
        write_tile(rows, ox, inner(ox));
        ox += 1;
    }
    while ox < len
        invariant
            0 < F,
            rows.len() == F,
            forall|g: int| 0 <= g < F ==> #[trigger] rows[g].len() == len,
            len == old(rows)[0].len(),
            ox <= len,
            forall|x: usize| x < len ==> border.requires((x,)),
            forall|g: int, i: int| 0 <= g < F && 0 <= i < ox ==>
                #[trigger] rows[g][i].as_option() is Some,
        decreases len - ox,
    {
        write_tile(rows, ox, border(ox));
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

    #[test]
    fn miri_cpu_buffers_dense_row_segments_are_initialized() {
        for (len, lo, hi) in [(1, 0, 1), (7, 1, 6), (9, 2, 9), (12, 0, 12), (5, 3, 3)] {
            let (mut a, mut b) = (
                vec![MaybeUninit::uninit(); len],
                vec![MaybeUninit::uninit(); len],
            );
            let mut rows = [&mut a[..], &mut b[..]];
            write_tiled_rows(
                &mut rows,
                lo,
                hi,
                |x| [[x], [100 + x]],
                |x| {
                    assert!(lo <= x && x < hi, "inner called outside the interior");
                    [[x], [100 + x]]
                },
                |x| {
                    assert!(lo <= x && x + 3 <= hi, "tile called outside the interior");
                    [[x, x + 1, x + 2], [100 + x, 101 + x, 102 + x]]
                },
            );
            // SAFETY: the dense row writer initializes every slot of every segment.
            assert!((0..len).all(|i| unsafe { a[i].assume_init() } == i));
            assert!((0..len).all(|i| unsafe { b[i].assume_init() } == 100 + i));
        }
    }
}
