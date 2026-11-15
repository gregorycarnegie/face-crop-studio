//! Writes into uninitialized channel-plane rows, checked by tools/verus.

use vstd::prelude::*;
#[cfg(verus_only)]
use vstd::std_specs::maybe_uninit::MaybeUninitAdditionalSpecFns;

verus! {

use std::mem::MaybeUninit;

/// Write pixel `x` of one row into three channel planes at once: `value(x)[0]` into `first`,
/// `[1]` into `second`, `[2]` into `third`, for every `x` of rows that share one length.
#[inline(always)]
pub fn write_planes<T: Copy, F: Fn(usize) -> [T; 3]>(
    first: &mut [MaybeUninit<T>],
    second: &mut [MaybeUninit<T>],
    third: &mut [MaybeUninit<T>],
    value: F,
)
    requires
        old(second).len() == old(first).len(),
        old(third).len() == old(first).len(),
        forall|x: usize| x < old(first).len() ==> value.requires((x,)),
    ensures
        final(first).len() == old(first).len(),
        final(second).len() == old(first).len(),
        final(third).len() == old(first).len(),
        forall|x: int| 0 <= x < old(first).len() ==> {
            &&& #[trigger] final(first)[x].as_option() is Some
            &&& final(second)[x].as_option() is Some
            &&& final(third)[x].as_option() is Some
            &&& exists|v: [T; 3]| value.ensures((x as usize,), v)
                && final(first)[x].as_option()->Some_0 == v[0]
                && final(second)[x].as_option()->Some_0 == v[1]
                && final(third)[x].as_option()->Some_0 == v[2]
        },
{
    let len = first.len();
    let mut x = 0;
    while x < len
        invariant
            len == old(first).len(),
            first.len() == len,
            second.len() == len,
            third.len() == len,
            forall|i: usize| i < len ==> value.requires((i,)),
            x <= len,
            forall|i: int| 0 <= i < x ==> {
                &&& #[trigger] first[i].as_option() is Some
                &&& second[i].as_option() is Some
                &&& third[i].as_option() is Some
                &&& exists|v: [T; 3]| value.ensures((i as usize,), v)
                    && first[i].as_option()->Some_0 == v[0]
                    && second[i].as_option()->Some_0 == v[1]
                    && third[i].as_option()->Some_0 == v[2]
            },
        decreases len - x,
    {
        let v = value(x);
        first[x].write(v[0]);
        second[x].write(v[1]);
        third[x].write(v[2]);
        x += 1;
    }
}

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_slot_of_all_three_rows_is_written() {
        for len in [0, 1, 7] {
            let mut planes = vec![MaybeUninit::uninit(); 3 * len];
            let (a, rest) = planes.split_at_mut(len);
            let (b, c) = rest.split_at_mut(len);
            write_planes(a, b, c, |x| [x, 10 + x, 20 + x]);
            // SAFETY: the plane writer initializes every slot of every row.
            let got: Vec<usize> = planes.iter().map(|v| unsafe { v.assume_init() }).collect();
            let want: Vec<usize> = (0..3)
                .flat_map(|c| (0..len).map(move |x| 10 * c + x))
                .collect();
            assert_eq!(got, want);
        }
    }
}
