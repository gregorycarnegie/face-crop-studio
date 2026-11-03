//! Contiguous channel ranges for merged output heads.

// shortcut: Aeneas lacks Option's ? model, remove this exception when supported.
#[allow(clippy::question_mark, clippy::type_complexity)]
pub(super) fn ranges(counts: &[usize]) -> Option<(Vec<(usize, usize)>, usize)> {
    let mut ranges = Vec::new();
    let mut first = 0usize;
    let mut i = 0;
    while i < counts.len() {
        let count = counts[i];
        let Some(end) = first.checked_add(count) else {
            return None;
        };
        ranges.push((first, count));
        first = end;
        i += 1;
    }
    Some((ranges, first))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_are_contiguous_even_for_empty_heads_and_overflow_is_rejected() {
        assert_eq!(
            ranges(&[0, 2, 0, 3]),
            Some((vec![(0, 0), (0, 2), (2, 0), (2, 3)], 5))
        );
        assert_eq!(ranges(&[usize::MAX, 1]), None);
        assert_eq!(ranges(&[]), Some((vec![], 0)));
    }
}
