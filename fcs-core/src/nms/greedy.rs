/// Visit detections best first; each one still kept suppresses every later one it overlaps.
pub(super) fn greedy_suppression<F: Fn(usize, usize) -> bool>(
    len: usize,
    overlaps: F,
) -> Vec<bool> {
    let mut suppressed = vec![false; len];
    let mut i = 0;
    while i < len {
        if !suppressed[i] {
            let mut j = i + 1;
            while j < len {
                if !suppressed[j] && overlaps(i, j) {
                    suppressed[j] = true;
                }
                j += 1;
            }
        }
        i += 1;
    }
    suppressed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_suppressed_detection_cannot_suppress_others() {
        // 0 overlaps 1, 1 overlaps 2, 0 and 2 are apart: 1 goes, so 2 survives.
        let overlaps = |i: usize, j: usize| j == i + 1;
        assert_eq!(greedy_suppression(3, overlaps), [false, true, false]);
        assert_eq!(greedy_suppression(0, overlaps), Vec::<bool>::new());
    }
}
