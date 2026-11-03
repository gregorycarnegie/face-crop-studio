//! Last-reader schedules over graph slot indices, independent of tensor storage.

pub(super) fn readers(steps: &[(usize, usize)]) -> Vec<usize> {
    // Zero means unread; a reader is stored as its index plus one.
    let mut last = vec![0; steps.len() + 1];
    let mut index = 0;
    while index < steps.len() {
        let (a, b) = steps[index];
        last[a] = index + 1;
        last[b] = index + 1;
        index += 1;
    }
    last
}

pub(super) fn retained(outputs: &[usize], slot: usize) -> bool {
    let mut i = 0;
    while i < outputs.len() {
        if outputs[i] == slot {
            return true;
        }
        i += 1;
    }
    false
}

pub(super) fn empty_schedule(steps: usize) -> Vec<Vec<usize>> {
    // Explicit initialization lets the source proof track each release row.
    let mut frees = Vec::with_capacity(steps);
    let mut i = 0;
    while i < steps {
        frees.push(Vec::new());
        i += 1;
    }
    frees
}

pub(super) fn release_schedule(last: &[usize], outputs: &[usize], steps: usize) -> Vec<Vec<usize>> {
    let mut frees = empty_schedule(steps);
    let mut slot = 1;
    while slot < last.len() {
        let at = last[slot];
        if at > 0 && !retained(outputs, slot) {
            frees[at - 1].push(slot);
        }
        slot += 1;
    }
    frees
}

/// Release each non-output slot after its final reader; slot zero belongs to the caller.
pub(super) fn last_reads(steps: &[(usize, usize)], outputs: &[usize]) -> Vec<Vec<usize>> {
    release_schedule(&readers(steps), outputs, steps.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedules_only_the_last_reader_and_retains_input_and_outputs() {
        let reads = [(0, 0), (1, 0), (1, 2), (3, 3)];
        assert_eq!(
            last_reads(&reads, &[3, 4]),
            vec![vec![], vec![], vec![1, 2], vec![]]
        );
        assert!(last_reads(&[], &[]).is_empty());
        assert_eq!(last_reads(&[(0, 0), (1, 1)], &[]), vec![vec![], vec![1]]);
    }
}
