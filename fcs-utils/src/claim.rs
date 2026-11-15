//! The claiming step of a work queue whose workers share one job counter.

/// The job a worker takes when the shared counter's `fetch_add(1)` hands it `ticket`:
/// `Some(ticket)` while jobs remain, `None` once every job is taken. A worker stops at its
/// first `None`. tools/verification (ClaimProofs) proves that, however the workers'
/// `fetch_add`s interleave, every job is then claimed by exactly one ticket.
pub fn claim(ticket: usize, jobs: usize) -> Option<usize> {
    if ticket < jobs { Some(ticket) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tickets_past_the_last_job_claim_nothing() {
        assert_eq!(claim(0, 2), Some(0));
        assert_eq!(claim(1, 2), Some(1));
        assert_eq!(claim(2, 2), None);
        assert_eq!(claim(usize::MAX, usize::MAX), None);
        assert_eq!(claim(0, 0), None);
    }
}
