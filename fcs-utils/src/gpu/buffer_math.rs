//! Overflow-safe budget checks and stable minimum-buffer selection.

pub(super) fn budget_fits(current: u64, size: u64, limit: u64) -> bool {
    current <= limit && size <= limit - current
}

/// Usage after reserving `size`, or `None` when the budget refuses it.
pub(super) fn reserve(current: u64, size: u64, limit: u64) -> Option<u64> {
    if budget_fits(current, size, limit) {
        Some(current + size)
    } else {
        None
    }
}

/// Usage after releasing `freed`; saturates so a foreign or repeated release cannot wrap.
pub(super) fn release(current: u64, freed: u64) -> u64 {
    current.saturating_sub(freed)
}

pub(super) fn choose_fit(
    best: Option<(usize, u64)>,
    index: usize,
    size: u64,
    requested: u64,
) -> Option<(usize, u64)> {
    if size < requested {
        return best;
    }
    match best {
        Some((_, best_size)) if best_size <= size => best,
        _ => Some((index, size)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_and_best_fit_handle_integer_boundaries() {
        assert!(!budget_fits(u64::MAX, 1, u64::MAX));
        assert!(budget_fits(u64::MAX - 1, 1, u64::MAX));
        assert!(!budget_fits(4, 0, 3));
        assert_eq!(reserve(u64::MAX - 1, 1, u64::MAX), Some(u64::MAX));
        assert_eq!(reserve(u64::MAX, 1, u64::MAX), None);
        assert_eq!(reserve(512, 600, 1024), None);
        assert_eq!(release(3, 5), 0);
        assert_eq!(release(5, 3), 2);
        assert_eq!(choose_fit(None, 0, u64::MAX, 1), Some((0, u64::MAX)));
        assert_eq!(choose_fit(Some((0, 7)), 1, 7, 4), Some((0, 7)));
        assert_eq!(choose_fit(Some((0, 7)), 1, 5, 4), Some((1, 5)));
        assert_eq!(choose_fit(None, 1, 3, 4), None);
    }
}
