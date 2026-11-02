//! Overflow-safe budget checks and stable minimum-buffer selection.

pub(super) fn budget_fits(current: u64, size: u64, limit: u64) -> bool {
    current <= limit && size <= limit - current
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
        assert_eq!(choose_fit(None, 0, u64::MAX, 1), Some((0, u64::MAX)));
        assert_eq!(choose_fit(Some((0, 7)), 1, 7, 4), Some((0, 7)));
        assert_eq!(choose_fit(Some((0, 7)), 1, 5, 4), Some((1, 5)));
        assert_eq!(choose_fit(None, 1, 3, 4), None);
    }
}
