//! Checked byte ranges for metadata walkers and length fields.

// shortcut: Aeneas lacks Option's ? model, remove this exception when supported.
#![allow(clippy::question_mark)]

pub(super) fn span(
    cursor: usize,
    header: usize,
    payload: usize,
    trailer: usize,
    length: usize,
) -> Option<(usize, usize, usize)> {
    let Some(start) = cursor.checked_add(header) else {
        return None;
    };
    let Some(end) = start.checked_add(payload) else {
        return None;
    };
    let Some(next) = end.checked_add(trailer) else {
        return None;
    };
    if next > length {
        None
    } else {
        Some((start, end, next))
    }
}

pub(super) fn png_text_layout(keyword: usize, value: usize) -> Option<(usize, usize)> {
    let Some(data) = keyword.checked_add(value) else {
        return None;
    };
    let Some(data) = data.checked_add(1) else {
        return None;
    };
    if data > u32::MAX as usize {
        return None;
    }
    let Some(total) = data.checked_add(12) else {
        return None;
    };
    if total > isize::MAX as usize {
        None
    } else {
        Some((data, total))
    }
}

pub(super) fn jpeg_length(payload: usize) -> Option<usize> {
    let Some(length) = payload.checked_add(2) else {
        return None;
    };
    if length > u16::MAX as usize {
        None
    } else {
        Some(length)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_and_segment_ranges_do_not_wrap_or_accept_truncation() {
        assert_eq!(span(8, 8, 13, 4, 33), Some((16, 29, 33)));
        assert_eq!(span(8, 8, 13, 4, 32), None);
        assert_eq!(span(usize::MAX - 1, 8, 0, 0, usize::MAX), None);
        assert_eq!(span(8, 8, usize::MAX, 4, usize::MAX), None);
        assert_eq!(jpeg_length(65533), Some(65535));
        assert_eq!(jpeg_length(65534), None);
        assert_eq!(jpeg_length(usize::MAX), None);
        assert_eq!(png_text_layout(3, 4), Some((8, 20)));
        assert_eq!(png_text_layout(79, usize::MAX), None);
    }

    /// On 64-bit the PNG length field is the binding limit: a chunk holds u32::MAX bytes of
    /// data, and nothing larger.
    #[cfg(target_pointer_width = "64")]
    #[test]
    fn text_chunks_stop_at_the_length_field() {
        let largest = u32::MAX as usize;
        assert_eq!(
            png_text_layout(3, largest - 4),
            Some((largest, largest + 12))
        );
        assert_eq!(png_text_layout(3, largest - 3), None);
    }

    /// On 32-bit isize::MAX is below u32::MAX, so the allocation limit binds first: a chunk
    /// totalling exactly isize::MAX bytes is accepted, one byte more is not.
    #[cfg(target_pointer_width = "32")]
    #[test]
    fn text_chunks_stop_at_the_allocation_limit() {
        let fits = isize::MAX as usize;
        assert_eq!(png_text_layout(3, fits - 16), Some((fits - 12, fits)));
        assert_eq!(png_text_layout(3, fits - 15), None);
    }
}
