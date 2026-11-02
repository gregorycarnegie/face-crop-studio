//! ASCII hex decoding, shared with the verifier.

// shortcut: Aeneas lacks Option's ? model, remove this exception when supported.
#![allow(clippy::question_mark)]

// shortcut: Aeneas lacks RangeInclusive::contains's model, use it when supported.
#[allow(clippy::manual_range_contains)]
pub(super) fn digit(byte: u8) -> Option<u8> {
    if byte >= b'0' && byte <= b'9' {
        Some(byte - b'0')
    } else if byte >= b'A' && byte <= b'F' {
        Some(byte - b'A' + 10)
    } else if byte >= b'a' && byte <= b'f' {
        Some(byte - b'a' + 10)
    } else {
        None
    }
}

pub(super) fn pair(first: u8, second: u8) -> Option<u8> {
    // Preserve from_str_radix's acceptance of a leading plus in a two-byte field.
    if first == b'+' {
        return digit(second);
    }
    let Some(hi) = digit(first) else {
        return None;
    };
    let Some(lo) = digit(second) else {
        return None;
    };
    Some(hi * 16 + lo)
}

pub(super) fn short(byte: u8) -> Option<u8> {
    let Some(d) = digit(byte) else {
        return None;
    };
    Some(d * 17)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_byte_pair_matches_the_standard_parser() {
        for a in 0..=u8::MAX {
            for b in 0..=u8::MAX {
                let bytes = [a, b];
                let expected = std::str::from_utf8(&bytes)
                    .ok()
                    .and_then(|s| u8::from_str_radix(s, 16).ok());
                assert_eq!(pair(a, b), expected, "{a} {b}");
            }
        }
    }
}
