//! Basic color utilities shared across CLI and GUI surfaces.

use serde::{Deserialize, Serialize};

/// Simple RGBA color stored in 8-bit channels.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct RgbaColor {
    pub red: u8,
    pub green: u8,
    pub blue: u8,
    pub alpha: u8,
}

impl RgbaColor {
    /// Constructs an opaque RGB color.
    pub const fn opaque(red: u8, green: u8, blue: u8) -> Self {
        Self {
            red,
            green,
            blue,
            alpha: 255,
        }
    }

    /// Returns the color as normalized HSV tuple (hue in degrees, saturation/value 0.0..1.0).
    pub fn to_hsv(self) -> (f32, f32, f32) {
        rgb_to_hsv(self.red, self.green, self.blue)
    }

    /// Builds a color from HSV values (hue in degrees, saturation/value 0.0..1.0).
    pub fn from_hsv(h: f32, s: f32, v: f32) -> Self {
        let (r, g, b) = hsv_to_rgb(h, s, v);
        Self::opaque(r, g, b)
    }
}

impl Default for RgbaColor {
    fn default() -> Self {
        Self::opaque(0, 0, 0)
    }
}

/// Convert RGB channels (0-255) to HSV (hue in degrees 0-360, saturation/value 0-1).
pub fn rgb_to_hsv(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let rf = r as f32 / 255.0;
    let gf = g as f32 / 255.0;
    let bf = b as f32 / 255.0;

    let max = rf.max(gf).max(bf);
    let min = rf.min(gf).min(bf);
    let delta = max - min;

    let hue = if delta.abs() < f32::EPSILON {
        0.0
    } else if (max - rf).abs() < f32::EPSILON {
        60.0 * (((gf - bf) / delta) % 6.0)
    } else if (max - gf).abs() < f32::EPSILON {
        60.0 * (((bf - rf) / delta) + 2.0)
    } else {
        60.0 * (((rf - gf) / delta) + 4.0)
    };

    let hue = if hue < 0.0 { hue + 360.0 } else { hue };
    let saturation = if max.abs() < f32::EPSILON {
        0.0
    } else {
        delta / max
    };
    (hue, saturation, max)
}

/// Convert HSV (hue in degrees, saturation/value 0-1) to RGB channels (0-255).
pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    if s <= 0.0 {
        let val = (v * 255.0) as u8;
        return (val, val, val);
    }

    let hue = if h.is_nan() { 0.0 } else { h.rem_euclid(360.0) };
    let c = v * s;
    let x = c * (1.0 - ((hue / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r1, g1, b1) = match hue {
        h if h < 60.0 => (c, x, 0.0),
        h if h < 120.0 => (x, c, 0.0),
        h if h < 180.0 => (0.0, c, x),
        h if h < 240.0 => (0.0, x, c),
        h if h < 300.0 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    let to_byte = |value: f32| -> u8 { ((value + m) * 255.0) as u8 };

    (to_byte(r1), to_byte(g1), to_byte(b1))
}

/// Parse a hexadecimal color string. Accepts `#RGB`, `#RRGGBB`, `#RRGGBBAA`, with or without `#`.
pub fn parse_hex_color(input: &str) -> Option<RgbaColor> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return None;
    }

    let mut hex = trimmed;
    if let Some(stripped) = hex.strip_prefix('#') {
        hex = stripped;
    } else if let Some(stripped) = hex.strip_prefix("0x") {
        hex = stripped;
    }
    let hex = hex.replace('_', "");
    match hex.len() {
        3 => Some(RgbaColor::opaque(
            replicate_nibble(hex.get(0..1)?)?,
            replicate_nibble(hex.get(1..2)?)?,
            replicate_nibble(hex.get(2..3)?)?,
        )),
        4 => Some(RgbaColor {
            red: replicate_nibble(hex.get(0..1)?)?,
            green: replicate_nibble(hex.get(1..2)?)?,
            blue: replicate_nibble(hex.get(2..3)?)?,
            alpha: replicate_nibble(hex.get(3..4)?)?,
        }),
        6 => Some(RgbaColor {
            red: parse_byte(hex.get(0..2)?)?,
            green: parse_byte(hex.get(2..4)?)?,
            blue: parse_byte(hex.get(4..6)?)?,
            alpha: 255,
        }),
        8 => Some(RgbaColor {
            red: parse_byte(hex.get(0..2)?)?,
            green: parse_byte(hex.get(2..4)?)?,
            blue: parse_byte(hex.get(4..6)?)?,
            alpha: parse_byte(hex.get(6..8)?)?,
        }),
        _ => None,
    }
}

fn parse_byte(slice: &str) -> Option<u8> {
    u8::from_str_radix(slice, 16).ok()
}

fn replicate_nibble(slice: &str) -> Option<u8> {
    let nib = u8::from_str_radix(slice, 16).ok()?;
    Some((nib << 4) | nib)
}

/// Convert RGB channels (0-255) to HSL (hue in degrees 0-360, saturation 0-1, lightness 0-1).
pub fn rgb_to_hsl(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let rf = r as f32 / 255.0;
    let gf = g as f32 / 255.0;
    let bf = b as f32 / 255.0;

    let max = rf.max(gf).max(bf);
    let min = rf.min(gf).min(bf);
    let delta = max - min;

    let hue = if delta.abs() < f32::EPSILON {
        0.0
    } else if (max - rf).abs() < f32::EPSILON {
        60.0 * (((gf - bf) / delta) % 6.0)
    } else if (max - gf).abs() < f32::EPSILON {
        60.0 * (((bf - rf) / delta) + 2.0)
    } else {
        60.0 * (((rf - gf) / delta) + 4.0)
    };

    let hue = if hue < 0.0 { hue + 360.0 } else { hue };

    let lightness = (max + min) * 0.5;

    let saturation = if delta.abs() < f32::EPSILON {
        0.0
    } else {
        delta / (1.0 - (lightness.mul_add(2.0, -1.0)).abs())
    };

    (hue, saturation, lightness)
}

/// Convert HSL (hue in degrees, saturation 0-1, lightness 0-1) to RGB channels (0-255).
pub fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (u8, u8, u8) {
    let c = (1.0 - (l.mul_add(2.0, -1.0)).abs()) * s;
    let hue = if h.is_nan() { 0.0 } else { h.rem_euclid(360.0) };
    let x = c * (1.0 - ((hue / 60.0) % 2.0 - 1.0).abs());
    let m = l - c * 0.5;

    let (r1, g1, b1) = match hue {
        h if h < 60.0 => (c, x, 0.0),
        h if h < 120.0 => (x, c, 0.0),
        h if h < 180.0 => (0.0, c, x),
        h if h < 240.0 => (0.0, x, c),
        h if h < 300.0 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    let to_byte = |value: f32| -> u8 { ((value + m) * 255.0) as u8 };

    (to_byte(r1), to_byte(g1), to_byte(b1))
}

/// Convert RGB channels (0-255) to CMYK (0-1 for all channels).
pub fn rgb_to_cmyk(r: u8, g: u8, b: u8) -> (f32, f32, f32, f32) {
    let rf = r as f32 / 255.0;
    let gf = g as f32 / 255.0;
    let bf = b as f32 / 255.0;

    let k = 1.0 - rf.max(gf).max(bf);
    if (1.0 - k).abs() < f32::EPSILON {
        return (0.0, 0.0, 0.0, 1.0);
    }

    let rgb_channel_to_cymk = |value: f32| -> f32 { (1.0 - value - k) / (1.0 - k) };

    (
        rgb_channel_to_cymk(rf), // cyan
        rgb_channel_to_cymk(gf), // magenta
        rgb_channel_to_cymk(bf), // yellow
        k,                       // black
    )
}

/// Convert CMYK (0-1 for all channels) to RGB channels (0-255).
pub fn cmyk_to_rgb(c: f32, m: f32, y: f32, k: f32) -> (u8, u8, u8) {
    let to_rgb_channel = |value: f32| -> u8 { (255.0 * (1.0 - value) * (1.0 - k)) as u8 };
    (
        to_rgb_channel(c), // red
        to_rgb_channel(m), // green
        to_rgb_channel(y), // blue
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! small_diff {
        ($expected:ident, $got:ident) => {
            assert!(
                ($expected as i16 - $got as i16).abs() <= 1,
                "{} mismatch: got {}, expected {}",
                stringify!($expected).to_uppercase(),
                $got,
                $expected
            );
        };
    }

    fn colspace_assertion(
        r: u8,
        g: u8,
        b: u8,
        colspace_val: f32,
        exp_colspace_val: f32,
        colspace_name: &str,
        tolerance: f32,
    ) {
        assert!(
            (colspace_val - exp_colspace_val).abs() < tolerance,
            "{} mismatch for ({}, {}, {}): got {}, expected {}",
            colspace_name,
            r,
            g,
            b,
            colspace_val,
            exp_colspace_val
        );
    }

    // ------------------------------------------------------------------
    // Golden values.
    //
    // The round-trip tests below cover only saturated primaries, secondaries
    // and greys. For every one of those, lightness is exactly 0.5 (or the
    // colour is achromatic) and k is exactly 0 or 1 — so the `1 - |2L - 1|`
    // denominator is always 1, the `x` interpolation term is always 0 or c,
    // and the general CMYK path never runs. Converting a value and converting
    // it back also hides any error the inverse repeats. These pin intermediate
    // hues, off-centre lightness, and mid-range k against reference values.

    #[track_caller]
    fn close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-3,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn rgb_to_hsv_intermediate_hues() {
        // Orange: red is max, green sits partway, so hue lands mid-sextant.
        let (h, s, v) = rgb_to_hsv(255, 128, 0);
        close(h, 30.117647);
        close(s, 1.0);
        close(v, 1.0);

        // Green max — the second branch, +2 sextants.
        let (h, s, v) = rgb_to_hsv(0, 255, 128);
        close(h, 150.11765);
        close(s, 1.0);
        close(v, 1.0);

        // Blue max — the third branch, +4 sextants, with s and v both partial.
        let (h, s, v) = rgb_to_hsv(64, 128, 192);
        close(h, 210.0);
        close(s, 2.0 / 3.0);
        close(v, 0.7529412);
    }

    #[test]
    fn rgb_to_hsv_green_max_with_a_partial_delta() {
        // The green-max probe above uses a saturated colour where delta is
        // exactly 1.0, and dividing by one is indistinguishable from
        // multiplying by it or taking a remainder. (32, 200, 100) gives
        // delta = 168/255, so the division actually has to happen:
        // hue = 60 * (68/168 + 2) = 144.286.
        let (h, s, v) = rgb_to_hsv(32, 200, 100);
        close(h, 144.28572);
        close(s, 168.0 / 200.0);
        close(v, 200.0 / 255.0);
    }

    #[test]
    fn rgb_to_hsl_green_max_with_a_partial_delta() {
        // Same input through the HSL path, which has its own copy of the
        // sextant arithmetic. L = 0.4549 also keeps the saturation
        // denominator away from 1.
        let (h, s, l) = rgb_to_hsl(32, 200, 100);
        close(h, 144.28572);
        close(s, 168.0 / 232.0);
        close(l, 232.0 / 510.0);
    }

    #[test]
    fn hsv_to_rgb_covers_every_sextant() {
        // The probes above land on 30, 210 and 330, which leaves three match
        // guards never decided in the affirmative. A hue inside each
        // remaining sextant pins which arm runs.
        assert_eq!(hsv_to_rgb(90.0, 1.0, 1.0), (127, 255, 0), "second sextant");
        assert_eq!(hsv_to_rgb(150.0, 1.0, 1.0), (0, 255, 127), "third sextant");
        assert_eq!(hsv_to_rgb(270.0, 1.0, 1.0), (127, 0, 255), "fifth sextant");
    }

    #[test]
    fn hsl_to_rgb_covers_every_sextant() {
        // At L = 0.5 and S = 1 the chroma is 1 and m is 0, so each sextant
        // maps to a clean primary/partial pair.
        assert_eq!(hsl_to_rgb(90.0, 1.0, 0.5), (127, 255, 0), "second sextant");
        assert_eq!(hsl_to_rgb(150.0, 1.0, 0.5), (0, 255, 127), "third sextant");
        assert_eq!(hsl_to_rgb(270.0, 1.0, 0.5), (127, 0, 255), "fifth sextant");
        // Past 300 the final arm swaps which channel takes the partial value.
        assert_eq!(hsl_to_rgb(330.0, 1.0, 0.5), (255, 0, 127), "sixth sextant");
    }

    #[test]
    fn rgb_to_hsv_wraps_negative_hue_to_the_top_of_the_circle() {
        // gf < bf with red max makes the raw hue negative (-30.1), so the
        // +360 correction has to fire. The existing test only asserts the
        // result is non-negative, which any wrap value satisfies.
        let (h, s, v) = rgb_to_hsv(255, 0, 128);
        close(h, 329.88235);
        close(s, 1.0);
        close(v, 1.0);
    }

    #[test]
    fn rgb_to_hsv_achromatic_inputs() {
        assert_eq!(rgb_to_hsv(0, 0, 0), (0.0, 0.0, 0.0));
        let (h, s, v) = rgb_to_hsv(128, 128, 128);
        close(h, 0.0);
        close(s, 0.0);
        close(v, 0.5019608);
    }

    #[test]
    fn hsv_to_rgb_partial_sextant_values() {
        // Halfway through the first sextant: x = c/2, so the green channel is
        // half of red. With a primary hue x is 0 or c and this term vanishes.
        assert_eq!(hsv_to_rgb(30.0, 1.0, 1.0), (255, 127, 0));
        // Fourth sextant with partial saturation and value: c = 0.4, x = 0.2,
        // m = 0.4 gives (0.4, 0.6, 0.8) -> #6699CC.
        assert_eq!(hsv_to_rgb(210.0, 0.5, 0.8), (102, 153, 204));
        // Sixth sextant, reached by wrapping a negative hue.
        assert_eq!(hsv_to_rgb(-30.0, 1.0, 1.0), (255, 0, 127));
        // A full turn is the same as none.
        assert_eq!(hsv_to_rgb(360.0, 1.0, 1.0), (255, 0, 0));
    }

    #[test]
    fn hsv_to_rgb_degenerate_inputs() {
        // Zero saturation short-circuits to grey without touching the sextants.
        assert_eq!(hsv_to_rgb(123.0, 0.0, 0.5), (127, 127, 127));
        // NaN hue is treated as zero rather than propagating.
        assert_eq!(hsv_to_rgb(f32::NAN, 1.0, 1.0), (255, 0, 0));
    }

    #[test]
    fn rgb_to_hsl_off_centre_lightness() {
        // Dark: L = 0.251, so the saturation denominator is 0.502 rather than
        // the 1.0 every primary-colour case produces.
        let (h, s, l) = rgb_to_hsl(32, 64, 96);
        close(h, 210.0);
        close(s, 0.5);
        close(l, 0.2509804);

        // Light: L = 0.876, denominator 0.247, which happens to equal delta
        // so saturation returns to 1.
        let (h, s, l) = rgb_to_hsl(192, 224, 255);
        close(h, 209.52382);
        close(s, 1.0);
        close(l, 0.8764706);
    }

    #[test]
    fn hsl_to_rgb_off_centre_lightness() {
        // L = 0.25 makes c = 0.25 rather than s, and m = 0.125 rather than 0.
        assert_eq!(hsl_to_rgb(210.0, 0.5, 0.25), (31, 63, 95));
        // Zero saturation collapses to a grey at the given lightness.
        assert_eq!(hsl_to_rgb(210.0, 0.0, 0.25), (63, 63, 63));
    }

    #[test]
    fn rgb_to_cmyk_general_path_with_mid_range_key() {
        // Every existing case has k = 0 or k = 1, so `(1 - v - k) / (1 - k)`
        // either divides by one or is skipped by the early return.
        let (c, m, y, k) = rgb_to_cmyk(128, 64, 192);
        close(c, 1.0 / 3.0);
        close(m, 2.0 / 3.0);
        close(y, 0.0);
        close(k, 0.2470588);
    }

    #[test]
    fn cmyk_to_rgb_scales_by_both_ink_and_key() {
        // k = 0: only the per-channel ink applies.
        assert_eq!(cmyk_to_rgb(0.5, 0.0, 0.25, 0.0), (127, 255, 191));
        // k = 0.5 halves everything on top of the ink.
        assert_eq!(cmyk_to_rgb(0.0, 0.5, 1.0, 0.5), (127, 63, 0));
    }

    #[test]
    fn replicate_nibble_duplicates_the_digit() {
        // 0xA -> 0xAA, not 0x0A or 0xA0.
        let c = parse_hex_color("#ABC").unwrap();
        assert_eq!((c.red, c.green, c.blue), (0xAA, 0xBB, 0xCC));
    }

    #[test]
    fn test_rgb_to_hsl_and_back() {
        let cases = [
            ((0xFF, 0x00, 0x00), (0.0, 1.0, 0.5)),   // Red
            ((0x00, 0xFF, 0x00), (120.0, 1.0, 0.5)), // Green
            ((0x00, 0x00, 0xFF), (240.0, 1.0, 0.5)), // Blue
            ((0x00, 0xFF, 0xFF), (180.0, 1.0, 0.5)), // Cyan
            ((0xFF, 0x00, 0xFF), (300.0, 1.0, 0.5)), // Magenta
            ((0xFF, 0xFF, 0x00), (60.0, 1.0, 0.5)),  // Yellow
            ((0x00, 0x00, 0x00), (0.0, 0.0, 0.0)),   // Black
            ((0xFF, 0xFF, 0xFF), (0.0, 0.0, 1.0)),   // White
            ((0x80, 0x80, 0x80), (0.0, 0.0, 0.5)),   // Gray
        ];

        for ((r, g, b), (exp_h, exp_s, exp_l)) in cases {
            let (h, s, l) = rgb_to_hsl(r, g, b);

            // Hue is undefined for grayscale (often 0), checking close enough or exact 0 if expected
            if s > 0.001 {
                colspace_assertion(r, g, b, h, exp_h, "Hue", 0.1);
            }
            colspace_assertion(r, g, b, s, exp_s, "Saturation", 0.01);
            colspace_assertion(r, g, b, l, exp_l, "Lightness", 0.01);

            let (r2, g2, b2) = hsl_to_rgb(h, s, l);
            // Allow small rounding diffs
            small_diff!(r, r2);
            small_diff!(g, g2);
            small_diff!(b, b2);
        }
    }

    #[test]
    fn test_rgb_to_hsv_blue_dominant() {
        // Blue is max channel — exercises the third branch in rgb_to_hsv
        let (h, s, v) = rgb_to_hsv(0, 0, 255);
        assert!((h - 240.0).abs() < 0.1, "hue should be ~240, got {h}");
        assert!((s - 1.0).abs() < 0.01);
        assert!((v - 1.0).abs() < 0.01);

        // Negative hue wrap-around path: when gf < bf the hue formula yields < 0
        let (h2, _, _) = rgb_to_hsv(255, 0, 128);
        assert!(h2 >= 0.0, "hue must always be non-negative, got {h2}");
    }

    #[test]
    fn test_rgba_color_to_from_hsv() {
        let original = RgbaColor::opaque(200, 100, 50);
        let (h, s, v) = original.to_hsv();
        let restored = RgbaColor::from_hsv(h, s, v);
        // Allow ±2 rounding error per channel
        assert!(
            (original.red as i16 - restored.red as i16).abs() <= 2,
            "red mismatch: {} vs {}",
            original.red,
            restored.red
        );
        assert!(
            (original.green as i16 - restored.green as i16).abs() <= 2,
            "green mismatch"
        );
        assert!(
            (original.blue as i16 - restored.blue as i16).abs() <= 2,
            "blue mismatch"
        );
        // from_hsv always returns opaque
        assert_eq!(restored.alpha, 255);
    }

    #[test]
    fn test_parse_hex_color_3_char() {
        let c = parse_hex_color("#F80").unwrap();
        assert_eq!(c.red, 0xFF);
        assert_eq!(c.green, 0x88);
        assert_eq!(c.blue, 0x00);
        assert_eq!(c.alpha, 255);
    }

    #[test]
    fn test_parse_hex_color_4_char_rgba() {
        let c = parse_hex_color("#F80A").unwrap();
        assert_eq!(c.red, 0xFF);
        assert_eq!(c.green, 0x88);
        assert_eq!(c.blue, 0x00);
        assert_eq!(c.alpha, 0xAA);
    }

    #[test]
    fn test_parse_hex_color_8_char_with_alpha() {
        let c = parse_hex_color("FF8800CC").unwrap();
        assert_eq!(c.red, 0xFF);
        assert_eq!(c.green, 0x88);
        assert_eq!(c.blue, 0x00);
        assert_eq!(c.alpha, 0xCC);
    }

    #[test]
    fn test_parse_hex_color_0x_prefix() {
        let c = parse_hex_color("0xFF8800").unwrap();
        assert_eq!(c.red, 0xFF);
        assert_eq!(c.green, 0x88);
        assert_eq!(c.blue, 0x00);
        assert_eq!(c.alpha, 255);
    }

    #[test]
    fn test_parse_hex_color_invalid() {
        assert!(parse_hex_color("").is_none());
        assert!(parse_hex_color("#ZZZZZ").is_none());
        assert!(parse_hex_color("#12345").is_none()); // 5 digits → no match
    }

    #[test]
    fn test_rgb_to_cmyk_and_back() {
        let cases = [
            ((0xFF, 0x00, 0x00), (0.0, 1.0, 1.0, 0.0)), // Red
            ((0x00, 0xFF, 0x00), (1.0, 0.0, 1.0, 0.0)), // Green
            ((0x00, 0x00, 0xFF), (1.0, 1.0, 0.0, 0.0)), // Blue
            ((0x00, 0xFF, 0xFF), (1.0, 0.0, 0.0, 0.0)), // Cyan
            ((0xFF, 0x00, 0xFF), (0.0, 1.0, 0.0, 0.0)), // Magenta
            ((0xFF, 0xFF, 0x00), (0.0, 0.0, 1.0, 0.0)), // Yellow
            ((0x00, 0x00, 0x00), (0.0, 0.0, 0.0, 1.0)), // Black
            ((0xFF, 0xFF, 0xFF), (0.0, 0.0, 0.0, 0.0)), // White
        ];

        for ((r, g, b), (exp_c, exp_m, exp_y, exp_k)) in cases {
            let (c, m, y, k) = rgb_to_cmyk(r, g, b);
            colspace_assertion(r, g, b, c, exp_c, "Cyan", 0.01);
            colspace_assertion(r, g, b, m, exp_m, "Magenta", 0.01);
            colspace_assertion(r, g, b, y, exp_y, "Yellow", 0.01);
            colspace_assertion(r, g, b, k, exp_k, "Key", 0.01);

            let (r2, g2, b2) = cmyk_to_rgb(c, m, y, k);
            // Allow small rounding diffs
            small_diff!(r, r2);
            small_diff!(g, g2);
            small_diff!(b, b2);
        }
    }
}
