//! Color parsing utilities for CLI color specifications.

use fcs_utils::{RgbaColor, hsv_to_rgb, parse_hex_color};

/// Parse fill color specification from CLI argument.
/// Accepts formats: #RRGGBB, rgb(), hsv(), or comma-separated values.
pub fn parse_fill_color_spec(raw: &str) -> Result<RgbaColor, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("fill color value is empty".to_string());
    }

    if let Some(color) = parse_hex_color(trimmed) {
        return Ok(color);
    }
    if let Some(args) = parse_fn_args(trimmed, "rgb") {
        let (r, g, b) = parse_rgb_components(&args)?;
        let alpha = args
            .get(3)
            .map(|value| parse_alpha_value(value))
            .transpose()?
            .unwrap_or(255);
        return Ok(RgbaColor {
            red: r,
            green: g,
            blue: b,
            alpha,
        });
    }
    if let Some(args) = parse_fn_args(trimmed, "hsv") {
        if args.len() < 3 {
            return Err("hsv() requires three values: hue,saturation,value".to_string());
        }
        let hue = parse_hue_value(args[0])?;
        let sat = parse_percentage_value(args[1])?;
        let val = parse_percentage_value(args[2])?;
        let (r, g, b) = hsv_to_rgb(hue, sat, val);
        return Ok(RgbaColor::opaque(r, g, b));
    }

    if trimmed.contains(',') {
        let parts: Vec<_> = trimmed
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();
        if parts.len() >= 3 {
            let (r, g, b) = parse_rgb_components(&parts)?;
            return Ok(RgbaColor::opaque(r, g, b));
        }
    }

    Err(format!(
        "unrecognized fill color format '{}'; expected #RRGGBB, rgb(), or hsv()",
        trimmed
    ))
}

fn parse_fn_args<'a>(input: &'a str, name: &str) -> Option<Vec<&'a str>> {
    let trimmed = input.trim();
    let open = trimmed.find('(')?;
    let close = trimmed.rfind(')')?;
    if close <= open {
        return None;
    }
    if !trimmed[..open].trim().eq_ignore_ascii_case(name) {
        return None;
    }
    let inner = &trimmed[open + 1..close];
    let args = inner
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>();
    if args.is_empty() { None } else { Some(args) }
}

fn parse_rgb_components(parts: &[&str]) -> Result<(u8, u8, u8), String> {
    if parts.len() < 3 {
        return Err("expected three values for rgb()".to_string());
    }
    Ok((
        parse_rgb_value(parts[0])?,
        parse_rgb_value(parts[1])?,
        parse_rgb_value(parts[2])?,
    ))
}

fn parse_rgb_value(token: &str) -> Result<u8, String> {
    let value: f32 = token
        .parse()
        .map_err(|_| format!("invalid RGB component '{}'", token))?;
    if !(0.0..=255.0).contains(&value) {
        return Err(format!(
            "RGB component '{}' must be between 0 and 255",
            token
        ));
    }
    Ok(value.round() as u8)
}

fn parse_alpha_value(token: &str) -> Result<u8, String> {
    let trimmed = token.trim();
    let normalized = if let Some(stripped) = trimmed.strip_suffix('%') {
        let pct = stripped
            .trim()
            .parse::<f32>()
            .map_err(|_| format!("invalid alpha percentage '{}'", token))?;
        pct * 0.01
    } else {
        let value: f32 = trimmed
            .parse()
            .map_err(|_| format!("invalid alpha value '{}'", token))?;
        if value > 1.0 { value / 255.0 } else { value }
    };
    Ok((normalized.clamp(0.0, 1.0) * 255.0).round() as u8)
}

fn parse_hue_value(token: &str) -> Result<f32, String> {
    let mut raw = token.trim().to_string();
    if raw.len() >= 3 && raw[raw.len() - 3..].eq_ignore_ascii_case("deg") {
        raw.truncate(raw.len() - 3);
        raw = raw.trim_end().to_string();
    }
    if raw.ends_with('°') {
        raw.pop();
        raw = raw.trim_end().to_string();
    }
    let value: f32 = raw
        .parse()
        .map_err(|_| format!("invalid hue '{}'", token))?;
    Ok(value.rem_euclid(360.0))
}

fn parse_percentage_value(token: &str) -> Result<f32, String> {
    let trimmed = token.trim();
    if let Some(stripped) = trimmed.strip_suffix('%') {
        let pct = stripped
            .trim()
            .parse::<f32>()
            .map_err(|_| format!("invalid percentage '{}'", token))?;
        return Ok((pct * 0.01).clamp(0.0, 1.0));
    }
    let value: f32 = trimmed
        .parse()
        .map_err(|_| format!("invalid component '{}'", token))?;
    if value > 1.0 {
        Ok((value * 0.01).clamp(0.0, 1.0))
    } else {
        Ok(value.clamp(0.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_fill_color_spec_accepts_hex_and_comma_separated_rgb() {
        assert_eq!(
            parse_fill_color_spec("#112233").expect("valid hex color"),
            RgbaColor::opaque(0x11, 0x22, 0x33)
        );
        assert_eq!(
            parse_fill_color_spec("0x44556677").expect("valid 0x hex color with alpha"),
            RgbaColor {
                red: 0x44,
                green: 0x55,
                blue: 0x66,
                alpha: 0x77,
            }
        );
        assert_eq!(
            parse_fill_color_spec(" 1, 2, 3 ").expect("valid comma-separated RGB"),
            RgbaColor::opaque(1, 2, 3)
        );
    }

    #[test]
    fn parse_fill_color_spec_accepts_rgb_function_with_multiple_alpha_formats() {
        assert_eq!(
            parse_fill_color_spec("rgb(10, 20, 30)").expect("valid rgb() without alpha"),
            RgbaColor::opaque(10, 20, 30)
        );
        assert_eq!(
            parse_fill_color_spec("RGB(10, 20, 30, 0.5)").expect("valid RGB() with float alpha"),
            RgbaColor {
                red: 10,
                green: 20,
                blue: 30,
                alpha: 128,
            }
        );
        assert_eq!(
            parse_fill_color_spec("rgb(10, 20, 30, 50%)").expect("valid rgb() with percent alpha"),
            RgbaColor {
                red: 10,
                green: 20,
                blue: 30,
                alpha: 128,
            }
        );
        assert_eq!(
            parse_fill_color_spec("rgb(10, 20, 30, 128)").expect("valid rgb() with integer alpha"),
            RgbaColor {
                red: 10,
                green: 20,
                blue: 30,
                alpha: 128,
            }
        );
    }

    #[test]
    fn parse_fill_color_spec_accepts_hsv_function() {
        assert_eq!(
            parse_fill_color_spec("hsv(0deg, 100%, 100%)").expect("valid hsv() with deg suffix"),
            RgbaColor::opaque(255, 0, 0)
        );
        assert_eq!(
            parse_fill_color_spec("HSV(240°, 100, 100)").expect("valid HSV() with degree symbol"),
            RgbaColor::opaque(0, 0, 255)
        );
    }

    #[test]
    fn parse_fill_color_spec_rejects_empty_or_invalid_inputs() {
        assert_eq!(
            parse_fill_color_spec("   ").unwrap_err(),
            "fill color value is empty"
        );
        assert!(parse_fill_color_spec("rgb(10, 20)").is_err());
        assert!(parse_fill_color_spec("rgb(300, 20, 30)").is_err());
        assert!(parse_fill_color_spec("hsv(120, 50)").is_err());
        assert!(parse_fill_color_spec("not-a-color").is_err());
    }

    #[test]
    fn parse_fill_color_spec_hsv_plain_hue_number() {
        // parse_hue_value: no deg/° suffix — plain f32
        let result =
            parse_fill_color_spec("hsv(0, 100%, 100%)").expect("valid hsv() with plain hue number");
        assert_eq!(result, RgbaColor::opaque(255, 0, 0));
    }

    #[test]
    fn parse_fill_color_spec_hsv_fractional_percentage() {
        // parse_percentage_value: value ≤ 1.0 — treated as a direct fraction
        let result = parse_fill_color_spec("hsv(0, 1.0, 1.0)")
            .expect("valid hsv() with fractional percentage");
        assert_eq!(result, RgbaColor::opaque(255, 0, 0));
    }

    #[test]
    fn parse_fill_color_spec_empty_fn_args_returns_error() {
        // parse_fn_args: empty parens → args.is_empty() → None → falls to final error
        assert!(parse_fill_color_spec("rgb()").is_err());
        assert!(parse_fill_color_spec("hsv()").is_err());
    }

    #[test]
    fn parse_fill_color_spec_bad_alpha_value_returns_error() {
        assert!(parse_fill_color_spec("rgb(10, 20, 30, notanumber)").is_err());
        assert!(parse_fill_color_spec("rgb(10, 20, 30, 50notpct)").is_err());
    }

    #[test]
    fn parse_fill_color_spec_comma_separated_too_few_parts_returns_error() {
        // Comma path with < 3 parts falls through to the final unrecognized-format error
        assert!(parse_fill_color_spec("10, 20").is_err());
    }

    // ------------------------------------------------------------------
    // The three component parsers, exercised directly. They were only ever
    // reached through `parse_fill_color_spec` with values that did not
    // distinguish their branches, so the scaling arithmetic and the boundary
    // between "fraction" and "0-255 / percentage" were untested.
    // ------------------------------------------------------------------

    #[test]
    fn parse_alpha_value_treats_above_one_as_0_255_and_at_most_one_as_a_fraction() {
        // The branch pivots on `value > 1.0`, so 1.0 itself must stay a fraction:
        // as a 0-255 value it would collapse to 1/255 -> 0.
        assert_eq!(parse_alpha_value("1").unwrap(), 255);
        assert_eq!(parse_alpha_value("1.0").unwrap(), 255);
        assert_eq!(parse_alpha_value("0").unwrap(), 0);
        assert_eq!(parse_alpha_value("0.5").unwrap(), 128);

        // Above 1.0 is a 0-255 byte value, divided by 255.
        assert_eq!(parse_alpha_value("255").unwrap(), 255);
        assert_eq!(parse_alpha_value("128").unwrap(), 128);
        assert_eq!(parse_alpha_value("2").unwrap(), 2);

        // Out of range clamps rather than wrapping.
        assert_eq!(parse_alpha_value("999").unwrap(), 255);
        assert_eq!(parse_alpha_value("-5").unwrap(), 0);
    }

    #[test]
    fn parse_alpha_value_scales_percentages_by_one_hundredth() {
        assert_eq!(parse_alpha_value("100%").unwrap(), 255);
        assert_eq!(parse_alpha_value("50%").unwrap(), 128);
        assert_eq!(parse_alpha_value("0%").unwrap(), 0);
        // Whitespace inside the percentage form is tolerated.
        assert_eq!(parse_alpha_value(" 25 % ").unwrap(), 64);
        assert!(parse_alpha_value("abc%").is_err());
        assert!(parse_alpha_value("nonsense").is_err());
    }

    #[test]
    fn parse_hue_value_strips_units_and_wraps_into_zero_to_360() {
        for (input, expected) in [
            ("90", 90.0f32),
            ("90deg", 90.0),
            ("90DEG", 90.0),
            ("90 deg", 90.0),
            ("90°", 90.0),
            ("90 °", 90.0),
        ] {
            let got = parse_hue_value(input).unwrap();
            assert!(
                (got - expected).abs() < 1e-3,
                "hue {input}: got {got}, expected {expected}"
            );
        }

        // rem_euclid, so negatives wrap up and multiples of 360 collapse to 0.
        assert!((parse_hue_value("-90").unwrap() - 270.0).abs() < 1e-3);
        assert!(parse_hue_value("360").unwrap().abs() < 1e-3);
        assert!((parse_hue_value("450").unwrap() - 90.0).abs() < 1e-3);

        // "deg" stripping is length-guarded: a bare unit is not a number.
        assert!(parse_hue_value("deg").is_err());
        assert!(parse_hue_value("").is_err());
    }

    #[test]
    fn parse_percentage_value_distinguishes_fractions_from_percentages() {
        // Explicit percent sign: scaled by 0.01.
        for (input, expected) in [("100%", 1.0f32), ("50%", 0.5), ("0%", 0.0)] {
            let got = parse_percentage_value(input).unwrap();
            assert!(
                (got - expected).abs() < 1e-4,
                "{input}: got {got}, expected {expected}"
            );
        }

        // No percent sign and <= 1.0: already a fraction, used as-is. 1.0 sits on
        // the boundary and must not be rescaled to 0.01.
        assert!((parse_percentage_value("1").unwrap() - 1.0).abs() < 1e-4);
        assert!((parse_percentage_value("0.25").unwrap() - 0.25).abs() < 1e-4);

        // No percent sign and > 1.0: read as a percentage, so 50 -> 0.5.
        assert!((parse_percentage_value("50").unwrap() - 0.5).abs() < 1e-4);
        assert!((parse_percentage_value("100").unwrap() - 1.0).abs() < 1e-4);

        // Clamped at both ends.
        assert!((parse_percentage_value("500").unwrap() - 1.0).abs() < 1e-4);
        assert!(parse_percentage_value("-1").unwrap().abs() < 1e-4);

        assert!(parse_percentage_value("x%").is_err());
        assert!(parse_percentage_value("x").is_err());
    }
}
