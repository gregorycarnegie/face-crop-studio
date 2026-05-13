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
}
