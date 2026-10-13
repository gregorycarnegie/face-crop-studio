#![no_main]

extern crate self as fcs_utils;

use libfuzzer_sys::fuzz_target;

#[allow(dead_code)]
#[path = "../../fcs-utils/src/color.rs"]
mod color;
pub use color::{RgbaColor, hsv_to_rgb, parse_hex_color};

#[path = "../../fcs-cli/src/color.rs"]
mod cli_color;

fuzz_target!(|data: &str| {
    let _ = cli_color::parse_fill_color_spec(data);
    let _ = parse_hex_color(data);
});
