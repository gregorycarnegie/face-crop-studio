#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../../fcs-utils/src/output/exif.rs"]
mod exif;
#[path = "../../fcs-utils/src/output/ranges.rs"]
#[allow(dead_code)]
mod ranges;

fuzz_target!(|data: &[u8]| {
    let _ = exif::png_exif_chunks(data);
    let _ = exif::jpeg_exif(data);
});
