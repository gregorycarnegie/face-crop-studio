//! Shared helpers for the GPU operation tests.
//!
//! Two problems these exist to fix, both found by a full `cargo mutants` run:
//!
//! 1. Every GPU test module had its own copy of `test_context()`.
//! 2. More importantly, the tests built their inputs with
//!    `RgbaImage::from_pixel` — a single flat colour. A uniform image makes
//!    almost every interesting defect invisible: swap a workgroup count, an
//!    index stride, or a buffer usage flag and a flat image still comes back
//!    flat. Combined with tests that only asserted "did not error", an
//!    operation could be replaced wholesale by `Ok(Default::default())` and the
//!    suite stayed green.
//!
//! [`gradient_image`] gives every pixel a distinct value so spatial arithmetic
//! has somewhere to go wrong, and [`assert_plausible_output`] pins down the
//! properties that a returned image must satisfy for the operation to have run
//! at all.

use super::{GpuAvailability, GpuContext, GpuContextOptions};
use image::{DynamicImage, GenericImageView, RgbaImage};
use std::sync::Arc;

/// A GPU context, or `None` when the machine has no usable adapter.
///
/// Tests self-skip rather than fail so the suite still runs on hosted CI
/// runners without a GPU. `FCS_STRICT_TESTS` does not apply here: absence of an
/// adapter is a property of the machine, not a broken fixture.
pub(crate) fn test_context() -> Option<Arc<GpuContext>> {
    match GpuContext::init_with_fallback(&GpuContextOptions::default()) {
        GpuAvailability::Available(ctx) => Some(ctx),
        _ => None,
    }
}

/// An image where every pixel differs from its neighbours in all four channels.
///
/// Deliberately not a flat fill: the per-pixel variation is what lets a test
/// notice a shader that reads the wrong texel, a dispatch that covers the wrong
/// area, or a stride computed with the wrong operator.
pub(crate) fn gradient_image(width: u32, height: u32) -> DynamicImage {
    let mut img = RgbaImage::new(width, height);
    for (x, y, px) in img.enumerate_pixels_mut() {
        // Coprime multipliers so no two pixels collide within realistic sizes,
        // and so horizontal and vertical mixups produce different values.
        let r = ((x * 7 + y * 13) % 256) as u8;
        let g = ((x * 29 + y * 3) % 256) as u8;
        let b = ((x * 11 + y * 37) % 256) as u8;
        *px = image::Rgba([r, g, b, 255]);
    }
    DynamicImage::ImageRgba8(img)
}

/// Assert that `result` is a real output image for an operation over `input`.
///
/// Checks, in order of how bluntly they fail:
/// - dimensions match the input, which alone rules out an operation replaced by
///   a default-constructed (0x0) image;
/// - the pixel buffer is the size those dimensions imply;
/// - the content is not uniformly zero, which catches a dispatch that never
///   wrote anything as distinct from one that wrote the wrong thing.
pub(crate) fn assert_plausible_output(result: &DynamicImage, input: &DynamicImage, op: &str) {
    assert_eq!(
        result.dimensions(),
        input.dimensions(),
        "{op} must preserve image dimensions"
    );

    let pixels = result.to_rgba8();
    let expected_len = (input.width() * input.height() * 4) as usize;
    assert_eq!(
        pixels.as_raw().len(),
        expected_len,
        "{op} returned a buffer of the wrong length"
    );

    assert!(
        pixels.as_raw().iter().any(|&b| b != 0),
        "{op} returned an all-zero image, so nothing was written"
    );
}

/// Assert that an operation actually changed the image.
///
/// The counterpart to [`assert_plausible_output`]: for a filter that is supposed
/// to have a visible effect, returning the input untouched is as wrong as
/// returning a blank buffer, and a surprising number of mutants do exactly that.
pub(crate) fn assert_changed(result: &DynamicImage, input: &DynamicImage, op: &str) {
    assert_plausible_output(result, input, op);
    assert_ne!(
        result.to_rgba8().as_raw(),
        input.to_rgba8().as_raw(),
        "{op} left the image unchanged"
    );
}
