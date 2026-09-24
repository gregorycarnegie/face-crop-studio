//! CPU enhancement pipeline orchestration.

use crate::gpu::red_eye::RedEye;

use super::{
    EPSILON,
    detail::{background_blur_from_rgba, unsharp_with_preblur_rgba},
    red_eye::red_eye_in_place,
    settings::EnhancementSettings,
    skin::skin_smooth_rgba,
    tone::{apply_lut_in_place, equalize_histogram_in_place, saturation_in_place, tone_lut},
};
use image::DynamicImage;

/// Apply the configured enhancements to the input image and return the result.
///
/// Converts to RGBA8 once up front; every stage then mutates that buffer in
/// place (or swaps it for filters that can't run in place), avoiding the
/// full-image copy per stage that chaining the `DynamicImage` helpers incurs.
pub fn apply_enhancements(
    img: &DynamicImage,
    settings: &EnhancementSettings,
    eyes: Option<&[RedEye]>,
) -> DynamicImage {
    let mut buf = img.to_rgba8();

    if settings.auto_color {
        equalize_histogram_in_place(&mut buf);
    }

    if settings.red_eye_removal {
        red_eye_in_place(&mut buf, settings.red_eye_threshold, eyes);
    }

    if let Some(lut) = tone_lut(settings) {
        apply_lut_in_place(&mut buf, &lut);
    }

    if (settings.saturation - 1.0).abs() >= EPSILON {
        saturation_in_place(&mut buf, settings.saturation);
    }

    if settings.skin_smooth_amount > 0.0 {
        buf = skin_smooth_rgba(
            &buf,
            settings.skin_smooth_amount,
            settings.skin_smooth_sigma_space,
            settings.skin_smooth_sigma_color,
        );
    }

    let combined_sharp = (settings.unsharp_amount + settings.sharpness).clamp(0.0, 2.0);
    if combined_sharp > 0.0 && settings.unsharp_radius > 0.0 {
        let blurred = image::imageops::fast_blur(&buf, settings.unsharp_radius);
        buf = unsharp_with_preblur_rgba(&buf, &blurred, combined_sharp);
    }

    if settings.background_blur && settings.background_blur_radius > 0.0 {
        let blurred = image::imageops::fast_blur(&buf, settings.background_blur_radius);
        buf = background_blur_from_rgba(&buf, &blurred, settings.background_blur_mask_size);
    }

    DynamicImage::ImageRgba8(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Noise, not a flat colour or a gradient: `fast_blur` at radius 0 has to visibly change
    /// something for a test to see it run.
    fn noisy(w: u32, h: u32) -> DynamicImage {
        DynamicImage::ImageRgba8(image::RgbaImage::from_fn(w, h, |x, y| {
            let v = (x * 37 + y * 91 + x * y * 13) % 256;
            image::Rgba([v as u8, (v * 7 % 256) as u8, (255 - v) as u8, 255])
        }))
    }

    /// A zero radius switches the blur-based effects off even when their amount is set. That
    /// is more than a fast path: `fast_blur` at radius 0 still changes pixels, so running an
    /// effect there alters an image the settings leave alone.
    #[test]
    fn a_zero_radius_switches_the_blur_based_effects_off() {
        let base = EnhancementSettings {
            unsharp_amount: 0.0,
            sharpness: 0.0,
            ..EnhancementSettings::default()
        };
        let untouched = apply_enhancements(&noisy(24, 18), &base, None).to_rgba8();
        for (name, settings) in [
            (
                "sharpening",
                EnhancementSettings {
                    sharpness: 0.8,
                    unsharp_radius: 0.0,
                    ..base.clone()
                },
            ),
            (
                "background blur",
                EnhancementSettings {
                    background_blur: true,
                    background_blur_radius: 0.0,
                    ..base.clone()
                },
            ),
        ] {
            let out = apply_enhancements(&noisy(24, 18), &settings, None).to_rgba8();
            assert_eq!(out, untouched, "{name} at radius 0 changed the image");
        }
    }
}
