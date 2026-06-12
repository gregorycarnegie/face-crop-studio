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
