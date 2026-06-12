//! Portrait blur and sharpening helpers.

use image::{DynamicImage, ImageBuffer, Rgba, RgbaImage};
use rayon::prelude::*;

/// Apply background blur with centered elliptical mask (portrait mode effect).
///
/// Blurs the background while keeping a centered elliptical region (the face) sharp.
/// Creates a professional portrait look similar to smartphone portrait mode.
pub(super) fn apply_background_blur(
    img: &DynamicImage,
    radius: f32,
    mask_size: f32,
) -> DynamicImage {
    if radius <= 0.0 {
        return img.clone();
    }
    let sharp = img.to_rgba8();
    // fast_blur is a linear-time box-blur approximation of the Gaussian; at
    // portrait-blur radii it is visually indistinguishable and far cheaper.
    let blurred = image::imageops::fast_blur(&sharp, radius);
    DynamicImage::ImageRgba8(background_blur_from_rgba(&sharp, &blurred, mask_size))
}

pub(super) fn apply_background_blur_with_preblur(
    img: &DynamicImage,
    blurred: &DynamicImage,
    mask_size: f32,
) -> DynamicImage {
    let sharp = img.to_rgba8();
    let blurred = blurred.to_rgba8();
    DynamicImage::ImageRgba8(background_blur_from_rgba(&sharp, &blurred, mask_size))
}

pub(super) fn background_blur_from_rgba(
    sharp: &RgbaImage,
    blurred: &RgbaImage,
    mask_size: f32,
) -> RgbaImage {
    let (w, h) = sharp.dimensions();
    if blurred.dimensions() != (w, h) {
        return sharp.clone();
    }
    if w == 0 || h == 0 {
        return sharp.clone();
    }

    let cx = w as f32 * 0.5;
    let cy = h as f32 * 0.5;
    let mask_size = mask_size.clamp(0.3, 1.0);
    let rx = (w as f32 * 0.5) * mask_size;
    let ry = (h as f32 * 0.5) * mask_size;

    // Squared radii for distance checks
    let rx_sq = rx * rx;
    let ry_sq = ry * ry;

    // Thresholds for transition zone (0.9 to 1.1)
    // We check dist_sq against 0.9^2 and 1.1^2
    let inner_thresh_sq = 0.81; // 0.9 * 0.9
    let outer_thresh_sq = 1.21; // 1.1 * 1.1

    let mut out: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(w, h);
    let row_stride = (w as usize) << 2; // Optimized: w * 4
    let sharp_raw = sharp.as_raw();
    let blur_raw = blurred.as_raw();
    let out_raw = out.as_mut();

    out_raw
        .par_chunks_mut(row_stride)
        .enumerate()
        .for_each(|(y, row)| {
            let dy = y as f32 - cy;
            let dy_sq_norm = (dy * dy) / ry_sq;
            let sharp_row = &sharp_raw[y * row_stride..(y + 1) * row_stride];
            let blur_row = &blur_raw[y * row_stride..(y + 1) * row_stride];

            for x in 0..w as usize {
                let dx = x as f32 - cx;
                let dx_sq_norm = (dx * dx) / rx_sq;
                let dist_sq = dx_sq_norm + dy_sq_norm;

                let blend = if dist_sq < inner_thresh_sq {
                    0.0
                } else if dist_sq > outer_thresh_sq {
                    1.0
                } else {
                    let dist = dist_sq.sqrt();
                    (dist - 0.9) * 5.0
                };

                let idx = x << 2; // Optimized: x * 4
                if blend <= 0.0 {
                    row[idx..idx + 4].copy_from_slice(&sharp_row[idx..idx + 4]);
                } else if blend >= 1.0 {
                    row[idx..idx + 4].copy_from_slice(&blur_row[idx..idx + 4]);
                } else {
                    let sharp_px = &sharp_row[idx..idx + 4];
                    let blur_px = &blur_row[idx..idx + 4];
                    for c in 0..4 {
                        let sharp_val = sharp_px[c] as f32;
                        let mix = blend.mul_add(blur_px[c] as f32 - sharp_val, sharp_val);
                        row[idx + c] = mix.round().clamp(0.0, 255.0) as u8;
                    }
                }
            }
        });

    out
}

/// Apply a simple unsharp mask to an RGBA image.
pub(super) fn apply_unsharp_mask(img: &DynamicImage, amount: f32, radius: f32) -> DynamicImage {
    if amount <= 0.0 || radius <= 0.0 {
        return img.clone();
    }

    let src = img.to_rgba8();
    let blurred = image::imageops::fast_blur(&src, radius);
    DynamicImage::ImageRgba8(unsharp_with_preblur_rgba(&src, &blurred, amount))
}

pub(super) fn unsharp_with_preblur_rgba(
    src: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    blurred: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    amount: f32,
) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let amount = amount.clamp(0.0, 2.0);
    let (w, h) = src.dimensions();
    assert_eq!(
        blurred.dimensions(),
        (w, h),
        "unsharp inputs must have matching dimensions"
    );
    let mut out: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(w, h);

    for ((dst, s), b) in out
        .as_mut()
        .chunks_exact_mut(4)
        .zip(src.as_raw().chunks_exact(4))
        .zip(blurred.as_raw().chunks_exact(4))
    {
        for c in 0..3usize {
            let src_val = s[c] as f32;
            let diff = src_val - b[c] as f32;
            let val = amount.mul_add(diff, src_val);
            dst[c] = val.round().clamp(0.0, 255.0) as u8;
        }
        dst[3] = s[3];
    }

    out
}

pub(super) fn apply_unsharp_with_preblur(
    src: &DynamicImage,
    blurred: &DynamicImage,
    amount: f32,
) -> DynamicImage {
    let src_rgba = src.to_rgba8();
    let blur_rgba = blurred.to_rgba8();
    DynamicImage::ImageRgba8(unsharp_with_preblur_rgba(&src_rgba, &blur_rgba, amount))
}
