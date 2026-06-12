//! Automated red-eye correction.

use crate::gpu::red_eye::RedEye;

use super::EPSILON;
use image::{DynamicImage, RgbaImage};

/// Apply automated red-eye reduction.
///
/// Detects and desaturates pixels where the red channel is significantly
/// higher than the green and blue channels, which is characteristic of red-eye.
pub(super) fn apply_red_eye_removal(
    img: &DynamicImage,
    threshold: f32,
    eyes: Option<&[RedEye]>,
) -> DynamicImage {
    let mut out = img.to_rgba8();
    red_eye_in_place(&mut out, threshold, eyes);
    DynamicImage::ImageRgba8(out)
}

pub(super) fn red_eye_in_place(out: &mut RgbaImage, threshold: f32, eyes: Option<&[RedEye]>) {
    let (w, h) = out.dimensions();
    if w == 0 || h == 0 {
        return;
    }

    match eyes.filter(|list| !list.is_empty()) {
        // With known eye locations only the pixels inside each eye's bounding
        // box need testing, instead of scanning the whole image.
        Some(eyes_list) => {
            for eye in eyes_list {
                correct_red_eye_region(out, threshold, eye);
            }
        }
        None => {
            for px in out.as_mut().chunks_exact_mut(4) {
                correct_red_pixel(px, threshold);
            }
        }
    }
}

/// Desaturate a single RGBA pixel if red is dominant (typical red-eye has a
/// red/(avg green+blue) ratio > 1.5) by replacing red with that average.
/// Idempotent, so overlapping eye regions may safely re-apply it.
#[inline]
fn correct_red_pixel(px: &mut [u8], threshold: f32) {
    let r = px[0] as f32;
    let g = px[1] as f32;
    let b = px[2] as f32;

    // Check red dominance without dividing by the green/blue average.
    let avg_gb = (g + b).mul_add(0.5, EPSILON);

    if r > avg_gb * threshold && r > 80.0 {
        px[0] = avg_gb.round().clamp(0.0, 255.0) as u8;
    }
}

fn correct_red_eye_region(out: &mut RgbaImage, threshold: f32, eye: &RedEye) {
    let (w, h) = out.dimensions();
    // `as u32` saturates negative/NaN coordinates to 0; an eye entirely
    // outside the image yields an empty range.
    let min_x = (eye.x - eye.radius).floor().max(0.0) as u32;
    let max_x = ((eye.x + eye.radius).ceil() as u32).min(w - 1);
    let min_y = (eye.y - eye.radius).floor().max(0.0) as u32;
    let max_y = ((eye.y + eye.radius).ceil() as u32).min(h - 1);
    if min_x > max_x || min_y > max_y {
        return;
    }

    let radius_sq = eye.radius * eye.radius;
    let row_stride = w as usize * 4;
    let data = out.as_mut();

    for y in min_y..=max_y {
        let dy = y as f32 - eye.y;
        let dy_sq = dy * dy;
        let row = &mut data[y as usize * row_stride..(y as usize + 1) * row_stride];
        for x in min_x..=max_x {
            let dx = x as f32 - eye.x;
            // Plain multiply-add (not fused) to match the original membership
            // test bit-for-bit on boundary pixels.
            if dx * dx + dy_sq <= radius_sq {
                let idx = x as usize * 4;
                correct_red_pixel(&mut row[idx..idx + 4], threshold);
            }
        }
    }
}
