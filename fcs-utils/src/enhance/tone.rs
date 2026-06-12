//! Tone, contrast, saturation, and histogram equalization helpers.

use super::{EPSILON, settings::EnhancementSettings};
use image::{DynamicImage, RgbaImage};
use wide::f32x4;

fn identity_lut() -> [u8; 256] {
    let mut lut = [0u8; 256];
    for (i, item) in lut.iter_mut().enumerate() {
        *item = i as u8;
    }
    lut
}

#[inline]
fn build_lut(mut mapper: impl FnMut(u8) -> u8) -> [u8; 256] {
    let mut lut = [0u8; 256];
    for (value, slot) in lut.iter_mut().enumerate() {
        *slot = mapper(value as u8);
    }
    lut
}

pub(super) fn apply_lut_in_place(buf: &mut RgbaImage, lut: &[u8; 256]) {
    for pixel in buf.as_mut().chunks_exact_mut(4) {
        pixel[0] = lut[pixel[0] as usize];
        pixel[1] = lut[pixel[1] as usize];
        pixel[2] = lut[pixel[2] as usize];
    }
}

fn apply_lut_rgb(img: &DynamicImage, lut: &[u8; 256]) -> DynamicImage {
    let mut buf = img.to_rgba8();
    apply_lut_in_place(&mut buf, lut);
    DynamicImage::ImageRgba8(buf)
}

/// Compose two LUTs into one: `out[i] = second[first[i]]`.
fn compose_luts(first: &[u8; 256], second: &[u8; 256]) -> [u8; 256] {
    let mut out = [0u8; 256];
    for (slot, &mapped) in out.iter_mut().zip(first.iter()) {
        *slot = second[mapped as usize];
    }
    out
}

fn exposure_lut(stops: f32) -> [u8; 256] {
    let factor = 2f32.powf(stops.clamp(-2.0, 2.0));
    build_lut(|value| {
        let boosted = (value as f32 * factor).round().clamp(0.0, 255.0);
        boosted as u8
    })
}

fn brightness_lut(offset: i32) -> [u8; 256] {
    build_lut(|value| {
        let value = value as i32 + offset;
        value.clamp(0, 255) as u8
    })
}

fn contrast_lut(multiplier: f32) -> [u8; 256] {
    let multiplier = multiplier.clamp(0.5, 2.0);
    build_lut(|value| {
        let normalized = value as f32 / 255.0;
        let contrasted = multiplier.mul_add(normalized - 0.5, 0.5).clamp(0.0, 1.0) * 255.0;
        contrasted.round() as u8
    })
}

/// Fold the active tone stages (exposure → brightness → contrast) into a
/// single LUT so the pipeline applies them in one pass over the image.
/// Returns `None` when every stage is a no-op.
pub(super) fn tone_lut(settings: &EnhancementSettings) -> Option<[u8; 256]> {
    let mut lut: Option<[u8; 256]> = None;
    let fold = |stage: [u8; 256], lut: &mut Option<[u8; 256]>| {
        *lut = Some(match lut {
            Some(prev) => compose_luts(prev, &stage),
            None => stage,
        });
    };
    if settings.exposure_stops.abs() >= EPSILON {
        fold(exposure_lut(settings.exposure_stops), &mut lut);
    }
    if settings.brightness != 0 {
        fold(brightness_lut(settings.brightness), &mut lut);
    }
    if (settings.contrast - 1.0).abs() >= EPSILON {
        fold(contrast_lut(settings.contrast), &mut lut);
    }
    lut
}

#[inline]
fn clamp_vec_to_u8(vec: f32x4) -> [u8; 4] {
    let rounded: [f32; 4] = vec.round().into();
    let mut out = [0u8; 4];
    for (idx, value) in rounded.iter().enumerate() {
        out[idx] = value.clamp(0.0, 255.0) as u8;
    }
    out
}

pub(super) fn build_equalization_lut(hist: &[u32; 256], total: u32) -> [u8; 256] {
    if total == 0 {
        return identity_lut();
    }

    let mut cdf = [0u32; 256];
    let mut cumulative = 0u32;
    let mut cdf_min = None;
    for (idx, count) in hist.iter().enumerate() {
        cumulative += *count;
        cdf[idx] = cumulative;
        if cdf_min.is_none() && *count > 0 {
            cdf_min = Some(cumulative);
        }
    }

    let cdf_min = match cdf_min {
        Some(v) => v,
        None => return identity_lut(),
    };

    if cdf_min == total {
        return identity_lut();
    }

    let denom = (total - cdf_min) as f32;
    let mut lut = [0u8; 256];
    for i in 0..=255 {
        let cdf_val = cdf[i];
        let numerator = if cdf_val > cdf_min {
            (cdf_val - cdf_min) as f32
        } else {
            0.0
        };
        let mapped = (numerator / denom * 255.0).round().clamp(0.0, 255.0) as u8;
        lut[i] = mapped;
    }

    lut
}

pub(super) fn equalize_histogram_in_place(buf: &mut RgbaImage) {
    let (w, h) = buf.dimensions();
    if w == 0 || h == 0 {
        return;
    }

    let mut hist_r = [0u32; 256];
    let mut hist_g = [0u32; 256];
    let mut hist_b = [0u32; 256];

    for px in buf.pixels() {
        hist_r[px[0] as usize] += 1;
        hist_g[px[1] as usize] += 1;
        hist_b[px[2] as usize] += 1;
    }
    let total = w * h;
    let lut_r = build_equalization_lut(&hist_r, total);
    let lut_g = build_equalization_lut(&hist_g, total);
    let lut_b = build_equalization_lut(&hist_b, total);

    for px in buf.as_mut().chunks_exact_mut(4) {
        px[0] = lut_r[px[0] as usize];
        px[1] = lut_g[px[1] as usize];
        px[2] = lut_b[px[2] as usize];
    }
}

pub(super) fn apply_histogram_equalization(img: &DynamicImage) -> DynamicImage {
    let mut buf = img.to_rgba8();
    equalize_histogram_in_place(&mut buf);
    DynamicImage::ImageRgba8(buf)
}

pub(super) fn apply_exposure(img: &DynamicImage, stops: f32) -> DynamicImage {
    if stops.abs() < EPSILON {
        return img.clone();
    }
    apply_lut_rgb(img, &exposure_lut(stops))
}

pub(super) fn apply_brightness(img: &DynamicImage, offset: i32) -> DynamicImage {
    if offset == 0 {
        return img.clone();
    }
    apply_lut_rgb(img, &brightness_lut(offset))
}

pub(super) fn apply_contrast(img: &DynamicImage, multiplier: f32) -> DynamicImage {
    if (multiplier - 1.0).abs() < EPSILON {
        return img.clone();
    }
    apply_lut_rgb(img, &contrast_lut(multiplier))
}

/// Adjust saturation by mixing with per-pixel luminance: new = gray*(1-s) + orig*s.
pub(super) fn apply_saturation(img: &DynamicImage, saturation: f32) -> DynamicImage {
    if (saturation - 1.0).abs() < EPSILON {
        return img.clone();
    }
    let mut buf = img.to_rgba8();
    saturation_in_place(&mut buf, saturation);
    DynamicImage::ImageRgba8(buf)
}

pub(super) fn saturation_in_place(buf: &mut RgbaImage, saturation: f32) {
    let multiplier = saturation.clamp(0.0, 2.5);
    let data = buf.as_mut();
    let vec_inv = f32x4::splat(1.0 - multiplier);
    let vec_mul = f32x4::splat(multiplier);
    let coeff_r = f32x4::splat(0.299);
    let coeff_g = f32x4::splat(0.587);
    let coeff_b = f32x4::splat(0.114);
    let mut idx = 0;

    while idx + 16 <= data.len() {
        let mut r = [0.0f32; 4];
        let mut g = [0.0f32; 4];
        let mut b = [0.0f32; 4];
        let mut a = [0u8; 4];
        for lane in 0..4 {
            let base = idx + (lane << 2); // Optimized: lane * 4
            r[lane] = data[base] as f32;
            g[lane] = data[base + 1] as f32;
            b[lane] = data[base + 2] as f32;
            a[lane] = data[base + 3];
        }

        let rv = f32x4::from(r);
        let gv = f32x4::from(g);
        let bv = f32x4::from(b);
        let gray = rv * coeff_r + gv * coeff_g + bv * coeff_b;

        let new_r = clamp_vec_to_u8(gray * vec_inv + rv * vec_mul);
        let new_g = clamp_vec_to_u8(gray * vec_inv + gv * vec_mul);
        let new_b = clamp_vec_to_u8(gray * vec_inv + bv * vec_mul);

        for lane in 0..4 {
            let base = idx + (lane << 2); // Optimized: lane * 4
            data[base] = new_r[lane];
            data[base + 1] = new_g[lane];
            data[base + 2] = new_b[lane];
            data[base + 3] = a[lane];
        }

        idx += 16;
    }

    for pixel in data[idx..].chunks_exact_mut(4) {
        let r = pixel[0] as f32;
        let g = pixel[1] as f32;
        let b = pixel[2] as f32;
        let gray = 0.587f32.mul_add(g, 0.114f32.mul_add(b, 0.299f32 * r));
        let r_adj = multiplier.mul_add(r - gray, gray);
        let g_adj = multiplier.mul_add(g - gray, gray);
        let b_adj = multiplier.mul_add(b - gray, gray);
        pixel[0] = r_adj.round().clamp(0.0, 255.0) as u8;
        pixel[1] = g_adj.round().clamp(0.0, 255.0) as u8;
        pixel[2] = b_adj.round().clamp(0.0, 255.0) as u8;
    }
}
