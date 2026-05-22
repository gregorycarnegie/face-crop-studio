//! Basic image enhancement utilities for Phase 5.
//!
//! Provides a small, deterministic enhancement pipeline used by the CLI
//! when `--enhance` is requested. The implementations are intentionally
//! lightweight and pure-Rust using the `image` crate.

use crate::{
    gpu::{
        GpuBackgroundBlur, GpuBilateralFilter, GpuContext, GpuGaussianBlur, GpuHistogramEqualizer,
        GpuPixelAdjust, GpuRedEyeRemoval, GpuShapeMask, red_eye::RedEye,
    },
    shape::CropShape,
};

use anyhow::{Context, Result};
use image::{DynamicImage, ImageBuffer, Rgba};
use rayon::prelude::*;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};
use wide::f32x4;

const EPSILON: f32 = 1e-6;

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct SkinKernelKey {
    radius: i32,
    sigma_space_bits: u32,
    sigma_color_bits: u32,
}

#[derive(Clone)]
struct SkinKernel {
    kernel_side: usize,
    spatial_weights: Arc<Vec<f32>>,
    color_lut: Arc<Vec<f32>>,
}

static SKIN_KERNEL_CACHE: OnceLock<Mutex<HashMap<SkinKernelKey, Arc<SkinKernel>>>> =
    OnceLock::new();

fn skin_kernel(radius: i32, sigma_space: f32, sigma_color: f32) -> Arc<SkinKernel> {
    let key = SkinKernelKey {
        radius,
        sigma_space_bits: sigma_space.to_bits(),
        sigma_color_bits: sigma_color.to_bits(),
    };
    let cache = SKIN_KERNEL_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(hit) = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(&key)
    {
        return hit.clone();
    }

    let side = (2 * radius + 1) as usize;
    let mut spatial = Vec::with_capacity(side * side);
    let spatial_coeff = -0.5 / (sigma_space * sigma_space);
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let dist_sq = (dx * dx + dy * dy) as f32;
            spatial.push((spatial_coeff * dist_sq).exp());
        }
    }

    let max_dist_sq = 255 * 255 * 3;
    let color_coeff = -0.5 / (sigma_color * sigma_color);
    let color_lut: Vec<f32> = (0..=max_dist_sq)
        .map(|d| (color_coeff * d as f32).exp())
        .collect();

    let kernel = Arc::new(SkinKernel {
        kernel_side: side,
        spatial_weights: Arc::new(spatial),
        color_lut: Arc::new(color_lut),
    });
    cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(key, kernel.clone());
    kernel
}

/// Settings for the enhancement pipeline.
#[derive(Debug, Clone)]
pub struct EnhancementSettings {
    /// Apply histogram-equalization based auto color correction.
    pub auto_color: bool,
    /// Exposure adjustment expressed in stops (-2.0..=2.0).
    pub exposure_stops: f32,
    /// Additional brightness offset applied after exposure.
    pub brightness: i32,
    /// Contrast multiplier (0.5..=2.0, 1.0 = unchanged).
    pub contrast: f32,
    /// Saturation multiplier (0.0..=2.5, 1.0 = unchanged).
    pub saturation: f32,
    /// Strength of unsharp mask (0.0..=2.0).
    pub unsharp_amount: f32,
    /// Blur radius used for the unsharp mask in pixels.
    pub unsharp_radius: f32,
    /// Additional sharpening control layered on top of the base amount.
    pub sharpness: f32,
    /// Skin smoothing strength (0.0 = off, 1.0 = maximum).
    pub skin_smooth_amount: f32,
    /// Spatial sigma for bilateral filter (controls spatial extent).
    pub skin_smooth_sigma_space: f32,
    /// Color sigma for bilateral filter (controls color similarity threshold).
    pub skin_smooth_sigma_color: f32,
    /// Enable automated red-eye removal.
    pub red_eye_removal: bool,
    /// Red-eye detection threshold (higher = more selective).
    pub red_eye_threshold: f32,
    /// Enable background blur (portrait mode effect).
    pub background_blur: bool,
    /// Background blur strength (radius in pixels).
    pub background_blur_radius: f32,
    /// Background blur mask size (0.0-1.0, larger = more area kept sharp).
    pub background_blur_mask_size: f32,
}

impl Default for EnhancementSettings {
    fn default() -> Self {
        Self {
            auto_color: false,
            exposure_stops: 0.0,
            brightness: 0,
            contrast: 1.0,
            saturation: 1.0,
            unsharp_amount: 0.6,
            unsharp_radius: 1.0,
            sharpness: 0.0,
            skin_smooth_amount: 0.0,
            skin_smooth_sigma_space: 3.0,
            skin_smooth_sigma_color: 25.0,
            red_eye_removal: false,
            red_eye_threshold: 1.5,
            background_blur: false,
            background_blur_radius: 15.0,
            background_blur_mask_size: 0.6,
        }
    }
}

impl EnhancementSettings {
    /// Gentle preset: light tonal lift with subtle sharpening.
    pub fn natural() -> Self {
        Self {
            auto_color: true,
            exposure_stops: 0.1,
            contrast: 1.1,
            saturation: 1.05,
            sharpness: 0.2,
            ..Self::default()
        }
    }

    /// Punchier preset: warmer exposure, higher contrast and saturation.
    pub fn vivid() -> Self {
        Self {
            exposure_stops: 0.3,
            brightness: 10,
            contrast: 1.25,
            saturation: 1.3,
            unsharp_amount: 0.9,
            unsharp_radius: 1.2,
            sharpness: 0.5,
            ..Self::default()
        }
    }

    /// Headshot preset: balanced tone with stronger detail enhancement.
    pub fn professional() -> Self {
        Self {
            auto_color: true,
            exposure_stops: 0.2,
            contrast: 1.15,
            saturation: 1.05,
            unsharp_amount: 1.2,
            sharpness: 0.8,
            ..Self::default()
        }
    }

    /// Resolve a preset name (case-sensitive lowercase). Returns `None` for unknown names.
    pub fn preset_by_name(name: &str) -> Option<Self> {
        match name {
            "natural" => Some(Self::natural()),
            "vivid" => Some(Self::vivid()),
            "professional" => Some(Self::professional()),
            _ => None,
        }
    }
}

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

fn apply_lut_rgb(img: &DynamicImage, lut: &[u8; 256]) -> DynamicImage {
    let mut buf = img.to_rgba8();
    for pixel in buf.as_mut().chunks_exact_mut(4) {
        pixel[0] = lut[pixel[0] as usize];
        pixel[1] = lut[pixel[1] as usize];
        pixel[2] = lut[pixel[2] as usize];
    }
    DynamicImage::ImageRgba8(buf)
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

fn build_equalization_lut(hist: &[u32; 256], total: u32) -> [u8; 256] {
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

fn apply_histogram_equalization(img: &DynamicImage) -> DynamicImage {
    let mut buf = img.to_rgba8();
    let (w, h) = buf.dimensions();
    if w == 0 || h == 0 {
        return DynamicImage::ImageRgba8(buf);
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

    DynamicImage::ImageRgba8(buf)
}

fn apply_exposure(img: &DynamicImage, stops: f32) -> DynamicImage {
    if stops.abs() < EPSILON {
        return img.clone();
    }
    let factor = 2f32.powf(stops.clamp(-2.0, 2.0));
    let lut = build_lut(|value| {
        let boosted = (value as f32 * factor).round().clamp(0.0, 255.0);
        boosted as u8
    });
    apply_lut_rgb(img, &lut)
}

fn apply_brightness(img: &DynamicImage, offset: i32) -> DynamicImage {
    if offset == 0 {
        return img.clone();
    }
    let lut = build_lut(|value| {
        let value = value as i32 + offset;
        value.clamp(0, 255) as u8
    });
    apply_lut_rgb(img, &lut)
}

fn apply_contrast(img: &DynamicImage, multiplier: f32) -> DynamicImage {
    if (multiplier - 1.0).abs() < EPSILON {
        return img.clone();
    }
    let multiplier = multiplier.clamp(0.5, 2.0);
    let lut = build_lut(|value| {
        let normalized = value as f32 / 255.0;
        let contrasted = multiplier.mul_add(normalized - 0.5, 0.5).clamp(0.0, 1.0) * 255.0;
        contrasted.round() as u8
    });
    apply_lut_rgb(img, &lut)
}

/// Adjust saturation by mixing with per-pixel luminance: new = gray*(1-s) + orig*s.
fn apply_saturation(img: &DynamicImage, saturation: f32) -> DynamicImage {
    if (saturation - 1.0).abs() < EPSILON {
        return img.clone();
    }
    let multiplier = saturation.clamp(0.0, 2.5);
    let mut buf = img.to_rgba8();
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

    DynamicImage::ImageRgba8(buf)
}

/// Apply bilateral filter for skin smoothing (edge-preserving blur).
///
/// The bilateral filter smooths flat regions while preserving edges by
/// weighting pixels based on both spatial distance and color similarity.
fn apply_skin_smoothing(
    img: &DynamicImage,
    amount: f32,
    sigma_space: f32,
    sigma_color: f32,
) -> DynamicImage {
    if amount <= 0.0 {
        return img.clone();
    }

    let amount = amount.clamp(0.0, 1.0);
    let src = img.to_rgba8();
    let (w, h) = src.dimensions();
    if w == 0 || h == 0 {
        return DynamicImage::ImageRgba8(src);
    }

    let mut out_buffer = ImageBuffer::new(w, h);

    let radius = (sigma_space * 2.0).ceil() as i32;
    let kernel = skin_kernel(radius, sigma_space, sigma_color);
    let spatial_weights = kernel.spatial_weights.clone();
    let color_lut = kernel.color_lut.clone();
    let kernel_side = kernel.kernel_side;

    let max_y = h as i32 - 1;
    let max_x = w as i32 - 1;
    let row_stride = (w as usize) << 2; // Optimized: w * 4
    let src_data = src.as_raw();
    let out_data = out_buffer.as_mut();

    out_data
        .par_chunks_mut(row_stride)
        .enumerate()
        .for_each(|(y, row)| {
            let y_i = y as i32;
            let base_y = y * row_stride;
            for x in 0..w as usize {
                let src_idx = base_y + (x << 2); // Optimized: x * 4
                let center = &src_data[src_idx..src_idx + 4];
                let mut sum_r = 0.0f32;
                let mut sum_g = 0.0f32;
                let mut sum_b = 0.0f32;
                let mut sum_weight = 0.0f32;

                for dy in -radius..=radius {
                    let ny = (y_i + dy).clamp(0, max_y) as usize;
                    let ny_offset = ny * row_stride;
                    let spatial_row = (dy + radius) as usize * kernel_side;
                    for dx in -radius..=radius {
                        let nx = (x as i32 + dx).clamp(0, max_x) as usize;
                        let neighbor_idx = ny_offset + (nx << 2); // Optimized: nx * 4
                        let neighbor = &src_data[neighbor_idx..neighbor_idx + 4];

                        let spatial_w = spatial_weights[spatial_row + (dx + radius) as usize];
                        let dr = (center[0] as i32 - neighbor[0] as i32).abs();
                        let dg = (center[1] as i32 - neighbor[1] as i32).abs();
                        let db = (center[2] as i32 - neighbor[2] as i32).abs();
                        let color_dist_sq = (dr * dr + dg * dg + db * db) as usize;
                        let weight = spatial_w * color_lut[color_dist_sq];

                        sum_r = weight.mul_add(neighbor[0] as f32, sum_r);
                        sum_g = weight.mul_add(neighbor[1] as f32, sum_g);
                        sum_b = weight.mul_add(neighbor[2] as f32, sum_b);
                        sum_weight += weight;
                    }
                }

                if sum_weight > 0.0 {
                    let filtered_r = (sum_r / sum_weight).round().clamp(0.0, 255.0) as u8;
                    let filtered_g = (sum_g / sum_weight).round().clamp(0.0, 255.0) as u8;
                    let filtered_b = (sum_b / sum_weight).round().clamp(0.0, 255.0) as u8;

                    let center_r = center[0] as f32;
                    let center_g = center[1] as f32;
                    let center_b = center[2] as f32;
                    let final_r = amount
                        .mul_add(filtered_r as f32 - center_r, center_r)
                        .round()
                        .clamp(0.0, 255.0) as u8;
                    let final_g = amount
                        .mul_add(filtered_g as f32 - center_g, center_g)
                        .round()
                        .clamp(0.0, 255.0) as u8;
                    let final_b = amount
                        .mul_add(filtered_b as f32 - center_b, center_b)
                        .round()
                        .clamp(0.0, 255.0) as u8;

                    let out_idx = x << 2; // Optimized: x * 4
                    row[out_idx] = final_r;
                    row[out_idx + 1] = final_g;
                    row[out_idx + 2] = final_b;
                    row[out_idx + 3] = center[3];
                } else {
                    let out_idx = x << 2; // Optimized: x * 4
                    row[out_idx..out_idx + 4].copy_from_slice(center);
                }
            }
        });

    DynamicImage::ImageRgba8(out_buffer)
}

/// Apply automated red-eye reduction.
///
/// Detects and desaturates pixels where the red channel is significantly
/// higher than the green and blue channels, which is characteristic of red-eye.
fn apply_red_eye_removal(
    img: &DynamicImage,
    threshold: f32,
    eyes: Option<&[RedEye]>,
) -> DynamicImage {
    let mut out = img.to_rgba8();
    let (w, h) = out.dimensions();
    let active_eyes = eyes.filter(|list| !list.is_empty());

    for y in 0..h {
        for x in 0..w {
            if let Some(eyes_list) = active_eyes {
                let mut in_eye = false;
                for eye in eyes_list {
                    let dx = x as f32 - eye.x;
                    let dy = y as f32 - eye.y;
                    if dx * dx + dy * dy <= eye.radius * eye.radius {
                        in_eye = true;
                        break;
                    }
                }
                if !in_eye {
                    continue;
                }
            }

            let px = out.get_pixel(x, y).0;
            let r = px[0] as f32;
            let g = px[1] as f32;
            let b = px[2] as f32;

            // Calculate red dominance: red / average(green, blue)
            let avg_gb = (g + b).mul_add(0.5, EPSILON);
            let red_ratio = r / avg_gb;

            // If red is significantly dominant (typical red-eye has ratio > 1.5)
            if red_ratio > threshold && r > 80.0 {
                // Desaturate by replacing red with the average of green and blue
                let corrected_r = avg_gb.round().clamp(0.0, 255.0) as u8;
                out.put_pixel(x, y, image::Rgba([corrected_r, px[1], px[2], px[3]]));
            }
        }
    }

    DynamicImage::ImageRgba8(out)
}

/// Apply background blur with centered elliptical mask (portrait mode effect).
///
/// Blurs the background while keeping a centered elliptical region (the face) sharp.
/// Creates a professional portrait look similar to smartphone portrait mode.
fn apply_background_blur(img: &DynamicImage, radius: f32, mask_size: f32) -> DynamicImage {
    if radius <= 0.0 {
        return img.clone();
    }
    let sharp = img.to_rgba8();
    let blurred = image::imageops::blur(&sharp, radius);
    background_blur_from_rgba(&sharp, &blurred, mask_size)
}

fn apply_background_blur_with_preblur(
    img: &DynamicImage,
    blurred: &DynamicImage,
    mask_size: f32,
) -> DynamicImage {
    let sharp = img.to_rgba8();
    let blurred = blurred.to_rgba8();
    background_blur_from_rgba(&sharp, &blurred, mask_size)
}

fn background_blur_from_rgba(
    sharp: &image::RgbaImage,
    blurred: &image::RgbaImage,
    mask_size: f32,
) -> DynamicImage {
    let (w, h) = sharp.dimensions();
    if blurred.dimensions() != (w, h) {
        return DynamicImage::ImageRgba8(sharp.clone());
    }
    if w == 0 || h == 0 {
        return DynamicImage::ImageRgba8(sharp.clone());
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

    DynamicImage::ImageRgba8(out)
}

/// Apply a simple unsharp mask to an RGBA image.
fn apply_unsharp_mask(img: &DynamicImage, amount: f32, radius: f32) -> DynamicImage {
    if amount <= 0.0 || radius <= 0.0 {
        return img.clone();
    }

    let src = img.to_rgba8();
    let blurred = image::imageops::blur(&src, radius);
    DynamicImage::ImageRgba8(unsharp_with_preblur_rgba(&src, &blurred, amount))
}

fn unsharp_with_preblur_rgba(
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

fn apply_unsharp_with_preblur(
    src: &DynamicImage,
    blurred: &DynamicImage,
    amount: f32,
) -> DynamicImage {
    let src_rgba = src.to_rgba8();
    let blur_rgba = blurred.to_rgba8();
    DynamicImage::ImageRgba8(unsharp_with_preblur_rgba(&src_rgba, &blur_rgba, amount))
}

/// Apply the configured enhancements to the input image and return the result.
pub fn apply_enhancements(
    img: &DynamicImage,
    settings: &EnhancementSettings,
    eyes: Option<&[RedEye]>,
) -> DynamicImage {
    let mut out = img.clone();

    if settings.auto_color {
        out = apply_histogram_equalization(&out);
    }

    if settings.red_eye_removal {
        out = apply_red_eye_removal(&out, settings.red_eye_threshold, eyes);
    }

    if settings.exposure_stops.abs() >= EPSILON {
        out = apply_exposure(&out, settings.exposure_stops);
    }

    if settings.brightness != 0 {
        out = apply_brightness(&out, settings.brightness);
    }

    if (settings.contrast - 1.0).abs() >= EPSILON {
        out = apply_contrast(&out, settings.contrast);
    }

    if (settings.saturation - 1.0).abs() >= EPSILON {
        out = apply_saturation(&out, settings.saturation);
    }

    if settings.skin_smooth_amount > 0.0 {
        out = apply_skin_smoothing(
            &out,
            settings.skin_smooth_amount,
            settings.skin_smooth_sigma_space,
            settings.skin_smooth_sigma_color,
        );
    }

    let combined_sharp = (settings.unsharp_amount + settings.sharpness).clamp(0.0, 2.0);
    if combined_sharp > 0.0 && settings.unsharp_radius > 0.0 {
        out = apply_unsharp_mask(&out, combined_sharp, settings.unsharp_radius);
    }

    if settings.background_blur {
        out = apply_background_blur(
            &out,
            settings.background_blur_radius,
            settings.background_blur_mask_size,
        );
    }

    out
}

/// GPU-accelerated enhancement pipeline that currently offloads pixel adjustments.
#[derive(Clone)]
pub struct WgpuEnhancer {
    context: Arc<GpuContext>,
    pixel_adjust: GpuPixelAdjust,
    gaussian_blur: GpuGaussianBlur,
    bilateral_filter: GpuBilateralFilter,
    background_blur: GpuBackgroundBlur,
    red_eye: GpuRedEyeRemoval,
    shape_mask: GpuShapeMask,
    histogram_equalizer: GpuHistogramEqualizer,
}

impl WgpuEnhancer {
    /// Create a new GPU-backed enhancer using the shared [`GpuContext`].
    pub fn new(context: Arc<GpuContext>) -> Result<Self> {
        let pixel_adjust = GpuPixelAdjust::new(context.clone())
            .context("failed to create GPU pixel adjust pipeline")?;
        let gaussian_blur = GpuGaussianBlur::new(context.clone())
            .context("failed to create GPU gaussian blur pipeline")?;
        let bilateral_filter = GpuBilateralFilter::new(context.clone())
            .context("failed to create GPU bilateral filter pipeline")?;
        let background_blur = GpuBackgroundBlur::new(context.clone())
            .context("failed to create GPU background blur pipeline")?;
        let red_eye = GpuRedEyeRemoval::new(context.clone())
            .context("failed to create GPU red-eye pipeline")?;
        let shape_mask = GpuShapeMask::new(context.clone())
            .context("failed to create GPU shape mask pipeline")?;
        let histogram_equalizer = GpuHistogramEqualizer::new(context.clone())
            .context("failed to create GPU histogram equalization pipeline")?;
        Ok(Self {
            context,
            pixel_adjust,
            gaussian_blur,
            bilateral_filter,
            background_blur,
            red_eye,
            shape_mask,
            histogram_equalizer,
        })
    }

    /// Apply the configured enhancements, using GPU kernels where available.
    pub fn apply(
        &self,
        img: &DynamicImage,
        settings: &EnhancementSettings,
        eyes: Option<&[RedEye]>,
    ) -> Result<DynamicImage> {
        let mut out = img.clone();

        if settings.auto_color {
            out = match self.histogram_equalizer.equalize(&out) {
                Ok(eq) => eq,
                Err(err) => {
                    log::warn!("GPU histogram equalization failed: {err}");
                    apply_histogram_equalization(&out)
                }
            };
        }

        if settings.red_eye_removal {
            if let Some(corrected) = self.try_gpu_red_eye(&out, settings.red_eye_threshold, eyes)? {
                out = corrected;
            } else {
                out = apply_red_eye_removal(&out, settings.red_eye_threshold, eyes);
            }
        }

        if GpuPixelAdjust::needs_adjustment(settings) {
            out = self
                .pixel_adjust
                .apply(&out, settings)
                .context("gpu pixel adjust failed")?;
        } else {
            if settings.exposure_stops.abs() >= EPSILON {
                out = apply_exposure(&out, settings.exposure_stops);
            }
            if settings.brightness != 0 {
                out = apply_brightness(&out, settings.brightness);
            }
            if (settings.contrast - 1.0).abs() >= EPSILON {
                out = apply_contrast(&out, settings.contrast);
            }
            if (settings.saturation - 1.0).abs() >= EPSILON {
                out = apply_saturation(&out, settings.saturation);
            }
        }

        if settings.skin_smooth_amount > 0.0 {
            if let Some(smoothed) = self.try_gpu_skin_smoothing(settings, &out)? {
                out = smoothed;
            } else {
                out = apply_skin_smoothing(
                    &out,
                    settings.skin_smooth_amount,
                    settings.skin_smooth_sigma_space,
                    settings.skin_smooth_sigma_color,
                );
            }
        }

        let combined_sharp = (settings.unsharp_amount + settings.sharpness).clamp(0.0, 2.0);
        if combined_sharp > 0.0 && settings.unsharp_radius > 0.0 {
            if let Some(blurred) = self.try_gpu_blur(&out, settings.unsharp_radius)? {
                out = apply_unsharp_with_preblur(&out, &blurred, combined_sharp);
            } else {
                out = apply_unsharp_mask(&out, combined_sharp, settings.unsharp_radius);
            }
        }

        if settings.background_blur {
            if let Some(result) = self.try_gpu_background_blur(&out, settings)? {
                out = result;
            } else if let Some(blurred) =
                self.try_gpu_blur(&out, settings.background_blur_radius)?
            {
                out = apply_background_blur_with_preblur(
                    &out,
                    &blurred,
                    settings.background_blur_mask_size,
                );
            } else {
                out = apply_background_blur(
                    &out,
                    settings.background_blur_radius,
                    settings.background_blur_mask_size,
                );
            }
        }

        Ok(out)
    }

    /// Access the underlying GPU context (handy for logging/tests).
    pub fn context(&self) -> &Arc<GpuContext> {
        &self.context
    }

    fn try_gpu_blur(&self, image: &DynamicImage, radius: f32) -> Result<Option<DynamicImage>> {
        if radius <= 0.0 {
            return Ok(None);
        }
        match self.gaussian_blur.blur(image, radius) {
            Ok(blurred) => Ok(Some(blurred)),
            Err(err) => {
                log::warn!("GPU gaussian blur failed: {err}");
                Ok(None)
            }
        }
    }

    fn try_gpu_skin_smoothing(
        &self,
        settings: &EnhancementSettings,
        image: &DynamicImage,
    ) -> Result<Option<DynamicImage>> {
        if settings.skin_smooth_amount <= 0.0 {
            return Ok(None);
        }
        match self.bilateral_filter.smooth(
            image,
            settings.skin_smooth_amount,
            settings.skin_smooth_sigma_space,
            settings.skin_smooth_sigma_color,
        ) {
            Ok(result) => Ok(Some(result)),
            Err(err) => {
                log::warn!("GPU skin smoothing failed: {err}");
                Ok(None)
            }
        }
    }

    fn try_gpu_background_blur(
        &self,
        image: &DynamicImage,
        settings: &EnhancementSettings,
    ) -> Result<Option<DynamicImage>> {
        if !settings.background_blur || settings.background_blur_radius <= 0.0 {
            return Ok(None);
        }
        let blurred = match self.try_gpu_blur(image, settings.background_blur_radius)? {
            Some(b) => b,
            None => return Ok(None),
        };
        match self
            .background_blur
            .blend(image, &blurred, settings.background_blur_mask_size)
        {
            Ok(result) => Ok(Some(result)),
            Err(err) => {
                log::warn!("GPU background blur failed: {err}");
                Ok(None)
            }
        }
    }

    fn try_gpu_red_eye(
        &self,
        image: &DynamicImage,
        threshold: f32,
        eyes: Option<&[RedEye]>,
    ) -> Result<Option<DynamicImage>> {
        if threshold <= 0.0 {
            return Ok(None);
        }
        match self.red_eye.apply(image, threshold, eyes) {
            Ok(result) => Ok(Some(result)),
            Err(err) => {
                log::warn!("GPU red-eye removal failed: {err}");
                Ok(None)
            }
        }
    }

    pub fn apply_shape_mask_gpu(
        &self,
        image: &DynamicImage,
        shape: &CropShape,
        vignette_softness: f32,
        vignette_intensity: f32,
        vignette_color: crate::color::RgbaColor,
    ) -> Result<Option<DynamicImage>> {
        self.shape_mask.apply(
            image,
            shape,
            vignette_softness,
            vignette_intensity,
            vignette_color,
        )
    }

    /// Clears any internal GPU buffer pools to free memory.
    pub fn clear_caches(&self) {
        self.gaussian_blur.clear_cache();
        self.background_blur.clear_cache();
    }

    /// Returns the estimated total size in bytes of internal GPU buffer pools.
    pub fn memory_usage(&self) -> u64 {
        self.gaussian_blur.memory_usage() + self.background_blur.memory_usage()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::{
        apply_background_blur, apply_background_blur_with_preblur, apply_red_eye_removal,
        apply_saturation, apply_unsharp_mask, apply_unsharp_with_preblur, build_equalization_lut,
        skin_kernel,
    };
    use image::{GenericImageView, RgbaImage};
    use std::sync::Arc;

    fn solid(color: [u8; 4]) -> DynamicImage {
        DynamicImage::ImageRgba8(RgbaImage::from_pixel(4, 4, image::Rgba(color)))
    }

    #[test]
    fn histogram_equalization_stretches_levels() {
        let mut img = RgbaImage::new(4, 1);
        for x in 0..2 {
            img.put_pixel(x, 0, image::Rgba([40, 80, 120, 255]));
        }
        for x in 2..4 {
            img.put_pixel(x, 0, image::Rgba([200, 160, 100, 255]));
        }
        let out = apply_histogram_equalization(&DynamicImage::ImageRgba8(img));
        let buf = out.to_rgba8();
        let mut mins = [u8::MAX; 3];
        let mut maxs = [0u8; 3];
        for pixel in buf.pixels() {
            for c in 0..3 {
                mins[c] = mins[c].min(pixel[c]);
                maxs[c] = maxs[c].max(pixel[c]);
            }
        }
        for c in 0..3 {
            assert!(
                maxs[c] > mins[c],
                "expected channel {} max {} > min {}",
                c,
                maxs[c],
                mins[c]
            );
            assert!(
                (maxs[c] as i16 - mins[c] as i16) >= 100,
                "expected channel {} spread >=100, got {}",
                c,
                maxs[c] as i16 - mins[c] as i16
            );
        }
    }

    #[test]
    fn exposure_positive_increases_values() {
        let img = solid([64, 64, 64, 255]);
        let out = apply_exposure(&img, 1.0);
        let buf = out.to_rgba8();
        let px = buf.get_pixel(0, 0);
        assert_eq!(px[0], 128);
    }

    #[test]
    fn exposure_negative_darkens_values() {
        let img = solid([200, 200, 200, 255]);
        let out = apply_exposure(&img, -1.0);
        let buf = out.to_rgba8();
        let px = buf.get_pixel(0, 0);
        assert_eq!(px[0], 100);
    }

    #[test]
    fn brightness_offsets_channels() {
        let img = solid([100, 100, 100, 255]);
        let out = apply_brightness(&img, 20);
        let buf = out.to_rgba8();
        let px = buf.get_pixel(0, 0);
        assert_eq!(px[0], 120);
    }

    #[test]
    fn contrast_multiplier_expands_range() {
        let mut img = RgbaImage::from_pixel(4, 1, image::Rgba([128, 128, 128, 255]));
        img.put_pixel(0, 0, image::Rgba([80, 80, 80, 255]));
        img.put_pixel(3, 0, image::Rgba([180, 180, 180, 255]));
        let out = apply_contrast(&DynamicImage::ImageRgba8(img), 1.5);
        let buf = out.to_rgba8();
        assert!(buf.get_pixel(0, 0)[0] < 80);
        assert!(buf.get_pixel(3, 0)[0] > 180);
    }

    #[test]
    fn saturation_zero_grays_image() {
        let img = solid([200, 100, 50, 255]);
        let out = apply_saturation(&img, 0.0);
        let buf = out.to_rgba8();
        let px = buf.get_pixel(0, 0);
        assert_eq!(px[0], px[1]);
        assert_eq!(px[1], px[2]);
    }

    #[test]
    fn unsharp_preserves_alpha() {
        let mut img = RgbaImage::new(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                let val = if (x + y) % 2 == 0 { 0 } else { 255 };
                img.put_pixel(x, y, image::Rgba([val, val, val, 128]));
            }
        }
        let dyn_img = DynamicImage::ImageRgba8(img);
        let out = apply_unsharp_mask(&dyn_img, 1.5, 1.5).to_rgba8();
        assert_eq!(out.get_pixel(1, 1)[3], 128);
    }

    #[test]
    fn skin_smoothing_reduces_noise() {
        // Create a noisy image with alternating pixel values
        let mut img = RgbaImage::new(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                let val = if (x + y) % 2 == 0 { 100 } else { 110 };
                img.put_pixel(x, y, image::Rgba([val, val, val, 255]));
            }
        }
        let dyn_img = DynamicImage::ImageRgba8(img);

        // Apply skin smoothing
        let smoothed = apply_skin_smoothing(&dyn_img, 0.8, 3.0, 25.0).to_rgba8();

        // Check that neighboring pixels have become more similar (variance reduced)
        let p1 = smoothed.get_pixel(1, 1)[0];
        let p2 = smoothed.get_pixel(1, 2)[0];
        let diff = (p1 as i32 - p2 as i32).abs();

        // The difference should be less than the original 10
        assert!(diff < 10, "Smoothing should reduce pixel differences");

        // Alpha should be preserved
        assert_eq!(smoothed.get_pixel(1, 1)[3], 255);
    }

    #[test]
    fn skin_smoothing_zero_amount_unchanged() {
        let img = solid([150, 120, 100, 255]);
        let out = apply_skin_smoothing(&img, 0.0, 3.0, 25.0);
        assert_eq!(
            out.to_rgba8().get_pixel(0, 0),
            img.to_rgba8().get_pixel(0, 0)
        );
    }

    #[test]
    fn red_eye_removal_reduces_red_dominance() {
        // Create an image with a red-eye pixel (high red, low green/blue)
        let mut img = RgbaImage::new(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                // Normal pixel
                img.put_pixel(x, y, image::Rgba([100, 100, 100, 255]));
            }
        }
        // Add a red-eye pixel in the center (very high red)
        img.put_pixel(1, 1, image::Rgba([200, 50, 50, 255]));
        img.put_pixel(2, 1, image::Rgba([220, 60, 55, 255]));

        let dyn_img = DynamicImage::ImageRgba8(img);
        let corrected = apply_red_eye_removal(&dyn_img, 1.5, None).to_rgba8();

        // Check that the red-eye pixels have been corrected
        let px1 = corrected.get_pixel(1, 1);
        let px2 = corrected.get_pixel(2, 1);

        // Red channel should be reduced (closer to green/blue average)
        assert!(px1[0] < 200, "Red channel should be reduced from 200");
        assert!(px2[0] < 220, "Red channel should be reduced from 220");

        // Normal pixels should be unchanged
        let normal_px = corrected.get_pixel(0, 0);
        assert_eq!(normal_px[0], 100);
        assert_eq!(normal_px[1], 100);
    }

    #[test]
    fn red_eye_removal_preserves_alpha() {
        let img = solid([200, 50, 50, 128]);
        let out = apply_red_eye_removal(&img, 1.5, None).to_rgba8();
        assert_eq!(out.get_pixel(0, 0)[3], 128);
    }

    #[test]
    fn background_blur_keeps_center_sharp() {
        // Create a simple gradient image
        let mut img = RgbaImage::new(100, 100);
        for y in 0..100 {
            for x in 0..100 {
                let val = ((x + y) / 2) as u8;
                img.put_pixel(x, y, image::Rgba([val, val, val, 255]));
            }
        }
        let dyn_img = DynamicImage::ImageRgba8(img.clone());

        // Apply background blur
        let blurred = apply_background_blur(&dyn_img, 10.0, 0.5).to_rgba8();

        // Center pixel should be nearly unchanged (inside mask)
        let center_orig = img.get_pixel(50, 50)[0];
        let center_blur = blurred.get_pixel(50, 50)[0];
        let diff = (center_orig as i32 - center_blur as i32).abs();
        assert!(diff < 3, "Center pixel should be nearly unchanged");

        // Corner pixel should be blurred (outside mask)
        // We can't directly test blur, but dimensions should be preserved
        assert_eq!(blurred.dimensions(), img.dimensions());
    }

    #[test]
    fn background_blur_preserves_dimensions() {
        let img = solid([128, 128, 128, 255]);
        let out = apply_background_blur(&img, 15.0, 0.6);
        assert_eq!(out.dimensions(), img.dimensions());
    }

    #[test]
    fn pipeline_preserves_dimensions() {
        let img = solid([128, 128, 128, 255]);
        let out = apply_enhancements(
            &img,
            &EnhancementSettings {
                auto_color: true,
                exposure_stops: 0.5,
                brightness: 10,
                contrast: 1.2,
                saturation: 1.1,
                unsharp_amount: 0.8,
                unsharp_radius: 1.2,
                sharpness: 0.1,
                skin_smooth_amount: 0.5,
                skin_smooth_sigma_space: 3.0,
                skin_smooth_sigma_color: 25.0,
                red_eye_removal: true,
                red_eye_threshold: 1.5,
                background_blur: true,
                background_blur_radius: 15.0,
                background_blur_mask_size: 0.6,
            },
            None,
        );
        assert_eq!(out.width(), img.width());
        assert_eq!(out.height(), img.height());
    }

    #[test]
    fn pipeline_auto_color_matches_direct_equalization() {
        let mut img = RgbaImage::new(8, 1);
        for x in 0..4 {
            img.put_pixel(x, 0, image::Rgba([32, 64, 128, 255]));
        }
        for x in 4..8 {
            img.put_pixel(x, 0, image::Rgba([220, 180, 140, 255]));
        }
        let source = DynamicImage::ImageRgba8(img);

        let settings = EnhancementSettings {
            auto_color: true,
            unsharp_amount: 0.0,
            unsharp_radius: 0.0,
            sharpness: 0.0,
            ..EnhancementSettings::default()
        };

        let pipeline = apply_enhancements(&source, &settings, None).to_rgba8();
        let expected = apply_histogram_equalization(&source).to_rgba8();

        assert_eq!(pipeline, expected, "auto_color should reuse equalization");
        assert_ne!(
            pipeline,
            source.to_rgba8(),
            "auto_color should adjust levels"
        );
    }

    #[test]
    fn pipeline_sharpness_combines_with_unsharp_amount() {
        let mut img = RgbaImage::new(5, 1);
        for x in 0..5 {
            let val = (x * 40 + 40) as u8;
            img.put_pixel(x, 0, image::Rgba([val, val, val, 255]));
        }
        let source = DynamicImage::ImageRgba8(img);

        let settings = EnhancementSettings {
            unsharp_amount: 0.0,
            sharpness: 0.6,
            unsharp_radius: 1.0,
            ..EnhancementSettings::default()
        };

        let pipeline = apply_enhancements(&source, &settings, None).to_rgba8();
        let expected = apply_unsharp_mask(&source, 0.6, 1.0).to_rgba8();

        assert_eq!(
            pipeline, expected,
            "sharpness setting should fold into unsharp mask amount"
        );
        assert_ne!(
            pipeline,
            source.to_rgba8(),
            "sharpening should modify pixels"
        );
    }

    #[test]
    fn pipeline_red_eye_removal_matches_direct_call() {
        let mut img = RgbaImage::new(4, 2);
        for y in 0..2 {
            for x in 0..4 {
                img.put_pixel(x, y, image::Rgba([90, 90, 90, 255]));
            }
        }
        img.put_pixel(1, 0, image::Rgba([220, 40, 40, 255]));
        img.put_pixel(2, 0, image::Rgba([210, 45, 60, 255]));
        let source = DynamicImage::ImageRgba8(img);

        let settings = EnhancementSettings {
            red_eye_removal: true,
            red_eye_threshold: 1.2,
            unsharp_amount: 0.0,
            unsharp_radius: 0.0,
            sharpness: 0.0,
            ..EnhancementSettings::default()
        };

        let pipeline = apply_enhancements(&source, &settings, None).to_rgba8();
        let expected = apply_red_eye_removal(&source, 1.2, None).to_rgba8();

        assert_eq!(pipeline, expected);
        assert!(pipeline.get_pixel(1, 0)[0] < 200);
        assert!(pipeline.get_pixel(2, 0)[0] < 210);
    }

    #[test]
    fn pipeline_background_blur_matches_direct_call() {
        let mut img = RgbaImage::new(32, 32);
        for y in 0..32 {
            for x in 0..32 {
                let val = ((x + y) * 4).clamp(0, 255) as u8;
                img.put_pixel(x, y, image::Rgba([val, 255 - val, val / 2, 255]));
            }
        }
        let source = DynamicImage::ImageRgba8(img);

        let settings = EnhancementSettings {
            background_blur: true,
            background_blur_radius: 6.0,
            background_blur_mask_size: 0.5,
            unsharp_amount: 0.0,
            unsharp_radius: 0.0,
            sharpness: 0.0,
            ..EnhancementSettings::default()
        };

        let pipeline = apply_enhancements(&source, &settings, None).to_rgba8();
        let expected = apply_background_blur(&source, 6.0, 0.5).to_rgba8();

        assert_eq!(pipeline, expected);
        // ensure the blur keeps center mostly intact while affecting a corner
        let center_diff = (pipeline.get_pixel(16, 16)[0] as i16
            - source.to_rgba8().get_pixel(16, 16)[0] as i16)
            .abs();
        assert!(center_diff < 5, "central region should remain sharp");
        let corner_diff = (pipeline.get_pixel(0, 0)[0] as i16
            - source.to_rgba8().get_pixel(0, 0)[0] as i16)
            .abs();
        assert!(corner_diff > 0, "corner should show blur impact");
    }

    // -----------------------------------------------------------------------
    // equalization edge cases

    #[test]
    fn equalization_lut_zero_total_returns_identity() {
        // An all-zero histogram (total == 0) must not divide by zero and must
        // return the identity mapping.
        let hist = [0u32; 256];
        let lut = build_equalization_lut(&hist, 0);
        for (i, &v) in lut.iter().enumerate() {
            assert_eq!(v, i as u8, "identity expected at index {i}");
        }
    }

    #[test]
    fn equalization_lut_single_value_image_returns_identity() {
        // Every pixel has the same intensity: cdf_min == total, so the result
        // is undefined — we return identity to avoid divide-by-zero.
        let mut hist = [0u32; 256];
        hist[128] = 16; // all 16 pixels are intensity 128
        let lut = build_equalization_lut(&hist, 16);
        for (i, &v) in lut.iter().enumerate() {
            assert_eq!(v, i as u8, "identity expected at index {i}");
        }
    }

    // -----------------------------------------------------------------------
    // red-eye removal with eye regions

    #[test]
    fn red_eye_removal_with_eye_region_only_corrects_inside() {
        use crate::gpu::red_eye::RedEye;

        let mut img = RgbaImage::new(8, 8);
        // Fill with neutral gray
        for y in 0..8 {
            for x in 0..8 {
                img.put_pixel(x, y, image::Rgba([90, 90, 90, 255]));
            }
        }
        // Plant two red-eye pixels inside the eye radius (cx=4, cy=4, r=2)
        img.put_pixel(4, 4, image::Rgba([220, 50, 50, 255]));
        img.put_pixel(3, 4, image::Rgba([210, 45, 45, 255]));
        // And a red pixel clearly outside the radius
        img.put_pixel(0, 0, image::Rgba([220, 50, 50, 255]));

        let source = DynamicImage::ImageRgba8(img);
        let eyes = [RedEye {
            x: 4.0,
            y: 4.0,
            radius: 2.0,
            _pad: 0.0,
        }];
        let out = apply_red_eye_removal(&source, 1.5, Some(&eyes)).to_rgba8();

        // Pixels inside the eye circle must be corrected (red reduced)
        assert!(
            out.get_pixel(4, 4)[0] < 220,
            "red inside eye region should be reduced"
        );
        assert!(
            out.get_pixel(3, 4)[0] < 210,
            "red inside eye region should be reduced"
        );
        // Pixel outside the eye region must be UNCHANGED despite being red
        assert_eq!(
            out.get_pixel(0, 0)[0],
            220,
            "pixel outside eye region must not be corrected"
        );
    }

    // -----------------------------------------------------------------------
    // background blur helpers

    #[test]
    fn background_blur_with_preblur_dimension_mismatch_returns_sharp() {
        // If the pre-blurred image has different dimensions, background_blur_from_rgba
        // bails out and returns the original sharp image unchanged.
        let sharp = RgbaImage::from_pixel(4, 4, image::Rgba([100u8, 150, 200, 255]));
        let wrong_size_blur = RgbaImage::from_pixel(8, 8, image::Rgba([0u8, 0, 0, 255]));
        let sharp_dyn = DynamicImage::ImageRgba8(sharp.clone());
        let blur_dyn = DynamicImage::ImageRgba8(wrong_size_blur);

        let out = apply_background_blur_with_preblur(&sharp_dyn, &blur_dyn, 0.6).to_rgba8();
        assert_eq!(
            out, sharp,
            "mismatched dimensions should return the sharp image"
        );
    }

    #[test]
    fn background_blur_with_preblur_matches_direct_apply() {
        let mut img = RgbaImage::new(20, 20);
        for y in 0..20 {
            for x in 0..20 {
                img.put_pixel(x, y, image::Rgba([((x + y) * 6) as u8, 100, 50, 255]));
            }
        }
        let source = DynamicImage::ImageRgba8(img);
        let mask_size = 0.5f32;
        let radius = 4.0f32;

        // Produce the pre-blurred image the same way apply_background_blur does
        let pre_blurred =
            DynamicImage::ImageRgba8(image::imageops::blur(&source.to_rgba8(), radius));
        let via_preblur =
            apply_background_blur_with_preblur(&source, &pre_blurred, mask_size).to_rgba8();
        let via_direct = apply_background_blur(&source, radius, mask_size).to_rgba8();

        assert_eq!(
            via_preblur, via_direct,
            "pre-blurred and direct paths must produce identical output"
        );
    }

    // -----------------------------------------------------------------------
    // unsharp-mask with pre-blurred image

    #[test]
    fn unsharp_with_preblur_direct_matches_indirect() {
        let mut img = RgbaImage::new(6, 6);
        for y in 0..6 {
            for x in 0..6 {
                let v = ((x * 20 + y * 15) % 200 + 30) as u8;
                img.put_pixel(x, y, image::Rgba([v, 255 - v, v / 2, 200]));
            }
        }
        let source = DynamicImage::ImageRgba8(img);
        let amount = 1.2f32;
        let radius = 1.0f32;

        let pre_blurred =
            DynamicImage::ImageRgba8(image::imageops::blur(&source.to_rgba8(), radius));
        let via_preblur = apply_unsharp_with_preblur(&source, &pre_blurred, amount).to_rgba8();
        let via_direct = apply_unsharp_mask(&source, amount, radius).to_rgba8();

        assert_eq!(
            via_preblur, via_direct,
            "pre-blurred and direct unsharp paths must match"
        );
    }

    // -----------------------------------------------------------------------
    // skin kernel cache

    #[test]
    fn skin_kernel_cache_hit_returns_same_arc() {
        let a = skin_kernel(3, 3.0, 25.0);
        let b = skin_kernel(3, 3.0, 25.0);
        // Same Arc pointer means the cache returned the same allocation
        assert!(
            Arc::ptr_eq(&a, &b),
            "second call with identical params must hit the cache"
        );
        // Different params must produce a distinct kernel
        let c = skin_kernel(4, 3.0, 25.0);
        assert!(
            !Arc::ptr_eq(&a, &c),
            "different params must produce a distinct kernel"
        );
    }

    // -----------------------------------------------------------------------
    // saturation scalar tail path

    #[test]
    fn saturation_scalar_tail_is_exercised_by_odd_pixel_count() {
        // 5 pixels = 1 SIMD batch of 4 + 1 scalar remainder pixel
        let mut img = RgbaImage::new(5, 1);
        img.put_pixel(0, 0, image::Rgba([200, 100, 50, 255]));
        img.put_pixel(1, 0, image::Rgba([180, 90, 40, 255]));
        img.put_pixel(2, 0, image::Rgba([160, 80, 30, 255]));
        img.put_pixel(3, 0, image::Rgba([140, 70, 20, 255]));
        img.put_pixel(4, 0, image::Rgba([120, 60, 10, 255]));

        let source = DynamicImage::ImageRgba8(img.clone());
        let out = apply_saturation(&source, 0.0).to_rgba8();

        // At saturation=0 every pixel must be fully desaturated (all channels equal)
        for x in 0..5 {
            let px = out.get_pixel(x, 0);
            assert_eq!(
                px[0], px[1],
                "pixel {x}: R and G must be equal at zero saturation"
            );
            assert_eq!(
                px[1], px[2],
                "pixel {x}: G and B must be equal at zero saturation"
            );
        }
    }

    #[test]
    fn saturation_simd_and_scalar_paths_agree() {
        // Run saturation on 4 pixels (SIMD only) and 5 pixels (SIMD + scalar)
        // and verify the first 4 pixels match in both outputs.
        let make_img = |width: u32| {
            let mut img = RgbaImage::new(width, 1);
            for x in 0..width {
                img.put_pixel(x, 0, image::Rgba([(x * 40 + 40) as u8, 100, 50, 255]));
            }
            DynamicImage::ImageRgba8(img)
        };

        let out4 = apply_saturation(&make_img(4), 1.5).to_rgba8();
        let out5 = apply_saturation(&make_img(5), 1.5).to_rgba8();

        for x in 0..4 {
            assert_eq!(
                out4.get_pixel(x, 0),
                out5.get_pixel(x, 0),
                "pixel {x}: SIMD and scalar paths must agree"
            );
        }
    }
}

#[cfg(test)]
mod benches {
    use super::*;
    use image::{DynamicImage, ImageBuffer, Rgba, RgbaImage};
    use rayon::iter::ParallelBridge;
    use std::time::Instant;

    fn baseline_skin_smoothing(
        img: &DynamicImage,
        amount: f32,
        sigma_space: f32,
        sigma_color: f32,
    ) -> DynamicImage {
        if amount <= 0.0 {
            return img.clone();
        }

        let amount = amount.clamp(0.0, 1.0);
        let src = img.to_rgba8();
        let (w, h) = src.dimensions();
        let mut out_buffer = src.clone();
        let radius = (sigma_space * 2.0).ceil() as i32;

        let mut spatial_weights =
            vec![vec![0.0f32; (2 * radius + 1) as usize]; (2 * radius + 1) as usize];
        let spatial_coeff = -0.5 / (sigma_space * sigma_space);
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let dist_sq = (dx * dx + dy * dy) as f32;
                spatial_weights[(dy + radius) as usize][(dx + radius) as usize] =
                    (spatial_coeff * dist_sq).exp();
            }
        }

        let color_coeff = -0.5 / (sigma_color * sigma_color);
        let max_dist_sq = 255 * 255 * 3;
        let color_lut: Vec<f32> = (0..=max_dist_sq)
            .map(|d| (color_coeff * d as f32).exp())
            .collect();

        out_buffer
            .enumerate_rows_mut()
            .par_bridge()
            .for_each(|(y, row)| {
                for (x, _y, pixel) in row {
                    let center = src.get_pixel(x, y).0;
                    let mut sum_r = 0.0f32;
                    let mut sum_g = 0.0f32;
                    let mut sum_b = 0.0f32;
                    let mut sum_weight = 0.0f32;

                    for dy in -radius..=radius {
                        let ny = (y as i32 + dy).clamp(0, h as i32 - 1) as u32;
                        for dx in -radius..=radius {
                            let nx = (x as i32 + dx).clamp(0, w as i32 - 1) as u32;
                            let neighbor = src.get_pixel(nx, ny).0;

                            let dr = (center[0] as i32 - neighbor[0] as i32).abs();
                            let dg = (center[1] as i32 - neighbor[1] as i32).abs();
                            let db = (center[2] as i32 - neighbor[2] as i32).abs();
                            let color_dist_sq = (dr * dr + dg * dg + db * db) as usize;

                            let spatial_w =
                                spatial_weights[(dy + radius) as usize][(dx + radius) as usize];
                            let color_w = color_lut[color_dist_sq];
                            let weight = spatial_w * color_w;

                            sum_r = weight.mul_add(neighbor[0] as f32, sum_r);
                            sum_g = weight.mul_add(neighbor[1] as f32, sum_g);
                            sum_b = weight.mul_add(neighbor[2] as f32, sum_b);
                            sum_weight += weight;
                        }
                    }

                    if sum_weight > 0.0 {
                        let filtered_r = (sum_r / sum_weight).round().clamp(0.0, 255.0) as u8;
                        let filtered_g = (sum_g / sum_weight).round().clamp(0.0, 255.0) as u8;
                        let filtered_b = (sum_b / sum_weight).round().clamp(0.0, 255.0) as u8;

                        let center_r = center[0] as f32;
                        let center_g = center[1] as f32;
                        let center_b = center[2] as f32;
                        let final_r = amount
                            .mul_add(filtered_r as f32 - center_r, center_r)
                            .round()
                            .clamp(0.0, 255.0) as u8;
                        let final_g = amount
                            .mul_add(filtered_g as f32 - center_g, center_g)
                            .round()
                            .clamp(0.0, 255.0) as u8;
                        let final_b = amount
                            .mul_add(filtered_b as f32 - center_b, center_b)
                            .round()
                            .clamp(0.0, 255.0) as u8;

                        *pixel = image::Rgba([final_r, final_g, final_b, center[3]]);
                    }
                }
            });

        DynamicImage::ImageRgba8(out_buffer)
    }

    fn baseline_background_blur(
        sharp: &image::RgbaImage,
        blurred: &image::RgbaImage,
        mask_size: f32,
    ) -> DynamicImage {
        let (w, h) = sharp.dimensions();
        if blurred.dimensions() != (w, h) {
            return DynamicImage::ImageRgba8(sharp.clone());
        }

        let cx = w as f32 * 0.5;
        let cy = h as f32 * 0.5;
        let mask_size = mask_size.clamp(0.3, 1.0);
        let rx = (w as f32 * 0.5) * mask_size;
        let ry = (h as f32 * 0.5) * mask_size;

        let rx_sq = rx * rx;
        let ry_sq = ry * ry;
        let inner_thresh_sq = 0.81;
        let outer_thresh_sq = 1.21;

        let mut out: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(w, h);
        out.enumerate_rows_mut().par_bridge().for_each(|(y, row)| {
            let dy = y as f32 - cy;
            let dy_sq_norm = (dy * dy) / ry_sq;

            for (x, _y, pixel) in row {
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

                let sharp_px = sharp.get_pixel(x, y).0;

                if blend <= 0.0 {
                    *pixel = Rgba(sharp_px);
                } else if blend >= 1.0 {
                    *pixel = *blurred.get_pixel(x, y);
                } else {
                    let blur_px = blurred.get_pixel(x, y).0;
                    let mut result = [0u8; 4];
                    for c in 0..4 {
                        let sharp_val = sharp_px[c] as f32;
                        let mix = blend.mul_add(blur_px[c] as f32 - sharp_val, sharp_val);
                        result[c] = mix.round().clamp(0.0, 255.0) as u8;
                    }
                    *pixel = Rgba(result);
                }
            }
        });

        DynamicImage::ImageRgba8(out)
    }

    #[test]
    #[ignore]
    fn bench_skin_smoothing_variants() {
        let base = DynamicImage::ImageRgba8(ImageBuffer::from_pixel(
            256,
            256,
            Rgba([140, 120, 110, 255]),
        ));

        let iterations = 5;
        for _ in 0..2 {
            let _ = apply_skin_smoothing(&base, 0.8, 3.0, 25.0);
            let _ = baseline_skin_smoothing(&base, 0.8, 3.0, 25.0);
        }

        let start_new = Instant::now();
        for _ in 0..iterations {
            let _ = apply_skin_smoothing(&base, 0.8, 3.0, 25.0);
        }
        let new_time = start_new.elapsed();

        let start_old = Instant::now();
        for _ in 0..iterations {
            let _ = baseline_skin_smoothing(&base, 0.8, 3.0, 25.0);
        }
        let old_time = start_old.elapsed();

        println!(
            "skin smoothing optimized avg: {:?}, baseline avg: {:?}",
            new_time / iterations,
            old_time / iterations
        );
    }

    #[test]
    #[ignore]
    fn bench_background_blur_variants() {
        let sharp = RgbaImage::from_pixel(512, 512, Rgba([120, 120, 120, 255]));
        let blurred = image::imageops::blur(&sharp, 12.0);

        let iterations = 10;
        for _ in 0..2 {
            let _ = super::background_blur_from_rgba(&sharp, &blurred, 0.6);
            let _ = baseline_background_blur(&sharp, &blurred, 0.6);
        }

        let start_new = Instant::now();
        for _ in 0..iterations {
            let _ = super::background_blur_from_rgba(&sharp, &blurred, 0.6);
        }
        let new_time = start_new.elapsed();

        let start_old = Instant::now();
        for _ in 0..iterations {
            let _ = baseline_background_blur(&sharp, &blurred, 0.6);
        }
        let old_time = start_old.elapsed();

        println!(
            "background blur optimized avg: {:?}, baseline avg: {:?}",
            new_time / iterations,
            old_time / iterations
        );
    }
}
