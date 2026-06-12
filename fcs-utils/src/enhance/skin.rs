//! Skin smoothing filters.

use image::{DynamicImage, ImageBuffer, RgbaImage};
use rayon::prelude::*;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct SkinKernelKey {
    radius: i32,
    sigma_space_bits: u32,
    sigma_color_bits: u32,
}

#[derive(Clone)]
pub(super) struct SkinKernel {
    kernel_side: usize,
    spatial_weights: Arc<Vec<f32>>,
    color_lut: Arc<Vec<f32>>,
}

static SKIN_KERNEL_CACHE: OnceLock<Mutex<HashMap<SkinKernelKey, Arc<SkinKernel>>>> =
    OnceLock::new();

pub(super) fn skin_kernel(radius: i32, sigma_space: f32, sigma_color: f32) -> Arc<SkinKernel> {
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

/// Apply bilateral filter for skin smoothing (edge-preserving blur).
///
/// The bilateral filter smooths flat regions while preserving edges by
/// weighting pixels based on both spatial distance and color similarity.
pub(super) fn apply_skin_smoothing(
    img: &DynamicImage,
    amount: f32,
    sigma_space: f32,
    sigma_color: f32,
) -> DynamicImage {
    if amount <= 0.0 {
        return img.clone();
    }
    let src = img.to_rgba8();
    DynamicImage::ImageRgba8(skin_smooth_rgba(&src, amount, sigma_space, sigma_color))
}

pub(super) fn skin_smooth_rgba(
    src: &RgbaImage,
    amount: f32,
    sigma_space: f32,
    sigma_color: f32,
) -> RgbaImage {
    let amount = amount.clamp(0.0, 1.0);
    let (w, h) = src.dimensions();
    if w == 0 || h == 0 {
        return src.clone();
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

    out_buffer
}
