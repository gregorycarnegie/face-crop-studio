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
    let row_stride = w as usize * 4;
    let src_data = src.as_raw();
    let out_data = out_buffer.as_mut();

    out_data
        .par_chunks_mut(row_stride)
        .enumerate()
        .for_each(|(y, row)| {
            let y_i = y as i32;
            let base_y = y * row_stride;
            for x in 0..w as usize {
                let src_idx = base_y + x * 4;
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
                        let neighbor_idx = ny_offset + nx * 4;
                        let neighbor = &src_data[neighbor_idx..neighbor_idx + 4];

                        let spatial_w = spatial_weights[spatial_row + (dx + radius) as usize];
                        let dr = (center[0] as i32 - neighbor[0] as i32).abs();
                        let dg = (center[1] as i32 - neighbor[1] as i32).abs();
                        let db = (center[2] as i32 - neighbor[2] as i32).abs();
                        let color_dist_sq = (dr * dr + dg * dg + db * db) as usize;
                        let weight = spatial_w * color_lut[color_dist_sq];

                        // ponytail: plain a*b+c, not mul_add — fma is a libm
                        // call on the SSE2 baseline this project targets.
                        sum_r += weight * neighbor[0] as f32;
                        sum_g += weight * neighbor[1] as f32;
                        sum_b += weight * neighbor[2] as f32;
                        sum_weight += weight;
                    }
                }

                if sum_weight > 0.0 {
                    let filtered_r = (sum_r / sum_weight + 0.5) as u8;
                    let filtered_g = (sum_g / sum_weight + 0.5) as u8;
                    let filtered_b = (sum_b / sum_weight + 0.5) as u8;

                    let center_r = center[0] as f32;
                    let center_g = center[1] as f32;
                    let center_b = center[2] as f32;
                    let final_r = (center_r + amount * (filtered_r as f32 - center_r) + 0.5) as u8;
                    let final_g = (center_g + amount * (filtered_g as f32 - center_g) + 0.5) as u8;
                    let final_b = (center_b + amount * (filtered_b as f32 - center_b) + 0.5) as u8;

                    let out_idx = x * 4;
                    row[out_idx] = final_r;
                    row[out_idx + 1] = final_g;
                    row[out_idx + 2] = final_b;
                    row[out_idx + 3] = center[3];
                } else {
                    let out_idx = x * 4;
                    row[out_idx..out_idx + 4].copy_from_slice(center);
                }
            }
        });

    out_buffer
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deliberately naive, single-threaded bilateral filter written straight
    /// from the definition.
    ///
    /// Hand-computing expected pixels for this filter is not practical — a
    /// single output value is a ratio of two 25-term sums of products of two
    /// exponentials — and rounding the ratio to `u8` puts several probes
    /// uncomfortably close to a `.5` boundary. Comparing against an
    /// independent implementation avoids that: the reference is test code, so
    /// it is never mutated, and any arithmetic change in the real filter shows
    /// up as a pixel mismatch. The loop order matches the implementation so
    /// float accumulation is bit-identical rather than merely close.
    fn reference_skin_smooth(
        src: &RgbaImage,
        amount: f32,
        sigma_space: f32,
        sigma_color: f32,
    ) -> RgbaImage {
        let amount = amount.clamp(0.0, 1.0);
        let (w, h) = src.dimensions();
        let radius = (sigma_space * 2.0).ceil() as i32;
        let spatial_coeff = -0.5 / (sigma_space * sigma_space);
        let color_coeff = -0.5 / (sigma_color * sigma_color);

        let mut out = RgbaImage::new(w, h);
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                let center = src.get_pixel(x as u32, y as u32).0;
                let (mut sum_r, mut sum_g, mut sum_b, mut sum_w) = (0.0f32, 0.0, 0.0, 0.0);

                for dy in -radius..=radius {
                    let ny = (y + dy).clamp(0, h as i32 - 1);
                    for dx in -radius..=radius {
                        let nx = (x + dx).clamp(0, w as i32 - 1);
                        let n = src.get_pixel(nx as u32, ny as u32).0;

                        let spatial = (spatial_coeff * (dx * dx + dy * dy) as f32).exp();
                        let dr = (center[0] as i32 - n[0] as i32).abs();
                        let dg = (center[1] as i32 - n[1] as i32).abs();
                        let db = (center[2] as i32 - n[2] as i32).abs();
                        let colour = (color_coeff * (dr * dr + dg * dg + db * db) as f32).exp();
                        let weight = spatial * colour;

                        sum_r += weight * n[0] as f32;
                        sum_g += weight * n[1] as f32;
                        sum_b += weight * n[2] as f32;
                        sum_w += weight;
                    }
                }

                let px = if sum_w > 0.0 {
                    let filtered = |sum: f32| (sum / sum_w + 0.5) as u8;
                    let blend =
                        |c: u8, f: u8| (c as f32 + amount * (f as f32 - c as f32) + 0.5) as u8;
                    [
                        blend(center[0], filtered(sum_r)),
                        blend(center[1], filtered(sum_g)),
                        blend(center[2], filtered(sum_b)),
                        center[3],
                    ]
                } else {
                    center
                };
                out.put_pixel(x as u32, y as u32, image::Rgba(px));
            }
        }
        out
    }

    /// Varied colours, a hard edge, and varying alpha, at a size where the
    /// clamped borders are a meaningful fraction of the image.
    fn sample_image() -> RgbaImage {
        let mut img = RgbaImage::new(7, 5);
        for y in 0..5u32 {
            for x in 0..7u32 {
                // A vertical edge at x = 3 plus a diagonal gradient, so the
                // colour term actually does work rather than degenerating.
                let base = if x < 3 { 40u32 } else { 190 };
                let r = (base + x * 6 + y * 3).min(255) as u8;
                let g = (base + y * 9).min(255) as u8;
                let b = (base + x * 4).min(255) as u8;
                img.put_pixel(x, y, image::Rgba([r, g, b, (200 + y * 10) as u8]));
            }
        }
        img
    }

    #[test]
    fn skin_kernel_spatial_weights_follow_the_gaussian() {
        // sigma_space 1.0 -> radius 1, a 3x3 kernel, coefficient -0.5.
        let kernel = skin_kernel(1, 1.0, 25.0);
        assert_eq!(kernel.kernel_side, 3);
        assert_eq!(kernel.spatial_weights.len(), 9);

        let near = (-0.5f32).exp(); // distance^2 == 1
        let far = (-1.0f32).exp(); // distance^2 == 2
        let expected = [near, 1.0, near];
        // Row ordering is dy outer, dx inner, so the centre lands at index 4.
        for (i, want) in [far, near, far, near, 1.0, near, far, near, far]
            .into_iter()
            .enumerate()
        {
            assert!(
                (kernel.spatial_weights[i] - want).abs() < 1e-6,
                "spatial weight {i}: expected {want}, got {}",
                kernel.spatial_weights[i]
            );
        }
        // The middle row is the symmetric one either side of the centre.
        assert!((kernel.spatial_weights[3] - expected[0]).abs() < 1e-6);
        assert!((kernel.spatial_weights[5] - expected[2]).abs() < 1e-6);
    }

    #[test]
    fn skin_kernel_colour_lut_covers_the_whole_distance_range() {
        let kernel = skin_kernel(1, 1.0, 25.0);
        // Squared distance maxes out at 255^2 * 3 across three channels, and
        // the table is inclusive of that endpoint.
        assert_eq!(kernel.color_lut.len(), 255 * 255 * 3 + 1);

        // coefficient = -0.5 / 25^2 = -0.0008, so lut[1250] = exp(-1).
        assert!((kernel.color_lut[0] - 1.0).abs() < 1e-6);
        assert!((kernel.color_lut[1250] - (-1.0f32).exp()).abs() < 1e-6);
        // Monotonically decreasing: identical colours weigh most.
        assert!(kernel.color_lut[1] < kernel.color_lut[0]);
    }

    #[test]
    fn skin_kernel_radius_sets_the_side_length() {
        assert_eq!(skin_kernel(2, 1.0, 25.0).kernel_side, 5);
        assert_eq!(skin_kernel(3, 1.5, 25.0).kernel_side, 7);
        assert_eq!(skin_kernel(2, 1.0, 25.0).spatial_weights.len(), 25);
    }

    #[test]
    fn skin_smooth_matches_the_reference_filter() {
        let src = sample_image();
        let got = skin_smooth_rgba(&src, 0.7, 1.0, 30.0);
        let want = reference_skin_smooth(&src, 0.7, 1.0, 30.0);
        assert_eq!(got, want);
    }

    #[test]
    fn skin_smooth_matches_the_reference_across_parameters() {
        let src = sample_image();
        // Vary each knob independently: a wider kernel, a tighter colour
        // sigma (edge-preserving), and a partial blend amount.
        for (amount, sigma_space, sigma_color) in
            [(1.0, 1.5, 10.0), (0.25, 0.5, 60.0), (0.5, 2.0, 25.0)]
        {
            let got = skin_smooth_rgba(&src, amount, sigma_space, sigma_color);
            let want = reference_skin_smooth(&src, amount, sigma_space, sigma_color);
            assert_eq!(
                got, want,
                "mismatch at amount {amount}, sigma_space {sigma_space}, sigma_color {sigma_color}"
            );
        }
    }

    #[test]
    fn skin_smooth_clamps_amount_above_one() {
        // amount is clamped to 1.0, so 5.0 must behave exactly like 1.0.
        let src = sample_image();
        assert_eq!(
            skin_smooth_rgba(&src, 5.0, 1.0, 30.0),
            skin_smooth_rgba(&src, 1.0, 1.0, 30.0)
        );
    }

    #[test]
    fn skin_smooth_preserves_alpha_and_flat_colour() {
        // Every neighbour is identical, so the filtered value equals the
        // centre and the output is the input — whatever the blend amount.
        let src = RgbaImage::from_pixel(5, 5, image::Rgba([150u8, 120, 100, 173]));
        assert_eq!(skin_smooth_rgba(&src, 1.0, 1.0, 25.0), src);
        assert_eq!(skin_smooth_rgba(&src, 0.3, 2.0, 60.0), src);
    }

    #[test]
    fn skin_smooth_zero_dimensions_returns_input() {
        // Guards the row-stride chunking against a zero-sized image.
        let empty = RgbaImage::new(0, 4);
        assert_eq!(
            skin_smooth_rgba(&empty, 1.0, 1.0, 25.0).dimensions(),
            (0, 4)
        );
    }

    #[test]
    fn apply_skin_smoothing_zero_amount_is_a_passthrough() {
        let src = DynamicImage::ImageRgba8(sample_image());
        assert_eq!(
            apply_skin_smoothing(&src, 0.0, 1.0, 25.0).to_rgba8(),
            src.to_rgba8()
        );
    }
}
