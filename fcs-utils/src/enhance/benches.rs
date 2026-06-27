use super::*;
use image::{DynamicImage, ImageBuffer, Rgba, RgbaImage};
use rayon::iter::{ParallelBridge, ParallelIterator};
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

/// Sequential (pre-rayon) unsharp, kept only as the bench baseline.
fn baseline_unsharp(src: &RgbaImage, blurred: &RgbaImage, amount: f32) -> RgbaImage {
    let amount = amount.clamp(0.0, 2.0);
    let (w, h) = src.dimensions();
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

#[test]
#[ignore]
fn bench_unsharp_variants() {
    let src = RgbaImage::from_pixel(1920, 1080, Rgba([130, 110, 90, 255]));
    let blurred = image::imageops::fast_blur(&src, 3.0);

    let iterations = 50;
    for _ in 0..3 {
        let _ = super::detail::unsharp_with_preblur_rgba(&src, &blurred, 0.8);
        let _ = baseline_unsharp(&src, &blurred, 0.8);
    }

    let start_new = Instant::now();
    for _ in 0..iterations {
        let _ = super::detail::unsharp_with_preblur_rgba(&src, &blurred, 0.8);
    }
    let new_time = start_new.elapsed();

    let start_old = Instant::now();
    for _ in 0..iterations {
        let _ = baseline_unsharp(&src, &blurred, 0.8);
    }
    let old_time = start_old.elapsed();

    println!(
        "unsharp rayon avg: {:?}, sequential baseline avg: {:?}",
        new_time / iterations,
        old_time / iterations
    );
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
