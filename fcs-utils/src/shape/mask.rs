//! Shape mask application for RGBA and dynamic images.

use crate::color::RgbaColor;

use super::{outline::build_path, types::CropShape};
use image::{DynamicImage, RgbaImage};
use rayon::prelude::*;
use std::f32::consts::FRAC_1_SQRT_2;
use tiny_skia::{FillRule, Paint, Pixmap, Transform};

/// Maximum mask resolution for raster-based shape masking.
const MAX_MASK_RESOLUTION: f32 = 2048.0;

/// Apply the shape mask to the supplied RGBA image in-place.
pub fn apply_shape_mask(
    image: &mut RgbaImage,
    shape: &CropShape,
    vignette_softness: f32,
    vignette_intensity: f32,
    vignette_color: RgbaColor,
) {
    if matches!(shape, CropShape::Rectangle) && vignette_softness <= 0.0 {
        return;
    }

    // Use analytical SDFs for simple shapes to avoid rasterization + blur cost.
    if matches!(
        shape,
        CropShape::Rectangle
            | CropShape::Ellipse
            | CropShape::RoundedRectangle { .. }
            | CropShape::ChamferedRectangle { .. }
    ) {
        apply_analytical_mask(
            image,
            shape,
            vignette_softness,
            vignette_intensity,
            vignette_color,
        );
        return;
    }

    apply_raster_mask_optimized(
        image,
        shape,
        vignette_softness,
        vignette_intensity,
        vignette_color,
    );
}

#[derive(Clone, Copy)]
struct AnalyticalMaskParams {
    width: f32,
    height: f32,
    cx: f32,
    cy: f32,
    softness_px: f32,
    shape_param: f32,
}

fn precompute_analytical_mask_params(
    shape: &CropShape,
    width: u32,
    height: u32,
    vignette_softness: f32,
) -> AnalyticalMaskParams {
    let width = width as f32;
    let height = height as f32;

    AnalyticalMaskParams {
        width,
        height,
        cx: width * 0.5,
        cy: height * 0.5,
        softness_px: if vignette_softness > 0.0 {
            (width.min(height) * 0.5 * vignette_softness).max(1.0)
        } else {
            0.0
        },
        shape_param: analytical_shape_param(shape, width, height),
    }
}

fn analytical_shape_param(shape: &CropShape, width: f32, height: f32) -> f32 {
    match shape {
        CropShape::RoundedRectangle { radius_pct } => {
            let limit = width.min(height) * 0.5;
            (width.min(height) * radius_pct).clamp(0.0, limit)
        }
        CropShape::ChamferedRectangle { size_pct } => {
            let limit = width.min(height) * 0.5;
            (width.min(height) * size_pct).clamp(0.0, limit)
        }
        _ => 0.0,
    }
}

#[inline]
fn axis_aligned_rect_signed_distance(p_abs_x: f32, p_abs_y: f32, half_w: f32, half_h: f32) -> f32 {
    let dx = p_abs_x - half_w;
    let dy = p_abs_y - half_h;
    dx.max(0.0).hypot(dy.max(0.0)) + dx.max(dy).min(0.0)
}

fn analytical_signed_distance(
    shape: &CropShape,
    px: f32,
    py: f32,
    params: &AnalyticalMaskParams,
) -> f32 {
    let p_abs_x = (px - params.cx).abs();
    let p_abs_y = (py - params.cy).abs();

    match shape {
        CropShape::Ellipse => {
            let rx = params.width * 0.5;
            let ry = params.height * 0.5;
            let val = (p_abs_x * p_abs_x) / (rx * rx) + (p_abs_y * p_abs_y) / (ry * ry);
            (val.sqrt() - 1.0) * params.width.min(params.height) * 0.5
        }
        CropShape::Rectangle => {
            let bx = params.width * 0.5;
            let by = params.height * 0.5;
            axis_aligned_rect_signed_distance(p_abs_x, p_abs_y, bx, by)
        }
        CropShape::RoundedRectangle { .. } => {
            let radius = params.shape_param;
            let bx = params.width * 0.5 - radius;
            let by = params.height * 0.5 - radius;
            axis_aligned_rect_signed_distance(p_abs_x, p_abs_y, bx, by) - radius
        }
        CropShape::ChamferedRectangle { .. } => {
            let chamfer = params.shape_param;
            let bx = params.width * 0.5;
            let by = params.height * 0.5;
            let rect_dist = axis_aligned_rect_signed_distance(p_abs_x, p_abs_y, bx, by);

            let diag_dist = (p_abs_x + p_abs_y - (bx + by - chamfer)) * FRAC_1_SQRT_2;

            rect_dist.max(diag_dist)
        }
        _ => unreachable!(),
    }
}

fn mask_alpha_from_distance(dist: f32, softness_px: f32) -> f32 {
    if softness_px > 0.0 {
        let t = dist / softness_px;
        (0.5 - 0.5 * t).clamp(0.0, 1.0)
    } else if dist <= 0.0 {
        1.0
    } else {
        0.0
    }
}

fn apply_analytical_mask(
    image: &mut RgbaImage,
    shape: &CropShape,
    vignette_softness: f32,
    vignette_intensity: f32,
    vignette_color: RgbaColor,
) {
    let (w, h) = image.dimensions();
    if w == 0 || h == 0 {
        return;
    }
    let params = precompute_analytical_mask_params(shape, w, h, vignette_softness);

    image
        .par_chunks_mut(4 * w as usize)
        .enumerate()
        .for_each(|(y, row)| {
            let py = y as f32 + 0.5;
            for x in 0..w as usize {
                let px = x as f32 + 0.5;
                let dist = analytical_signed_distance(shape, px, py, &params);
                let mask_alpha = mask_alpha_from_distance(dist, params.softness_px);

                process_pixel(
                    &mut row[x * 4..x * 4 + 4],
                    mask_alpha,
                    vignette_intensity,
                    &vignette_color,
                );
            }
        });
}

fn raster_mask_scale(width: u32, height: u32) -> f32 {
    if width.max(height) > MAX_MASK_RESOLUTION as u32 {
        MAX_MASK_RESOLUTION / width.max(height) as f32
    } else {
        1.0
    }
}

fn build_raster_hard_mask(mask_w: u32, mask_h: u32, shape: &CropShape) -> Option<RgbaImage> {
    let mut pixmap = Pixmap::new(mask_w, mask_h)?;
    pixmap.fill(tiny_skia::Color::from_rgba8(0, 0, 0, 0));

    if let Some(path) = build_path(mask_w, mask_h, shape) {
        let mut paint = Paint::default();
        paint.set_color_rgba8(255, 255, 255, 255);
        paint.anti_alias = true;

        pixmap.fill_path(
            &path,
            &paint,
            FillRule::Winding,
            Transform::identity(),
            None,
        );
    }

    RgbaImage::from_raw(mask_w, mask_h, pixmap.data().to_vec())
}

fn build_raster_soft_mask(hard_mask: RgbaImage, vignette_softness: f32) -> RgbaImage {
    if vignette_softness <= 0.0 {
        return hard_mask;
    }

    let (mask_w, mask_h) = hard_mask.dimensions();
    let radius = (mask_w.min(mask_h) as f32 * 0.5 * vignette_softness).max(1.0);
    let soft_mask = image::imageops::blur(&hard_mask, radius);

    let mut combined = hard_mask.clone();
    for (c_pixel, s_pixel) in combined.pixels_mut().zip(soft_mask.pixels()) {
        let hard_a = c_pixel[3] as f32 / 255.0;
        let soft_a = s_pixel[3] as f32 / 255.0;
        let final_a = (hard_a * soft_a * 255.0 + 0.5) as u8;
        c_pixel[3] = final_a;
    }

    combined
}

fn sample_mask_alpha_bilinear(
    mask_raw: &[u8],
    mask_w: usize,
    mask_h: usize,
    sample_x: f32,
    sample_y: f32,
) -> f32 {
    let x0 = sample_x.floor() as i32;
    let y0 = sample_y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let wx = sample_x - x0 as f32;
    let wy = sample_y - y0 as f32;

    let get_alpha = |ix: i32, iy: i32| -> f32 {
        let cx = ix.clamp(0, mask_w as i32 - 1) as usize;
        let cy = iy.clamp(0, mask_h as i32 - 1) as usize;
        mask_raw[(cy * mask_w + cx) * 4 + 3] as f32 / 255.0
    };

    let tl = get_alpha(x0, y0);
    let tr = get_alpha(x1, y0);
    let bl = get_alpha(x0, y1);
    let br = get_alpha(x1, y1);

    let top = tl * (1.0 - wx) + tr * wx;
    let bot = bl * (1.0 - wx) + br * wx;
    top * (1.0 - wy) + bot * wy
}

fn apply_raster_mask_optimized(
    image: &mut RgbaImage,
    shape: &CropShape,
    vignette_softness: f32,
    vignette_intensity: f32,
    vignette_color: RgbaColor,
) {
    let width = image.width();
    let height = image.height();
    if width == 0 || height == 0 {
        return;
    }

    let scale = raster_mask_scale(width, height);

    let mask_w = (width as f32 * scale).ceil() as u32;
    let mask_h = (height as f32 * scale).ceil() as u32;

    let hard_mask = match build_raster_hard_mask(mask_w, mask_h, shape) {
        Some(mask) => mask,
        None => return,
    };

    let mask_buffer = build_raster_soft_mask(hard_mask, vignette_softness);
    let mask_raw = mask_buffer.as_raw();
    let mask_w_usize = mask_w as usize;

    image
        .par_chunks_mut(4 * width as usize)
        .enumerate()
        .for_each(|(y, row)| {
            let v = (y as f32 + 0.5) * scale;
            for x in 0..width as usize {
                let u = (x as f32 + 0.5) * scale;

                let mask_alpha = sample_mask_alpha_bilinear(
                    mask_raw,
                    mask_w_usize,
                    mask_h as usize,
                    u - 0.5,
                    v - 0.5,
                );

                process_pixel(
                    &mut row[x * 4..x * 4 + 4],
                    mask_alpha,
                    vignette_intensity,
                    &vignette_color,
                );
            }
        });
}

fn process_pixel(
    pixel: &mut [u8],
    mask_alpha: f32,
    vignette_intensity: f32,
    vignette_color: &RgbaColor,
) {
    let inv_mask = 1.0 - mask_alpha;

    let vign_helper = |pixel: u8, vig: u8, mix_factor: f32| {
        (pixel as f32 + mix_factor * (vig as f32 - pixel as f32)).clamp(0.0, 255.0) as u8
    };

    if vignette_intensity > 0.0 && inv_mask > 0.0 {
        let mix_factor = inv_mask * vignette_intensity;

        pixel[0] = vign_helper(pixel[0], vignette_color.red, mix_factor);
        pixel[1] = vign_helper(pixel[1], vignette_color.green, mix_factor);
        pixel[2] = vign_helper(pixel[2], vignette_color.blue, mix_factor);
    }

    pixel[3] = (pixel[3] as f32 * mask_alpha).round() as u8;
}

/// Apply the shape mask to a dynamic image, upgrading to RGBA as needed.
pub fn apply_shape_mask_dynamic(
    image: &mut DynamicImage,
    shape: &CropShape,
    vignette_softness: f32,
    vignette_intensity: f32,
    vignette_color: RgbaColor,
) {
    if matches!(shape, CropShape::Rectangle) && vignette_softness <= 0.0 {
        return;
    }

    let mut rgba = image.to_rgba8();
    apply_shape_mask(
        &mut rgba,
        shape,
        vignette_softness,
        vignette_intensity,
        vignette_color,
    );
    *image = DynamicImage::ImageRgba8(rgba);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analytical_signed_distance_marks_center_inside_and_outside_positive() {
        let params = precompute_analytical_mask_params(&CropShape::Rectangle, 20, 10, 0.0);
        assert!(analytical_signed_distance(&CropShape::Rectangle, 10.0, 5.0, &params) <= 0.0);
        assert!(analytical_signed_distance(&CropShape::Rectangle, 25.0, 5.0, &params) > 0.0);
    }

    #[test]
    fn analytical_signed_distance_handles_ellipse_rounded_and_chamfered_shapes() {
        let ellipse = precompute_analytical_mask_params(&CropShape::Ellipse, 20, 12, 0.0);
        assert!(analytical_signed_distance(&CropShape::Ellipse, 10.0, 6.0, &ellipse) < 0.0);
        assert!(analytical_signed_distance(&CropShape::Ellipse, 21.0, 6.0, &ellipse) > 0.0);

        let rounded = precompute_analytical_mask_params(
            &CropShape::RoundedRectangle { radius_pct: 0.2 },
            20,
            12,
            0.0,
        );
        assert!(
            analytical_signed_distance(
                &CropShape::RoundedRectangle { radius_pct: 0.2 },
                10.0,
                6.0,
                &rounded
            ) < 0.0
        );
        assert!(
            analytical_signed_distance(
                &CropShape::RoundedRectangle { radius_pct: 0.2 },
                21.0,
                6.0,
                &rounded
            ) > 0.0
        );

        let chamfered = precompute_analytical_mask_params(
            &CropShape::ChamferedRectangle { size_pct: 0.2 },
            20,
            12,
            0.0,
        );
        assert!(
            analytical_signed_distance(
                &CropShape::ChamferedRectangle { size_pct: 0.2 },
                10.0,
                6.0,
                &chamfered
            ) < 0.0
        );
        assert!(
            analytical_signed_distance(
                &CropShape::ChamferedRectangle { size_pct: 0.2 },
                21.0,
                6.0,
                &chamfered
            ) > 0.0
        );
    }

    #[test]
    fn mask_alpha_from_distance_handles_hard_and_soft_edges() {
        assert_eq!(mask_alpha_from_distance(-1.0, 0.0), 1.0);
        assert_eq!(mask_alpha_from_distance(1.0, 0.0), 0.0);
        assert!((mask_alpha_from_distance(0.0, 10.0) - 0.5).abs() < f32::EPSILON);
        assert_eq!(mask_alpha_from_distance(-10.0, 10.0), 1.0);
        assert_eq!(mask_alpha_from_distance(10.0, 10.0), 0.0);
    }

    #[test]
    fn raster_mask_scale_only_downscales_above_threshold() {
        assert_eq!(raster_mask_scale(2048, 1024), 1.0);
        assert_eq!(raster_mask_scale(1024, 2048), 1.0);
        assert_eq!(raster_mask_scale(4096, 2048), 0.5);
    }

    #[test]
    fn sample_mask_alpha_bilinear_clamps_to_mask_borders() {
        let mask = RgbaImage::from_raw(
            2,
            2,
            vec![
                0, 0, 0, 10, 0, 0, 0, 20, //
                0, 0, 0, 30, 0, 0, 0, 40,
            ],
        )
        .expect("valid test mask");
        let raw = mask.as_raw();

        assert_eq!(
            sample_mask_alpha_bilinear(raw, 2, 2, -10.0, -10.0),
            10.0 / 255.0
        );
        assert_eq!(
            sample_mask_alpha_bilinear(raw, 2, 2, 10.0, 10.0),
            40.0 / 255.0
        );
    }
}
