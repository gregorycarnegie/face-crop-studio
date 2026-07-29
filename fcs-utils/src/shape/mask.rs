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

    /// Signed distances are exact small rationals for the probes below, but
    /// the diagonal and elliptical cases involve a sqrt, so compare with a
    /// tolerance far tighter than any operator swap could survive.
    #[track_caller]
    fn approx(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-4,
            "expected {expected}, got {actual}"
        );
    }

    // -------------------------------------------------------------------
    // Golden values.
    //
    // The sign-only assertions further down establish that a point is inside
    // or outside, which any scaling or offset error preserves. These pin the
    // distances themselves.

    #[test]
    fn analytical_signed_distance_rectangle_exact_values() {
        // 20x10 rectangle: cx = 10, cy = 5, half extents 10 and 5.
        let params = precompute_analytical_mask_params(&CropShape::Rectangle, 20, 10, 0.0);
        let d = |x, y| analytical_signed_distance(&CropShape::Rectangle, x, y, &params);

        // Centre: both deltas negative, so the result is the *larger* of them
        // (the nearest edge), i.e. -5 from the top/bottom edge, not -10.
        approx(d(10.0, 5.0), -5.0);
        // Directly right of the box: 15 - 10 = 5 across, still inside vertically.
        approx(d(25.0, 5.0), 5.0);
        // Diagonally outside both edges: hypot(5, 2) = sqrt(29).
        approx(d(25.0, 12.0), 29f32.sqrt());
    }

    #[test]
    fn analytical_signed_distance_ellipse_exact_values() {
        // 20x12 ellipse: rx = 10, ry = 6, scaled by min(w,h)*0.5 = 6.
        let params = precompute_analytical_mask_params(&CropShape::Ellipse, 20, 12, 0.0);
        let d = |x, y| analytical_signed_distance(&CropShape::Ellipse, x, y, &params);

        // Centre: (0 - 1) * 6 = -6.
        approx(d(10.0, 6.0), -6.0);
        // x = 21 -> normalized 11/10, sqrt(1.21) = 1.1, (1.1 - 1) * 6 = 0.6.
        approx(d(21.0, 6.0), 0.6);
    }

    #[test]
    fn analytical_signed_distance_rounded_rect_exact_values() {
        // radius = min(20,12) * 0.2 = 2.4, so the inner box is 7.6 x 3.6.
        let shape = CropShape::RoundedRectangle { radius_pct: 0.2 };
        let params = precompute_analytical_mask_params(&shape, 20, 12, 0.0);
        let d = |x, y| analytical_signed_distance(&shape, x, y, &params);

        // Centre: -3.6 from the inner box, minus the 2.4 radius.
        approx(d(10.0, 6.0), -6.0);
        // x = 21: 11 - 7.6 = 3.4 outside the inner box, minus the radius.
        approx(d(21.0, 6.0), 1.0);
    }

    #[test]
    fn analytical_signed_distance_chamfered_rect_exact_values() {
        // chamfer = 2.4, so the diagonal cut sits at x + y = 10 + 6 - 2.4 = 13.6.
        let shape = CropShape::ChamferedRectangle { size_pct: 0.2 };
        let params = precompute_analytical_mask_params(&shape, 20, 12, 0.0);
        let d = |x, y| analytical_signed_distance(&shape, x, y, &params);

        approx(d(10.0, 6.0), -6.0);
        // Straight out the side: the box term wins over the diagonal.
        approx(d(21.0, 6.0), 1.0);
        // Near the cut corner the *diagonal* term wins: still inside the box
        // (-1.0) but outside the chamfer, (9 + 5 - 13.6)/sqrt(2) = 0.2828.
        approx(d(19.0, 11.0), 0.4 * FRAC_1_SQRT_2);
    }

    #[test]
    fn analytical_shape_param_scales_and_clamps() {
        // min(20,12) = 12, limit = 6.
        approx(
            analytical_shape_param(&CropShape::RoundedRectangle { radius_pct: 0.2 }, 20.0, 12.0),
            2.4,
        );
        approx(
            analytical_shape_param(&CropShape::ChamferedRectangle { size_pct: 0.1 }, 20.0, 12.0),
            1.2,
        );
        // 12 * 0.9 = 10.8 exceeds the half-extent limit and clamps to 6.
        approx(
            analytical_shape_param(&CropShape::RoundedRectangle { radius_pct: 0.9 }, 20.0, 12.0),
            6.0,
        );
        // Shapes without a corner parameter contribute nothing.
        approx(
            analytical_shape_param(&CropShape::Rectangle, 20.0, 12.0),
            0.0,
        );
        approx(analytical_shape_param(&CropShape::Ellipse, 20.0, 12.0), 0.0);
    }

    #[test]
    fn precompute_analytical_mask_params_derives_centre_and_softness() {
        let p = precompute_analytical_mask_params(&CropShape::Rectangle, 20, 12, 0.5);
        approx(p.width, 20.0);
        approx(p.height, 12.0);
        approx(p.cx, 10.0);
        approx(p.cy, 6.0);
        // min(20,12) * 0.5 * 0.5 = 3.
        approx(p.softness_px, 3.0);

        // Zero softness stays zero rather than falling through the .max(1.0).
        let hard = precompute_analytical_mask_params(&CropShape::Rectangle, 20, 12, 0.0);
        approx(hard.softness_px, 0.0);

        // Tiny images floor at one pixel: 2 * 0.5 * 0.1 = 0.1 -> 1.0.
        let tiny = precompute_analytical_mask_params(&CropShape::Rectangle, 2, 2, 0.1);
        approx(tiny.softness_px, 1.0);
    }

    #[test]
    fn mask_alpha_from_distance_on_the_boundary_is_opaque() {
        // Exactly on the edge with a hard mask. Treating softness_px >= 0 as
        // "soft" divides 0 by 0 here and yields NaN instead of 1.0.
        assert_eq!(mask_alpha_from_distance(0.0, 0.0), 1.0);
    }

    #[test]
    fn sample_mask_alpha_bilinear_interpolates_interior() {
        // 2x2 alphas: 10 20 / 30 40.
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

        // Dead centre averages all four: (10+20+30+40)/4 = 25.
        approx(
            sample_mask_alpha_bilinear(raw, 2, 2, 0.5, 0.5),
            25.0 / 255.0,
        );
        // wx = 0.25, wy = 0.75 -> top 12.5, bottom 32.5 -> 27.5.
        approx(
            sample_mask_alpha_bilinear(raw, 2, 2, 0.25, 0.75),
            27.5 / 255.0,
        );
        // Asymmetric weights catch an x/y swap: wx = 0.75, wy = 0.25 -> 22.5.
        approx(
            sample_mask_alpha_bilinear(raw, 2, 2, 0.75, 0.25),
            22.5 / 255.0,
        );
    }

    #[test]
    fn process_pixel_mixes_vignette_and_scales_alpha() {
        // Distinct channels and a distinct vignette colour so a channel mix-up
        // shows up. mix_factor = inv_mask * intensity = 0.5 * 1.0 = 0.5.
        //   R: 100 + 0.5*(10 - 100)  =  55
        //   G: 150 + 0.5*(20 - 150)  =  85
        //   B: 200 + 0.5*(30 - 200)  = 115
        //   A: 255 * 0.5 = 127.5, rounded = 128
        let colour = RgbaColor {
            red: 10,
            green: 20,
            blue: 30,
            alpha: 255,
        };
        let mut px = [100u8, 150, 200, 255];
        process_pixel(&mut px, 0.5, 1.0, &colour);
        assert_eq!(px, [55, 85, 115, 128]);
    }

    #[test]
    fn process_pixel_scales_the_mix_by_intensity() {
        // The case above uses intensity 1.0, where `inv_mask * intensity` and
        // `inv_mask / intensity` agree. A partial intensity separates them:
        // mix_factor = 0.5 * 0.5 = 0.25, not 1.0.
        //   R: 100 + 0.25*(10 - 100)  =  77.5 -> 77
        //   G: 150 + 0.25*(20 - 150)  = 117.5 -> 117
        //   B: 200 + 0.25*(30 - 200)  = 157.5 -> 157
        let colour = RgbaColor {
            red: 10,
            green: 20,
            blue: 30,
            alpha: 255,
        };
        let mut px = [100u8, 150, 200, 255];
        process_pixel(&mut px, 0.5, 0.5, &colour);
        assert_eq!(px, [77, 117, 157, 128]);
    }

    #[test]
    fn dynamic_rectangle_with_softness_still_gets_a_vignette() {
        // `apply_shape_mask_dynamic` carries its own copy of the rectangle
        // short-circuit, so it needs its own coverage of the softness case.
        let mut img = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            8,
            8,
            image::Rgba([200u8, 100, 50, 255]),
        ));
        apply_shape_mask_dynamic(
            &mut img,
            &CropShape::Rectangle,
            0.5,
            0.0,
            RgbaColor::opaque(0, 0, 0),
        );
        let buf = img.to_rgba8();
        assert_eq!(buf.get_pixel(0, 0)[3], 159);
        assert_eq!(buf.get_pixel(4, 4)[3], 255);
    }

    #[test]
    fn dynamic_rectangle_without_softness_is_untouched() {
        let source = RgbaImage::from_pixel(4, 4, image::Rgba([200u8, 100, 50, 255]));
        let mut img = DynamicImage::ImageRgba8(source.clone());
        apply_shape_mask_dynamic(
            &mut img,
            &CropShape::Rectangle,
            0.0,
            1.0,
            RgbaColor::opaque(0, 0, 0),
        );
        assert_eq!(img.to_rgba8(), source);
    }

    #[test]
    fn process_pixel_leaves_colour_alone_when_fully_inside_or_unvignetted() {
        let colour = RgbaColor {
            red: 10,
            green: 20,
            blue: 30,
            alpha: 255,
        };

        // mask_alpha 1.0 -> inv_mask 0.0 -> no vignette contribution at all.
        let mut inside = [100u8, 150, 200, 200];
        process_pixel(&mut inside, 1.0, 1.0, &colour);
        assert_eq!(inside, [100, 150, 200, 200]);

        // Zero intensity -> colours untouched, but alpha still gets masked.
        let mut unvignetted = [100u8, 150, 200, 200];
        process_pixel(&mut unvignetted, 0.25, 0.0, &colour);
        assert_eq!(unvignetted, [100, 150, 200, 50]);
    }

    #[test]
    fn build_raster_soft_mask_passthrough_and_solid_mask() {
        let hard = RgbaImage::from_pixel(4, 4, image::Rgba([255u8, 255, 255, 128]));

        // Non-positive softness returns the input untouched.
        let same = build_raster_soft_mask(hard.clone(), 0.0);
        assert_eq!(same, hard);

        // A fully opaque mask blurs to itself, so hard_a * soft_a stays 1.0
        // and the alpha survives the round trip at 255.
        let solid = RgbaImage::from_pixel(8, 8, image::Rgba([255u8, 255, 255, 255]));
        let softened = build_raster_soft_mask(solid, 0.5);
        assert_eq!(softened.get_pixel(4, 4)[3], 255);
    }

    #[test]
    fn analytical_signed_distance_ellipse_off_the_horizontal_axis() {
        // The probes above all sit on y = cy, which zeroes the whole vertical
        // term and hides anything wrong with ry.
        let params = precompute_analytical_mask_params(&CropShape::Ellipse, 20, 12, 0.0);
        // p_abs_y = 9, ry = 6 -> 81/36 = 2.25, sqrt = 1.5, (1.5 - 1) * 6 = 3.
        approx(
            analytical_signed_distance(&CropShape::Ellipse, 10.0, 15.0, &params),
            3.0,
        );
    }

    #[test]
    fn analytical_shape_param_clamps_the_chamfer_too() {
        // The rounded case exercises the clamp above; the chamfered branch has
        // its own copy of the limit and needs its own oversized input.
        approx(
            analytical_shape_param(&CropShape::ChamferedRectangle { size_pct: 0.9 }, 20.0, 12.0),
            6.0,
        );
    }

    #[test]
    fn mask_alpha_from_distance_ramps_linearly_across_the_falloff() {
        // Only partial values distinguish `dist / softness` from `dist *
        // softness` — at the extremes both saturate to the same 0 or 1.
        approx(mask_alpha_from_distance(5.0, 10.0), 0.25);
        approx(mask_alpha_from_distance(-5.0, 10.0), 0.75);
        approx(mask_alpha_from_distance(2.5, 10.0), 0.375);
    }

    #[test]
    fn build_raster_soft_mask_multiplies_hard_and_soft_alpha() {
        // A solid mask blurs to itself, so the alpha is squared in normalized
        // space: (128/255)^2 * 255 + 0.5 = 64.75 -> 64. A fully opaque fixture
        // would square to 1.0 and hide the division entirely.
        let half = RgbaImage::from_pixel(8, 8, image::Rgba([255u8, 255, 255, 128]));
        let softened = build_raster_soft_mask(half, 0.5);
        assert_eq!(softened.get_pixel(4, 4)[3], 64);
    }

    #[test]
    fn analytical_mask_samples_at_pixel_centres() {
        // 4x4 ellipse, hard edge. Pixel centres are at 0.5..3.5 about (2, 2),
        // so the four corners fall outside the unit ellipse and everything
        // else falls inside. Dropping the half-pixel offset shifts the whole
        // pattern by one pixel and lights up the wrong corners.
        let mut img = RgbaImage::from_pixel(4, 4, image::Rgba([200u8, 100, 50, 255]));
        apply_shape_mask(
            &mut img,
            &CropShape::Ellipse,
            0.0,
            0.0,
            RgbaColor::opaque(0, 0, 0),
        );

        for y in 0..4u32 {
            for x in 0..4u32 {
                let corner = (x == 0 || x == 3) && (y == 0 || y == 3);
                let want = if corner { 0 } else { 255 };
                assert_eq!(
                    img.get_pixel(x, y)[3],
                    want,
                    "alpha at ({x}, {y}) should be {want}"
                );
            }
        }
    }

    #[test]
    fn rectangle_with_softness_still_gets_a_vignette() {
        // A rectangle short-circuits only when softness is zero. With softness
        // the analytical path must run and feather the border.
        let mut img = RgbaImage::from_pixel(8, 8, image::Rgba([200u8, 100, 50, 255]));
        apply_shape_mask(
            &mut img,
            &CropShape::Rectangle,
            0.5,
            0.0,
            RgbaColor::opaque(0, 0, 0),
        );

        // softness_px = 8 * 0.5 * 0.5 = 2. The corner pixel centre sits 0.5
        // inside the edge: alpha = 0.5 + 0.5*(0.5/2) = 0.625 -> 159.
        assert_eq!(img.get_pixel(0, 0)[3], 159);
        // The centre is 3.5 in, well past the falloff, so it stays opaque.
        assert_eq!(img.get_pixel(4, 4)[3], 255);
    }

    #[test]
    fn rectangle_without_softness_is_untouched() {
        let mut img = RgbaImage::from_pixel(4, 4, image::Rgba([200u8, 100, 50, 255]));
        let before = img.clone();
        apply_shape_mask(
            &mut img,
            &CropShape::Rectangle,
            0.0,
            1.0,
            RgbaColor::opaque(0, 0, 0),
        );
        assert_eq!(img, before);
    }

    #[test]
    fn zero_sized_images_are_left_alone_on_both_paths() {
        // Row chunking uses `4 * width`, which rejects a zero chunk size, so
        // the guard has to fire before the parallel loop.
        let mut analytical = RgbaImage::new(0, 4);
        apply_shape_mask(
            &mut analytical,
            &CropShape::Ellipse,
            0.0,
            0.0,
            RgbaColor::opaque(0, 0, 0),
        );
        assert_eq!(analytical.dimensions(), (0, 4));

        let mut raster = RgbaImage::new(0, 4);
        apply_shape_mask(
            &mut raster,
            &CropShape::Star {
                points: 5,
                inner_radius_pct: 0.5,
                rotation_deg: 0.0,
            },
            0.0,
            0.0,
            RgbaColor::opaque(0, 0, 0),
        );
        assert_eq!(raster.dimensions(), (0, 4));
    }

    /// Sequential re-derivation of the raster masking loop.
    ///
    /// The mask itself comes from tiny-skia, whose antialiased coverage is not
    /// something to hand-predict — and it is not perfectly mirror-symmetric
    /// either, so a symmetry assertion cannot stand in for it. What this pins
    /// instead is everything the loop does *around* the rasterizer: deriving
    /// the mask size from the scale, mapping pixel centres to mask
    /// coordinates, and indexing rows. The shared helpers it calls are covered
    /// by their own golden tests above.
    fn reference_raster_mask(
        image: &RgbaImage,
        shape: &CropShape,
        vignette_softness: f32,
        vignette_intensity: f32,
        vignette_color: RgbaColor,
    ) -> RgbaImage {
        let (width, height) = image.dimensions();
        let scale = raster_mask_scale(width, height);
        let mask_w = (width as f32 * scale).ceil() as u32;
        let mask_h = (height as f32 * scale).ceil() as u32;

        let hard = build_raster_hard_mask(mask_w, mask_h, shape).expect("mask builds");
        let mask = build_raster_soft_mask(hard, vignette_softness);
        let raw = mask.as_raw();

        let mut out = image.clone();
        for y in 0..height {
            for x in 0..width {
                let u = (x as f32 + 0.5) * scale;
                let v = (y as f32 + 0.5) * scale;
                let alpha = sample_mask_alpha_bilinear(
                    raw,
                    mask_w as usize,
                    mask_h as usize,
                    u - 0.5,
                    v - 0.5,
                );
                let px = out.get_pixel_mut(x, y);
                process_pixel(&mut px.0[..], alpha, vignette_intensity, &vignette_color);
            }
        }
        out
    }

    fn star() -> CropShape {
        CropShape::Star {
            points: 4,
            inner_radius_pct: 0.5,
            rotation_deg: 0.0,
        }
    }

    #[test]
    fn raster_mask_matches_the_reference_loop() {
        let source = RgbaImage::from_pixel(16, 16, image::Rgba([200u8, 100, 50, 255]));
        let colour = RgbaColor::opaque(10, 20, 30);

        for (softness, intensity) in [(0.0, 0.0), (0.0, 0.8), (0.4, 0.6)] {
            let mut got = source.clone();
            apply_shape_mask(&mut got, &star(), softness, intensity, colour);
            let want = reference_raster_mask(&source, &star(), softness, intensity, colour);
            assert_eq!(
                got, want,
                "mismatch at softness {softness}, intensity {intensity}"
            );
        }

        // Sanity: the star must actually carve something out, otherwise the
        // comparison above would hold on an untouched image.
        let mut carved = source.clone();
        apply_shape_mask(&mut carved, &star(), 0.0, 0.0, colour);
        assert!(carved.pixels().any(|p| p[3] == 0));
        assert!(carved.pixels().any(|p| p[3] == 255));
    }

    #[test]
    fn raster_mask_downscales_above_the_resolution_cap() {
        // Wider than MAX_MASK_RESOLUTION, so scale = 0.5 and the mask is built
        // at half size. That makes `* scale` and `/ scale` diverge, which they
        // cannot at the scale of 1.0 every other test uses.
        let source = RgbaImage::from_pixel(4096, 8, image::Rgba([200u8, 100, 50, 255]));
        assert_eq!(raster_mask_scale(4096, 8), 0.5);

        let mut got = source.clone();
        apply_shape_mask(&mut got, &star(), 0.0, 0.0, RgbaColor::opaque(0, 0, 0));
        let want = reference_raster_mask(&source, &star(), 0.0, 0.0, RgbaColor::opaque(0, 0, 0));

        assert_eq!(got.dimensions(), (4096, 8));
        assert_eq!(got, want);
        assert!(got.pixels().any(|p| p[3] == 0));
    }

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
