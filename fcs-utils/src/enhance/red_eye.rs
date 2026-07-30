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

#[cfg(test)]
mod tests {
    use super::*;

    /// The shared tests in `enhance/tests.rs` can only reach the public entry
    /// point, and assert things like "red was reduced from 200". That holds
    /// for almost any replacement value and for a correction region of almost
    /// any shape. These reach the private helpers directly and pin both.
    fn eye(x: f32, y: f32, radius: f32) -> RedEye {
        RedEye {
            x,
            y,
            radius,
            _pad: 0.0,
        }
    }

    fn canvas(w: u32, h: u32) -> RgbaImage {
        // Every pixel is red-dominant enough to be corrected, so whichever
        // ones come back unchanged map out the region that was tested.
        RgbaImage::from_pixel(w, h, image::Rgba([200, 20, 80, 255]))
    }

    #[test]
    fn correct_red_pixel_replaces_red_with_the_green_blue_mean() {
        // avg_gb = (20 + 80) * 0.5 = 50, and 200 > 50 * 1.5 and > 80, so the
        // red channel becomes 50. Green and blue are left alone, and the
        // asymmetric pair catches a mangled mean.
        let mut px = [200u8, 20, 80, 255];
        correct_red_pixel(&mut px, 1.5);
        assert_eq!(px, [50, 20, 80, 255]);
    }

    #[test]
    fn correct_red_pixel_needs_red_above_the_absolute_floor() {
        // The `r > 80.0` guard, probed either side of the boundary. Both
        // pixels clear the ratio test, so only the floor separates them.
        let mut just_under = [79u8, 10, 10, 255];
        correct_red_pixel(&mut just_under, 1.5);
        assert_eq!(just_under, [79, 10, 10, 255], "79 is below the floor");

        let mut just_over = [81u8, 10, 10, 255];
        correct_red_pixel(&mut just_over, 1.5);
        assert_eq!(just_over, [10, 10, 10, 255], "81 clears the floor");
    }

    #[test]
    fn correct_red_pixel_scales_the_ratio_test_by_the_threshold() {
        // avg_gb = 50. At 1.5 the bar is 75 and 100 clears it; at 2.5 the bar
        // is 125 and the same pixel is left alone.
        let mut lenient = [100u8, 50, 50, 255];
        correct_red_pixel(&mut lenient, 1.5);
        assert_eq!(lenient, [50, 50, 50, 255]);

        let mut strict = [100u8, 50, 50, 255];
        correct_red_pixel(&mut strict, 2.5);
        assert_eq!(strict, [100, 50, 50, 255]);
    }

    #[test]
    fn correct_red_pixel_floor_is_strictly_above_eighty() {
        // Exactly 80 must not qualify — the guard is `>`, not `>=`.
        let mut boundary = [80u8, 10, 10, 255];
        correct_red_pixel(&mut boundary, 1.5);
        assert_eq!(boundary, [80, 10, 10, 255]);
    }

    #[test]
    fn correct_red_eye_region_handles_a_zero_radius_eye() {
        // A radius of zero collapses the bounding box to a single pixel, where
        // min and max coincide. The early return has to trigger on min being
        // strictly greater than max, not on equality.
        let mut img = canvas(5, 5);
        correct_red_eye_region(&mut img, 1.5, &eye(0.0, 0.0, 0.0));
        assert_eq!(img.get_pixel(0, 0).0, [50, 20, 80, 255], "the single pixel");
        assert_eq!(
            img.get_pixel(1, 0).0,
            [200, 20, 80, 255],
            "and only that one"
        );
    }

    #[test]
    fn correct_red_eye_region_clamps_a_box_running_off_the_right_and_bottom() {
        // The box for this eye reaches x = 8 and y = 8 on a 7x7 image, so the
        // `w - 1` / `h - 1` ceilings have to bind. Without them the row slice
        // is indexed past its end.
        let mut img = canvas(7, 7);
        correct_red_eye_region(&mut img, 1.5, &eye(6.0, 6.0, 2.0));

        assert_eq!(img.get_pixel(6, 6).0, [50, 20, 80, 255], "centre");
        assert_eq!(img.get_pixel(4, 6).0, [50, 20, 80, 255], "two left");
        assert_eq!(img.get_pixel(6, 4).0, [50, 20, 80, 255], "two up");
        assert_eq!(img.get_pixel(5, 5).0, [50, 20, 80, 255], "diagonal, inside");
        assert_eq!(
            img.get_pixel(4, 5).0,
            [200, 20, 80, 255],
            "outside the disc"
        );
    }

    #[test]
    fn correct_red_pixel_is_idempotent() {
        // Documented behaviour: overlapping eye regions may re-apply it.
        let mut px = [200u8, 20, 80, 255];
        correct_red_pixel(&mut px, 1.5);
        let once = px;
        correct_red_pixel(&mut px, 1.5);
        assert_eq!(px, once);
    }

    #[test]
    fn correct_red_eye_region_corrects_exactly_the_enclosed_disc() {
        // Radius 2 about (3, 3) on a 7x7 canvas. The membership test is
        // `dx^2 + dy^2 <= 4`, which includes the axis pixels two out but
        // excludes the (1,2) and (2,2) diagonals — a diamond, not the 5x5
        // bounding box the loop actually walks.
        let mut img = canvas(7, 7);
        correct_red_eye_region(&mut img, 1.5, &eye(3.0, 3.0, 2.0));

        let inside = |dx: i32, dy: i32| dx * dx + dy * dy <= 4;
        let mut corrected = 0;
        for y in 0..7i32 {
            for x in 0..7i32 {
                let px = img.get_pixel(x as u32, y as u32).0;
                if inside(x - 3, y - 3) {
                    assert_eq!(px, [50, 20, 80, 255], "({x}, {y}) should be corrected");
                    corrected += 1;
                } else {
                    assert_eq!(px, [200, 20, 80, 255], "({x}, {y}) should be untouched");
                }
            }
        }
        assert_eq!(corrected, 13, "the disc covers 13 of the 49 pixels");
    }

    #[test]
    fn correct_red_eye_region_clamps_to_the_image_edges() {
        // An eye at the corner: the box would run negative and past the right
        // edge, so both ends have to clamp rather than wrap or panic.
        let mut img = canvas(5, 5);
        correct_red_eye_region(&mut img, 1.5, &eye(0.0, 0.0, 2.0));

        assert_eq!(img.get_pixel(0, 0).0, [50, 20, 80, 255], "centre");
        assert_eq!(img.get_pixel(2, 0).0, [50, 20, 80, 255], "two out on x");
        assert_eq!(img.get_pixel(0, 2).0, [50, 20, 80, 255], "two out on y");
        // Outside the radius, still inside the image.
        assert_eq!(img.get_pixel(2, 2).0, [200, 20, 80, 255]);
        assert_eq!(img.get_pixel(4, 4).0, [200, 20, 80, 255]);
    }

    #[test]
    fn correct_red_eye_region_ignores_an_eye_off_the_image() {
        let mut img = canvas(5, 5);
        let before = img.clone();
        correct_red_eye_region(&mut img, 1.5, &eye(100.0, 100.0, 1.0));
        assert_eq!(
            img, before,
            "an eye past the right/bottom edge does nothing"
        );

        correct_red_eye_region(&mut img, 1.5, &eye(-20.0, 2.0, 1.0));
        assert_eq!(img, before, "an eye past the left edge does nothing");
    }

    #[test]
    fn red_eye_in_place_scans_everything_without_eye_locations() {
        let mut img = canvas(3, 2);
        red_eye_in_place(&mut img, 1.5, None);
        for px in img.pixels() {
            assert_eq!(px.0, [50, 20, 80, 255]);
        }
    }

    #[test]
    fn red_eye_in_place_treats_an_empty_eye_list_as_no_locations() {
        // `eyes.filter(|l| !l.is_empty())` has to fall through to the
        // whole-image scan, not skip the correction entirely.
        let mut img = canvas(3, 2);
        red_eye_in_place(&mut img, 1.5, Some(&[]));
        for px in img.pixels() {
            assert_eq!(px.0, [50, 20, 80, 255], "an empty list means scan it all");
        }
    }

    #[test]
    fn red_eye_in_place_restricts_to_the_listed_eyes() {
        let mut img = canvas(7, 7);
        red_eye_in_place(&mut img, 1.5, Some(&[eye(3.0, 3.0, 1.0)]));
        // Radius 1 leaves a plus of five pixels.
        assert_eq!(img.get_pixel(3, 3).0, [50, 20, 80, 255]);
        assert_eq!(img.get_pixel(2, 3).0, [50, 20, 80, 255]);
        assert_eq!(img.get_pixel(3, 2).0, [50, 20, 80, 255]);
        assert_eq!(img.get_pixel(2, 2).0, [200, 20, 80, 255], "diagonal is out");
        assert_eq!(
            img.get_pixel(0, 0).0,
            [200, 20, 80, 255],
            "far corner is out"
        );
    }

    #[test]
    fn red_eye_in_place_guards_zero_sized_images() {
        // With eye locations the region walker computes `w - 1`, which
        // underflows if a zero-width image is not rejected first.
        let mut empty = RgbaImage::new(0, 4);
        red_eye_in_place(&mut empty, 1.5, Some(&[eye(0.0, 0.0, 1.0)]));
        assert_eq!(empty.dimensions(), (0, 4));

        let mut flat = RgbaImage::new(4, 0);
        red_eye_in_place(&mut flat, 1.5, Some(&[eye(0.0, 0.0, 1.0)]));
        assert_eq!(flat.dimensions(), (4, 0));
    }
}
