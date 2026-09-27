//! Automated red-eye correction.
//!
//! Three steps per region (an eye disc, or the whole image when no eyes are known):
//!
//! 1. The core: pixels whose red clears `threshold` times the green/blue mean.
//! 2. The fringe: pixels within [`FRINGE_RADIUS`] of the core, corrected in proportion to
//!    how red they are. Antialiasing and JPEG leave a half-red ring round a pupil that the
//!    hard test misses, and correcting only the core leaves that ring standing out.
//! 3. The black level: the darkest non-red pixels round the eye. A red pupil with its red
//!    removed is near 0, which reads as a black hole in a low-contrast photo whose darkest
//!    shade is a grey, so the corrected pixels are kept no darker than that grey.
//!
//! Steps 2 and 3 need neighbours and region statistics, so this runs on the CPU on both
//! paths. It only walks the eye discs, which is cheaper than a GPU upload and readback.

use crate::gpu::red_eye::RedEye;

use super::EPSILON;
use image::{DynamicImage, RgbaImage};

/// How far past the red core, in pixels, the fringe correction reaches.
const FRINGE_RADIUS: usize = 2;

/// Apply automated red-eye reduction.
///
/// Replaces red with the green/blue mean where red dominates, blends the half-red fringe
/// round each red area, and keeps the result no darker than the black level round it.
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
        // With known eye locations only the pixels inside each eye's disc are considered,
        // and each eye gets its own black level.
        Some(eyes_list) => {
            for eye in eyes_list {
                correct_red_eye_region(out, threshold, eye);
            }
        }
        None => correct_region(out, threshold, (0, 0, w - 1, h - 1), |_, _| true),
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

    let radius_sq = eye.radius * eye.radius;
    correct_region(out, threshold, (min_x, min_y, max_x, max_y), |x, y| {
        let dx = x as f32 - eye.x;
        let dy = y as f32 - eye.y;
        // Plain multiply-add (not fused) to match the original membership
        // test bit-for-bit on boundary pixels.
        dx * dx + dy * dy <= radius_sq
    });
}

/// Correct the pixels of the inclusive box `(x0, y0, x1, y1)` for which `inside` holds.
fn correct_region(
    out: &mut RgbaImage,
    threshold: f32,
    (x0, y0, x1, y1): (u32, u32, u32, u32),
    inside: impl Fn(u32, u32) -> bool,
) {
    if x0 > x1 || y0 > y1 {
        return;
    }
    let rw = (x1 - x0 + 1) as usize;
    let rh = (y1 - y0 + 1) as usize;
    let xy = |i: usize| (x0 + (i % rw) as u32, y0 + (i / rw) as u32);

    let member: Vec<bool> = (0..rw * rh)
        .map(|i| {
            let (x, y) = xy(i);
            inside(x, y)
        })
        .collect();
    let core: Vec<bool> = (0..rw * rh)
        .map(|i| member[i] && is_red(out.get_pixel(xy(i).0, xy(i).1).0, threshold))
        .collect();
    if !core.contains(&true) {
        return;
    }

    // The core grown by FRINGE_RADIUS, square rather than round: the fringe weight, not the
    // shape, decides how much each of these pixels changes.
    let mut near = vec![false; rw * rh];
    for i in (0..rw * rh).filter(|&i| core[i]) {
        let (cx, cy) = (i % rw, i / rw);
        for ny in cy.saturating_sub(FRINGE_RADIUS)..=(cy + FRINGE_RADIUS).min(rh - 1) {
            for nx in cx.saturating_sub(FRINGE_RADIUS)..=(cx + FRINGE_RADIUS).min(rw - 1) {
                near[ny * rw + nx] = true;
            }
        }
    }

    let black = black_level(
        (0..rw * rh)
            .filter(|&i| member[i] && !near[i])
            .map(|i| out.get_pixel(xy(i).0, xy(i).1).0),
    );

    for i in (0..rw * rh).filter(|&i| member[i] && near[i]) {
        let (x, y) = xy(i);
        let px = out.get_pixel_mut(x, y);
        let weight = if core[i] {
            1.0
        } else {
            fringe_weight(px.0, threshold)
        };
        if weight > 0.0 {
            px.0 = corrected(px.0, black, weight);
        }
    }
}

/// The red-eye core test: red above `threshold` times the green/blue mean, and above an
/// absolute floor so dark reddish-brown irises are left alone.
#[inline]
fn is_red(px: [u8; 4], threshold: f32) -> bool {
    let r = px[0] as f32;
    // Check red dominance without dividing by the green/blue average.
    let avg_gb = (px[1] as f32 + px[2] as f32).mul_add(0.5, EPSILON);
    r > avg_gb * threshold && r > 80.0
}

/// How much of the correction a fringe pixel gets: 0 for red no stronger than two thirds
/// of `threshold` (neutral at the default 1.5), rising to 1 at `threshold`. No absolute
/// floor, since a fringe pixel is a dim mix of red pupil and iris.
fn fringe_weight(px: [u8; 4], threshold: f32) -> f32 {
    let avg_gb = (px[1] as f32 + px[2] as f32).mul_add(0.5, EPSILON);
    let start = threshold * (2.0 / 3.0);
    ((px[0] as f32 / avg_gb - start) / (threshold - start).max(EPSILON)).clamp(0.0, 1.0)
}

/// Mean colour of the darkest 5% of `pixels` by luma; black when there are none.
fn black_level(pixels: impl Iterator<Item = [u8; 4]>) -> [f32; 3] {
    let mut pixels: Vec<[u8; 4]> = pixels.collect();
    if pixels.is_empty() {
        return [0.0; 3];
    }
    let k = (pixels.len() / 20).max(1);
    pixels.select_nth_unstable_by_key(k - 1, |p| {
        299 * p[0] as u32 + 587 * p[1] as u32 + 114 * p[2] as u32
    });
    let mut sum = [0.0f32; 3];
    for p in &pixels[..k] {
        for c in 0..3 {
            sum[c] += p[c] as f32;
        }
    }
    sum.map(|s| s / k as f32)
}

/// `px` with red replaced by the green/blue mean, no darker than `black` per channel, and
/// blended in by `weight`. Alpha is kept.
///
/// A clamp rather than a lift of the whole range: green and blue are already in the photo's
/// tones and only the values under its black level are wrong, which in a red pupil is all of
/// them. A catchlight or a fringe pixel keeps its own brightness.
fn corrected(px: [u8; 4], black: [f32; 3], weight: f32) -> [u8; 4] {
    let g = px[1] as f32;
    let b = px[2] as f32;
    let target = [(g + b) * 0.5, g, b];
    let mut out = px;
    for c in 0..3 {
        let lifted = target[c].max(black[c]);
        let v = px[c] as f32;
        out[c] = (v + (lifted - v) * weight).round().clamp(0.0, 255.0) as u8;
    }
    out
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
        // ones come back unchanged map out the region that was tested. With
        // no non-red pixel anywhere the black level is 0.
        RgbaImage::from_pixel(w, h, image::Rgba([200, 20, 80, 255]))
    }

    #[test]
    fn corrected_replaces_red_with_the_green_blue_mean_on_a_zero_black_level() {
        // (20 + 80) / 2 = 50. Green and blue are left alone, and the
        // asymmetric pair catches a mangled mean.
        assert_eq!(
            corrected([200, 20, 80, 255], [0.0; 3], 1.0),
            [50, 20, 80, 255]
        );
    }

    #[test]
    fn corrected_stops_at_the_black_level_and_blends_by_weight() {
        assert_eq!(
            corrected([200, 20, 20, 128], [70.0; 3], 1.0),
            [70, 70, 70, 128]
        );
        // Half weight lands halfway.
        assert_eq!(
            corrected([200, 20, 20, 255], [70.0; 3], 0.5),
            [135, 45, 45, 255]
        );
        // Channels already above the black level keep their value.
        assert_eq!(
            corrected([200, 100, 120, 255], [70.0; 3], 1.0),
            [110, 100, 120, 255]
        );
    }

    #[test]
    fn is_red_needs_red_above_the_absolute_floor() {
        // Both pixels clear the ratio test, so only the floor separates them,
        // and exactly 80 must not qualify.
        assert!(!is_red([79, 10, 10, 255], 1.5));
        assert!(!is_red([80, 10, 10, 255], 1.5));
        assert!(is_red([81, 10, 10, 255], 1.5));
    }

    #[test]
    fn is_red_scales_the_ratio_test_by_the_threshold() {
        // avg_gb = 50. At 1.5 the bar is 75 and 100 clears it; at 2.5 it is 125.
        assert!(is_red([100, 50, 50, 255], 1.5));
        assert!(!is_red([100, 50, 50, 255], 2.5));
    }

    #[test]
    fn is_red_leaves_a_ratio_exactly_at_the_threshold() {
        // avg_gb = 50.0 exactly (the 1e-6 epsilon is under half an f32 ulp at 50 and rounds
        // away), so 100 sits on the line at threshold 2.0, not over it.
        assert!(!is_red([100, 40, 60, 255], 2.0));
    }

    #[test]
    fn fringe_weight_ramps_from_neutral_to_the_threshold() {
        assert_eq!(fringe_weight([100, 100, 100, 255], 1.5), 0.0, "neutral");
        assert_eq!(fringe_weight([80, 100, 100, 255], 1.5), 0.0, "cyan-ish");
        assert!((fringe_weight([125, 100, 100, 255], 1.5) - 0.5).abs() < 1e-4);
        assert_eq!(fringe_weight([200, 100, 100, 255], 1.5), 1.0, "past it");
    }

    #[test]
    fn a_corrected_pixel_is_no_longer_red() {
        // Overlapping eye regions re-apply the correction, so its output must
        // not qualify as a core a second time.
        let once = corrected([200, 20, 80, 255], [60.0, 55.0, 50.0], 1.0);
        assert!(!is_red(once, 1.5));
        assert_eq!(fringe_weight(once, 1.5), 0.0);
    }

    #[test]
    fn black_level_is_the_mean_of_the_darkest_twentieth() {
        assert_eq!(black_level(std::iter::empty()), [0.0; 3]);
        // 40 pixels: the darkest 2 are averaged, the rest ignored.
        let mut px = vec![[200u8, 200, 200, 255]; 38];
        px.push([10, 20, 30, 255]);
        px.push([30, 40, 50, 255]);
        assert_eq!(black_level(px.into_iter()), [20.0, 30.0, 40.0]);
    }

    /// A low-contrast photo: nothing round the eye is darker than 70, and the
    /// pupil is red with a half-red fringe.
    fn washed_out_eye() -> RgbaImage {
        let mut img = RgbaImage::from_pixel(21, 21, image::Rgba([150, 150, 150, 255]));
        for y in 5..=15 {
            for x in 5..=15 {
                img.put_pixel(x, y, image::Rgba([70, 70, 70, 255]));
            }
        }
        for y in 9..=11 {
            for x in 9..=11 {
                img.put_pixel(x, y, image::Rgba([200, 20, 20, 255]));
            }
        }
        img.put_pixel(12, 10, image::Rgba([130, 100, 100, 255]));
        // The same half-red colour, far from any red core.
        img.put_pixel(3, 10, image::Rgba([130, 100, 100, 255]));
        img
    }

    #[test]
    fn the_pupil_lands_on_the_local_black_level_not_zero() {
        let mut img = washed_out_eye();
        correct_red_eye_region(&mut img, 1.5, &eye(10.0, 10.0, 9.0));
        // The darkest twentieth round the eye is the 70 grey, not the pupil's 20.
        assert_eq!(img.get_pixel(10, 10).0, [70, 70, 70, 255]);
    }

    #[test]
    fn the_fringe_is_corrected_only_next_to_the_core() {
        let mut img = washed_out_eye();
        correct_red_eye_region(&mut img, 1.5, &eye(10.0, 10.0, 9.0));
        // Ratio 1.3 at threshold 1.5 is weight 0.6: 130 - 0.6 * 30 = 112.
        assert_eq!(
            img.get_pixel(12, 10).0,
            [112, 100, 100, 255],
            "the fringe next to the pupil loses most of its red"
        );
        assert_eq!(
            img.get_pixel(3, 10).0,
            [130, 100, 100, 255],
            "the same colour away from any red core is left alone"
        );
    }

    #[test]
    fn correct_red_eye_region_handles_a_zero_radius_eye() {
        // A radius of zero collapses the bounding box to a single pixel, where
        // min and max coincide, and that one-pixel range must still be walked.
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
        // `w - 1` / `h - 1` ceilings have to bind. Without them the pixel
        // lookups run past the image.
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
    fn correct_red_eye_region_corrects_exactly_the_enclosed_disc() {
        // Radius 2 about (3, 3) on a 7x7 canvas. The membership test is
        // `dx^2 + dy^2 <= 4`, which includes the axis pixels two out but
        // excludes the (1,2) and (2,2) diagonals — a diamond, not the 5x5
        // bounding box the loop actually walks, and the fringe must not
        // leak past it either.
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
