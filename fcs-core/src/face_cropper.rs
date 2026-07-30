//! Face extraction and resizing utilities.
//!
//! Provides a simple `crop_face_from_image` helper that ties a `Detection` to
//! a `CropSettings` and returns an owned `DynamicImage` sized to the requested output.

use crate::{
    cropper::{CropRegion, CropSettings, calculate_crop_region},
    postprocess::Detection,
};

use image::{DynamicImage, GenericImageView, Rgba, RgbaImage, imageops::FilterType};
use imageproc::geometric_transformations::{Border, Interpolation, rotate_about_center};

/// Crop a face from `img` according to `detection` and `settings`.
///
/// The returned image is resized to `settings.output_width` x `settings.output_height`.
pub fn crop_face_from_image(
    img: &DynamicImage,
    detection: &Detection,
    settings: &CropSettings,
) -> DynamicImage {
    let (img_w, img_h) = img.dimensions();

    let region: CropRegion = calculate_crop_region(img_w, img_h, detection.bbox, settings);

    let canvas_width = region.width.max(1);
    let canvas_height = region.height.max(1);
    let fill = Rgba([
        settings.fill_color.red,
        settings.fill_color.green,
        settings.fill_color.blue,
        settings.fill_color.alpha,
    ]);
    let mut canvas = RgbaImage::from_pixel(canvas_width, canvas_height, fill);

    if let Some((src_x, src_y, src_w, src_h)) = region
        .in_bounds_rect(img_w, img_h)
        .filter(|(_, _, w, h)| *w > 0 && *h > 0)
    {
        let sub = image::imageops::crop_imm(img, src_x, src_y, src_w, src_h).to_image();
        let offset_x = region.pad_left.min(canvas_width.saturating_sub(1));
        let offset_y = region.pad_top.min(canvas_height.saturating_sub(1));
        for y in 0..sub.height() {
            for x in 0..sub.width() {
                let dest_x = offset_x + x;
                let dest_y = offset_y + y;
                if dest_x < canvas_width && dest_y < canvas_height {
                    let pixel = sub.get_pixel(x, y);
                    canvas.put_pixel(dest_x, dest_y, *pixel);
                }
            }
        }
    }

    // If output dimensions are zero, return the raw (possibly padded) crop as DynamicImage.
    if settings.output_width == 0 || settings.output_height == 0 {
        return DynamicImage::ImageRgba8(canvas);
    }

    let canvas = if settings.eye_line_align {
        let re = &detection.landmarks[0]; // right eye (viewer's right)
        let le = &detection.landmarks[1]; // left eye  (viewer's left)
        let both_zero = re.x == 0.0 && re.y == 0.0 && le.x == 0.0 && le.y == 0.0;
        if !both_zero {
            // Angle of the eye line relative to horizontal in source image coords.
            // Positive angle = right eye is above left eye; we rotate by -angle to level.
            let dx = le.x - re.x;
            let dy = le.y - re.y;
            let angle = dy.atan2(dx); // radians; counter-clockwise positive
            let fill = Rgba([
                settings.fill_color.red,
                settings.fill_color.green,
                settings.fill_color.blue,
                settings.fill_color.alpha,
            ]);
            // rotate_about_center rotates counter-clockwise, so pass -angle to level the eyes.
            rotate_about_center(
                &canvas,
                -angle,
                Interpolation::Bilinear,
                Border::Constant(fill),
            )
        } else {
            canvas
        }
    } else {
        canvas
    };

    let resized = image::imageops::resize(
        &DynamicImage::ImageRgba8(canvas),
        settings.output_width,
        settings.output_height,
        FilterType::Lanczos3,
    );

    DynamicImage::ImageRgba8(resized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cropper::{CropSettings, FillColor},
        postprocess::BoundingBox,
    };
    use image::{DynamicImage, Rgba, RgbaImage};

    #[test]
    fn crop_face_resizes_to_output() {
        // Create a simple synthetic image with a neutral color
        let mut img = RgbaImage::from_pixel(800, 600, Rgba([128u8, 128u8, 128u8, 255u8]));
        // draw a bright square where face would be (not necessary for this test but helpful)
        for y in 250..350 {
            for x in 350..450 {
                img.put_pixel(x, y, Rgba([200u8, 100u8, 100u8, 255u8]));
            }
        }

        let img_dyn = DynamicImage::ImageRgba8(img);

        let detection = Detection {
            bbox: BoundingBox {
                x: 350.0,
                y: 250.0,
                width: 100.0,
                height: 100.0,
            },
            landmarks: [
                crate::postprocess::Landmark { x: 360.0, y: 260.0 },
                crate::postprocess::Landmark { x: 390.0, y: 260.0 },
                crate::postprocess::Landmark { x: 375.0, y: 285.0 },
                crate::postprocess::Landmark { x: 365.0, y: 310.0 },
                crate::postprocess::Landmark { x: 385.0, y: 310.0 },
            ],
            score: 0.95,
        };

        let settings = CropSettings {
            output_width: 200,
            output_height: 300,
            face_height_pct: 60.0,
            positioning_mode: crate::cropper::PositioningMode::Center,
            horizontal_offset: 0.0,
            vertical_offset: 0.0,
            fill_color: crate::cropper::FillColor::default(),
            eye_line_align: false,
        };

        let out = crop_face_from_image(&img_dyn, &detection, &settings);
        assert_eq!(out.width(), 200);
        assert_eq!(out.height(), 300);
    }

    /// Landmarks that are all zero, which switches the eye-line branch off.
    fn no_landmarks() -> [crate::postprocess::Landmark; 5] {
        [crate::postprocess::Landmark { x: 0.0, y: 0.0 }; 5]
    }

    fn detection_at(bbox: BoundingBox) -> Detection {
        Detection {
            bbox,
            landmarks: no_landmarks(),
            score: 0.9,
        }
    }

    /// Source pixels carrying their own coordinates, so a misplaced copy is
    /// visible rather than just "some colour".
    fn coded_source(w: u32, h: u32) -> DynamicImage {
        DynamicImage::ImageRgba8(RgbaImage::from_fn(w, h, |x, y| {
            Rgba([(x * 10 + 1) as u8, (y * 10 + 1) as u8, 7, 255])
        }))
    }

    #[test]
    fn zero_output_dimensions_return_the_unresized_canvas() {
        // The early return hands back the padded crop at its natural size
        // instead of resizing it to nothing.
        let img = coded_source(8, 8);
        let detection = detection_at(BoundingBox {
            x: 2.0,
            y: 2.0,
            width: 4.0,
            height: 4.0,
        });
        let settings = CropSettings {
            output_width: 0,
            output_height: 0,
            face_height_pct: 100.0,
            positioning_mode: crate::cropper::PositioningMode::Center,
            horizontal_offset: 0.0,
            vertical_offset: 0.0,
            fill_color: FillColor::opaque(1, 2, 3),
            eye_line_align: false,
        };

        let region = calculate_crop_region(8, 8, detection.bbox, &settings);
        let out = crop_face_from_image(&img, &detection, &settings);
        assert_eq!(out.width(), region.width.max(1));
        assert_eq!(out.height(), region.height.max(1));

        // A single zero dimension is enough to take the same path.
        let half = CropSettings {
            output_width: 16,
            output_height: 0,
            ..settings.clone()
        };
        let out = crop_face_from_image(&img, &detection, &half);
        assert_eq!(out.width(), region.width.max(1));
    }

    #[test]
    fn source_pixels_land_at_the_padding_offset() {
        // A bbox hanging off the top-left forces padding on those two sides.
        // With resizing disabled the canvas is the raw crop, so the copied
        // block has to start exactly at (pad_left, pad_top) and carry the
        // source pixel values from the in-bounds rectangle.
        let img = coded_source(8, 8);
        let detection = detection_at(BoundingBox {
            x: -3.0,
            y: -3.0,
            width: 6.0,
            height: 6.0,
        });
        let settings = CropSettings {
            output_width: 0,
            output_height: 0,
            face_height_pct: 100.0,
            positioning_mode: crate::cropper::PositioningMode::Center,
            horizontal_offset: 0.0,
            vertical_offset: 0.0,
            fill_color: FillColor::opaque(9, 9, 9),
            eye_line_align: false,
        };

        let region = calculate_crop_region(8, 8, detection.bbox, &settings);
        let (src_x, src_y, src_w, src_h) = region
            .in_bounds_rect(8, 8)
            .expect("part of the region overlaps the image");
        assert!(
            region.pad_left > 0 || region.pad_top > 0,
            "expected padding"
        );

        let out = crop_face_from_image(&img, &detection, &settings).to_rgba8();
        let source = img.to_rgba8();

        // Every copied pixel keeps its source value at the shifted position.
        for y in 0..src_h {
            for x in 0..src_w {
                let dest_x = region.pad_left + x;
                let dest_y = region.pad_top + y;
                if dest_x < out.width() && dest_y < out.height() {
                    assert_eq!(
                        out.get_pixel(dest_x, dest_y),
                        source.get_pixel(src_x + x, src_y + y),
                        "pixel ({x}, {y}) of the crop landed wrong"
                    );
                }
            }
        }

        // And the padded corner is still fill, not a stray source pixel.
        if region.pad_left > 0 && region.pad_top > 0 {
            assert_eq!(*out.get_pixel(0, 0), Rgba([9, 9, 9, 255]));
        }
    }

    #[test]
    fn regions_entirely_outside_the_image_are_all_fill() {
        // `in_bounds_rect` returns None, so nothing is copied and the canvas
        // stays uniformly the fill colour.
        let img = coded_source(8, 8);
        let detection = detection_at(BoundingBox {
            x: 500.0,
            y: 500.0,
            width: 10.0,
            height: 10.0,
        });
        let settings = CropSettings {
            output_width: 0,
            output_height: 0,
            face_height_pct: 100.0,
            positioning_mode: crate::cropper::PositioningMode::Center,
            horizontal_offset: 0.0,
            vertical_offset: 0.0,
            fill_color: FillColor::opaque(70, 80, 90),
            eye_line_align: false,
        };

        let out = crop_face_from_image(&img, &detection, &settings).to_rgba8();
        assert!(out.width() > 0 && out.height() > 0);
        for px in out.pixels() {
            assert_eq!(*px, Rgba([70, 80, 90, 255]));
        }
    }

    #[test]
    fn eye_line_alignment_rotates_only_when_landmarks_are_present() {
        // Both existing tests set `eye_line_align: false`, so the whole
        // rotation branch — including the all-zero landmark guard — never ran.
        let img = coded_source(32, 32);
        let bbox = BoundingBox {
            x: 8.0,
            y: 8.0,
            width: 16.0,
            height: 16.0,
        };
        let base = CropSettings {
            output_width: 24,
            output_height: 24,
            face_height_pct: 80.0,
            positioning_mode: crate::cropper::PositioningMode::Center,
            horizontal_offset: 0.0,
            vertical_offset: 0.0,
            fill_color: FillColor::opaque(0, 0, 0),
            eye_line_align: true,
        };

        // All-zero landmarks mean "no landmarks", so no rotation happens and
        // the result matches the unaligned path exactly.
        let zeroed = detection_at(bbox);
        let unaligned = CropSettings {
            eye_line_align: false,
            ..base.clone()
        };
        assert_eq!(
            crop_face_from_image(&img, &zeroed, &base).to_rgba8(),
            crop_face_from_image(&img, &zeroed, &unaligned).to_rgba8(),
            "zeroed landmarks must skip the rotation"
        );

        // Level eyes give an angle of zero, so alignment is still a no-op.
        let mut level = detection_at(bbox);
        level.landmarks[0] = crate::postprocess::Landmark { x: 12.0, y: 14.0 };
        level.landmarks[1] = crate::postprocess::Landmark { x: 20.0, y: 14.0 };
        assert_eq!(
            crop_face_from_image(&img, &level, &base).to_rgba8(),
            crop_face_from_image(&img, &level, &unaligned).to_rgba8(),
            "a horizontal eye line needs no rotation"
        );

        // Tilted eyes must actually change the output.
        let mut tilted = detection_at(bbox);
        tilted.landmarks[0] = crate::postprocess::Landmark { x: 12.0, y: 10.0 };
        tilted.landmarks[1] = crate::postprocess::Landmark { x: 20.0, y: 18.0 };
        assert_ne!(
            crop_face_from_image(&img, &tilted, &base).to_rgba8(),
            crop_face_from_image(&img, &tilted, &unaligned).to_rgba8(),
            "a tilted eye line must rotate the crop"
        );
    }

    #[test]
    fn eye_line_rotation_matches_an_explicit_rotation() {
        // Asserting only that a tilt "changes something" leaves the angle
        // itself unpinned: a swapped subtraction or a dropped negation still
        // produces a different-but-wrong image. This rebuilds the expected
        // result from the same canvas, rotating by the angle the eye line
        // implies, so the arithmetic has to be exactly right.
        let img = coded_source(32, 32);
        let bbox = BoundingBox {
            x: 8.0,
            y: 8.0,
            width: 16.0,
            height: 16.0,
        };
        let (right_eye, left_eye) = ((12.0f32, 10.0f32), (20.0f32, 18.0f32));

        let mut detection = detection_at(bbox);
        detection.landmarks[0] = crate::postprocess::Landmark {
            x: right_eye.0,
            y: right_eye.1,
        };
        detection.landmarks[1] = crate::postprocess::Landmark {
            x: left_eye.0,
            y: left_eye.1,
        };

        let fill = FillColor::opaque(3, 5, 7);
        let aligned = CropSettings {
            output_width: 24,
            output_height: 24,
            face_height_pct: 80.0,
            positioning_mode: crate::cropper::PositioningMode::Center,
            horizontal_offset: 0.0,
            vertical_offset: 0.0,
            fill_color: fill,
            eye_line_align: true,
        };
        let got = crop_face_from_image(&img, &detection, &aligned).to_rgba8();

        // The unrotated canvas: same crop, no alignment, no resize.
        let raw = CropSettings {
            output_width: 0,
            output_height: 0,
            eye_line_align: false,
            ..aligned.clone()
        };
        let canvas = crop_face_from_image(&img, &detection, &raw).to_rgba8();

        // dx and dy are left-minus-right, and the canvas is turned by the
        // negation of that angle to level the eyes.
        let dx = left_eye.0 - right_eye.0;
        let dy = left_eye.1 - right_eye.1;
        let angle = dy.atan2(dx);
        let rotated = rotate_about_center(
            &canvas,
            -angle,
            Interpolation::Bilinear,
            Border::Constant(Rgba([fill.red, fill.green, fill.blue, fill.alpha])),
        );
        let want = image::imageops::resize(
            &DynamicImage::ImageRgba8(rotated),
            24,
            24,
            FilterType::Lanczos3,
        );

        assert_eq!(got, want);
    }

    #[test]
    fn eye_line_needs_all_four_landmark_values_zero_to_skip() {
        // The guard is "no landmarks at all", so a single non-zero coordinate
        // means the eyes are real and the rotation must happen. Treating it as
        // "any coordinate is zero" would skip alignment for an eye that
        // genuinely sits on x = 0 or y = 0.
        let img = coded_source(32, 32);
        let bbox = BoundingBox {
            x: 8.0,
            y: 8.0,
            width: 16.0,
            height: 16.0,
        };
        let settings = CropSettings {
            output_width: 24,
            output_height: 24,
            face_height_pct: 80.0,
            positioning_mode: crate::cropper::PositioningMode::Center,
            horizontal_offset: 0.0,
            vertical_offset: 0.0,
            fill_color: FillColor::opaque(0, 0, 0),
            eye_line_align: true,
        };
        let unaligned = CropSettings {
            eye_line_align: false,
            ..settings.clone()
        };

        // Right eye at the origin, left eye offset diagonally: not "no
        // landmarks", so this must rotate.
        let mut partial = detection_at(bbox);
        partial.landmarks[0] = crate::postprocess::Landmark { x: 0.0, y: 0.0 };
        partial.landmarks[1] = crate::postprocess::Landmark { x: 5.0, y: 5.0 };
        assert_ne!(
            crop_face_from_image(&img, &partial, &settings).to_rgba8(),
            crop_face_from_image(&img, &partial, &unaligned).to_rgba8(),
            "one populated landmark is enough to align"
        );

        // A single zero component elsewhere behaves the same way.
        let mut one_axis = detection_at(bbox);
        one_axis.landmarks[0] = crate::postprocess::Landmark { x: 12.0, y: 0.0 };
        one_axis.landmarks[1] = crate::postprocess::Landmark { x: 20.0, y: 8.0 };
        assert_ne!(
            crop_face_from_image(&img, &one_axis, &settings).to_rgba8(),
            crop_face_from_image(&img, &one_axis, &unaligned).to_rgba8(),
        );
    }

    #[test]
    fn eye_line_rotation_direction_depends_on_the_tilt() {
        // Mirrored tilts must rotate opposite ways. A dropped sign on the
        // angle, or swapping which landmark is subtracted, makes these two
        // identical.
        let img = coded_source(32, 32);
        let bbox = BoundingBox {
            x: 8.0,
            y: 8.0,
            width: 16.0,
            height: 16.0,
        };
        let settings = CropSettings {
            output_width: 24,
            output_height: 24,
            face_height_pct: 80.0,
            positioning_mode: crate::cropper::PositioningMode::Center,
            horizontal_offset: 0.0,
            vertical_offset: 0.0,
            fill_color: FillColor::opaque(0, 0, 0),
            eye_line_align: true,
        };

        let mut down = detection_at(bbox);
        down.landmarks[0] = crate::postprocess::Landmark { x: 12.0, y: 10.0 };
        down.landmarks[1] = crate::postprocess::Landmark { x: 20.0, y: 18.0 };

        let mut up = detection_at(bbox);
        up.landmarks[0] = crate::postprocess::Landmark { x: 12.0, y: 18.0 };
        up.landmarks[1] = crate::postprocess::Landmark { x: 20.0, y: 10.0 };

        assert_ne!(
            crop_face_from_image(&img, &down, &settings).to_rgba8(),
            crop_face_from_image(&img, &up, &settings).to_rgba8(),
            "opposite tilts must not produce the same rotation"
        );
    }

    #[test]
    fn pads_with_fill_color_when_region_extends() {
        let img = RgbaImage::from_pixel(32, 32, Rgba([40, 50, 60, 255]));
        let img_dyn = DynamicImage::ImageRgba8(img);
        let detection = Detection {
            bbox: BoundingBox {
                x: -5.0,
                y: -5.0,
                width: 20.0,
                height: 20.0,
            },
            landmarks: [
                crate::postprocess::Landmark { x: 0.0, y: 0.0 },
                crate::postprocess::Landmark { x: 0.0, y: 0.0 },
                crate::postprocess::Landmark { x: 0.0, y: 0.0 },
                crate::postprocess::Landmark { x: 0.0, y: 0.0 },
                crate::postprocess::Landmark { x: 0.0, y: 0.0 },
            ],
            score: 0.8,
        };
        let settings = CropSettings {
            output_width: 16,
            output_height: 16,
            face_height_pct: 80.0,
            positioning_mode: crate::cropper::PositioningMode::Center,
            horizontal_offset: 0.0,
            vertical_offset: 0.0,
            fill_color: FillColor::opaque(200, 10, 50),
            eye_line_align: false,
        };

        let out = crop_face_from_image(&img_dyn, &detection, &settings).to_rgba8();
        assert_eq!(out.width(), 16);
        assert_eq!(out.height(), 16);
        let top_left = out.get_pixel(0, 0);
        assert_eq!(top_left[0], 200);
        assert_eq!(top_left[1], 10);
        assert_eq!(top_left[2], 50);
    }
}
