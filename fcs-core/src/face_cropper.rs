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

/// Copy the in-bounds source region `(x, y, w, h)` onto the padded canvas at `offset`.
///
/// This used to be `imageops::crop_imm(img, ..).to_image()` followed by a per-pixel loop
/// copying that into the canvas, and it was the third largest cost in a folder job --
/// `DynamicImage::get_pixel` at 4.8% of all CPU (experiment 98). Two reasons, both from
/// `to_image`: it allocates and fills a whole second copy of the region, and it reads the
/// source through `DynamicImage`'s `get_pixel`, which matches on the enum variant for
/// every single pixel.
///
/// The two variants the decoders actually produce are handled a row at a time instead --
/// `load_image` returns `ImageRgb8` for every JPEG through libjpeg-turbo, and `ImageRgba8`
/// for anything with alpha. Everything else (Luma, 16-bit, f32) keeps the generic
/// per-pixel path, which is correct for all of them and is what the fast paths are
/// checked against.
fn blit_region(
    img: &DynamicImage,
    (src_x, src_y, src_w, src_h): (u32, u32, u32, u32),
    canvas: &mut RgbaImage,
    (offset_x, offset_y): (u32, u32),
) {
    let (canvas_w, canvas_h) = canvas.dimensions();
    // The old loop skipped any destination pixel outside the canvas; clipping the extent
    // up front is the same thing without testing it per pixel.
    let w = src_w.min(canvas_w.saturating_sub(offset_x));
    let h = src_h.min(canvas_h.saturating_sub(offset_y));
    if w == 0 || h == 0 {
        return;
    }
    let dst_stride = canvas_w as usize * 4;

    match img {
        DynamicImage::ImageRgba8(src) => {
            let src_stride = src.width() as usize * 4;
            let src_raw = src.as_raw();
            let dst_raw = canvas.as_mut();
            let len = w as usize * 4;
            for row in 0..h as usize {
                let s = (src_y as usize + row) * src_stride + src_x as usize * 4;
                let d = (offset_y as usize + row) * dst_stride + offset_x as usize * 4;
                dst_raw[d..d + len].copy_from_slice(&src_raw[s..s + len]);
            }
        }
        DynamicImage::ImageRgb8(src) => {
            let src_stride = src.width() as usize * 3;
            let src_raw = src.as_raw();
            let dst_raw = canvas.as_mut();
            for row in 0..h as usize {
                let s = (src_y as usize + row) * src_stride + src_x as usize * 3;
                let d = (offset_y as usize + row) * dst_stride + offset_x as usize * 4;
                let src_row = &src_raw[s..s + w as usize * 3];
                let dst_row = &mut dst_raw[d..d + w as usize * 4];
                let (dst_px, _) = dst_row.as_chunks_mut::<4>();
                let (src_px, _) = src_row.as_chunks::<3>();
                for (out, rgb) in dst_px.iter_mut().zip(src_px) {
                    // `get_pixel` on an `ImageRgb8` returns an opaque Rgba, so 255 is the
                    // alpha the per-pixel path produced.
                    *out = [rgb[0], rgb[1], rgb[2], 255];
                }
            }
        }
        other => {
            for row in 0..h {
                for col in 0..w {
                    let pixel = other.get_pixel(src_x + col, src_y + row);
                    canvas.put_pixel(offset_x + col, offset_y + row, pixel);
                }
            }
        }
    }
}

/// Map the two eye landmarks into output-crop coordinates, for targeted red-eye removal.
///
/// Empty when either eye is absent: one point is not a pair, and there is nothing to aim at.
///
/// This lived in the GUI until 2.0, which is why the CLI passed `None` for the eye positions
/// and its red-eye removal had nothing to aim at -- the mapping simply was not reachable from
/// there. It belongs next to `calculate_crop_region`, whose geometry it inverts.
pub fn eye_positions(
    detection: &Detection,
    img_w: u32,
    img_h: u32,
    settings: &CropSettings,
) -> Vec<fcs_utils::RedEye> {
    let (Some(right), Some(left)) = (detection.landmarks[0], detection.landmarks[1]) else {
        return Vec::new();
    };
    let region = calculate_crop_region(img_w, img_h, detection.bbox, settings);
    let sx = settings.output_width as f32 / region.width.max(1) as f32;
    let sy = settings.output_height as f32 / region.height.max(1) as f32;
    let face_h_out =
        detection.bbox.height / region.height.max(1) as f32 * settings.output_height as f32;
    let radius = (face_h_out * 0.12).max(4.0);
    [right, left]
        .iter()
        .map(|lm| fcs_utils::RedEye {
            x: (lm.x - region.x as f32) * sx,
            y: (lm.y - region.y as f32) * sy,
            radius,
            _pad: 0.0,
        })
        .collect()
}

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

    // `in_bounds_rect` returns `None` rather than a zero-sized rect, so there is
    // nothing left to filter out here.
    if let Some((src_x, src_y, src_w, src_h)) = region.in_bounds_rect(img_w, img_h) {
        let offset_x = region.pad_left.min(canvas_width.saturating_sub(1));
        let offset_y = region.pad_top.min(canvas_height.saturating_sub(1));
        blit_region(
            img,
            (src_x, src_y, src_w, src_h),
            &mut canvas,
            (offset_x, offset_y),
        );
    }

    // If output dimensions are zero, return the raw (possibly padded) crop as DynamicImage.
    if settings.output_width == 0 || settings.output_height == 0 {
        return DynamicImage::ImageRgba8(canvas);
    }

    let canvas = if settings.eye_line_align {
        // Index 0 lies on the viewer's left of the face and index 1 on the viewer's right --
        // measured across 10,880 large detections, that holds for 99.8% of them. It is the
        // usual RetinaFace/YuNet order, where index 0 is the subject's *own* right eye. These
        // names are screen-relative on purpose: read anatomically, "left eye" is the mirror of
        // this, and that reading has already produced one mirrored landmark mapping elsewhere.
        // Both eyes or nothing. Aligning from one point is not possible, and the sentinel this
        // replaced made a missing eye look like a real one at the origin -- which rotated the
        // crop 45 degrees when the other eye happened to sit diagonally from it.
        if let (Some(left_eye), Some(right_eye)) = (detection.landmarks[0], detection.landmarks[1])
        {
            // Angle of the eye line relative to horizontal in source image coords, taken from
            // the viewer's left eye towards the viewer's right. Positive angle = the eye on
            // the right sits lower on screen; rotate by -angle to level them.
            let dx = right_eye.x - left_eye.x;
            let dy = right_eye.y - left_eye.y;
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

    // Through `fast_image_resize` rather than `image::imageops::resize`: the latter samples
    // pixel-by-pixel through `GenericImageView`, and a batch profile put this one call at
    // 10.9% of all CPU over a 1239-image folder. Same Lanczos3 kernel either side, so output
    // is equivalent up to rounding (experiment 88). `None` means fir declined the request --
    // a zero dimension, say -- and the original path still has to answer for it.
    let resized = fcs_utils::resize_rgba_fast(
        &canvas,
        settings.output_width,
        settings.output_height,
        FilterType::Lanczos3,
    )
    .unwrap_or_else(|| {
        image::imageops::resize(
            &DynamicImage::ImageRgba8(canvas),
            settings.output_width,
            settings.output_height,
            FilterType::Lanczos3,
        )
    });

    DynamicImage::ImageRgba8(resized)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Clipped to zero width but not zero height: without the early return the row loop would
    /// slice far past the end of the canvas.
    #[test]
    fn a_region_placed_past_the_canvas_writes_nothing() {
        let src = DynamicImage::ImageRgba8(RgbaImage::from_pixel(3, 3, Rgba([1, 2, 3, 4])));
        let fill = Rgba([9, 8, 7, 200]);
        let mut canvas = RgbaImage::from_pixel(4, 4, fill);
        blit_region(&src, (0, 0, 3, 3), &mut canvas, (40, 1));
        assert!(canvas.pixels().all(|p| *p == fill));
    }

    /// The RGB and RGBA fast paths must produce exactly what the generic `get_pixel` loop
    /// produced, including at the clipped edges and with a non-zero source offset, because
    /// the crops they feed are compared byte for byte against previous releases
    /// (experiment 98).
    #[test]
    fn the_blit_fast_paths_agree_with_the_generic_one() {
        /// The pre-experiment-98 loop, kept here as the reference.
        fn reference(
            img: &DynamicImage,
            (src_x, src_y, src_w, src_h): (u32, u32, u32, u32),
            canvas: &mut RgbaImage,
            (offset_x, offset_y): (u32, u32),
        ) {
            let (canvas_width, canvas_height) = canvas.dimensions();
            let sub = image::imageops::crop_imm(img, src_x, src_y, src_w, src_h).to_image();
            for y in 0..sub.height() {
                for x in 0..sub.width() {
                    let dest_x = offset_x + x;
                    let dest_y = offset_y + y;
                    if dest_x < canvas_width && dest_y < canvas_height {
                        canvas.put_pixel(dest_x, dest_y, *sub.get_pixel(x, y));
                    }
                }
            }
        }

        // Every pixel distinct, so a wrong stride or a swapped axis cannot survive.
        let mut rgb = image::RgbImage::new(23, 17);
        for (x, y, px) in rgb.enumerate_pixels_mut() {
            *px = image::Rgb([
                (x * 7 % 251) as u8,
                (y * 13 % 241) as u8,
                ((x + y) % 239) as u8,
            ]);
        }
        let mut rgba = RgbaImage::new(23, 17);
        for (x, y, px) in rgba.enumerate_pixels_mut() {
            *px = Rgba([
                (x * 5 % 251) as u8,
                (y * 11 % 241) as u8,
                ((x * y) % 239) as u8,
                ((x + y * 3) % 253) as u8,
            ]);
        }
        let luma = DynamicImage::ImageLuma8(image::GrayImage::from_fn(23, 17, |x, y| {
            image::Luma([((x * 3 + y * 29) % 251) as u8])
        }));

        let sources = [
            DynamicImage::ImageRgb8(rgb),
            DynamicImage::ImageRgba8(rgba),
            luma,
        ];
        // Exact fit, clipped on both axes, offset into the canvas, and a region that
        // starts away from the origin.
        let cases = [
            ((0, 0, 23, 17), (20, 20), (0, 0)),
            ((0, 0, 23, 17), (10, 8), (0, 0)),
            ((0, 0, 23, 17), (30, 30), (7, 5)),
            ((4, 3, 11, 9), (30, 30), (2, 6)),
            ((4, 3, 11, 9), (8, 8), (5, 5)),
        ];

        for src in &sources {
            for (rect, (cw, ch), offset) in cases {
                let fill = Rgba([9, 8, 7, 200]);
                let mut want = RgbaImage::from_pixel(cw, ch, fill);
                let mut got = RgbaImage::from_pixel(cw, ch, fill);
                reference(src, rect, &mut want, offset);
                blit_region(src, rect, &mut got, offset);
                assert_eq!(
                    got.as_raw(),
                    want.as_raw(),
                    "blit disagreed for {src:?} rect {rect:?} canvas {cw}x{ch} offset {offset:?}"
                );
            }
        }
    }
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
                Some(crate::postprocess::Landmark { x: 360.0, y: 260.0 }),
                Some(crate::postprocess::Landmark { x: 390.0, y: 260.0 }),
                Some(crate::postprocess::Landmark { x: 375.0, y: 285.0 }),
                Some(crate::postprocess::Landmark { x: 365.0, y: 310.0 }),
                Some(crate::postprocess::Landmark { x: 385.0, y: 310.0 }),
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

    /// No landmarks at all, which switches the eye-line branch off.
    fn no_landmarks() -> [Option<crate::postprocess::Landmark>; 5] {
        [None; 5]
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
        level.landmarks[0] = Some(crate::postprocess::Landmark { x: 12.0, y: 14.0 });
        level.landmarks[1] = Some(crate::postprocess::Landmark { x: 20.0, y: 14.0 });
        assert_eq!(
            crop_face_from_image(&img, &level, &base).to_rgba8(),
            crop_face_from_image(&img, &level, &unaligned).to_rgba8(),
            "a horizontal eye line needs no rotation"
        );

        // Tilted eyes must actually change the output.
        let mut tilted = detection_at(bbox);
        tilted.landmarks[0] = Some(crate::postprocess::Landmark { x: 12.0, y: 10.0 });
        tilted.landmarks[1] = Some(crate::postprocess::Landmark { x: 20.0, y: 18.0 });
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
        detection.landmarks[0] = Some(crate::postprocess::Landmark {
            x: right_eye.0,
            y: right_eye.1,
        });
        detection.landmarks[1] = Some(crate::postprocess::Landmark {
            x: left_eye.0,
            y: left_eye.1,
        });

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

        // Compared with a tolerance rather than for equality: the subject here is the
        // rotation angle, and the reference scales with `image::imageops` while the
        // production path scales with `fast_image_resize` (experiment 88). Two Lanczos3
        // implementations round differently. A wrong angle moves pixels by far more than
        // this -- the eyes are 30 degrees off level in this fixture.
        assert_eq!(got.dimensions(), want.dimensions());
        let worst = got
            .as_raw()
            .iter()
            .zip(want.as_raw())
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .expect("non-empty");
        assert!(
            worst <= 4,
            "max channel difference {worst} is larger than resampling rounding"
        );
    }

    #[test]
    fn eye_line_needs_both_eyes_present_to_align() {
        // The guard is "both eyes are `Some`". The name and comment here used to say "all four
        // landmark values zero", which was the pre-2.0 sentinel and is the opposite of what the
        // assertions below now check -- a stale name on a rewritten test is worse than no name.
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

        // One eye present and the other absent must NOT rotate. This is the assertion the
        // `Option` change inverted, and it is worth being explicit about why.
        //
        // Until 2.0 the absent marker was an all-zero point, and this test asserted the
        // opposite: that a landmark at exactly `(0, 0)` beside a real one "is enough to
        // align". It is not. There is no eye line through one point, so what actually
        // happened was that the origin was read as a coordinate and the crop was rotated by
        // the angle from `(0, 0)` to the other eye -- 45 degrees, here, from a value that
        // meant "nothing was predicted".
        let mut one_eye = detection_at(bbox);
        one_eye.landmarks[0] = None;
        one_eye.landmarks[1] = Some(crate::postprocess::Landmark { x: 5.0, y: 5.0 });
        assert_eq!(
            crop_face_from_image(&img, &one_eye, &settings).to_rgba8(),
            crop_face_from_image(&img, &one_eye, &unaligned).to_rgba8(),
            "one eye is not an eye line, so alignment must be skipped"
        );

        // A real landmark that happens to sit on an axis is still a real landmark, and still
        // aligns. Under the sentinel this was indistinguishable from the case above.
        let mut on_axis = detection_at(bbox);
        on_axis.landmarks[0] = Some(crate::postprocess::Landmark { x: 12.0, y: 0.0 });
        on_axis.landmarks[1] = Some(crate::postprocess::Landmark { x: 20.0, y: 8.0 });
        assert_ne!(
            crop_face_from_image(&img, &on_axis, &settings).to_rgba8(),
            crop_face_from_image(&img, &on_axis, &unaligned).to_rgba8(),
            "a zero component is a coordinate, not an absence"
        );

        // And the origin itself: a genuine eye at (0, 0) with its pair is now usable, which
        // the sentinel made impossible to express.
        let mut at_origin = detection_at(bbox);
        at_origin.landmarks[0] = Some(crate::postprocess::Landmark { x: 0.0, y: 0.0 });
        at_origin.landmarks[1] = Some(crate::postprocess::Landmark { x: 8.0, y: 0.0 });
        assert_eq!(
            crop_face_from_image(&img, &at_origin, &settings).to_rgba8(),
            crop_face_from_image(&img, &at_origin, &unaligned).to_rgba8(),
            "a level pair needs no rotation, wherever it sits"
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
        down.landmarks[0] = Some(crate::postprocess::Landmark { x: 12.0, y: 10.0 });
        down.landmarks[1] = Some(crate::postprocess::Landmark { x: 20.0, y: 18.0 });

        let mut up = detection_at(bbox);
        up.landmarks[0] = Some(crate::postprocess::Landmark { x: 12.0, y: 18.0 });
        up.landmarks[1] = Some(crate::postprocess::Landmark { x: 20.0, y: 10.0 });

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
            landmarks: [None; 5],
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

    #[test]
    fn eye_line_aligns_from_any_non_level_pair() {
        // Both eyes are present in every case; what varies is where they sit. The tilts are
        // quarter or half turns, which are unmistakable in the output -- an eye line that is
        // merely horizontal rotates by zero and looks exactly like skipping the alignment,
        // which is why the left eye's x is negative here.
        //
        // The previous name said "any single landmark coordinate is set", describing the
        // all-zero sentinel that 2.0 replaced with `Option`.
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
        let baseline = crop_face_from_image(&img, &detection_at(bbox), &unaligned).to_rgba8();

        for (label, right_eye, left_eye) in [
            ("right eye x only", (7.0, 0.0), (0.0, 0.0)),
            ("right eye y only", (0.0, 7.0), (0.0, 0.0)),
            ("left eye y only", (0.0, 0.0), (0.0, 7.0)),
            ("left eye x only", (0.0, 0.0), (-7.0, 0.0)),
        ] {
            let mut detection = detection_at(bbox);
            detection.landmarks[0] = Some(crate::postprocess::Landmark {
                x: right_eye.0,
                y: right_eye.1,
            });
            detection.landmarks[1] = Some(crate::postprocess::Landmark {
                x: left_eye.0,
                y: left_eye.1,
            });
            assert_ne!(
                crop_face_from_image(&img, &detection, &settings).to_rgba8(),
                baseline,
                "{label}: a populated eye coordinate must still align"
            );
        }
    }

    /// `eye_positions` maps the two eyes out of source pixels and into crop pixels.
    ///
    /// It arrived in 2.0 by being moved out of the GUI, and it arrived with no tests: all 19 of
    /// this file's surviving mutants were its arithmetic. The fixture is chosen so none of that
    /// arithmetic can collapse -- a non-square image and box, an output size that is neither a
    /// multiple nor a divisor of the crop region, and eyes away from the region's origin so
    /// `lm.x - region.x` is a number rather than zero.
    #[test]
    fn eye_positions_map_into_output_crop_coordinates() {
        let (img_w, img_h) = (800u32, 600u32);
        let bbox = BoundingBox {
            x: 150.0,
            y: 120.0,
            width: 200.0,
            height: 160.0,
        };
        let settings = CropSettings {
            output_width: 320,
            output_height: 240,
            face_height_pct: 50.0,
            positioning_mode: crate::cropper::PositioningMode::Center,
            horizontal_offset: 0.0,
            vertical_offset: 0.0,
            fill_color: FillColor::opaque(0, 0, 0),
            eye_line_align: false,
        };
        let mut detection = detection_at(bbox);
        detection.landmarks[0] = Some(crate::postprocess::Landmark { x: 203.0, y: 171.0 });
        detection.landmarks[1] = Some(crate::postprocess::Landmark { x: 297.0, y: 183.0 });

        let eyes = eye_positions(&detection, img_w, img_h, &settings);
        assert_eq!(eyes.len(), 2, "both eyes present means two targets");

        // The expected mapping, written out here rather than taken from the function.
        let region = calculate_crop_region(img_w, img_h, bbox, &settings);
        let sx = settings.output_width as f32 / region.width.max(1) as f32;
        let sy = settings.output_height as f32 / region.height.max(1) as f32;
        let face_h_out = bbox.height / region.height.max(1) as f32 * settings.output_height as f32;
        let expected_radius = (face_h_out * 0.12).max(4.0);

        // A fixture where the radius sits on its floor would hide every term feeding it.
        assert!(
            expected_radius > 4.0,
            "fixture must clear the 4 px floor, got {expected_radius}"
        );

        for (eye, landmark) in eyes
            .iter()
            .zip([detection.landmarks[0], detection.landmarks[1]])
        {
            let landmark = landmark.expect("set above");
            let want_x = (landmark.x - region.x as f32) * sx;
            let want_y = (landmark.y - region.y as f32) * sy;
            assert!((eye.x - want_x).abs() < 1e-3, "x {} vs {want_x}", eye.x);
            assert!((eye.y - want_y).abs() < 1e-3, "y {} vs {want_y}", eye.y);
            assert!(
                (eye.radius - expected_radius).abs() < 1e-3,
                "radius {} vs {expected_radius}",
                eye.radius
            );
            // Non-zero and distinct, or the comparisons above prove less than they look.
            assert!(want_x.abs() > 1.0 && want_y.abs() > 1.0);
            assert!((want_x - want_y).abs() > 1.0);
        }
        // The two eyes must not land on the same point, which a dropped landmark index would do.
        assert!(
            (eyes[0].x - eyes[1].x).abs() > 1.0,
            "both eyes mapped to the same x"
        );
    }

    /// One eye absent means no targets: red-eye removal has nothing to aim at with half a pair.
    #[test]
    fn eye_positions_are_empty_unless_both_eyes_are_present() {
        let bbox = BoundingBox {
            x: 10.0,
            y: 20.0,
            width: 60.0,
            height: 80.0,
        };
        let settings = CropSettings {
            output_width: 128,
            output_height: 96,
            face_height_pct: 60.0,
            positioning_mode: crate::cropper::PositioningMode::Center,
            horizontal_offset: 0.0,
            vertical_offset: 0.0,
            fill_color: FillColor::opaque(0, 0, 0),
            eye_line_align: false,
        };
        let eye = Some(crate::postprocess::Landmark { x: 30.0, y: 45.0 });

        for (left, right) in [(None, None), (eye, None), (None, eye)] {
            let mut detection = detection_at(bbox);
            detection.landmarks[0] = left;
            detection.landmarks[1] = right;
            assert!(
                eye_positions(&detection, 200, 300, &settings).is_empty(),
                "a half pair must produce no targets"
            );
        }

        // And with both, it does produce them -- so the emptiness above is the guard, not a
        // function that always returns nothing.
        let mut both = detection_at(bbox);
        both.landmarks[0] = eye;
        both.landmarks[1] = Some(crate::postprocess::Landmark { x: 52.0, y: 49.0 });
        assert_eq!(eye_positions(&both, 200, 300, &settings).len(), 2);
    }

    /// A tiny face clamps the radius to its 4 px floor rather than vanishing.
    #[test]
    fn a_tiny_face_keeps_a_usable_eye_radius() {
        let bbox = BoundingBox {
            x: 100.0,
            y: 100.0,
            width: 9.0,
            height: 7.0,
        };
        let settings = CropSettings {
            output_width: 64,
            output_height: 48,
            face_height_pct: 3.0,
            positioning_mode: crate::cropper::PositioningMode::Center,
            horizontal_offset: 0.0,
            vertical_offset: 0.0,
            fill_color: FillColor::opaque(0, 0, 0),
            eye_line_align: false,
        };
        let mut detection = detection_at(bbox);
        detection.landmarks[0] = Some(crate::postprocess::Landmark { x: 102.0, y: 103.0 });
        detection.landmarks[1] = Some(crate::postprocess::Landmark { x: 106.0, y: 104.0 });
        let eyes = eye_positions(&detection, 400, 300, &settings);
        assert_eq!(eyes.len(), 2);
        assert!(
            (eyes[0].radius - 4.0).abs() < 1e-3,
            "radius should sit on the floor, got {}",
            eyes[0].radius
        );
    }
}
