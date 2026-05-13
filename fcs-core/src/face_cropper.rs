//! Face extraction and resizing utilities.
//!
//! Provides a simple `crop_face_from_image` helper that ties a `Detection` to
//! a `CropSettings` and returns an owned `DynamicImage` sized to the requested output.

use crate::cropper::{CropRegion, CropSettings, calculate_crop_region};
use crate::postprocess::Detection;

use image::{DynamicImage, GenericImageView, Rgba, RgbaImage, imageops::FilterType};
use imageproc::geometric_transformations::{Interpolation, rotate_about_center};

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
            let rotated = rotate_about_center(&canvas, -angle, Interpolation::Bilinear, fill);
            rotated
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
    use crate::cropper::{CropSettings, FillColor};
    use crate::postprocess::BoundingBox;
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
