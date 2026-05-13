//! Image annotation functionality for drawing detections.

use std::{fs, path::Path};

use anyhow::{Context, Result};
use fcs_core::{BoundingBox, Detection};
use image::Rgba;
use imageproc::{
    drawing::{draw_filled_circle_mut, draw_hollow_rect_mut},
    rect::Rect,
};

/// Draw detections on an image and save it to a directory.
pub fn annotate_image(
    image_path: &Path,
    detections: &[Detection],
    output_dir: &Path,
) -> Result<std::path::PathBuf> {
    let mut image = image::open(image_path)
        .with_context(|| format!("failed to open image {}", image_path.display()))?
        .to_rgba8();
    let (img_w, img_h) = image.dimensions();

    if img_w == 0 || img_h == 0 {
        anyhow::bail!(
            "cannot annotate image with zero dimensions: {}",
            image_path.display()
        );
    }

    let rect_color = Rgba([255, 0, 0, 255]);
    let landmark_color = Rgba([0, 255, 0, 255]);

    for detection in detections {
        let rect = rect_from_bbox(&detection.bbox, img_w, img_h);
        draw_hollow_rect_mut(&mut image, rect, rect_color);
        for lm in &detection.landmarks {
            let cx = clamp_to_i32(lm.x, img_w);
            let cy = clamp_to_i32(lm.y, img_h);
            draw_filled_circle_mut(&mut image, (cx, cy), 2, landmark_color);
        }
    }

    let file_name = image_path
        .file_name()
        .unwrap_or_else(|| std::ffi::OsStr::new("frame.png"));
    let output_path = output_dir.join(file_name);

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    image
        .save(&output_path)
        .with_context(|| format!("failed to save annotated image {}", output_path.display()))?;

    Ok(output_path)
}

/// Convert a floating-point `BoundingBox` to an integer `imageproc::rect::Rect`.
fn rect_from_bbox(bbox: &BoundingBox, img_w: u32, img_h: u32) -> Rect {
    let max_x = if img_w == 0 { 0.0 } else { (img_w - 1) as f32 };
    let max_y = if img_h == 0 { 0.0 } else { (img_h - 1) as f32 };

    let x1 = bbox.x.clamp(0.0, max_x);
    let y1 = bbox.y.clamp(0.0, max_y);
    let x2 = (bbox.x + bbox.width).clamp(0.0, max_x);
    let y2 = (bbox.y + bbox.height).clamp(0.0, max_y);

    let width = (x2 - x1).max(1.0) as u32;
    let height = (y2 - y1).max(1.0) as u32;

    Rect::at(x1 as i32, y1 as i32).of_size(width, height)
}

/// Clamp a floating-point coordinate to a valid integer pixel index.
#[inline]
fn clamp_to_i32(value: f32, max_extent: u32) -> i32 {
    if max_extent == 0 {
        return 0;
    }
    let max = (max_extent - 1) as f32;
    value.clamp(0.0, max) as i32
}

#[cfg(test)]
mod tests {
    use super::*;
    use fcs_core::{BoundingBox, Landmark};
    use image::{DynamicImage, RgbaImage};
    use tempfile::tempdir;

    // --- clamp_to_i32 ---

    #[test]
    fn clamp_to_i32_zero_extent_returns_zero() {
        assert_eq!(clamp_to_i32(999.0, 0), 0);
        assert_eq!(clamp_to_i32(-5.0, 0), 0);
    }

    #[test]
    fn clamp_to_i32_normal_value() {
        assert_eq!(clamp_to_i32(50.0, 100), 50);
    }

    #[test]
    fn clamp_to_i32_negative_clamps_to_zero() {
        assert_eq!(clamp_to_i32(-10.0, 100), 0);
    }

    #[test]
    fn clamp_to_i32_over_max_clamps_to_max_minus_one() {
        assert_eq!(clamp_to_i32(200.0, 100), 99);
    }

    #[test]
    fn clamp_to_i32_exactly_at_max_minus_one() {
        assert_eq!(clamp_to_i32(99.0, 100), 99);
    }

    // --- rect_from_bbox ---

    fn make_bbox(x: f32, y: f32, w: f32, h: f32) -> BoundingBox {
        BoundingBox {
            x,
            y,
            width: w,
            height: h,
        }
    }

    #[test]
    fn rect_from_bbox_normal() {
        let r = rect_from_bbox(&make_bbox(10.0, 20.0, 50.0, 60.0), 200, 200);
        assert_eq!(r.left(), 10);
        assert_eq!(r.top(), 20);
        assert_eq!(r.width(), 50);
        assert_eq!(r.height(), 60);
    }

    #[test]
    fn rect_from_bbox_clamps_to_image_bounds() {
        // bbox extends well past image edges
        let r = rect_from_bbox(&make_bbox(0.0, 0.0, 500.0, 500.0), 100, 80);
        assert_eq!(r.width(), 99); // clamped to (99 - 0).max(1)
        assert_eq!(r.height(), 79);
    }

    #[test]
    fn rect_from_bbox_negative_origin_clamps_to_zero() {
        let r = rect_from_bbox(&make_bbox(-20.0, -10.0, 60.0, 60.0), 200, 200);
        assert_eq!(r.left(), 0);
        assert_eq!(r.top(), 0);
        // x2 = (-20 + 60).clamp(0, 199) = 40; width = (40 - 0).max(1) = 40
        assert_eq!(r.width(), 40);
    }

    #[test]
    fn rect_from_bbox_zero_image_dimensions_uses_zero_max() {
        // With zero dimensions max_x/max_y = 0, so everything clamps to 0 and width/height = 1
        let r = rect_from_bbox(&make_bbox(5.0, 5.0, 10.0, 10.0), 0, 0);
        assert_eq!(r.width(), 1);
        assert_eq!(r.height(), 1);
    }

    // --- annotate_image ---

    fn make_detection(x: f32, y: f32, w: f32, h: f32) -> Detection {
        Detection {
            bbox: BoundingBox {
                x,
                y,
                width: w,
                height: h,
            },
            landmarks: [
                Landmark {
                    x: x + 10.0,
                    y: y + 10.0,
                },
                Landmark {
                    x: x + 20.0,
                    y: y + 10.0,
                },
                Landmark {
                    x: x + 15.0,
                    y: y + 20.0,
                },
                Landmark {
                    x: x + 8.0,
                    y: y + 30.0,
                },
                Landmark {
                    x: x + 22.0,
                    y: y + 30.0,
                },
            ],
            score: 0.95,
        }
    }

    #[test]
    fn annotate_image_creates_output_file() {
        let dir = tempdir().expect("tempdir");
        let img_path = dir.path().join("input.png");
        let out_dir = dir.path().join("annotated");
        std::fs::create_dir_all(&out_dir).expect("create output dir");

        // Save a small RGBA image
        let img = RgbaImage::from_pixel(100, 100, image::Rgba([128, 128, 128, 255]));
        DynamicImage::ImageRgba8(img)
            .save(&img_path)
            .expect("save test image");

        let det = make_detection(20.0, 20.0, 40.0, 40.0);
        let result = annotate_image(&img_path, &[det], &out_dir);
        assert!(result.is_ok(), "annotate_image failed: {:?}", result.err());
        assert!(out_dir.join("input.png").exists());
    }

    #[test]
    fn annotate_image_no_detections_copies_image_unchanged() {
        let dir = tempdir().expect("tempdir");
        let img_path = dir.path().join("empty.png");
        let out_dir = dir.path().join("out");
        std::fs::create_dir_all(&out_dir).expect("create output dir");

        let img = RgbaImage::from_pixel(50, 50, image::Rgba([255, 0, 0, 255]));
        DynamicImage::ImageRgba8(img)
            .save(&img_path)
            .expect("save test image");

        let result = annotate_image(&img_path, &[], &out_dir);
        assert!(result.is_ok());
        assert!(out_dir.join("empty.png").exists());
    }

    #[test]
    fn annotate_image_returns_err_for_missing_file() {
        let dir = tempdir().expect("tempdir");
        let missing = dir.path().join("does_not_exist.png");
        let out_dir = dir.path().join("out");
        std::fs::create_dir_all(&out_dir).expect("create output dir");

        let result = annotate_image(&missing, &[], &out_dir);
        assert!(result.is_err());
    }
}
