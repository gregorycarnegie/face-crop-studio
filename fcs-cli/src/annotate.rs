//! Image annotation functionality for drawing detections.

use std::{fs, path::Path};

use anyhow::{Context, Result};
use fcs_core::{BoundingBox, Detection};
use image::{DynamicImage, Rgba};
use imageproc::{
    drawing::{draw_filled_circle_mut, draw_hollow_rect_mut},
    rect::Rect,
};

/// Draw detections on an already-decoded image and save it to a directory.
/// `image_path` is used only to derive the output filename and for error messages.
pub fn annotate_image(
    image: &DynamicImage,
    image_path: &Path,
    detections: &[Detection],
    output_dir: &Path,
) -> Result<std::path::PathBuf> {
    let mut image = image.to_rgba8();
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
        // `None` means "not predicted": SCRFD's landmark head was only ever trained on eyes,
        // so nose and mouth come back absent. `flatten` is the whole of what used to be a
        // hand-rolled check for an all-zero sentinel.
        for lm in detection.landmarks.iter().flatten() {
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

    // The output keeps the source's extension, so the format follows from it. A RAW or HEIC
    // source names a format nothing here can encode, which is an error worth saying out loud
    // rather than a silent miss.
    let format = image::ImageFormat::from_path(&output_path).with_context(|| {
        format!(
            "cannot write an annotated image to {}: no encoder for that extension",
            output_path.display()
        )
    })?;

    // Drawing needs RGBA, but JPEG has no alpha channel and refuses to encode it -- which is
    // why `--annotate` produced nothing but a 0-byte file for every `.jpg` input, the most
    // common one there is. Dropping alpha here is lossless: the annotation is drawn opaque.
    let annotated = DynamicImage::ImageRgba8(image);
    let annotated = if format == image::ImageFormat::Jpeg {
        DynamicImage::ImageRgb8(annotated.to_rgb8())
    } else {
        annotated
    };

    // Encoded to memory and replaced atomically rather than through `image::save`, which
    // truncates the destination before encoding: the failure above used to leave that 0-byte
    // file in place of whatever annotation was there before.
    let mut encoded = std::io::Cursor::new(Vec::new());
    annotated
        .write_to(&mut encoded, format)
        .with_context(|| format!("failed to encode annotated image {}", output_path.display()))?;
    fcs_utils::write_atomically(&output_path, &encoded.into_inner())
        .with_context(|| format!("failed to save annotated image {}", output_path.display()))?;

    Ok(output_path)
}

/// Convert a floating-point `BoundingBox` to an integer `imageproc::rect::Rect`.
fn rect_from_bbox(bbox: &BoundingBox, img_w: u32, img_h: u32) -> Rect {
    let max_x = if img_w == 0 { 0.0 } else { (img_w - 1) as f32 };
    let max_y = if img_h == 0 { 0.0 } else { (img_h - 1) as f32 };

    let x1 = bbox.x.clamp(0.0, max_x);
    let y1 = bbox.y.clamp(0.0, max_y);
    let x2 = bbox.right().clamp(0.0, max_x);
    let y2 = bbox.bottom().clamp(0.0, max_y);

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
                Some(Landmark {
                    x: x + 10.0,
                    y: y + 10.0,
                }),
                Some(Landmark {
                    x: x + 20.0,
                    y: y + 10.0,
                }),
                Some(Landmark {
                    x: x + 15.0,
                    y: y + 20.0,
                }),
                Some(Landmark {
                    x: x + 8.0,
                    y: y + 30.0,
                }),
                Some(Landmark {
                    x: x + 22.0,
                    y: y + 30.0,
                }),
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

        let img = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            100,
            100,
            image::Rgba([128, 128, 128, 255]),
        ));

        let det = make_detection(20.0, 20.0, 40.0, 40.0);
        let result = annotate_image(&img, &img_path, &[det], &out_dir);
        assert!(result.is_ok(), "annotate_image failed: {:?}", result.err());
        assert!(out_dir.join("input.png").exists());
    }

    #[test]
    fn annotate_image_no_detections_copies_image_unchanged() {
        let dir = tempdir().expect("tempdir");
        let img_path = dir.path().join("empty.png");
        let out_dir = dir.path().join("out");
        std::fs::create_dir_all(&out_dir).expect("create output dir");

        let img =
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(50, 50, image::Rgba([255, 0, 0, 255])));

        let result = annotate_image(&img, &img_path, &[], &out_dir);
        assert!(result.is_ok());
        assert!(out_dir.join("empty.png").exists());
    }

    #[test]
    fn annotate_rejects_an_image_with_either_dimension_zero() {
        // Both dimensions have to be non-zero, not just one: drawing into a
        // zero-width buffer has nowhere to put the boxes.
        let dir = tempdir().expect("temp directory");
        for (w, h) in [(0u32, 4u32), (4, 0), (0, 0)] {
            let image = DynamicImage::ImageRgba8(RgbaImage::new(w, h));
            // Which error matters: past the guard, saving a zero-sized PNG fails too.
            let err = annotate_image(&image, Path::new("input.png"), &[], dir.path())
                .expect_err(&format!("{w}x{h} should be refused"));
            assert!(
                format!("{err}").contains("zero dimensions"),
                "{w}x{h}: {err}"
            );
        }
    }

    /// The bug the existing tests missed by only ever using `.png`.
    ///
    /// Drawing happens in RGBA and JPEG has no alpha channel, so encoding refused outright and
    /// `--annotate` wrote a 0-byte file for every `.jpg` input -- the commonest source there is.
    /// It has to come back out as a decodable JPEG of the same size as the input.
    /// Only JPEG, which has no alpha channel, is flattened; a PNG keeps its transparency. The
    /// JPEG test cannot see `!=` in place of `==`, because `write_to` flattens JPEG by itself --
    /// the mutant's real effect lands on every other format.
    #[test]
    fn annotating_a_png_keeps_its_alpha_channel() {
        let dir = tempdir().expect("tempdir");
        let img_path = dir.path().join("cutout.png");
        let out_dir = dir.path().join("annotated");
        std::fs::create_dir_all(&out_dir).expect("create output dir");

        let mut rgba = RgbaImage::from_pixel(80, 60, image::Rgba([90, 120, 150, 255]));
        // Transparent, and well away from the box and landmarks drawn below.
        rgba.put_pixel(79, 59, image::Rgba([0, 0, 0, 0]));
        let det = make_detection(10.0, 10.0, 30.0, 30.0);

        let written = annotate_image(&DynamicImage::ImageRgba8(rgba), &img_path, &[det], &out_dir)
            .expect("annotate a png");
        let decoded = image::open(&written).expect("the output must be a decodable image");
        assert_eq!(
            decoded.color(),
            image::ColorType::Rgba8,
            "the alpha channel must survive"
        );
        assert_eq!(decoded.to_rgba8().get_pixel(79, 59)[3], 0);
    }

    #[test]
    fn annotating_a_jpeg_source_writes_a_real_jpeg() {
        let dir = tempdir().expect("tempdir");
        let img_path = dir.path().join("photo.jpg");
        let out_dir = dir.path().join("annotated");
        std::fs::create_dir_all(&out_dir).expect("create output dir");

        let img = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            80,
            60,
            image::Rgba([90, 120, 150, 255]),
        ));
        let det = make_detection(10.0, 10.0, 30.0, 30.0);

        let written = annotate_image(&img, &img_path, &[det], &out_dir).expect("annotate a jpeg");
        assert_eq!(written, out_dir.join("photo.jpg"));

        let bytes = std::fs::metadata(&written).expect("stat").len();
        assert!(bytes > 0, "a 0-byte file is the bug this guards");
        let decoded = image::open(&written).expect("the output must be a decodable image");
        assert_eq!((decoded.width(), decoded.height()), (80, 60));
    }

    /// A source extension with no encoder must fail loudly and leave nothing behind, rather
    /// than the truncated file `image::save` used to produce.
    #[test]
    fn a_source_extension_with_no_encoder_writes_nothing() {
        let dir = tempdir().expect("tempdir");
        let img_path = dir.path().join("capture.nef");
        let out_dir = dir.path().join("annotated");
        std::fs::create_dir_all(&out_dir).expect("create output dir");

        let img =
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(20, 20, image::Rgba([1, 2, 3, 255])));
        let err = annotate_image(&img, &img_path, &[], &out_dir).expect_err("no NEF encoder");
        assert!(
            format!("{err:#}").contains("no encoder"),
            "the error should say why: {err:#}"
        );
        assert!(
            std::fs::read_dir(&out_dir)
                .expect("read_dir")
                .next()
                .is_none(),
            "nothing may be left in the output directory"
        );
    }
}
