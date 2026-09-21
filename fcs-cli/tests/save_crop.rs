use fcs_core::{
    Detection, Landmark, crop_face_from_image, cropper::CropSettings, postprocess::BoundingBox,
};
use image::{DynamicImage, Rgba, RgbaImage};
use tempfile::tempdir;

#[test]
fn saves_crops_in_various_formats() {
    // Create a temp dir and write a synthetic image file
    let td = tempdir().unwrap();
    let img_path = td.path().join("test_image.png");
    let img = RgbaImage::from_pixel(400, 400, Rgba([120u8, 120u8, 200u8, 255u8]));
    img.save(&img_path).unwrap();

    // Make a synthetic detection
    let det = Detection {
        bbox: BoundingBox {
            x: 150.0,
            y: 150.0,
            width: 100.0,
            height: 100.0,
        },
        landmarks: [
            Some(Landmark { x: 160.0, y: 160.0 }),
            Some(Landmark { x: 190.0, y: 160.0 }),
            Some(Landmark { x: 175.0, y: 180.0 }),
            Some(Landmark { x: 165.0, y: 200.0 }),
            Some(Landmark { x: 185.0, y: 200.0 }),
        ],
        score: 0.9,
    };

    let dyn_img = DynamicImage::ImageRgba8(img);

    let settings = CropSettings {
        output_width: 200,
        output_height: 200,
        face_height_pct: 60.0,
        ..Default::default()
    };

    // Crop and save as PNG
    let crop = crop_face_from_image(&dyn_img, &det, &settings);
    let out_png = td.path().join("out.png");
    crop.save(&out_png).unwrap();
    assert!(out_png.exists());
    assert!(out_png.metadata().unwrap().len() > 0);

    // Save as JPEG with quality
    let out_jpeg = td.path().join("out.jpg");
    crop.save_with_format(&out_jpeg, image::ImageFormat::Jpeg)
        .unwrap();
    assert!(out_jpeg.exists());
    assert!(out_jpeg.metadata().unwrap().len() > 0);

    // Save as WebP (if supported)
    let out_webp = td.path().join("out.webp");
    let _ = crop.save_with_format(&out_webp, image::ImageFormat::WebP);
    // If WebP is supported by the image crate build, the file should exist and be non-empty.
    if out_webp.exists() {
        assert!(out_webp.metadata().unwrap().len() > 0);
    }
}
