use std::fs;

use image::{DynamicImage, RgbaImage};
use tempfile::tempdir;

use fcs_core::{
    BoundingBox, CropSettings, Detection, FillColor, Landmark, PositioningMode,
    crop_face_from_image,
};
use fcs_utils::{EnhancementSettings, apply_enhancements};

#[test]
fn cli_like_crop_and_enhance_saves_file() {
    // Create a temp directory
    let dir = tempdir().expect("tempdir");
    let base = dir.path();

    // Create synthetic input image
    let img_path = base.join("input.png");
    let img = RgbaImage::from_pixel(300, 300, image::Rgba([150, 120, 200, 255]));
    let dyn_img = DynamicImage::ImageRgba8(img.clone());
    dyn_img.save(&img_path).expect("save input");

    // Build a fake detection centered in the image
    let det = Detection {
        bbox: BoundingBox {
            x: 75.0,
            y: 75.0,
            width: 150.0,
            height: 150.0,
        },
        landmarks: [
            Landmark { x: 100.0, y: 100.0 },
            Landmark { x: 200.0, y: 100.0 },
            Landmark { x: 150.0, y: 140.0 },
            Landmark { x: 115.0, y: 200.0 },
            Landmark { x: 185.0, y: 200.0 },
        ],
        score: 0.98,
    };

    // Crop using default settings
    let settings = CropSettings {
        output_width: 256,
        output_height: 256,
        face_height_pct: 60.0,
        positioning_mode: PositioningMode::Center,
        horizontal_offset: 0.0,
        vertical_offset: 0.0,
        fill_color: FillColor::default(),
        eye_line_align: false,
    };

    let cropped = crop_face_from_image(&dyn_img, &det, &settings);

    // Build a small enhancement and apply
    let enh = EnhancementSettings {
        auto_color: false,
        exposure_stops: 0.0,
        brightness: 0,
        contrast: 1.1,
        saturation: 1.0,
        unsharp_amount: 0.5,
        unsharp_radius: 1.0,
        sharpness: 0.2,
        skin_smooth_amount: 0.0,
        skin_smooth_sigma_space: 3.0,
        skin_smooth_sigma_color: 25.0,
        red_eye_removal: false,
        red_eye_threshold: 1.5,
        background_blur: false,
        background_blur_radius: 15.0,
        background_blur_mask_size: 0.6,
    };
    let final_crop = apply_enhancements(&cropped, &enh, None);

    // Save final crop to disk and assert file exists and is non-empty
    let out = base.join("out.png");
    final_crop.save(&out).expect("save crop");
    let md = fs::metadata(&out).expect("metadata");
    assert!(md.len() > 0, "saved file should be non-empty");

    // Cleanup handled by tempdir
}
