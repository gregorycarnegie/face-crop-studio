use criterion::{Criterion, criterion_group, criterion_main};
use fcs_core::{
    BoundingBox, CropSettings, Detection, FillColor, Landmark, PositioningMode,
    crop_face_from_image,
};
use fcs_utils::{EnhancementSettings, apply_enhancements};
use image::{DynamicImage, RgbaImage};
use std::hint::black_box;

fn build_source_image() -> DynamicImage {
    let mut img = RgbaImage::from_pixel(1024, 1024, image::Rgba([64, 64, 64, 255]));
    // Add a mixed-frequency region to give the enhancement pipeline work to do.
    for y in 256..768 {
        for x in 256..768 {
            let val = ((x + y) % 255) as u8;
            img.put_pixel(
                x,
                y,
                image::Rgba([val, 255u8.saturating_sub(val), val / 2 + 60, 255]),
            );
        }
    }
    DynamicImage::ImageRgba8(img)
}

fn build_detection() -> Detection {
    Detection {
        bbox: BoundingBox {
            x: 320.0,
            y: 320.0,
            width: 384.0,
            height: 384.0,
        },
        landmarks: [
            Landmark { x: 360.0, y: 380.0 },
            Landmark { x: 620.0, y: 380.0 },
            Landmark { x: 490.0, y: 520.0 },
            Landmark { x: 420.0, y: 620.0 },
            Landmark { x: 560.0, y: 620.0 },
        ],
        score: 0.95,
    }
}

fn crop_settings() -> CropSettings {
    CropSettings {
        output_width: 512,
        output_height: 512,
        face_height_pct: 75.0,
        positioning_mode: PositioningMode::Center,
        horizontal_offset: 0.0,
        vertical_offset: 0.0,
        fill_color: FillColor::default(),
        eye_line_align: false,
    }
}

fn enhancement_settings() -> EnhancementSettings {
    EnhancementSettings {
        auto_color: true,
        exposure_stops: 0.15,
        brightness: 8,
        contrast: 1.15,
        saturation: 1.2,
        unsharp_amount: 0.6,
        unsharp_radius: 1.2,
        sharpness: 0.2,
        skin_smooth_amount: 0.3,
        skin_smooth_sigma_space: 3.0,
        skin_smooth_sigma_color: 25.0,
        red_eye_removal: false,
        red_eye_threshold: 1.5,
        background_blur: false,
        background_blur_radius: 12.0,
        background_blur_mask_size: 0.6,
    }
}

fn crop_enhance_benchmark(c: &mut Criterion) {
    let image = build_source_image();
    let detection = build_detection();
    let crop_settings = crop_settings();
    let enhance_settings = enhancement_settings();

    c.bench_function("crop_and_enhance_pipeline", |b| {
        b.iter(|| {
            let crop =
                crop_face_from_image(black_box(&image), black_box(&detection), &crop_settings);
            let enhanced = apply_enhancements(black_box(&crop), &enhance_settings, None);
            black_box(enhanced);
        });
    });
}

criterion_group!(benches, crop_enhance_benchmark);
criterion_main!(benches);
