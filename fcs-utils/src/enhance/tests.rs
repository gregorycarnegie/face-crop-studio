use super::*;
use image::{GenericImageView, RgbaImage};
use std::sync::Arc;

fn solid(color: [u8; 4]) -> DynamicImage {
    DynamicImage::ImageRgba8(RgbaImage::from_pixel(4, 4, image::Rgba(color)))
}

#[test]
fn histogram_equalization_stretches_levels() {
    let mut img = RgbaImage::new(4, 1);
    for x in 0..2 {
        img.put_pixel(x, 0, image::Rgba([40, 80, 120, 255]));
    }
    for x in 2..4 {
        img.put_pixel(x, 0, image::Rgba([200, 160, 100, 255]));
    }
    let out = apply_histogram_equalization(&DynamicImage::ImageRgba8(img));
    let buf = out.to_rgba8();
    let mut mins = [u8::MAX; 3];
    let mut maxs = [0u8; 3];
    for pixel in buf.pixels() {
        for c in 0..3 {
            mins[c] = mins[c].min(pixel[c]);
            maxs[c] = maxs[c].max(pixel[c]);
        }
    }
    for c in 0..3 {
        assert!(
            maxs[c] > mins[c],
            "expected channel {} max {} > min {}",
            c,
            maxs[c],
            mins[c]
        );
        assert!(
            (maxs[c] as i16 - mins[c] as i16) >= 100,
            "expected channel {} spread >=100, got {}",
            c,
            maxs[c] as i16 - mins[c] as i16
        );
    }
}

#[test]
fn exposure_positive_increases_values() {
    let img = solid([64, 64, 64, 255]);
    let out = apply_exposure(&img, 1.0);
    let buf = out.to_rgba8();
    let px = buf.get_pixel(0, 0);
    assert_eq!(px[0], 128);
}

#[test]
fn exposure_negative_darkens_values() {
    let img = solid([200, 200, 200, 255]);
    let out = apply_exposure(&img, -1.0);
    let buf = out.to_rgba8();
    let px = buf.get_pixel(0, 0);
    assert_eq!(px[0], 100);
}

#[test]
fn brightness_offsets_channels() {
    let img = solid([100, 100, 100, 255]);
    let out = apply_brightness(&img, 20);
    let buf = out.to_rgba8();
    let px = buf.get_pixel(0, 0);
    assert_eq!(px[0], 120);
}

#[test]
fn contrast_multiplier_expands_range() {
    let mut img = RgbaImage::from_pixel(4, 1, image::Rgba([128, 128, 128, 255]));
    img.put_pixel(0, 0, image::Rgba([80, 80, 80, 255]));
    img.put_pixel(3, 0, image::Rgba([180, 180, 180, 255]));
    let out = apply_contrast(&DynamicImage::ImageRgba8(img), 1.5);
    let buf = out.to_rgba8();
    assert!(buf.get_pixel(0, 0)[0] < 80);
    assert!(buf.get_pixel(3, 0)[0] > 180);
}

#[test]
fn saturation_zero_grays_image() {
    let img = solid([200, 100, 50, 255]);
    let out = apply_saturation(&img, 0.0);
    let buf = out.to_rgba8();
    let px = buf.get_pixel(0, 0);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
}

#[test]
fn unsharp_preserves_alpha() {
    let mut img = RgbaImage::new(4, 4);
    for y in 0..4 {
        for x in 0..4 {
            let val = if (x + y) % 2 == 0 { 0 } else { 255 };
            img.put_pixel(x, y, image::Rgba([val, val, val, 128]));
        }
    }
    let dyn_img = DynamicImage::ImageRgba8(img);
    let out = apply_unsharp_mask(&dyn_img, 1.5, 1.5).to_rgba8();
    assert_eq!(out.get_pixel(1, 1)[3], 128);
}

#[test]
fn skin_smoothing_reduces_noise() {
    // Create a noisy image with alternating pixel values
    let mut img = RgbaImage::new(8, 8);
    for y in 0..8 {
        for x in 0..8 {
            let val = if (x + y) % 2 == 0 { 100 } else { 110 };
            img.put_pixel(x, y, image::Rgba([val, val, val, 255]));
        }
    }
    let dyn_img = DynamicImage::ImageRgba8(img);

    // Apply skin smoothing
    let smoothed = apply_skin_smoothing(&dyn_img, 0.8, 3.0, 25.0).to_rgba8();

    // Check that neighboring pixels have become more similar (variance reduced)
    let p1 = smoothed.get_pixel(1, 1)[0];
    let p2 = smoothed.get_pixel(1, 2)[0];
    let diff = (p1 as i32 - p2 as i32).abs();

    // The difference should be less than the original 10
    assert!(diff < 10, "Smoothing should reduce pixel differences");

    // Alpha should be preserved
    assert_eq!(smoothed.get_pixel(1, 1)[3], 255);
}

#[test]
fn skin_smoothing_zero_amount_unchanged() {
    let img = solid([150, 120, 100, 255]);
    let out = apply_skin_smoothing(&img, 0.0, 3.0, 25.0);
    assert_eq!(
        out.to_rgba8().get_pixel(0, 0),
        img.to_rgba8().get_pixel(0, 0)
    );
}

#[test]
fn red_eye_removal_reduces_red_dominance() {
    // Create an image with a red-eye pixel (high red, low green/blue)
    let mut img = RgbaImage::new(4, 4);
    for y in 0..4 {
        for x in 0..4 {
            // Normal pixel
            img.put_pixel(x, y, image::Rgba([100, 100, 100, 255]));
        }
    }
    // Add a red-eye pixel in the center (very high red)
    img.put_pixel(1, 1, image::Rgba([200, 50, 50, 255]));
    img.put_pixel(2, 1, image::Rgba([220, 60, 55, 255]));

    let dyn_img = DynamicImage::ImageRgba8(img);
    let corrected = apply_red_eye_removal(&dyn_img, 1.5, None).to_rgba8();

    // Check that the red-eye pixels have been corrected
    let px1 = corrected.get_pixel(1, 1);
    let px2 = corrected.get_pixel(2, 1);

    // Red channel should be reduced (closer to green/blue average)
    assert!(px1[0] < 200, "Red channel should be reduced from 200");
    assert!(px2[0] < 220, "Red channel should be reduced from 220");

    // Normal pixels should be unchanged
    let normal_px = corrected.get_pixel(0, 0);
    assert_eq!(normal_px[0], 100);
    assert_eq!(normal_px[1], 100);
}

#[test]
fn red_eye_removal_preserves_alpha() {
    let img = solid([200, 50, 50, 128]);
    let out = apply_red_eye_removal(&img, 1.5, None).to_rgba8();
    assert_eq!(out.get_pixel(0, 0)[3], 128);
}

#[test]
fn background_blur_keeps_center_sharp() {
    // Create a simple gradient image
    let mut img = RgbaImage::new(100, 100);
    for y in 0..100 {
        for x in 0..100 {
            let val = ((x + y) / 2) as u8;
            img.put_pixel(x, y, image::Rgba([val, val, val, 255]));
        }
    }
    let dyn_img = DynamicImage::ImageRgba8(img.clone());

    // Apply background blur
    let blurred = apply_background_blur(&dyn_img, 10.0, 0.5).to_rgba8();

    // Center pixel should be nearly unchanged (inside mask)
    let center_orig = img.get_pixel(50, 50)[0];
    let center_blur = blurred.get_pixel(50, 50)[0];
    let diff = (center_orig as i32 - center_blur as i32).abs();
    assert!(diff < 3, "Center pixel should be nearly unchanged");

    // Corner pixel should be blurred (outside mask)
    // We can't directly test blur, but dimensions should be preserved
    assert_eq!(blurred.dimensions(), img.dimensions());
}

#[test]
fn background_blur_preserves_dimensions() {
    let img = solid([128, 128, 128, 255]);
    let out = apply_background_blur(&img, 15.0, 0.6);
    assert_eq!(out.dimensions(), img.dimensions());
}

#[test]
fn pipeline_preserves_dimensions() {
    let img = solid([128, 128, 128, 255]);
    let out = apply_enhancements(
        &img,
        &EnhancementSettings {
            auto_color: true,
            exposure_stops: 0.5,
            brightness: 10,
            contrast: 1.2,
            saturation: 1.1,
            unsharp_amount: 0.8,
            unsharp_radius: 1.2,
            sharpness: 0.1,
            skin_smooth_amount: 0.5,
            skin_smooth_sigma_space: 3.0,
            skin_smooth_sigma_color: 25.0,
            red_eye_removal: true,
            red_eye_threshold: 1.5,
            background_blur: true,
            background_blur_radius: 15.0,
            background_blur_mask_size: 0.6,
        },
        None,
    );
    assert_eq!(out.width(), img.width());
    assert_eq!(out.height(), img.height());
}

#[test]
fn pipeline_auto_color_matches_direct_equalization() {
    let mut img = RgbaImage::new(8, 1);
    for x in 0..4 {
        img.put_pixel(x, 0, image::Rgba([32, 64, 128, 255]));
    }
    for x in 4..8 {
        img.put_pixel(x, 0, image::Rgba([220, 180, 140, 255]));
    }
    let source = DynamicImage::ImageRgba8(img);

    let settings = EnhancementSettings {
        auto_color: true,
        unsharp_amount: 0.0,
        unsharp_radius: 0.0,
        sharpness: 0.0,
        ..EnhancementSettings::default()
    };

    let pipeline = apply_enhancements(&source, &settings, None).to_rgba8();
    let expected = apply_histogram_equalization(&source).to_rgba8();

    assert_eq!(pipeline, expected, "auto_color should reuse equalization");
    assert_ne!(
        pipeline,
        source.to_rgba8(),
        "auto_color should adjust levels"
    );
}

#[test]
fn pipeline_sharpness_combines_with_unsharp_amount() {
    let mut img = RgbaImage::new(5, 1);
    for x in 0..5 {
        let val = (x * 40 + 40) as u8;
        img.put_pixel(x, 0, image::Rgba([val, val, val, 255]));
    }
    let source = DynamicImage::ImageRgba8(img);

    let settings = EnhancementSettings {
        unsharp_amount: 0.0,
        sharpness: 0.6,
        unsharp_radius: 1.0,
        ..EnhancementSettings::default()
    };

    let pipeline = apply_enhancements(&source, &settings, None).to_rgba8();
    let expected = apply_unsharp_mask(&source, 0.6, 1.0).to_rgba8();

    assert_eq!(
        pipeline, expected,
        "sharpness setting should fold into unsharp mask amount"
    );
    assert_ne!(
        pipeline,
        source.to_rgba8(),
        "sharpening should modify pixels"
    );
}

#[test]
fn pipeline_red_eye_removal_matches_direct_call() {
    let mut img = RgbaImage::new(4, 2);
    for y in 0..2 {
        for x in 0..4 {
            img.put_pixel(x, y, image::Rgba([90, 90, 90, 255]));
        }
    }
    img.put_pixel(1, 0, image::Rgba([220, 40, 40, 255]));
    img.put_pixel(2, 0, image::Rgba([210, 45, 60, 255]));
    let source = DynamicImage::ImageRgba8(img);

    let settings = EnhancementSettings {
        red_eye_removal: true,
        red_eye_threshold: 1.2,
        unsharp_amount: 0.0,
        unsharp_radius: 0.0,
        sharpness: 0.0,
        ..EnhancementSettings::default()
    };

    let pipeline = apply_enhancements(&source, &settings, None).to_rgba8();
    let expected = apply_red_eye_removal(&source, 1.2, None).to_rgba8();

    assert_eq!(pipeline, expected);
    assert!(pipeline.get_pixel(1, 0)[0] < 200);
    assert!(pipeline.get_pixel(2, 0)[0] < 210);
}

#[test]
fn pipeline_background_blur_matches_direct_call() {
    let mut img = RgbaImage::new(32, 32);
    for y in 0..32 {
        for x in 0..32 {
            let val = ((x + y) * 4).clamp(0, 255) as u8;
            img.put_pixel(x, y, image::Rgba([val, 255 - val, val / 2, 255]));
        }
    }
    let source = DynamicImage::ImageRgba8(img);

    let settings = EnhancementSettings {
        background_blur: true,
        background_blur_radius: 6.0,
        background_blur_mask_size: 0.5,
        unsharp_amount: 0.0,
        unsharp_radius: 0.0,
        sharpness: 0.0,
        ..EnhancementSettings::default()
    };

    let pipeline = apply_enhancements(&source, &settings, None).to_rgba8();
    let expected = apply_background_blur(&source, 6.0, 0.5).to_rgba8();

    assert_eq!(pipeline, expected);
    // ensure the blur keeps center mostly intact while affecting a corner
    let center_diff = (pipeline.get_pixel(16, 16)[0] as i16
        - source.to_rgba8().get_pixel(16, 16)[0] as i16)
        .abs();
    assert!(center_diff < 5, "central region should remain sharp");
    let corner_diff =
        (pipeline.get_pixel(0, 0)[0] as i16 - source.to_rgba8().get_pixel(0, 0)[0] as i16).abs();
    assert!(corner_diff > 0, "corner should show blur impact");
}

// -----------------------------------------------------------------------
// equalization edge cases

#[test]
fn equalization_lut_zero_total_returns_identity() {
    // An all-zero histogram (total == 0) must not divide by zero and must
    // return the identity mapping.
    let hist = [0u32; 256];
    let lut = build_equalization_lut(&hist, 0);
    for (i, &v) in lut.iter().enumerate() {
        assert_eq!(v, i as u8, "identity expected at index {i}");
    }
}

#[test]
fn equalization_lut_single_value_image_returns_identity() {
    // Every pixel has the same intensity: cdf_min == total, so the result
    // is undefined — we return identity to avoid divide-by-zero.
    let mut hist = [0u32; 256];
    hist[128] = 16; // all 16 pixels are intensity 128
    let lut = build_equalization_lut(&hist, 16);
    for (i, &v) in lut.iter().enumerate() {
        assert_eq!(v, i as u8, "identity expected at index {i}");
    }
}

// -----------------------------------------------------------------------
// red-eye removal with eye regions

#[test]
fn red_eye_removal_with_eye_region_only_corrects_inside() {
    use crate::gpu::red_eye::RedEye;

    let mut img = RgbaImage::new(8, 8);
    // Fill with neutral gray
    for y in 0..8 {
        for x in 0..8 {
            img.put_pixel(x, y, image::Rgba([90, 90, 90, 255]));
        }
    }
    // Plant two red-eye pixels inside the eye radius (cx=4, cy=4, r=2)
    img.put_pixel(4, 4, image::Rgba([220, 50, 50, 255]));
    img.put_pixel(3, 4, image::Rgba([210, 45, 45, 255]));
    // And a red pixel clearly outside the radius
    img.put_pixel(0, 0, image::Rgba([220, 50, 50, 255]));

    let source = DynamicImage::ImageRgba8(img);
    let eyes = [RedEye {
        x: 4.0,
        y: 4.0,
        radius: 2.0,
        _pad: 0.0,
    }];
    let out = apply_red_eye_removal(&source, 1.5, Some(&eyes)).to_rgba8();

    // Pixels inside the eye circle must be corrected (red reduced)
    assert!(
        out.get_pixel(4, 4)[0] < 220,
        "red inside eye region should be reduced"
    );
    assert!(
        out.get_pixel(3, 4)[0] < 210,
        "red inside eye region should be reduced"
    );
    // Pixel outside the eye region must be UNCHANGED despite being red
    assert_eq!(
        out.get_pixel(0, 0)[0],
        220,
        "pixel outside eye region must not be corrected"
    );
}

// -----------------------------------------------------------------------
// background blur helpers

#[test]
fn background_blur_with_preblur_dimension_mismatch_returns_sharp() {
    // If the pre-blurred image has different dimensions, background_blur_from_rgba
    // bails out and returns the original sharp image unchanged.
    let sharp = RgbaImage::from_pixel(4, 4, image::Rgba([100u8, 150, 200, 255]));
    let wrong_size_blur = RgbaImage::from_pixel(8, 8, image::Rgba([0u8, 0, 0, 255]));
    let sharp_dyn = DynamicImage::ImageRgba8(sharp.clone());
    let blur_dyn = DynamicImage::ImageRgba8(wrong_size_blur);

    let out = apply_background_blur_with_preblur(&sharp_dyn, &blur_dyn, 0.6).to_rgba8();
    assert_eq!(
        out, sharp,
        "mismatched dimensions should return the sharp image"
    );
}

#[test]
fn background_blur_with_preblur_matches_direct_apply() {
    let mut img = RgbaImage::new(20, 20);
    for y in 0..20 {
        for x in 0..20 {
            img.put_pixel(x, y, image::Rgba([((x + y) * 6) as u8, 100, 50, 255]));
        }
    }
    let source = DynamicImage::ImageRgba8(img);
    let mask_size = 0.5f32;
    let radius = 4.0f32;

    // Produce the pre-blurred image the same way apply_background_blur does
    let pre_blurred =
        DynamicImage::ImageRgba8(image::imageops::fast_blur(&source.to_rgba8(), radius));
    let via_preblur =
        apply_background_blur_with_preblur(&source, &pre_blurred, mask_size).to_rgba8();
    let via_direct = apply_background_blur(&source, radius, mask_size).to_rgba8();

    assert_eq!(
        via_preblur, via_direct,
        "pre-blurred and direct paths must produce identical output"
    );
}

// -----------------------------------------------------------------------
// unsharp-mask with pre-blurred image

#[test]
fn unsharp_with_preblur_direct_matches_indirect() {
    let mut img = RgbaImage::new(6, 6);
    for y in 0..6 {
        for x in 0..6 {
            let v = ((x * 20 + y * 15) % 200 + 30) as u8;
            img.put_pixel(x, y, image::Rgba([v, 255 - v, v / 2, 200]));
        }
    }
    let source = DynamicImage::ImageRgba8(img);
    let amount = 1.2f32;
    let radius = 1.0f32;

    let pre_blurred =
        DynamicImage::ImageRgba8(image::imageops::fast_blur(&source.to_rgba8(), radius));
    let via_preblur = apply_unsharp_with_preblur(&source, &pre_blurred, amount).to_rgba8();
    let via_direct = apply_unsharp_mask(&source, amount, radius).to_rgba8();

    assert_eq!(
        via_preblur, via_direct,
        "pre-blurred and direct unsharp paths must match"
    );
}

// -----------------------------------------------------------------------
// skin kernel cache

#[test]
fn skin_kernel_cache_hit_returns_same_arc() {
    let a = skin_kernel(3, 3.0, 25.0);
    let b = skin_kernel(3, 3.0, 25.0);
    // Same Arc pointer means the cache returned the same allocation
    assert!(
        Arc::ptr_eq(&a, &b),
        "second call with identical params must hit the cache"
    );
    // Different params must produce a distinct kernel
    let c = skin_kernel(4, 3.0, 25.0);
    assert!(
        !Arc::ptr_eq(&a, &c),
        "different params must produce a distinct kernel"
    );
}

// -----------------------------------------------------------------------
// saturation scalar tail path

#[test]
fn saturation_scalar_tail_is_exercised_by_odd_pixel_count() {
    // 5 pixels = 1 SIMD batch of 4 + 1 scalar remainder pixel
    let mut img = RgbaImage::new(5, 1);
    img.put_pixel(0, 0, image::Rgba([200, 100, 50, 255]));
    img.put_pixel(1, 0, image::Rgba([180, 90, 40, 255]));
    img.put_pixel(2, 0, image::Rgba([160, 80, 30, 255]));
    img.put_pixel(3, 0, image::Rgba([140, 70, 20, 255]));
    img.put_pixel(4, 0, image::Rgba([120, 60, 10, 255]));

    let source = DynamicImage::ImageRgba8(img.clone());
    let out = apply_saturation(&source, 0.0).to_rgba8();

    // At saturation=0 every pixel must be fully desaturated (all channels equal)
    for x in 0..5 {
        let px = out.get_pixel(x, 0);
        assert_eq!(
            px[0], px[1],
            "pixel {x}: R and G must be equal at zero saturation"
        );
        assert_eq!(
            px[1], px[2],
            "pixel {x}: G and B must be equal at zero saturation"
        );
    }
}

#[test]
fn saturation_simd_and_scalar_paths_agree() {
    // Run saturation on 4 pixels (SIMD only) and 5 pixels (SIMD + scalar)
    // and verify the first 4 pixels match in both outputs.
    let make_img = |width: u32| {
        let mut img = RgbaImage::new(width, 1);
        for x in 0..width {
            img.put_pixel(x, 0, image::Rgba([(x * 40 + 40) as u8, 100, 50, 255]));
        }
        DynamicImage::ImageRgba8(img)
    };

    let out4 = apply_saturation(&make_img(4), 1.5).to_rgba8();
    let out5 = apply_saturation(&make_img(5), 1.5).to_rgba8();

    for x in 0..4 {
        assert_eq!(
            out4.get_pixel(x, 0),
            out5.get_pixel(x, 0),
            "pixel {x}: SIMD and scalar paths must agree"
        );
    }
}
