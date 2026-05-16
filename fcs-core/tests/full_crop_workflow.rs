use fcs_core::{
    BoundingBox, CropSettings, Detection, FillColor, Landmark, PositioningMode,
    crop_face_from_image,
};
use fcs_utils::{
    EnhancementSettings, MetadataContext, OutputOptions, Quality, QualityFilter,
    append_suffix_to_filename, apply_enhancements, config::CropSettings as ConfigCropSettings,
    estimate_sharpness, save_dynamic_image,
};
use image::{DynamicImage, GenericImageView, Rgba, RgbaImage};
use tempfile::tempdir;

fn make_detection(x: f32, y: f32, width: f32, height: f32, score: f32) -> Detection {
    Detection {
        bbox: BoundingBox {
            x,
            y,
            width,
            height,
        },
        landmarks: [
            Landmark {
                x: width.mul_add(0.3, x),
                y: height.mul_add(0.3, y),
            },
            Landmark {
                x: width.mul_add(0.7, x),
                y: height.mul_add(0.3, y),
            },
            Landmark {
                x: width.mul_add(0.5, x),
                y: height.mul_add(0.55, y),
            },
            Landmark {
                x: width.mul_add(0.35, x),
                y: height.mul_add(0.75, y),
            },
            Landmark {
                x: width.mul_add(0.65, x),
                y: height.mul_add(0.75, y),
            },
        ],
        score,
    }
}

fn build_app_crop_settings(core: &CropSettings) -> ConfigCropSettings {
    ConfigCropSettings {
        preset: "custom".to_string(),
        output_width: core.output_width,
        output_height: core.output_height,
        face_height_pct: core.face_height_pct,
        positioning_mode: core.positioning_mode,
        output_format: fcs_utils::ImageFormatHint::Png,
        auto_detect_format: false,
        ..ConfigCropSettings::default()
    }
}

#[test]
fn full_crop_workflow_selects_best_quality_face() {
    // Synthetic source image with one high-detail face region and one flat region.
    let mut base = RgbaImage::from_pixel(512, 512, Rgba([40, 40, 40, 255]));

    // High-detail chessboard pattern near the center.
    for y in 140..300 {
        for x in 160..320 {
            let val = if (x + y) % 2 == 0 { 230 } else { 30 };
            base.put_pixel(x, y, Rgba([val, 255 - val, val.saturating_sub(60), 255]));
        }
    }

    // Low-detail flat patch near the bottom left.
    for y in 320..420 {
        for x in 40..140 {
            base.put_pixel(x, y, Rgba([90, 92, 94, 255]));
        }
    }

    let source = DynamicImage::ImageRgba8(base);

    let detections = [
        make_detection(160.0, 160.0, 140.0, 140.0, 0.94),
        make_detection(40.0, 320.0, 90.0, 90.0, 0.78),
    ];

    let core_settings = CropSettings {
        output_width: 256,
        output_height: 256,
        face_height_pct: 70.0,
        positioning_mode: PositioningMode::Center,
        horizontal_offset: 0.0,
        vertical_offset: 0.0,
        fill_color: FillColor::default(),
        eye_line_align: false,
    };

    let cfg_crop = build_app_crop_settings(&core_settings);
    let output_opts = OutputOptions::from_crop_settings(&cfg_crop);

    let mut quality_filter = QualityFilter::new(Some(Quality::Medium));
    quality_filter.suffix_enabled = true;

    let enhancement_settings = EnhancementSettings {
        auto_color: true,
        contrast: 1.1,
        unsharp_amount: 0.4,
        unsharp_radius: 1.2,
        sharpness: 0.2,
        ..Default::default()
    };

    struct Candidate {
        detection: Detection,
        image: DynamicImage,
        quality: Quality,
        variance: f64,
        file_stub: String,
    }

    let mut candidates = Vec::new();
    for (idx, detection) in detections.iter().cloned().enumerate() {
        let crop = crop_face_from_image(&source, &detection, &core_settings);
        assert_eq!(crop.width(), core_settings.output_width);
        assert_eq!(crop.height(), core_settings.output_height);

        let enhanced = apply_enhancements(&crop, &enhancement_settings, None);
        let (variance, quality) = estimate_sharpness(&enhanced);

        candidates.push(Candidate {
            detection,
            image: enhanced,
            quality,
            variance,
            file_stub: format!("face_{idx}.png"),
        });
    }

    assert!(
        !candidates.is_empty(),
        "expected at least one candidate crop"
    );

    let quality_ranking: Vec<(Quality, f64)> =
        candidates.iter().map(|c| (c.quality, c.variance)).collect();

    let best_idx = quality_filter
        .select_best_index(&quality_ranking)
        .expect("quality filter should choose a best candidate");
    let best = &candidates[best_idx];

    assert!(
        !quality_filter.should_skip(best.quality),
        "selected crop should satisfy minimum quality threshold"
    );

    for (idx, candidate) in candidates.iter().enumerate() {
        if idx == best_idx {
            continue;
        }
        assert!(
            candidate.quality <= best.quality
                || (candidate.quality == best.quality && candidate.variance <= best.variance),
            "non-selected crop should not outrank the chosen candidate"
        );
    }

    let temp_dir = tempdir().expect("create temp dir");
    let mut file_name = best.file_stub.clone();
    if let Some(suffix) = quality_filter.suffix_for(best.quality) {
        file_name = append_suffix_to_filename(&file_name, suffix);
    }
    let destination = temp_dir.path().join(file_name);

    let metadata = MetadataContext {
        crop_settings: Some(&cfg_crop),
        detection_score: Some(best.detection.score),
        quality: Some(best.quality),
        quality_score: Some(best.variance),
        ..MetadataContext::default()
    };

    save_dynamic_image(&best.image, &destination, &output_opts, &metadata)
        .expect("export enhanced crop");

    assert!(destination.exists(), "exported crop should exist on disk");
    assert!(
        best.variance
            >= candidates
                .iter()
                .enumerate()
                .filter(|(idx, _)| *idx != best_idx)
                .map(|(_, c)| c.variance)
                .fold(0.0, f64::max),
        "selected crop should have the highest variance score"
    );

    let decoded = image::open(&destination).expect("re-open saved crop");
    assert_eq!(
        decoded.dimensions(),
        (core_settings.output_width, core_settings.output_height),
        "exported crop dimensions should match requested output size"
    );

    if let Some(expected_suffix) = quality_filter.suffix_for(best.quality) {
        let name = destination.file_name().unwrap().to_string_lossy();
        assert!(
            name.contains(expected_suffix),
            "exported file name should include quality suffix"
        );
    }
}
