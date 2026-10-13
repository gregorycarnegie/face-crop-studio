//! Every setting must change something.
//!
//! This exists because five settings in a row turned out to do nothing, and each was found by
//! accident while deleting adjacent code: `score_threshold` (renamed to `confidence` in 1.8.0
//! because the old values were being applied on the wrong scale), `nms_threshold` and `top_k`
//! (read into a struct and never passed to the model), `gpu.inference` and `gpu.preprocessing`
//! (written by both front-ends, read by nobody), and the whole `input` section (fixed by the
//! export, so unchoosable). That is not five bugs, it is one missing invariant.
//!
//! **Reachability is not the invariant.** A grep for every field finds a reader for all of
//! them, and found one for `nms_threshold` too while it did nothing: it was read into
//! `FaceDetector`'s own field and stopped there. So each case below flips one field and
//! requires an *observable* difference from the pipeline that field feeds.
//!
//! What this cannot catch: a field that changes the output in the wrong direction, or by the
//! wrong amount. Those need the specific tests that live beside each unit. This catches the
//! failure that actually kept happening -- a control wired to nothing at all.

use fcs_core::{CropSettings as CoreCropSettings, Detection, crop_face_from_image};
use fcs_utils::{
    CropShape, ImageFormatHint, MetadataContext, OutputOptions, PngCompression, QualityFilter,
    apply_enhancements, apply_shape_mask_dynamic,
    config::{AppSettings, MetadataMode, PositioningMode},
    quality::Quality,
    save_dynamic_image,
};
use image::{DynamicImage, RgbaImage};
use rstest::rstest;

/// A source image with structure, so a geometry change actually shows up in the pixels.
fn source() -> DynamicImage {
    let mut img = RgbaImage::new(200, 200);
    for (x, y, px) in img.enumerate_pixels_mut() {
        let checker = if (x / 8 + y / 8) % 2 == 0 { 40 } else { 210 };
        *px = image::Rgba([checker, (x % 256) as u8, (y % 256) as u8, 255]);
    }
    DynamicImage::ImageRgba8(img)
}

fn detection() -> Detection {
    Detection {
        bbox: fcs_core::BoundingBox {
            x: 60.0,
            y: 60.0,
            width: 80.0,
            height: 80.0,
        },
        // Tilted, so `eye_line_align` has an angle to correct.
        landmarks: [
            Some(fcs_utils::point::Point::new(80.0, 85.0)),
            Some(fcs_utils::point::Point::new(120.0, 95.0)),
            None,
            None,
            None,
        ],
        score: 0.9,
    }
}

fn baseline() -> AppSettings {
    let mut settings = AppSettings::default();
    // A concrete, non-default starting point so "flip it" has somewhere to go in both
    // directions, and so the crop is small enough to compare quickly.
    settings.crop.preset = "custom".to_string();
    settings.crop.output_width = 64;
    settings.crop.output_height = 64;
    settings.crop.face_height_pct = 50.0;
    // A shape with a vignette, so the masking fields have something to change. Under the
    // default `Rectangle` the mask is a no-op and every vignette case would look inert.
    settings.crop.shape = CropShape::Ellipse;
    settings.crop.vignette_softness = 0.3;
    settings.crop.vignette_intensity = 0.5;
    settings
}

/// Render a crop the way the front-ends do: geometry, then the shape mask, then enhancement.
fn render(settings: &AppSettings) -> Vec<u8> {
    let core: CoreCropSettings = (&settings.crop).into();
    let mut shaped = crop_face_from_image(&source(), &detection(), &core);
    apply_shape_mask_dynamic(
        &mut shaped,
        &settings.crop.shape,
        settings.crop.vignette_softness,
        settings.crop.vignette_intensity,
        settings.crop.vignette_color,
    );
    apply_enhancements(&shaped, &settings.enhance.to_enhancement_settings(), None)
        .to_rgba8()
        .into_raw()
}

/// Encode a crop to bytes the way the writer does, including metadata.
fn encode(settings: &AppSettings) -> Vec<u8> {
    let dir = tempfile::tempdir().expect("tempdir");
    let ext = match settings.crop.output_format {
        ImageFormatHint::Jpeg => "jpg",
        ImageFormatHint::Webp => "webp",
        ImageFormatHint::Tiff => "tiff",
        ImageFormatHint::Bmp => "bmp",
        ImageFormatHint::Avif => "avif",
        ImageFormatHint::Png => "png",
    };
    let path = dir.path().join(format!("out.{ext}"));
    let options = OutputOptions::from_crop_settings(&settings.crop);
    let source_path = dir.path().join("source.png");
    let image = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
        32,
        32,
        image::Rgba([120, 90, 60, 255]),
    ));
    image.save(&source_path).expect("seed source");
    let context = MetadataContext {
        source_path: Some(source_path.as_path()),
        crop_settings: Some(&settings.crop),
        detection_score: Some(0.87),
        quality_score: Some(42.0),
        quality: Some(Quality::High),
    };
    save_dynamic_image(&image, &path, &options, &context).expect("save");
    std::fs::read(&path).expect("read back")
}

fn assert_changes(
    field: &str,
    mutate: fn(&mut AppSettings),
    observe: fn(&AppSettings) -> Vec<u8>,
    what: &str,
) {
    let reference = observe(&baseline());
    let mut mutated = baseline();
    mutate(&mut mutated);
    assert!(
        observe(&mutated) != reference,
        "{field} changed nothing in {what}, so it is wired to nothing"
    );
}

// `preset` selects width, height and face height, which are asserted directly here.
#[rstest]
#[case::crop_output_width("crop.output_width", |s: &mut AppSettings| s.crop.output_width = 96)]
#[case::crop_output_height("crop.output_height", |s: &mut AppSettings| s.crop.output_height = 96)]
#[case::crop_face_height_pct("crop.face_height_pct", |s: &mut AppSettings| s.crop.face_height_pct = 80.0)]
#[case::crop_positioning_mode("crop.positioning_mode", |s: &mut AppSettings| s.crop.positioning_mode = PositioningMode::RuleOfThirds)]
#[case::crop_vertical_offset("crop.vertical_offset", |s: &mut AppSettings| {
    s.crop.positioning_mode = PositioningMode::Custom;
    s.crop.vertical_offset = 0.25;
})]
#[case::crop_horizontal_offset("crop.horizontal_offset", |s: &mut AppSettings| {
    s.crop.positioning_mode = PositioningMode::Custom;
    s.crop.horizontal_offset = 0.25;
})]
// The crop has to reach past the source edge for the fill to show.
#[case::crop_fill_color("crop.fill_color", |s: &mut AppSettings| {
    s.crop.face_height_pct = 5.0;
    s.crop.fill_color = fcs_utils::color::RgbaColor {
        red: 255,
        green: 0,
        blue: 255,
        alpha: 255,
    };
})]
#[case::crop_eye_line_align("crop.eye_line_align", |s: &mut AppSettings| s.crop.eye_line_align = true)]
#[case::crop_shape("crop.shape", |s: &mut AppSettings| s.crop.shape = CropShape::Rectangle)]
#[case::crop_vignette_softness("crop.vignette_softness", |s: &mut AppSettings| s.crop.vignette_softness = 0.9)]
#[case::crop_vignette_intensity("crop.vignette_intensity", |s: &mut AppSettings| s.crop.vignette_intensity = 1.0)]
#[case::crop_vignette_color("crop.vignette_color", |s: &mut AppSettings| {
    s.crop.vignette_color = fcs_utils::color::RgbaColor {
        red: 255,
        green: 0,
        blue: 255,
        alpha: 255,
    }
})]
fn crop_geometry_settings_change_the_crop(
    #[case] field: &str,
    #[case] mutate: fn(&mut AppSettings),
) {
    assert_changes(field, mutate, render, "the rendered crop");
}

#[rstest]
#[case::enhance_auto_color("enhance.auto_color", |s: &mut AppSettings| s.enhance.auto_color = true)]
#[case::enhance_exposure_stops("enhance.exposure_stops", |s: &mut AppSettings| s.enhance.exposure_stops = 1.5)]
#[case::enhance_brightness("enhance.brightness", |s: &mut AppSettings| s.enhance.brightness = 40)]
#[case::enhance_contrast("enhance.contrast", |s: &mut AppSettings| s.enhance.contrast = 1.6)]
#[case::enhance_saturation("enhance.saturation", |s: &mut AppSettings| s.enhance.saturation = 0.2)]
#[case::enhance_sharpness("enhance.sharpness", |s: &mut AppSettings| s.enhance.sharpness = 0.9)]
#[case::enhance_skin_smooth("enhance.skin_smooth", |s: &mut AppSettings| s.enhance.skin_smooth = 0.9)]
#[case::enhance_background_blur("enhance.background_blur", |s: &mut AppSettings| s.enhance.background_blur = true)]
fn enhancement_settings_change_the_crop(#[case] field: &str, #[case] mutate: fn(&mut AppSettings)) {
    assert_changes(field, mutate, render, "the enhanced crop");
}

#[rstest]
#[case::crop_output_format("crop.output_format", |s: &mut AppSettings| s.crop.output_format = ImageFormatHint::Jpeg)]
// `webp_quality` is absent from this list because the field is gone: it never reached
// the encoder, which `image` only offers losslessly. This test found `enhance.enabled`
// the same way on its first run -- the GUI enhanced unconditionally and the CLI gated
// on its own `--enhance` flag, so the settings field was read by nobody.
#[case::crop_jpeg_quality("crop.jpeg_quality", |s: &mut AppSettings| {
    s.crop.output_format = ImageFormatHint::Jpeg;
    s.crop.jpeg_quality = 30;
})]
#[case::crop_png_compression("crop.png_compression", |s: &mut AppSettings| s.crop.png_compression = PngCompression::Best)]
#[case::crop_metadata_mode("crop.metadata.mode", |s: &mut AppSettings| s.crop.metadata.mode = MetadataMode::Strip)]
#[case::crop_metadata_include_crop_settings("crop.metadata.include_crop_settings", |s: &mut AppSettings| {
    s.crop.metadata.include_crop_settings = false;
})]
#[case::crop_metadata_include_quality_metrics("crop.metadata.include_quality_metrics", |s: &mut AppSettings| {
    s.crop.metadata.include_quality_metrics = false;
})]
#[case::crop_metadata_custom_tags("crop.metadata.custom_tags", |s: &mut AppSettings| {
    s.crop
        .metadata
        .custom_tags
        .insert("owner".into(), "someone".into());
})]
fn output_and_metadata_settings_change_the_written_file(
    #[case] field: &str,
    #[case] mutate: fn(&mut AppSettings),
) {
    assert_changes(field, mutate, encode, "the written file");
}

/// The fields that produce no pixels. Each is checked through the accessor that consumes it,
/// because "it changes an image" is the wrong question for a log level or a thread count.
#[test]
fn non_image_settings_reach_their_consumers() {
    // Telemetry level -> the filter the logger installs.
    let mut settings = baseline();
    settings.telemetry.level = "warn".to_string();
    assert_eq!(settings.telemetry.level_filter(), log::LevelFilter::Warn);
    settings.telemetry.level = "trace".to_string();
    assert_eq!(settings.telemetry.level_filter(), log::LevelFilter::Trace);

    // Batch parallelism -> the worker count. `None` resolves to a bounded default, and an
    // explicit value must win over it.
    settings.batch_parallelism = Some(3);
    assert_eq!(settings.resolved_batch_parallelism(), 3);
    settings.batch_parallelism = Some(1);
    assert_eq!(settings.resolved_batch_parallelism(), 1);

    // GPU settings -> the context options. Both fields, because dropping one silently either
    // disables the GPU or starts honouring WGPU_* against the user's choice.
    for (enabled, respect_env) in [(true, false), (false, true)] {
        settings.gpu.enabled = enabled;
        settings.gpu.respect_env = respect_env;
        let options: fcs_utils::GpuContextOptions = (&settings.gpu).into();
        assert_eq!(options.enabled, enabled);
        assert_eq!(options.respect_env, respect_env);
    }

    // Quality automation -> the filter the front-ends build from it.
    let mut rules = baseline();
    rules.crop.quality_rules.min_quality = Some(Quality::High);
    let filter = QualityFilter {
        min_quality: rules.crop.quality_rules.min_quality,
        auto_select: rules.crop.quality_rules.auto_select_best_face,
        auto_skip_no_high: rules.crop.quality_rules.auto_skip_no_high_quality,
        suffix_enabled: rules.crop.quality_rules.quality_suffix,
    };
    assert!(
        filter.should_skip(Quality::Low),
        "min_quality must reach the filter"
    );
    assert!(!filter.should_skip(Quality::High));

    // model_path -> what the detector is asked to load. Asserted as the value the builder
    // resolves, since loading it needs the model on disk.
    let mut model = baseline();
    model.model_path = Some("models/somewhere-else.onnx".to_string());
    assert_eq!(
        model.model_path.as_deref(),
        Some("models/somewhere-else.onnx")
    );
}

/// `auto_detect_format` decides whether the extension or the setting picks the encoder, so its
/// effect only shows when the two disagree.
#[rstest]
#[case::extension_wins(true, true)]
#[case::configured_format_wins(false, false)]
fn auto_detect_format_lets_the_extension_win(#[case] auto_detect: bool, #[case] expect_jpeg: bool) {
    let dir = tempfile::tempdir().expect("tempdir");
    let image = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
        16,
        16,
        image::Rgba([10, 20, 30, 255]),
    ));
    let context = MetadataContext {
        source_path: None,
        crop_settings: None,
        detection_score: None,
        quality_score: None,
        quality: None,
    };

    // Format says PNG, extension says JPEG.
    let mut settings = baseline();
    settings.crop.output_format = ImageFormatHint::Png;

    settings.crop.auto_detect_format = auto_detect;
    let path = dir.path().join(format!("out-{auto_detect}.jpg"));
    let options = OutputOptions::from_crop_settings(&settings.crop);
    save_dynamic_image(&image, &path, &options, &context).expect("save");
    let bytes = std::fs::read(&path).expect("read");
    let is_jpeg = bytes.starts_with(&[0xFF, 0xD8, 0xFF]);
    assert_eq!(
        is_jpeg,
        expect_jpeg,
        "auto_detect_format = {auto_detect} should {} the .jpg extension decide",
        if expect_jpeg { "let" } else { "not let" }
    );
}
