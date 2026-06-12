use super::*;
use crate::{
    color::RgbaColor,
    config::{CropSettings, PositioningMode},
};
use image::{DynamicImage, Rgba, RgbaImage};
use serde_json::Value;
use std::{collections::BTreeMap, path::PathBuf};
use tempfile::tempdir;

fn sample_image() -> DynamicImage {
    DynamicImage::ImageRgba8(RgbaImage::from_pixel(2, 2, Rgba([12, 34, 56, 255])))
}

fn make_png_chunk(chunk_type: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut chunk = Vec::with_capacity(data.len() + 12);
    chunk.extend_from_slice(&(data.len() as u32).to_be_bytes());
    chunk.extend_from_slice(chunk_type);
    chunk.extend_from_slice(data);

    let mut hasher = Crc32::new();
    hasher.update(chunk_type);
    hasher.update(data);
    chunk.extend_from_slice(&hasher.finalize().to_be_bytes());
    chunk
}

fn make_exif_segment(payload_suffix: &[u8]) -> Vec<u8> {
    let mut payload = b"Exif\0\0".to_vec();
    payload.extend_from_slice(payload_suffix);

    let mut segment = Vec::with_capacity(payload.len() + 4);
    segment.extend_from_slice(&[0xFF, 0xE1]);
    segment.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
    segment.extend_from_slice(&payload);
    segment
}

#[test]
fn image_format_hint_from_extension_accepts_common_aliases() {
    assert_eq!(
        ImageFormatHint::from_extension("png"),
        Some(ImageFormatHint::Png)
    );
    assert_eq!(
        ImageFormatHint::from_extension("JPG"),
        Some(ImageFormatHint::Jpeg)
    );
    assert_eq!(
        ImageFormatHint::from_extension("jpeg"),
        Some(ImageFormatHint::Jpeg)
    );
    assert_eq!(
        ImageFormatHint::from_extension("tif"),
        Some(ImageFormatHint::Tiff)
    );
    assert_eq!(
        ImageFormatHint::from_extension("bmp"),
        Some(ImageFormatHint::Bmp)
    );
    assert_eq!(ImageFormatHint::from_extension("gif"), None);
}

#[test]
fn png_compression_parse_maps_keywords_and_numeric_levels() {
    assert_eq!(PngCompression::parse("fast"), PngCompression::Fast);
    assert_eq!(PngCompression::parse("default"), PngCompression::Default);
    assert_eq!(PngCompression::parse("best"), PngCompression::Best);
    assert_eq!(PngCompression::parse("0"), PngCompression::Fast);
    assert_eq!(PngCompression::parse("3"), PngCompression::Fast);
    assert_eq!(PngCompression::parse("5"), PngCompression::Default);
    assert_eq!(PngCompression::parse("9"), PngCompression::Best);
    assert_eq!(PngCompression::parse("invalid"), PngCompression::Default);
}

#[test]
fn output_options_from_crop_settings_clamps_values() {
    let mut settings = CropSettings {
        output_format: ImageFormatHint::Jpeg,
        jpeg_quality: 0,
        png_compression: PngCompression::Best,
        webp_quality: 200,
        auto_detect_format: false,
        ..CropSettings::default()
    };
    settings.metadata.mode = MetadataMode::Custom;

    let options = OutputOptions::from_crop_settings(&settings);

    assert_eq!(options.format, Some(ImageFormatHint::Jpeg));
    assert!(!options.auto_detect);
    assert_eq!(options.jpeg_quality, 1);
    assert_eq!(options.png_compression, PngCompression::Best);
    assert_eq!(options.webp_quality, 100);
    assert_eq!(options.metadata.mode, MetadataMode::Custom);
}

#[test]
fn determine_format_prefers_extension_when_auto_detect_is_enabled() {
    let options = OutputOptions {
        format: Some(ImageFormatHint::Png),
        auto_detect: true,
        jpeg_quality: 90,
        png_compression: PngCompression::Default,
        webp_quality: 90,
        metadata: MetadataSettings::default(),
    };

    assert_eq!(
        determine_format(Path::new("output.jpeg"), &options),
        ImageFormatHint::Jpeg
    );
    assert_eq!(
        determine_format(Path::new("output.unknown"), &options),
        ImageFormatHint::Png
    );
}

#[test]
fn append_suffix_to_filename_preserves_extension() {
    assert_eq!(
        append_suffix_to_filename("portrait.png", "_highq"),
        "portrait_highq.png"
    );
    assert_eq!(
        append_suffix_to_filename("archive.tar.gz", "_v2"),
        "archive.tar_v2.gz"
    );
    assert_eq!(
        append_suffix_to_filename("portrait", "_highq"),
        "portrait_highq"
    );
    assert_eq!(
        append_suffix_to_filename("portrait.png", ""),
        "portrait.png"
    );
}

#[test]
fn save_dynamic_image_creates_missing_parent_directories() {
    let dir = tempdir().unwrap();
    let destination = dir.path().join("nested").join("exports").join("face.png");
    let options = OutputOptions {
        format: Some(ImageFormatHint::Png),
        auto_detect: false,
        jpeg_quality: 90,
        png_compression: PngCompression::Default,
        webp_quality: 90,
        metadata: MetadataSettings {
            mode: MetadataMode::Strip,
            ..MetadataSettings::default()
        },
    };

    save_dynamic_image(
        &sample_image(),
        &destination,
        &options,
        &MetadataContext::default(),
    )
    .unwrap();

    assert!(destination.exists());
    let bytes = fs::read(&destination).unwrap();
    assert!(bytes.starts_with(b"\x89PNG\r\n\x1a\n"));
}

#[test]
fn save_dynamic_image_auto_detects_format_from_destination_extension() {
    let dir = tempdir().unwrap();
    let destination = dir.path().join("face.jpg");
    let options = OutputOptions {
        format: Some(ImageFormatHint::Png),
        auto_detect: true,
        jpeg_quality: 90,
        png_compression: PngCompression::Default,
        webp_quality: 90,
        metadata: MetadataSettings {
            mode: MetadataMode::Strip,
            ..MetadataSettings::default()
        },
    };

    save_dynamic_image(
        &sample_image(),
        &destination,
        &options,
        &MetadataContext::default(),
    )
    .unwrap();

    let bytes = fs::read(&destination).unwrap();
    assert_eq!(&bytes[..2], &[0xFF, 0xD8]);
}

#[test]
fn load_png_exif_chunks_returns_empty_for_non_png_sources() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("source.jpg");
    fs::write(&path, b"not-a-png").unwrap();

    assert!(load_png_exif_chunks(None).is_empty());
    assert!(load_png_exif_chunks(Some(&path)).is_empty());
}

#[test]
fn load_png_exif_chunks_extracts_embedded_exif_chunks() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("source.png");
    let encoded = encode_png(&sample_image(), PngCompression::Default).unwrap();
    let exif_chunk = make_png_chunk(b"eXIf", b"exif-payload");
    let png_with_exif = inject_png_metadata(encoded, std::slice::from_ref(&exif_chunk), None);
    fs::write(&path, png_with_exif).unwrap();

    assert_eq!(load_png_exif_chunks(Some(&path)), vec![exif_chunk]);
}

#[test]
fn load_jpeg_exif_returns_none_for_non_jpeg_sources() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("source.png");
    fs::write(&path, b"not-a-jpeg").unwrap();

    assert!(load_jpeg_exif(None).is_none());
    assert!(load_jpeg_exif(Some(&path)).is_none());
}

#[test]
fn load_jpeg_exif_extracts_embedded_exif_segment() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("source.jpg");
    let encoded = encode_jpeg(&sample_image(), 90).unwrap();
    let exif_segment = make_exif_segment(b"minimal-exif");
    let jpeg_with_exif = inject_jpeg_metadata(encoded, Some(exif_segment.clone()), None);
    fs::write(&path, jpeg_with_exif).unwrap();

    assert_eq!(load_jpeg_exif(Some(&path)), Some(exif_segment));
}

#[test]
fn inject_png_metadata_inserts_chunks_after_ihdr() {
    let encoded = encode_png(&sample_image(), PngCompression::Default).unwrap();
    let exif_chunk = make_png_chunk(b"eXIf", b"payload");
    let output = inject_png_metadata(
        encoded,
        std::slice::from_ref(&exif_chunk),
        Some("{\"meta\":true}"),
    );

    let mut cursor = 8usize;
    let ihdr_length = u32::from_be_bytes(output[cursor..cursor + 4].try_into().unwrap()) as usize;
    let ihdr_total = 8 + ihdr_length + 4;
    cursor += ihdr_total;

    assert_eq!(
        &output[cursor..cursor + exif_chunk.len()],
        exif_chunk.as_slice()
    );
    cursor += exif_chunk.len();

    let text_len = u32::from_be_bytes(output[cursor..cursor + 4].try_into().unwrap()) as usize;
    assert_eq!(&output[cursor + 4..cursor + 8], b"tEXt");
    let text_data = &output[cursor + 8..cursor + 8 + text_len];
    assert!(text_data.starts_with(b"IronCropper\0"));
    assert!(text_data.ends_with(b"{\"meta\":true}"));
}

#[test]
fn inject_jpeg_metadata_inserts_exif_and_xmp_after_soi() {
    let encoded = encode_jpeg(&sample_image(), 90).unwrap();
    let exif_segment = make_exif_segment(b"payload");
    let output = inject_jpeg_metadata(
        encoded.clone(),
        Some(exif_segment.clone()),
        Some("{\"quality\":\"high\"}"),
    );

    assert_eq!(&output[..2], &[0xFF, 0xD8]);
    assert_eq!(&output[2..2 + exif_segment.len()], exif_segment.as_slice());

    let xmp_start = 2 + exif_segment.len();
    assert_eq!(&output[xmp_start..xmp_start + 2], &[0xFF, 0xE1]);
    let xmp_len =
        u16::from_be_bytes(output[xmp_start + 2..xmp_start + 4].try_into().unwrap()) as usize;
    let xmp_payload = &output[xmp_start + 4..xmp_start + 2 + xmp_len];
    assert!(xmp_payload.starts_with(b"http://ns.adobe.com/xap/1.0/\0"));
    assert!(String::from_utf8_lossy(xmp_payload).contains("iron:Metadata"));
    assert!(output.ends_with(&encoded[2..]));
}

#[test]
fn build_custom_metadata_payload_returns_none_for_strip_mode() {
    let settings = MetadataSettings {
        mode: MetadataMode::Strip,
        ..MetadataSettings::default()
    };

    assert!(
        build_custom_metadata_payload(&settings, &MetadataContext::default())
            .unwrap()
            .is_none()
    );
}

#[test]
fn build_custom_metadata_payload_includes_crop_quality_and_custom_tags() {
    let mut custom_tags = BTreeMap::new();
    custom_tags.insert("job_id".to_string(), "1234".to_string());

    let settings = MetadataSettings {
        mode: MetadataMode::Custom,
        include_crop_settings: true,
        include_quality_metrics: true,
        custom_tags,
    };

    let crop = CropSettings {
        preset: "linkedin".to_string(),
        output_width: 400,
        output_height: 500,
        face_height_pct: 72.5,
        positioning_mode: PositioningMode::Custom,
        horizontal_offset: 0.25,
        vertical_offset: -0.1,
        fill_color: RgbaColor::opaque(1, 2, 3),
        ..CropSettings::default()
    };
    let source = PathBuf::from("source.jpg");

    let payload = build_custom_metadata_payload(
        &settings,
        &MetadataContext {
            source_path: Some(&source),
            crop_settings: Some(&crop),
            detection_score: Some(0.91),
            quality: Some(Quality::High),
            quality_score: Some(1234.5),
        },
    )
    .unwrap()
    .unwrap();

    let parsed: Value = serde_json::from_str(&payload).unwrap();
    assert_eq!(parsed["job_id"].as_str(), Some("1234"));
    assert_eq!(parsed["quality"].as_str(), Some("high"));
    assert_eq!(parsed["crop_settings"]["preset"].as_str(), Some("linkedin"));
    assert_eq!(parsed["crop_settings"]["output_width"].as_u64(), Some(400));
    assert_eq!(parsed["crop_settings"]["output_height"].as_u64(), Some(500));
    assert_eq!(
        parsed["crop_settings"]["positioning_mode"].as_str(),
        Some("custom")
    );
    assert_eq!(parsed["generator"].as_str(), Some("face-crop-studio"));
    assert_eq!(
        parsed["generator_version"].as_str(),
        Some(env!("CARGO_PKG_VERSION"))
    );
    let face_confidence = parsed["face_confidence"].as_f64().unwrap();
    assert!((face_confidence - 0.91).abs() < 1e-6);
    assert_eq!(parsed["quality_score"].as_f64(), Some(1234.5));
}
