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

// ---------------------------------------------------------------------------
// Metadata builders and parsers.
//
// The tests above drive the injectors through real encoded images, which
// covers the happy path but never reaches the keyword validation, the
// marker-walking branches, or the orientation clearing.

// `output` re-exports only what it calls; these are `pub(super)` in the
// metadata module and reachable from here as a descendant of `output`.
use super::metadata::{build_png_text_chunk, inject_webp_exif};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};

/// Pull the XMP APP1 segment out of an injected JPEG.
///
/// `build_jpeg_xmp_segment` is private, so it is exercised through its only
/// caller rather than by widening its visibility for the tests.
fn xmp_segment_of(encoded: Vec<u8>, json: &str) -> Option<Vec<u8>> {
    let injected = inject_jpeg_metadata(encoded.clone(), None, Some(json));
    if injected.len() == encoded.len() {
        return None; // nothing was inserted
    }
    let declared = u16::from_be_bytes(injected[4..6].try_into().unwrap()) as usize;
    Some(injected[2..2 + declared + 2].to_vec())
}

/// Assemble a JPEG from a sequence of `(marker, payload)` segments, writing
/// the two-byte length prefix the format requires for each.
fn make_jpeg(segments: &[(u8, Vec<u8>)]) -> Vec<u8> {
    let mut out = vec![0xFF, 0xD8];
    for (marker, payload) in segments {
        out.push(0xFF);
        out.push(*marker);
        out.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
        out.extend_from_slice(payload);
    }
    out.extend_from_slice(&[0xFF, 0xD9]);
    out
}

fn write_bytes(dir: &tempfile::TempDir, name: &str, bytes: &[u8]) -> PathBuf {
    let path = dir.path().join(name);
    fs::write(&path, bytes).unwrap();
    path
}

#[test]
fn build_png_text_chunk_lays_out_a_valid_text_chunk() {
    let chunk = build_png_text_chunk("Ab", "cd").expect("valid keyword");
    // length | "tEXt" | keyword NUL value | crc, so 4 + 4 + 5 + 4.
    assert_eq!(chunk.len(), 17);
    assert_eq!(&chunk[0..4], &5u32.to_be_bytes());
    assert_eq!(&chunk[4..8], b"tEXt");
    assert_eq!(&chunk[8..13], b"Ab\0cd");

    // The CRC covers the chunk type as well as the data, not just one.
    let mut hasher = Crc32::new();
    hasher.update(&chunk[4..13]);
    assert_eq!(&chunk[13..17], &hasher.finalize().to_be_bytes());
}

#[test]
fn build_png_text_chunk_rejects_invalid_keywords() {
    assert!(build_png_text_chunk("", "v").is_none(), "empty");
    assert!(
        build_png_text_chunk(&"k".repeat(80), "v").is_none(),
        "80 characters exceeds the PNG limit"
    );
    assert!(
        build_png_text_chunk(&"k".repeat(79), "v").is_some(),
        "79 characters is the maximum allowed"
    );
    assert!(build_png_text_chunk("a\nb", "v").is_none(), "newline");
    assert!(
        build_png_text_chunk("a\rb", "v").is_none(),
        "carriage return"
    );
    assert!(build_png_text_chunk("a\0b", "v").is_none(), "nul");
    assert!(
        build_png_text_chunk("caf\u{e9}", "v").is_none(),
        "non-ascii"
    );
}

#[test]
fn build_png_text_chunk_allows_an_empty_value() {
    let chunk = build_png_text_chunk("K", "").expect("empty values are legal");
    assert_eq!(&chunk[0..4], &2u32.to_be_bytes());
    assert_eq!(&chunk[8..10], b"K\0");
}

#[test]
fn xmp_segment_wraps_base64_json_in_an_app1_marker() {
    let json = "{\"a\":1}";
    let jpeg = encode_jpeg(&sample_image(), 90).unwrap();
    let segment = xmp_segment_of(jpeg, json).expect("small payload is embedded");

    assert_eq!(&segment[..2], &[0xFF, 0xE1]);
    let declared = u16::from_be_bytes(segment[2..4].try_into().unwrap()) as usize;
    // The declared length counts itself but not the two marker bytes.
    assert_eq!(declared, segment.len() - 2);
    assert!(segment[4..].starts_with(b"http://ns.adobe.com/xap/1.0/\0"));

    let text = String::from_utf8_lossy(&segment[4..]);
    assert!(
        text.contains(&BASE64.encode(json.as_bytes())),
        "the JSON should be embedded base64-encoded"
    );
    assert!(text.contains("<iron:Metadata>"));
}

#[test]
fn xmp_segment_is_dropped_when_it_exceeds_the_segment_limit() {
    // A JPEG segment length is a u16, so a payload that cannot be represented
    // has to be dropped rather than silently truncated. The rest of the file
    // must still come through untouched.
    let jpeg = encode_jpeg(&sample_image(), 90).unwrap();
    let huge = "x".repeat(60_000);
    assert_eq!(
        inject_jpeg_metadata(jpeg.clone(), None, Some(&huge)),
        jpeg,
        "an oversized payload leaves the image unchanged"
    );
}

#[test]
fn load_jpeg_exif_skips_segments_before_the_exif_one() {
    let dir = tempdir().unwrap();
    // A JFIF APP0 and a non-Exif APP1 both precede the real EXIF segment, so
    // the scanner has to step over each rather than stopping or misreading.
    let mut exif_payload = b"Exif\0\0".to_vec();
    exif_payload.extend_from_slice(b"tiff-bytes");
    let jpeg = make_jpeg(&[
        (0xE0, b"JFIF\0stuff".to_vec()),
        (0xE1, b"http://ns.adobe.com/xap/1.0/\0xmp".to_vec()),
        (0xE1, exif_payload),
    ]);
    let path = write_bytes(&dir, "multi.jpg", &jpeg);

    let found = load_jpeg_exif(Some(&path)).expect("should find the Exif APP1");
    assert_eq!(&found[..2], &[0xFF, 0xE1]);
    assert!(found.ends_with(b"tiff-bytes"));
}

#[test]
fn load_jpeg_exif_stops_at_the_start_of_scan_data() {
    let dir = tempdir().unwrap();
    // Bytes after SOS are entropy-coded image data, not segments, so a stray
    // "Exif" pattern in there must not be picked up.
    let mut jpeg = vec![0xFF, 0xD8, 0xFF, 0xDA, 0x00, 0x02];
    jpeg.extend_from_slice(b"\xFF\xE1\x00\x0cExif\0\0junk");
    let path = write_bytes(&dir, "sos.jpg", &jpeg);

    assert!(load_jpeg_exif(Some(&path)).is_none());
}

#[test]
fn load_jpeg_exif_rejects_files_without_the_soi_marker() {
    let dir = tempdir().unwrap();
    let path = write_bytes(&dir, "bad-magic.jpg", b"\x00\x00not really a jpeg");
    assert!(load_jpeg_exif(Some(&path)).is_none());

    let short = write_bytes(&dir, "short.jpg", b"\xFF\xD8");
    assert!(load_jpeg_exif(Some(&short)).is_none());
}

#[test]
fn load_jpeg_exif_stops_when_a_marker_byte_is_missing() {
    let dir = tempdir().unwrap();
    // Without a 0xFF lead-in the stream is no longer on a marker boundary;
    // scanning must stop rather than try to resynchronise.
    let mut jpeg = vec![0xFF, 0xD8, 0x12, 0x34, 0x56, 0x78];
    jpeg.extend_from_slice(b"\xFF\xE1\x00\x0cExif\0\0junk");
    let path = write_bytes(&dir, "desync.jpg", &jpeg);

    assert!(load_jpeg_exif(Some(&path)).is_none());
}

#[test]
fn load_jpeg_exif_clears_the_orientation_tag() {
    let dir = tempdir().unwrap();
    // Minimal little-endian TIFF carrying a single Orientation entry set to 6.
    let mut tiff = b"II\x2a\x00".to_vec();
    tiff.extend_from_slice(&8u32.to_le_bytes()); // IFD0 offset
    tiff.extend_from_slice(&1u16.to_le_bytes()); // entry count
    tiff.extend_from_slice(&0x0112u16.to_le_bytes()); // Orientation tag
    tiff.extend_from_slice(&3u16.to_le_bytes()); // type SHORT
    tiff.extend_from_slice(&1u32.to_le_bytes()); // count
    tiff.extend_from_slice(&6u16.to_le_bytes()); // value: rotate 90
    tiff.extend_from_slice(&0u16.to_le_bytes()); // value padding
    tiff.extend_from_slice(&0u32.to_le_bytes()); // no next IFD

    let mut payload = b"Exif\0\0".to_vec();
    payload.extend_from_slice(&tiff);
    let jpeg = make_jpeg(&[(0xE1, payload.clone())]);
    let path = write_bytes(&dir, "oriented.jpg", &jpeg);

    let found = load_jpeg_exif(Some(&path)).expect("segment present");
    // The copy that comes back has the orientation stripped, so it cannot be
    // byte-identical to what went in.
    assert_ne!(
        &found[4..],
        payload.as_slice(),
        "orientation should have been cleared"
    );
}

#[test]
fn load_jpeg_exif_leaves_a_malformed_exif_header_untouched() {
    let dir = tempdir().unwrap();
    // "Exif" is present so the segment is accepted, but the two NUL bytes
    // that must follow are not — the orientation pass has to bail out and
    // hand the bytes back verbatim.
    let mut payload = b"Exif??".to_vec();
    payload.extend_from_slice(b"\x2a\x00\x08\x00\x00\x00");
    let jpeg = make_jpeg(&[(0xE1, payload.clone())]);
    let path = write_bytes(&dir, "malformed.jpg", &jpeg);

    let found = load_jpeg_exif(Some(&path)).expect("segment present");
    assert_eq!(&found[4..], payload.as_slice(), "returned unchanged");
}

#[test]
fn load_png_exif_chunks_collects_every_exif_chunk_and_stops_at_iend() {
    let dir = tempdir().unwrap();
    let first = make_png_chunk(b"eXIf", b"one");
    let second = make_png_chunk(b"eXIf", b"two");
    let after_end = make_png_chunk(b"eXIf", b"ignored");

    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    png.extend_from_slice(&make_png_chunk(b"IHDR", &[0u8; 13]));
    png.extend_from_slice(&first);
    png.extend_from_slice(&make_png_chunk(b"IDAT", b"pixels"));
    png.extend_from_slice(&second);
    png.extend_from_slice(&make_png_chunk(b"IEND", b""));
    png.extend_from_slice(&after_end);
    let path = write_bytes(&dir, "multi.png", &png);

    assert_eq!(load_png_exif_chunks(Some(&path)), vec![first, second]);
}

#[test]
fn load_png_exif_chunks_rejects_a_bad_signature_or_truncation() {
    let dir = tempdir().unwrap();

    let wrong = write_bytes(&dir, "wrong.png", b"\x89PNGbroken-signature");
    assert!(load_png_exif_chunks(Some(&wrong)).is_empty());

    // A chunk header claiming more data than the file holds must not panic.
    let mut truncated = b"\x89PNG\r\n\x1a\n".to_vec();
    truncated.extend_from_slice(&9999u32.to_be_bytes());
    truncated.extend_from_slice(b"eXIf");
    truncated.extend_from_slice(b"short");
    let path = write_bytes(&dir, "truncated.png", &truncated);
    assert!(load_png_exif_chunks(Some(&path)).is_empty());
}

#[test]
fn injectors_pass_through_when_there_is_nothing_to_add() {
    let png = encode_png(&sample_image(), PngCompression::Default).unwrap();
    assert_eq!(inject_png_metadata(png.clone(), &[], None), png);

    let jpeg = encode_jpeg(&sample_image(), 90).unwrap();
    assert_eq!(inject_jpeg_metadata(jpeg.clone(), None, None), jpeg);

    // Too short to hold a header at all.
    assert_eq!(
        inject_png_metadata(vec![1, 2, 3], &[], Some("{}")),
        vec![1, 2, 3]
    );
    assert_eq!(
        inject_jpeg_metadata(vec![0xFF], None, Some("{}")),
        vec![0xFF]
    );
    // Long enough, but not a JPEG.
    assert_eq!(
        inject_jpeg_metadata(vec![0x00, 0x00, 0x01], None, Some("{}")),
        vec![0x00, 0x00, 0x01]
    );
}

#[test]
fn inject_webp_exif_returns_the_input_unchanged() {
    // WebP metadata is not implemented; the contract is that the encoded
    // bytes come back untouched whether or not metadata was supplied.
    let encoded = vec![b'R', b'I', b'F', b'F', 1, 2, 3, 4];
    assert_eq!(inject_webp_exif(encoded.clone(), None, None), encoded);
    assert_eq!(
        inject_webp_exif(encoded.clone(), Some(vec![0xFF, 0xE1]), None),
        encoded
    );
    assert_eq!(
        inject_webp_exif(encoded.clone(), None, Some("{\"a\":1}")),
        encoded
    );
}
