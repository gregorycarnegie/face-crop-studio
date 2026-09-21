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
        auto_detect_format: false,
        ..CropSettings::default()
    };
    settings.metadata.mode = MetadataMode::Custom;

    let options = OutputOptions::from_crop_settings(&settings);

    assert_eq!(options.format, Some(ImageFormatHint::Jpeg));
    assert!(!options.auto_detect);
    assert_eq!(options.jpeg_quality, 1);
    assert_eq!(options.png_compression, PngCompression::Best);
    assert_eq!(options.metadata.mode, MetadataMode::Custom);
}

#[test]
fn determine_format_prefers_extension_when_auto_detect_is_enabled() {
    let options = OutputOptions {
        format: Some(ImageFormatHint::Png),
        auto_detect: true,
        jpeg_quality: 90,
        png_compression: PngCompression::Default,
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

/// Both loaders are fed whatever the user selected, so the header checks are a
/// trust boundary: they run before any indexing and must reject a file too short
/// to index into. Written as a table over every truncation length because the
/// interesting failures are all off-by-one — losing the short-circuit in
/// `len < 8 || &bytes[..8] != sig` turns each of these into a slice-out-of-bounds
/// panic rather than an empty result.
#[test]
fn metadata_loaders_reject_truncated_headers_without_panicking() {
    let dir = tempdir().unwrap();
    let png_signature = b"\x89PNG\r\n\x1a\n";

    // Every prefix of a PNG signature, including empty, plus one full signature
    // with no chunks after it.
    for len in 0..=png_signature.len() {
        let path = write_bytes(&dir, &format!("trunc_{len}.png"), &png_signature[..len]);
        assert!(
            load_png_exif_chunks(Some(&path)).is_empty(),
            "PNG truncated to {len} byte(s) must yield no chunks"
        );
    }

    // A correct signature followed by a partial chunk header: the loop guard has
    // to stop before reading a length field that isn't fully present.
    for extra in 1..8 {
        let mut bytes = png_signature.to_vec();
        bytes.extend(std::iter::repeat_n(0u8, extra));
        let path = write_bytes(&dir, &format!("partial_chunk_{extra}.png"), &bytes);
        assert!(load_png_exif_chunks(Some(&path)).is_empty());
    }

    for len in 0..=4 {
        let bytes = vec![0xFFu8, 0xD8, 0xFF, 0xE1][..len].to_vec();
        let path = write_bytes(&dir, &format!("trunc_{len}.jpg"), &bytes);
        assert!(
            load_jpeg_exif(Some(&path)).is_none(),
            "JPEG truncated to {len} byte(s) must yield no EXIF"
        );
    }
}

/// A JPEG segment's length field counts its own two bytes, so a value below 2 is
/// malformed. The subtraction used to happen before any check, which underflowed
/// on a crafted file: a panic in debug builds and a wrapped offset in release.
#[test]
fn load_jpeg_exif_survives_undersized_segment_length() {
    let dir = tempdir().unwrap();

    for bad_length in [0u16, 1] {
        let mut bytes = vec![0xFF, 0xD8, 0xFF, 0xE1];
        bytes.extend_from_slice(&bad_length.to_be_bytes());
        bytes.extend_from_slice(b"Exif\0\0trailing");
        let path = write_bytes(&dir, &format!("len_{bad_length}.jpg"), &bytes);

        assert!(
            load_jpeg_exif(Some(&path)).is_none(),
            "segment length {bad_length} is malformed and must be rejected"
        );
    }

    // Length exactly 2 means an empty payload: valid arithmetic, still no EXIF.
    let mut bytes = vec![0xFF, 0xD8, 0xFF, 0xE1, 0x00, 0x02];
    bytes.extend_from_slice(&[0xFF, 0xD9]);
    let path = write_bytes(&dir, "len_2.jpg", &bytes);
    assert!(load_jpeg_exif(Some(&path)).is_none());
}

/// The PNG chunk walk breaks when `data_end + 4 > len`. A final chunk that ends
/// exactly at EOF sits on that boundary, so a `>=` there would silently drop the
/// last chunk in the file — which is precisely where an appended eXIf lands.
#[test]
fn load_png_exif_chunks_keeps_a_chunk_that_ends_exactly_at_eof() {
    let dir = tempdir().unwrap();
    let exif_chunk = make_png_chunk(b"eXIf", b"ends-at-eof");

    let mut bytes = b"\x89PNG\r\n\x1a\n".to_vec();
    bytes.extend_from_slice(&make_png_chunk(b"IHDR", &[0u8; 13]));
    bytes.extend_from_slice(&exif_chunk);
    // Deliberately no IEND: the eXIf chunk is the final byte of the file.
    let path = write_bytes(&dir, "exif_at_eof.png", &bytes);

    assert_eq!(
        load_png_exif_chunks(Some(&path)),
        vec![exif_chunk],
        "a chunk ending exactly at EOF must still be collected"
    );
}

/// `load_png_exif_chunks` stops at IEND. A file with trailing bytes after IEND
/// must not have them parsed as further chunks.
#[test]
fn load_png_exif_chunks_stops_at_iend_and_ignores_trailing_bytes() {
    let dir = tempdir().unwrap();

    let mut bytes = b"\x89PNG\r\n\x1a\n".to_vec();
    bytes.extend_from_slice(&make_png_chunk(b"IHDR", &[0u8; 13]));
    bytes.extend_from_slice(&make_png_chunk(b"IEND", b""));
    // An eXIf chunk hidden after IEND is not part of the image.
    bytes.extend_from_slice(&make_png_chunk(b"eXIf", b"after-iend"));
    let path = write_bytes(&dir, "trailing.png", &bytes);

    assert!(
        load_png_exif_chunks(Some(&path)).is_empty(),
        "chunks after IEND must be ignored"
    );
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
use super::metadata::build_png_text_chunk;
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
fn png_compression_level_changes_the_encoded_size() {
    // Ramps compress well, so the strategies must actually differ in size.
    let image = DynamicImage::ImageRgb8(image::RgbImage::from_fn(96, 64, |x, y| {
        image::Rgb([(x * 2) as u8, (y * 3) as u8, (x + y) as u8])
    }));
    let fast = encode_png(&image, PngCompression::Fast).unwrap();
    let best = encode_png(&image, PngCompression::Best).unwrap();
    assert!(
        best.len() < fast.len(),
        "best {} bytes, fast {} bytes",
        best.len(),
        fast.len()
    );
}

#[test]
fn lossless_encoders_round_trip_the_pixels() {
    // Opaque only: BMP and WebP round trips are what is being checked here, not
    // each codec's alpha handling.
    let img = DynamicImage::ImageRgba8(RgbaImage::from_fn(3, 2, |x, y| {
        Rgba([
            (20 + x * 40) as u8,
            (60 + y * 70) as u8,
            200 - (x * 15) as u8,
            255,
        ])
    }));

    let encoded: [(&str, Vec<u8>); 4] = [
        ("bmp", encode_bmp(&img).expect("encode bmp")),
        ("tiff", encode_tiff(&img).expect("encode tiff")),
        ("webp", encode_webp(&img).expect("encode webp")),
        (
            "png",
            encode_png(&img, PngCompression::Default).expect("encode png"),
        ),
    ];

    for (name, bytes) in encoded {
        let decoded = image::load_from_memory(&bytes)
            .unwrap_or_else(|err| panic!("{name} output should decode: {err}"));
        assert_eq!(decoded.to_rgba8(), img.to_rgba8(), "{name} round trip");
    }
}

#[test]
fn load_jpeg_exif_needs_both_soi_bytes_to_match() {
    let dir = tempdir().unwrap();
    let mut payload = b"Exif\0\0".to_vec();
    payload.extend_from_slice(b"tiff-ish");
    let jpeg = make_jpeg(&[(0xE1, payload)]);

    let good = write_bytes(&dir, "soi.jpg", &jpeg);
    assert!(
        load_jpeg_exif(Some(&good)).is_some(),
        "the fixture itself carries EXIF"
    );

    // One correct SOI byte is not enough: 0xFF followed by anything other than
    // 0xD8 is not a JPEG, and parsing it as one reads segments out of noise.
    let mut half = jpeg.clone();
    half[1] = 0x00;
    let half_path = write_bytes(&dir, "half-soi.jpg", &half);
    assert!(load_jpeg_exif(Some(&half_path)).is_none());

    let mut other_half = jpeg;
    other_half[0] = 0x00;
    let other_path = write_bytes(&dir, "other-half-soi.jpg", &other_half);
    assert!(load_jpeg_exif(Some(&other_path)).is_none());
}

#[test]
fn load_png_exif_chunks_skips_a_chunk_whose_crc_is_missing() {
    let dir = tempdir().unwrap();
    // A chunk whose declared length runs past the end of the file once its
    // four CRC bytes are accounted for: the scan must stop rather than slice
    // past the buffer.
    let chunk = make_png_chunk(b"eXIf", b"Exif\0\0payload");
    let mut bytes = b"\x89PNG\r\n\x1a\n".to_vec();
    bytes.extend_from_slice(&chunk[..chunk.len() - 4]);
    let path = write_bytes(&dir, "truncated-crc.png", &bytes);

    assert!(load_png_exif_chunks(Some(&path)).is_empty());
}

#[test]
fn load_jpeg_exif_steps_over_an_empty_segment() {
    // A declared length of exactly 2 is a segment with no payload: legal, and
    // the scan has to step past it to reach the EXIF segment behind it.
    let dir = tempdir().unwrap();
    let mut exif = b"Exif\0\0".to_vec();
    exif.extend_from_slice(b"tiff-ish");
    let jpeg = make_jpeg(&[(0xE0, Vec::new()), (0xE1, exif)]);
    let path = write_bytes(&dir, "empty-segment.jpg", &jpeg);

    assert!(load_jpeg_exif(Some(&path)).is_some());
}

#[test]
fn load_jpeg_exif_takes_a_segment_ending_at_eof_but_not_one_past_it() {
    let dir = tempdir().unwrap();
    let payload = b"Exif\0\0tiff";
    let mut jpeg = vec![0xFF, 0xD8, 0xFF, 0xE1];
    jpeg.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
    jpeg.extend_from_slice(payload);

    // The segment's last byte is the file's last byte, which is in bounds.
    let exact = write_bytes(&dir, "exif-at-eof.jpg", &jpeg);
    assert!(load_jpeg_exif(Some(&exact)).is_some());

    // Declaring four more bytes than the file holds must be refused rather
    // than sliced out of the buffer.
    let mut overrun = jpeg.clone();
    overrun[4..6].copy_from_slice(&((payload.len() + 6) as u16).to_be_bytes());
    let past = write_bytes(&dir, "exif-past-eof.jpg", &overrun);
    assert!(load_jpeg_exif(Some(&past)).is_none());
}

/// Signature plus an IHDR chunk carrying the 13 bytes the format requires.
fn minimal_png_header() -> Vec<u8> {
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    png.extend_from_slice(&make_png_chunk(b"IHDR", &[0u8; 13]));
    png
}

#[test]
fn inject_png_metadata_needs_a_whole_ihdr_chunk_to_insert_after() {
    let chunk = make_png_chunk(b"eXIf", b"Exif\0\0payload");
    let chunks = [chunk.clone()];

    // Signature only: there is no IHDR to insert behind, so nothing is added
    // and nothing is read past the end either.
    let signature = b"\x89PNG\r\n\x1a\n".to_vec();
    assert_eq!(
        inject_png_metadata(signature.clone(), &chunks, None),
        signature
    );

    // An IHDR whose declared length runs past the end of the buffer.
    let mut truncated = minimal_png_header();
    truncated.truncate(20);
    assert_eq!(
        inject_png_metadata(truncated.clone(), &chunks, None),
        truncated
    );
}

#[test]
fn inject_png_metadata_inserts_after_an_ihdr_that_ends_the_file() {
    // The IHDR chunk finishes exactly at EOF: still a complete chunk, so the
    // new chunks go straight after it.
    let header = minimal_png_header();
    let chunk = make_png_chunk(b"eXIf", b"Exif\0\0payload");

    let injected = inject_png_metadata(header.clone(), std::slice::from_ref(&chunk), None);
    assert_eq!(
        injected.len(),
        header.len() + chunk.len(),
        "the chunk should have been added"
    );
    assert_eq!(&injected[..header.len()], &header[..]);
    assert_eq!(&injected[header.len()..], &chunk[..]);
}

#[test]
fn inject_jpeg_metadata_needs_both_soi_bytes_to_match() {
    // 0xFF alone is not an SOI marker, so the file is left alone.
    let mut half_soi = vec![0xFF, 0x00, 0x11, 0x22];
    let untouched = inject_jpeg_metadata(half_soi.clone(), None, Some("{\"a\":1}"));
    assert_eq!(untouched, half_soi);

    half_soi[0] = 0x00;
    half_soi[1] = 0xD8;
    assert_eq!(
        inject_jpeg_metadata(half_soi.clone(), None, Some("{\"a\":1}")),
        half_soi
    );

    // A bare SOI is a valid, if empty, place to inject.
    let bare = vec![0xFF, 0xD8];
    let injected = inject_jpeg_metadata(bare.clone(), None, Some("{\"a\":1}"));
    assert!(
        injected.len() > bare.len(),
        "metadata should be inserted after the SOI"
    );
    assert_eq!(&injected[..2], &[0xFF, 0xD8]);
}

#[test]
fn xmp_segment_length_prefix_counts_its_own_two_bytes() {
    // The declared length covers the prefix and payload but not the marker, so the original EOI
    // must follow straight after. Checked against the untouched bytes: xmp_segment_of slices by
    // the declared length, so a wrong prefix would still look self-consistent through it.
    let injected = inject_jpeg_metadata(vec![0xFF, 0xD8, 0xFF, 0xD9], None, Some("{\"k\":\"v\"}"));
    let declared = u16::from_be_bytes(injected[4..6].try_into().unwrap()) as usize;
    assert_eq!(&injected[2 + 2 + declared..], &[0xFF, 0xD9]);
}

#[test]
fn xmp_segment_fits_exactly_at_the_u16_limit_and_not_one_block_past_it() {
    // 48 882 JSON bytes base64 to 65 176 chars, making the declared length exactly u16::MAX.
    // If the XMP template changes these numbers move; the assert_eq names the drift.
    let at_limit =
        xmp_segment_of(vec![0xFF, 0xD8], &"x".repeat(48_882)).expect("exactly u16::MAX fits");
    assert_eq!(
        u16::from_be_bytes(at_limit[2..4].try_into().unwrap()),
        u16::MAX
    );
    // One more byte adds a whole base64 block, four chars over.
    assert!(xmp_segment_of(vec![0xFF, 0xD8], &"x".repeat(48_883)).is_none());
}
