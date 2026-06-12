//! Export orchestration and filesystem writing.

use super::{
    encoders::{encode_avif, encode_bmp, encode_jpeg, encode_png, encode_tiff, encode_webp},
    metadata::{
        build_custom_metadata_payload, inject_jpeg_metadata, inject_png_metadata, inject_webp_exif,
        load_jpeg_exif, load_png_exif_chunks,
    },
    types::{ImageFormatHint, MetadataContext, OutputOptions},
};
use crate::config::MetadataMode;
use anyhow::{Context, Result};
use image::DynamicImage;
use log::debug;
use std::{
    fs,
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

/// Save an image using the provided options and metadata context.
pub fn save_dynamic_image(
    image: &DynamicImage,
    destination: &Path,
    options: &OutputOptions,
    metadata: &MetadataContext<'_>,
) -> Result<()> {
    if let Some(parent) = destination.parent().filter(|p| !p.exists()) {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let format = determine_format(destination, options);
    debug!(
        "Saving crop to {} using {:?} format",
        destination.display(),
        format
    );

    let mut encoded = match format {
        ImageFormatHint::Png => encode_png(image, options.png_compression)?,
        ImageFormatHint::Jpeg => encode_jpeg(image, options.jpeg_quality)?,
        ImageFormatHint::Webp => encode_webp(image)?,
        ImageFormatHint::Tiff => encode_tiff(image)?,
        ImageFormatHint::Bmp => encode_bmp(image)?,
        ImageFormatHint::Avif => encode_avif(image)?,
    };

    // Prepare metadata payload if applicable.
    let custom_payload = build_custom_metadata_payload(&options.metadata, metadata)?;

    match format {
        ImageFormatHint::Png => {
            if !matches!(options.metadata.mode, MetadataMode::Strip) {
                let mut exif_chunks = Vec::new();
                if matches!(options.metadata.mode, MetadataMode::Preserve) {
                    exif_chunks = load_png_exif_chunks(metadata.source_path);
                }
                encoded = inject_png_metadata(encoded, &exif_chunks, custom_payload.as_deref());
            }
        }
        ImageFormatHint::Jpeg => {
            encoded = inject_jpeg_metadata(
                encoded,
                if matches!(options.metadata.mode, MetadataMode::Preserve) {
                    load_jpeg_exif(metadata.source_path)
                } else {
                    None
                },
                if matches!(options.metadata.mode, MetadataMode::Strip) {
                    None
                } else {
                    custom_payload.as_deref()
                },
            );
        }
        ImageFormatHint::Webp => {
            if matches!(options.metadata.mode, MetadataMode::Preserve) {
                if let Some(exif) = load_jpeg_exif(metadata.source_path) {
                    encoded = inject_webp_exif(encoded, Some(exif), custom_payload.as_deref());
                } else {
                    encoded = inject_webp_exif(encoded, None, custom_payload.as_deref());
                }
            } else if matches!(options.metadata.mode, MetadataMode::Strip) {
                // Nothing extra to embed.
            } else {
                encoded = inject_webp_exif(encoded, None, custom_payload.as_deref());
            }
        }
        ImageFormatHint::Tiff | ImageFormatHint::Bmp | ImageFormatHint::Avif => {
            // Metadata injection not yet implemented for these formats
        }
    }

    write_bytes(destination, &encoded)?;
    Ok(())
}

pub(super) fn determine_format(path: &Path, options: &OutputOptions) -> ImageFormatHint {
    if !options.auto_detect {
        return options.format.unwrap_or_default();
    }

    if let Some(fmt) = path
        .extension()
        .and_then(|e| e.to_str())
        .and_then(ImageFormatHint::from_extension)
    {
        fmt
    } else {
        options.format.unwrap_or_default()
    }
}

/// Append a suffix to a filename, preserving the existing extension.
pub fn append_suffix_to_filename(name: &str, suffix: &str) -> String {
    if suffix.is_empty() {
        return name.to_string();
    }
    if let Some(idx) = name.rfind('.') {
        let (base, ext) = name.split_at(idx);
        format!("{base}{suffix}{ext}")
    } else {
        format!("{name}{suffix}")
    }
}

fn write_bytes(path: &Path, bytes: &[u8]) -> Result<()> {
    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    writer
        .write_all(bytes)
        .with_context(|| format!("failed to write {}", path.display()))?;
    writer.flush().ok();
    Ok(())
}
