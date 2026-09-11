//! Output format and metadata option types.

use crate::{
    config::{CropSettings, MetadataSettings},
    quality::Quality,
};

use image::codecs::png::CompressionType;
use log::warn;
use std::path::Path;

/// Canonical image formats supported by the exporter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageFormatHint {
    /// Lossless PNG with alpha support; the default output format.
    #[default]
    Png,
    /// Lossy JPEG; output is converted to RGB and alpha is discarded.
    #[serde(alias = "jpg")]
    Jpeg,
    /// Lossless WebP with alpha support.
    Webp,
    /// TIFF output; custom metadata injection is not supported.
    #[serde(alias = "tif")]
    Tiff,
    /// BMP output; custom metadata injection is not supported.
    Bmp,
    /// AVIF output using the exporter's fixed quality 80 and speed 4.
    Avif,
}

impl ImageFormatHint {
    /// Determine format from a filesystem extension.
    pub fn from_extension(ext: &str) -> Option<Self> {
        ext.parse().ok()
    }
}

impl std::str::FromStr for ImageFormatHint {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "png" => Ok(Self::Png),
            "jpg" | "jpeg" => Ok(Self::Jpeg),
            "webp" => Ok(Self::Webp),
            "tif" | "tiff" => Ok(Self::Tiff),
            "bmp" => Ok(Self::Bmp),
            "avif" => Ok(Self::Avif),
            other => Err(format!("unknown image format '{other}'")),
        }
    }
}

/// Simplified PNG compression strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PngCompression {
    /// Favor encoding speed over compressed size; pixel data remains lossless.
    Fast,
    /// Use the image encoder's default lossless compression strategy.
    #[default]
    Default,
    /// Favor smaller files at the cost of encoding time; pixel data remains lossless.
    Best,
}

impl PngCompression {
    /// Parse compression string/level into a compression strategy.
    pub fn parse(input: &str) -> Self {
        let normalized = input.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "fast" => Self::Fast,
            "best" => Self::Best,
            "default" => Self::Default,
            _ => {
                if let Ok(level) = normalized.parse::<u8>() {
                    match level {
                        0..=3 => Self::Fast,
                        7..=9 => Self::Best,
                        _ => Self::Default,
                    }
                } else {
                    warn!(
                        "Unknown PNG compression '{}', falling back to default strategy",
                        input
                    );
                    Self::Default
                }
            }
        }
    }

    pub(super) fn into_image(self) -> CompressionType {
        match self {
            Self::Fast => CompressionType::Fast,
            Self::Default => CompressionType::Default,
            Self::Best => CompressionType::Best,
        }
    }
}

// Accept either a keyword ("fast"/"default"/"best") or a numeric string ("0".."9"),
// preserving back-compat with hand-written config files.
impl<'de> serde::Deserialize<'de> for PngCompression {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        Ok(Self::parse(&raw))
    }
}

/// Immutable configuration derived from the user's crop settings.
#[derive(Debug, Clone)]
pub struct OutputOptions {
    /// Explicit format, or PNG when unset; also the fallback for an unknown extension.
    pub format: Option<ImageFormatHint>,
    /// Let a recognized destination extension override `format` when true.
    pub auto_detect: bool,
    /// JPEG quality in 1..=100; [`Self::from_crop_settings`] clamps it to this range.
    pub jpeg_quality: u8,
    /// Lossless PNG compression strategy.
    pub png_compression: PngCompression,
    /// Stored WebP quality preference (0..=100); currently unused by the lossless encoder.
    pub webp_quality: u8,
    /// Source EXIF and custom metadata policy; currently supported for PNG and JPEG output.
    pub metadata: MetadataSettings,
}

impl OutputOptions {
    /// Build `OutputOptions` from persistent crop settings.
    pub fn from_crop_settings(settings: &CropSettings) -> Self {
        Self {
            format: Some(settings.output_format),
            auto_detect: settings.auto_detect_format,
            jpeg_quality: settings.jpeg_quality.clamp(1, 100),
            png_compression: settings.png_compression,
            webp_quality: settings.webp_quality.min(100),
            metadata: settings.metadata.clone(),
        }
    }
}

/// Runtime metadata passed in from the caller when exporting a single crop.
#[derive(Debug, Clone, Default)]
pub struct MetadataContext<'a> {
    /// Original file from which supported EXIF metadata may be copied.
    pub source_path: Option<&'a Path>,
    /// Crop settings to embed when `include_crop_settings` is enabled.
    pub crop_settings: Option<&'a CropSettings>,
    /// Detection confidence to embed when `include_quality_metrics` is enabled.
    pub detection_score: Option<f32>,
    /// Sharpness category to embed when `include_quality_metrics` is enabled.
    pub quality: Option<Quality>,
    /// Raw Laplacian variance to embed when `include_quality_metrics` is enabled.
    pub quality_score: Option<f64>,
}
