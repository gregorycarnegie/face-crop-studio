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
    #[default]
    Png,
    #[serde(alias = "jpg")]
    Jpeg,
    Webp,
    #[serde(alias = "tif")]
    Tiff,
    Bmp,
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
    Fast,
    #[default]
    Default,
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
    pub format: Option<ImageFormatHint>,
    pub auto_detect: bool,
    pub jpeg_quality: u8,
    pub png_compression: PngCompression,
    pub webp_quality: u8,
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
    pub source_path: Option<&'a Path>,
    pub crop_settings: Option<&'a CropSettings>,
    pub detection_score: Option<f32>,
    pub quality: Option<Quality>,
    pub quality_score: Option<f64>,
}
