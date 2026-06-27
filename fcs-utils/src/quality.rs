//! Image quality analysis utilities.
//!
//! The quality subsystem uses Laplacian variance to approximate how much
//! high-frequency detail exists inside a face crop. A high variance indicates
//! crisp edges and well-focused features, while a low variance points to blur
//! or motion smearing. We bucket raw variance values into three coarse bands:
//! `Low` (≤300), `Medium` (300‒1000), and `High` (>1000). Those thresholds
//! were calibrated against the YuNet fixtures so that High roughly aligns with
//! the faces we would confidently ship to customers, Medium covers "usable but
//! soft" captures, and Low highlights frames that should be skipped or
//! re-shot. The helpers in this module expose both the raw variance score and
//! the derived `Quality` so higher layers (CLI/GUI) can implement automation
//! such as auto-selecting the sharpest face or applying filename suffixes.

use image::{DynamicImage, GenericImageView, GrayImage};
use serde::{Deserialize, Serialize};

/// Laplacian variance threshold for high quality images.
pub const QUALITY_HIGH_THRESHOLD: f64 = 1000.0;
/// Laplacian variance threshold for medium quality images.
pub const QUALITY_MEDIUM_THRESHOLD: f64 = 300.0;
/// Maximum dimension for quality analysis (images are downscaled for performance).
const QUALITY_MAX_DIM: u32 = 512;

/// Quality levels derived from Laplacian variance thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Quality {
    Low,
    Medium,
    High,
}

impl Quality {
    /// Map a Laplacian variance score into a `Quality` bucket.
    pub const fn from_variance(v: f64) -> Self {
        if v > QUALITY_HIGH_THRESHOLD {
            Quality::High
        } else if v > QUALITY_MEDIUM_THRESHOLD {
            Quality::Medium
        } else {
            Quality::Low
        }
    }
}

impl std::str::FromStr for Quality {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "low" => Ok(Quality::Low),
            "medium" | "med" => Ok(Quality::Medium),
            "high" => Ok(Quality::High),
            other => Err(format!(
                "unknown quality '{}', expected low/medium/high",
                other
            )),
        }
    }
}

/// A small configuration object used to centralize quality filtering behavior.
#[derive(Debug, Clone)]
pub struct QualityFilter {
    /// Minimum quality required to accept an image.
    pub min_quality: Option<Quality>,
    /// If true, choose the highest available crop automatically (future use).
    pub auto_select: bool,
    /// Skip exporting when no detection meets `Quality::High`.
    pub auto_skip_no_high: bool,
    /// Append quality suffixes to exported filenames.
    pub suffix_enabled: bool,
}

impl QualityFilter {
    /// Simple constructor.
    pub fn new(min_quality: Option<Quality>) -> Self {
        Self {
            min_quality,
            auto_select: false,
            auto_skip_no_high: false,
            suffix_enabled: false,
        }
    }

    /// Return true if the provided detection quality should be skipped.
    pub fn should_skip(&self, q: Quality) -> bool {
        if let Some(min) = self.min_quality {
            return q < min;
        }
        false
    }

    /// When `auto_skip_no_high` is enabled, return true if the best available quality
    /// does not reach `Quality::High`.
    pub fn should_skip_image(&self, best_quality: Option<Quality>) -> bool {
        if self.auto_skip_no_high {
            !matches!(best_quality, Some(Quality::High))
        } else {
            false
        }
    }

    /// Pick the index of the best quality entry from `(Quality, score)` tuples.
    /// `Quality::High` beats `Medium`, which beats `Low`. Ties fall back to the raw score.
    pub fn select_best_index(&self, qualities: &[(Quality, f64)]) -> Option<usize> {
        qualities
            .iter()
            .enumerate()
            .max_by(|(_, (qa, sa)), (_, (qb, sb))| match qa.cmp(qb) {
                std::cmp::Ordering::Equal => {
                    sa.partial_cmp(sb).unwrap_or(std::cmp::Ordering::Equal)
                }
                other => other,
            })
            .map(|(idx, _)| idx)
    }

    /// Return a filename suffix (e.g., `_highq`) when suffixing is enabled.
    pub fn suffix_for(&self, q: Quality) -> Option<&'static str> {
        if !self.suffix_enabled {
            return None;
        }
        match q {
            Quality::High => Some("_highq"),
            Quality::Medium => Some("_medq"),
            Quality::Low => Some("_lowq"),
        }
    }
}

/// Compute the Laplacian variance for an image region. Higher values mean
/// the image is sharper (less blurry).
pub fn laplacian_variance(img: &DynamicImage) -> f64 {
    // Optimization: Downscale large images to speed up variance calculation
    let (w, h) = img.dimensions();
    let img_to_process = if w > QUALITY_MAX_DIM || h > QUALITY_MAX_DIM {
        std::borrow::Cow::Owned(img.resize(
            QUALITY_MAX_DIM,
            QUALITY_MAX_DIM,
            image::imageops::FilterType::Triangle,
        ))
    } else {
        std::borrow::Cow::Borrowed(img)
    };

    let gray: GrayImage = img_to_process.to_luma8();
    let (width, height) = gray.dimensions();

    if width < 3 || height < 3 {
        return 0.0;
    }

    let raw = gray.as_raw();
    let stride = width as usize;

    let mut sum: i64 = 0;
    let mut sum_sq: i64 = 0;

    // Single-pass Laplacian calculation using integer arithmetic
    // Kernel:
    //  0  1  0
    //  1 -4  1
    //  0  1  0

    // Iterate over the inner pixels
    for y in 1..height - 1 {
        let row_offset = (y as usize) * stride;
        for x in 1..width - 1 {
            let idx = row_offset + (x as usize);

            // Direct buffer access (safe because we are within bounds)
            let p_c = raw[idx] as i16;
            let p_u = raw[idx - stride] as i16;
            let p_d = raw[idx + stride] as i16;
            let p_l = raw[idx - 1] as i16;
            let p_r = raw[idx + 1] as i16;

            let lap = (p_u + p_d + p_l + p_r - 4 * p_c) as i64;

            sum += lap;
            sum_sq += lap * lap;
        }
    }

    let count = ((width - 2) * (height - 2)) as f64;
    let mean = (sum as f64) / count;
    ((sum_sq as f64) / count) - (mean * mean)
}

/// Estimate the quality bucket for the image using Laplacian variance.
pub fn estimate_sharpness(img: &DynamicImage) -> (f64, Quality) {
    let v = laplacian_variance(img);
    (v, Quality::from_variance(v))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbaImage;

    #[test]
    fn quality_from_variance_thresholds() {
        assert_eq!(Quality::from_variance(50.0), Quality::Low);
        assert_eq!(Quality::from_variance(300.0), Quality::Low);
        assert_eq!(Quality::from_variance(300.1), Quality::Medium);
        assert_eq!(Quality::from_variance(1000.0), Quality::Medium);
        assert_eq!(Quality::from_variance(1500.0), Quality::High);
    }

    #[test]
    fn variance_of_blank_image_is_low() {
        let img = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            64,
            64,
            image::Rgba([128, 128, 128, 255]),
        ));
        let (v, q) = estimate_sharpness(&img);
        assert!(v >= 0.0);
        assert_eq!(q, Quality::Low);
    }

    #[test]
    fn variance_of_sharp_image_is_higher() {
        let mut img = RgbaImage::from_pixel(64, 64, image::Rgba([128, 128, 128, 255]));
        // draw a high-frequency pattern
        for y in 0..64 {
            for x in 0..64 {
                let v = if (x + y) % 2 == 0 { 255u8 } else { 0u8 };
                img.put_pixel(x, y, image::Rgba([v, v, v, 255]));
            }
        }
        let dynimg = DynamicImage::ImageRgba8(img);
        let (v, q) = estimate_sharpness(&dynimg);
        assert!(v > 0.0);
        assert!(q == Quality::Medium || q == Quality::High);
    }

    #[test]
    fn variance_on_empty_image_is_zero() {
        let img = DynamicImage::ImageRgba8(RgbaImage::new(0, 0));
        assert_eq!(laplacian_variance(&img), 0.0);
    }

    #[test]
    fn quality_filter_respects_min_quality() {
        let filter = QualityFilter::new(Some(Quality::Medium));
        assert!(filter.should_skip(Quality::Low));
        assert!(!filter.should_skip(Quality::Medium));
        assert!(!filter.should_skip(Quality::High));
    }

    #[test]
    fn quality_filter_auto_skip_requires_high() {
        let mut filter = QualityFilter::new(None);
        filter.auto_skip_no_high = true;
        assert!(filter.should_skip_image(None));
        assert!(filter.should_skip_image(Some(Quality::Medium)));
        assert!(!filter.should_skip_image(Some(Quality::High)));
    }

    #[test]
    fn select_best_index_prefers_higher_quality_then_score() {
        let filter = QualityFilter::new(None);
        let samples = [
            (Quality::Medium, 0.4),
            (Quality::High, 0.3),
            (Quality::High, 0.8),
            (Quality::Low, 0.9),
        ];
        assert_eq!(filter.select_best_index(&samples), Some(2));

        let tie_samples = [(Quality::Medium, 0.2), (Quality::Medium, 0.7)];
        assert_eq!(filter.select_best_index(&tie_samples), Some(1));
    }

    #[test]
    fn suffix_for_respects_flag() {
        let mut filter = QualityFilter::new(None);
        assert_eq!(filter.suffix_for(Quality::High), None);

        filter.suffix_enabled = true;
        assert_eq!(filter.suffix_for(Quality::High), Some("_highq"));
        assert_eq!(filter.suffix_for(Quality::Medium), Some("_medq"));
        assert_eq!(filter.suffix_for(Quality::Low), Some("_lowq"));
    }
}

#[cfg(test)]
mod benchmarks {
    use super::*;
    use image::{DynamicImage, RgbaImage};
    use std::time::Instant;

    #[test]
    fn bench_laplacian_variance() {
        // Create a 512x512 image (typical face crop size)
        let img = DynamicImage::ImageRgba8(RgbaImage::new(512, 512));

        // Warmup
        for _ in 0..10 {
            let _ = laplacian_variance(&img);
        }

        let start = Instant::now();
        let iterations = 100;
        for _ in 0..iterations {
            let _ = laplacian_variance(&img);
        }
        let duration = start.elapsed();

        println!(
            "laplacian_variance (512x512) avg time: {:?}",
            duration / iterations
        );
    }
}
