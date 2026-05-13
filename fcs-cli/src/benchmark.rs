//! Preprocessing benchmarking functionality.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use fcs_core::{CpuPreprocessor, PreprocessConfig, Preprocessor, WgpuPreprocessor};
use fcs_utils::{GpuContext, load_image};
use image::DynamicImage;
use log::{info, warn};
use serde::Serialize;

use crate::input::ProcessingItem;

#[derive(Serialize)]
pub struct PreprocessBenchmarkSummary {
    label: String,
    samples: usize,
    iterations_per_sample: usize,
    total_ms: f64,
    avg_ms: f64,
    min_ms: f64,
    max_ms: f64,
}

impl PreprocessBenchmarkSummary {
    pub fn with_label(mut self, label: &str) -> Self {
        self.label = label.to_string();
        self
    }
}

pub fn run_preprocess_benchmark(
    items: &[ProcessingItem],
    config: &PreprocessConfig,
    gpu_context: Option<&Arc<GpuContext>>,
) -> Result<()> {
    anyhow::ensure!(
        !items.is_empty(),
        "preprocess benchmark requires at least one input image"
    );

    info!(
        "Running preprocessing benchmark across {} image(s)...",
        items.len()
    );

    let images = load_benchmark_images(items)?;
    let iterations = 3;

    let cpu = CpuPreprocessor;
    let cpu_summary =
        benchmark_preprocessor("cpu", &cpu, &images, config, iterations)?.with_label("cpu");
    info!(
        "preprocess_benchmark {}",
        serde_json::to_string(&cpu_summary)?
    );

    if let Some(ctx) = gpu_context {
        match WgpuPreprocessor::new(ctx.clone()) {
            Ok(gpu) => {
                let summary = benchmark_preprocessor("gpu", &gpu, &images, config, iterations)?
                    .with_label(&format!(
                        "gpu:{}:{:?}",
                        ctx.adapter_info().name,
                        ctx.adapter_info().backend
                    ));
                info!("preprocess_benchmark {}", serde_json::to_string(&summary)?);
            }
            Err(err) => warn!("Skipping GPU benchmark (initialization failed: {err})"),
        }
    } else {
        info!("GPU context unavailable; skipping GPU preprocessing benchmark.");
    }

    Ok(())
}

fn load_benchmark_images(items: &[ProcessingItem]) -> Result<Vec<DynamicImage>> {
    let mut images = Vec::with_capacity(items.len());
    for item in items {
        let img = load_image(&item.source)
            .with_context(|| format!("failed to load benchmark image {}", item.source.display()))?;
        images.push(img);
    }
    Ok(images)
}

fn benchmark_preprocessor<T: Preprocessor + ?Sized>(
    label: &str,
    preprocessor: &T,
    images: &[DynamicImage],
    config: &PreprocessConfig,
    iterations: usize,
) -> Result<PreprocessBenchmarkSummary> {
    let mut timings = Vec::with_capacity(images.len() * iterations);
    for image in images {
        for _ in 0..iterations {
            let start = Instant::now();
            preprocessor.preprocess(image, config)?;
            timings.push(start.elapsed());
        }
    }
    Ok(PreprocessBenchmarkSummary {
        label: label.to_string(),
        samples: images.len(),
        iterations_per_sample: iterations,
        total_ms: sum_durations_ms(&timings),
        avg_ms: avg_duration_ms(&timings),
        min_ms: timings
            .iter()
            .map(|d| duration_to_ms(*d))
            .fold(f64::MAX, f64::min),
        max_ms: timings
            .iter()
            .map(|d| duration_to_ms(*d))
            .fold(0.0, f64::max),
    })
}

fn sum_durations_ms(samples: &[Duration]) -> f64 {
    samples.iter().map(|d| duration_to_ms(*d)).sum()
}

fn avg_duration_ms(samples: &[Duration]) -> f64 {
    if samples.is_empty() {
        0.0
    } else {
        sum_durations_ms(samples) / samples.len() as f64
    }
}

fn duration_to_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use fcs_core::{CpuPreprocessor, InputSize};
    use image::{DynamicImage, GenericImageView, RgbaImage};
    use tempfile::tempdir;

    fn benchmark_config() -> PreprocessConfig {
        PreprocessConfig {
            input_size: InputSize::new(8, 8),
            ..Default::default()
        }
    }

    fn write_image(path: &std::path::Path) {
        let image = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            16,
            16,
            image::Rgba([32, 64, 96, 255]),
        ));
        image.save(path).expect("save test image");
    }

    #[test]
    fn duration_to_ms_converts_correctly() {
        assert!((duration_to_ms(Duration::from_millis(1)) - 1.0).abs() < 1e-9);
        assert!((duration_to_ms(Duration::from_secs(1)) - 1000.0).abs() < 1e-9);
        assert_eq!(duration_to_ms(Duration::ZERO), 0.0);
    }

    #[test]
    fn avg_duration_ms_returns_zero_for_empty_slice() {
        assert_eq!(avg_duration_ms(&[]), 0.0);
    }

    #[test]
    fn avg_duration_ms_returns_correct_average() {
        let samples = [Duration::from_millis(10), Duration::from_millis(20)];
        assert!((avg_duration_ms(&samples) - 15.0).abs() < 1e-9);
    }

    #[test]
    fn sum_durations_ms_returns_correct_sum() {
        let samples = [Duration::from_millis(5), Duration::from_millis(10)];
        assert!((sum_durations_ms(&samples) - 15.0).abs() < 1e-9);
        assert_eq!(sum_durations_ms(&[]), 0.0);
    }

    #[test]
    fn preprocess_benchmark_summary_with_label_replaces_label() {
        let summary = PreprocessBenchmarkSummary {
            label: "old".to_string(),
            samples: 1,
            iterations_per_sample: 1,
            total_ms: 1.0,
            avg_ms: 1.0,
            min_ms: 1.0,
            max_ms: 1.0,
        };
        let updated = summary.with_label("new");
        assert_eq!(updated.label, "new");
    }

    #[test]
    fn load_benchmark_images_reads_all_sources() {
        let dir = tempdir().expect("tempdir");
        let image_a = dir.path().join("a.png");
        let image_b = dir.path().join("b.png");
        write_image(&image_a);
        write_image(&image_b);
        let items = vec![
            ProcessingItem {
                source: image_a,
                output_override: None,
                mapping_row: None,
            },
            ProcessingItem {
                source: image_b,
                output_override: None,
                mapping_row: None,
            },
        ];

        let images = load_benchmark_images(&items).expect("load benchmark images");
        assert_eq!(images.len(), 2);
        assert_eq!(images[0].dimensions(), (16, 16));
        assert_eq!(images[1].dimensions(), (16, 16));
    }

    #[test]
    fn load_benchmark_images_adds_path_context_on_error() {
        let dir = tempdir().expect("tempdir");
        let missing = dir.path().join("missing.png");
        let items = vec![ProcessingItem {
            source: missing.clone(),
            output_override: None,
            mapping_row: None,
        }];

        let err = load_benchmark_images(&items).unwrap_err().to_string();
        assert!(err.contains("failed to load benchmark image"));
        assert!(err.contains("missing.png"));
    }

    #[test]
    fn benchmark_preprocessor_records_samples_and_iterations() {
        let images = vec![
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(16, 16, image::Rgba([1, 2, 3, 255]))),
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(16, 16, image::Rgba([4, 5, 6, 255]))),
        ];
        let summary =
            benchmark_preprocessor("cpu", &CpuPreprocessor, &images, &benchmark_config(), 2)
                .expect("benchmark_preprocessor should succeed");

        assert_eq!(summary.label, "cpu");
        assert_eq!(summary.samples, 2);
        assert_eq!(summary.iterations_per_sample, 2);
        assert!(summary.total_ms >= 0.0);
        assert!(summary.avg_ms >= 0.0);
        assert!(summary.max_ms >= summary.min_ms);
    }

    #[test]
    fn run_preprocess_benchmark_requires_inputs() {
        let err = run_preprocess_benchmark(&[], &benchmark_config(), None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("requires at least one input image"));
    }

    #[test]
    fn run_preprocess_benchmark_succeeds_without_gpu_context() {
        let dir = tempdir().expect("tempdir");
        let image = dir.path().join("bench.png");
        write_image(&image);
        let items = vec![ProcessingItem {
            source: image,
            output_override: None,
            mapping_row: None,
        }];

        run_preprocess_benchmark(&items, &benchmark_config(), None)
            .expect("preprocess benchmark should succeed");
    }
}
