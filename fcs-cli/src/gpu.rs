//! GPU runtime and context management for fcs-cli.

use std::sync::Arc;

use anyhow::Result;
use fcs_utils::{
    CropShape, EnhancementSettings, GpuAvailability, GpuContext, GpuContextOptions, WgpuEnhancer,
    apply_enhancements, apply_shape_mask_dynamic,
    config::AppSettings,
    gpu::{GpuStatusIndicator, GpuStatusMode},
};
use image::DynamicImage;
use log::{debug, info, warn};

/// The GPU work the CLI actually does: enhancement and shape masks.
///
/// It no longer keeps the `GpuContext` itself. The only caller that wanted one was the
/// preprocessing benchmark, and preprocessing is the detector's own business now -- the
/// enhancer holds the `Arc` it needs.
pub struct CliGpuRuntime {
    status: GpuStatusIndicator,
    enhancer: Option<Arc<WgpuEnhancer>>,
}

impl CliGpuRuntime {
    pub fn log_status(&self) {
        log_gpu_status(&self.status);
    }

    pub fn enhance(&self, image: &DynamicImage, settings: &EnhancementSettings) -> DynamicImage {
        if let Some(enhancer) = &self.enhancer {
            match enhancer.apply(image, settings, None) {
                Ok(output) => return output,
                Err(err) => {
                    warn!("GPU enhancement failed: {err}; falling back to CPU pipeline.");
                }
            }
        }
        apply_enhancements(image, settings, None)
    }

    pub fn apply_shape_mask(
        &self,
        image: &DynamicImage,
        shape: &CropShape,
        vignette_softness: f32,
        vignette_intensity: f32,
        vignette_color: fcs_utils::color::RgbaColor,
    ) -> DynamicImage {
        if let Some(enhancer) = &self.enhancer {
            match enhancer.apply_shape_mask_gpu(
                image,
                shape,
                vignette_softness,
                vignette_intensity,
                vignette_color,
            ) {
                Ok(Some(masked)) => return masked,
                Ok(None) => {}
                Err(err) => warn!("GPU shape mask failed: {err}; falling back to CPU path."),
            }
        }
        let mut cpu = image.clone();
        apply_shape_mask_dynamic(
            &mut cpu,
            shape,
            vignette_softness,
            vignette_intensity,
            vignette_color,
        );
        cpu
    }
}

fn log_gpu_status(status: &GpuStatusIndicator) {
    let detail = status.detail.as_deref();
    match status.mode {
        GpuStatusMode::Available => {
            let adapter = status.adapter_name.as_deref().unwrap_or("GPU adapter");
            let backend = status.backend.as_deref().unwrap_or("wgpu");
            let driver = status.driver.as_deref().unwrap_or("driver n/a");
            let vendor = status
                .vendor_id
                .map(|v| format!("{v:#06x}"))
                .unwrap_or_else(|| "n/a".to_string());
            let device = status
                .device_id
                .map(|d| format!("{d:#06x}"))
                .unwrap_or_else(|| "n/a".to_string());
            info!(
                "GPU ready: {adapter} via {backend} (driver: {driver}, vendor={vendor}, device={device})."
            );
        }
        GpuStatusMode::Disabled => {
            if let Some(reason) = detail {
                info!("GPU disabled: {reason}");
            } else {
                info!("GPU disabled.");
            }
        }
        GpuStatusMode::Fallback => {
            if let Some(reason) = detail {
                warn!("GPU fallback to CPU: {reason}");
            } else {
                warn!("GPU fallback to CPU.");
            }
        }
        GpuStatusMode::Error => {
            if let Some(reason) = detail {
                warn!("GPU unavailable: {reason}");
            } else {
                warn!("GPU unavailable.");
            }
        }
        GpuStatusMode::Pending => {
            debug!("GPU status pending...");
        }
    }
    status.emit_telemetry();
}

pub fn init_cli_gpu_runtime(settings: &AppSettings) -> Result<CliGpuRuntime> {
    let options: GpuContextOptions = (&settings.gpu).into();
    let availability = GpuContext::init_with_fallback(&options);

    let (context, status) = match &availability {
        GpuAvailability::Available(context) => {
            let info = context.adapter_info();
            let status = GpuStatusIndicator::available(
                info.name.clone(),
                format!("{:?}", info.backend),
                Some(info.driver.clone()),
                Some(info.vendor),
                Some(info.device),
            );
            (Some(context.clone()), status)
        }
        GpuAvailability::Disabled { reason } => {
            (None, GpuStatusIndicator::disabled(reason.clone()))
        }
        GpuAvailability::Unavailable { error } => {
            (None, GpuStatusIndicator::error(error.to_string()))
        }
    };

    let enhancer = match &context {
        Some(ctx) => match WgpuEnhancer::new(ctx.clone()) {
            Ok(enhancer) => {
                info!(
                    "GPU enhancement pipeline ready on '{}' ({:?})",
                    ctx.adapter_info().name,
                    ctx.adapter_info().backend
                );
                Some(Arc::new(enhancer))
            }
            Err(err) => {
                warn!("GPU enhancer initialization failed: {err}");
                None
            }
        },
        None => None,
    };

    let runtime = CliGpuRuntime { status, enhancer };
    runtime.log_status();
    Ok(runtime)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fcs_utils::{
        EnhancementSettings,
        config::AppSettings,
        gpu::{GpuStatusIndicator, GpuStatusMode},
    };
    use image::{DynamicImage, RgbaImage};

    fn no_gpu_settings() -> AppSettings {
        let mut s = AppSettings::default();
        s.gpu.enabled = false;
        s
    }

    fn manual_runtime(status: GpuStatusIndicator) -> CliGpuRuntime {
        CliGpuRuntime {
            status,
            enhancer: None,
        }
    }

    // --- log_gpu_status: smoke-test each GpuStatusMode variant ---

    #[test]
    fn log_gpu_status_disabled_does_not_panic() {
        let status = GpuStatusIndicator::disabled("unit test");
        log_gpu_status(&status);
        assert_eq!(status.mode, GpuStatusMode::Disabled);
    }

    #[test]
    fn log_gpu_status_error_does_not_panic() {
        let status = GpuStatusIndicator::error("unit test error");
        log_gpu_status(&status);
        assert_eq!(status.mode, GpuStatusMode::Error);
    }

    #[test]
    fn log_gpu_status_fallback_does_not_panic() {
        let status = GpuStatusIndicator::fallback("no reason", None, None);
        log_gpu_status(&status);
        assert_eq!(status.mode, GpuStatusMode::Fallback);
    }

    #[test]
    fn log_gpu_status_pending_does_not_panic() {
        let status = GpuStatusIndicator::pending();
        log_gpu_status(&status);
        assert_eq!(status.mode, GpuStatusMode::Pending);
    }

    #[test]
    fn log_gpu_status_available_without_pool_does_not_panic() {
        let status = GpuStatusIndicator::available(
            "UnitTest GPU",
            "Vulkan",
            Some("driver".to_string()),
            Some(0x1234),
            Some(0xabcd),
        );
        log_gpu_status(&status);
        assert_eq!(status.mode, GpuStatusMode::Available);
    }

    #[test]
    fn runtime_accessors_return_expected_state() {
        let runtime = manual_runtime(GpuStatusIndicator::pending());
        runtime.log_status();
    }

    // --- CliGpuRuntime methods with GPU disabled ---

    /// With the GPU off there must be no enhancer, so `enhance` takes the CPU pipeline.
    ///
    /// This used to assert the absence of the stored `GpuContext`, which the runtime no longer
    /// keeps; the enhancer is the observable consequence, and the thing a caller notices.
    #[test]
    fn init_runtime_gpu_disabled_has_no_enhancer() {
        let runtime = init_cli_gpu_runtime(&no_gpu_settings()).expect("init");
        assert!(runtime.enhancer.is_none());
    }

    #[test]
    fn log_status_does_not_panic() {
        let runtime = init_cli_gpu_runtime(&no_gpu_settings()).expect("init");
        runtime.log_status();
    }

    #[test]
    fn enhance_falls_back_to_cpu_and_preserves_dimensions() {
        let runtime = init_cli_gpu_runtime(&no_gpu_settings()).expect("init");
        let img = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            20,
            20,
            image::Rgba([100, 120, 140, 255]),
        ));
        let enh = EnhancementSettings::default();
        let result = runtime.enhance(&img, &enh);
        assert_eq!(result.width(), 20);
        assert_eq!(result.height(), 20);
    }

    #[test]
    fn apply_shape_mask_falls_back_to_cpu_and_preserves_dimensions() {
        use fcs_utils::{CropShape, color::RgbaColor};
        let runtime = init_cli_gpu_runtime(&no_gpu_settings()).expect("init");
        let img = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            30,
            30,
            image::Rgba([200, 200, 200, 255]),
        ));
        let result =
            runtime.apply_shape_mask(&img, &CropShape::Rectangle, 0.0, 0.0, RgbaColor::default());
        assert_eq!(result.width(), 30);
        assert_eq!(result.height(), 30);
    }
}
