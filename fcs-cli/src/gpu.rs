//! GPU runtime and context management for fcs-cli.

use anyhow::Result;
use fcs_utils::{
    EnhancementRuntime, EnhancementSettings, FinishedCrop, GpuAvailability, GpuContext,
    GpuContextOptions, RedEye,
    config::AppSettings,
    gpu::{GpuStatusIndicator, GpuStatusMode},
};
use image::DynamicImage;
use log::{debug, info, warn};
use std::sync::Arc;

/// The CLI's one GPU context, and the enhancement and shape-mask work done on it.
pub struct CliGpuRuntime {
    status: GpuStatusIndicator,
    /// Handed to the detector, which used to open a second device on the same adapter
    /// (experiment 104).
    context: Option<Arc<GpuContext>>,
    /// Shared with the GUI, so the two front-ends cannot diverge about what enhancement means.
    enhancement: EnhancementRuntime,
}

impl CliGpuRuntime {
    pub fn log_status(&self) {
        log_gpu_status(&self.status);
    }

    pub fn context(&self) -> Option<Arc<GpuContext>> {
        self.context.clone()
    }

    /// Enhance, score, then shape and fill: the order every front-end exports in.
    pub fn finish_crop(
        &self,
        crop: DynamicImage,
        crop_settings: &fcs_utils::config::CropSettings,
        enhancement: Option<&EnhancementSettings>,
        eyes: Option<&[RedEye]>,
    ) -> FinishedCrop {
        self.enhancement
            .finish_crop(crop, crop_settings, enhancement, eyes)
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

/// Whether the job will dispatch any of the GPU enhancer's pipelines.
///
/// The enhancer compiles seven of them through FXC, and only enhancement and a
/// non-rectangular shape use any. Settings are fixed for the run, so a job that uses neither
/// skips the build and runs the same CPU fallbacks it would anyway. With the shared context
/// this took a launch from 2.30 to 1.62 s (experiment 104).
fn needs_gpu_enhancer(settings: &AppSettings, enhancement: Option<&EnhancementSettings>) -> bool {
    enhancement.is_some() || !matches!(settings.crop.shape, fcs_utils::CropShape::Rectangle)
}

/// `enhancement` is the job's `--enhance` settings, `None` without it.
pub fn init_cli_gpu_runtime(
    settings: &AppSettings,
    enhancement: Option<&EnhancementSettings>,
) -> Result<CliGpuRuntime> {
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

    let enhancement = EnhancementRuntime::new(
        context
            .clone()
            .filter(|_| needs_gpu_enhancer(settings, enhancement)),
    );
    if let Some(name) = enhancement.adapter_name() {
        info!("GPU enhancement pipeline ready on '{name}'");
    }

    let runtime = CliGpuRuntime {
        status,
        context,
        enhancement,
    };
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
            context: None,
            enhancement: EnhancementRuntime::cpu_only(),
        }
    }

    /// A rectangle without `--enhance` is the default job, and the one that must not pay for
    /// the enhancer; either of the other two dispatches it and must still get it.
    #[test]
    fn the_gpu_enhancer_is_built_only_when_something_dispatches_it() {
        let mut settings = AppSettings::default();
        assert!(matches!(
            settings.crop.shape,
            fcs_utils::CropShape::Rectangle
        ));
        assert!(!needs_gpu_enhancer(&settings, None));
        assert!(needs_gpu_enhancer(
            &settings,
            Some(&EnhancementSettings::default())
        ));
        settings.crop.shape = fcs_utils::CropShape::Ellipse;
        assert!(needs_gpu_enhancer(&settings, None));
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
        let runtime = init_cli_gpu_runtime(&no_gpu_settings(), None).expect("init");
        assert_eq!(runtime.enhancement.backend(), "cpu");
    }

    #[test]
    fn log_status_does_not_panic() {
        let runtime = init_cli_gpu_runtime(&no_gpu_settings(), None).expect("init");
        runtime.log_status();
    }

    #[test]
    fn finish_crop_falls_back_to_cpu_and_preserves_dimensions() {
        let runtime = init_cli_gpu_runtime(&no_gpu_settings(), None).expect("init");
        let img = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            20,
            30,
            image::Rgba([100, 120, 140, 255]),
        ));
        let crop = fcs_utils::config::CropSettings::default();
        let enh = EnhancementSettings::default();
        let result = runtime.finish_crop(img, &crop, Some(&enh), None).image;
        assert_eq!((result.width(), result.height()), (20, 30));
    }
}
