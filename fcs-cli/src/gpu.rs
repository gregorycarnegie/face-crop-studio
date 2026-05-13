//! GPU runtime and context management for fcs-cli.

use std::{num::NonZeroUsize, sync::Arc};

use anyhow::{Result, anyhow};
use fcs_core::{CropSettings, Detection, calculate_crop_region};
use fcs_utils::{
    BatchCropRequest, CropShape, EnhancementSettings, GpuAvailability, GpuBatchCropper, GpuContext,
    GpuContextOptions, GpuContextPool, GpuPoolError, WgpuEnhancer, apply_enhancements,
    apply_shape_mask_dynamic,
    config::AppSettings,
    gpu::{GpuStatusIndicator, GpuStatusMode},
};
use image::{DynamicImage, GenericImageView};
use log::{debug, info, warn};

pub struct CliGpuRuntime {
    context: Option<Arc<GpuContext>>,
    pool: Option<GpuContextPool>,
    status: GpuStatusIndicator,
    enhancer: Option<Arc<WgpuEnhancer>>,
    cropper: Option<Arc<GpuBatchCropper>>,
}

impl CliGpuRuntime {
    pub fn context(&self) -> Option<&Arc<GpuContext>> {
        self.context.as_ref()
    }

    pub fn log_pool_state(&self) {
        if let Some(pool) = self.pool.as_ref() {
            debug!(
                "GPU context pool ready (available {} of {})",
                pool.available(),
                pool.capacity()
            );
        }
    }

    pub fn log_status(&self) {
        log_gpu_status(&self.status, self.pool.as_ref());
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

    pub fn crop_faces_gpu(
        &self,
        image: &DynamicImage,
        detections: &[Detection],
        settings: &CropSettings,
    ) -> Option<Vec<DynamicImage>> {
        if settings.output_width == 0 || settings.output_height == 0 {
            return None;
        }
        let cropper = self.cropper.as_ref()?;
        if detections.is_empty() {
            return Some(Vec::new());
        }

        let (img_w, img_h) = image.dimensions();
        let mut jobs = Vec::with_capacity(detections.len());
        for det in detections {
            let region = calculate_crop_region(img_w, img_h, det.bbox, settings);
            if region.requires_padding() {
                return None;
            }
            let (source_x, source_y, source_width, source_height) =
                region.in_bounds_rect(img_w, img_h)?;
            jobs.push(BatchCropRequest {
                source_x,
                source_y,
                source_width: source_width.max(1),
                source_height: source_height.max(1),
                output_width: settings.output_width,
                output_height: settings.output_height,
            });
        }

        match cropper.crop(image, &jobs) {
            Ok(images) => {
                if images.len() == detections.len() {
                    Some(images)
                } else {
                    warn!(
                        "GPU crop count mismatch (expected {}, got {}); reverting to CPU crops.",
                        detections.len(),
                        images.len()
                    );
                    None
                }
            }
            Err(err) => {
                warn!("GPU batch cropping failed: {err}; reverting to CPU crops.");
                None
            }
        }
    }

    #[allow(dead_code)]
    pub fn pool(&self) -> Option<&GpuContextPool> {
        self.pool.as_ref()
    }

    #[allow(dead_code)]
    pub fn status(&self) -> &GpuStatusIndicator {
        &self.status
    }
}

fn log_gpu_status(status: &GpuStatusIndicator, pool: Option<&GpuContextPool>) {
    let detail = status.detail.as_deref();
    let pool_capacity = pool.map(|p| p.capacity());
    let pool_available = pool.map(|p| p.available());
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
            if let Some(pool) = pool {
                info!(
                    "GPU ready: {adapter} via {backend} (driver: {driver}, vendor={vendor}, device={device}). Pool capacity {} ({} idle).",
                    pool.capacity(),
                    pool.available()
                );
            } else {
                info!(
                    "GPU ready: {adapter} via {backend} (driver: {driver}, vendor={vendor}, device={device})."
                );
            }
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
    status.emit_telemetry(pool_capacity, pool_available);
}

#[cfg(test)]
mod tests {
    use super::*;
    use fcs_core::{CropSettings, FillColor, PositioningMode};
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
            context: None,
            pool: None,
            status,
            enhancer: None,
            cropper: None,
        }
    }

    // --- log_gpu_status: smoke-test each GpuStatusMode variant ---

    #[test]
    fn log_gpu_status_disabled_does_not_panic() {
        let status = GpuStatusIndicator::disabled("unit test");
        log_gpu_status(&status, None);
        assert_eq!(status.mode, GpuStatusMode::Disabled);
    }

    #[test]
    fn log_gpu_status_error_does_not_panic() {
        let status = GpuStatusIndicator::error("unit test error");
        log_gpu_status(&status, None);
        assert_eq!(status.mode, GpuStatusMode::Error);
    }

    #[test]
    fn log_gpu_status_fallback_does_not_panic() {
        let status = GpuStatusIndicator::fallback("no reason", None, None);
        log_gpu_status(&status, None);
        assert_eq!(status.mode, GpuStatusMode::Fallback);
    }

    #[test]
    fn log_gpu_status_pending_does_not_panic() {
        let status = GpuStatusIndicator::pending();
        log_gpu_status(&status, None);
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
        log_gpu_status(&status, None);
        assert_eq!(status.mode, GpuStatusMode::Available);
    }

    #[test]
    fn runtime_accessors_return_expected_state() {
        let runtime = manual_runtime(GpuStatusIndicator::pending());
        assert!(runtime.pool().is_none());
        assert_eq!(runtime.status().mode, GpuStatusMode::Pending);
        runtime.log_pool_state();
        runtime.log_status();
    }

    // --- CliGpuRuntime methods with GPU disabled ---

    #[test]
    fn init_runtime_gpu_disabled_has_no_context() {
        let runtime = init_cli_gpu_runtime(&no_gpu_settings()).expect("init");
        assert!(runtime.context().is_none());
    }

    #[test]
    fn log_pool_state_and_log_status_do_not_panic() {
        let runtime = init_cli_gpu_runtime(&no_gpu_settings()).expect("init");
        runtime.log_pool_state();
        runtime.log_status();
    }

    #[test]
    fn crop_faces_gpu_returns_none_for_zero_output_width() {
        let runtime = init_cli_gpu_runtime(&no_gpu_settings()).expect("init");
        let img = DynamicImage::ImageRgba8(RgbaImage::new(100, 100));
        let settings = CropSettings {
            output_width: 0,
            output_height: 100,
            face_height_pct: 70.0,
            positioning_mode: PositioningMode::Center,
            horizontal_offset: 0.0,
            vertical_offset: 0.0,
            fill_color: FillColor::default(),
            eye_line_align: false,
        };
        assert!(runtime.crop_faces_gpu(&img, &[], &settings).is_none());
    }

    #[test]
    fn crop_faces_gpu_returns_none_when_cropper_absent() {
        // GPU disabled → cropper is None → returns None regardless of detections
        let runtime = init_cli_gpu_runtime(&no_gpu_settings()).expect("init");
        let img = DynamicImage::ImageRgba8(RgbaImage::new(100, 100));
        let settings = CropSettings {
            output_width: 50,
            output_height: 50,
            face_height_pct: 70.0,
            positioning_mode: PositioningMode::Center,
            horizontal_offset: 0.0,
            vertical_offset: 0.0,
            fill_color: FillColor::default(),
            eye_line_align: false,
        };
        assert!(runtime.crop_faces_gpu(&img, &[], &settings).is_none());
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

pub fn init_cli_gpu_runtime(settings: &AppSettings) -> Result<CliGpuRuntime> {
    let options: GpuContextOptions = (&settings.gpu).into();
    let availability = GpuContext::init_with_fallback(&options);

    let (context, pool, status) = match &availability {
        GpuAvailability::Available(context) => {
            let pool_size = NonZeroUsize::new(rayon::current_num_threads())
                .unwrap_or_else(|| NonZeroUsize::new(1).expect("non-zero pool size fallback"));
            let pool =
                GpuContextPool::new(context.clone(), pool_size).map_err(|err| match err {
                    GpuPoolError::Closed => anyhow!("failed to initialize GPU pool: {err}"),
                })?;

            let info = context.adapter_info();
            let status = GpuStatusIndicator::available(
                info.name.clone(),
                format!("{:?}", info.backend),
                Some(info.driver.clone()),
                Some(info.vendor),
                Some(info.device),
            );
            (Some(context.clone()), Some(pool), status)
        }
        GpuAvailability::Disabled { reason } => {
            (None, None, GpuStatusIndicator::disabled(reason.clone()))
        }
        GpuAvailability::Unavailable { error } => {
            (None, None, GpuStatusIndicator::error(error.to_string()))
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

    let cropper = match &context {
        Some(ctx) => match GpuBatchCropper::new(ctx.clone()) {
            Ok(cropper) => {
                info!(
                    "GPU batch cropper ready on '{}' ({:?})",
                    ctx.adapter_info().name,
                    ctx.adapter_info().backend
                );
                Some(Arc::new(cropper))
            }
            Err(err) => {
                warn!("GPU batch cropper initialization failed: {err}");
                None
            }
        },
        None => None,
    };

    let runtime = CliGpuRuntime {
        context,
        pool,
        status,
        enhancer,
        cropper,
    };
    runtime.log_status();
    runtime.log_pool_state();
    Ok(runtime)
}
