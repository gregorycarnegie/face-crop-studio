//! GPU-backed enhancement pipeline.

use crate::{
    gpu::{
        GpuBackgroundBlur, GpuBilateralFilter, GpuContext, GpuGaussianBlur, GpuHistogramEqualizer,
        GpuPixelAdjust, GpuRedEyeRemoval, GpuShapeMask, red_eye::RedEye,
    },
    shape::CropShape,
};

use super::{
    EPSILON,
    detail::{
        apply_background_blur, apply_background_blur_with_preblur, apply_unsharp_mask,
        apply_unsharp_with_preblur,
    },
    red_eye::apply_red_eye_removal,
    settings::EnhancementSettings,
    skin::apply_skin_smoothing,
    tone::{
        apply_brightness, apply_contrast, apply_exposure, apply_histogram_equalization,
        apply_saturation,
    },
};
use anyhow::{Context, Result};
use image::DynamicImage;
use std::sync::Arc;

/// GPU-accelerated enhancement pipeline that currently offloads pixel adjustments.
#[derive(Clone)]
pub struct WgpuEnhancer {
    context: Arc<GpuContext>,
    pixel_adjust: GpuPixelAdjust,
    gaussian_blur: GpuGaussianBlur,
    bilateral_filter: GpuBilateralFilter,
    background_blur: GpuBackgroundBlur,
    red_eye: GpuRedEyeRemoval,
    shape_mask: GpuShapeMask,
    histogram_equalizer: GpuHistogramEqualizer,
}

impl WgpuEnhancer {
    /// Create a new GPU-backed enhancer using the shared [`GpuContext`].
    pub fn new(context: Arc<GpuContext>) -> Result<Self> {
        let pixel_adjust = GpuPixelAdjust::new(context.clone())
            .context("failed to create GPU pixel adjust pipeline")?;
        let gaussian_blur = GpuGaussianBlur::new(context.clone())
            .context("failed to create GPU gaussian blur pipeline")?;
        let bilateral_filter = GpuBilateralFilter::new(context.clone())
            .context("failed to create GPU bilateral filter pipeline")?;
        let background_blur = GpuBackgroundBlur::new(context.clone())
            .context("failed to create GPU background blur pipeline")?;
        let red_eye = GpuRedEyeRemoval::new(context.clone())
            .context("failed to create GPU red-eye pipeline")?;
        let shape_mask = GpuShapeMask::new(context.clone())
            .context("failed to create GPU shape mask pipeline")?;
        let histogram_equalizer = GpuHistogramEqualizer::new(context.clone())
            .context("failed to create GPU histogram equalization pipeline")?;
        Ok(Self {
            context,
            pixel_adjust,
            gaussian_blur,
            bilateral_filter,
            background_blur,
            red_eye,
            shape_mask,
            histogram_equalizer,
        })
    }

    /// Apply the configured enhancements, using GPU kernels where available.
    pub fn apply(
        &self,
        img: &DynamicImage,
        settings: &EnhancementSettings,
        eyes: Option<&[RedEye]>,
    ) -> Result<DynamicImage> {
        let mut out = img.clone();

        if settings.auto_color {
            out = match self.histogram_equalizer.equalize(&out) {
                Ok(eq) => eq,
                Err(err) => {
                    log::warn!("GPU histogram equalization failed: {err}");
                    apply_histogram_equalization(&out)
                }
            };
        }

        if settings.red_eye_removal {
            if let Some(corrected) = self.try_gpu_red_eye(&out, settings.red_eye_threshold, eyes)? {
                out = corrected;
            } else {
                out = apply_red_eye_removal(&out, settings.red_eye_threshold, eyes);
            }
        }

        if GpuPixelAdjust::needs_adjustment(settings) {
            out = self
                .pixel_adjust
                .apply(&out, settings)
                .context("gpu pixel adjust failed")?;
        } else {
            if settings.exposure_stops.abs() >= EPSILON {
                out = apply_exposure(&out, settings.exposure_stops);
            }
            if settings.brightness != 0 {
                out = apply_brightness(&out, settings.brightness);
            }
            if (settings.contrast - 1.0).abs() >= EPSILON {
                out = apply_contrast(&out, settings.contrast);
            }
            if (settings.saturation - 1.0).abs() >= EPSILON {
                out = apply_saturation(&out, settings.saturation);
            }
        }

        if settings.skin_smooth_amount > 0.0 {
            if let Some(smoothed) = self.try_gpu_skin_smoothing(settings, &out)? {
                out = smoothed;
            } else {
                out = apply_skin_smoothing(
                    &out,
                    settings.skin_smooth_amount,
                    settings.skin_smooth_sigma_space,
                    settings.skin_smooth_sigma_color,
                );
            }
        }

        let combined_sharp = (settings.unsharp_amount + settings.sharpness).clamp(0.0, 2.0);
        if combined_sharp > 0.0 && settings.unsharp_radius > 0.0 {
            if let Some(blurred) = self.try_gpu_blur(&out, settings.unsharp_radius)? {
                out = apply_unsharp_with_preblur(&out, &blurred, combined_sharp);
            } else {
                out = apply_unsharp_mask(&out, combined_sharp, settings.unsharp_radius);
            }
        }

        if settings.background_blur {
            if let Some(result) = self.try_gpu_background_blur(&out, settings)? {
                out = result;
            } else if let Some(blurred) =
                self.try_gpu_blur(&out, settings.background_blur_radius)?
            {
                out = apply_background_blur_with_preblur(
                    &out,
                    &blurred,
                    settings.background_blur_mask_size,
                );
            } else {
                out = apply_background_blur(
                    &out,
                    settings.background_blur_radius,
                    settings.background_blur_mask_size,
                );
            }
        }

        Ok(out)
    }

    /// Access the underlying GPU context (handy for logging/tests).
    pub fn context(&self) -> &Arc<GpuContext> {
        &self.context
    }

    fn try_gpu_blur(&self, image: &DynamicImage, radius: f32) -> Result<Option<DynamicImage>> {
        if radius <= 0.0 {
            return Ok(None);
        }
        match self.gaussian_blur.blur(image, radius) {
            Ok(blurred) => Ok(Some(blurred)),
            Err(err) => {
                log::warn!("GPU gaussian blur failed: {err}");
                Ok(None)
            }
        }
    }

    fn try_gpu_skin_smoothing(
        &self,
        settings: &EnhancementSettings,
        image: &DynamicImage,
    ) -> Result<Option<DynamicImage>> {
        if settings.skin_smooth_amount <= 0.0 {
            return Ok(None);
        }
        match self.bilateral_filter.smooth(
            image,
            settings.skin_smooth_amount,
            settings.skin_smooth_sigma_space,
            settings.skin_smooth_sigma_color,
        ) {
            Ok(result) => Ok(Some(result)),
            Err(err) => {
                log::warn!("GPU skin smoothing failed: {err}");
                Ok(None)
            }
        }
    }

    fn try_gpu_background_blur(
        &self,
        image: &DynamicImage,
        settings: &EnhancementSettings,
    ) -> Result<Option<DynamicImage>> {
        if !settings.background_blur || settings.background_blur_radius <= 0.0 {
            return Ok(None);
        }
        let blurred = match self.try_gpu_blur(image, settings.background_blur_radius)? {
            Some(b) => b,
            None => return Ok(None),
        };
        match self
            .background_blur
            .blend(image, &blurred, settings.background_blur_mask_size)
        {
            Ok(result) => Ok(Some(result)),
            Err(err) => {
                log::warn!("GPU background blur failed: {err}");
                Ok(None)
            }
        }
    }

    fn try_gpu_red_eye(
        &self,
        image: &DynamicImage,
        threshold: f32,
        eyes: Option<&[RedEye]>,
    ) -> Result<Option<DynamicImage>> {
        if threshold <= 0.0 {
            return Ok(None);
        }
        match self.red_eye.apply(image, threshold, eyes) {
            Ok(result) => Ok(Some(result)),
            Err(err) => {
                log::warn!("GPU red-eye removal failed: {err}");
                Ok(None)
            }
        }
    }

    pub fn apply_shape_mask_gpu(
        &self,
        image: &DynamicImage,
        shape: &CropShape,
        vignette_softness: f32,
        vignette_intensity: f32,
        vignette_color: crate::color::RgbaColor,
    ) -> Result<Option<DynamicImage>> {
        self.shape_mask.apply(
            image,
            shape,
            vignette_softness,
            vignette_intensity,
            vignette_color,
        )
    }

    /// Clears any internal GPU buffer pools to free memory.
    pub fn clear_caches(&self) {
        self.gaussian_blur.clear_cache();
        self.background_blur.clear_cache();
    }

    /// Returns the estimated total size in bytes of internal GPU buffer pools.
    pub fn memory_usage(&self) -> u64 {
        self.gaussian_blur.memory_usage() + self.background_blur.memory_usage()
    }
}
