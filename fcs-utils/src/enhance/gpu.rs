//! GPU-backed enhancement pipeline.

use crate::{
    gpu::{
        GpuBackgroundBlur, GpuBilateralFilter, GpuContext, GpuGaussianBlur, GpuHistogramEqualizer,
        GpuPixelAdjust, GpuRedEyeRemoval, GpuShapeMask, red_eye::RedEye,
    },
    shape::CropShape,
};

use super::{
    detail::{
        apply_background_blur, apply_background_blur_with_preblur, apply_unsharp_mask,
        apply_unsharp_with_preblur,
    },
    red_eye::apply_red_eye_removal,
    settings::EnhancementSettings,
    skin::apply_skin_smoothing,
    tone::apply_histogram_equalization,
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

        // `needs_adjustment` covers exposure, brightness, contrast and saturation, so when it
        // is false there is nothing for a CPU fallback to do.
        if GpuPixelAdjust::needs_adjustment(settings) {
            out = self
                .pixel_adjust
                .apply(&out, settings)
                .context("gpu pixel adjust failed")?;
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
        // A zero radius comes back from `try_gpu_blur` as None, which falls through to the CPU.
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

    /// Apply the GPU shape mask and optional edge vignette.
    ///
    /// See [`crate::gpu::shape_mask::GpuShapeMask::apply`] for parameter units,
    /// outline limits, and when `Ok(None)` is returned. GPU errors are propagated.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::test_support::{gradient_image, test_context};

    /// `apply` is a chain of `if setting is active { run stage }` guards, and it
    /// had no tests at all. Every guard was therefore free to invert: a mutated
    /// comparison would either skip a requested enhancement or run one nobody
    /// asked for, and nothing noticed.
    ///
    /// Note that `EnhancementSettings::default()` is *not* neutral — it ships
    /// `unsharp_amount: 0.6`, so sharpening is on. A genuine no-op baseline has
    /// to zero that as well, which is itself worth pinning down.
    fn neutral() -> EnhancementSettings {
        EnhancementSettings {
            unsharp_amount: 0.0,
            sharpness: 0.0,
            ..Default::default()
        }
    }

    #[test]
    fn default_settings_are_not_a_no_op() {
        let d = EnhancementSettings::default();
        assert!(
            d.unsharp_amount > 0.0,
            "defaults are expected to sharpen; neutral() exists because of it"
        );
    }

    #[test]
    fn neutral_settings_leave_the_image_untouched() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping WgpuEnhancer test: no adapter");
            return;
        };
        let enhancer = WgpuEnhancer::new(ctx).expect("init");
        let image = gradient_image(24, 18);

        let out = enhancer.apply(&image, &neutral(), None).expect("apply");
        assert_eq!(
            out.to_rgba8().as_raw(),
            image.to_rgba8().as_raw(),
            "no active enhancement should mean no change; a guard that inverted \
             would run a stage nobody asked for"
        );
    }

    #[test]
    fn each_enhancement_changes_the_image_when_its_setting_is_active() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping WgpuEnhancer test: no adapter");
            return;
        };
        let enhancer = WgpuEnhancer::new(ctx).expect("init");
        let image = gradient_image(32, 24);
        let baseline = image.to_rgba8().into_raw();

        // One stage at a time, each starting from the neutral baseline, so a
        // failure names the stage whose guard broke rather than "something".
        let cases: Vec<(&str, EnhancementSettings)> = vec![
            (
                "auto_color",
                EnhancementSettings {
                    auto_color: true,
                    ..neutral()
                },
            ),
            (
                "exposure_stops",
                EnhancementSettings {
                    exposure_stops: 1.0,
                    ..neutral()
                },
            ),
            (
                "brightness",
                EnhancementSettings {
                    brightness: 30,
                    ..neutral()
                },
            ),
            (
                "contrast",
                EnhancementSettings {
                    contrast: 1.6,
                    ..neutral()
                },
            ),
            (
                "saturation",
                EnhancementSettings {
                    saturation: 1.8,
                    ..neutral()
                },
            ),
            (
                "skin_smooth_amount",
                EnhancementSettings {
                    skin_smooth_amount: 0.8,
                    ..neutral()
                },
            ),
            (
                "unsharp_amount",
                EnhancementSettings {
                    unsharp_amount: 1.2,
                    unsharp_radius: 2.0,
                    ..neutral()
                },
            ),
            (
                "sharpness",
                EnhancementSettings {
                    sharpness: 1.0,
                    unsharp_radius: 2.0,
                    ..neutral()
                },
            ),
            (
                "background_blur",
                EnhancementSettings {
                    background_blur: true,
                    ..neutral()
                },
            ),
        ];

        for (name, settings) in cases {
            let out = enhancer
                .apply(&image, &settings, None)
                .unwrap_or_else(|e| panic!("apply failed for {name}: {e}"));
            assert_eq!(
                (out.width(), out.height()),
                (image.width(), image.height()),
                "{name} must preserve dimensions"
            );
            assert_ne!(
                out.to_rgba8().as_raw(),
                &baseline,
                "{name} is active but changed nothing, so its guard did not fire"
            );
        }
    }

    #[test]
    fn negative_exposure_is_active_too() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping WgpuEnhancer test: no adapter");
            return;
        };
        let enhancer = WgpuEnhancer::new(ctx).expect("init");
        let image = gradient_image(16, 16);

        // The guard tests an absolute value, so a sign-blind comparison would
        // silently drop darkening while keeping brightening.
        let settings = EnhancementSettings {
            exposure_stops: -1.0,
            ..neutral()
        };
        let out = enhancer.apply(&image, &settings, None).expect("apply");
        assert_ne!(out.to_rgba8().as_raw(), image.to_rgba8().as_raw());
    }

    #[test]
    fn unsharp_needs_both_an_amount_and_a_radius() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping WgpuEnhancer test: no adapter");
            return;
        };
        let enhancer = WgpuEnhancer::new(ctx).expect("init");
        let image = gradient_image(16, 16);

        // The stage is gated on `combined > 0.0 && radius > 0.0`; with either
        // half missing it must not run. An `||` there would sharpen with a zero
        // radius, and a zero amount would sharpen by nothing.
        for (amount, radius) in [(1.0f32, 0.0f32), (0.0, 2.0)] {
            let settings = EnhancementSettings {
                unsharp_amount: amount,
                unsharp_radius: radius,
                ..neutral()
            };
            let out = enhancer.apply(&image, &settings, None).expect("apply");
            assert_eq!(
                out.to_rgba8().as_raw(),
                image.to_rgba8().as_raw(),
                "unsharp with amount={amount} radius={radius} must be inert"
            );
        }
    }

    #[test]
    fn memory_usage_and_clear_caches_track_the_pools() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping WgpuEnhancer test: no adapter");
            return;
        };
        let enhancer = WgpuEnhancer::new(ctx).expect("init");
        let image = gradient_image(32, 32);

        // Sharpening fills only the gaussian pool, so a product of the two pools would be 0.
        let sharpen = EnhancementSettings {
            unsharp_amount: 1.2,
            unsharp_radius: 2.0,
            ..neutral()
        };
        enhancer.apply(&image, &sharpen, None).expect("apply");
        assert!(
            enhancer.memory_usage() > 0,
            "the gaussian pool alone must count"
        );

        // background_blur exercises both the gaussian and background pools,
        // which are the two that memory_usage sums.
        let settings = EnhancementSettings {
            background_blur: true,
            ..neutral()
        };
        enhancer.apply(&image, &settings, None).expect("apply");
        assert!(
            enhancer.memory_usage() > 0,
            "pools should hold buffers after a blur pass"
        );

        enhancer.clear_caches();
        assert_eq!(enhancer.memory_usage(), 0, "clear_caches must empty them");
    }

    #[test]
    fn gpu_stages_run_on_the_gpu_rather_than_falling_back() {
        // Each try_gpu_* returning None silently routes to the CPU filter, whose output is
        // close enough that apply()-level tests cannot tell. Ask the helpers directly.
        let Some(ctx) = test_context() else {
            return;
        };
        let enhancer = WgpuEnhancer::new(ctx).expect("init");
        let image = gradient_image(16, 16);
        let s = EnhancementSettings {
            skin_smooth_amount: 0.8,
            background_blur: true,
            ..neutral()
        };
        // The size, not just `is_some`: a defaulted empty image is also Some.
        let dims = |result: Result<Option<DynamicImage>>| {
            result.unwrap().map(|img| (img.width(), img.height()))
        };
        assert_eq!(dims(enhancer.try_gpu_skin_smoothing(&s, &image)), Some((16, 16)));
        assert_eq!(dims(enhancer.try_gpu_background_blur(&image, &s)), Some((16, 16)));
        assert_eq!(
            dims(enhancer.try_gpu_red_eye(&image, s.red_eye_threshold, None)),
            Some((16, 16))
        );
    }

    #[test]
    fn gpu_bilateral_matches_the_cpu_filter() {
        // Same exponentials both sides; the CPU rounds the filtered value before blending, so
        // allow 2 levels. gradient_image wraps mod 256, giving hard edges where the sampling
        // radius changes the answer.
        let Some(ctx) = test_context() else {
            return;
        };
        let enhancer = WgpuEnhancer::new(ctx).expect("init");
        let image = gradient_image(29, 23);
        let gpu = enhancer
            .bilateral_filter
            .smooth(&image, 1.0, 3.0, 25.0)
            .unwrap()
            .to_rgba8();
        let cpu = crate::enhance::skin::skin_smooth_rgba(&image.to_rgba8(), 1.0, 3.0, 25.0);
        let worst = gpu
            .as_raw()
            .iter()
            .zip(cpu.as_raw())
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        assert!(worst <= 2, "GPU and CPU bilateral differ by {worst}");
    }

    #[test]
    fn gpu_histogram_equalization_matches_the_cpu_lut() {
        // Every pixel compared, so a wrong pixel total or a short histogram buffer both show.
        let Some(ctx) = test_context() else {
            return;
        };
        let enhancer = WgpuEnhancer::new(ctx).expect("init");
        let image = gradient_image(41, 23);
        let gpu = enhancer
            .histogram_equalizer
            .equalize(&image)
            .unwrap()
            .to_rgba8();
        let cpu = crate::enhance::tone::apply_histogram_equalization(&image).to_rgba8();
        let worst = gpu
            .as_raw()
            .iter()
            .zip(cpu.as_raw())
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        assert!(worst <= 1, "GPU and CPU equalization differ by {worst}");
    }

    #[test]
    fn shape_mask_delegates_to_the_gpu_mask() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping GPU shape-mask delegation test: no GPU");
            return;
        };
        let enhancer = WgpuEnhancer::new(ctx).expect("init enhancer");
        let image = gradient_image(32, 32);

        let masked = enhancer
            .apply_shape_mask_gpu(
                &image,
                &CropShape::Ellipse,
                0.0,
                0.0,
                crate::color::RgbaColor::opaque(0, 0, 0),
            )
            .expect("shape mask should not error")
            .expect("a non-rectangular shape produces an image")
            .to_rgba8();

        assert_eq!(masked.dimensions(), (32, 32));
        assert_eq!(
            masked.get_pixel(0, 0)[3],
            0,
            "the corner falls outside the ellipse"
        );
        assert_eq!(
            masked.get_pixel(16, 16)[3],
            255,
            "the centre is kept opaque"
        );
    }
}
