//! One place that decides whether enhancement runs on the GPU or the CPU.
//!
//! Both front-ends enhance crops, and until 2.0 they each did it their own way. The results
//! diverged in both directions, quietly:
//!
//! * The CLI built a [`WgpuEnhancer`] and used it, so enhancement ran on the WGSL shaders — but
//!   it passed `None` for the eye positions, so **red-eye removal had nothing to aim at** and
//!   fell back to scanning the whole crop.
//! * The GUI passed the eye positions, so red-eye removal was targeted — but it called
//!   [`apply_enhancements`] directly, so **none of the shaders ever ran**. It held a
//!   `GpuContext` the whole time and used it for a status label.
//!
//! Neither was a decision; each side simply grew the half its author was looking at. With one
//! runtime there is one answer, and a front-end that forgets to pass the eyes is passing them
//! to the same function either way.

use std::sync::Arc;

use image::DynamicImage;
use log::warn;

use super::{EnhancementSettings, WgpuEnhancer, apply_enhancements};
use crate::gpu::red_eye::RedEye;
use crate::{color::RgbaColor, gpu::GpuContext, shape::CropShape, shape::apply_shape_mask_dynamic};

/// Enhancement and shape masking, on the GPU when one is available.
///
/// Cheap to clone: the enhancer is behind an `Arc` so a rayon batch can share one, which is how
/// the CLI has always used it.
#[derive(Clone)]
pub struct EnhancementRuntime {
    enhancer: Option<Arc<WgpuEnhancer>>,
}

impl std::fmt::Debug for EnhancementRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnhancementRuntime")
            .field("backend", &self.backend())
            .finish()
    }
}

impl EnhancementRuntime {
    /// CPU only. Every method still works; nothing is optional about the results.
    pub fn cpu_only() -> Self {
        Self { enhancer: None }
    }

    /// Build the GPU path on `context`, falling back to the CPU if the pipelines will not build.
    ///
    /// `None` for the context is the ordinary CPU case, not an error.
    pub fn new(context: Option<Arc<GpuContext>>) -> Self {
        let Some(context) = context else {
            return Self::cpu_only();
        };
        match WgpuEnhancer::new(context) {
            Ok(enhancer) => Self {
                enhancer: Some(Arc::new(enhancer)),
            },
            Err(err) => {
                warn!("GPU enhancer initialization failed: {err}; enhancement will run on the CPU");
                Self::cpu_only()
            }
        }
    }

    /// Which path enhancement will take, for logs and the status bar.
    pub fn backend(&self) -> &'static str {
        if self.enhancer.is_some() {
            "wgsl-gpu"
        } else {
            "cpu"
        }
    }

    /// The adapter behind the GPU path, when there is one.
    pub fn adapter_name(&self) -> Option<String> {
        self.enhancer
            .as_ref()
            .map(|e| e.context().adapter_info().name.clone())
    }

    /// Enhance `image`, on the GPU when available.
    ///
    /// `eyes` are the red-eye targets in `image`'s own coordinates. Passing `None` means
    /// "nowhere in particular", which is weaker, not faster — so callers that know where the
    /// eyes are should say.
    pub fn enhance(
        &self,
        image: &DynamicImage,
        settings: &EnhancementSettings,
        eyes: Option<&[RedEye]>,
    ) -> DynamicImage {
        if let Some(enhancer) = &self.enhancer {
            match enhancer.apply(image, settings, eyes) {
                Ok(output) => return output,
                // A GPU failure mid-batch must not lose the crop: the CPU pipeline computes the
                // same thing, so the fallback is a slowdown rather than a different result.
                Err(err) => warn!("GPU enhancement failed: {err}; falling back to the CPU"),
            }
        }
        apply_enhancements(image, settings, eyes)
    }

    /// Apply a crop shape and its vignette, on the GPU when available.
    pub fn apply_shape_mask(
        &self,
        image: &DynamicImage,
        shape: &CropShape,
        vignette_softness: f32,
        vignette_intensity: f32,
        vignette_color: RgbaColor,
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
                // `Ok(None)` means the shape needs no GPU work, not that it failed.
                Ok(None) => {}
                Err(err) => warn!("GPU shape mask failed: {err}; falling back to the CPU"),
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The CPU path must produce exactly what calling the pipeline directly produces, because
    /// that is what both front-ends used to do by hand.
    #[test]
    fn the_cpu_path_matches_the_pipeline_it_replaces() {
        let runtime = EnhancementRuntime::cpu_only();
        assert_eq!(runtime.backend(), "cpu");
        assert_eq!(runtime.adapter_name(), None);

        let image = DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
            8,
            8,
            image::Rgba([90, 120, 150, 255]),
        ));
        let settings = EnhancementSettings {
            brightness: 12,
            contrast: 1.2,
            ..EnhancementSettings::default()
        };
        assert_eq!(
            runtime.enhance(&image, &settings, None).to_rgba8(),
            apply_enhancements(&image, &settings, None).to_rgba8()
        );
    }

    /// `None` for the context is the CPU case, not a failure.
    #[test]
    fn no_context_means_the_cpu_path() {
        assert_eq!(EnhancementRuntime::new(None).backend(), "cpu");
    }

    /// The eye positions have to reach the pipeline. This is the CLI's half of the divergence:
    /// it passed `None`, so red-eye removal had nothing to aim at.
    #[test]
    fn eye_positions_reach_the_pipeline() {
        let runtime = EnhancementRuntime::cpu_only();
        // A red blob for red-eye removal to find, and an eye sitting on it.
        let image = DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
            16,
            16,
            image::Rgba([220, 30, 30, 255]),
        ));
        let settings = EnhancementSettings {
            red_eye_removal: true,
            ..EnhancementSettings::default()
        };
        let eyes = [RedEye {
            x: 8.0,
            y: 8.0,
            radius: 6.0,
            _pad: 0.0,
        }];

        let targeted = runtime.enhance(&image, &settings, Some(&eyes)).to_rgba8();
        let untargeted = runtime.enhance(&image, &settings, None).to_rgba8();
        assert_ne!(
            targeted, untargeted,
            "passing eye positions must change what red-eye removal does, or the argument is \
             decoration"
        );
    }
}
