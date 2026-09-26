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
use crate::{
    color::RgbaColor,
    config::CropSettings,
    gpu::GpuContext,
    quality::{Quality, estimate_sharpness},
    shape::CropShape,
    shape::apply_shape_mask_dynamic,
};

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
                // `Ok(None)` means the GPU pass does not take this shape (a rectangle, or an
                // outline too dense for it), not that it failed.
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

    /// Everything after the geometric crop, in the one order every export path uses: enhance,
    /// score sharpness, then shape and fill.
    ///
    /// The GUI used to shape before enhancing and score last, the CLI the other way round and
    /// without the fill, so the same face could pass the quality rules in one front-end and fail
    /// them in the other. Scoring comes before the shape because the mask edge and the fill are
    /// not detail in the subject.
    pub fn finish_crop(
        &self,
        crop: DynamicImage,
        crop_settings: &CropSettings,
        enhancement: Option<&EnhancementSettings>,
        eyes: Option<&[RedEye]>,
    ) -> FinishedCrop {
        let enhanced = match enhancement {
            Some(settings) => self.enhance(&crop, settings, eyes),
            None => crop,
        };
        let (quality_score, quality) = estimate_sharpness(&enhanced);
        let shaped = self.apply_shape_mask(
            &enhanced,
            &crop_settings.shape,
            crop_settings.vignette_softness,
            crop_settings.vignette_intensity,
            crop_settings.vignette_color,
        );
        FinishedCrop {
            image: fill_transparent(shaped, crop_settings.fill_color),
            quality,
            quality_score,
        }
    }
}

/// A crop ready to save, with the sharpness it was judged by.
#[derive(Clone, Debug)]
pub struct FinishedCrop {
    /// Enhanced, shaped and filled.
    pub image: DynamicImage,
    /// Sharpness band of the enhanced crop before the shape.
    pub quality: Quality,
    /// The Laplacian variance behind `quality`.
    pub quality_score: f64,
}

/// Composite `image` over `fill` ("over", straight alpha). A transparent fill keeps the shape's
/// transparency for PNG/WebP; an opaque one is what makes a shaped JPEG show its shape at all,
/// since the JPEG encoder drops alpha and would otherwise expose the pixels under the mask.
fn fill_transparent(image: DynamicImage, fill: RgbaColor) -> DynamicImage {
    let mut rgba = image.into_rgba8();
    let bg_a = fill.alpha as f32 / 255.0;
    let bg = [fill.red as f32, fill.green as f32, fill.blue as f32];
    for px in rgba.pixels_mut() {
        let src_a = px[3] as f32 / 255.0;
        if src_a >= 1.0 {
            continue;
        }
        let out_a = src_a + bg_a * (1.0 - src_a);
        if out_a <= 0.0 {
            px.0 = [0, 0, 0, 0];
            continue;
        }
        for c in 0..3 {
            px[c] = ((px[c] as f32 * src_a + bg[c] * bg_a * (1.0 - src_a)) / out_a).round() as u8;
        }
        px[3] = (out_a * 255.0).round() as u8;
    }
    DynamicImage::ImageRgba8(rgba)
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

    #[test]
    fn debug_names_the_backend() {
        assert_eq!(
            format!("{:?}", EnhancementRuntime::cpu_only()),
            r#"EnhancementRuntime { backend: "cpu" }"#
        );
    }

    /// With a GPU context the runtime names the adapter it runs on; the CPU case is above. The
    /// shared test context, never a new device.
    #[test]
    fn a_gpu_runtime_names_its_adapter() {
        let Some(context) = crate::gpu::test_support::test_context() else {
            return;
        };
        let expected = context.adapter_info().name.clone();
        let runtime = EnhancementRuntime::new(Some(context));
        assert_eq!(runtime.adapter_name(), Some(expected));
    }

    /// The CPU shape mask is the pipeline's, applied to a copy.
    #[test]
    fn the_cpu_shape_mask_matches_the_pipeline() {
        let runtime = EnhancementRuntime::cpu_only();
        let image = DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
            12,
            9,
            image::Rgba([90, 120, 150, 255]),
        ));
        let color = RgbaColor::opaque(10, 20, 30);
        let masked = runtime.apply_shape_mask(&image, &CropShape::Ellipse, 0.3, 0.5, color);
        let mut expected = image.clone();
        apply_shape_mask_dynamic(&mut expected, &CropShape::Ellipse, 0.3, 0.5, color);
        assert_eq!(masked.to_rgba8(), expected.to_rgba8());
    }

    /// Scored before the shape, then shaped and composited over the fill: an ellipse's corners
    /// take an opaque fill, a transparent fill leaves them transparent, and the score is the
    /// unshaped crop's.
    #[test]
    fn finish_crop_scores_then_shapes_and_fills() {
        let runtime = EnhancementRuntime::cpu_only();
        let checker = DynamicImage::ImageRgba8(image::RgbaImage::from_fn(16, 16, |x, y| {
            let v = if (x + y) % 2 == 0 { 250 } else { 5 };
            image::Rgba([v, v, v, 255])
        }));
        let mut settings = CropSettings {
            shape: CropShape::Ellipse,
            fill_color: RgbaColor::opaque(200, 10, 50),
            ..CropSettings::default()
        };

        let filled = runtime.finish_crop(checker.clone(), &settings, None, None);
        assert_eq!(filled.quality_score, estimate_sharpness(&checker).0);
        assert_eq!(
            filled.image.to_rgba8().get_pixel(0, 0).0,
            [200, 10, 50, 255]
        );

        settings.fill_color = RgbaColor {
            alpha: 0,
            ..settings.fill_color
        };
        let clear = runtime
            .finish_crop(checker, &settings, None, None)
            .image
            .to_rgba8();
        assert_eq!(clear.get_pixel(0, 0)[3], 0);
        // The centre is inside the ellipse and must come through untouched.
        assert_eq!(clear.get_pixel(8, 8).0, [250, 250, 250, 255]);
    }

    /// Dense outlines -- a Koch rectangle at 4 iterations is 1,024 points, a 9-sided Koch
    /// polygon at 3 is 576 -- used to be cut to their first 512 points on the GPU, closing the
    /// shape along a diagonal. They now take the CPU mask, so the two paths agree exactly.
    #[test]
    fn dense_outlines_match_the_cpu_mask_on_the_gpu_path() {
        let Some(context) = crate::gpu::test_support::test_context() else {
            eprintln!("skipped: no GPU");
            return;
        };
        let runtime = EnhancementRuntime::new(Some(context));
        let image = DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
            96,
            96,
            image::Rgba([90, 120, 150, 255]),
        ));
        let color = RgbaColor::opaque(0, 0, 0);
        for shape in [
            CropShape::KochRectangle { iterations: 4 },
            CropShape::KochPolygon {
                sides: 9,
                rotation_deg: 0.0,
                iterations: 3,
            },
        ] {
            let mut expected = image.clone();
            apply_shape_mask_dynamic(&mut expected, &shape, 0.0, 1.0, color);
            let masked = runtime.apply_shape_mask(&image, &shape, 0.0, 1.0, color);
            assert_eq!(masked.to_rgba8(), expected.to_rgba8(), "{shape:?}");
        }
    }

    /// The CPU rasteriser at the densest Koch rectangle keeps the shape's 180-degree symmetry;
    /// a truncated outline covers one side and not the other.
    #[test]
    fn the_densest_koch_rectangle_is_symmetric() {
        let size = 128;
        let mut image = DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
            size,
            size,
            image::Rgba([255, 255, 255, 255]),
        ));
        let shape = CropShape::KochRectangle { iterations: 5 };
        apply_shape_mask_dynamic(&mut image, &shape, 0.0, 1.0, RgbaColor::opaque(0, 0, 0));
        let rgba = image.to_rgba8();
        let covered = |ys: std::ops::Range<u32>| {
            ys.flat_map(|y| (0..size).map(move |x| (x, y)))
                .filter(|&(x, y)| rgba.get_pixel(x, y)[3] > 127)
                .count() as f64
        };
        let (top, bottom) = (covered(0..size / 2), covered(size / 2..size));
        assert!(
            top > 0.0 && (top - bottom).abs() / top < 0.02,
            "{top} vs {bottom}"
        );
    }
}
