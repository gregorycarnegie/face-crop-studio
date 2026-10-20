//! CUDA equivalents of the production WESL image passes.
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use image::{DynamicImage, RgbaImage};

use super::{
    EnhancementSettings, detail::apply_unsharp_with_preblur, red_eye::apply_red_eye_removal,
};
use crate::{
    color::RgbaColor,
    cuda::runtime::Runtime,
    gpu::{
        GpuPixelAdjust, RedEye, gaussian_blur::build_kernel, pack_rgba_pixels, unpack_rgba_pixels,
    },
    shape::{CropShape, outline_points_for_rect},
};

pub(super) struct CudaEnhancer {
    pub runtime: Arc<Runtime>,
}

fn config(len: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (len.div_ceil(128), 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    }
}

impl CudaEnhancer {
    pub fn new(runtime: Arc<Runtime>) -> Self {
        Self { runtime }
    }

    fn upload(&self, image: &DynamicImage) -> Result<(Arc<CudaStream>, CudaSlice<u32>, u32)> {
        let rgba = image.to_rgba8();
        let pixels = pack_rgba_pixels(rgba.as_raw());
        ensure!(pixels.len() <= i32::MAX as usize, "CUDA image is too large");
        ensure!(
            image.width() <= i32::MAX as u32 - 12 && image.height() <= i32::MAX as u32 - 12,
            "CUDA dimensions are too large"
        );
        let stream = self.runtime.context.new_stream()?;
        let data = stream.clone_htod(&pixels)?;
        Ok((stream, data, pixels.len() as u32))
    }

    fn download(
        stream: &Arc<CudaStream>,
        pixels: &CudaSlice<u32>,
        width: u32,
        height: u32,
    ) -> Result<DynamicImage> {
        let bytes = unpack_rgba_pixels(&stream.clone_dtoh(pixels)?);
        Ok(DynamicImage::ImageRgba8(
            RgbaImage::from_raw(width, height, bytes).context("invalid CUDA image size")?,
        ))
    }

    fn blur_pixels(
        &self,
        stream: &Arc<CudaStream>,
        pixels: &CudaSlice<u32>,
        width: u32,
        height: u32,
        radius: f32,
    ) -> Result<CudaSlice<u32>> {
        let radius = (radius.ceil() as i32).clamp(1, 12) as u32;
        let weights = stream.clone_htod(&build_kernel(radius))?;
        let mut temp = stream.alloc_zeros::<u32>((width * height) as usize)?;
        let mut output = stream.alloc_zeros::<u32>((width * height) as usize)?;
        // SAFETY: equal-sized disjoint images, validated dimensions, and 2*radius+1 weights.
        unsafe {
            stream
                .launch_builder(&self.runtime.blur)
                .arg(pixels)
                .arg(&mut temp)
                .arg(&weights)
                .arg(&width)
                .arg(&height)
                .arg(&radius)
                .arg(&0u32)
                .launch(config(width * height))?;
            stream
                .launch_builder(&self.runtime.blur)
                .arg(&temp)
                .arg(&mut output)
                .arg(&weights)
                .arg(&width)
                .arg(&height)
                .arg(&radius)
                .arg(&1u32)
                .launch(config(width * height))?;
        }
        Ok(output)
    }

    pub fn apply(
        &self,
        image: &DynamicImage,
        settings: &EnhancementSettings,
        eyes: Option<&[RedEye]>,
    ) -> Result<DynamicImage> {
        if image.width() == 0 || image.height() == 0 {
            return Ok(image.clone());
        }
        let amount = (settings.unsharp_amount + settings.sharpness).clamp(0.0, 2.0);
        if !settings.auto_color
            && !settings.red_eye_removal
            && !GpuPixelAdjust::needs_adjustment(settings)
            && settings.skin_smooth_amount <= 0.0
            && (amount <= 0.0 || settings.unsharp_radius <= 0.0)
            && (!settings.background_blur || settings.background_blur_radius <= 0.0)
        {
            return Ok(image.clone());
        }
        let (width, height) = (image.width(), image.height());
        let (stream, mut pixels, len) = self.upload(image)?;
        // SAFETY: upload validates dimensions; every allocation and launch stays on this stream.
        unsafe {
            if settings.auto_color {
                let mut histogram = stream.alloc_zeros::<u32>(768)?;
                let mut lut = stream.alloc_zeros::<u32>(768)?;
                stream
                    .launch_builder(&self.runtime.histogram)
                    .arg(&pixels)
                    .arg(&mut histogram)
                    .arg(&len)
                    .launch(config(len))?;
                stream
                    .launch_builder(&self.runtime.lut)
                    .arg(&histogram)
                    .arg(&mut lut)
                    .arg(&len)
                    .launch(config(3))?;
                stream
                    .launch_builder(&self.runtime.equalize)
                    .arg(&mut pixels)
                    .arg(&lut)
                    .arg(&len)
                    .launch(config(len))?;
            }
            // Red-eye statistics and sharpening recombination stay on the same CPU helpers as WESL.
            if settings.red_eye_removal {
                let image = Self::download(&stream, &pixels, width, height)?;
                let image = apply_red_eye_removal(&image, settings.red_eye_threshold, eyes);
                pixels = stream.clone_htod(&pack_rgba_pixels(image.to_rgba8().as_raw()))?;
            }
            if GpuPixelAdjust::needs_adjustment(settings) {
                let flags = u32::from(settings.exposure_stops.abs() >= 1e-6)
                    | (u32::from(settings.brightness != 0) << 1)
                    | (u32::from((settings.contrast - 1.0).abs() >= 1e-6) << 2)
                    | (u32::from((settings.saturation - 1.0).abs() >= 1e-6) << 3);
                stream
                    .launch_builder(&self.runtime.pixel)
                    .arg(&mut pixels)
                    .arg(&len)
                    .arg(&2f32.powf(settings.exposure_stops))
                    .arg(&(settings.brightness as f32))
                    .arg(&settings.contrast.clamp(0.5, 2.0))
                    .arg(&settings.saturation.clamp(0.0, 2.5))
                    .arg(&flags)
                    .launch(config(len))?;
            }
            if settings.skin_smooth_amount > 0.0 {
                let mut output = stream.alloc_zeros::<u32>(len as usize)?;
                let radius =
                    ((settings.skin_smooth_sigma_space.max(0.1) * 2.0).ceil() as u32).clamp(1, 8);
                stream
                    .launch_builder(&self.runtime.bilateral)
                    .arg(&pixels)
                    .arg(&mut output)
                    .arg(&width)
                    .arg(&height)
                    .arg(&radius)
                    .arg(&settings.skin_smooth_sigma_space.max(0.1))
                    .arg(&settings.skin_smooth_sigma_color.max(0.1))
                    .arg(&settings.skin_smooth_amount.clamp(0.0, 1.0))
                    .launch(config(len))?;
                pixels = output;
            }
            if amount > 0.0 && settings.unsharp_radius > 0.0 {
                let blurred =
                    self.blur_pixels(&stream, &pixels, width, height, settings.unsharp_radius)?;
                let image = Self::download(&stream, &pixels, width, height)?;
                let blurred = Self::download(&stream, &blurred, width, height)?;
                let image = apply_unsharp_with_preblur(&image, &blurred, amount);
                pixels = stream.clone_htod(&pack_rgba_pixels(image.to_rgba8().as_raw()))?;
            }
            if settings.background_blur && settings.background_blur_radius > 0.0 {
                let blurred = self.blur_pixels(
                    &stream,
                    &pixels,
                    width,
                    height,
                    settings.background_blur_radius,
                )?;
                let mut output = stream.alloc_zeros::<u32>(len as usize)?;
                stream
                    .launch_builder(&self.runtime.background)
                    .arg(&pixels)
                    .arg(&blurred)
                    .arg(&mut output)
                    .arg(&width)
                    .arg(&height)
                    .arg(&settings.background_blur_mask_size.clamp(0.3, 1.0))
                    .launch(config(len))?;
                pixels = output;
            }
        }
        Self::download(&stream, &pixels, width, height)
    }

    pub fn apply_shape(
        &self,
        image: &DynamicImage,
        shape: &CropShape,
        softness: f32,
        intensity: f32,
        color: RgbaColor,
    ) -> Result<Option<DynamicImage>> {
        if matches!(shape, CropShape::Rectangle) || image.width() == 0 || image.height() == 0 {
            return Ok(None);
        }
        let (width, height) = (image.width(), image.height());
        let points = outline_points_for_rect(width as f32, height as f32, shape);
        if points.len() < 3 || points.len() > 512 {
            return Ok(None);
        }
        let data: Vec<_> = points.iter().flat_map(|&(x, y)| [x, y]).collect();
        let (stream, mut pixels, len) = self.upload(image)?;
        let polygon = stream.clone_htod(&data)?;
        let color = u32::from_le_bytes([color.red, color.green, color.blue, color.alpha]);
        // SAFETY: the polygon has at least three points and pixels matches the validated extent.
        unsafe {
            stream
                .launch_builder(&self.runtime.shape)
                .arg(&mut pixels)
                .arg(&polygon)
                .arg(&width)
                .arg(&height)
                .arg(&(points.len() as u32))
                .arg(&softness)
                .arg(&intensity)
                .arg(&color)
                .launch(config(len))?;
        }
        Ok(Some(Self::download(&stream, &pixels, width, height)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enhance::WgpuEnhancer;

    fn compare(actual: &DynamicImage, expected: &DynamicImage, tolerance: u8, label: &str) {
        assert_eq!(
            (actual.width(), actual.height()),
            (expected.width(), expected.height())
        );
        let error = actual
            .to_rgba8()
            .as_raw()
            .iter()
            .zip(expected.to_rgba8().as_raw())
            .map(|(&a, &b)| a.abs_diff(b))
            .max()
            .unwrap_or(0);
        assert!(
            error <= tolerance,
            "{label}: max channel error {error} > {tolerance}"
        );
    }

    #[test]
    #[ignore = "requires a compatible NVIDIA GPU and a WESL GPU adapter"]
    fn cuda_image_passes_match_wesl_and_support_concurrent_exports() -> Result<()> {
        let context = crate::gpu::test_support::test_context().context("WESL GPU required")?;
        let wesl = WgpuEnhancer::new(context)?;
        let neutral = EnhancementSettings {
            unsharp_amount: 0.0,
            sharpness: 0.0,
            ..Default::default()
        };
        let cases = [
            ("neutral", neutral.clone()),
            (
                "histogram",
                EnhancementSettings {
                    auto_color: true,
                    ..neutral.clone()
                },
            ),
            (
                "tone",
                EnhancementSettings {
                    exposure_stops: 0.4,
                    brightness: -13,
                    contrast: 1.2,
                    saturation: 0.7,
                    ..neutral.clone()
                },
            ),
            (
                "bilateral",
                EnhancementSettings {
                    skin_smooth_amount: 0.7,
                    skin_smooth_sigma_space: 2.5,
                    skin_smooth_sigma_color: 35.0,
                    ..neutral.clone()
                },
            ),
            (
                "unsharp",
                EnhancementSettings {
                    unsharp_amount: 0.6,
                    unsharp_radius: 2.5,
                    ..neutral.clone()
                },
            ),
            (
                "background",
                EnhancementSettings {
                    background_blur: true,
                    background_blur_radius: 3.2,
                    background_blur_mask_size: 0.65,
                    ..neutral.clone()
                },
            ),
            (
                "combined",
                EnhancementSettings {
                    auto_color: true,
                    exposure_stops: -0.3,
                    brightness: 8,
                    contrast: 1.1,
                    saturation: 1.2,
                    skin_smooth_amount: 0.4,
                    unsharp_amount: 0.5,
                    background_blur: true,
                    red_eye_removal: true,
                    ..neutral.clone()
                },
            ),
        ];
        let shapes = [
            CropShape::Ellipse,
            CropShape::RoundedRectangle { radius_pct: 0.25 },
            CropShape::ChamferedRectangle { size_pct: 0.2 },
            CropShape::Star {
                points: 5,
                inner_radius_pct: 0.4,
                rotation_deg: 13.0,
            },
            CropShape::Polygon {
                sides: 7,
                rotation_deg: 27.0,
                corner_style: crate::shape::PolygonCornerStyle::Rounded { radius_pct: 0.1 },
            },
            CropShape::KochRectangle { iterations: 3 },
            CropShape::Rectangle,
            CropShape::KochRectangle { iterations: 4 },
        ];
        for ptx in [
            include_str!("../cuda/windows.ptx"),
            include_str!("../cuda/linux.ptx"),
        ] {
            let cuda = CudaEnhancer::new(Arc::new(Runtime::from_ptx(ptx)?));
            let sixteen = DynamicImage::ImageLuma16(image::ImageBuffer::from_pixel(
                3,
                2,
                image::Luma([40000u16]),
            ));
            let unchanged = cuda.apply(&sixteen, &neutral, None)?;
            assert_eq!(unchanged.color(), sixteen.color());
            assert_eq!(unchanged.as_bytes(), sixteen.as_bytes());
            for (width, height) in [(1, 1), (1, 7), (17, 13), (96, 65)] {
                let image = DynamicImage::ImageRgba8(RgbaImage::from_fn(width, height, |x, y| {
                    image::Rgba([
                        ((x * 17 + y * 7) % 256) as u8,
                        ((x * 3 + y * 23) % 256) as u8,
                        ((x * 11 + y * 13) % 256) as u8,
                        ((x * 31 + y * 19) % 256) as u8,
                    ])
                }));
                for (label, settings) in &cases {
                    let expected = wesl.apply(&image, settings, None)?;
                    let actual = cuda.apply(&image, settings, None)?;
                    compare(
                        &actual,
                        &expected,
                        if *label == "combined" { 4 } else { 1 },
                        label,
                    );
                }
                for shape in &shapes {
                    for (softness, intensity) in [(0.0, 0.0), (0.3, 0.7)] {
                        let expected = wesl.apply_shape_mask_gpu(
                            &image,
                            shape,
                            softness,
                            intensity,
                            RgbaColor::opaque(20, 50, 90),
                        )?;
                        let actual = cuda.apply_shape(
                            &image,
                            shape,
                            softness,
                            intensity,
                            RgbaColor::opaque(20, 50, 90),
                        )?;
                        match (actual, expected) {
                            (Some(a), Some(b)) => compare(
                                &a,
                                &b,
                                1,
                                &format!(
                                    "shape {shape:?}, {width}x{height}, softness={softness}, intensity={intensity}"
                                ),
                            ),
                            (None, None) => {}
                            _ => panic!("CUDA/WESL shape fallback differs"),
                        }
                    }
                }
            }
            let image = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
                33,
                19,
                image::Rgba([230, 30, 40, 128]),
            ));
            let settings = &cases.last().unwrap().1;
            let expected = cuda.apply(&image, settings, None)?;
            std::thread::scope(|scope| -> Result<()> {
                let first = scope.spawn(|| cuda.apply(&image, settings, None));
                let second = scope.spawn(|| cuda.apply(&image, settings, None));
                compare(&first.join().unwrap()?, &expected, 0, "concurrent export");
                compare(&second.join().unwrap()?, &expected, 0, "concurrent export");
                Ok(())
            })?;
            assert_eq!(
                cuda.apply(&DynamicImage::new_rgba8(0, 0), &neutral, None)?
                    .width(),
                0
            );
        }
        Ok(())
    }
}
