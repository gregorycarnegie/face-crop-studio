use std::sync::Arc;

use anyhow::{Context, Result};
use bytemuck::{bytes_of, cast_slice};
use image::{DynamicImage, RgbaImage};
use wgpu::util::DeviceExt;

use super::{
    BACKGROUND_BLUR_WGSL, GpuBufferPool, GpuContext, pack_rgba_pixels, unpack_rgba_pixels,
};
use crate::{
    create_gpu_pipeline, gpu_readback, gpu_uniforms, storage_buffer_entry, uniform_buffer_entry,
};

gpu_uniforms!(BackgroundBlurUniforms, 1, {
    width: u32,
    height: u32,
    mask_size: f32,
});

#[derive(Clone)]
pub struct GpuBackgroundBlur {
    context: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    buffer_pool: Arc<GpuBufferPool>,
}

impl GpuBackgroundBlur {
    pub fn new(context: Arc<GpuContext>) -> Result<Self> {
        let device = context.device();

        let (pipeline, bind_group_layout) = create_gpu_pipeline!(
            device,
            "background_blur",
            BACKGROUND_BLUR_WGSL,
            [
                storage_buffer_entry!(0, read_only),
                storage_buffer_entry!(1, read_only),
                storage_buffer_entry!(2, read_write),
                uniform_buffer_entry!(3),
            ]
        );

        let buffer_pool = Arc::new(GpuBufferPool::new(context.clone(), None));

        Ok(Self {
            context,
            pipeline,
            bind_group_layout,
            buffer_pool,
        })
    }

    pub fn blend(
        &self,
        sharp: &DynamicImage,
        blurred: &DynamicImage,
        mask_size: f32,
    ) -> Result<DynamicImage> {
        let sharp_rgba = sharp.to_rgba8();
        let blur_rgba = blurred.to_rgba8();
        let (width, height) = sharp_rgba.dimensions();

        if blur_rgba.dimensions() != (width, height) {
            anyhow::bail!("background blur images must have matching dimensions");
        }

        let mask_size = mask_size.clamp(0.3, 1.0);
        let sharp_u32 = pack_rgba_pixels(sharp_rgba.as_raw());
        let blur_u32 = pack_rgba_pixels(blur_rgba.as_raw());

        let device = self.context.device();
        let queue = self.context.queue();
        let buffer_size = (sharp_u32.len() * std::mem::size_of::<u32>()) as wgpu::BufferAddress;

        // Acquire buffers from pool
        let sharp_buffer = self.buffer_pool.acquire(
            buffer_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            Some("background_blur_sharp"),
        )?;
        let blur_buffer = self.buffer_pool.acquire(
            buffer_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            Some("background_blur_blur"),
        )?;
        let output_buffer = self.buffer_pool.acquire(
            buffer_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            Some("background_blur_output"),
        )?;
        let readback = self.buffer_pool.acquire(
            buffer_size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            Some("background_blur_readback"),
        )?;

        // Upload input data
        queue.write_buffer(&sharp_buffer, 0, cast_slice(&sharp_u32));
        queue.write_buffer(&blur_buffer, 0, cast_slice(&blur_u32));

        let uniforms = BackgroundBlurUniforms {
            width,
            height,
            mask_size,
            __padding: [0; 1],
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("background_blur_uniforms"),
            contents: bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("background_blur_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sharp_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: blur_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("background_blur_encoder"),
        });
        {
            let workgroups_x = width.div_ceil(16);
            let workgroups_y = height.div_ceil(16);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("background_blur_pass"),
                timestamp_writes: self.context.timestamp_writes("background_blur"),
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, buffer_size);
        queue.submit(std::iter::once(encoder.finish()));

        let out_pixels = gpu_readback!(readback, device, sharp_u32.len(), "background blur")?;
        let out_bytes = unpack_rgba_pixels(&out_pixels);

        // Recycle buffers back to pool
        self.buffer_pool.recycle(
            sharp_buffer,
            buffer_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        self.buffer_pool.recycle(
            blur_buffer,
            buffer_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        self.buffer_pool.recycle(
            output_buffer,
            buffer_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        self.buffer_pool.recycle(
            readback,
            buffer_size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        let image = RgbaImage::from_raw(width, height, out_bytes)
            .context("failed to build blurred image")?;
        Ok(DynamicImage::ImageRgba8(image))
    }

    /// Clear pooled buffers to free up GPU memory.
    pub fn clear_cache(&self) {
        self.buffer_pool.clear();
    }

    /// Returns the estimated size in bytes of pooled buffers.
    pub fn memory_usage(&self) -> u64 {
        self.buffer_pool.memory_usage()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::{GpuAvailability, GpuContextOptions};
    use image::RgbaImage;

    fn test_context() -> Option<Arc<GpuContext>> {
        match GpuContext::init_with_fallback(&GpuContextOptions::default()) {
            GpuAvailability::Available(ctx) => Some(ctx),
            _ => None,
        }
    }

    #[test]
    fn clear_cache_and_memory_usage() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping background_blur cache test: no GPU");
            return;
        };
        let blurrer = GpuBackgroundBlur::new(ctx).expect("init");
        // Run blend to populate pool
        let img = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            8,
            8,
            image::Rgba([100, 100, 100, 255]),
        ));
        blurrer.blend(&img, &img, 0.5).expect("blend");
        // Verify accessors don't panic
        blurrer.clear_cache();
        let _ = blurrer.memory_usage();
    }

    #[test]
    fn blend_rejects_mismatched_dimensions() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping background_blur dimension test: no GPU");
            return;
        };
        let blurrer = GpuBackgroundBlur::new(ctx).expect("init");
        let sharp = DynamicImage::ImageRgba8(RgbaImage::new(4, 4));
        let blurred = DynamicImage::ImageRgba8(RgbaImage::new(8, 8));
        assert!(blurrer.blend(&sharp, &blurred, 0.5).is_err());
    }

    // ------------------------------------------------------------------
    // Until these existed the only `blend` coverage was the error path above,
    // so the successful path — its buffer usages, uniforms and dispatch — never
    // ran, and the whole operation could return a blank image unnoticed.
    // ------------------------------------------------------------------

    use crate::gpu::test_support::assert_plausible_output;

    #[test]
    fn blend_keeps_the_centre_sharp_and_the_edges_blurred() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping background_blur blend test: no GPU");
            return;
        };
        let blurrer = GpuBackgroundBlur::new(ctx).expect("init");

        // Two flat but very different images, so every output pixel must be a
        // mix of two known values: the centre should favour `sharp`, the corner
        // `blurred`. That pins the mask orientation without modelling the ellipse.
        let sharp =
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(33, 33, image::Rgba([255, 0, 0, 255])));
        let blurred =
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(33, 33, image::Rgba([0, 0, 255, 255])));

        let result = blurrer.blend(&sharp, &blurred, 0.6).expect("blend");
        assert_plausible_output(&result, &sharp, "blend");

        let out = result.to_rgba8();
        let centre = out.get_pixel(16, 16);
        let corner = out.get_pixel(0, 0);
        assert!(
            centre[0] > centre[2],
            "centre should favour the sharp image, got {centre:?}"
        );
        assert!(
            corner[2] > corner[0],
            "corner should favour the blurred image, got {corner:?}"
        );
    }

    #[test]
    fn blend_with_identical_inputs_returns_that_image() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping background_blur blend test: no GPU");
            return;
        };
        let blurrer = GpuBackgroundBlur::new(ctx).expect("init");

        // Blending an image with itself has one correct answer whatever the mask
        // does, which exposes interpolation errors with no geometry to reason
        // about. Odd size so a mis-rounded dispatch leaves a tail behind.
        let image = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            27,
            13,
            image::Rgba([90, 140, 60, 255]),
        ));
        let result = blurrer.blend(&image, &image, 0.5).expect("blend");
        assert_plausible_output(&result, &image, "blend (identical inputs)");

        for px in result.to_rgba8().pixels() {
            for (ch, expected) in [90i16, 140, 60].iter().enumerate() {
                assert!(
                    (px[ch] as i16 - expected).abs() <= 1,
                    "expected ~{expected}, got {}",
                    px[ch]
                );
            }
        }
    }

    #[test]
    fn blend_clamps_the_mask_size_rather_than_failing() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping background_blur blend test: no GPU");
            return;
        };
        let blurrer = GpuBackgroundBlur::new(ctx).expect("init");
        let sharp = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            12,
            12,
            image::Rgba([200, 200, 200, 255]),
        ));
        let blurred = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            12,
            12,
            image::Rgba([20, 20, 20, 255]),
        ));

        // Out of range on both sides: mask_size clamps to 0.3 / 1.0.
        for mask in [-5.0f32, 0.0, 1.0, 99.0] {
            let result = blurrer.blend(&sharp, &blurred, mask).expect("blend");
            assert_plausible_output(&result, &sharp, "blend (clamped mask)");
        }
    }
}
