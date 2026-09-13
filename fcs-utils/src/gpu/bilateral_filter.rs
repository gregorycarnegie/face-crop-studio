use std::sync::Arc;

use anyhow::{Context, Result};
use bytemuck::{bytes_of, cast_slice};
use image::{DynamicImage, RgbaImage};
use wgpu::util::DeviceExt;

use super::{
    BILATERAL_FILTER_WGSL, GpuBufferPool, GpuContext, pack_rgba_pixels, unpack_rgba_pixels,
};
use crate::{
    create_gpu_pipeline, gpu_readback, gpu_uniforms, storage_buffer_entry, uniform_buffer_entry,
};

const MAX_RADIUS: u32 = 8;

gpu_uniforms!(BilateralUniforms, 2, {
    width: u32,
    height: u32,
    radius: u32,
    sigma_space: f32,
    sigma_color: f32,
    amount: f32,
});

/// Pixels sampled either side of the centre: two standard deviations of the (floored) spatial
/// sigma, at least one and at most [`MAX_RADIUS`], the WGSL loop's own bound.
fn sampling_radius(sigma_space: f32) -> u32 {
    ((sigma_space.max(0.1) * 2.0).ceil() as u32).clamp(1, MAX_RADIUS)
}

/// Reusable GPU bilateral filter for smoothing while preserving color edges.
#[derive(Clone)]
pub struct GpuBilateralFilter {
    context: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    pool: Arc<GpuBufferPool>,
}

impl GpuBilateralFilter {
    /// Create the bilateral-filter pipeline and pool on an existing GPU context.
    pub fn new(context: Arc<GpuContext>) -> Result<Self> {
        let device = context.device();

        let (pipeline, bind_group_layout) = create_gpu_pipeline!(
            device,
            "bilateral_filter",
            BILATERAL_FILTER_WGSL,
            [
                storage_buffer_entry!(0, read_only),
                storage_buffer_entry!(1, read_write),
                uniform_buffer_entry!(2),
            ]
        );

        let pool = Arc::new(GpuBufferPool::new(context.clone(), None));

        Ok(Self {
            context,
            pipeline,
            bind_group_layout,
            pool,
        })
    }

    /// Clear pooled buffers to free up GPU memory.
    pub fn clear_cache(&self) {
        self.pool.clear();
    }

    /// Returns the estimated size in bytes of pooled buffers.
    pub fn memory_usage(&self) -> u64 {
        self.pool.memory_usage()
    }

    /// Blend bilateral smoothing with the original image.
    ///
    /// `amount` is clamped to 0..=1; zero returns an unchanged clone.
    /// `sigma_space` is measured in pixels and `sigma_color` in 8-bit RGB channel
    /// units; both are at least 0.1. The sampling radius is capped at 8 pixels.
    /// Returns RGBA8 or an error on buffer allocation or readback failure.
    pub fn smooth(
        &self,
        image: &DynamicImage,
        amount: f32,
        sigma_space: f32,
        sigma_color: f32,
    ) -> Result<DynamicImage> {
        let amount = amount.clamp(0.0, 1.0);
        if amount <= 0.0 {
            return Ok(image.clone());
        }

        let radius = sampling_radius(sigma_space);
        let sigma_space = sigma_space.max(0.1);
        let sigma_color = sigma_color.max(0.1);

        let rgba = image.to_rgba8();
        let (width, height) = rgba.dimensions();
        let data_u32 = pack_rgba_pixels(rgba.as_raw());

        let device = self.context.device();
        let queue = self.context.queue();
        let buffer_size = (data_u32.len() * std::mem::size_of::<u32>()) as wgpu::BufferAddress;

        let storage_usage = super::buffer_pool::STORAGE_RW;
        let readback_usage = super::buffer_pool::READBACK;

        let input_buffer =
            self.pool
                .acquire(buffer_size, storage_usage, Some("bilateral_filter_input"))?;
        queue.write_buffer(&input_buffer, 0, cast_slice(&data_u32));
        let output_buffer =
            self.pool
                .acquire(buffer_size, storage_usage, Some("bilateral_filter_output"))?;
        let readback = self.pool.acquire(
            buffer_size,
            readback_usage,
            Some("bilateral_filter_readback"),
        )?;

        let uniforms = BilateralUniforms {
            width,
            height,
            radius,
            sigma_space,
            sigma_color,
            amount,
            __padding: [0; 2],
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bilateral_filter_uniforms"),
            contents: bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bilateral_filter_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bilateral_filter_encoder"),
        });
        {
            let workgroups_x = width.div_ceil(8);
            let workgroups_y = height.div_ceil(8);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bilateral_filter_pass"),
                timestamp_writes: self.context.timestamp_writes("bilateral_filter"),
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, buffer_size);
        queue.submit(std::iter::once(encoder.finish()));

        let out_pixels = gpu_readback!(readback, device, data_u32.len(), "bilateral filter")?;
        let out_bytes = unpack_rgba_pixels(&out_pixels);

        self.pool.recycle(input_buffer, buffer_size, storage_usage);
        self.pool.recycle(output_buffer, buffer_size, storage_usage);
        self.pool.recycle(readback, buffer_size, readback_usage);

        let image = RgbaImage::from_raw(width, height, out_bytes)
            .context("failed to build smoothed image")?;
        Ok(DynamicImage::ImageRgba8(image))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbaImage;

    use crate::gpu::test_support::test_context;

    /// Output parity cannot see a radius off by one ring: the outermost weights are small, and
    /// flat test images make the radius irrelevant. So pin the radius itself.
    #[test]
    fn the_sampling_radius_is_two_sigmas_clamped_to_the_shader_bound() {
        assert_eq!([0.0, 1.2, 3.0, 9.0].map(sampling_radius), [1, 3, 6, 8]);
    }

    #[test]
    fn smooth_zero_amount_returns_clone() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping bilateral_filter test: no GPU");
            return;
        };
        let filter = GpuBilateralFilter::new(ctx).expect("init");
        let image = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            4,
            4,
            image::Rgba([100, 150, 200, 255]),
        ));
        let result = filter.smooth(&image, 0.0, 3.0, 50.0).expect("smooth");
        assert_eq!(result.to_rgba8().as_raw(), image.to_rgba8().as_raw());
    }

    #[test]
    fn smooth_amount_above_one_is_clamped_and_succeeds() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping bilateral_filter test: no GPU");
            return;
        };
        let filter = GpuBilateralFilter::new(ctx).expect("init");
        let image = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            4,
            4,
            image::Rgba([128, 128, 128, 255]),
        ));
        // amount=2.0 clamps to 1.0. "Does not error" was the whole assertion
        // here, which let the entire operation be replaced by a blank image.
        let result = filter
            .smooth(&image, 2.0, 3.0, 50.0)
            .expect("smooth with clamped amount");
        crate::gpu::test_support::assert_plausible_output(
            &result,
            &image,
            "smooth (clamped amount)",
        );
    }

    #[test]
    fn smooth_small_sigma_is_clamped_and_succeeds() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping bilateral_filter test: no GPU");
            return;
        };
        let filter = GpuBilateralFilter::new(ctx).expect("init");
        let image = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            4,
            4,
            image::Rgba([64, 128, 192, 255]),
        ));
        // sigma_space=0.0 and sigma_color=0.0 are clamped to 0.1. A flat input
        // is preserved by a smoothing filter, so the exact output is knowable
        // here rather than merely "no error".
        let result = filter
            .smooth(&image, 0.5, 0.0, 0.0)
            .expect("smooth with clamped sigmas");
        crate::gpu::test_support::assert_plausible_output(
            &result,
            &image,
            "smooth (clamped sigmas)",
        );
        assert_eq!(
            result.to_rgba8().as_raw(),
            image.to_rgba8().as_raw(),
            "smoothing a uniform image must leave it unchanged"
        );
    }

    #[test]
    fn smooth_nonzero_amount_executes_gpu_path() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping bilateral_filter test: no GPU");
            return;
        };
        let filter = GpuBilateralFilter::new(ctx).expect("init");
        let image = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            8,
            8,
            image::Rgba([200, 100, 50, 255]),
        ));
        let result = filter.smooth(&image, 0.5, 3.0, 50.0).expect("smooth");
        let (w, h) = result.to_rgba8().dimensions();
        assert_eq!((w, h), (8, 8));
    }

    #[test]
    fn clear_cache_and_memory_usage() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping bilateral_filter test: no GPU");
            return;
        };
        let filter = GpuBilateralFilter::new(ctx).expect("init");
        let image = crate::gpu::test_support::gradient_image(8, 8);
        filter.smooth(&image, 0.5, 3.0, 50.0).expect("smooth");
        assert!(filter.memory_usage() > 0, "a pass pools its buffers");
        filter.clear_cache();
        assert_eq!(filter.memory_usage(), 0);
    }
}
