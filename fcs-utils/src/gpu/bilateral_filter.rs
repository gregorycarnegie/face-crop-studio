use std::sync::Arc;

use anyhow::{Context, Result};
use bytemuck::{bytes_of, cast_slice};
use image::{DynamicImage, RgbaImage};
use wgpu::util::DeviceExt;

use super::{BILATERAL_FILTER_WGSL, GpuContext, pack_rgba_pixels, unpack_rgba_pixels};
use crate::{
    create_gpu_pipeline, gpu_readback, gpu_uniforms, storage_buffer_entry, uniform_buffer_entry,
};

const MAX_RADIUS: u32 = 8;

gpu_uniforms!(BilateralUniforms, 1, {
    width: u32,
    height: u32,
    radius: u32,
    pixel_count: u32,
    sigma_space: f32,
    sigma_color: f32,
    amount: f32,
});

#[derive(Clone)]
pub struct GpuBilateralFilter {
    context: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuBilateralFilter {
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

        Ok(Self {
            context,
            pipeline,
            bind_group_layout,
        })
    }

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

        let sigma_space = sigma_space.max(0.1);
        let sigma_color = sigma_color.max(0.1);
        let radius = ((sigma_space * 2.0).ceil() as u32).clamp(1, MAX_RADIUS);

        let rgba = image.to_rgba8();
        let (width, height) = rgba.dimensions();
        let data_u32 = pack_rgba_pixels(rgba.as_raw());

        let device = self.context.device();
        let queue = self.context.queue();
        let buffer_size = (data_u32.len() * std::mem::size_of::<u32>()) as wgpu::BufferAddress;

        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bilateral_filter_input"),
            contents: cast_slice(&data_u32),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bilateral_filter_output"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bilateral_filter_readback"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniforms = BilateralUniforms {
            width,
            height,
            radius,
            pixel_count: width * height,
            sigma_space,
            sigma_color,
            amount,
            __padding: [0; 1],
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
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, buffer_size);
        queue.submit(std::iter::once(encoder.finish()));

        let out_pixels = gpu_readback!(readback, device, data_u32.len(), "bilateral filter")?;
        let out_bytes = unpack_rgba_pixels(&out_pixels);

        let image = RgbaImage::from_raw(width, height, out_bytes)
            .context("failed to build smoothed image")?;
        Ok(DynamicImage::ImageRgba8(image))
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
}
