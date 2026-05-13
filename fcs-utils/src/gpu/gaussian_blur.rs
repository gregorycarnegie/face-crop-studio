use std::sync::Arc;

use anyhow::{Context, Result};
use bytemuck::{bytes_of, cast_slice};
use image::{DynamicImage, RgbaImage};
use wgpu::util::DeviceExt;

use super::{GAUSSIAN_BLUR_WGSL, GpuBufferPool, GpuContext, pack_rgba_pixels, unpack_rgba_pixels};
use crate::{
    create_gpu_pipeline, gpu_readback, gpu_uniforms, storage_buffer_entry, uniform_buffer_entry,
};

const MAX_RADIUS: u32 = 12;
const MAX_KERNEL_SIZE: usize = (MAX_RADIUS as usize * 2) + 1;

gpu_uniforms!(BlurUniforms, 0, {
    width: u32,
    height: u32,
    radius: u32,
    direction: u32,
});

#[derive(Clone)]
pub struct GpuGaussianBlur {
    context: Arc<GpuContext>,
    pool: Arc<GpuBufferPool>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuGaussianBlur {
    pub fn new(context: Arc<GpuContext>) -> Result<Self> {
        let device = context.device();

        let (pipeline, bind_group_layout) = create_gpu_pipeline!(
            device,
            "gaussian_blur",
            GAUSSIAN_BLUR_WGSL,
            [
                storage_buffer_entry!(0, read_only),
                storage_buffer_entry!(1, read_write),
                uniform_buffer_entry!(2),
                storage_buffer_entry!(3, read_only),
            ]
        );

        let pool = Arc::new(GpuBufferPool::new(context.clone(), None));

        Ok(Self {
            context,
            pool,
            pipeline,
            bind_group_layout,
        })
    }

    pub fn blur(&self, image: &DynamicImage, radius: f32) -> Result<DynamicImage> {
        let radius = radius.ceil() as i32;
        if radius <= 0 {
            return Ok(image.clone());
        }
        let radius = radius.clamp(1, MAX_RADIUS as i32) as u32;
        let weights = build_kernel(radius);

        let rgba = image.to_rgba8();
        let (width, height) = rgba.dimensions();
        let data_u32 = pack_rgba_pixels(rgba.as_raw());

        let device = self.context.device();
        let queue = self.context.queue();

        let buffer_size = (data_u32.len() * std::mem::size_of::<u32>()) as wgpu::BufferAddress;

        // Acquire buffers from pool
        let input_buffer = self.pool.acquire(
            buffer_size,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            Some("gaussian_blur_input"),
        )?;
        queue.write_buffer(&input_buffer, 0, cast_slice(&data_u32));

        let weights_buffer = self.pool.acquire(
            (weights.len() * std::mem::size_of::<f32>()) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            Some("gaussian_blur_weights"),
        )?;
        queue.write_buffer(&weights_buffer, 0, cast_slice(&weights));

        let temp_buffer = self.pool.acquire(
            buffer_size,
            wgpu::BufferUsages::STORAGE,
            Some("gaussian_blur_temp"),
        )?;

        let readback = self.pool.acquire(
            buffer_size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            Some("gaussian_blur_readback"),
        )?;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gaussian_blur_encoder"),
        });

        // Horizontal pass
        let horizontal_uniforms = BlurUniforms {
            width,
            height,
            radius,
            direction: 0,
            __padding: [],
        };
        let horizontal_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gaussian_blur_uniform_horizontal"),
            contents: bytes_of(&horizontal_uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let horizontal_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gaussian_blur_bg_horizontal"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: temp_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: horizontal_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: weights_buffer.as_entire_binding(),
                },
            ],
        });
        dispatch_blur(&mut encoder, &self.pipeline, &horizontal_bg, width, height);

        // Vertical pass
        let vertical_uniforms = BlurUniforms {
            direction: 1,
            ..horizontal_uniforms
        };
        let vertical_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gaussian_blur_uniform_vertical"),
            contents: bytes_of(&vertical_uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let vertical_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gaussian_blur_bg_vertical"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: temp_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: vertical_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: weights_buffer.as_entire_binding(),
                },
            ],
        });
        dispatch_blur(&mut encoder, &self.pipeline, &vertical_bg, width, height);

        encoder.copy_buffer_to_buffer(&input_buffer, 0, &readback, 0, buffer_size);
        queue.submit(std::iter::once(encoder.finish()));

        let out_pixels = gpu_readback!(readback, device, data_u32.len(), "gaussian blur")?;
        let out_bytes = unpack_rgba_pixels(&out_pixels);

        // Recycle buffers
        self.pool.recycle(
            input_buffer,
            buffer_size,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );
        self.pool.recycle(
            weights_buffer,
            (weights.len() * std::mem::size_of::<f32>()) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        self.pool
            .recycle(temp_buffer, buffer_size, wgpu::BufferUsages::STORAGE);
        self.pool.recycle(
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
        self.pool.clear();
    }

    /// Returns the estimated size in bytes of pooled buffers.
    pub fn memory_usage(&self) -> u64 {
        self.pool.memory_usage()
    }
}

fn dispatch_blur(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    width: u32,
    height: u32,
) {
    let workgroups_x = width.div_ceil(16);
    let workgroups_y = height.div_ceil(16);
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("gaussian_blur_pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
}

fn build_kernel(radius: u32) -> [f32; MAX_KERNEL_SIZE] {
    let sigma = (radius.max(1) as f32) * 0.5 + 0.5;
    let kernel_size = radius * 2 + 1;
    let mut weights = [0.0f32; MAX_KERNEL_SIZE];
    let mut sum = 0.0;
    for i in 0..kernel_size {
        let distance = i as i32 - radius as i32;
        let weight = gaussian(distance as f32, sigma);
        weights[i as usize] = weight;
        sum += weight;
    }
    if sum > 0.0 {
        for weight in weights.iter_mut().take(kernel_size as usize) {
            *weight /= sum;
        }
    }
    weights
}

fn gaussian(distance: f32, sigma: f32) -> f32 {
    let two_sigma_sq = 2.0 * sigma * sigma;
    (-distance * distance / two_sigma_sq).exp()
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
    fn build_kernel_sums_to_one() {
        for radius in [1u32, 3, 5, 12] {
            let weights = build_kernel(radius);
            let kernel_size = (radius * 2 + 1) as usize;
            let sum: f32 = weights[..kernel_size].iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "kernel radius {radius} sum = {sum}, expected ~1.0"
            );
        }
    }

    #[test]
    fn blur_zero_radius_returns_original() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping gaussian_blur zero-radius test: no GPU");
            return;
        };
        let blurrer = GpuGaussianBlur::new(ctx).expect("init");
        let image = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            4,
            4,
            image::Rgba([50, 100, 150, 255]),
        ));
        let result = blurrer.blur(&image, 0.0).expect("blur");
        assert_eq!(result.to_rgba8().as_raw(), image.to_rgba8().as_raw());
    }

    #[test]
    fn clear_cache_and_memory_usage() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping gaussian_blur cache test: no GPU");
            return;
        };
        let blurrer = GpuGaussianBlur::new(ctx).expect("init");
        // Run a blur to populate the pool
        let image =
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(8, 8, image::Rgba([1, 2, 3, 255])));
        blurrer.blur(&image, 1.0).expect("blur");
        // clear_cache should not panic
        blurrer.clear_cache();
        // memory_usage is u64, no specific value required
        let _ = blurrer.memory_usage();
    }
}
