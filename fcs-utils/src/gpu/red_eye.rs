use std::sync::Arc;

use anyhow::{Context, Result};
use bytemuck::{bytes_of, cast_slice};
use image::{DynamicImage, RgbaImage};
use wgpu::util::DeviceExt;

use super::{GpuContext, RED_EYE_WGSL, pack_rgba_pixels, unpack_rgba_pixels};
use crate::{
    create_gpu_pipeline, gpu_readback, gpu_uniforms, storage_buffer_entry, uniform_buffer_entry,
};
use bytemuck::{Pod, Zeroable};

gpu_uniforms!(RedEyeUniforms, 3, {
    pixel_count: u32,
    width: u32, // Added width for coordinate calculation
    threshold: f32,
    min_red: f32,
    eye_count: u32,
});

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct RedEye {
    pub x: f32,
    pub y: f32,
    pub radius: f32,
    pub _pad: f32,
}

#[derive(Clone)]
pub struct GpuRedEyeRemoval {
    context: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuRedEyeRemoval {
    pub fn new(context: Arc<GpuContext>) -> Result<Self> {
        let device = context.device();

        let (pipeline, bind_group_layout) = create_gpu_pipeline!(
            device,
            "red_eye",
            RED_EYE_WGSL,
            [
                storage_buffer_entry!(0, read_write),
                uniform_buffer_entry!(1),
                storage_buffer_entry!(2, read_only), // Eyes buffer
            ]
        );

        Ok(Self {
            context,
            pipeline,
            bind_group_layout,
        })
    }

    pub fn apply(
        &self,
        image: &DynamicImage,
        threshold: f32,
        eyes: Option<&[RedEye]>,
    ) -> Result<DynamicImage> {
        let rgba = image.to_rgba8();
        let (width, height) = rgba.dimensions();
        let pixel_count = (width as usize) * (height as usize);
        if pixel_count == 0 {
            return Ok(image.clone());
        }
        let data_u32 = pack_rgba_pixels(rgba.as_raw());

        let device = self.context.device();
        let queue = self.context.queue();

        let buffer_size = (data_u32.len() * std::mem::size_of::<u32>()) as wgpu::BufferAddress;
        let storage = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("red_eye_storage"),
            contents: cast_slice(&data_u32),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("red_eye_readback"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let eyes_slice = eyes.unwrap_or(&[]);
        let (eyes_data, eye_count) = if eyes_slice.is_empty() {
            // Provide a dummy buffer if no eyes, but eye_count 0 prevents access
            (
                vec![RedEye {
                    x: 0.0,
                    y: 0.0,
                    radius: 0.0,
                    _pad: 0.0,
                }],
                0,
            )
        } else {
            (eyes_slice.to_vec(), eyes_slice.len() as u32)
        };

        let eyes_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("red_eye_locs"),
            contents: cast_slice(&eyes_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let uniforms = RedEyeUniforms {
            pixel_count: pixel_count as u32,
            width,
            threshold,
            min_red: 80.0,
            eye_count,
            __padding: [0; 3], // Adjusted padding for alignment (total size must by 16-byte aligned)
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("red_eye_uniforms"),
            contents: bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("red_eye_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: storage.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: eyes_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("red_eye_encoder"),
        });
        {
            let dispatch = div_ceil(pixel_count as u32, 256);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("red_eye_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&storage, 0, &readback, 0, buffer_size);
        queue.submit(std::iter::once(encoder.finish()));

        let out_pixels = gpu_readback!(readback, device, data_u32.len(), "red-eye removal")?;
        let out_bytes = unpack_rgba_pixels(&out_pixels);

        let image = RgbaImage::from_raw(width, height, out_bytes)
            .context("failed to build red-eye image")?;
        Ok(DynamicImage::ImageRgba8(image))
    }
}

fn div_ceil(value: u32, divisor: u32) -> u32 {
    value.div_ceil(divisor)
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
    fn div_ceil_basic() {
        assert_eq!(div_ceil(0, 256), 0);
        assert_eq!(div_ceil(256, 256), 1);
        assert_eq!(div_ceil(257, 256), 2);
        assert_eq!(div_ceil(1, 1), 1);
    }

    #[test]
    fn apply_zero_pixel_image_returns_clone() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping red_eye zero-pixel test: no GPU");
            return;
        };
        let remover = GpuRedEyeRemoval::new(ctx).expect("init");
        // 0×0 image has pixel_count == 0 → early return
        let image = DynamicImage::ImageRgba8(RgbaImage::new(0, 0));
        let result = remover.apply(&image, 0.3, None).expect("apply");
        assert_eq!(result.width(), 0);
        assert_eq!(result.height(), 0);
    }

    #[test]
    fn apply_with_no_eyes_preserves_low_red_pixels() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping red_eye no-eyes test: no GPU");
            return;
        };
        let remover = GpuRedEyeRemoval::new(ctx).expect("init");
        // Use a green pixel: r=0 < min_red(80) → shader returns early for every pixel.
        let image = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            16,
            16,
            image::Rgba([0, 200, 50, 255]),
        ));
        let result = remover.apply(&image, 0.3, None).expect("apply");
        assert_eq!(result.width(), 16);
        assert_eq!(result.height(), 16);
        let original = image.to_rgba8();
        let output = result.to_rgba8();
        assert_eq!(
            original.as_raw(),
            output.as_raw(),
            "Low-red pixels should not be modified"
        );
    }

    #[test]
    fn apply_with_eye_region_desaturates_red_channel() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping red_eye correction test: no GPU");
            return;
        };
        let remover = GpuRedEyeRemoval::new(ctx).expect("init");
        // Fill image with a strongly red pixel (classic red-eye color).
        let width = 16u32;
        let height = 16u32;
        let image = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            width,
            height,
            image::Rgba([220, 30, 30, 255]),
        ));
        let eye = RedEye {
            x: (width / 2) as f32,
            y: (height / 2) as f32,
            radius: (width / 2) as f32,
            _pad: 0.0,
        };
        let result = remover.apply(&image, 0.3, Some(&[eye])).expect("apply");
        assert_eq!(result.width(), width);
        assert_eq!(result.height(), height);
        // Center pixel should have its red channel reduced after correction.
        let output = result.to_rgba8();
        let center = output.get_pixel(width / 2, height / 2);
        assert!(
            center[0] < 220,
            "Red channel should be reduced by red-eye removal (got {})",
            center[0]
        );
    }
}
