use std::sync::Arc;

use anyhow::{Context, Result};
use bytemuck::{bytes_of, cast_slice};
use image::{DynamicImage, RgbaImage};
use wgpu::util::DeviceExt;

use crate::enhance::EnhancementSettings;
use crate::{
    create_gpu_pipeline, gpu_readback, gpu_uniforms, storage_buffer_entry, uniform_buffer_entry,
};

use super::{GpuContext, PIXEL_ADJUST_WGSL, pack_rgba_pixels, unpack_rgba_pixels};

const EPSILON: f32 = 1e-6;
const FLAG_EXPOSURE: u32 = 1 << 0;
const FLAG_BRIGHTNESS: u32 = 1 << 1;
const FLAG_CONTRAST: u32 = 1 << 2;
const FLAG_SATURATION: u32 = 1 << 3;

gpu_uniforms!(PixelAdjustUniforms, 2, {
    exposure_multiplier: f32,
    brightness_offset: f32,
    contrast_factor: f32,
    saturation: f32,
    pixel_count: u32,
    flags: u32,
});

#[derive(Clone)]
pub struct GpuPixelAdjust {
    context: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuPixelAdjust {
    pub fn new(context: Arc<GpuContext>) -> Result<Self> {
        let device = context.device();

        let (pipeline, bind_group_layout) = create_gpu_pipeline!(
            device,
            "pixel_adjust",
            PIXEL_ADJUST_WGSL,
            [
                storage_buffer_entry!(0, read_write),
                uniform_buffer_entry!(1),
            ]
        );

        Ok(Self {
            context,
            pipeline,
            bind_group_layout,
        })
    }

    pub fn needs_adjustment(settings: &EnhancementSettings) -> bool {
        Self::activity(settings).has_any()
    }

    pub fn apply(
        &self,
        image: &DynamicImage,
        settings: &EnhancementSettings,
    ) -> Result<DynamicImage> {
        let activity = Self::activity(settings);
        if !activity.has_any() {
            return Ok(image.clone());
        }

        let rgba = image.to_rgba8();
        let (width, height) = rgba.dimensions();
        let pixel_count = (width as usize) * (height as usize);
        let data_u32 = pack_rgba_pixels(rgba.as_raw());

        let device = self.context.device();
        let queue = self.context.queue();

        let storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pixel_adjust_storage"),
            contents: cast_slice(&data_u32),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let uniforms = PixelAdjustUniforms {
            exposure_multiplier: if activity.exposure {
                2f32.powf(settings.exposure_stops)
            } else {
                1.0
            },
            brightness_offset: settings.brightness as f32,
            contrast_factor: settings.contrast.clamp(0.5, 2.0),
            saturation: settings.saturation.clamp(0.0, 2.5),
            pixel_count: pixel_count as u32,
            flags: activity.flags(),
            __padding: [0; 2],
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pixel_adjust_uniforms"),
            contents: bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pixel_adjust_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: storage_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let buffer_size_bytes =
            (data_u32.len() * std::mem::size_of::<u32>()) as wgpu::BufferAddress;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pixel_adjust_readback"),
            size: buffer_size_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("pixel_adjust_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pixel_adjust_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let dispatch = div_ceil(pixel_count as u32, 256);
            pass.dispatch_workgroups(dispatch, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&storage_buffer, 0, &readback, 0, buffer_size_bytes);
        queue.submit(std::iter::once(encoder.finish()));

        let out_pixels = gpu_readback!(readback, device, data_u32.len(), "pixel adjust")?;
        let out_bytes = unpack_rgba_pixels(&out_pixels);

        let image =
            RgbaImage::from_raw(width, height, out_bytes).context("failed to build RGBA image")?;
        Ok(DynamicImage::ImageRgba8(image))
    }
}

fn div_ceil(value: u32, divisor: u32) -> u32 {
    value.div_ceil(divisor)
}

#[derive(Clone, Copy)]
struct AdjustmentActivity {
    exposure: bool,
    brightness: bool,
    contrast: bool,
    saturation: bool,
}

impl AdjustmentActivity {
    fn has_any(&self) -> bool {
        self.exposure || self.brightness || self.contrast || self.saturation
    }

    fn flags(&self) -> u32 {
        let mut flags = 0;
        if self.exposure {
            flags |= FLAG_EXPOSURE;
        }
        if self.brightness {
            flags |= FLAG_BRIGHTNESS;
        }
        if self.contrast {
            flags |= FLAG_CONTRAST;
        }
        if self.saturation {
            flags |= FLAG_SATURATION;
        }
        flags
    }
}

impl GpuPixelAdjust {
    fn activity(settings: &EnhancementSettings) -> AdjustmentActivity {
        AdjustmentActivity {
            exposure: settings.exposure_stops.abs() >= EPSILON,
            brightness: settings.brightness != 0,
            contrast: (settings.contrast - 1.0).abs() >= EPSILON,
            saturation: (settings.saturation - 1.0).abs() >= EPSILON,
        }
    }
}
