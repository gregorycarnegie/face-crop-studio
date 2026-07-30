use std::sync::Arc;

use anyhow::{Context, Result};
use bytemuck::{bytes_of, cast_slice};
use image::{DynamicImage, RgbaImage};
use wgpu::util::DeviceExt;

use crate::{
    create_gpu_pipeline, enhance::EnhancementSettings, gpu_readback, gpu_uniforms,
    storage_buffer_entry, uniform_buffer_entry,
};

use super::{GpuBufferPool, GpuContext, PIXEL_ADJUST_WGSL, pack_rgba_pixels, unpack_rgba_pixels};

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
    pool: Arc<GpuBufferPool>,
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

        let buffer_size_bytes =
            (data_u32.len() * std::mem::size_of::<u32>()) as wgpu::BufferAddress;
        let storage_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let readback_usage = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;

        let storage_buffer = self.pool.acquire(
            buffer_size_bytes,
            storage_usage,
            Some("pixel_adjust_storage"),
        )?;
        queue.write_buffer(&storage_buffer, 0, cast_slice(&data_u32));

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

        let readback = self.pool.acquire(
            buffer_size_bytes,
            readback_usage,
            Some("pixel_adjust_readback"),
        )?;

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

        self.pool
            .recycle(storage_buffer, buffer_size_bytes, storage_usage);
        self.pool
            .recycle(readback, buffer_size_bytes, readback_usage);

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

#[cfg(test)]
mod tests {
    use super::*;

    // Which adjustments are live, and the bitmask handed to the shader,
    // decide whether the GPU path runs at all and which branches it takes.
    // Both are plain CPU logic and neither had any coverage.

    fn none_active() -> AdjustmentActivity {
        AdjustmentActivity {
            exposure: false,
            brightness: false,
            contrast: false,
            saturation: false,
        }
    }

    #[test]
    fn flags_pack_each_adjustment_into_its_own_bit() {
        assert_eq!(none_active().flags(), 0);

        let cases = [
            (
                AdjustmentActivity {
                    exposure: true,
                    ..none_active()
                },
                1,
            ),
            (
                AdjustmentActivity {
                    brightness: true,
                    ..none_active()
                },
                2,
            ),
            (
                AdjustmentActivity {
                    contrast: true,
                    ..none_active()
                },
                4,
            ),
            (
                AdjustmentActivity {
                    saturation: true,
                    ..none_active()
                },
                8,
            ),
        ];
        for (activity, expected) in cases {
            assert_eq!(activity.flags(), expected);
        }

        // All four combine rather than overwrite one another.
        let all = AdjustmentActivity {
            exposure: true,
            brightness: true,
            contrast: true,
            saturation: true,
        };
        assert_eq!(all.flags(), 0b1111);
    }

    #[test]
    fn has_any_is_true_for_each_adjustment_alone() {
        assert!(!none_active().has_any());
        for setter in [
            |a: &mut AdjustmentActivity| a.exposure = true,
            |a: &mut AdjustmentActivity| a.brightness = true,
            |a: &mut AdjustmentActivity| a.contrast = true,
            |a: &mut AdjustmentActivity| a.saturation = true,
        ] {
            let mut activity = none_active();
            setter(&mut activity);
            assert!(activity.has_any(), "one live adjustment is enough");
        }
    }

    #[test]
    fn activity_treats_neutral_settings_as_inactive() {
        // Defaults are the identity transform: no exposure shift, no
        // brightness offset, and unit contrast and saturation.
        let activity = GpuPixelAdjust::activity(&EnhancementSettings::default());
        assert!(!activity.has_any());
        assert_eq!(activity.flags(), 0);
    }

    #[test]
    fn activity_detects_each_adjustment_independently() {
        // Contrast and saturation are measured as a distance from 1.0, not
        // from zero, so a setting of 1.0 must read as inactive while 0.0 is a
        // real change.
        let exposure = EnhancementSettings {
            exposure_stops: 0.5,
            ..Default::default()
        };
        assert_eq!(GpuPixelAdjust::activity(&exposure).flags(), 1);

        let brightness = EnhancementSettings {
            brightness: -10,
            ..Default::default()
        };
        assert_eq!(GpuPixelAdjust::activity(&brightness).flags(), 2);

        let contrast = EnhancementSettings {
            contrast: 0.0,
            ..Default::default()
        };
        assert_eq!(
            GpuPixelAdjust::activity(&contrast).flags(),
            4,
            "zero contrast is a change from the 1.0 default"
        );

        let saturation = EnhancementSettings {
            saturation: 1.4,
            ..Default::default()
        };
        assert_eq!(GpuPixelAdjust::activity(&saturation).flags(), 8);
    }

    #[test]
    fn activity_ignores_negligible_differences() {
        // Below EPSILON the adjustment is not worth a GPU pass.
        let settings = EnhancementSettings {
            exposure_stops: EPSILON * 0.5,
            contrast: 1.0 + EPSILON * 0.5,
            saturation: 1.0 - EPSILON * 0.5,
            ..Default::default()
        };
        assert!(!GpuPixelAdjust::activity(&settings).has_any());

        // A negative exposure of real size still counts: the test is on the
        // absolute value.
        let negative = EnhancementSettings {
            exposure_stops: -0.5,
            ..Default::default()
        };
        assert!(GpuPixelAdjust::activity(&negative).has_any());
    }
}
