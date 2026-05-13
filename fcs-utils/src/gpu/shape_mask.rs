use std::sync::Arc;

use anyhow::{Context, Result};
use bytemuck::cast_slice;
use image::{DynamicImage, RgbaImage};
use wgpu::util::DeviceExt;

use crate::shape::CropShape;
use crate::shape::outline_points_for_rect;
use crate::{create_gpu_pipeline, gpu_readback, storage_buffer_entry, uniform_buffer_entry};

use super::{GpuContext, SHAPE_MASK_WGSL, pack_rgba_pixels, unpack_rgba_pixels};

const MAX_POINTS: usize = 512;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
struct ShapeMaskUniforms {
    pub width: u32,
    pub height: u32,
    pub point_count: u32,
    pub samples: u32,
    pub vignette_softness: f32,
    pub vignette_intensity: f32,
    pub vignette_color: u32,
    pub __padding: u32,
}

#[derive(Clone)]
pub struct GpuShapeMask {
    context: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuShapeMask {
    pub fn new(context: Arc<GpuContext>) -> Result<Self> {
        let device = context.device();

        let (pipeline, bind_group_layout) = create_gpu_pipeline!(
            device,
            "shape_mask",
            SHAPE_MASK_WGSL,
            [
                storage_buffer_entry!(0, read_write),
                storage_buffer_entry!(1, read_only),
                uniform_buffer_entry!(2),
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
        shape: &CropShape,
        vignette_softness: f32,
        vignette_intensity: f32,
        vignette_color: crate::color::RgbaColor,
    ) -> Result<Option<DynamicImage>> {
        if matches!(shape, CropShape::Rectangle) {
            return Ok(None);
        }
        let width = image.width();
        let height = image.height();

        let points = outline_points_for_rect(width as f32, height as f32, shape);
        if points.len() < 3 {
            return Ok(None);
        }
        let clamped = points
            .iter()
            .take(MAX_POINTS)
            .map(|(x, y)| [*x, *y])
            .collect::<Vec<_>>();
        if clamped.len() < 3 {
            return Ok(None);
        }

        let rgba = image.to_rgba8();
        let pixels_u32 = pack_rgba_pixels(rgba.as_raw());

        let device = self.context.device();
        let queue = self.context.queue();
        let buffer_size = (pixels_u32.len() * std::mem::size_of::<u32>()) as wgpu::BufferAddress;

        let storage = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shape_mask_pixels"),
            contents: cast_slice(&pixels_u32),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shape_mask_readback"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let points_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shape_mask_points"),
            contents: cast_slice(&clamped),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let uniforms = ShapeMaskUniforms {
            width,
            height,
            point_count: clamped.len() as u32,
            samples: 4,
            vignette_softness,
            vignette_intensity,
            vignette_color: ((vignette_color.alpha as u32) << 24)
                | ((vignette_color.blue as u32) << 16)
                | ((vignette_color.green as u32) << 8)
                | (vignette_color.red as u32),
            ..Default::default()
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shape_mask_uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shape_mask_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: storage.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: points_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("shape_mask_encoder"),
        });
        {
            let workgroups_x = width.div_ceil(16);
            let workgroups_y = height.div_ceil(16);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("shape_mask_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }
        encoder.copy_buffer_to_buffer(&storage, 0, &readback, 0, buffer_size);
        queue.submit(std::iter::once(encoder.finish()));

        let out_pixels = gpu_readback!(readback, device, pixels_u32.len(), "shape mask")?;
        let out_bytes = unpack_rgba_pixels(&out_pixels);

        let masked = RgbaImage::from_raw(width, height, out_bytes)
            .context("failed to build masked image")?;
        Ok(Some(DynamicImage::ImageRgba8(masked)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::RgbaColor;
    use crate::gpu::{GpuAvailability, GpuContextOptions};
    use image::RgbaImage;

    fn test_context() -> Option<Arc<GpuContext>> {
        match GpuContext::init_with_fallback(&GpuContextOptions::default()) {
            GpuAvailability::Available(ctx) => Some(ctx),
            _ => None,
        }
    }

    #[test]
    fn rectangle_variant_returns_none() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping shape_mask rectangle test: no GPU");
            return;
        };
        let mask = GpuShapeMask::new(ctx).expect("init");
        let image =
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(8, 8, image::Rgba([255, 0, 0, 255])));
        let result = mask
            .apply(
                &image,
                &CropShape::Rectangle,
                0.0,
                1.0,
                RgbaColor::opaque(0, 0, 0),
            )
            .expect("apply");
        assert!(
            result.is_none(),
            "Rectangle shape should return None (no masking needed)"
        );
    }

    #[test]
    fn ellipse_shape_produces_masked_image_with_correct_dimensions() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping shape_mask ellipse test: no GPU");
            return;
        };
        let mask = GpuShapeMask::new(ctx).expect("init");
        let width = 32u32;
        let height = 32u32;
        let image = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            width,
            height,
            image::Rgba([255, 128, 64, 255]),
        ));
        let result = mask
            .apply(
                &image,
                &CropShape::Ellipse,
                0.5,
                0.8,
                RgbaColor::opaque(0, 0, 0),
            )
            .expect("apply should not error");
        let output = result.expect("Ellipse shape should produce a masked image");
        assert_eq!(output.width(), width);
        assert_eq!(output.height(), height);
        // Corner pixels (outside the ellipse) should be transparent.
        let rgba = output.to_rgba8();
        let corner_alpha = rgba.get_pixel(0, 0)[3];
        assert_eq!(
            corner_alpha, 0,
            "Corner pixel should be masked to transparent"
        );
        // Center pixel (inside the ellipse) should remain opaque.
        let center_alpha = rgba.get_pixel(width / 2, height / 2)[3];
        assert_eq!(center_alpha, 255, "Center pixel should remain opaque");
    }

    #[test]
    fn polygon_shape_produces_masked_image() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping shape_mask polygon test: no GPU");
            return;
        };
        let mask = GpuShapeMask::new(ctx).expect("init");
        let image = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            64,
            64,
            image::Rgba([200, 200, 200, 255]),
        ));
        let shape = CropShape::Polygon {
            sides: 6,
            rotation_deg: 0.0,
            corner_style: Default::default(),
        };
        let result = mask
            .apply(&image, &shape, 0.0, 0.0, RgbaColor::opaque(0, 0, 0))
            .expect("apply should not error");
        let output = result.expect("Polygon shape should produce a masked image");
        assert_eq!(output.width(), 64);
        assert_eq!(output.height(), 64);
    }
}
