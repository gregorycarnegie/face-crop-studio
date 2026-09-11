use std::sync::Arc;

use anyhow::{Context, Result};
use bytemuck::cast_slice;
use image::{DynamicImage, RgbaImage};
use wgpu::util::DeviceExt;

use crate::{
    create_gpu_pipeline, gpu_readback,
    shape::{CropShape, outline_points_for_rect},
    storage_buffer_entry, uniform_buffer_entry,
};

use super::{GpuBufferPool, GpuContext, SHAPE_MASK_WGSL, pack_rgba_pixels, unpack_rgba_pixels};

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

/// Reusable GPU pipeline for antialiased crop outlines and edge vignettes.
#[derive(Clone)]
pub struct GpuShapeMask {
    context: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    pool: Arc<GpuBufferPool>,
}

impl GpuShapeMask {
    /// Create the shape-mask pipeline and pool on an existing GPU context.
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

    /// Mask an image to a crop shape, optionally fading and tinting its edge.
    ///
    /// `vignette_softness` is the fade width as a fraction of the shorter image
    /// side; use 0..=1. `vignette_intensity` is the color blend amount (0..=1);
    /// alpha coverage is independent of this tint.
    ///
    /// Returns `Ok(None)` for rectangles or outlines with fewer than three points.
    /// Otherwise returns RGBA8, or an error on allocation/readback failure.
    /// Only the first 512 outline points are used; use the CPU mask for denser outlines.
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

        let storage_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let readback_usage = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;

        let storage = self
            .pool
            .acquire(buffer_size, storage_usage, Some("shape_mask_pixels"))?;
        queue.write_buffer(&storage, 0, cast_slice(&pixels_u32));
        let readback =
            self.pool
                .acquire(buffer_size, readback_usage, Some("shape_mask_readback"))?;

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
                timestamp_writes: self.context.timestamp_writes("shape_mask"),
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }
        encoder.copy_buffer_to_buffer(&storage, 0, &readback, 0, buffer_size);
        queue.submit(std::iter::once(encoder.finish()));

        let out_pixels = gpu_readback!(readback, device, pixels_u32.len(), "shape mask")?;
        let out_bytes = unpack_rgba_pixels(&out_pixels);

        self.pool.recycle(storage, buffer_size, storage_usage);
        self.pool.recycle(readback, buffer_size, readback_usage);

        let masked = RgbaImage::from_raw(width, height, out_bytes)
            .context("failed to build masked image")?;
        Ok(Some(DynamicImage::ImageRgba8(masked)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::RgbaColor;
    use image::RgbaImage;

    use crate::gpu::test_support::test_context;

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

    #[test]
    fn vignette_colour_and_softness_reach_the_masked_edges() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping shape_mask vignette test: no GPU");
            return;
        };
        let mask = GpuShapeMask::new(ctx).expect("init");
        let image = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            64,
            64,
            image::Rgba([255, 255, 255, 255]),
        ));

        // A colour with three different channels: a packing shift or a dropped
        // OR in the RGBA -> u32 uniform shows up as the wrong channel value,
        // where black (the other tests' colour) would look identical.
        let out = mask
            .apply(
                &image,
                &CropShape::Ellipse,
                0.6,
                1.0,
                RgbaColor::opaque(255, 128, 32),
            )
            .expect("apply should not error")
            .expect("Ellipse shape should produce a masked image")
            .to_rgba8();

        // Fully outside the ellipse: the pixel is entirely vignette colour.
        let corner = out.get_pixel(0, 0).0;
        assert_eq!(
            [corner[0], corner[1], corner[2]],
            [255, 128, 32],
            "the corner should be painted the vignette colour"
        );
        assert_eq!(corner[3], 0, "the corner is still masked out");

        // Deep inside, the shape is untouched by both mask and vignette.
        let centre = out.get_pixel(32, 32).0;
        assert_eq!(centre, [255, 255, 255, 255], "the centre stays as it was");

        // Just inside the outline the softness fade is partway through, which
        // is the only place a dropped softness parameter is visible.
        let edge_alpha = out.get_pixel(32, 4).0[3];
        assert!(
            (1..=254).contains(&edge_alpha),
            "the softness fade should partially mask the edge, got alpha {edge_alpha}"
        );
    }

    #[test]
    fn a_three_sided_polygon_is_still_masked() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping shape_mask triangle test: no GPU");
            return;
        };
        let mask = GpuShapeMask::new(ctx).expect("init");
        let image = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            32,
            32,
            image::Rgba([180, 90, 40, 255]),
        ));
        let shape = CropShape::Polygon {
            sides: 3,
            rotation_deg: 0.0,
            corner_style: crate::shape::PolygonCornerStyle::Sharp,
        };

        // Three points is the minimum a polygon can have, not one too few.
        let out = mask
            .apply(&image, &shape, 0.0, 0.0, RgbaColor::opaque(0, 0, 0))
            .expect("apply should not error")
            .expect("a triangle has enough points to mask");
        assert_eq!(out.to_rgba8().get_pixel(0, 0)[3], 0, "corner masked out");
    }

    #[test]
    fn clear_cache_releases_the_pooled_buffers() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping shape_mask cache test: no GPU");
            return;
        };
        let mask = GpuShapeMask::new(ctx).expect("init");
        let image = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            16,
            16,
            image::Rgba([10, 20, 30, 255]),
        ));
        mask.apply(
            &image,
            &CropShape::Ellipse,
            0.0,
            0.0,
            RgbaColor::opaque(0, 0, 0),
        )
        .expect("apply should not error");

        assert!(
            mask.memory_usage() > 0,
            "applying the mask allocates pooled buffers"
        );
        mask.clear_cache();
        assert_eq!(mask.memory_usage(), 0, "clearing frees them again");
    }

    #[test]
    fn the_mask_edge_is_supersampled() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping shape_mask supersampling test: no GPU");
            return;
        };
        let mask = GpuShapeMask::new(ctx).expect("init");
        let image = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            64,
            64,
            image::Rgba([255, 255, 255, 255]),
        ));

        // No vignette at all, so partial alpha can only come from the shader's
        // multi-sample coverage. One sample per pixel would leave every pixel
        // fully in or fully out, and the outline visibly stepped.
        let out = mask
            .apply(
                &image,
                &CropShape::Ellipse,
                0.0,
                0.0,
                RgbaColor::opaque(0, 0, 0),
            )
            .expect("apply should not error")
            .expect("Ellipse shape should produce a masked image")
            .to_rgba8();

        assert!(
            out.pixels().any(|px| (1..=254).contains(&px.0[3])),
            "the ellipse edge should have partially covered pixels"
        );
    }
}
