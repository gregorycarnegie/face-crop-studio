use std::sync::Arc;

use anyhow::{Context, Result};
use bytemuck::{bytes_of, cast_slice};
use image::{DynamicImage, RgbaImage};
use wgpu::util::DeviceExt;

use super::{
    GpuBufferPool, GpuContext, HIST_EQUALIZE_WGSL,
    buffer_pool::{READBACK, STORAGE_RW},
    pack_rgba_pixels, unpack_rgba_pixels,
};
use crate::{gpu_readback, gpu_uniforms, storage_buffer_entry, uniform_buffer_entry};

gpu_uniforms!(HistogramUniforms, 3, {
    pixel_count: u32,
});

gpu_uniforms!(CdfUniforms, 3, {
    total_pixels: u32,
});

gpu_uniforms!(LutUniforms, 3, {
    pixel_count: u32,
});

/// Reusable GPU histogram equalizer with independent red, green, and blue lookup tables.
#[derive(Clone)]
pub struct GpuHistogramEqualizer {
    context: Arc<GpuContext>,
    histogram_pipeline: wgpu::ComputePipeline,
    cdf_pipeline: wgpu::ComputePipeline,
    apply_pipeline: wgpu::ComputePipeline,
    histogram_bgl: wgpu::BindGroupLayout,
    cdf_bgl: wgpu::BindGroupLayout,
    apply_bgl: wgpu::BindGroupLayout,
    pool: Arc<GpuBufferPool>,
}

impl GpuHistogramEqualizer {
    /// Create histogram, lookup-table, and application pipelines on an existing context.
    pub fn new(context: Arc<GpuContext>) -> Result<Self> {
        let device = context.device();
        // Panics if WGSL compilation fails; the label appears in the panic message.
        // If this panics, inspect hist_equalize.wgsl and verify wgpu feature support.
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hist_equalize.wgsl shader"),
            source: wgpu::ShaderSource::Wgsl(HIST_EQUALIZE_WGSL.into()),
        });

        let histogram_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("histogram_bgl"),
            entries: &[
                storage_buffer_entry!(0, read_only),
                storage_buffer_entry!(1, read_write),
                uniform_buffer_entry!(2),
            ],
        });
        let cdf_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hist_cdf_bgl"),
            entries: &[
                storage_buffer_entry!(0, read_only),
                storage_buffer_entry!(1, read_write),
                uniform_buffer_entry!(2),
            ],
        });
        let apply_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hist_apply_bgl"),
            entries: &[
                storage_buffer_entry!(0, read_write),
                storage_buffer_entry!(1, read_only),
                uniform_buffer_entry!(2),
            ],
        });

        let histogram_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("histogram_layout"),
            bind_group_layouts: &[Some(&histogram_bgl)],
            immediate_size: 0,
        });
        let cdf_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hist_cdf_layout"),
            bind_group_layouts: &[Some(&cdf_bgl)],
            immediate_size: 0,
        });
        let apply_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hist_apply_layout"),
            bind_group_layouts: &[Some(&apply_bgl)],
            immediate_size: 0,
        });

        let histogram_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hist_equalize.wgsl build_histogram pipeline — check wgpu feature support if this panics"),
            layout: Some(&histogram_layout),
            module: &module,
            entry_point: Some("build_histogram"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let cdf_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hist_equalize.wgsl compute_lut pipeline — check wgpu feature support if this panics"),
            layout: Some(&cdf_layout),
            module: &module,
            entry_point: Some("compute_lut"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let apply_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hist_equalize.wgsl apply_equalization pipeline — check wgpu feature support if this panics"),
            layout: Some(&apply_layout),
            module: &module,
            entry_point: Some("apply_equalization"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let pool = Arc::new(GpuBufferPool::new(context.clone(), None));

        Ok(Self {
            context,
            histogram_pipeline,
            cdf_pipeline,
            apply_pipeline,
            histogram_bgl,
            cdf_bgl,
            apply_bgl,
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

    /// Equalize each RGB channel's histogram, preserving alpha.
    /// Returns RGBA8 (an unchanged clone for an empty image), or an error on
    /// buffer allocation or GPU readback failure.
    pub fn equalize(&self, image: &DynamicImage) -> Result<DynamicImage> {
        let rgba = image.to_rgba8();
        let (width, height) = rgba.dimensions();
        let pixel_count = (width as usize) * (height as usize);
        if pixel_count == 0 {
            return Ok(image.clone());
        }

        let data_u32 = pack_rgba_pixels(rgba.as_raw());

        let (lut_buffer, pixel_buffer) =
            self.build_histogram_and_lut(&data_u32, pixel_count as u32)?;

        self.apply_lut(pixel_buffer, lut_buffer, width, height)
    }

    fn build_histogram_and_lut(
        &self,
        pixels: &[u32],
        pixel_count: u32,
    ) -> Result<(wgpu::Buffer, wgpu::Buffer)> {
        let device = self.context.device();
        let queue = self.context.queue();

        let pixel_buffer_size = std::mem::size_of_val(pixels) as wgpu::BufferAddress;
        let pixel_buffer =
            self.pool
                .acquire(pixel_buffer_size, STORAGE_RW, Some("hist_pixels"))?;
        queue.write_buffer(&pixel_buffer, 0, cast_slice(pixels));
        // 256 bins per channel, three channels; the LUT below has the same shape.
        let histogram_size = (256 * 3 * std::mem::size_of::<u32>()) as wgpu::BufferAddress;
        let histogram_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("histogram_buffer"),
            size: histogram_size,
            usage: STORAGE_RW,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &histogram_buffer,
            0,
            vec![0u8; histogram_size as usize].as_slice(),
        );

        let uniform = HistogramUniforms {
            pixel_count,
            __padding: [0; 3],
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hist_uniform"),
            contents: bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("histogram_bg"),
            layout: &self.histogram_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pixel_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: histogram_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let lut_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hist_lut_buffer"),
            size: histogram_size,
            usage: STORAGE_RW,
            mapped_at_creation: false,
        });

        let cdf_uniform = CdfUniforms {
            total_pixels: pixel_count,
            __padding: [0; 3],
        };
        let cdf_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hist_cdf_uniform"),
            contents: bytes_of(&cdf_uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let cdf_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hist_cdf_bg"),
            layout: &self.cdf_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: histogram_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lut_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cdf_uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hist_encoder"),
        });
        {
            let dispatch = pixel_count.div_ceil(256);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hist_pass"),
                timestamp_writes: self.context.timestamp_writes("hist"),
            });
            pass.set_pipeline(&self.histogram_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hist_cdf_pass"),
                timestamp_writes: self.context.timestamp_writes("hist_cdf"),
            });
            pass.set_pipeline(&self.cdf_pipeline);
            pass.set_bind_group(0, &cdf_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));

        Ok((lut_buffer, pixel_buffer))
    }

    fn apply_lut(
        &self,
        pixel_buffer: wgpu::Buffer,
        lut_buffer: wgpu::Buffer,
        width: u32,
        height: u32,
    ) -> Result<DynamicImage> {
        let device = self.context.device();
        let queue = self.context.queue();
        let pixel_count = (width as usize * height as usize) as u32;
        let buffer_size =
            (pixel_count as usize * std::mem::size_of::<u32>()) as wgpu::BufferAddress;

        let uniform = LutUniforms {
            pixel_count,
            __padding: [0; 3],
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hist_apply_uniform"),
            contents: bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hist_apply_bg"),
            layout: &self.apply_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pixel_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lut_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hist_apply_encoder"),
        });
        {
            let dispatch = pixel_count.div_ceil(256);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hist_apply_pass"),
                timestamp_writes: self.context.timestamp_writes("hist_apply"),
            });
            pass.set_pipeline(&self.apply_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch, 1, 1);
        }

        let readback = self
            .pool
            .acquire(buffer_size, READBACK, Some("hist_apply_readback"))?;
        encoder.copy_buffer_to_buffer(&pixel_buffer, 0, &readback, 0, buffer_size);
        queue.submit(std::iter::once(encoder.finish()));

        let expected_len = pixel_count as usize;
        let packed = gpu_readback!(readback, device, expected_len, "histogram equalization")?;
        let bytes = unpack_rgba_pixels(&packed);

        self.pool.recycle(pixel_buffer, buffer_size, STORAGE_RW);
        self.pool.recycle(readback, buffer_size, READBACK);

        let result =
            RgbaImage::from_raw(width, height, bytes).context("failed to build equalized image")?;
        Ok(DynamicImage::ImageRgba8(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::gpu::test_support::{assert_plausible_output, gradient_image, test_context};

    #[test]
    fn memory_usage_grows_after_a_pass_and_resets_when_cleared() {
        let Some(ctx) = test_context() else {
            return;
        };
        let eq = GpuHistogramEqualizer::new(ctx).expect("init");
        eq.equalize(&gradient_image(16, 16)).expect("equalize");
        assert!(
            eq.memory_usage() > 0,
            "the pixel and readback buffers are pooled after a pass"
        );
        eq.clear_cache();
        assert_eq!(eq.memory_usage(), 0, "clear_cache must release them");
    }

    #[test]
    fn equalize_expands_a_low_contrast_image() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping hist_equalize GPU test: no adapter");
            return;
        };
        let eq = GpuHistogramEqualizer::new(ctx).expect("init");

        // A narrow band of dark values: equalisation should spread these out,
        // so the output range must be wider than the input's.
        let mut img = image::RgbaImage::new(32, 32);
        for (x, y, px) in img.enumerate_pixels_mut() {
            let v = 100 + ((x + y) % 8) as u8; // range of 7
            *px = image::Rgba([v, v, v, 255]);
        }
        let image = DynamicImage::ImageRgba8(img);

        let result = eq.equalize(&image).expect("equalize");
        assert_plausible_output(&result, &image, "equalize");

        let span = |i: &DynamicImage| {
            let rgba = i.to_rgba8();
            let lums: Vec<u8> = rgba.pixels().map(|p| p[0]).collect();
            let lo = *lums.iter().min().unwrap();
            let hi = *lums.iter().max().unwrap();
            hi - lo
        };

        let before = span(&image);
        let after = span(&result);
        assert!(
            after > before,
            "equalisation must widen the tonal range: {before} -> {after}"
        );
    }

    #[test]
    fn equalize_covers_dimensions_that_are_not_workgroup_multiples() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping hist_equalize GPU test: no adapter");
            return;
        };
        let eq = GpuHistogramEqualizer::new(ctx).expect("init");
        // Odd size on purpose: a dispatch that rounds the wrong way leaves the
        // trailing pixels untouched, which a flat test image would never reveal.
        let image = gradient_image(41, 23);

        let result = eq.equalize(&image).expect("equalize");
        assert_plausible_output(&result, &image, "equalize (odd dimensions)");

        let after = result.to_rgba8();
        let last_row: Vec<_> = (0..41).map(|x| after.get_pixel(x, 22)[0]).collect();
        assert!(
            last_row.iter().any(|&v| v != last_row[0]),
            "final row is uniform, so the dispatch missed the end of the image"
        );
    }
}
