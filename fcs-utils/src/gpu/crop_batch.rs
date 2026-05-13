use std::sync::Arc;

use anyhow::{Context, Result};
use bytemuck::{bytes_of, cast_slice};
use image::{DynamicImage, RgbaImage};
use wgpu::util::DeviceExt;

use crate::{
    create_gpu_pipeline, gpu_readback, gpu_uniforms, storage_buffer_entry, uniform_buffer_entry,
};

use super::{CROP_WGSL, GpuContext, pack_rgba_pixels, unpack_rgba_pixels};

gpu_uniforms!(CropUniforms, 0, {
    src_width: u32,
    src_height: u32,
    crop_x: u32,
    crop_y: u32,
    crop_width: u32,
    crop_height: u32,
    dst_width: u32,
    dst_height: u32,
});

/// Describes a single crop-and-resize job inside a batch.
#[derive(Debug, Clone, Copy)]
pub struct BatchCropRequest {
    /// Top-left source coordinate of the crop rectangle.
    pub source_x: u32,
    /// Top-left source coordinate of the crop rectangle.
    pub source_y: u32,
    /// Width of the source crop region.
    pub source_width: u32,
    /// Height of the source crop region.
    pub source_height: u32,
    /// Desired output width in pixels.
    pub output_width: u32,
    /// Desired output height in pixels.
    pub output_height: u32,
}

impl BatchCropRequest {
    fn validate(&self, src_width: u32, src_height: u32) -> Result<()> {
        anyhow::ensure!(
            self.output_width > 0 && self.output_height > 0,
            "GPU batch crop requires non-zero output dimensions"
        );
        anyhow::ensure!(
            self.source_width > 0 && self.source_height > 0,
            "GPU batch crop source region must be non-empty"
        );
        anyhow::ensure!(
            self.source_x < src_width && self.source_y < src_height,
            "GPU batch crop outside image bounds"
        );
        let end_x = self.source_x.saturating_add(self.source_width);
        let end_y = self.source_y.saturating_add(self.source_height);
        anyhow::ensure!(
            end_x <= src_width && end_y <= src_height,
            "GPU batch crop rectangle exceeds image dimensions"
        );
        Ok(())
    }
}

/// GPU compute pipeline that crops and resizes multiple face regions without re-uploading the source image.
#[derive(Clone)]
pub struct GpuBatchCropper {
    context: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuBatchCropper {
    /// Initialize the compute pipeline.
    pub fn new(context: Arc<GpuContext>) -> Result<Self> {
        let device = context.device();
        let (pipeline, bind_group_layout) = create_gpu_pipeline!(
            device,
            "batch_crop",
            CROP_WGSL,
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

    /// Execute the batch crop on the provided image and requests.
    pub fn crop(
        &self,
        source: &DynamicImage,
        requests: &[BatchCropRequest],
    ) -> Result<Vec<DynamicImage>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        let rgba = source.to_rgba8();
        let (src_width, src_height) = rgba.dimensions();
        anyhow::ensure!(
            src_width > 0 && src_height > 0,
            "source image must be non-empty"
        );

        let source_data = pack_rgba_pixels(rgba.as_raw());
        let source_buffer_size = (source_data.len() * std::mem::size_of::<u32>()) as u64;
        let max_binding = self.context.limits().max_storage_buffer_binding_size;
        anyhow::ensure!(
            source_buffer_size <= max_binding,
            "Source image too large for GPU batch cropping ({:.2} MB exceeds {:.2} MB WebGPU buffer binding limit). Image dimensions: {}x{}. Use CPU cropping for large images.",
            source_buffer_size as f64 / (1024.0 * 1024.0),
            max_binding as f64 / (1024.0 * 1024.0),
            src_width,
            src_height
        );

        let device = self.context.device();
        let queue = self.context.queue();

        let source_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("batch_crop_source"),
            contents: cast_slice(&source_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        struct JobInfo {
            element_offset: usize,
            element_count: usize,
            dst_width: u32,
            dst_height: u32,
            uniform: CropUniforms,
        }

        let mut jobs = Vec::with_capacity(requests.len());
        let mut total_elements = 0usize;
        for req in requests {
            req.validate(src_width, src_height)?;
            let crop_width = req.source_width.min(src_width - req.source_x).max(1);
            let crop_height = req.source_height.min(src_height - req.source_y).max(1);
            let dst_pixels = (req.output_width as usize) * (req.output_height as usize);
            let element_count = dst_pixels;
            jobs.push(JobInfo {
                element_offset: total_elements,
                element_count,
                dst_width: req.output_width,
                dst_height: req.output_height,
                uniform: CropUniforms {
                    src_width,
                    src_height,
                    crop_x: req.source_x,
                    crop_y: req.source_y,
                    crop_width,
                    crop_height,
                    dst_width: req.output_width,
                    dst_height: req.output_height,
                    __padding: [],
                },
            });
            total_elements += element_count;
        }

        let total_bytes = (total_elements * std::mem::size_of::<u32>()) as wgpu::BufferAddress;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("batch_crop_readback"),
            size: total_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("batch_crop_encoder"),
        });
        let mut _transient_buffers = Vec::with_capacity(requests.len());

        for (req, job) in requests.iter().zip(jobs.iter()) {
            let buffer_len_bytes =
                (job.element_count * std::mem::size_of::<u32>()) as wgpu::BufferAddress;

            let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("batch_crop_output"),
                size: buffer_len_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("batch_crop_uniforms"),
                contents: bytes_of(&job.uniform),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("batch_crop_bg"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: source_buffer.as_entire_binding(),
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

            {
                let workgroups_x = req.output_width.div_ceil(16);
                let workgroups_y = req.output_height.div_ceil(16);
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("batch_crop_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups_x.max(1), workgroups_y.max(1), 1);
            }

            let readback_offset =
                (job.element_offset * std::mem::size_of::<u32>()) as wgpu::BufferAddress;
            encoder.copy_buffer_to_buffer(
                &output_buffer,
                0,
                &readback,
                readback_offset,
                buffer_len_bytes,
            );
            _transient_buffers.push(output_buffer);
        }

        queue.submit(std::iter::once(encoder.finish()));

        let out_pixels = gpu_readback!(readback, device, total_elements, "batch crop")?;

        let mut outputs = Vec::with_capacity(requests.len());
        for job in jobs {
            let start = job.element_offset;
            let end = start + job.element_count;
            let slice = &out_pixels[start..end];
            let bytes = unpack_rgba_pixels(slice);
            let image = RgbaImage::from_raw(job.dst_width, job.dst_height, bytes)
                .context("failed to build GPU crop image")?;
            outputs.push(DynamicImage::ImageRgba8(image));
        }

        Ok(outputs)
    }
}
