use std::sync::Arc;

use anyhow::{Context, Result};
use bytemuck::{bytes_of, cast_slice};
use image::{DynamicImage, RgbaImage};
use wgpu::util::DeviceExt;

use crate::{
    create_gpu_pipeline, gpu_readback, gpu_uniforms, storage_buffer_entry, uniform_buffer_entry,
};

use super::{CROP_WGSL, GpuBufferPool, GpuContext, pack_rgba_pixels, unpack_rgba_pixels};

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
    pool: Arc<GpuBufferPool>,
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

        let source_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let output_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let readback_usage = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;

        let source_buffer =
            self.pool
                .acquire(source_buffer_size, source_usage, Some("batch_crop_source"))?;
        queue.write_buffer(&source_buffer, 0, cast_slice(&source_data));

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
        let readback =
            self.pool
                .acquire(total_bytes, readback_usage, Some("batch_crop_readback"))?;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("batch_crop_encoder"),
        });
        let mut transient_output_buffers: Vec<(wgpu::Buffer, wgpu::BufferAddress)> =
            Vec::with_capacity(requests.len());

        for (req, job) in requests.iter().zip(jobs.iter()) {
            let buffer_len_bytes =
                (job.element_count * std::mem::size_of::<u32>()) as wgpu::BufferAddress;

            let output_buffer =
                self.pool
                    .acquire(buffer_len_bytes, output_usage, Some("batch_crop_output"))?;

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
                    timestamp_writes: self.context.timestamp_writes("batch_crop"),
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
            transient_output_buffers.push((output_buffer, buffer_len_bytes));
        }

        queue.submit(std::iter::once(encoder.finish()));

        let out_pixels = gpu_readback!(readback, device, total_elements, "batch crop")?;

        for (buffer, size) in transient_output_buffers {
            self.pool.recycle(buffer, size, output_usage);
        }
        self.pool
            .recycle(source_buffer, source_buffer_size, source_usage);
        self.pool.recycle(readback, total_bytes, readback_usage);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::test_support::{assert_plausible_output, gradient_image, test_context};

    fn request(x: u32, y: u32, w: u32, h: u32, out_w: u32, out_h: u32) -> BatchCropRequest {
        BatchCropRequest {
            source_x: x,
            source_y: y,
            source_width: w,
            source_height: h,
            output_width: out_w,
            output_height: out_h,
        }
    }

    // `validate` is the guard between a user-supplied rectangle and a GPU
    // dispatch that would read outside the source buffer, so each rejection
    // deserves its own case rather than one happy-path check.

    #[test]
    fn validate_accepts_a_rectangle_inside_the_image() {
        assert!(request(0, 0, 10, 10, 5, 5).validate(10, 10).is_ok());
        // Touching the far edge exactly is in bounds: end == dimension.
        assert!(request(5, 5, 5, 5, 4, 4).validate(10, 10).is_ok());
    }

    #[test]
    fn validate_rejects_zero_sized_output() {
        for (w, h) in [(0, 4), (4, 0), (0, 0)] {
            assert!(
                request(0, 0, 8, 8, w, h).validate(16, 16).is_err(),
                "output {w}x{h} must be rejected"
            );
        }
    }

    #[test]
    fn validate_rejects_an_empty_source_region() {
        for (w, h) in [(0, 4), (4, 0), (0, 0)] {
            assert!(
                request(0, 0, w, h, 4, 4).validate(16, 16).is_err(),
                "source {w}x{h} must be rejected"
            );
        }
    }

    #[test]
    fn validate_rejects_an_origin_outside_the_image() {
        // Equal to the dimension is already past the last valid pixel.
        assert!(request(16, 0, 1, 1, 4, 4).validate(16, 16).is_err());
        assert!(request(0, 16, 1, 1, 4, 4).validate(16, 16).is_err());
    }

    #[test]
    fn validate_rejects_a_rectangle_that_runs_past_the_edge() {
        assert!(request(8, 0, 9, 4, 4, 4).validate(16, 16).is_err());
        assert!(request(0, 8, 4, 9, 4, 4).validate(16, 16).is_err());
        // saturating_add means a huge width must not wrap into a valid range.
        assert!(request(1, 1, u32::MAX, 1, 4, 4).validate(16, 16).is_err());
    }

    #[test]
    fn crop_with_no_requests_returns_no_images() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping crop_batch GPU test: no adapter");
            return;
        };
        let cropper = GpuBatchCropper::new(ctx).expect("init");
        let source = gradient_image(16, 16);
        assert!(cropper.crop(&source, &[]).expect("crop").is_empty());
    }

    #[test]
    fn crop_returns_one_image_per_request_at_the_requested_size() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping crop_batch GPU test: no adapter");
            return;
        };
        let cropper = GpuBatchCropper::new(ctx).expect("init");
        let source = gradient_image(64, 48);

        // Differing output sizes so a single shared stride cannot satisfy all
        // three, and one non-power-of-two size to catch dispatch rounding.
        let requests = [
            request(0, 0, 32, 24, 16, 12),
            request(32, 24, 32, 24, 8, 8),
            request(10, 10, 20, 20, 13, 7),
        ];

        let outputs = cropper.crop(&source, &requests).expect("crop");
        assert_eq!(outputs.len(), requests.len(), "one output per request");

        for (out, req) in outputs.iter().zip(requests.iter()) {
            assert_eq!(
                (out.width(), out.height()),
                (req.output_width, req.output_height),
                "output must match the requested size"
            );
            let pixels = out.to_rgba8();
            assert!(
                pixels.as_raw().iter().any(|&b| b != 0),
                "crop produced an all-zero image, so the dispatch wrote nothing"
            );
        }
    }

    #[test]
    fn crop_regions_differ_when_the_source_rectangles_differ() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping crop_batch GPU test: no adapter");
            return;
        };
        let cropper = GpuBatchCropper::new(ctx).expect("init");
        let source = gradient_image(64, 64);

        // Same output size, different source corners. A shader that ignores the
        // source offset returns identical tiles for both.
        let outputs = cropper
            .crop(
                &source,
                &[
                    request(0, 0, 16, 16, 16, 16),
                    request(48, 48, 16, 16, 16, 16),
                ],
            )
            .expect("crop");

        assert_ne!(
            outputs[0].to_rgba8().as_raw(),
            outputs[1].to_rgba8().as_raw(),
            "crops from different source offsets must not be identical"
        );
    }

    #[test]
    fn crop_rejects_a_request_outside_the_source() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping crop_batch GPU test: no adapter");
            return;
        };
        let cropper = GpuBatchCropper::new(ctx).expect("init");
        let source = gradient_image(16, 16);

        assert!(
            cropper
                .crop(&source, &[request(0, 0, 32, 32, 8, 8)])
                .is_err(),
            "a rectangle larger than the source must be rejected, not clamped"
        );
    }

    /// A batch with fewer crops than the one before it must still read back.
    ///
    /// The pool sizes its readback buffer for the largest batch it has seen and hands that
    /// buffer to later, smaller batches. `gpu_readback!` used to map the whole buffer, so the
    /// length check compared the output against the pooled capacity and failed -- an image with
    /// one face after an image with three was enough to break it, with no memory pressure
    /// involved at all.
    #[test]
    fn a_smaller_batch_after_a_larger_one_still_reads_back() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping batch crop test: no GPU");
            return;
        };
        let cropper = GpuBatchCropper::new(ctx).expect("init");
        let image = gradient_image(400, 400);
        let at = |x: u32| request(x, 0, 100, 100, 64, 64);

        let many = cropper
            .crop(&image, &[at(0), at(100), at(200)])
            .expect("three crops");
        assert_eq!(many.len(), 3);

        let few = cropper.crop(&image, &[at(0)]).expect("one crop");
        assert_eq!(few.len(), 1);
        assert_eq!((few[0].width(), few[0].height()), (64, 64));
    }

    #[test]
    fn memory_usage_reports_pooled_buffers_until_cleared() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping crop_batch GPU test: no adapter");
            return;
        };
        let cropper = GpuBatchCropper::new(ctx).expect("init");
        let source = gradient_image(32, 32);

        cropper
            .crop(&source, &[request(0, 0, 16, 16, 8, 8)])
            .expect("crop");
        assert!(cropper.memory_usage() > 0, "buffers should be pooled");

        cropper.clear_cache();
        assert_eq!(cropper.memory_usage(), 0, "clear_cache must release them");
    }

    #[test]
    fn crop_output_is_a_plausible_image_for_a_full_frame_request() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping crop_batch GPU test: no adapter");
            return;
        };
        let cropper = GpuBatchCropper::new(ctx).expect("init");
        let source = gradient_image(24, 24);

        // Crop the whole frame at its own size: the result should be a faithful
        // image of the same dimensions, which pins down the no-scaling path.
        let outputs = cropper
            .crop(&source, &[request(0, 0, 24, 24, 24, 24)])
            .expect("crop");
        assert_plausible_output(&outputs[0], &source, "crop (identity)");
    }
}
