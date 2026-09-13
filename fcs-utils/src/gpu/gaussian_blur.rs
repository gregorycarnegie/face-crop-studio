use std::sync::Arc;

use anyhow::{Context, Result};
use bytemuck::{bytes_of, cast_slice};
use image::{DynamicImage, RgbaImage};
use wgpu::util::DeviceExt;

use super::{
    GAUSSIAN_BLUR_WGSL, GpuBufferPool, GpuContext,
    buffer_pool::{READBACK, STORAGE_RW},
    pack_rgba_pixels, unpack_rgba_pixels,
};
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

/// Reusable two-pass Gaussian blur pipeline with pooled GPU buffers.
#[derive(Clone)]
pub struct GpuGaussianBlur {
    context: Arc<GpuContext>,
    pool: Arc<GpuBufferPool>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuGaussianBlur {
    /// Create the blur pipeline and buffer pool on an existing GPU context.
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

    /// Blur an image with a pixel radius rounded up and capped at 12.
    /// Nonpositive radii return an unchanged clone. Active blurs return RGBA8;
    /// buffer allocation or readback failures return an error.
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
        let input_buffer =
            self.pool
                .acquire(buffer_size, STORAGE_RW, Some("gaussian_blur_input"))?;
        queue.write_buffer(&input_buffer, 0, cast_slice(&data_u32));

        let weights_size = std::mem::size_of_val(&weights) as u64;
        let weights_buffer =
            self.pool
                .acquire(weights_size, STORAGE_RW, Some("gaussian_blur_weights"))?;
        queue.write_buffer(&weights_buffer, 0, cast_slice(&weights));

        let temp_buffer = self.pool.acquire(
            buffer_size,
            wgpu::BufferUsages::STORAGE,
            Some("gaussian_blur_temp"),
        )?;

        let readback = self
            .pool
            .acquire(buffer_size, READBACK, Some("gaussian_blur_readback"))?;

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
        dispatch_blur(
            &mut encoder,
            &self.context,
            "gaussian_blur_h",
            &self.pipeline,
            &horizontal_bg,
            width,
            height,
        );

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
        dispatch_blur(
            &mut encoder,
            &self.context,
            "gaussian_blur_v",
            &self.pipeline,
            &vertical_bg,
            width,
            height,
        );

        encoder.copy_buffer_to_buffer(&input_buffer, 0, &readback, 0, buffer_size);
        queue.submit(std::iter::once(encoder.finish()));

        let out_pixels = gpu_readback!(readback, device, data_u32.len(), "gaussian blur")?;
        let out_bytes = unpack_rgba_pixels(&out_pixels);

        // Recycle buffers
        self.pool.recycle(input_buffer, buffer_size, STORAGE_RW);
        self.pool.recycle(weights_buffer, weights_size, STORAGE_RW);
        self.pool
            .recycle(temp_buffer, buffer_size, wgpu::BufferUsages::STORAGE);
        self.pool.recycle(readback, buffer_size, READBACK);

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
    context: &GpuContext,
    label: &str,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    width: u32,
    height: u32,
) {
    let workgroups_x = width.div_ceil(16);
    let workgroups_y = height.div_ceil(16);
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("gaussian_blur_pass"),
        timestamp_writes: context.timestamp_writes(label),
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
    // The centre tap is gaussian(0) = 1, so sum >= 1.
    for weight in weights.iter_mut().take(kernel_size as usize) {
        *weight /= sum;
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
    use image::RgbaImage;

    use crate::gpu::test_support::test_context;

    #[track_caller]
    fn close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn gaussian_matches_the_reference_curve() {
        // exp(-d^2 / 2*sigma^2). Normalisation happens later, so these are the
        // raw weights and any change to the exponent shows up directly.
        close(gaussian(0.0, 1.0), 1.0);
        close(gaussian(1.0, 1.0), (-0.5f32).exp());
        close(gaussian(2.0, 1.0), (-2.0f32).exp());
        // Widening sigma flattens the curve.
        close(gaussian(1.0, 2.0), (-0.125f32).exp());
        // Symmetric about zero.
        close(gaussian(-1.5, 1.0), gaussian(1.5, 1.0));
    }

    #[test]
    fn build_kernel_has_the_expected_shape() {
        // Summing to one holds for *any* normalised kernel, so it cannot see a
        // mangled exponent or a wrong sigma. These pin the actual weights.
        //
        // radius 1 -> sigma 1.0, three taps of exp(-0.5), 1, exp(-0.5)
        // normalised by their sum of 2.2130613.
        let w = build_kernel(1);
        close(w[0], 0.27406862);
        close(w[1], 0.45186275);
        close(w[2], 0.27406862);
        assert_eq!(w[3], 0.0, "entries past the kernel stay zero");
    }

    #[test]
    fn build_kernel_ties_sigma_to_the_radius() {
        // sigma = radius * 0.5 + 0.5, so radius 3 gives sigma 2.0. The ratio
        // between adjacent taps survives normalisation, which makes it a clean
        // probe: one step out at sigma 2 is exp(-1/8).
        let w = build_kernel(3);
        close(w[2] / w[3], (-0.125f32).exp());
        close(w[1] / w[3], (-0.5f32).exp());

        // radius 1 has sigma 1.0 instead, a visibly tighter curve.
        let narrow = build_kernel(1);
        close(narrow[0] / narrow[1], (-0.5f32).exp());
    }

    #[test]
    fn build_kernel_spans_two_radius_plus_one_taps() {
        // radius 3 fills seven slots; `radius + 2` would fill only five.
        let w = build_kernel(3);
        assert!(w[6] > 0.0, "the seventh tap must be populated");
        assert_eq!(w[7], 0.0, "and the eighth must not be");

        // Symmetry about the centre tap.
        for i in 0..3 {
            close(w[i], w[6 - i]);
        }
        // The centre is the largest weight.
        assert!(w[3] > w[2] && w[2] > w[1] && w[1] > w[0]);
    }

    #[test]
    fn build_kernel_floors_the_radius_at_one() {
        // A zero radius still produces a single valid tap rather than an
        // all-zero kernel or a division by zero.
        let w = build_kernel(0);
        close(w[0], 1.0);
        assert_eq!(w[1], 0.0);
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

    /// The only test that ran `blur` used radius 0, which returns the input
    /// unchanged before touching the GPU — so the two-pass separable blur, its
    /// buffer usages and its dispatch arithmetic were entirely unexercised.
    #[test]
    fn blur_softens_a_hard_edge_in_both_directions() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping gaussian_blur test: no GPU");
            return;
        };
        let blurrer = GpuGaussianBlur::new(ctx).expect("init");

        // A single white pixel on black. After blurring, its energy must spread
        // to both horizontal and vertical neighbours — a blur that only runs one
        // of the two passes leaves one axis untouched.
        let mut img = RgbaImage::from_pixel(15, 15, image::Rgba([0, 0, 0, 255]));
        img.put_pixel(7, 7, image::Rgba([255, 255, 255, 255]));
        let image = DynamicImage::ImageRgba8(img);

        let result = blurrer.blur(&image, 2.0).expect("blur");
        crate::gpu::test_support::assert_changed(&result, &image, "blur");

        let out = result.to_rgba8();
        let centre = out.get_pixel(7, 7)[0];
        let left = out.get_pixel(6, 7)[0];
        let right = out.get_pixel(8, 7)[0];
        let above = out.get_pixel(7, 6)[0];
        let below = out.get_pixel(7, 8)[0];

        assert!(centre < 255, "the peak must be attenuated, got {centre}");
        assert!(
            left > 0 && right > 0,
            "horizontal pass did not spread: {left}, {right}"
        );
        assert!(
            above > 0 && below > 0,
            "vertical pass did not spread: {above}, {below}"
        );
        // Separable Gaussian is symmetric, so opposite neighbours must match.
        assert_eq!(left, right, "blur must be horizontally symmetric");
        assert_eq!(above, below, "blur must be vertically symmetric");
    }

    #[test]
    fn blur_preserves_a_uniform_image_and_covers_odd_dimensions() {
        let Some(ctx) = test_context() else {
            eprintln!("Skipping gaussian_blur test: no GPU");
            return;
        };
        let blurrer = GpuGaussianBlur::new(ctx).expect("init");

        // Blurring a flat field is the identity up to rounding, which pins the
        // kernel normalisation: a kernel that does not sum to 1 shifts the level.
        // Odd, non-workgroup-multiple size so a mis-rounded dispatch shows up.
        let image = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            37,
            19,
            image::Rgba([120, 120, 120, 255]),
        ));
        let result = blurrer.blur(&image, 3.0).expect("blur");
        crate::gpu::test_support::assert_plausible_output(&result, &image, "blur (uniform)");

        for px in result.to_rgba8().pixels() {
            for ch in 0..3 {
                let d = px[ch] as i16 - 120;
                assert!(
                    d.abs() <= 1,
                    "uniform field must survive blurring, got {} vs 120",
                    px[ch]
                );
            }
        }
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
        assert!(blurrer.memory_usage() > 0, "a pass pools its buffers");
        blurrer.clear_cache();
        assert_eq!(blurrer.memory_usage(), 0, "clear_cache must release them");
    }
}
