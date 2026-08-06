//! Preprocessing utilities for preparing images for YuNet inference.
//!
//! The helpers in this module resize images, convert them into the expected tensor layout, and
//! return the scale factors necessary to map detections back to the source image.

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable, bytes_of};
use fcs_utils::{
    compute_resize_scales,
    config::{InputDimensions, ResizeQuality},
    gpu::{GpuContext, PREPROCESS_WGSL},
    load_image, resize_image, rgb_to_bgr_chw,
    telemetry::timing_guard,
};
use image::{DynamicImage, GenericImageView, RgbImage, imageops::FilterType};
use std::{
    borrow::Cow,
    path::Path,
    sync::{Arc, Mutex, mpsc},
};
use tract_onnx::prelude::{IntoTensor, Tensor, tract_ndarray};

/// Desired input resolution for YuNet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InputSize {
    /// The width of the input tensor.
    pub width: u32,
    /// The height of the input tensor.
    pub height: u32,
}

impl InputSize {
    /// Creates a new `InputSize`.
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

impl Default for InputSize {
    fn default() -> Self {
        Self {
            width: 640,
            height: 640,
        }
    }
}

/// Configuration for preprocessing an image before inference.
#[derive(Debug, Clone, Default)]
pub struct PreprocessConfig {
    /// The target input size for the model.
    pub input_size: InputSize,
    /// Resize filter preference controlling the quality vs speed trade-off.
    pub resize_quality: ResizeQuality,
}

impl PreprocessConfig {
    fn resize_filter(&self) -> FilterType {
        match self.resize_quality {
            ResizeQuality::Quality => FilterType::Triangle,
            ResizeQuality::Speed => FilterType::Nearest,
        }
    }
}

/// Output of preprocessing: tensor plus metadata for rescaling detections.
#[derive(Debug)]
pub struct PreprocessOutput {
    /// The preprocessed image tensor, ready for inference.
    pub tensor: Tensor,
    /// The horizontal scale factor to convert detection coordinates to the original image space.
    pub scale_x: f32,
    /// The vertical scale factor to convert detection coordinates to the original image space.
    pub scale_y: f32,
    /// The original dimensions of the input image.
    pub original_size: (u32, u32),
}

/// Preprocess an image file into a YuNet-ready tensor in `[1, 3, H, W]` (CHW) BGR format matching OpenCV's `blobFromImage`.
///
/// # Arguments
///
/// * `path` - The path to the image file.
/// * `config` - The configuration for preprocessing.
pub fn preprocess_image<P: AsRef<Path>>(
    path: P,
    config: &PreprocessConfig,
) -> Result<PreprocessOutput> {
    let default_cpu = CpuPreprocessor;
    preprocess_image_with(&default_cpu, path, config)
}

/// Preprocess an image from disk using a specific preprocessor implementation.
///
/// This is primarily useful for injecting GPU-backed preprocessors in tests/benchmarks.
pub fn preprocess_image_with<P, T>(
    preprocessor: &T,
    path: P,
    config: &PreprocessConfig,
) -> Result<PreprocessOutput>
where
    P: AsRef<Path>,
    T: Preprocessor + ?Sized,
{
    let _guard = timing_guard("fcs_core::preprocess_image", log::Level::Debug);
    let path_ref = path.as_ref();
    anyhow::ensure!(
        path_ref.exists(),
        "input image does not exist: {}",
        path_ref.display()
    );

    let image = load_image(path_ref)
        .with_context(|| format!("failed to load image from {}", path_ref.display()))?;
    preprocessor.preprocess(&image, config)
}

/// Preprocess an in-memory image (useful for tests).
///
/// # Arguments
///
/// * `image` - The dynamic image to process.
/// * `config` - The configuration for preprocessing.
pub fn preprocess_dynamic_image(
    image: &DynamicImage,
    config: &PreprocessConfig,
) -> Result<PreprocessOutput> {
    let cpu = CpuPreprocessor;
    cpu.preprocess(image, config)
}

impl From<InputDimensions> for InputSize {
    fn from(dimensions: InputDimensions) -> Self {
        InputSize::new(dimensions.width, dimensions.height)
    }
}

impl From<&InputDimensions> for InputSize {
    fn from(dimensions: &InputDimensions) -> Self {
        (*dimensions).into()
    }
}

impl From<InputDimensions> for PreprocessConfig {
    fn from(dimensions: InputDimensions) -> Self {
        let InputDimensions {
            width,
            height,
            resize_quality,
        } = dimensions;
        PreprocessConfig {
            input_size: InputSize::new(width, height),
            resize_quality,
        }
    }
}

impl From<&InputDimensions> for PreprocessConfig {
    fn from(dimensions: &InputDimensions) -> Self {
        PreprocessConfig {
            input_size: (*dimensions).into(),
            resize_quality: dimensions.resize_quality,
        }
    }
}

/// Abstraction over preprocessing backends (CPU, GPU).
pub trait Preprocessor: Send + Sync + std::fmt::Debug {
    /// Convert the provided dynamic image into a YuNet-ready tensor.
    fn preprocess(
        &self,
        image: &DynamicImage,
        config: &PreprocessConfig,
    ) -> Result<PreprocessOutput>;
}

/// Default CPU implementation backed by `image` + ndarray utilities.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuPreprocessor;

impl Preprocessor for CpuPreprocessor {
    fn preprocess(
        &self,
        image: &DynamicImage,
        config: &PreprocessConfig,
    ) -> Result<PreprocessOutput> {
        cpu_preprocess(image, config)
    }
}

fn cpu_preprocess(image: &DynamicImage, config: &PreprocessConfig) -> Result<PreprocessOutput> {
    let _guard = timing_guard("fcs_core::preprocess_dynamic_image", log::Level::Trace);
    let input_w = config.input_size.width;
    let input_h = config.input_size.height;
    anyhow::ensure!(
        input_w > 0 && input_h > 0,
        "input dimensions must be greater than zero"
    );

    let (orig_w, orig_h) = image.dimensions();
    anyhow::ensure!(
        orig_w > 0 && orig_h > 0,
        "source image dimensions must be greater than zero"
    );
    let resized_rgb: Cow<'_, RgbImage> = if orig_w == input_w && orig_h == input_h {
        match image.as_rgb8() {
            Some(rgb) => Cow::Borrowed(rgb),
            None => Cow::Owned(image.to_rgb8()),
        }
    } else {
        Cow::Owned(resize_image(
            image,
            input_w,
            input_h,
            config.resize_filter(),
        ))
    };
    let data = rgb_to_bgr_chw(&resized_rgb);
    let tensor = chw_tensor_from_vec(data, input_w, input_h)?;

    let (scale_x, scale_y) = compute_resize_scales((orig_w, orig_h), (input_w, input_h))?;

    Ok(PreprocessOutput {
        tensor,
        scale_x,
        scale_y,
        original_size: (orig_w, orig_h),
    })
}

/// GPU-backed preprocessor that uses `wgpu` compute shaders for resize + color conversion.
#[derive(Clone)]
pub struct WgpuPreprocessor {
    context: Arc<GpuContext>,
    pipeline: Arc<WgpuPreprocessPipeline>,
    pool: Arc<Mutex<GpuResourcePool>>,
}

impl std::fmt::Debug for WgpuPreprocessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuPreprocessor")
            .field("adapter", self.context.adapter_info())
            .finish()
    }
}

impl WgpuPreprocessor {
    /// Create a GPU preprocessor from an existing `GpuContext`.
    pub fn new(context: Arc<GpuContext>) -> Result<Self> {
        let pipeline = WgpuPreprocessPipeline::new(context.device())?;
        Ok(Self {
            context,
            pipeline: Arc::new(pipeline),
            pool: Arc::new(Mutex::new(GpuResourcePool::default())),
        })
    }
}

impl Preprocessor for WgpuPreprocessor {
    fn preprocess(
        &self,
        image: &DynamicImage,
        config: &PreprocessConfig,
    ) -> Result<PreprocessOutput> {
        gpu_preprocess(
            image,
            config,
            self.context.as_ref(),
            &self.pipeline,
            self.pool.as_ref(),
        )
    }
}

struct WgpuPreprocessPipeline {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    sampler: wgpu::Sampler,
}

impl WgpuPreprocessPipeline {
    fn new(device: &wgpu::Device) -> Result<Self> {
        // Panics if WGSL compilation fails; the label appears in the panic message.
        // If this panics, inspect preprocess.wgsl and verify wgpu feature support.
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("preprocess.wgsl shader"),
            source: wgpu::ShaderSource::Wgsl(PREPROCESS_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("preprocess_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("preprocess_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("preprocess.wgsl pipeline — check wgpu feature support if this panics"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("preprocess_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        Ok(Self {
            bind_group_layout,
            pipeline,
            sampler,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PreprocessUniforms {
    src_size: [u32; 2],
    dst_size: [u32; 2],
}

#[derive(Default)]
struct GpuResourcePool {
    idle: Vec<GpuWorkBuffers>,
}

struct GpuWorkBuffers {
    texture: wgpu::Texture,
    extent: wgpu::Extent3d,
    storage: wgpu::Buffer,
    storage_size: u64,
    readback: wgpu::Buffer,
    readback_size: u64,
    uniform: wgpu::Buffer,
    staging: Vec<u8>,
}

const UNIFORM_BUFFER_SIZE: u64 = std::mem::size_of::<PreprocessUniforms>() as u64;

impl GpuResourcePool {
    fn acquire(
        &mut self,
        device: &wgpu::Device,
        extent: wgpu::Extent3d,
        output_bytes: u64,
    ) -> GpuWorkBuffers {
        if let Some(mut buffers) = self.idle.pop() {
            buffers.ensure_texture(device, extent);
            buffers.ensure_output_buffers(device, output_bytes);
            buffers
        } else {
            GpuWorkBuffers::new(device, extent, output_bytes)
        }
    }

    fn recycle(&mut self, mut buffers: GpuWorkBuffers) {
        // Shrink oversized staging buffers to avoid carrying high-water-mark
        // allocations across batch items (a single large image can grow the
        // staging Vec to ~100 MB which is then never freed).
        const STAGING_SHRINK_THRESHOLD: usize = 1 << 24; // 16 MB
        if buffers.staging.capacity() > STAGING_SHRINK_THRESHOLD {
            buffers.staging = Vec::new();
        }
        self.idle.push(buffers);
    }
}

impl GpuWorkBuffers {
    fn new(device: &wgpu::Device, extent: wgpu::Extent3d, output_bytes: u64) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("preprocess_input_texture"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let storage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("preprocess_output_storage"),
            size: output_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("preprocess_readback"),
            size: output_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("preprocess_uniforms"),
            size: UNIFORM_BUFFER_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            texture,
            extent,
            storage,
            storage_size: output_bytes,
            readback,
            readback_size: output_bytes,
            uniform,
            staging: Vec::new(),
        }
    }

    fn ensure_texture(&mut self, device: &wgpu::Device, extent: wgpu::Extent3d) {
        if self.extent == extent {
            return;
        }
        self.texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("preprocess_input_texture"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.extent = extent;
    }

    fn ensure_output_buffers(&mut self, device: &wgpu::Device, size: u64) {
        if self.storage_size < size {
            self.storage = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("preprocess_output_storage"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            self.storage_size = size;
        }
        if self.readback_size < size {
            self.readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("preprocess_readback"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            self.readback_size = size;
        }
    }

    fn uniform_buffer(&self) -> &wgpu::Buffer {
        &self.uniform
    }

    fn storage_buffer(&self) -> &wgpu::Buffer {
        &self.storage
    }

    fn readback_buffer(&self) -> &wgpu::Buffer {
        &self.readback
    }

    fn prepare_upload<'a>(&'a mut self, data: &'a [u8], width: u32) -> (&'a [u8], u32) {
        let bytes_per_row = 4 * width as usize;
        let aligned = align_to(bytes_per_row, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize);
        if aligned == bytes_per_row {
            return (data, bytes_per_row as u32);
        }

        let rows = data.len() / bytes_per_row;
        let required = aligned * rows;
        self.staging.resize(required, 0);
        for row in 0..rows {
            let src_start = row * bytes_per_row;
            let dst_start = row * aligned;
            self.staging[dst_start..dst_start + bytes_per_row]
                .copy_from_slice(&data[src_start..src_start + bytes_per_row]);
        }
        (self.staging.as_slice(), aligned as u32)
    }
}

fn gpu_preprocess(
    image: &DynamicImage,
    config: &PreprocessConfig,
    context: &GpuContext,
    pipeline: &WgpuPreprocessPipeline,
    pool: &Mutex<GpuResourcePool>,
) -> Result<PreprocessOutput> {
    let input_w = config.input_size.width;
    let input_h = config.input_size.height;
    anyhow::ensure!(
        input_w > 0 && input_h > 0,
        "input dimensions must be greater than zero"
    );

    let (orig_w, orig_h) = image.dimensions();
    let device = context.device();
    let queue = context.queue();

    // The source image is uploaded as a single wgpu texture, which is capped at the
    // device's max 2D texture dimension (commonly 8192). Full-resolution camera RAWs
    // routinely exceed this, so fall back to CPU preprocessing rather than tripping a
    // fatal wgpu validation error. CPU resize handles arbitrarily large inputs.
    let max_dim = device.limits().max_texture_dimension_2d;
    if orig_w > max_dim || orig_h > max_dim {
        log::debug!(
            "source {orig_w}x{orig_h} exceeds GPU max texture dimension {max_dim}; using CPU preprocess"
        );
        return cpu_preprocess(image, config);
    }

    let rgba = image.to_rgba8();

    let src_size = wgpu::Extent3d {
        width: orig_w,
        height: orig_h,
        depth_or_array_layers: 1,
    };

    let output_pixels = (input_w * input_h) as usize;
    let output_f32_len = output_pixels * 3;
    let output_size_bytes = (output_f32_len * std::mem::size_of::<f32>()) as u64;

    let mut pool_guard = pool
        .lock()
        .map_err(|_| anyhow::anyhow!("GPU resource pool lock was poisoned"))?;
    let mut buffers = pool_guard.acquire(device, src_size, output_size_bytes);
    drop(pool_guard);

    let texture_handle = buffers.texture.clone();
    let texture_view = texture_handle.create_view(&wgpu::TextureViewDescriptor::default());
    let storage_buffer = buffers.storage_buffer().clone();
    let readback_buffer = buffers.readback_buffer().clone();
    let uniform_buffer = buffers.uniform_buffer().clone();

    let (input_bytes, bytes_per_row) = buffers.prepare_upload(rgba.as_raw(), orig_w);
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture_handle,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        input_bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(bytes_per_row),
            rows_per_image: Some(orig_h),
        },
        src_size,
    );
    let uniforms = PreprocessUniforms {
        src_size: [orig_w, orig_h],
        dst_size: [input_w, input_h],
    };
    queue.write_buffer(&uniform_buffer, 0, bytes_of(&uniforms));

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("preprocess_bind_group"),
        layout: &pipeline.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&pipeline.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: storage_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("preprocess_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("preprocess_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups_x = input_w.div_ceil(8);
        let workgroups_y = input_h.div_ceil(8);
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }
    encoder.copy_buffer_to_buffer(&storage_buffer, 0, &readback_buffer, 0, output_size_bytes);
    queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = readback_buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = sender.send(res);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .map_err(|e| anyhow::anyhow!("device poll failed during preprocessing: {e}"))?;
    receiver
        .recv()
        .map_err(|_| anyhow::anyhow!("GPU map callback was dropped"))?
        .map_err(|e| anyhow::anyhow!("failed to map GPU preprocessing buffer: {e}"))?;
    let data = buffer_slice
        .get_mapped_range()
        .map_err(|e| anyhow::anyhow!("failed to read mapped GPU preprocessing buffer: {e}"))?;
    let floats: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    readback_buffer.unmap();

    let mut pool_guard = pool
        .lock()
        .map_err(|_| anyhow::anyhow!("GPU resource pool lock was poisoned"))?;
    pool_guard.recycle(buffers);

    anyhow::ensure!(
        floats.len() == output_f32_len,
        "unexpected GPU output size (expected {}, got {})",
        output_f32_len,
        floats.len()
    );

    let tensor = chw_tensor_from_vec(floats, input_w, input_h)?;

    let (scale_x, scale_y) = compute_resize_scales((orig_w, orig_h), (input_w, input_h))?;

    Ok(PreprocessOutput {
        tensor,
        scale_x,
        scale_y,
        original_size: (orig_w, orig_h),
    })
}

/// Build a `[1, 3, H, W]` tensor by taking ownership of an existing CHW
/// buffer. `Array4::from_shape_vec` + `into_tensor` reuse the allocation,
/// unlike `Tensor::from_shape`, which copies the slice.
fn chw_tensor_from_vec(data: Vec<f32>, input_w: u32, input_h: u32) -> Result<Tensor> {
    let array =
        tract_ndarray::Array4::from_shape_vec((1, 3, input_h as usize, input_w as usize), data)
            .map_err(|e| anyhow::anyhow!("failed to build tensor: {e}"))?;
    Ok(array.into_tensor())
}

fn align_to(value: usize, alignment: usize) -> usize {
    debug_assert!(
        alignment.is_power_of_two(),
        "alignment must be a power of two for bitwise optimization"
    );
    // Optimized for power-of-2 alignment using bitwise operations
    // (value + alignment - 1) & !(alignment - 1)
    (value + alignment - 1) & !(alignment - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fcs_utils::config::{InputDimensions, ResizeQuality};
    use image::{ImageBuffer, Rgb};

    #[test]
    fn preprocess_generates_bgr_tensor() {
        let mut img = ImageBuffer::<Rgb<u8>, _>::new(4, 4);
        for (x, y, pixel) in img.enumerate_pixels_mut() {
            let value = ((x + y) * 32) as u8;
            *pixel = Rgb([value, value / 2, 255]);
        }

        let dynamic = DynamicImage::ImageRgb8(img);
        let config = PreprocessConfig {
            input_size: InputSize::new(2, 2),
            ..Default::default()
        };

        let output =
            preprocess_dynamic_image(&dynamic, &config).expect("preprocess should succeed");

        assert_eq!(output.original_size, (4, 4));
        assert_eq!(output.scale_x, 2.0);
        assert_eq!(output.scale_y, 2.0);
        assert_eq!(output.tensor.shape(), &[1, 3, 2, 2]);

        let data = output
            .tensor
            .try_as_plain()
            .and_then(|view| view.as_slice::<f32>())
            .unwrap();
        assert!(data.iter().all(|v| *v >= 0.0 && *v <= 255.0));
    }

    /// Read a plain `f32` tensor back out as a slice.
    fn tensor_data(output: &PreprocessOutput) -> Vec<f32> {
        output
            .tensor
            .try_as_plain()
            .and_then(|view| view.as_slice::<f32>())
            .expect("tensor should be plain f32")
            .to_vec()
    }

    #[test]
    fn preprocess_lays_out_planar_bgr_not_rgb() {
        // Input size matches the image, so no resampling stands between the
        // source pixels and the tensor and the expected values are exact.
        //
        // YuNet wants `[1, 3, H, W]` with the channels in B, G, R order — the
        // layout OpenCV's blobFromImage produces. `preprocess_generates_bgr_tensor`
        // only checks every value is in 0..=255, which holds just as well for
        // RGB order, an interleaved layout, or transposed rows.
        let mut img = ImageBuffer::<Rgb<u8>, _>::new(2, 2);
        img.put_pixel(0, 0, Rgb([10, 20, 30]));
        img.put_pixel(1, 0, Rgb([40, 50, 60]));
        img.put_pixel(0, 1, Rgb([70, 80, 90]));
        img.put_pixel(1, 1, Rgb([100, 110, 120]));

        let config = PreprocessConfig {
            input_size: InputSize::new(2, 2),
            ..Default::default()
        };
        let out = preprocess_dynamic_image(&DynamicImage::ImageRgb8(img), &config)
            .expect("preprocess should succeed");

        assert_eq!(out.tensor.shape(), &[1, 3, 2, 2]);
        assert_eq!(
            tensor_data(&out),
            vec![
                30.0, 60.0, 90.0, 120.0, // blue plane, row-major
                20.0, 50.0, 80.0, 110.0, // green plane
                10.0, 40.0, 70.0, 100.0, // red plane
            ]
        );
    }

    #[test]
    fn preprocess_converts_non_rgb_sources_without_reordering() {
        // A luma source has to be widened to RGB first. Every channel ends up
        // equal, so this pins the conversion rather than the channel order:
        // a dropped conversion would panic or produce the wrong length.
        let img = DynamicImage::ImageLuma8(ImageBuffer::from_fn(2, 1, |x, _| {
            image::Luma([if x == 0 { 64u8 } else { 192 }])
        }));
        let config = PreprocessConfig {
            input_size: InputSize::new(2, 1),
            ..Default::default()
        };
        let out = preprocess_dynamic_image(&img, &config).expect("luma should preprocess");

        assert_eq!(out.tensor.shape(), &[1, 3, 1, 2]);
        assert_eq!(
            tensor_data(&out),
            vec![64.0, 192.0, 64.0, 192.0, 64.0, 192.0]
        );
    }

    #[test]
    fn preprocess_scales_report_the_source_to_input_ratio() {
        // Non-square and non-integer ratios in one go: 30x8 -> 4x16 means
        // scale_x = 30/4 = 7.5 and scale_y = 8/16 = 0.5. A swapped axis or an
        // inverted ratio is indistinguishable when both factors are 2.0, which
        // is all the existing tests use.
        let img = DynamicImage::ImageRgb8(ImageBuffer::from_fn(30, 8, |_, _| Rgb([1u8, 2, 3])));
        let config = PreprocessConfig {
            input_size: InputSize::new(4, 16),
            ..Default::default()
        };
        let out = preprocess_dynamic_image(&img, &config).expect("preprocess should succeed");

        assert_eq!(out.original_size, (30, 8));
        assert_eq!(out.scale_x, 7.5);
        assert_eq!(out.scale_y, 0.5);
        assert_eq!(out.tensor.shape(), &[1, 3, 16, 4]);
    }

    #[test]
    fn cpu_preprocess_resizes_when_only_one_dimension_already_matches() {
        // The skip-resize fast path needs *both* dimensions to match. With
        // only the width matching, treating the condition as an `or` would
        // pass a 4x8 image off as a 4x16 tensor and the shape would not fit.
        let img = DynamicImage::ImageRgb8(ImageBuffer::from_fn(4, 8, |x, _| {
            Rgb([(x * 60) as u8, 10, 20])
        }));
        let config = PreprocessConfig {
            input_size: InputSize::new(4, 16),
            ..Default::default()
        };
        let out = preprocess_dynamic_image(&img, &config).expect("should resize the height");
        assert_eq!(out.tensor.shape(), &[1, 3, 16, 4]);
        assert_eq!(tensor_data(&out).len(), 3 * 16 * 4);
    }

    #[test]
    fn cpu_preprocess_errors_for_zero_source_dimension() {
        // The zero-input-dimension guard is covered below; this is the other
        // arm, where the *image* is degenerate.
        let img = DynamicImage::ImageRgb8(ImageBuffer::new(0, 4));
        let config = PreprocessConfig {
            input_size: InputSize::new(8, 8),
            ..Default::default()
        };
        assert!(preprocess_dynamic_image(&img, &config).is_err());
    }

    #[test]
    fn cpu_preprocess_errors_for_zero_input_height() {
        // `input_w > 0 && input_h > 0` needs both arms exercised; the existing
        // test only zeroes the width.
        let img = DynamicImage::ImageRgb8(ImageBuffer::from_fn(4, 4, |_, _| Rgb([1u8, 2, 3])));
        let config = PreprocessConfig {
            input_size: InputSize::new(32, 0),
            ..Default::default()
        };
        assert!(preprocess_dynamic_image(&img, &config).is_err());
    }

    #[test]
    fn align_to_leaves_small_alignments_alone() {
        // The existing coverage is all 256-byte alignment, where `& !(a - 1)`
        // and a plain round-up agree on every probe used.
        assert_eq!(align_to(5, 4), 8);
        assert_eq!(align_to(8, 4), 8);
        assert_eq!(align_to(9, 8), 16);
        assert_eq!(align_to(3, 1), 3);
        assert_eq!(align_to(1, 2), 2);
    }

    #[test]
    fn chw_tensor_from_vec_rejects_a_mismatched_buffer() {
        // 3 * 2 * 2 = 12 elements are required.
        assert!(chw_tensor_from_vec(vec![0.0; 11], 2, 2).is_err());
        assert!(chw_tensor_from_vec(vec![0.0; 13], 2, 2).is_err());
        assert!(chw_tensor_from_vec(vec![0.0; 12], 2, 2).is_ok());
    }

    #[test]
    fn input_size_default_is_the_yunet_resolution() {
        assert_eq!(InputSize::default(), InputSize::new(640, 640));
        // PreprocessConfig::default has to inherit it rather than zero it.
        assert_eq!(
            PreprocessConfig::default().input_size,
            InputSize::new(640, 640)
        );
    }

    #[test]
    fn converts_dimensions_into_configs() {
        let dims = InputDimensions {
            width: 320,
            height: 240,
            resize_quality: ResizeQuality::Quality,
        };

        let size: InputSize = dims.into();
        assert_eq!(size.width, 320);
        assert_eq!(size.height, 240);

        let config: PreprocessConfig = dims.into();
        assert_eq!(config.input_size.width, 320);
        assert_eq!(config.input_size.height, 240);
    }

    #[test]
    fn align_to_power_of_two() {
        // Already aligned
        assert_eq!(align_to(256, 256), 256);
        assert_eq!(align_to(512, 256), 512);
        // Rounds up
        assert_eq!(align_to(1, 256), 256);
        assert_eq!(align_to(257, 256), 512);
        assert_eq!(align_to(0, 256), 0);
    }

    #[test]
    fn preprocess_image_reads_from_disk() {
        use std::path::PathBuf;
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let path: PathBuf = dir.path().join("test.png");

        let img = ImageBuffer::<Rgb<u8>, _>::from_fn(64, 64, |x, y| {
            Rgb([((x + y) % 255) as u8, 100, 50])
        });
        DynamicImage::ImageRgb8(img).save(&path).unwrap();

        let config = PreprocessConfig {
            input_size: InputSize::new(32, 32),
            ..Default::default()
        };
        let out = preprocess_image(&path, &config).expect("should preprocess from disk");
        assert_eq!(out.original_size, (64, 64));
        assert_eq!(out.scale_x, 2.0);
        assert_eq!(out.scale_y, 2.0);
        assert_eq!(out.tensor.shape(), &[1, 3, 32, 32]);
    }

    #[test]
    fn preprocess_image_errors_for_missing_file() {
        let config = PreprocessConfig::default();
        let result = preprocess_image("does_not_exist_xyz.png", &config);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("does not exist") || msg.contains("does_not_exist"));
    }

    #[test]
    fn cpu_preprocess_errors_for_zero_input_dimension() {
        let img = DynamicImage::ImageRgb8(ImageBuffer::from_fn(4, 4, |_, _| Rgb([1u8, 2, 3])));
        let config = PreprocessConfig {
            input_size: InputSize::new(0, 32),
            ..Default::default()
        };
        assert!(preprocess_dynamic_image(&img, &config).is_err());
    }

    #[test]
    fn cpu_preprocess_skips_resize_when_image_already_matches_input_size() {
        let img =
            DynamicImage::ImageRgb8(ImageBuffer::from_fn(32, 32, |_, _| Rgb([100u8, 150, 200])));
        let config = PreprocessConfig {
            input_size: InputSize::new(32, 32),
            ..Default::default()
        };
        let out = preprocess_dynamic_image(&img, &config).unwrap();
        assert_eq!(out.original_size, (32, 32));
        assert_eq!(out.scale_x, 1.0);
        assert_eq!(out.scale_y, 1.0);
    }

    #[test]
    fn resize_quality_speed_yields_nearest_filter() {
        use fcs_utils::config::ResizeQuality;
        use image::imageops::FilterType;
        let config = PreprocessConfig {
            input_size: InputSize::new(32, 32),
            resize_quality: ResizeQuality::Speed,
        };
        assert_eq!(config.resize_filter(), FilterType::Nearest);
    }

    #[test]
    fn resize_quality_quality_yields_triangle_filter() {
        use fcs_utils::config::ResizeQuality;
        use image::imageops::FilterType;
        let config = PreprocessConfig {
            input_size: InputSize::new(32, 32),
            resize_quality: ResizeQuality::Quality,
        };
        assert_eq!(config.resize_filter(), FilterType::Triangle);
    }

    #[test]
    fn preprocess_image_with_accepts_custom_preprocessor() {
        use std::path::PathBuf;
        use tempfile::tempdir;

        #[derive(Debug)]
        struct DelegatingPreprocessor;

        impl Preprocessor for DelegatingPreprocessor {
            fn preprocess(
                &self,
                image: &DynamicImage,
                config: &PreprocessConfig,
            ) -> anyhow::Result<PreprocessOutput> {
                CpuPreprocessor.preprocess(image, config)
            }
        }

        let dir = tempdir().unwrap();
        let path: PathBuf = dir.path().join("test.png");
        let img = ImageBuffer::<Rgb<u8>, _>::from_fn(16, 16, |x, y| {
            Rgb([((x + y) % 255) as u8, 100, 50])
        });
        DynamicImage::ImageRgb8(img).save(&path).unwrap();

        let config = PreprocessConfig {
            input_size: InputSize::new(8, 8),
            ..Default::default()
        };
        let out = preprocess_image_with(&DelegatingPreprocessor, &path, &config).unwrap();
        assert_eq!(out.original_size, (16, 16));
        assert_eq!(out.scale_x, 2.0);
        assert_eq!(out.scale_y, 2.0);
        assert_eq!(out.tensor.shape(), &[1, 3, 8, 8]);
    }

    #[test]
    fn from_ref_input_dimensions_for_input_size() {
        use fcs_utils::config::ResizeQuality;
        let dims = InputDimensions {
            width: 128,
            height: 96,
            resize_quality: ResizeQuality::Speed,
        };
        let by_ref: InputSize = (&dims).into();
        assert_eq!(by_ref.width, 128);
        assert_eq!(by_ref.height, 96);
        // owned conversion must agree
        let by_owned: InputSize = dims.into();
        assert_eq!(by_ref.width, by_owned.width);
        assert_eq!(by_ref.height, by_owned.height);
    }

    #[test]
    fn from_ref_input_dimensions_for_preprocess_config() {
        use fcs_utils::config::ResizeQuality;
        let dims = InputDimensions {
            width: 320,
            height: 240,
            resize_quality: ResizeQuality::Speed,
        };
        let cfg: PreprocessConfig = (&dims).into();
        assert_eq!(cfg.input_size.width, 320);
        assert_eq!(cfg.input_size.height, 240);
        assert!(matches!(cfg.resize_quality, ResizeQuality::Speed));
    }

    #[test]
    fn cpu_preprocessor_trait_matches_helpers() {
        let mut img = ImageBuffer::<Rgb<u8>, _>::new(2, 2);
        for (i, pixel) in img.pixels_mut().enumerate() {
            *pixel = Rgb([(i * 10) as u8, 0, 255]);
        }
        let dynamic = DynamicImage::ImageRgb8(img);
        let config = PreprocessConfig {
            input_size: InputSize::new(2, 2),
            ..Default::default()
        };

        let cpu = CpuPreprocessor;
        let trait_output = cpu.preprocess(&dynamic, &config).expect("trait preprocess");
        let helper_output =
            preprocess_dynamic_image(&dynamic, &config).expect("function preprocess");

        assert_eq!(trait_output.original_size, helper_output.original_size);
        assert_eq!(trait_output.scale_x, helper_output.scale_x);
        assert_eq!(trait_output.scale_y, helper_output.scale_y);
        assert_eq!(trait_output.tensor.shape(), helper_output.tensor.shape());

        let trait_data = trait_output
            .tensor
            .try_as_plain()
            .and_then(|view| view.as_slice::<f32>())
            .unwrap();
        let helper_data = helper_output
            .tensor
            .try_as_plain()
            .and_then(|view| view.as_slice::<f32>())
            .unwrap();
        assert_eq!(trait_data, helper_data);
    }
}
