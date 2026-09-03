//! Preprocessing utilities for preparing images for YuNet inference.
//!
//! The helpers in this module resize images, convert them into the expected tensor layout, and
//! return the scale factors necessary to map detections back to the source image.

use crate::gpu::tensor::GpuTensor;
use crate::tensor::Tensor;
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

    /// The GPU implementation, when this preprocessor is one.
    ///
    /// Lets a detector pair a GPU preprocessor with GPU inference on the same device and keep
    /// the tensor there, instead of downloading it and uploading it straight back.
    fn as_wgpu(&self) -> Option<&WgpuPreprocessor> {
        None
    }
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

/// Scale factors from a preprocessing run whose tensor stayed on the GPU.
#[derive(Debug, Clone, Copy)]
pub struct PreprocessScales {
    /// Horizontal factor mapping detection coordinates back to the source image.
    pub scale_x: f32,
    /// Vertical factor mapping detection coordinates back to the source image.
    pub scale_y: f32,
    /// Dimensions of the source image.
    pub original_size: (u32, u32),
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
    /// The device this preprocessor renders on.
    ///
    /// Used to check that inference shares it before the two are fused; tensors cannot cross
    /// `wgpu::Device` boundaries.
    pub fn context(&self) -> &Arc<GpuContext> {
        &self.context
    }

    /// Preprocess directly into `output`, leaving the result on the GPU.
    ///
    /// `output` must be a `[1, 3, height, width]` tensor on this preprocessor's device. Returns
    /// `Ok(None)` if the image cannot take the GPU path (it is larger than the device's maximum
    /// texture dimension), leaving `output` untouched so the caller can fall back to the CPU.
    pub fn preprocess_into_tensor(
        &self,
        image: &DynamicImage,
        config: &PreprocessConfig,
        output: &GpuTensor,
    ) -> Result<Option<PreprocessScales>> {
        anyhow::ensure!(
            Arc::ptr_eq(output.context(), &self.context),
            "preprocess output tensor belongs to a different GPU context"
        );
        let expected = [
            1,
            3,
            config.input_size.height as usize,
            config.input_size.width as usize,
        ];
        anyhow::ensure!(
            output.shape().dims() == expected,
            "preprocess output tensor has shape {:?}, expected {:?}",
            output.shape().dims(),
            expected
        );

        let result = gpu_preprocess_to_tensor(
            image,
            config,
            self.context.as_ref(),
            &self.pipeline,
            self.pool.as_ref(),
            output,
        )?;
        Ok(result)
    }

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

    fn as_wgpu(&self) -> Option<&WgpuPreprocessor> {
        Some(self)
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

    fn recycle(&mut self, buffers: GpuWorkBuffers) {
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
}

/// Upload `image` and dispatch the resize/convert shader, writing f32 CHW BGR into `output`.
///
/// The work is submitted but **not** waited on: the caller decides whether to read it back or
/// leave it on the device. Callers that keep it on the device must consume it from the same
/// `wgpu::Queue`, which orders submissions, so no host synchronisation is needed between the
/// preprocess dispatch and whatever reads its output.
fn encode_preprocess(
    image: &DynamicImage,
    config: &PreprocessConfig,
    context: &GpuContext,
    pipeline: &WgpuPreprocessPipeline,
    buffers: &GpuWorkBuffers,
    output: &wgpu::Buffer,
) -> Result<PreprocessScales> {
    let input_w = config.input_size.width;
    let input_h = config.input_size.height;
    let (orig_w, orig_h) = image.dimensions();
    let device = context.device();
    let queue = context.queue();

    let rgba = image.to_rgba8();
    let texture_view = buffers
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    // Rows go up tightly packed. `Queue::write_texture` does not require
    // `COPY_BYTES_PER_ROW_ALIGNMENT` -- wgpu-core validates queue writes with alignment off, and
    // only `copy_buffer_to_texture` / `copy_texture_to_buffer` demand it. Padding every row into
    // a staging vec first, as this used to, cost a second full pass over the source (~14 ms on a
    // 2384x4240 image) to satisfy a rule that never applied here.
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &buffers.texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        rgba.as_raw(),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * orig_w),
            rows_per_image: Some(orig_h),
        },
        wgpu::Extent3d {
            width: orig_w,
            height: orig_h,
            depth_or_array_layers: 1,
        },
    );

    let uniforms = PreprocessUniforms {
        src_size: [orig_w, orig_h],
        dst_size: [input_w, input_h],
    };
    queue.write_buffer(buffers.uniform_buffer(), 0, bytes_of(&uniforms));

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
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buffers.uniform_buffer().as_entire_binding(),
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
        pass.dispatch_workgroups(input_w.div_ceil(8), input_h.div_ceil(8), 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    let (scale_x, scale_y) = compute_resize_scales((orig_w, orig_h), (input_w, input_h))?;
    Ok(PreprocessScales {
        scale_x,
        scale_y,
        original_size: (orig_w, orig_h),
    })
}

/// Shared preamble: validate the request and decide whether the GPU can take it at all.
fn gpu_preprocess_setup(
    image: &DynamicImage,
    config: &PreprocessConfig,
    context: &GpuContext,
) -> Result<Option<(wgpu::Extent3d, u64)>> {
    let input_w = config.input_size.width;
    let input_h = config.input_size.height;
    anyhow::ensure!(
        input_w > 0 && input_h > 0,
        "input dimensions must be greater than zero"
    );

    let (orig_w, orig_h) = image.dimensions();

    // The source image is uploaded as a single wgpu texture, which is capped at the
    // device's max 2D texture dimension (commonly 8192). Full-resolution camera RAWs
    // routinely exceed this, so fall back to CPU preprocessing rather than tripping a
    // fatal wgpu validation error. CPU resize handles arbitrarily large inputs.
    let max_dim = context.device().limits().max_texture_dimension_2d;
    if orig_w > max_dim || orig_h > max_dim {
        log::debug!(
            "source {orig_w}x{orig_h} exceeds GPU max texture dimension {max_dim}; using CPU preprocess"
        );
        return Ok(None);
    }

    let output_bytes = ((input_w * input_h) as usize * 3 * std::mem::size_of::<f32>()) as u64;
    Ok(Some((
        wgpu::Extent3d {
            width: orig_w,
            height: orig_h,
            depth_or_array_layers: 1,
        },
        output_bytes,
    )))
}

fn gpu_preprocess(
    image: &DynamicImage,
    config: &PreprocessConfig,
    context: &GpuContext,
    pipeline: &WgpuPreprocessPipeline,
    pool: &Mutex<GpuResourcePool>,
) -> Result<PreprocessOutput> {
    let Some((src_size, output_size_bytes)) = gpu_preprocess_setup(image, config, context)? else {
        return cpu_preprocess(image, config);
    };
    let input_w = config.input_size.width;
    let input_h = config.input_size.height;
    let output_f32_len = (input_w * input_h) as usize * 3;
    let device = context.device();

    let buffers = lock_pool(pool)?.acquire(device, src_size, output_size_bytes);

    let scales = encode_preprocess(
        image,
        config,
        context,
        pipeline,
        &buffers,
        buffers.storage_buffer(),
    )?;

    // The dispatch above is already submitted; copy its output where the host can map it.
    let readback_buffer = buffers.readback_buffer().clone();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("preprocess_readback_encoder"),
    });
    encoder.copy_buffer_to_buffer(
        buffers.storage_buffer(),
        0,
        &readback_buffer,
        0,
        output_size_bytes,
    );
    context.queue().submit(std::iter::once(encoder.finish()));

    // Slice the region this dispatch actually wrote, not the whole buffer.
    // `ensure_output_buffers` only ever grows the pooled buffers, so after a
    // larger tensor has been processed the buffer stays larger -- and mapping all
    // of it made the readback length the pooled capacity rather than the
    // requested size, failing the output-size check below for every subsequent
    // smaller tensor. Reachable by lowering the detection input size while the
    // preprocessor is reused. `gpu/runtime.rs` already sliced explicitly.
    let buffer_slice = readback_buffer.slice(0..output_size_bytes);
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

    lock_pool(pool)?.recycle(buffers);

    anyhow::ensure!(
        floats.len() == output_f32_len,
        "unexpected GPU output size (expected {}, got {})",
        output_f32_len,
        floats.len()
    );

    let tensor = chw_tensor_from_vec(floats, input_w, input_h)?;

    Ok(PreprocessOutput {
        tensor,
        scale_x: scales.scale_x,
        scale_y: scales.scale_y,
        original_size: scales.original_size,
    })
}

/// Preprocessing that leaves its result on the GPU.
///
/// The shader already writes exactly the layout a tensor wants -- f32, CHW, BGR -- so it is
/// pointed straight at `output`s buffer and nothing is copied. This skips the 4.9 MB download
/// and the blocking map that [`gpu_preprocess`] needs, and lets inference consume the result
/// without uploading it again.
///
/// Returns `Ok(None)` when the image cannot go through the GPU path at all, so the caller can
/// fall back rather than have a CPU tensor handed back through a GPU-shaped return type.
fn gpu_preprocess_to_tensor(
    image: &DynamicImage,
    config: &PreprocessConfig,
    context: &GpuContext,
    pipeline: &WgpuPreprocessPipeline,
    pool: &Mutex<GpuResourcePool>,
    output: &GpuTensor,
) -> Result<Option<PreprocessScales>> {
    let Some((src_size, output_size_bytes)) = gpu_preprocess_setup(image, config, context)? else {
        return Ok(None);
    };

    let buffers = lock_pool(pool)?.acquire(context.device(), src_size, output_size_bytes);
    let result = encode_preprocess(image, config, context, pipeline, &buffers, output.buffer())?;
    lock_pool(pool)?.recycle(buffers);
    Ok(Some(result))
}

fn lock_pool(pool: &Mutex<GpuResourcePool>) -> Result<std::sync::MutexGuard<'_, GpuResourcePool>> {
    pool.lock()
        .map_err(|_| anyhow::anyhow!("GPU resource pool lock was poisoned"))
}

fn chw_tensor_from_vec(data: Vec<f32>, input_w: u32, input_h: u32) -> Result<Tensor> {
    Tensor::from_vec(&[1, 3, input_h as usize, input_w as usize], data)
        .context("failed to build the preprocessed tensor")
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

        let data = output.tensor.as_slice();
        assert!(data.iter().all(|v| *v >= 0.0 && *v <= 255.0));
    }

    /// Read a plain `f32` tensor back out as a slice.
    fn tensor_data(output: &PreprocessOutput) -> Vec<f32> {
        output.tensor.as_slice().to_vec()
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

        let trait_data = trait_output.tensor.as_slice();
        let helper_data = helper_output.tensor.as_slice();
        assert_eq!(trait_data, helper_data);
    }
}

/// GPU preprocessing measured against the CPU implementation.
///
/// Every other test in this file exercises the CPU path, which is why the whole
/// `gpu_preprocess` chain carried mutation survivors — no assertion reached it.
/// The CPU path is well covered, so it makes a usable reference.
///
/// One caveat shapes these tests. The shader resamples with
/// `textureSampleLevel(..., 0.0)`: four bilinear taps at mip 0, while the CPU
/// `Triangle` filter averages every contributing source pixel. On minification
/// the two diverge sharply — downscaling 100x40 to 64x64 differs by up to 51 of
/// 255 per channel, against exactly 0 when no resize happens. So exact parity is
/// asserted only where no resampling occurs, which is also where the row
/// alignment padding can be isolated; the resampling paths get structural
/// assertions instead.
#[cfg(test)]
mod gpu_parity_tests {
    use super::*;
    use fcs_utils::gpu::{GpuAvailability, GpuContext, GpuContextOptions};
    use image::{DynamicImage, RgbaImage};

    /// The GPU-native path must produce exactly what the readback path produces.
    ///
    /// `preprocess_into_tensor` skips the download and re-upload by pointing the shader straight
    /// at an inference tensor, so it shares the dispatch with `preprocess` and should differ in
    /// nothing but where the bytes end up. Comparing them keeps a future change to either from
    /// silently diverging -- and the end-to-end parity suite would not catch it, since it would
    /// still pass if the detector quietly fell back to the readback path.
    #[test]
    fn preprocess_into_tensor_matches_the_readback_path() {
        let Some(gpu) = gpu_preprocessor() else {
            eprintln!("Skipping GPU tensor parity test: no adapter");
            return;
        };
        // Non-square and not a workgroup multiple, so a mis-rounded dispatch shows up, and large
        // enough relative to the target that the box filter takes several taps.
        let image = gradient(300, 130);
        let config = PreprocessConfig {
            input_size: InputSize::new(64, 48),
            resize_quality: ResizeQuality::Quality,
        };

        let expected = gpu
            .preprocess(&image, &config)
            .expect("readback preprocess");

        let tensor = GpuTensor::uninitialized(
            gpu.context().clone(),
            vec![1, 3, 48, 64],
            Some("parity_input"),
        )
        .expect("allocate tensor");
        let scales = gpu
            .preprocess_into_tensor(&image, &config, &tensor)
            .expect("gpu-native preprocess")
            .expect("image is small enough for the GPU path");

        assert_eq!(scales.scale_x, expected.scale_x);
        assert_eq!(scales.scale_y, expected.scale_y);
        assert_eq!(scales.original_size, expected.original_size);

        let actual = tensor.to_vec().expect("download tensor");
        let want = expected.tensor.into_vec();
        assert_eq!(actual.len(), want.len(), "tensor length");
        // Same shader, same inputs, no conversion on either route: this should be exact.
        assert_eq!(
            actual, want,
            "gpu-native tensor differs from the readback tensor"
        );
    }

    fn gpu_preprocessor() -> Option<WgpuPreprocessor> {
        match GpuContext::init_with_fallback(&GpuContextOptions::default()) {
            GpuAvailability::Available(ctx) => WgpuPreprocessor::new(ctx).ok(),
            _ => None,
        }
    }

    /// Distinct value per pixel and channel, so a transposed axis or a
    /// mis-strided row cannot coincidentally match.
    fn gradient(width: u32, height: u32) -> DynamicImage {
        let mut img = RgbaImage::new(width, height);
        for (x, y, px) in img.enumerate_pixels_mut() {
            let r = ((x * 7 + y * 13) % 256) as u8;
            let g = ((x * 29 + y * 3) % 256) as u8;
            let b = ((x * 11 + y * 37) % 256) as u8;
            *px = image::Rgba([r, g, b, 255]);
        }
        DynamicImage::ImageRgba8(img)
    }

    fn as_floats(out: &PreprocessOutput) -> Vec<f32> {
        out.tensor.as_slice().to_vec()
    }

    fn config_for(width: u32, height: u32) -> PreprocessConfig {
        PreprocessConfig {
            input_size: InputSize { width, height },
            resize_quality: ResizeQuality::Quality,
        }
    }

    /// With `input_size` equal to the image size no resampling happens, so the
    /// two tensors must be identical — which makes this the test that pins down
    /// the row padding in `prepare_upload`.
    ///
    /// wgpu requires each texture row to be a multiple of 256 bytes:
    ///   -  64 px -> 256 bytes, already aligned, takes the early return
    ///   -  65 px -> 260 bytes, pads to 512, one byte past the boundary
    ///   - 100 px -> 400 bytes, pads to 512, takes the row-copy loop
    ///   -  37 px -> 148 bytes, pads to 256
    #[test]
    fn gpu_matches_cpu_exactly_when_no_resampling_is_needed() {
        let Some(gpu) = gpu_preprocessor() else {
            eprintln!("Skipping GPU preprocess parity test: no adapter");
            return;
        };

        for (w, h) in [(64u32, 64u32), (65, 33), (100, 40), (37, 19), (128, 8)] {
            let image = gradient(w, h);
            let config = config_for(w, h);

            let cpu = preprocess_dynamic_image(&image, &config).expect("cpu preprocess");
            let out = gpu.preprocess(&image, &config).expect("gpu preprocess");

            let g = as_floats(&out);
            let c = as_floats(&cpu);
            assert_eq!(g.len(), c.len(), "{w}x{h}: tensor length must agree");

            let worst = g
                .iter()
                .zip(c.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                worst <= 1.0,
                "{w}x{h} needs no resampling, so the GPU tensor must match the CPU \
                 one; worst channel difference was {worst}. A row stride computed \
                 without the 256-byte alignment shows up here first."
            );
            assert!(
                g.iter().any(|&v| v > 1.0),
                "{w}x{h}: GPU tensor is empty, so nothing was written"
            );
        }
    }

    #[test]
    fn gpu_reports_the_same_geometry_as_cpu_when_resampling() {
        let Some(gpu) = gpu_preprocessor() else {
            eprintln!("Skipping GPU preprocess parity test: no adapter");
            return;
        };
        let config = config_for(64, 64);

        // Values diverge on minification (see the module comment), but the
        // metadata used to map detections back to source coordinates must not.
        for (w, h) in [(100u32, 40u32), (200, 200), (37, 19)] {
            let image = gradient(w, h);
            let cpu = preprocess_dynamic_image(&image, &config).expect("cpu");
            let out = gpu.preprocess(&image, &config).expect("gpu");

            assert_eq!(out.original_size, (w, h), "{w}x{h}: original_size");
            assert_eq!(out.original_size, cpu.original_size);
            assert!(
                (out.scale_x - cpu.scale_x).abs() < 1e-4
                    && (out.scale_y - cpu.scale_y).abs() < 1e-4,
                "{w}x{h}: scales must agree with the CPU path, gpu=({}, {}) cpu=({}, {})",
                out.scale_x,
                out.scale_y,
                cpu.scale_x,
                cpu.scale_y
            );
            assert_eq!(
                as_floats(&out).len(),
                as_floats(&cpu).len(),
                "{w}x{h}: tensor length"
            );
        }
    }

    /// The GPU sampler is built with `FilterMode::Linear` unconditionally, so
    /// `resize_quality` has no effect on the GPU path while it selects Nearest
    /// vs Triangle on the CPU. Recorded as it stands: if the shader ever honours
    /// the setting, update this test rather than let it fail obscurely.
    #[test]
    fn gpu_output_does_not_currently_vary_with_resize_quality() {
        let Some(gpu) = gpu_preprocessor() else {
            eprintln!("Skipping GPU preprocess parity test: no adapter");
            return;
        };
        let image = gradient(100, 40);
        let speed = gpu
            .preprocess(
                &image,
                &PreprocessConfig {
                    input_size: InputSize {
                        width: 64,
                        height: 64,
                    },
                    resize_quality: ResizeQuality::Speed,
                },
            )
            .expect("gpu speed");
        let quality = gpu
            .preprocess(&image, &config_for(64, 64))
            .expect("gpu quality");

        assert_eq!(
            as_floats(&speed),
            as_floats(&quality),
            "the GPU sampler ignores resize_quality; update this test if that changes"
        );
    }

    /// A source wider than the adapter's `max_texture_dimension_2d` cannot be
    /// uploaded as a texture, so `gpu_preprocess` falls back to the CPU instead
    /// of triggering a fatal wgpu validation error. Nothing exercised that guard.
    ///
    /// Only the width is oversized: the check is an `||`, so requiring both
    /// dimensions to exceed the limit would let this case through to the GPU and
    /// fail. Height stays small to keep the allocation to a few hundred KB.
    #[test]
    fn an_oversized_source_falls_back_to_cpu_instead_of_failing() {
        let Some(gpu) = gpu_preprocessor() else {
            eprintln!("Skipping GPU preprocess test: no adapter");
            return;
        };
        let max_dim = gpu.context.device().limits().max_texture_dimension_2d;
        let width = max_dim + 1;
        let height = 8u32;

        let image = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            width,
            height,
            image::Rgba([90, 140, 60, 255]),
        ));
        let config = config_for(64, 64);

        let out = gpu
            .preprocess(&image, &config)
            .expect("an oversized source must fall back to the CPU, not error");

        assert_eq!(out.original_size, (width, height));
        let floats = as_floats(&out);
        assert_eq!(floats.len(), 3 * 64 * 64);
        assert!(
            floats.iter().any(|&v| v > 1.0),
            "fallback produced an empty tensor"
        );
    }

    #[test]
    fn debug_impl_names_the_type_and_adapter() {
        let Some(gpu) = gpu_preprocessor() else {
            eprintln!("Skipping GPU preprocess test: no adapter");
            return;
        };
        let text = format!("{gpu:?}");
        assert!(
            text.contains("WgpuPreprocessor"),
            "Debug output should name the type, got {text}"
        );
        assert!(
            text.contains("adapter"),
            "Debug output should include the adapter field, got {text}"
        );
    }

    #[test]
    fn gpu_preprocess_rejects_zero_input_dimensions() {
        let Some(gpu) = gpu_preprocessor() else {
            eprintln!("Skipping GPU preprocess test: no adapter");
            return;
        };
        let image = gradient(16, 16);
        for (w, h) in [(0u32, 64u32), (64, 0)] {
            assert!(
                gpu.preprocess(&image, &config_for(w, h)).is_err(),
                "input {w}x{h} must be rejected"
            );
        }
    }

    /// Buffers are pooled across calls, so a run at a different size is where a
    /// stale capacity would surface: the third result must equal the first.
    #[test]
    fn gpu_preprocess_is_stable_when_pooled_buffers_are_reused() {
        let Some(gpu) = gpu_preprocessor() else {
            eprintln!("Skipping GPU preprocess test: no adapter");
            return;
        };
        let config = config_for(64, 64);
        let big = gradient(96, 96);
        let small = gradient(32, 32);

        let first = as_floats(&gpu.preprocess(&big, &config).expect("first"));
        let _ = gpu.preprocess(&small, &config).expect("second");
        let third = as_floats(&gpu.preprocess(&big, &config).expect("third"));

        assert_eq!(
            first, third,
            "reusing pooled buffers changed the result for identical input"
        );
    }
}
