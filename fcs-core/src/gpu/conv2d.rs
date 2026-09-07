use super::{
    activation::ActivationKind,
    utils::{ComputeDispatch, UniformCache, buffer_entry, compute_output_dim, uniform_entry},
};
use crate::gpu::GpuTensor;
use fcs_utils::create_gpu_pipeline;

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use fcs_utils::gpu::{GpuBufferPool, GpuContext};
use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

const CONV2D_WGSL: &str = include_str!("conv2d.wgsl");
const CONV_WORKGROUP_X: u32 = 8;
const CONV_WORKGROUP_Y: u32 = 8;
// Must match the pointwise channel tile in conv2d.wgsl.
const POINTWISE_CHANNEL_TILE: u32 = 4;

/// The tensor operands of one convolution.
///
/// Grouped because they are meaningless apart: every call site passes exactly these three, in
/// this order, and validation relates their shapes to each other.
#[derive(Debug, Clone, Copy)]
pub(super) struct Conv2dTensors<'a> {
    pub(super) input: &'a GpuTensor,
    pub(super) weights: &'a GpuTensor,
    pub(super) bias: &'a GpuTensor,
}

#[derive(Debug)]
pub(super) struct Conv2dPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    pixels_per_thread: u32,
    /// Creating one 56-byte buffer per dispatch measured at 0.445 ms per forward pass,
    /// substantial overhead beside the ~0.9 ms of GPU compute.
    uniforms: UniformCache<Conv2dUniforms>,
    /// Bind groups, reused when the same five buffers come back.
    ///
    /// Host encoding is 3.1 us per dispatch (`gpu_record` plus `gpu_finish` over 43
    /// dispatches) and `create_bind_group` is 0.8 us of it. The graph is static, so whether
    /// this pays at all depends on the buffer pool handing the same intermediates to the
    /// same layers on every inference -- which is a question about the pool's release order,
    /// not something to assume. Measured: the pool cycles through about three assignments
    /// before it settles, so the first few inferences miss and every one after hits.
    /// `bind_cache_stats` reports it, and a test asserts it stays that way -- a change to
    /// the pool's ordering would otherwise turn this cache into pure overhead in silence.
    ///
    /// Keyed on the buffers themselves, so a miss rebuilds a correct bind group rather than
    /// binding the wrong one. Cleared wholesale past `BIND_CACHE_LIMIT` so a caller sweeping
    /// input sizes cannot pin pooled buffers indefinitely.
    bind_groups: Mutex<HashMap<[wgpu::Buffer; 5], Arc<wgpu::BindGroup>>>,
    bind_hits: AtomicU64,
    bind_misses: AtomicU64,
}

/// Distinct buffer combinations kept before the bind-group cache is dropped and rebuilt.
/// The static 640x640 graph uses about 34; the headroom is for concurrent inferences.
const BIND_CACHE_LIMIT: usize = 512;

impl Conv2dPipeline {
    pub(super) fn new(device: &wgpu::Device, pixels_per_thread: u32) -> Result<Self> {
        let (pipeline, bind_group_layout) = create_gpu_pipeline!(
            device,
            "conv2d",
            CONV2D_WGSL,
            [
                buffer_entry(0, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
                uniform_entry(4),
            ]
        );
        Ok(Self {
            pipeline,
            bind_group_layout,
            pixels_per_thread,
            uniforms: UniformCache::new("yunet_conv2d_uniforms"),
            bind_groups: Mutex::new(HashMap::new()),
            bind_hits: AtomicU64::new(0),
            bind_misses: AtomicU64::new(0),
        })
    }

    /// Bind-group cache hits and misses since this pipeline was built.
    #[cfg(test)]
    pub(super) fn bind_cache_stats(&self) -> (u64, u64) {
        (
            self.bind_hits.load(Ordering::Relaxed),
            self.bind_misses.load(Ordering::Relaxed),
        )
    }

    /// Test hook for the uniform cache; the cache itself is an internal detail.
    #[cfg(test)]
    pub(super) fn uniform_buffer_for_test(
        &self,
        device: &wgpu::Device,
        config: &Conv2dConfig,
    ) -> Arc<wgpu::Buffer> {
        self.uniforms
            .buffer(device, Conv2dUniforms::from(config))
            .expect("uniform cache should not be poisoned in a test")
    }

    /// Record the Conv2D dispatch into `encoder` without submitting.
    pub(super) fn encode(
        &self,
        encoder: &mut impl ComputeDispatch,
        context: &Arc<GpuContext>,
        pool: &Arc<GpuBufferPool>,
        tensors: Conv2dTensors<'_>,
        config: &Conv2dConfig,
    ) -> Result<GpuTensor> {
        let Conv2dTensors {
            input,
            weights,
            bias,
        } = tensors;
        let device = context.device();
        let uniform_buffer = self.uniforms.buffer(device, Conv2dUniforms::from(config))?;
        let output = GpuTensor::uninitialized_with_pool(
            context.clone(),
            Some(pool.clone()),
            config.output_shape_dims(),
            Some("conv2d_output"),
        )?;

        let key = [
            input.buffer().clone(),
            weights.buffer().clone(),
            bias.buffer().clone(),
            output.buffer().clone(),
            wgpu::Buffer::clone(&uniform_buffer),
        ];
        let bind_group = {
            let mut cache = self
                .bind_groups
                .lock()
                .map_err(|_| anyhow::anyhow!("conv2d bind group cache poisoned"))?;
            if cache.len() >= BIND_CACHE_LIMIT {
                cache.clear();
            }
            match cache.get(&key) {
                Some(existing) => {
                    self.bind_hits.fetch_add(1, Ordering::Relaxed);
                    existing.clone()
                }
                None => {
                    self.bind_misses.fetch_add(1, Ordering::Relaxed);
                    let created = Arc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("conv2d_bg"),
                        layout: &self.bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: key[0].as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: key[1].as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: key[2].as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: key[3].as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: key[4].as_entire_binding(),
                            },
                        ],
                    }));
                    cache.insert(key, created.clone());
                    created
                }
            }
        };

        // These three must match `main` in conv2d.wgsl exactly: the shader picks its path
        // from the uniforms and the host has to size dispatch z for whichever it will pick.
        let pointwise = config.kernel_width == 1
            && config.kernel_height == 1
            && config.stride_x == 1
            && config.stride_y == 1
            && config.pad_x == 0
            && config.pad_y == 0
            && config.groups == 1;
        let depthwise = config.kernel_width == 3
            && config.kernel_height == 3
            && config.stride_x == 1
            && config.stride_y == 1
            && config.pad_x == 1
            && config.pad_y == 1
            && config.groups == config.input_channels
            && config.output_channels == config.input_channels;
        // An ungrouped general convolution gathers the same inputs for every output channel,
        // so it takes the same four-channel tile the pointwise path uses.
        let channel_tiled = pointwise || (!depthwise && config.groups == 1);
        encoder.record_dispatch(
            context,
            if config.kernel_width == 1 && config.kernel_height == 1 && config.groups == 1 {
                "conv2d/pointwise"
            } else if config.groups == config.input_channels
                && config.output_channels == config.input_channels
            {
                "conv2d/depthwise"
            } else {
                "conv2d/general"
            },
            &self.pipeline,
            &bind_group,
            [
                config
                    .output_width
                    .div_ceil(CONV_WORKGROUP_X * self.pixels_per_thread),
                config.output_height.div_ceil(CONV_WORKGROUP_Y),
                config.output_channels.div_ceil(if channel_tiled {
                    POINTWISE_CHANNEL_TILE
                } else {
                    1
                }),
            ],
        );

        Ok(output)
    }

    pub(super) fn execute(
        &self,
        context: &Arc<GpuContext>,
        pool: &Arc<GpuBufferPool>,
        tensors: Conv2dTensors<'_>,
        config: &Conv2dConfig,
    ) -> Result<GpuTensor> {
        let mut encoder =
            context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("conv2d_encoder"),
                });
        let output = self.encode(&mut encoder, context, pool, tensors, config)?;
        context.queue().submit(Some(encoder.finish()));
        Ok(output)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Conv2dChannels {
    pub input: u32,
    pub output: u32,
}

impl Conv2dChannels {
    pub const fn new(input: u32, output: u32) -> Self {
        Self { input, output }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SpatialDims {
    pub width: u32,
    pub height: u32,
}

impl SpatialDims {
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

impl From<(u32, u32)> for SpatialDims {
    fn from(value: (u32, u32)) -> Self {
        Self {
            width: value.0,
            height: value.1,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Conv2dOptions {
    pub groups: u32,
    pub activation: Option<ActivationKind>,
}

impl Conv2dOptions {
    pub const fn new(groups: u32, activation: Option<ActivationKind>) -> Self {
        Self { groups, activation }
    }
}

/// Geometry for a convolution layer.
#[derive(Debug, Clone)]
pub struct Conv2dConfig {
    pub batch: u32,
    pub input_channels: u32,
    pub output_channels: u32,
    pub input_width: u32,
    pub input_height: u32,
    pub kernel_width: u32,
    pub kernel_height: u32,
    pub stride_x: u32,
    pub stride_y: u32,
    pub pad_x: u32,
    pub pad_y: u32,
    pub groups: u32,
    pub output_width: u32,
    pub output_height: u32,
    pub activation: Option<ActivationKind>,
}

impl Conv2dConfig {
    /// Create a validated convolution configuration.
    pub fn new(
        batch: u32,
        channels: Conv2dChannels,
        input: SpatialDims,
        kernel: SpatialDims,
        stride: SpatialDims,
        pad: SpatialDims,
        options: Conv2dOptions,
    ) -> Result<Self> {
        let Conv2dOptions { groups, activation } = options;
        let Conv2dChannels {
            input: input_channels,
            output: output_channels,
        } = channels;
        let SpatialDims {
            width: input_width,
            height: input_height,
        } = input;
        let SpatialDims {
            width: kernel_width,
            height: kernel_height,
        } = kernel;
        let SpatialDims {
            width: stride_x,
            height: stride_y,
        } = stride;
        let SpatialDims {
            width: pad_x,
            height: pad_y,
        } = pad;
        anyhow::ensure!(batch == 1, "only batch size 1 is supported (got {batch})");
        anyhow::ensure!(input_channels > 0, "input channels must be > 0");
        anyhow::ensure!(output_channels > 0, "output channels must be > 0");
        anyhow::ensure!(
            kernel_width > 0 && kernel_height > 0,
            "kernel must be non-zero"
        );
        anyhow::ensure!(stride_x > 0 && stride_y > 0, "stride must be non-zero");
        anyhow::ensure!(groups > 0, "groups must be > 0");
        anyhow::ensure!(
            input_channels.is_multiple_of(groups),
            "input channels ({input_channels}) must be divisible by groups ({groups})"
        );
        anyhow::ensure!(
            output_channels.is_multiple_of(groups),
            "output channels ({output_channels}) must be divisible by groups ({groups})"
        );

        let output_width = compute_output_dim(input_width, pad_x, kernel_width, stride_x)
            .context("invalid convolution width configuration")?;
        let output_height = compute_output_dim(input_height, pad_y, kernel_height, stride_y)
            .context("invalid convolution height configuration")?;

        Ok(Self {
            batch,
            input_channels,
            output_channels,
            input_width,
            input_height,
            kernel_width,
            kernel_height,
            stride_x,
            stride_y,
            pad_x,
            pad_y,
            groups,
            output_width,
            output_height,
            activation,
        })
    }

    pub fn input_shape_dims(&self) -> [usize; 4] {
        [
            self.batch as usize,
            self.input_channels as usize,
            self.input_height as usize,
            self.input_width as usize,
        ]
    }

    pub fn output_shape_dims(&self) -> [usize; 4] {
        [
            self.batch as usize,
            self.output_channels as usize,
            self.output_height as usize,
            self.output_width as usize,
        ]
    }

    pub fn weight_shape_dims(&self) -> [usize; 4] {
        [
            self.output_channels as usize,
            (self.input_channels / self.groups) as usize,
            self.kernel_height as usize,
            self.kernel_width as usize,
        ]
    }

    pub fn bias_shape_dims(&self) -> [usize; 1] {
        [self.output_channels as usize]
    }

    pub fn validate(&self, input_len: usize, weight_len: usize, bias_len: usize) -> Result<()> {
        let expected_input = self.batch as usize
            * self.input_channels as usize
            * self.input_height as usize
            * self.input_width as usize;
        anyhow::ensure!(
            input_len == expected_input,
            "conv input tensor expected {expected_input} elements, got {input_len}"
        );

        let weights_per_out = (self.input_channels / self.groups) as usize
            * self.kernel_width as usize
            * self.kernel_height as usize;
        let expected_weights = self.output_channels as usize * weights_per_out;
        anyhow::ensure!(
            weight_len == expected_weights,
            "conv weights expected {expected_weights} elements, got {weight_len}"
        );
        anyhow::ensure!(
            bias_len == self.output_channels as usize,
            "conv bias expected {} elements, got {bias_len}",
            self.output_channels
        );
        Ok(())
    }

    #[cfg(test)]
    pub fn output_element_count(&self) -> usize {
        self.batch as usize
            * self.output_channels as usize
            * self.output_height as usize
            * self.output_width as usize
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, PartialEq, Eq, Hash)]
struct Conv2dUniforms {
    input_width: u32,
    input_height: u32,
    input_channels: u32,
    output_width: u32,
    output_height: u32,
    output_channels: u32,
    kernel_width: u32,
    kernel_height: u32,
    stride_x: u32,
    stride_y: u32,
    pad_x: u32,
    pad_y: u32,
    groups: u32,
    activation_mode: u32,
}

impl From<&Conv2dConfig> for Conv2dUniforms {
    fn from(value: &Conv2dConfig) -> Self {
        Self {
            input_width: value.input_width,
            input_height: value.input_height,
            input_channels: value.input_channels,
            output_width: value.output_width,
            output_height: value.output_height,
            output_channels: value.output_channels,
            kernel_width: value.kernel_width,
            kernel_height: value.kernel_height,
            stride_x: value.stride_x,
            stride_y: value.stride_y,
            pad_x: value.pad_x,
            pad_y: value.pad_y,
            groups: value.groups,
            activation_mode: value
                .activation
                .map(|k| match k {
                    ActivationKind::Relu => 1,
                    ActivationKind::Sigmoid => 2,
                })
                .unwrap_or(0),
        }
    }
}
