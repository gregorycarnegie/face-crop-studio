use super::{
    activation::ActivationKind,
    utils::{ComputeDispatch, UniformCache, buffer_entry, compute_output_dim, uniform_entry},
};
use crate::gpu::GpuTensor;

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use fcs_utils::gpu::{GpuBufferPool, GpuContext};
use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex, OnceLock,
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

/// Which specialized kernel a configuration dispatches.
///
/// The host has to know this anyway to size dispatch z, so selecting a pipeline by it costs
/// nothing extra. Mirrors the conditions in `conv2d.wgsl`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kernel {
    Pointwise,
    Depthwise,
    /// Ungrouped, anything else: the 640x640 3->16 stride-2 stem.
    General,
    /// Grouped and not depthwise. Nothing in the detector reaches it -- all 60 of its
    /// convolutions are `groups == 1` or `in_per_group == 1` -- so it is only the public
    /// `conv2d` API's fallback, which is why its pipeline is built on first use.
    Grouped,
}

#[derive(Debug)]
pub(super) struct Conv2dPipeline {
    /// One pipeline per kernel, from one shader module.
    ///
    /// `main` in the shader branches on the uniforms, so compiling it drags all four kernels
    /// through FXC: about 185 ms, which experiment 80 found to be 90% of all shader
    /// compilation and 22% of a cold start. Compiled per entry point instead, the three
    /// kernels the detector dispatches cost about 25 + 21 + 53 ms, because compilation is
    /// superlinear in what one entry point can reach
    /// (`examples/shader_compile_cost.rs`).
    pointwise_pipeline: wgpu::ComputePipeline,
    depthwise_pipeline: wgpu::ComputePipeline,
    general_pipeline: wgpu::ComputePipeline,
    /// Built on first use, because nothing in the detector is a grouped-not-depthwise
    /// convolution and compiling the branching entry point is the expensive half of the whole
    /// thing.
    grouped_pipeline: OnceLock<wgpu::ComputePipeline>,
    module: wgpu::ShaderModule,
    pipeline_layout: wgpu::PipelineLayout,
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
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("conv2d.wgsl shader"),
            source: wgpu::ShaderSource::Wgsl(CONV2D_WGSL.into()),
        });
        // All four entry points bind the same five resources, so one layout serves them and
        // the bind-group cache stays shared rather than one per pipeline.
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fcs_conv2d_bgl"),
            entries: &[
                buffer_entry(0, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
                uniform_entry(4),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fcs_conv2d_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let build = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        Ok(Self {
            pointwise_pipeline: build("main_pointwise"),
            depthwise_pipeline: build("main_depthwise"),
            general_pipeline: build("main_general"),
            grouped_pipeline: OnceLock::new(),
            module,
            pipeline_layout,
            bind_group_layout,
            pixels_per_thread,
            uniforms: UniformCache::new("fcs_conv2d_uniforms"),
            bind_groups: Mutex::new(HashMap::new()),
            bind_hits: AtomicU64::new(0),
            bind_misses: AtomicU64::new(0),
        })
    }

    /// The pipeline for one kernel, compiling the grouped fallback on first use.
    fn pipeline_for(&self, device: &wgpu::Device, kernel: Kernel) -> &wgpu::ComputePipeline {
        match kernel {
            Kernel::Pointwise => &self.pointwise_pipeline,
            Kernel::Depthwise => &self.depthwise_pipeline,
            Kernel::General => &self.general_pipeline,
            Kernel::Grouped => self.grouped_pipeline.get_or_init(|| {
                // `main` is the branching entry point, and the only one that reaches the
                // grouped path. Nothing in the detector gets here.
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("main"),
                    layout: Some(&self.pipeline_layout),
                    module: &self.module,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                })
            }),
        }
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

        let kernel = kernel_for(config);
        // Pointwise and ungrouped-general gather the same inputs for every output channel,
        // so both take the four-channel tile; the other two take one channel per thread.
        let channel_tiled = matches!(kernel, Kernel::Pointwise | Kernel::General);
        encoder.record_dispatch(
            context,
            match kernel {
                Kernel::Pointwise => "conv2d/pointwise",
                Kernel::Depthwise => "conv2d/depthwise",
                Kernel::General | Kernel::Grouped => "conv2d/general",
            },
            self.pipeline_for(device, kernel),
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

/// Input and output channel counts for a convolution.
#[derive(Debug, Clone, Copy)]
pub struct Conv2dChannels {
    /// Number of input channels.
    pub input: u32,
    /// Number of output channels.
    pub output: u32,
}

impl Conv2dChannels {
    /// Pair input/output channel counts; validation occurs in [`Conv2dConfig::new`].
    pub const fn new(input: u32, output: u32) -> Self {
        Self { input, output }
    }
}

/// Width and height used for convolution sizes, strides, or padding.
#[derive(Debug, Clone, Copy)]
pub struct SpatialDims {
    /// Extent along the horizontal axis.
    pub width: u32,
    /// Extent along the vertical axis.
    pub height: u32,
}

impl SpatialDims {
    /// Pair horizontal and vertical extents without validation.
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

/// Channel grouping and optional activation fused into a convolution.
#[derive(Debug, Clone, Copy)]
pub struct Conv2dOptions {
    /// Number of channel groups; 1 is dense, one per input channel is depthwise.
    pub groups: u32,
    /// Activation applied after bias, or `None` for linear output.
    pub activation: Option<ActivationKind>,
}

impl Conv2dOptions {
    /// Pair grouping and activation; validation occurs in [`Conv2dConfig::new`].
    pub const fn new(groups: u32, activation: Option<ActivationKind>) -> Self {
        Self { groups, activation }
    }
}

/// Geometry for a convolution layer.
#[derive(Debug, Clone)]
pub struct Conv2dConfig {
    /// Number of batch items; the GPU convolution supports only 1.
    pub batch: u32,
    /// Input channels, divisible by `groups`.
    pub input_channels: u32,
    /// Output channels, divisible by `groups`.
    pub output_channels: u32,
    /// Input width in pixels.
    pub input_width: u32,
    /// Input height in pixels.
    pub input_height: u32,
    /// Kernel width in pixels.
    pub kernel_width: u32,
    /// Kernel height in pixels.
    pub kernel_height: u32,
    /// Horizontal kernel step in input pixels.
    pub stride_x: u32,
    /// Vertical kernel step in input pixels.
    pub stride_y: u32,
    /// Zero-padding pixels on each horizontal side.
    pub pad_x: u32,
    /// Zero-padding pixels on each vertical side.
    pub pad_y: u32,
    /// Number of independent channel groups; must be nonzero.
    pub groups: u32,
    /// Output width computed from input width, kernel, stride, and padding.
    pub output_width: u32,
    /// Output height computed from input height, kernel, stride, and padding.
    pub output_height: u32,
    /// Activation applied after bias, or `None` for linear output.
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

    /// Return the expected input shape in NCHW order.
    pub fn input_shape_dims(&self) -> [usize; 4] {
        [
            self.batch as usize,
            self.input_channels as usize,
            self.input_height as usize,
            self.input_width as usize,
        ]
    }

    /// Return the computed output shape in NCHW order.
    pub fn output_shape_dims(&self) -> [usize; 4] {
        [
            self.batch as usize,
            self.output_channels as usize,
            self.output_height as usize,
            self.output_width as usize,
        ]
    }

    /// Return `[output_channels, input_channels / groups, kernel_height, kernel_width]`.
    pub fn weight_shape_dims(&self) -> [usize; 4] {
        [
            self.output_channels as usize,
            (self.input_channels / self.groups) as usize,
            self.kernel_height as usize,
            self.kernel_width as usize,
        ]
    }

    /// Return the bias shape: one value per output channel.
    pub fn bias_shape_dims(&self) -> [usize; 1] {
        [self.output_channels as usize]
    }

    /// Check input, weight, and bias element counts against this geometry.
    ///
    /// Returns an error on a length mismatch. Construct the geometry with
    /// [`Self::new`] first; this method does not revalidate its spatial parameters.
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

    /// Return the total number of output elements.
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

/// Which specialised kernel a config takes.
///
/// Pulled out of `encode` so it can be asserted without a GPU. Every one of these comparisons
/// survived mutation while it was inline: a mis-route sends the convolution to the *general*
/// kernel, which computes the same answer more slowly, so comparing output against a CPU
/// reference cannot see it. What is wrong in that case is the dispatch, and the only way to
/// check a dispatch is to look at it.
///
/// The selection order must match `main` in conv2d.wgsl, which is still the grouped fallback's
/// entry point: pointwise, then depthwise, then ungrouped-general.
fn kernel_for(config: &Conv2dConfig) -> Kernel {
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
    if pointwise {
        Kernel::Pointwise
    } else if depthwise {
        Kernel::Depthwise
    } else if config.groups == 1 {
        Kernel::General
    } else {
        Kernel::Grouped
    }
}

#[cfg(test)]
mod kernel_selection_tests {
    use super::*;
    use crate::gpu::conv2d::{Conv2dChannels, Conv2dOptions, SpatialDims};

    /// A config with every field distinct from every other, so one field standing in for
    /// another cannot go unnoticed.
    fn config(
        kernel: (u32, u32),
        stride: (u32, u32),
        pad: (u32, u32),
        groups: u32,
        channels: (u32, u32),
    ) -> Conv2dConfig {
        Conv2dConfig::new(
            1,
            Conv2dChannels::new(channels.0, channels.1),
            SpatialDims::new(12, 9),
            SpatialDims::new(kernel.0, kernel.1),
            SpatialDims::new(stride.0, stride.1),
            SpatialDims::new(pad.0, pad.1),
            Conv2dOptions::new(groups, None),
        )
        .expect("valid config")
    }

    #[test]
    fn a_1x1_unpadded_ungrouped_convolution_is_pointwise() {
        let c = config((1, 1), (1, 1), (0, 0), 1, (8, 16));
        assert_eq!(kernel_for(&c), Kernel::Pointwise);
    }

    #[test]
    fn a_3x3_same_padded_channelwise_convolution_is_depthwise() {
        let c = config((3, 3), (1, 1), (1, 1), 8, (8, 8));
        assert_eq!(kernel_for(&c), Kernel::Depthwise);
    }

    /// Each condition on its own: change one field and the kernel must change.
    ///
    /// This is what kills the `==` mutants. With them inline, flipping any one comparison sent
    /// the convolution to `General`, which is still correct, so nothing failed.
    #[test]
    fn every_pointwise_condition_is_load_bearing() {
        // Start from the pointwise config and break one thing at a time.
        for (what, c) in [
            ("kernel width", config((2, 1), (1, 1), (0, 0), 1, (8, 16))),
            ("kernel height", config((1, 2), (1, 1), (0, 0), 1, (8, 16))),
            ("stride x", config((1, 1), (2, 1), (0, 0), 1, (8, 16))),
            ("stride y", config((1, 1), (1, 2), (0, 0), 1, (8, 16))),
            ("pad x", config((1, 1), (1, 1), (1, 0), 1, (8, 16))),
            ("pad y", config((1, 1), (1, 1), (0, 1), 1, (8, 16))),
        ] {
            assert_ne!(
                kernel_for(&c),
                Kernel::Pointwise,
                "{what} should have disqualified the pointwise kernel"
            );
        }
        // Grouped disqualifies it too, and lands on the grouped kernel rather than general.
        let grouped = config((1, 1), (1, 1), (0, 0), 4, (8, 16));
        assert_eq!(kernel_for(&grouped), Kernel::Grouped);
    }

    #[test]
    fn every_depthwise_condition_is_load_bearing() {
        for (what, c) in [
            ("kernel width", config((2, 3), (1, 1), (1, 1), 8, (8, 8))),
            ("kernel height", config((3, 2), (1, 1), (1, 1), 8, (8, 8))),
            ("stride x", config((3, 3), (2, 1), (1, 1), 8, (8, 8))),
            ("stride y", config((3, 3), (1, 2), (1, 1), 8, (8, 8))),
            ("pad x", config((3, 3), (1, 1), (0, 1), 8, (8, 8))),
            ("pad y", config((3, 3), (1, 1), (1, 0), 8, (8, 8))),
            // groups != input channels, and output != input channels.
            ("groups", config((3, 3), (1, 1), (1, 1), 4, (8, 8))),
            (
                "output channels",
                config((3, 3), (1, 1), (1, 1), 8, (8, 16)),
            ),
        ] {
            assert_ne!(
                kernel_for(&c),
                Kernel::Depthwise,
                "{what} should have disqualified the depthwise kernel"
            );
        }
    }

    /// The stem: 3x3 stride 2, ungrouped. Neither fast path, and not the grouped fallback.
    #[test]
    fn the_strided_stem_takes_the_general_kernel() {
        let c = config((3, 3), (2, 2), (1, 1), 1, (3, 16));
        assert_eq!(kernel_for(&c), Kernel::General);
    }

    /// Grouped but not depthwise is the one the detector never reaches.
    #[test]
    fn grouped_but_not_channelwise_takes_the_grouped_kernel() {
        let c = config((3, 3), (1, 1), (1, 1), 4, (8, 16));
        assert_eq!(kernel_for(&c), Kernel::Grouped);
    }
}
