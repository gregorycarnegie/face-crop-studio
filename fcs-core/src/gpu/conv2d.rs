use super::activation::ActivationKind;
use super::utils::{buffer_entry, compute_output_dim, create_uniform_buffer, uniform_entry};
use crate::gpu::GpuTensor;
use fcs_utils::create_gpu_pipeline;

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use fcs_utils::gpu::{GpuBufferPool, GpuContext};
use std::sync::Arc;

const CONV2D_WGSL: &str = include_str!("conv2d.wgsl");
const CONV_WORKGROUP_X: u32 = 8;
const CONV_WORKGROUP_Y: u32 = 8;

#[derive(Debug)]
pub(super) struct Conv2dPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    pixels_per_thread: u32,
}

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
        })
    }

    /// Record the Conv2D dispatch into `encoder` without submitting.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        context: &Arc<GpuContext>,
        pool: &Arc<GpuBufferPool>,
        input: &GpuTensor,
        weights: &GpuTensor,
        bias: &GpuTensor,
        config: &Conv2dConfig,
    ) -> Result<GpuTensor> {
        let device = context.device();
        let uniforms = Conv2dUniforms::from(config);
        let uniform_buffer = create_uniform_buffer(device, "yunet_conv2d_uniforms", &uniforms);
        let output = GpuTensor::uninitialized_with_pool(
            context.clone(),
            Some(pool.clone()),
            config.output_shape_dims(),
            Some("conv2d_output"),
        )?;

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("conv2d_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weights.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bias.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("conv2d_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                config
                    .output_width
                    .div_ceil(CONV_WORKGROUP_X * self.pixels_per_thread),
                config.output_height.div_ceil(CONV_WORKGROUP_Y),
                config.output_channels,
            );
        }

        Ok(output)
    }

    pub(super) fn execute(
        &self,
        context: &Arc<GpuContext>,
        pool: &Arc<GpuBufferPool>,
        input: &GpuTensor,
        weights: &GpuTensor,
        bias: &GpuTensor,
        config: &Conv2dConfig,
    ) -> Result<GpuTensor> {
        let mut encoder =
            context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("conv2d_encoder"),
                });
        let output = self.encode(&mut encoder, context, pool, input, weights, bias, config)?;
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
#[derive(Clone, Copy, Pod, Zeroable)]
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
