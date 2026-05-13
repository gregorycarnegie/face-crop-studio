use super::utils::{buffer_entry, create_uniform_buffer, uniform_entry};
use crate::gpu::GpuTensor;
use fcs_utils::create_gpu_pipeline;

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use fcs_utils::gpu::GpuContext;
use std::sync::Arc;

const BATCH_NORM_WGSL: &str = include_str!("batch_norm.wgsl");
const BN_WORKGROUP_X: u32 = 8;
const BN_WORKGROUP_Y: u32 = 8;

#[derive(Debug)]
pub(super) struct BatchNormPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

#[derive(Clone, Copy)]
pub(super) struct BatchNormBindings<'a> {
    pub tensor: &'a GpuTensor,
    pub gamma: &'a GpuTensor,
    pub beta: &'a GpuTensor,
    pub mean: &'a GpuTensor,
    pub variance: &'a GpuTensor,
}

impl BatchNormPipeline {
    pub(super) fn new(device: &wgpu::Device) -> Result<Self> {
        let (pipeline, bind_group_layout) = create_gpu_pipeline!(
            device,
            "batch_norm",
            BATCH_NORM_WGSL,
            [
                buffer_entry(0, wgpu::BufferBindingType::Storage { read_only: false }),
                buffer_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, wgpu::BufferBindingType::Storage { read_only: true }),
                uniform_entry(5),
            ]
        );
        Ok(Self {
            pipeline,
            bind_group_layout,
        })
    }

    /// Record the batch-norm dispatch into `encoder` without submitting.
    pub(super) fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        context: &Arc<GpuContext>,
        tensors: BatchNormBindings,
        config: &BatchNormConfig,
    ) -> Result<GpuTensor> {
        let device = context.device();
        let output = tensors
            .tensor
            .uninitialized_like(Some("batch_norm_output"))?;
        tensors.tensor.encode_copy_to(encoder, &output);
        let uniforms = BatchNormUniforms::from(config);
        let uniform_buffer = create_uniform_buffer(device, "yunet_batch_norm_uniforms", &uniforms);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("batch_norm_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: output.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: tensors.gamma.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tensors.beta.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: tensors.mean.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: tensors.variance.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("batch_norm_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                config.width.div_ceil(BN_WORKGROUP_X),
                config.height.div_ceil(BN_WORKGROUP_Y),
                config.channels,
            );
        }

        Ok(output)
    }

    pub(super) fn execute(
        &self,
        context: &Arc<GpuContext>,
        tensors: BatchNormBindings,
        config: &BatchNormConfig,
    ) -> Result<GpuTensor> {
        let mut encoder =
            context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("batch_norm_encoder"),
                });
        let output = self.encode(&mut encoder, context, tensors, config)?;
        context.queue().submit(Some(encoder.finish()));
        Ok(output)
    }
}

/// Batch-norm tensor dimensions.
#[derive(Debug, Clone)]
pub struct BatchNormConfig {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub epsilon: f32,
}

impl BatchNormConfig {
    /// Create a validated batch-norm configuration.
    pub fn new(width: u32, height: u32, channels: u32, epsilon: f32) -> Result<Self> {
        anyhow::ensure!(width > 0 && height > 0, "spatial dimensions must be > 0");
        anyhow::ensure!(channels > 0, "channels must be > 0");
        anyhow::ensure!(epsilon >= 0.0, "epsilon must be >= 0");
        Ok(Self {
            width,
            height,
            channels,
            epsilon,
        })
    }

    pub fn tensor_shape_dims(&self) -> [usize; 3] {
        [
            self.channels as usize,
            self.height as usize,
            self.width as usize,
        ]
    }

    pub fn validate(
        &self,
        tensor_len: usize,
        gamma_len: usize,
        beta_len: usize,
        mean_len: usize,
        variance_len: usize,
    ) -> Result<()> {
        let expected_spatial = self.element_count();
        anyhow::ensure!(
            tensor_len == expected_spatial,
            "batch-norm tensor expected {expected_spatial} values, got {tensor_len}"
        );
        let channels = self.channels as usize;
        for (name, len) in [
            ("gamma", gamma_len),
            ("beta", beta_len),
            ("mean", mean_len),
            ("variance", variance_len),
        ] {
            anyhow::ensure!(
                len == channels,
                "batch-norm {name} expected {channels} values, got {len}"
            );
        }
        Ok(())
    }

    fn element_count(&self) -> usize {
        self.width as usize * self.height as usize * self.channels as usize
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BatchNormUniforms {
    width: u32,
    height: u32,
    channels: u32,
    epsilon: f32,
}

impl From<&BatchNormConfig> for BatchNormUniforms {
    fn from(value: &BatchNormConfig) -> Self {
        Self {
            width: value.width,
            height: value.height,
            channels: value.channels,
            epsilon: value.epsilon,
        }
    }
}
