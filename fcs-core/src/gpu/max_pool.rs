use super::utils::{buffer_entry, compute_output_dim, create_uniform_buffer, uniform_entry};
use fcs_utils::create_gpu_pipeline;

use crate::gpu::GpuTensor;
use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use fcs_utils::gpu::{GpuBufferPool, GpuContext};
use std::sync::Arc;

const POOL_WORKGROUP_X: u32 = 8;
const POOL_WORKGROUP_Y: u32 = 8;

#[derive(Debug)]
pub struct MaxPoolConfig {
    pub batch: u32,
    pub channels: u32,
    pub input_width: u32,
    pub input_height: u32,
    pub kernel: u32,
    pub stride: u32,
    pub pad: u32,
    pub output_width: u32,
    pub output_height: u32,
}

impl MaxPoolConfig {
    pub fn new(
        batch: u32,
        channels: u32,
        input_width: u32,
        input_height: u32,
        kernel: u32,
        stride: u32,
        pad: u32,
    ) -> Result<Self> {
        anyhow::ensure!(batch == 1, "only batch size 1 is supported (got {batch})");
        anyhow::ensure!(channels > 0, "channels must be > 0");
        anyhow::ensure!(
            input_width > 0 && input_height > 0,
            "spatial dims must be > 0"
        );
        anyhow::ensure!(kernel > 0, "kernel must be > 0");
        anyhow::ensure!(stride > 0, "stride must be > 0");
        let output_width = compute_output_dim(input_width, pad, kernel, stride)
            .context("pool width config invalid")?;
        let output_height = compute_output_dim(input_height, pad, kernel, stride)
            .context("pool height config invalid")?;
        Ok(Self {
            batch,
            channels,
            input_width,
            input_height,
            kernel,
            stride,
            pad,
            output_width,
            output_height,
        })
    }

    pub fn from_tensor(tensor: &GpuTensor, kernel: u32, stride: u32, pad: u32) -> Result<Self> {
        let dims = tensor.shape().dims();
        anyhow::ensure!(
            dims.len() == 4,
            "max pool only supports 4D tensors (got dims {:?})",
            dims
        );
        Self::new(
            dims[0] as u32,
            dims[1] as u32,
            dims[3] as u32,
            dims[2] as u32,
            kernel,
            stride,
            pad,
        )
    }

    pub fn output_dims(&self) -> [usize; 4] {
        [
            self.batch as usize,
            self.channels as usize,
            self.output_height as usize,
            self.output_width as usize,
        ]
    }
}

#[derive(Debug)]
pub(super) struct MaxPoolPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl MaxPoolPipeline {
    pub(super) fn new(device: &wgpu::Device) -> Result<Self> {
        let (pipeline, bind_group_layout) = create_gpu_pipeline!(
            device,
            "max_pool",
            super::MAX_POOL_WGSL,
            [
                buffer_entry(0, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_entry(1, wgpu::BufferBindingType::Storage { read_only: false }),
                uniform_entry(2),
            ]
        );
        Ok(Self {
            pipeline,
            bind_group_layout,
        })
    }

    /// Record the max-pool dispatch into `encoder` without submitting.
    pub(super) fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        context: &Arc<GpuContext>,
        pool: &Arc<GpuBufferPool>,
        tensor: &GpuTensor,
        config: &MaxPoolConfig,
    ) -> Result<GpuTensor> {
        let device = context.device();
        let output = GpuTensor::uninitialized_with_pool(
            context.clone(),
            Some(pool.clone()),
            config.output_dims().to_vec(),
            Some("max_pool_output"),
        )?;
        let uniforms = MaxPoolUniforms {
            input_width: config.input_width,
            input_height: config.input_height,
            channels: config.channels,
            output_width: config.output_width,
            output_height: config.output_height,
            kernel: config.kernel,
            stride: config.stride,
            pad: config.pad,
        };
        let uniform_buffer = create_uniform_buffer(device, "yunet_max_pool_uniforms", &uniforms);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("max_pool_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tensor.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("max_pool_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                config.output_width.div_ceil(POOL_WORKGROUP_X),
                config.output_height.div_ceil(POOL_WORKGROUP_Y),
                config.channels,
            );
        }

        Ok(output)
    }

    pub(super) fn execute(
        &self,
        context: &Arc<GpuContext>,
        pool: &Arc<GpuBufferPool>,
        tensor: &GpuTensor,
        config: &MaxPoolConfig,
    ) -> Result<GpuTensor> {
        let mut encoder =
            context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("max_pool_encoder"),
                });
        let output = self.encode(&mut encoder, context, pool, tensor, config)?;
        context.queue().submit(Some(encoder.finish()));
        Ok(output)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MaxPoolUniforms {
    input_width: u32,
    input_height: u32,
    channels: u32,
    output_width: u32,
    output_height: u32,
    kernel: u32,
    stride: u32,
    pad: u32,
}
