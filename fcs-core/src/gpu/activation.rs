use super::utils::{buffer_entry, create_uniform_buffer, div_ceil_uniform, uniform_entry};
use crate::gpu::GpuTensor;
use fcs_utils::create_gpu_pipeline;

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use fcs_utils::gpu::GpuContext;
use std::sync::Arc;

const ACTIVATION_WGSL: &str = include_str!("activation.wgsl");
const ACTIVATION_WORKGROUP_SIZE: u32 = 256;

/// Supported activation operations.
#[derive(Debug, Clone, Copy)]
#[repr(u32)]
pub enum ActivationKind {
    /// Rectified Linear Unit.
    Relu = 0,
    /// Sigmoid.
    Sigmoid = 1,
}

#[derive(Debug)]
pub(super) struct ActivationPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl ActivationPipeline {
    pub(super) fn new(device: &wgpu::Device) -> Result<Self> {
        let (pipeline, bind_group_layout) = create_gpu_pipeline!(
            device,
            "activation",
            ACTIVATION_WGSL,
            [
                buffer_entry(0, wgpu::BufferBindingType::Storage { read_only: false }),
                uniform_entry(1),
            ]
        );
        Ok(Self {
            pipeline,
            bind_group_layout,
        })
    }

    /// Record the activation dispatch into `encoder` without submitting.
    pub(super) fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        context: &Arc<GpuContext>,
        tensor: &GpuTensor,
        kind: ActivationKind,
    ) -> Result<GpuTensor> {
        let device = context.device();
        let len = tensor.shape().elements();
        let output = tensor.uninitialized_like(Some("activation_output"))?;
        tensor.encode_copy_to(encoder, &output);
        let uniforms = ActivationUniforms {
            len: len as u32,
            mode: kind as u32,
        };
        let uniform_buffer = create_uniform_buffer(device, "yunet_activation_uniforms", &uniforms);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("activation_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: output.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("activation_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                div_ceil_uniform(len as u32, ACTIVATION_WORKGROUP_SIZE),
                1,
                1,
            );
        }

        Ok(output)
    }

    pub(super) fn execute(
        &self,
        context: &Arc<GpuContext>,
        tensor: &GpuTensor,
        kind: ActivationKind,
    ) -> Result<GpuTensor> {
        let mut encoder =
            context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("activation_encoder"),
                });
        let output = self.encode(&mut encoder, context, tensor, kind)?;
        context.queue().submit(Some(encoder.finish()));
        Ok(output)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ActivationUniforms {
    len: u32,
    mode: u32,
}
