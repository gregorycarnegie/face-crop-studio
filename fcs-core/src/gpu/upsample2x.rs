use super::utils::{buffer_entry, create_uniform_buffer, uniform_entry};
use fcs_utils::create_gpu_pipeline;

use crate::gpu::GpuTensor;
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use fcs_utils::gpu::{GpuBufferPool, GpuContext};
use std::sync::Arc;

const UPSAMPLE_WORKGROUP_X: u32 = 8;
const UPSAMPLE_WORKGROUP_Y: u32 = 8;

#[derive(Debug)]
pub(super) struct Upsample2xPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl Upsample2xPipeline {
    pub(super) fn new(device: &wgpu::Device) -> Result<Self> {
        let (pipeline, bind_group_layout) = create_gpu_pipeline!(
            device,
            "resize2x",
            super::UPSAMPLE2X_WGSL,
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

    /// Record the 2x upsample dispatch into `encoder` without submitting.
    pub(super) fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        context: &Arc<GpuContext>,
        pool: &Arc<GpuBufferPool>,
        tensor: &GpuTensor,
    ) -> Result<GpuTensor> {
        let dims = tensor.shape().dims();
        anyhow::ensure!(
            dims.len() == 4,
            "upsample expects 4D tensor (got {:?})",
            dims
        );
        let output = GpuTensor::uninitialized_with_pool(
            context.clone(),
            Some(pool.clone()),
            [dims[0], dims[1], dims[2] * 2, dims[3] * 2],
            Some("resize2x_output"),
        )?;
        let uniforms = UpsampleUniforms {
            input_width: dims[3] as u32,
            input_height: dims[2] as u32,
            channels: dims[1] as u32,
            _padding: 0,
        };
        let uniform_buffer =
            create_uniform_buffer(context.device(), "yunet_resize2x_uniforms", &uniforms);

        let bind_group = context
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("resize2x_bg"),
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
                label: Some("resize2x_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                ((dims[3] * 2) as u32).div_ceil(UPSAMPLE_WORKGROUP_X),
                ((dims[2] * 2) as u32).div_ceil(UPSAMPLE_WORKGROUP_Y),
                dims[1] as u32,
            );
        }

        Ok(output)
    }

    pub(super) fn execute(
        &self,
        context: &Arc<GpuContext>,
        pool: &Arc<GpuBufferPool>,
        tensor: &GpuTensor,
    ) -> Result<GpuTensor> {
        let mut encoder =
            context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("resize2x_encoder"),
                });
        let output = self.encode(&mut encoder, context, pool, tensor)?;
        context.queue().submit(Some(encoder.finish()));
        Ok(output)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct UpsampleUniforms {
    input_width: u32,
    input_height: u32,
    channels: u32,
    _padding: u32,
}
