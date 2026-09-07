use super::utils::{ComputeDispatch, UniformCache, buffer_entry, uniform_entry};
use crate::gpu::GpuTensor;
use fcs_utils::create_gpu_pipeline;

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use fcs_utils::gpu::{GpuBufferPool, GpuContext};
use std::sync::Arc;

const ADD_WORKGROUP_SIZE: u32 = 256;

#[derive(Debug)]
pub(super) struct AddPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniforms: UniformCache<AddUniforms>,
}

impl AddPipeline {
    pub(super) fn new(device: &wgpu::Device) -> Result<Self> {
        let (pipeline, bind_group_layout) = create_gpu_pipeline!(
            device,
            "add",
            super::ADD_WGSL,
            [
                buffer_entry(0, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
                uniform_entry(3),
            ]
        );
        Ok(Self {
            pipeline,
            bind_group_layout,
            uniforms: UniformCache::new("yunet_add_uniforms"),
        })
    }

    /// Record the element-wise add dispatch into `encoder` without submitting.
    pub(super) fn encode(
        &self,
        encoder: &mut impl ComputeDispatch,
        context: &Arc<GpuContext>,
        pool: &Arc<GpuBufferPool>,
        lhs: &GpuTensor,
        rhs: &GpuTensor,
    ) -> Result<GpuTensor> {
        let output = GpuTensor::uninitialized_with_pool(
            context.clone(),
            Some(pool.clone()),
            lhs.shape().dims().to_vec(),
            Some("add_output"),
        )?;
        let uniforms = AddUniforms {
            len: lhs.shape().elements() as u32,
        };
        let uniform_buffer = self.uniforms.buffer(context.device(), uniforms)?;

        let bind_group = context
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("add_bg"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: lhs.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: rhs.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                ],
            });

        encoder.record_dispatch(
            context,
            "add",
            &self.pipeline,
            &bind_group,
            [uniforms.len.div_ceil(ADD_WORKGROUP_SIZE), 1, 1],
        );

        Ok(output)
    }

    pub(super) fn execute(
        &self,
        context: &Arc<GpuContext>,
        pool: &Arc<GpuBufferPool>,
        lhs: &GpuTensor,
        rhs: &GpuTensor,
    ) -> Result<GpuTensor> {
        let mut encoder =
            context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("add_encoder"),
                });
        let output = self.encode(&mut encoder, context, pool, lhs, rhs)?;
        context.queue().submit(Some(encoder.finish()));
        Ok(output)
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, PartialEq, Eq, Hash)]
struct AddUniforms {
    len: u32,
}
