use super::utils::{ComputeDispatch, UniformCache, buffer_entry, uniform_entry};
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
    uniforms: UniformCache<UpsampleUniforms>,
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
            uniforms: UniformCache::new("yunet_resize2x_uniforms"),
        })
    }

    /// Record the 2x upsample dispatch into `encoder` without submitting.
    pub(super) fn encode(
        &self,
        encoder: &mut impl ComputeDispatch,
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
        let uniform_buffer = self.uniforms.buffer(context.device(), uniforms)?;

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

        encoder.record_dispatch(
            context,
            "resize2x",
            &self.pipeline,
            &bind_group,
            [
                ((dims[3] * 2) as u32).div_ceil(UPSAMPLE_WORKGROUP_X),
                ((dims[2] * 2) as u32).div_ceil(UPSAMPLE_WORKGROUP_Y),
                dims[1] as u32,
            ],
        );

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

/// A nearest 2x upsample added to a skip connection, in one dispatch.
///
/// Both of the neck's upsamples are read by exactly one consumer, the add after them, so the
/// upsampled tensor never has to exist. The addition per element is the same one, in the same
/// order, so the result is bit-exact with the separate pair (experiment 36).
#[derive(Debug)]
pub(super) struct ResizeAddPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniforms: UniformCache<UpsampleUniforms>,
}

impl ResizeAddPipeline {
    pub(super) fn new(device: &wgpu::Device) -> Result<Self> {
        let (pipeline, bind_group_layout) = create_gpu_pipeline!(
            device,
            "resize2x_add",
            super::RESIZE2X_ADD_WGSL,
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
            uniforms: UniformCache::new("yunet_resize2x_add_uniforms"),
        })
    }

    /// Record `upsample2x(small) + skip` into `encoder` without submitting.
    pub(super) fn encode(
        &self,
        encoder: &mut impl ComputeDispatch,
        context: &Arc<GpuContext>,
        pool: &Arc<GpuBufferPool>,
        small: &GpuTensor,
        skip: &GpuTensor,
    ) -> Result<GpuTensor> {
        let dims = small.shape().dims();
        anyhow::ensure!(
            dims.len() == 4,
            "resize2x_add expects 4D tensors (got {:?})",
            dims
        );
        let out_dims = [dims[0], dims[1], dims[2] * 2, dims[3] * 2];
        anyhow::ensure!(
            skip.shape().dims() == out_dims,
            "resize2x_add skip must be the upsampled shape {:?} (got {:?})",
            out_dims,
            skip.shape().dims()
        );
        let output = GpuTensor::uninitialized_with_pool(
            context.clone(),
            Some(pool.clone()),
            out_dims,
            Some("resize2x_add_output"),
        )?;
        let uniforms = UpsampleUniforms {
            input_width: dims[3] as u32,
            input_height: dims[2] as u32,
            channels: dims[1] as u32,
            _padding: 0,
        };
        let uniform_buffer = self.uniforms.buffer(context.device(), uniforms)?;
        let bind_group = context
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("resize2x_add_bg"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: small.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: skip.buffer().as_entire_binding(),
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
            "resize2x_add",
            &self.pipeline,
            &bind_group,
            [
                ((dims[3] * 2) as u32).div_ceil(UPSAMPLE_WORKGROUP_X),
                ((dims[2] * 2) as u32).div_ceil(UPSAMPLE_WORKGROUP_Y),
                dims[1] as u32,
            ],
        );
        Ok(output)
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, PartialEq, Eq, Hash)]
struct UpsampleUniforms {
    input_width: u32,
    input_height: u32,
    channels: u32,
    _padding: u32,
}
