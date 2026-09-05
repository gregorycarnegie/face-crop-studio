use anyhow::{Context, Result};
use bytemuck::{Pod, bytes_of};

/// Records a dispatch in a new profiled pass or an already-open compute pass.
/// Both paths use the same graph and resource preparation code.
pub trait ComputeDispatch {
    fn record_dispatch(
        &mut self,
        context: &fcs_utils::gpu::GpuContext,
        label: &str,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        groups: [u32; 3],
    );
}

impl ComputeDispatch for wgpu::CommandEncoder {
    fn record_dispatch(
        &mut self,
        context: &fcs_utils::gpu::GpuContext,
        label: &str,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        groups: [u32; 3],
    ) {
        let mut pass = self.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: context.timestamp_writes(label),
        });
        pass.record_dispatch(context, label, pipeline, bind_group, groups);
    }
}

impl ComputeDispatch for wgpu::ComputePass<'_> {
    fn record_dispatch(
        &mut self,
        _context: &fcs_utils::gpu::GpuContext,
        _label: &str,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        [x, y, z]: [u32; 3],
    ) {
        self.set_pipeline(pipeline);
        self.set_bind_group(0, bind_group, &[]);
        self.dispatch_workgroups(x, y, z);
    }
}

pub(super) fn create_uniform_buffer(
    device: &wgpu::Device,
    label: &str,
    data: &impl Pod,
) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;

    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytes_of(data),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

pub(super) fn buffer_entry(
    binding: u32,
    ty: wgpu::BufferBindingType,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub(super) fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub(super) fn div_ceil_uniform(value: u32, divisor: u32) -> u32 {
    if value == 0 {
        0
    } else {
        value.div_ceil(divisor)
    }
}

pub(super) fn compute_output_dim(size: u32, pad: u32, kernel: u32, stride: u32) -> Result<u32> {
    anyhow::ensure!(stride > 0, "stride must be > 0");
    anyhow::ensure!(kernel > 0, "kernel must be > 0");
    let numerator = size
        .checked_add(pad * 2)
        .context("padding overflowed u32")?
        .checked_sub(kernel)
        .context("kernel larger than padded input")?;
    Ok(numerator / stride + 1)
}
