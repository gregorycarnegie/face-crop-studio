use anyhow::{Context, Result};
use bytemuck::{Pod, bytes_of};
use std::{
    collections::HashMap,
    hash::Hash,
    sync::{Arc, Mutex},
};

/// Uniform buffers for one pipeline, reused across dispatches with identical contents.
///
/// `wgpu::Device::create_buffer_init` measured **8.1 us** per 16-byte uniform on
/// RTX 4090 / D3D12 (`examples/encode_cost.rs`), against a whole forward pass that records
/// in ~0.19 ms. The graph is static, so a handful of distinct uniform values covers
/// every dispatch and after the first pass every lookup hits.
///
/// Keyed by contents rather than by layer, so it stays correct if a caller builds an
/// unexpected config; bounded by the number of distinct shapes, which for the detector is well
/// under twenty per pipeline. A caller sweeping many resolutions through one pipeline would
/// grow it without bound.
///
/// Safe to share across concurrent encodes: each buffer is written at creation and only
/// ever read by the shader afterwards.
#[derive(Debug)]
pub(super) struct UniformCache<T> {
    label: &'static str,
    entries: Mutex<HashMap<T, Arc<wgpu::Buffer>>>,
}

impl<T: Pod + Eq + Hash> UniformCache<T> {
    pub(super) fn new(label: &'static str) -> Self {
        Self {
            label,
            entries: Mutex::new(HashMap::new()),
        }
    }

    /// The buffer for these contents, created once and shared thereafter.
    pub(super) fn buffer(&self, device: &wgpu::Device, uniforms: T) -> Result<Arc<wgpu::Buffer>> {
        let mut entries = self
            .entries
            .lock()
            .map_err(|_| anyhow::anyhow!("{} uniform cache poisoned", self.label))?;
        Ok(entries
            .entry(uniforms)
            .or_insert_with(|| Arc::new(create_uniform_buffer(device, self.label, &uniforms)))
            .clone())
    }
}

/// Records a dispatch in a new profiled pass or an already-open compute pass.
/// Both paths use the same graph and resource preparation code.
pub trait ComputeDispatch {
    /// Bind `pipeline` and `bind_group` at index 0, then dispatch `[x, y, z]` workgroups.
    /// A command encoder opens a pass labelled and profiled through `context`;
    /// an existing compute pass records directly into that pass. Neither submits work.
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
