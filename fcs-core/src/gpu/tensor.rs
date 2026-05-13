use anyhow::{Context, Result, anyhow};
use bytemuck::cast_slice;
use fcs_utils::gpu::{GpuBufferPool, GpuContext};
use std::{
    fmt,
    sync::{Arc, mpsc},
};
use wgpu::util::DeviceExt;

/// Describes the dimensionality of a GPU tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorShape {
    dims: Vec<usize>,
    elements: usize,
}

impl TensorShape {
    /// Create a tensor shape from the provided dimensions.
    pub fn new<D>(dims: D) -> Result<Self>
    where
        D: Into<Vec<usize>>,
    {
        let dims_vec = dims.into();
        anyhow::ensure!(
            !dims_vec.is_empty(),
            "tensor shape must have at least one dimension"
        );
        let mut elements = 1usize;
        for (idx, dim) in dims_vec.iter().enumerate() {
            anyhow::ensure!(
                *dim > 0,
                "dimension {idx} must be greater than zero (got {dim})"
            );
            elements = elements
                .checked_mul(*dim)
                .with_context(|| format!("tensor shape would overflow usize at dimension {idx}"))?;
        }
        Ok(Self {
            dims: dims_vec,
            elements,
        })
    }

    /// Total number of elements described by this shape.
    pub fn elements(&self) -> usize {
        self.elements
    }

    /// Returns the underlying dimensions.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }
}

impl From<TensorShape> for Vec<usize> {
    fn from(value: TensorShape) -> Self {
        value.dims
    }
}

/// Wrapper around a GPU buffer representing an `f32` tensor.
pub struct GpuTensor {
    inner: Arc<GpuTensorInner>,
}

struct GpuTensorInner {
    context: Arc<GpuContext>,
    buffer: wgpu::Buffer,
    shape: TensorShape,
    size_bytes: u64,
    pool: Option<Arc<GpuBufferPool>>,
}

impl Drop for GpuTensorInner {
    fn drop(&mut self) {
        if let Some(pool) = &self.pool {
            pool.recycle(self.buffer.clone(), self.size_bytes, tensor_usage());
        }
    }
}

impl GpuTensor {
    /// Upload host data into a GPU tensor. The data length must match the shape's element count.
    pub fn from_slice<D>(
        context: Arc<GpuContext>,
        dims: D,
        data: &[f32],
        label: Option<&str>,
    ) -> Result<Self>
    where
        D: Into<Vec<usize>>,
    {
        Self::from_slice_with_pool(context, None, dims, data, label)
    }

    /// Upload host data into a GPU tensor backed by the provided pool.
    pub fn from_slice_with_pool<D>(
        context: Arc<GpuContext>,
        pool: Option<Arc<GpuBufferPool>>,
        dims: D,
        data: &[f32],
        label: Option<&str>,
    ) -> Result<Self>
    where
        D: Into<Vec<usize>>,
    {
        Self::from_slice_impl(context, pool, dims, data, label)
    }

    /// Update the contents of the tensor with new data.
    ///
    /// The data length must match the tensor's element count.
    pub fn write(&self, data: &[f32]) -> Result<()> {
        anyhow::ensure!(
            data.len() == self.shape().elements(),
            "write data length {} does not match tensor elements {}",
            data.len(),
            self.shape().elements()
        );
        self.context()
            .queue()
            .write_buffer(&self.inner.buffer, 0, cast_slice(data));
        Ok(())
    }

    /// Allocate an uninitialized GPU tensor (contents undefined) for the provided shape.
    pub fn uninitialized<D>(context: Arc<GpuContext>, dims: D, label: Option<&str>) -> Result<Self>
    where
        D: Into<Vec<usize>>,
    {
        Self::uninitialized_with_pool(context, None, dims, label)
    }

    /// Allocate an uninitialized tensor whose buffer will be recycled via the pool.
    pub fn uninitialized_with_pool<D>(
        context: Arc<GpuContext>,
        pool: Option<Arc<GpuBufferPool>>,
        dims: D,
        label: Option<&str>,
    ) -> Result<Self>
    where
        D: Into<Vec<usize>>,
    {
        Self::uninitialized_impl(context, pool, dims, label)
    }

    /// Download the tensor contents back to the host as a `Vec<f32>`.
    pub fn to_vec(&self) -> Result<Vec<f32>> {
        let device = self.context().device();
        let size_bytes = self.inner.size_bytes;
        let usage = wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ;

        let readback = if let Some(pool) = &self.inner.pool {
            pool.acquire(size_bytes, usage, Some("tensor_readback"))?
        } else {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("tensor_readback"),
                size: size_bytes,
                usage,
                mapped_at_creation: false,
            })
        };

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tensor_readback_encoder"),
        });
        encoder.copy_buffer_to_buffer(self.buffer(), 0, &readback, 0, size_bytes);
        self.context().queue().submit(Some(encoder.finish()));

        let result = read_buffer(
            device,
            &readback,
            self.shape().elements(),
            "gpu tensor readback",
        );

        // Recycle the readback buffer if we have a pool
        if let Some(pool) = &self.inner.pool {
            // Need to drop/unmap first? read_buffer unmaps it.
            // We clone the buffer handle to pass to recycle (wgpu buffers are Arc internals)
            // But wgpu::Buffer is a struct wrapper around an ID (in wgpu-core/hal) or Arc.
            // wgpu::Buffer implements Clone? Yes.
            pool.recycle(readback, size_bytes, usage);
        } else {
            // It will be dropped naturally
            drop(readback);
        }

        result
    }

    /// Returns the associated tensor shape.
    pub fn shape(&self) -> &TensorShape {
        &self.inner.shape
    }

    /// Returns the inner buffer for binding setups.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.inner.buffer
    }

    /// Returns a reference to the tensor's GPU context (useful for pointer equality checks).
    pub fn context(&self) -> &Arc<GpuContext> {
        &self.inner.context
    }

    /// Returns the size of the tensor's GPU buffer in bytes.
    pub fn size_bytes(&self) -> u64 {
        self.inner.size_bytes
    }

    /// Allocate an uninitialized tensor with the same shape, context, and pool as `self`.
    pub fn uninitialized_like(&self, label: Option<&str>) -> Result<Self> {
        let dims = self.shape().dims().to_vec();
        let pool = self.inner.pool.clone();
        Self::uninitialized_with_pool(self.context().clone(), pool, dims, label)
    }

    /// Record a GPU buffer copy from `self` into `dst` on the given encoder (no submit).
    pub fn encode_copy_to(&self, encoder: &mut wgpu::CommandEncoder, dst: &GpuTensor) {
        encoder.copy_buffer_to_buffer(self.buffer(), 0, dst.buffer(), 0, self.inner.size_bytes);
    }

    /// Clone the tensor into a new GPU buffer so callers can keep the original immutable.
    pub fn duplicate(&self, label: Option<&str>) -> Result<Self> {
        let dims = self.shape().dims().to_vec();
        let pool = self.inner.pool.clone();
        let clone = Self::uninitialized_with_pool(self.context().clone(), pool, dims, label)?;
        let size_bytes = self.inner.size_bytes;
        let mut encoder =
            self.context()
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("tensor_clone_encoder"),
                });
        encoder.copy_buffer_to_buffer(self.buffer(), 0, clone.buffer(), 0, size_bytes);
        self.context().queue().submit(Some(encoder.finish()));
        Ok(clone)
    }

    fn from_parts(
        context: Arc<GpuContext>,
        pool: Option<Arc<GpuBufferPool>>,
        buffer: wgpu::Buffer,
        shape: TensorShape,
        size_bytes: u64,
    ) -> Self {
        let inner = GpuTensorInner {
            context,
            buffer,
            shape,
            size_bytes,
            pool,
        };
        Self {
            inner: Arc::new(inner),
        }
    }

    fn from_slice_impl<D>(
        context: Arc<GpuContext>,
        pool: Option<Arc<GpuBufferPool>>,
        dims: D,
        data: &[f32],
        label: Option<&str>,
    ) -> Result<Self>
    where
        D: Into<Vec<usize>>,
    {
        let shape = TensorShape::new(dims)?;
        anyhow::ensure!(
            data.len() == shape.elements(),
            "tensor upload expected {} values, got {}",
            shape.elements(),
            data.len()
        );
        let size_bytes = (shape.elements() * std::mem::size_of::<f32>()) as u64;
        let usage = tensor_usage();
        let buffer = if let Some(pool_ref) = pool.as_ref() {
            let buffer = pool_ref.acquire(size_bytes, usage, label)?;
            context.queue().write_buffer(&buffer, 0, cast_slice(data));
            buffer
        } else {
            context
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label,
                    contents: cast_slice(data),
                    usage,
                })
        };
        Ok(Self::from_parts(context, pool, buffer, shape, size_bytes))
    }

    fn uninitialized_impl<D>(
        context: Arc<GpuContext>,
        pool: Option<Arc<GpuBufferPool>>,
        dims: D,
        label: Option<&str>,
    ) -> Result<Self>
    where
        D: Into<Vec<usize>>,
    {
        let shape = TensorShape::new(dims)?;
        let size_bytes = (shape.elements() * std::mem::size_of::<f32>()) as u64;
        let usage = tensor_usage();
        let buffer = if let Some(pool_ref) = pool.as_ref() {
            pool_ref.acquire(size_bytes, usage, label)?
        } else {
            context.device().create_buffer(&wgpu::BufferDescriptor {
                label,
                size: size_bytes,
                usage,
                mapped_at_creation: false,
            })
        };
        Ok(Self::from_parts(context, pool, buffer, shape, size_bytes))
    }
}

impl Clone for GpuTensor {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl fmt::Debug for GpuTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuTensor")
            .field("dims", &self.shape().dims())
            .finish()
    }
}

fn tensor_usage() -> wgpu::BufferUsages {
    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST
}

fn read_buffer(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    elements: usize,
    label: &str,
) -> Result<Vec<f32>> {
    let size_bytes = (elements * std::mem::size_of::<f32>()) as u64;
    let slice = buffer.slice(0..size_bytes);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .map_err(|e| anyhow!("device poll failed during {label}: {e}"))?;
    receiver
        .recv()
        .with_context(|| format!("{label} callback dropped"))?
        .map_err(|e| anyhow!("failed to map {label}: {e}"))?;

    let data = slice.get_mapped_range();
    let floats: Vec<f32> = cast_slice(&data).to_vec();
    drop(data);
    buffer.unmap();
    anyhow::ensure!(
        floats.len() == elements,
        "{label} returned {} elements, expected {}",
        floats.len(),
        elements
    );
    Ok(floats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fcs_utils::gpu::{GpuAvailability, GpuBufferPool, GpuContextOptions};
    use std::sync::Arc;

    fn test_context() -> Option<Arc<GpuContext>> {
        match GpuContext::init_with_fallback(&GpuContextOptions::default()) {
            GpuAvailability::Available(ctx) => Some(ctx),
            other => {
                eprintln!("Skipping GPU tensor test: {:?}", other);
                None
            }
        }
    }

    #[test]
    fn upload_download_roundtrip() {
        let Some(ctx) = test_context() else {
            return;
        };
        let data: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
        let tensor = GpuTensor::from_slice(ctx, [2usize, 2, 2, 2], &data, Some("tensor_test"))
            .expect("tensor upload");
        assert_eq!(tensor.shape().elements(), data.len());
        let roundtrip = tensor.to_vec().expect("download tensor");
        assert_eq!(roundtrip, data);
    }

    #[test]
    fn pooled_tensors_return_buffers() {
        let Some(ctx) = test_context() else {
            return;
        };
        let pool = Arc::new(GpuBufferPool::new(ctx.clone(), None));
        assert_eq!(pool.available(), 0);
        {
            let tensor = GpuTensor::uninitialized_with_pool(
                ctx.clone(),
                Some(pool.clone()),
                [1usize, 1, 1, 1],
                Some("tensor_pool_test"),
            )
            .expect("allocate pooled tensor");
            assert_eq!(tensor.shape().elements(), 1);
            assert_eq!(pool.available(), 0);
        }
        assert_eq!(pool.available(), 1);
    }

    #[test]
    fn test_tensor_write_update() {
        let Some(ctx) = test_context() else {
            return;
        };
        let initial_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = GpuTensor::from_slice(ctx, [4usize], &initial_data, Some("write_test"))
            .expect("tensor upload");

        let new_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
        tensor.write(&new_data).expect("tensor write");

        let roundtrip = tensor.to_vec().expect("download tensor");
        assert_eq!(roundtrip, new_data);
    }
}
