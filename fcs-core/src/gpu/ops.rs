use super::activation::{ActivationKind, ActivationPipeline};
use super::add::AddPipeline;
use super::batch_norm::{BatchNormBindings, BatchNormConfig, BatchNormPipeline};
use super::conv2d::{Conv2dConfig, Conv2dPipeline};
use super::max_pool::{MaxPoolConfig, MaxPoolPipeline};
use super::tensor::GpuTensor;
use super::upsample2x::Upsample2xPipeline;

use anyhow::Result;
use fcs_utils::gpu::{GpuBufferPool, GpuContext};
use std::sync::Arc;

/// Collection of GPU-backed YuNet primitives.
///
/// This type owns the compiled WGSL pipelines for convolution, batch
/// normalization, and activations so callers can reuse them across layers.
#[derive(Debug)]
pub struct GpuInferenceOps {
    context: Arc<GpuContext>,
    buffer_pool: Arc<GpuBufferPool>,
    conv2d: Conv2dPipeline,
    batch_norm: BatchNormPipeline,
    activation: ActivationPipeline,
    max_pool: MaxPoolPipeline,
    add: AddPipeline,
    upsample2x: Upsample2xPipeline,
}

impl GpuInferenceOps {
    /// Create the GPU pipelines from an existing [`GpuContext`].
    pub fn new(context: Arc<GpuContext>, memory_limit: Option<u64>) -> Result<Self> {
        let device = context.device();
        let buffer_pool = Arc::new(GpuBufferPool::new(context.clone(), memory_limit));
        Ok(Self {
            conv2d: Conv2dPipeline::new(device, 4)?,
            batch_norm: BatchNormPipeline::new(device)?,
            activation: ActivationPipeline::new(device)?,
            max_pool: MaxPoolPipeline::new(device)?,
            add: AddPipeline::new(device)?,
            upsample2x: Upsample2xPipeline::new(device)?,
            buffer_pool,
            context,
        })
    }

    /// Upload host data into a GPU tensor for reuse across layers.
    pub fn upload_tensor<D>(&self, dims: D, data: &[f32], label: Option<&str>) -> Result<GpuTensor>
    where
        D: Into<Vec<usize>>,
    {
        GpuTensor::from_slice_with_pool(
            self.context.clone(),
            Some(self.buffer_pool.clone()),
            dims,
            data,
            label,
        )
    }

    /// Upload host data into an existing GPU tensor.
    pub fn upload_to_tensor(&self, tensor: &GpuTensor, data: &[f32]) -> Result<()> {
        self.ensure_same_context(tensor, "upload target")?;
        tensor.write(data)
    }

    /// Download a tensor back to host memory.
    pub fn download_tensor(&self, tensor: &GpuTensor) -> Result<Vec<f32>> {
        tensor.to_vec()
    }

    /// Returns the GPU context backing this ops collection.
    pub fn context(&self) -> &Arc<GpuContext> {
        &self.context
    }

    /// Returns the estimated total memory usage (in bytes) of buffers managed by the pool.
    pub fn memory_usage(&self) -> u64 {
        self.buffer_pool.memory_usage()
    }

    fn ensure_same_context(&self, tensor: &GpuTensor, label: &str) -> Result<()> {
        anyhow::ensure!(
            Arc::ptr_eq(tensor.context(), &self.context),
            "{label} tensor was created from a different GPU context"
        );
        Ok(())
    }

    /// Execute a `Conv2D` layer using GPU-resident tensors.
    pub fn conv2d_tensor(
        &self,
        input: &GpuTensor,
        weights: &GpuTensor,
        bias: &GpuTensor,
        config: &Conv2dConfig,
    ) -> Result<GpuTensor> {
        config.validate(
            input.shape().elements(),
            weights.shape().elements(),
            bias.shape().elements(),
        )?;
        self.ensure_same_context(input, "conv2d input")?;
        self.ensure_same_context(weights, "conv2d weights")?;
        self.ensure_same_context(bias, "conv2d bias")?;
        self.conv2d.execute(
            &self.context,
            &self.buffer_pool,
            input,
            weights,
            bias,
            config,
        )
    }

    /// Execute a `Conv2D` layer using GPU-resident tensors with vectorized shader.
    pub fn conv2d_vec4_tensor(
        &self,
        input: &GpuTensor,
        weights: &GpuTensor,
        bias: &GpuTensor,
        config: &Conv2dConfig,
    ) -> Result<GpuTensor> {
        config.validate(
            input.shape().elements(),
            weights.shape().elements(),
            bias.shape().elements(),
        )?;
        self.ensure_same_context(input, "conv2d input")?;
        self.ensure_same_context(weights, "conv2d weights")?;
        self.ensure_same_context(bias, "conv2d bias")?;
        self.conv2d.execute(
            &self.context,
            &self.buffer_pool,
            input,
            weights,
            bias,
            config,
        )
    }

    /// Convenience wrapper that uploads host slices, runs Conv2D, and downloads the result.
    pub fn conv2d(
        &self,
        input: &[f32],
        weights: &[f32],
        bias: &[f32],
        config: &Conv2dConfig,
    ) -> Result<Vec<f32>> {
        let input_tensor =
            self.upload_tensor(config.input_shape_dims(), input, Some("conv_input"))?;
        let weight_tensor =
            self.upload_tensor(config.weight_shape_dims(), weights, Some("conv_weights"))?;
        let bias_tensor = self.upload_tensor(config.bias_shape_dims(), bias, Some("conv_bias"))?;
        let output = self.conv2d_tensor(&input_tensor, &weight_tensor, &bias_tensor, config)?;
        output.to_vec()
    }

    /// Batch-norm that keeps data on the GPU.
    pub fn batch_norm_tensor(
        &self,
        tensor: &GpuTensor,
        gamma: &GpuTensor,
        beta: &GpuTensor,
        mean: &GpuTensor,
        variance: &GpuTensor,
        config: &BatchNormConfig,
    ) -> Result<GpuTensor> {
        config.validate(
            tensor.shape().elements(),
            gamma.shape().elements(),
            beta.shape().elements(),
            mean.shape().elements(),
            variance.shape().elements(),
        )?;
        self.ensure_same_context(tensor, "batch_norm tensor")?;
        self.ensure_same_context(gamma, "batch_norm gamma")?;
        self.ensure_same_context(beta, "batch_norm beta")?;
        self.ensure_same_context(mean, "batch_norm mean")?;
        self.ensure_same_context(variance, "batch_norm variance")?;
        let tensors = BatchNormBindings {
            tensor,
            gamma,
            beta,
            mean,
            variance,
        };
        self.batch_norm.execute(&self.context, tensors, config)
    }

    pub fn batch_norm(
        &self,
        tensor: &[f32],
        gamma: &[f32],
        beta: &[f32],
        mean: &[f32],
        variance: &[f32],
        config: &BatchNormConfig,
    ) -> Result<Vec<f32>> {
        let tensor_gpu =
            self.upload_tensor(config.tensor_shape_dims(), tensor, Some("bn_tensor"))?;
        let channels = config.channels as usize;
        let gamma_gpu = self.upload_tensor([channels], gamma, Some("bn_gamma"))?;
        let beta_gpu = self.upload_tensor([channels], beta, Some("bn_beta"))?;
        let mean_gpu = self.upload_tensor([channels], mean, Some("bn_mean"))?;
        let variance_gpu = self.upload_tensor([channels], variance, Some("bn_variance"))?;
        let output = self.batch_norm_tensor(
            &tensor_gpu,
            &gamma_gpu,
            &beta_gpu,
            &mean_gpu,
            &variance_gpu,
            config,
        )?;
        output.to_vec()
    }

    /// Apply an activation to a GPU tensor (returns a new tensor copy).
    pub fn activation_tensor(&self, tensor: &GpuTensor, kind: ActivationKind) -> Result<GpuTensor> {
        self.ensure_same_context(tensor, "activation tensor")?;
        self.activation.execute(&self.context, tensor, kind)
    }

    pub fn activation(&self, tensor: &[f32], kind: ActivationKind) -> Result<Vec<f32>> {
        let tensor_gpu = self.upload_tensor([tensor.len()], tensor, Some("activation_tensor"))?;
        let output = self.activation_tensor(&tensor_gpu, kind)?;
        output.to_vec()
    }

    /// Max pool on GPU tensors.
    pub fn max_pool_tensor(&self, tensor: &GpuTensor, config: &MaxPoolConfig) -> Result<GpuTensor> {
        self.ensure_same_context(tensor, "max_pool tensor")?;
        self.max_pool
            .execute(&self.context, &self.buffer_pool, tensor, config)
    }

    /// Element-wise addition of two tensors.
    pub fn add_tensors(&self, lhs: &GpuTensor, rhs: &GpuTensor) -> Result<GpuTensor> {
        self.ensure_same_context(lhs, "add lhs")?;
        self.ensure_same_context(rhs, "add rhs")?;
        anyhow::ensure!(
            lhs.shape().dims() == rhs.shape().dims(),
            "add tensors require identical shapes (lhs={:?}, rhs={:?})",
            lhs.shape().dims(),
            rhs.shape().dims()
        );
        self.add.execute(&self.context, &self.buffer_pool, lhs, rhs)
    }

    /// Nearest-neighbour 2x upsample (spatial dimensions doubled).
    pub fn resize2x_tensor(&self, tensor: &GpuTensor) -> Result<GpuTensor> {
        self.ensure_same_context(tensor, "resize tensor")?;
        self.upsample2x
            .execute(&self.context, &self.buffer_pool, tensor)
    }

    // ── Batched-encoder variants ──────────────────────────────────────────────
    // These record GPU work into a shared encoder without submitting.
    // Use them to accumulate an entire inference pass into one command buffer,
    // then call `context().queue().submit(Some(encoder.finish()))` once.

    pub fn encode_conv2d_tensor(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuTensor,
        weights: &GpuTensor,
        bias: &GpuTensor,
        config: &Conv2dConfig,
    ) -> Result<GpuTensor> {
        config.validate(
            input.shape().elements(),
            weights.shape().elements(),
            bias.shape().elements(),
        )?;
        self.ensure_same_context(input, "conv2d input")?;
        self.ensure_same_context(weights, "conv2d weights")?;
        self.ensure_same_context(bias, "conv2d bias")?;
        self.conv2d.encode(
            encoder,
            &self.context,
            &self.buffer_pool,
            input,
            weights,
            bias,
            config,
        )
    }

    pub fn encode_max_pool_tensor(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        tensor: &GpuTensor,
        config: &MaxPoolConfig,
    ) -> Result<GpuTensor> {
        self.ensure_same_context(tensor, "max_pool tensor")?;
        self.max_pool
            .encode(encoder, &self.context, &self.buffer_pool, tensor, config)
    }

    pub fn encode_add_tensors(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        lhs: &GpuTensor,
        rhs: &GpuTensor,
    ) -> Result<GpuTensor> {
        self.ensure_same_context(lhs, "add lhs")?;
        self.ensure_same_context(rhs, "add rhs")?;
        anyhow::ensure!(
            lhs.shape().dims() == rhs.shape().dims(),
            "add tensors require identical shapes (lhs={:?}, rhs={:?})",
            lhs.shape().dims(),
            rhs.shape().dims()
        );
        self.add
            .encode(encoder, &self.context, &self.buffer_pool, lhs, rhs)
    }

    pub fn encode_resize2x_tensor(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        tensor: &GpuTensor,
    ) -> Result<GpuTensor> {
        self.ensure_same_context(tensor, "resize tensor")?;
        self.upsample2x
            .encode(encoder, &self.context, &self.buffer_pool, tensor)
    }
}
