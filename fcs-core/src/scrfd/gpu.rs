//! Running SCRFD's topology on the WGSL engine.
//!
//! The same generated table [`super::plan`] executes on the CPU, run against the GPU kernels
//! instead. Nothing new was needed from them: SCRFD's 60 convolutions are pointwise, depthwise
//! and dense 3x3, some at stride 2, and `Kernel::Grouped` already reads stride from its
//! uniforms. The neck's adds and its two nearest 2x upsamples are ops YuNet uses too.
//!
//! Weights are uploaded once at load and kept resident; only the input changes per image.

use anyhow::{Context, Result};

use crate::{
    gpu::{
        ActivationKind, GpuInferenceOps, GpuTensor,
        conv2d::{Conv2dChannels, Conv2dConfig, Conv2dOptions, SpatialDims},
        utils::ComputeDispatch,
    },
    scrfd::plan::Step,
};

/// SCRFD's weights, resident on the GPU.
#[derive(Debug)]
pub struct ScrfdGpuWeights {
    /// Per step: the uploaded weights and bias of a convolution, or `None` for other steps.
    convs: Vec<Option<(GpuTensor, GpuTensor)>>,
}

impl ScrfdGpuWeights {
    /// Upload every weight the topology names.
    ///
    /// Read by name through the shared ONNX initializer reader, so an export whose names moved
    /// fails here rather than part-way through a frame.
    pub fn load(ops: &GpuInferenceOps, path: &std::path::Path) -> Result<Self> {
        let mut names = Vec::new();
        for step in super::topology::STEPS {
            if let Step::Conv { weight, bias, .. } = step {
                names.push(weight);
                if !bias.is_empty() {
                    names.push(bias);
                }
            }
        }
        let map = crate::onnx::OnnxInitializerMap::load(path, &names)
            .with_context(|| format!("reading SCRFD weights from {}", path.display()))?;

        let mut convs = Vec::with_capacity(super::topology::STEPS.len());
        for step in super::topology::STEPS {
            convs.push(match step {
                Step::Conv {
                    weight,
                    bias,
                    out_channels,
                    in_per_group,
                    kernel,
                    ..
                } => {
                    let tensor = map.tensor(weight)?;
                    let expected = out_channels * in_per_group * kernel * kernel;
                    anyhow::ensure!(
                        tensor.data().len() == expected,
                        "{weight} holds {} values, the topology expects {expected}",
                        tensor.data().len()
                    );
                    let bias_values = if bias.is_empty() {
                        vec![0.0; out_channels]
                    } else {
                        map.tensor(bias)?.data().to_vec()
                    };
                    let weights = ops.upload_tensor(
                        vec![out_channels, in_per_group, kernel, kernel],
                        tensor.data(),
                        Some(weight),
                    )?;
                    let bias = ops.upload_tensor(vec![out_channels], &bias_values, Some("bias"))?;
                    Some((weights, bias))
                }
                _ => None,
            });
        }
        Ok(Self { convs })
    }
}

/// Run every step on the GPU and return the nine head outputs, in the topology's output order.
///
/// The whole forward pass is recorded into one command buffer and submitted once. Submitting op
/// by op -- 66 queue submissions per detection -- was what SCRFD's port had fallen back to when
/// it replaced YuNet, whose runtime had already made this change for the same reason.
pub fn run(
    ops: &GpuInferenceOps,
    input: &GpuTensor,
    weights: &ScrfdGpuWeights,
) -> Result<Vec<GpuTensor>> {
    let mut encoder =
        ops.context()
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("scrfd"),
            });
    // Every slot is held until after the submit. A tensor dropped while the pass is still being
    // recorded would hand its buffer back to the pool, where another thread could take it and
    // submit its own work first -- ahead of this pass, which still reads that buffer.
    let slots = if ops.context().profiler().is_some() {
        // Per-dispatch timestamps need a pass per dispatch, which recording through the encoder
        // gives.
        encode(&mut encoder, ops, input, weights)?
    } else {
        // One pass for everything: wgpu still inserts the barriers between dispatches that
        // pooled-buffer reuse needs.
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("scrfd_forward"),
            timestamp_writes: None,
        });
        encode(&mut pass, ops, input, weights)?
    };
    ops.context().queue().submit(Some(encoder.finish()));
    outputs(&slots)
}

/// Record every step into `dispatch` and return all the slots, intermediates included, so the
/// caller decides when they may be released.
fn encode(
    dispatch: &mut impl ComputeDispatch,
    ops: &GpuInferenceOps,
    input: &GpuTensor,
    weights: &ScrfdGpuWeights,
) -> Result<Vec<Option<GpuTensor>>> {
    let mut slots: Vec<Option<GpuTensor>> = Vec::with_capacity(super::topology::STEPS.len() + 1);
    slots.push(Some(input.clone()));

    for (index, step) in super::topology::STEPS.iter().enumerate() {
        let produced = match *step {
            Step::Conv {
                input,
                out_channels,
                in_per_group,
                kernel,
                stride,
                pad,
                groups,
                relu,
                ..
            } => {
                let source = slot(&slots, input, index)?;
                let dims = source.shape().dims().to_vec();
                let (height, width) = (dims[2] as u32, dims[3] as u32);
                let input_channels = (in_per_group * groups) as u32;
                let config = Conv2dConfig::new(
                    1,
                    Conv2dChannels::new(input_channels, out_channels as u32),
                    SpatialDims::new(width, height),
                    SpatialDims::new(kernel as u32, kernel as u32),
                    SpatialDims::new(stride as u32, stride as u32),
                    SpatialDims::new(pad as u32, pad as u32),
                    Conv2dOptions::new(groups as u32, relu.then_some(ActivationKind::Relu)),
                )
                .with_context(|| format!("step {index}: convolution configuration"))?;
                let (w, b) = weights.convs[index]
                    .as_ref()
                    .context("a Conv step without weights: the plan and the weights disagree")?;
                ops.encode_conv2d_tensor(dispatch, source, w, b, &config)
                    .with_context(|| format!("step {index}: conv"))?
            }
            Step::Add { a, b } => ops
                .encode_add_tensors(dispatch, slot(&slots, a, index)?, slot(&slots, b, index)?)
                .with_context(|| format!("step {index}: add"))?,
            Step::Upsample2x { input } => ops
                .encode_resize2x_tensor(dispatch, slot(&slots, input, index)?)
                .with_context(|| format!("step {index}: upsample"))?,
        };
        slots.push(Some(produced));
    }

    Ok(slots)
}

/// The head outputs, in the topology's output order.
fn outputs(slots: &[Option<GpuTensor>]) -> Result<Vec<GpuTensor>> {
    super::topology::OUTPUTS
        .iter()
        .map(|&at| {
            slots
                .get(at)
                .and_then(|s| s.clone())
                .with_context(|| format!("output slot {at} was never written"))
        })
        .collect()
}

fn slot(slots: &[Option<GpuTensor>], at: usize, step: usize) -> Result<&GpuTensor> {
    slots
        .get(at)
        .and_then(|s| s.as_ref())
        .with_context(|| format!("step {step} reads slot {at}, which does not exist yet"))
}
