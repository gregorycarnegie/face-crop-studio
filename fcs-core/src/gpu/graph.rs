use crate::{
    gpu::{
        activation::ActivationKind,
        conv2d::{Conv2dChannels, Conv2dConfig, Conv2dOptions, SpatialDims},
        ops::GpuInferenceOps,
        tensor::GpuTensor,
        utils::ComputeDispatch,
    },
    yunet::{BACKBONE_STAGES, NECK_BLOCKS, StageBlock},
};

use anyhow::{Context, Result, anyhow};
use std::collections::HashMap;

/// GPU-resident weights, keyed by ONNX initializer name.
pub type GpuWeights = HashMap<String, GpuTensor>;

fn weight(weights: &GpuWeights, name: &str) -> Result<GpuTensor> {
    weights
        .get(name)
        .cloned()
        .ok_or_else(|| anyhow!("cached weight '{name}' missing"))
}

/// Width and height of an NCHW tensor.
fn spatial_dims(tensor: &GpuTensor) -> Result<SpatialDims> {
    match *tensor.shape().dims() {
        [_, _, height, width] => Ok(SpatialDims::new(width as u32, height as u32)),
        ref dims => Err(anyhow!("expected an NCHW tensor, got {dims:?}")),
    }
}

/// Sized from the tensors it is given rather than a compiled-in 640, which is what made every
/// other input size fail at "stage0 conv" (experiments 74 and 75). Everything downstream already
/// read its sizes from its inputs.
fn encode_stage0_block(
    encoder: &mut impl ComputeDispatch,
    ops: &GpuInferenceOps,
    weights: &GpuWeights,
    input: &GpuTensor,
) -> Result<GpuTensor> {
    let conv0_weight = weight(weights, "420")?;
    let conv0_bias = weight(weights, "421")?;
    let pw_weight = weight(weights, "backbone.model0.conv2.conv1.weight")?;
    let pw_bias = weight(weights, "backbone.model0.conv2.conv1.bias")?;
    let dw_weight = weight(weights, "423")?;
    let dw_bias = weight(weights, "424")?;

    let conv_cfg = Conv2dConfig::new(
        1,
        Conv2dChannels::new(3, 16),
        spatial_dims(input)?,
        SpatialDims::new(3, 3),
        SpatialDims::new(2, 2),
        SpatialDims::new(1, 1),
        Conv2dOptions::new(1, Some(ActivationKind::Relu)),
    )?;
    let relu0 = ops
        .encode_conv2d_tensor(encoder, input, &conv0_weight, &conv0_bias, &conv_cfg)
        .context("stage0 conv")?;

    let point_cfg = Conv2dConfig::new(
        1,
        Conv2dChannels::new(16, 16),
        spatial_dims(&relu0)?,
        SpatialDims::new(1, 1),
        SpatialDims::new(1, 1),
        SpatialDims::new(0, 0),
        Conv2dOptions::new(1, None),
    )?;
    let point = ops
        .encode_conv2d_tensor(encoder, &relu0, &pw_weight, &pw_bias, &point_cfg)
        .context("stage0 pointwise")?;

    let depth_cfg = Conv2dConfig::new(
        1,
        Conv2dChannels::new(16, 16),
        spatial_dims(&point)?,
        SpatialDims::new(3, 3),
        SpatialDims::new(1, 1),
        SpatialDims::new(1, 1),
        Conv2dOptions::new(16, Some(ActivationKind::Relu)),
    )?;
    ops.encode_conv2d_tensor(encoder, &point, &dw_weight, &dw_bias, &depth_cfg)
        .context("stage0 depthwise")
}

fn encode_stage_blocks(
    encoder: &mut impl ComputeDispatch,
    ops: &GpuInferenceOps,
    weights: &GpuWeights,
    input: &GpuTensor,
    blocks: &[StageBlock],
) -> Result<GpuTensor> {
    let Some((first, rest)) = blocks.split_first() else {
        anyhow::bail!("stage block list cannot be empty");
    };
    let mut current = encode_stage_block(encoder, ops, weights, input, first)?;
    for block in rest {
        current = encode_stage_block(encoder, ops, weights, &current, block)?;
    }
    Ok(current)
}

fn encode_pool_tensor(
    encoder: &mut impl ComputeDispatch,
    ops: &GpuInferenceOps,
    tensor: &GpuTensor,
) -> Result<GpuTensor> {
    let cfg = crate::gpu::max_pool::MaxPoolConfig::from_tensor(tensor, 2, 2, 0)?;
    ops.encode_max_pool_tensor(encoder, tensor, &cfg)
}

/// The backbone, returning the outputs of its last three stages -- the only ones the neck reads.
///
/// It used to return all five, so the 160x160x64 outputs of stages 1 and 2 (6.5 MB each) stayed
/// referenced until the neck had been encoded. Intermediates released inside an execution scope
/// are reused by later layers of the same inference, so a tensor held for no reader is
/// allocation the rest of the graph cannot reuse (experiment 40).
pub fn encode_backbone_features(
    encoder: &mut impl ComputeDispatch,
    ops: &GpuInferenceOps,
    weights: &GpuWeights,
    input: &GpuTensor,
    stage_count: usize,
) -> Result<Vec<GpuTensor>> {
    let mut features = Vec::with_capacity(NECK_INPUTS);
    let mut current = encode_stage0_block(encoder, ops, weights, input)?;
    for (index, stage) in BACKBONE_STAGES.iter().take(stage_count).enumerate() {
        if stage.pool_before {
            current = encode_pool_tensor(encoder, ops, &current)?;
        }
        current = encode_stage_blocks(encoder, ops, weights, &current, stage.blocks)?;
        if index >= stage_count.saturating_sub(NECK_INPUTS) {
            features.push(current.clone());
        }
    }
    Ok(features)
}

/// Backbone stage outputs the neck consumes: the last three.
pub const NECK_INPUTS: usize = 3;

/// Device-resident feature and fused predictions for one detection level.
pub struct DetectionLevelOutputs {
    /// Neck feature map in NCHW order, used as input to this level's prediction branches.
    pub feature: GpuTensor,
    /// cls, obj, bbox and kps concatenated along the channel axis, in that order.
    ///
    /// The four branches read the same feature map, are the same 1x1-then-depthwise shape
    /// and differ only in output channels, so concatenating their weights turns eight
    /// dispatches per level into two. See [`HEAD_BRANCH_CHANNELS`] for the split.
    pub heads: GpuTensor,
}

/// Output channels of the cls, obj, bbox and kps branches, in concatenation order.
pub const HEAD_BRANCH_CHANNELS: [usize; 4] = [1, 1, 4, 10];

/// Key under which the fused weights for one level are stored in [`GpuWeights`].
///
/// Synthetic: these tensors are built by concatenating four ONNX initializers at upload
/// time, so they have no name in the model.
pub fn fused_head_key(level: usize, part: &str) -> String {
    format!("__fused_head{level}_{part}")
}

/// Record the feature-pyramid neck and all three detection heads without submitting.
///
/// `features` contains exactly the last three backbone stage outputs (c3, c4, c5);
/// `weights` must include the fused head entries named by [`fused_head_key`].
/// Returns stride-8, stride-16, and stride-32 outputs in that order, or an error
/// for missing weights or incompatible tensor shapes/contexts.
pub fn encode_neck_and_heads(
    encoder: &mut impl ComputeDispatch,
    ops: &GpuInferenceOps,
    weights: &GpuWeights,
    features: Vec<GpuTensor>,
) -> Result<[DetectionLevelOutputs; 3]> {
    // By value, so nothing outlives the encode that reads it.
    let [c3, c4, c5]: [GpuTensor; NECK_INPUTS] = features.try_into().map_err(|f: Vec<_>| {
        anyhow!(
            "need the last {NECK_INPUTS} backbone outputs (got {})",
            f.len()
        )
    })?;

    let p5_raw = encode_stage_blocks(encoder, ops, weights, &c5, &NECK_BLOCKS[2..3])?;
    let level2 = encode_detection_level(encoder, ops, weights, p5_raw.clone(), 2)?;

    let merged_p4_input = encode_upsample_add(encoder, ops, &p5_raw, &c4)?;
    let p4_raw = encode_stage_blocks(encoder, ops, weights, &merged_p4_input, &NECK_BLOCKS[1..2])?;
    let level1 = encode_detection_level(encoder, ops, weights, p4_raw.clone(), 1)?;

    let merged_p3_input = encode_upsample_add(encoder, ops, &p4_raw, &c3)?;
    let p3_raw = encode_stage_blocks(encoder, ops, weights, &merged_p3_input, &NECK_BLOCKS[0..1])?;
    let level0 = encode_detection_level(encoder, ops, weights, p3_raw.clone(), 0)?;

    Ok([level0, level1, level2])
}

/// `upsample2x(small) + skip`. Each upsample in the neck feeds only its add, so the pair is one
/// dispatch (experiment 36). `FCS_SEPARATE_RESIZE_ADD` restores the two-dispatch form for the
/// in-process A/B.
fn encode_upsample_add(
    encoder: &mut impl ComputeDispatch,
    ops: &GpuInferenceOps,
    small: &GpuTensor,
    skip: &GpuTensor,
) -> Result<GpuTensor> {
    if std::env::var_os("FCS_SEPARATE_RESIZE_ADD").is_some() {
        let up = ops.encode_resize2x_tensor(encoder, small)?;
        return ops.encode_add_tensors(encoder, &up, skip);
    }
    ops.encode_resize2x_add_tensors(encoder, small, skip)
}

fn encode_detection_level(
    encoder: &mut impl ComputeDispatch,
    ops: &GpuInferenceOps,
    weights: &GpuWeights,
    feature: GpuTensor,
    level: usize,
) -> Result<DetectionLevelOutputs> {
    let point_weight = weight(weights, &fused_head_key(level, "point_weight"))?;
    let point_bias = weight(weights, &fused_head_key(level, "point_bias"))?;
    let depth_weight = weight(weights, &fused_head_key(level, "depth_weight"))?;
    let depth_bias = weight(weights, &fused_head_key(level, "depth_bias"))?;

    let dims = feature.shape().dims();
    anyhow::ensure!(
        dims.len() == 4,
        "head branch expects NCHW tensor (got {:?})",
        dims
    );
    let batch = dims[0] as u32;
    let in_channels = dims[1] as u32;
    let height = dims[2] as u32;
    let width = dims[3] as u32;
    let out_channels = point_weight.shape().dims()[0] as u32;
    anyhow::ensure!(
        out_channels as usize == HEAD_BRANCH_CHANNELS.iter().sum::<usize>(),
        "fused head expects {} channels, got {out_channels}",
        HEAD_BRANCH_CHANNELS.iter().sum::<usize>()
    );

    let point_cfg = Conv2dConfig::new(
        batch,
        Conv2dChannels::new(in_channels, out_channels),
        SpatialDims::new(width, height),
        SpatialDims::new(1, 1),
        SpatialDims::new(1, 1),
        SpatialDims::new(0, 0),
        Conv2dOptions::new(1, None),
    )?;
    let reduced =
        ops.encode_conv2d_tensor(encoder, &feature, &point_weight, &point_bias, &point_cfg)?;

    // Depthwise is per-channel, so running one over the concatenation is exactly the four
    // separate depthwise convolutions provided the kernels were concatenated in the same
    // order. Nothing crosses a branch boundary.
    let depth_cfg = Conv2dConfig::new(
        batch,
        Conv2dChannels::new(out_channels, out_channels),
        SpatialDims::new(width, height),
        SpatialDims::new(3, 3),
        SpatialDims::new(1, 1),
        SpatialDims::new(1, 1),
        Conv2dOptions::new(out_channels, None),
    )?;
    let heads =
        ops.encode_conv2d_tensor(encoder, &reduced, &depth_weight, &depth_bias, &depth_cfg)?;

    Ok(DetectionLevelOutputs { feature, heads })
}

fn encode_stage_block(
    encoder: &mut impl ComputeDispatch,
    ops: &GpuInferenceOps,
    weights: &GpuWeights,
    input: &GpuTensor,
    block: &StageBlock,
) -> Result<GpuTensor> {
    let pw = weight(weights, block.point_weight)?;
    let pb = weight(weights, block.point_bias)?;
    let dw = weight(weights, block.depth_weight)?;
    let db = weight(weights, block.depth_bias)?;
    encode_separable_block(encoder, ops, input, &pw, &pb, &dw, &db)
}

fn encode_separable_block(
    encoder: &mut impl ComputeDispatch,
    ops: &GpuInferenceOps,
    input: &GpuTensor,
    point_weight: &GpuTensor,
    point_bias: &GpuTensor,
    depth_weight: &GpuTensor,
    depth_bias: &GpuTensor,
) -> Result<GpuTensor> {
    let dims = input.shape().dims();
    anyhow::ensure!(
        dims.len() == 4,
        "expected NCHW tensor for separable block (got {:?})",
        dims
    );
    let batch = dims[0] as u32;
    let channels = dims[1] as u32;
    let height = dims[2] as u32;
    let width = dims[3] as u32;

    let point_shape = point_weight.shape().dims();
    anyhow::ensure!(
        point_shape.len() == 4,
        "pointwise weights must be 4D (got {:?})",
        point_shape
    );
    let point_out = point_shape[0] as u32;
    let point_kernel_h = point_shape[2] as u32;
    let point_kernel_w = point_shape[3] as u32;
    anyhow::ensure!(
        point_kernel_h == 1 && point_kernel_w == 1,
        "pointwise kernels must be 1x1 (got {}x{})",
        point_kernel_h,
        point_kernel_w
    );

    let point_cfg = Conv2dConfig::new(
        batch,
        Conv2dChannels::new(channels, point_out),
        SpatialDims::new(width, height),
        SpatialDims::new(1, 1),
        SpatialDims::new(1, 1),
        SpatialDims::new(0, 0),
        Conv2dOptions::new(1, None),
    )?;
    let point = ops.encode_conv2d_tensor(encoder, input, point_weight, point_bias, &point_cfg)?;

    let depth_shape = depth_weight.shape().dims();
    anyhow::ensure!(
        depth_shape.len() == 4,
        "depthwise weights must be 4D (got {:?})",
        depth_shape
    );
    let depth_out = depth_shape[0] as u32;
    anyhow::ensure!(
        depth_out == point_out,
        "depthwise output ({depth_out}) must match pointwise output ({point_out})"
    );
    anyhow::ensure!(
        depth_shape[1] as u32 == 1,
        "depthwise weights expect channel multiplier 1 (got {})",
        depth_shape[1]
    );
    let depth_kernel_h = depth_shape[2] as u32;
    let depth_kernel_w = depth_shape[3] as u32;
    anyhow::ensure!(
        depth_kernel_h == depth_kernel_w,
        "depthwise kernels must be square (got {}x{})",
        depth_kernel_h,
        depth_kernel_w
    );
    let pad = crate::model_config::same_padding(depth_kernel_w as usize) as u32;

    let depth_cfg = Conv2dConfig::new(
        batch,
        Conv2dChannels::new(point_out, depth_out),
        SpatialDims::new(width, height),
        SpatialDims::new(depth_kernel_w, depth_kernel_h),
        SpatialDims::new(1, 1),
        SpatialDims::new(pad, pad),
        Conv2dOptions::new(depth_out, Some(ActivationKind::Relu)),
    )?;
    ops.encode_conv2d_tensor(encoder, &point, depth_weight, depth_bias, &depth_cfg)
}
