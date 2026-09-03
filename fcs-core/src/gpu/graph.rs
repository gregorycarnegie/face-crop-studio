use crate::{
    gpu::{
        activation::ActivationKind,
        conv2d::{Conv2dChannels, Conv2dConfig, Conv2dOptions, SpatialDims},
        ops::GpuInferenceOps,
        tensor::GpuTensor,
    },
    yunet::{
        BACKBONE_STAGES, DETECTION_HEADS, DetectionHeadConfig, HeadBlock, NECK_BLOCKS, StageBlock,
    },
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

fn encode_stage0_block(
    encoder: &mut wgpu::CommandEncoder,
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
        SpatialDims::new(640, 640),
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
        SpatialDims::new(320, 320),
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
        SpatialDims::new(320, 320),
        SpatialDims::new(3, 3),
        SpatialDims::new(1, 1),
        SpatialDims::new(1, 1),
        Conv2dOptions::new(16, Some(ActivationKind::Relu)),
    )?;
    ops.encode_conv2d_tensor(encoder, &point, &dw_weight, &dw_bias, &depth_cfg)
        .context("stage0 depthwise")
}

fn encode_stage_blocks(
    encoder: &mut wgpu::CommandEncoder,
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
    encoder: &mut wgpu::CommandEncoder,
    ops: &GpuInferenceOps,
    tensor: &GpuTensor,
) -> Result<GpuTensor> {
    let cfg = crate::gpu::max_pool::MaxPoolConfig::from_tensor(tensor, 2, 2, 0)?;
    ops.encode_max_pool_tensor(encoder, tensor, &cfg)
}

pub fn encode_backbone_features(
    encoder: &mut wgpu::CommandEncoder,
    ops: &GpuInferenceOps,
    weights: &GpuWeights,
    input: &GpuTensor,
    stage_count: usize,
) -> Result<Vec<GpuTensor>> {
    let mut features = Vec::with_capacity(stage_count);
    let mut current = encode_stage0_block(encoder, ops, weights, input)?;
    for stage in BACKBONE_STAGES.iter().take(stage_count) {
        if stage.pool_before {
            current = encode_pool_tensor(encoder, ops, &current)?;
        }
        current = encode_stage_blocks(encoder, ops, weights, &current, stage.blocks)?;
        features.push(current.clone());
    }
    Ok(features)
}

pub struct DetectionLevelOutputs {
    pub feature: GpuTensor,
    pub cls: GpuTensor,
    pub obj: GpuTensor,
    pub bbox: GpuTensor,
    pub kps: GpuTensor,
}

pub fn encode_neck_and_heads(
    encoder: &mut wgpu::CommandEncoder,
    ops: &GpuInferenceOps,
    weights: &GpuWeights,
    features: &[GpuTensor],
) -> Result<[DetectionLevelOutputs; 3]> {
    anyhow::ensure!(
        features.len() >= 5,
        "need at least five backbone outputs (got {})",
        features.len()
    );

    let c3 = features[2].clone();
    let c4 = features[3].clone();
    let c5 = features[4].clone();

    let p5_raw = encode_stage_blocks(encoder, ops, weights, &c5, &NECK_BLOCKS[2..3])?;
    let level2 =
        encode_detection_level(encoder, ops, weights, p5_raw.clone(), &DETECTION_HEADS[2])?;

    let up_p5 = ops.encode_resize2x_tensor(encoder, &p5_raw)?;
    let merged_p4_input = ops.encode_add_tensors(encoder, &up_p5, &c4)?;
    let p4_raw = encode_stage_blocks(encoder, ops, weights, &merged_p4_input, &NECK_BLOCKS[1..2])?;
    let level1 =
        encode_detection_level(encoder, ops, weights, p4_raw.clone(), &DETECTION_HEADS[1])?;

    let up_p4 = ops.encode_resize2x_tensor(encoder, &p4_raw)?;
    let merged_p3_input = ops.encode_add_tensors(encoder, &up_p4, &c3)?;
    let p3_raw = encode_stage_blocks(encoder, ops, weights, &merged_p3_input, &NECK_BLOCKS[0..1])?;
    let level0 =
        encode_detection_level(encoder, ops, weights, p3_raw.clone(), &DETECTION_HEADS[0])?;

    Ok([level0, level1, level2])
}

fn encode_detection_level(
    encoder: &mut wgpu::CommandEncoder,
    ops: &GpuInferenceOps,
    weights: &GpuWeights,
    feature: GpuTensor,
    head: &DetectionHeadConfig,
) -> Result<DetectionLevelOutputs> {
    let cls = encode_head_branch(encoder, ops, weights, &feature, &head.cls)?;
    let obj = encode_head_branch(encoder, ops, weights, &feature, &head.obj)?;
    let bbox = encode_head_branch(encoder, ops, weights, &feature, &head.bbox)?;
    let kps = encode_head_branch(encoder, ops, weights, &feature, &head.kps)?;
    Ok(DetectionLevelOutputs {
        feature,
        cls,
        obj,
        bbox,
        kps,
    })
}

fn encode_head_branch(
    encoder: &mut wgpu::CommandEncoder,
    ops: &GpuInferenceOps,
    weights: &GpuWeights,
    input: &GpuTensor,
    branch: &HeadBlock,
) -> Result<GpuTensor> {
    let point_weight = weight(weights, branch.conv1_weight)?;
    let point_bias = weight(weights, branch.conv1_bias)?;
    let depth_weight = weight(weights, branch.conv2_weight)?;
    let depth_bias = weight(weights, branch.conv2_bias)?;

    let dims = input.shape().dims();
    anyhow::ensure!(
        dims.len() == 4,
        "head branch expects NCHW tensor (got {:?})",
        dims
    );
    let batch = dims[0] as u32;
    let in_channels = dims[1] as u32;
    let height = dims[2] as u32;
    let width = dims[3] as u32;
    let point_out = point_weight.shape().dims()[0] as u32;

    let point_cfg = Conv2dConfig::new(
        batch,
        Conv2dChannels::new(in_channels, point_out),
        SpatialDims::new(width, height),
        SpatialDims::new(1, 1),
        SpatialDims::new(1, 1),
        SpatialDims::new(0, 0),
        Conv2dOptions::new(1, None),
    )?;
    let reduced =
        ops.encode_conv2d_tensor(encoder, input, &point_weight, &point_bias, &point_cfg)?;

    let depth_out = depth_weight.shape().dims()[0] as u32;
    anyhow::ensure!(
        depth_out == point_out,
        "depthwise conv expects {} channels but got {}",
        point_out,
        depth_out
    );
    let depth_cfg = Conv2dConfig::new(
        batch,
        Conv2dChannels::new(point_out, depth_out),
        SpatialDims::new(width, height),
        SpatialDims::new(3, 3),
        SpatialDims::new(1, 1),
        SpatialDims::new(1, 1),
        Conv2dOptions::new(depth_out, None),
    )?;
    ops.encode_conv2d_tensor(encoder, &reduced, &depth_weight, &depth_bias, &depth_cfg)
}

fn encode_stage_block(
    encoder: &mut wgpu::CommandEncoder,
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
    encoder: &mut wgpu::CommandEncoder,
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
    let pad = depth_kernel_w / 2;

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
