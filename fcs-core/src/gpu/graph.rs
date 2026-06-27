use crate::gpu::{
    activation::ActivationKind,
    conv2d::{Conv2dChannels, Conv2dConfig, Conv2dOptions, SpatialDims},
    onnx::OnnxInitializerMap,
    ops::GpuInferenceOps,
    tensor::GpuTensor,
};

use anyhow::{Context, Result};
use std::path::Path;

#[derive(Clone, Copy)]
pub struct StageBlock {
    pub point_weight: &'static str,
    pub point_bias: &'static str,
    pub depth_weight: &'static str,
    pub depth_bias: &'static str,
}

pub struct BackboneStage {
    pub blocks: &'static [StageBlock],
    pub pool_before: bool,
}

pub const STAGE1_BLOCKS: [StageBlock; 2] = [
    crate::backbone_block!("1", "1", "426", "427"),
    crate::backbone_block!("1", "2", "429", "430"),
];

pub const STAGE2_BLOCKS: [StageBlock; 2] = [
    crate::backbone_block!("2", "1", "432", "433"),
    crate::backbone_block!("2", "2", "435", "436"),
];

pub const STAGE3_BLOCKS: [StageBlock; 2] = [
    crate::backbone_block!("3", "1", "438", "439"),
    crate::backbone_block!("3", "2", "441", "442"),
];

pub const STAGE4_BLOCKS: [StageBlock; 2] = [
    crate::backbone_block!("4", "1", "444", "445"),
    crate::backbone_block!("4", "2", "447", "448"),
];

pub const STAGE5_BLOCKS: [StageBlock; 2] = [
    crate::backbone_block!("5", "1", "450", "451"),
    crate::backbone_block!("5", "2", "453", "454"),
];

pub const BACKBONE_STAGES: [BackboneStage; 5] = [
    BackboneStage {
        blocks: &STAGE1_BLOCKS,
        pool_before: true,
    },
    BackboneStage {
        blocks: &STAGE2_BLOCKS,
        pool_before: false,
    },
    BackboneStage {
        blocks: &STAGE3_BLOCKS,
        pool_before: true,
    },
    BackboneStage {
        blocks: &STAGE4_BLOCKS,
        pool_before: true,
    },
    BackboneStage {
        blocks: &STAGE5_BLOCKS,
        pool_before: true,
    },
];

pub const STAGE0_WEIGHT_NAMES: [&str; 6] = [
    "420",
    "421",
    "backbone.model0.conv2.conv1.weight",
    "backbone.model0.conv2.conv1.bias",
    "423",
    "424",
];

pub const NECK_BLOCKS: [StageBlock; 3] = [
    crate::neck_block!("0", "462", "463"),
    crate::neck_block!("1", "459", "460"),
    crate::neck_block!("2", "456", "457"),
];

#[derive(Clone, Copy)]
pub struct HeadBlock {
    pub conv1_weight: &'static str,
    pub conv1_bias: &'static str,
    pub conv2_weight: &'static str,
    pub conv2_bias: &'static str,
}

impl HeadBlock {
    pub fn names(&self) -> [&'static str; 4] {
        [
            self.conv1_weight,
            self.conv1_bias,
            self.conv2_weight,
            self.conv2_bias,
        ]
    }
}

pub struct DetectionHeadConfig {
    pub cls: HeadBlock,
    pub obj: HeadBlock,
    pub bbox: HeadBlock,
    pub kps: HeadBlock,
}

pub const DETECTION_HEADS: [DetectionHeadConfig; 3] = [
    crate::detection_head!("0"),
    crate::detection_head!("1"),
    crate::detection_head!("2"),
];

pub trait WeightProvider {
    fn tensor(&self, name: &str) -> Result<GpuTensor>;
}

pub fn load_backbone_weights(
    model_path: &Path,
    stage_count: usize,
    include_neck: bool,
    head_levels: usize,
) -> Result<OnnxInitializerMap> {
    anyhow::ensure!(
        stage_count <= BACKBONE_STAGES.len(),
        "requested {stage_count} backbone stages but only {} are available",
        BACKBONE_STAGES.len()
    );
    anyhow::ensure!(
        head_levels <= DETECTION_HEADS.len(),
        "requested {head_levels} head levels but only {} are available",
        DETECTION_HEADS.len()
    );
    let mut names: Vec<&str> = Vec::new();
    names.extend_from_slice(&STAGE0_WEIGHT_NAMES);
    for stage in BACKBONE_STAGES.iter().take(stage_count) {
        for block in stage.blocks.iter() {
            names.push(block.point_weight);
            names.push(block.point_bias);
            names.push(block.depth_weight);
            names.push(block.depth_bias);
        }
    }
    if include_neck {
        for block in NECK_BLOCKS.iter() {
            names.push(block.point_weight);
            names.push(block.point_bias);
            names.push(block.depth_weight);
            names.push(block.depth_bias);
        }
    }
    if head_levels > 0 {
        for head in DETECTION_HEADS.iter().take(head_levels) {
            names.extend_from_slice(&head.cls.names());
            names.extend_from_slice(&head.obj.names());
            names.extend_from_slice(&head.bbox.names());
            names.extend_from_slice(&head.kps.names());
        }
    }
    OnnxInitializerMap::load(model_path, &names)
}

pub fn run_stage0_block<W: WeightProvider>(
    ops: &GpuInferenceOps,
    weights: &W,
    input: &GpuTensor,
) -> Result<GpuTensor> {
    let conv0_weight = weights.tensor("420")?;
    let conv0_bias = weights.tensor("421")?;
    let pw_weight = weights.tensor("backbone.model0.conv2.conv1.weight")?;
    let pw_bias = weights.tensor("backbone.model0.conv2.conv1.bias")?;
    let dw_weight = weights.tensor("423")?;
    let dw_bias = weights.tensor("424")?;

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
        .conv2d_tensor(input, &conv0_weight, &conv0_bias, &conv_cfg)
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
        .conv2d_tensor(&relu0, &pw_weight, &pw_bias, &point_cfg)
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
    ops.conv2d_tensor(&point, &dw_weight, &dw_bias, &depth_cfg)
        .context("stage0 depthwise")
}

fn encode_stage0_block<W: WeightProvider>(
    encoder: &mut wgpu::CommandEncoder,
    ops: &GpuInferenceOps,
    weights: &W,
    input: &GpuTensor,
) -> Result<GpuTensor> {
    let conv0_weight = weights.tensor("420")?;
    let conv0_bias = weights.tensor("421")?;
    let pw_weight = weights.tensor("backbone.model0.conv2.conv1.weight")?;
    let pw_bias = weights.tensor("backbone.model0.conv2.conv1.bias")?;
    let dw_weight = weights.tensor("423")?;
    let dw_bias = weights.tensor("424")?;

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

pub fn run_stage_blocks<W: WeightProvider>(
    ops: &GpuInferenceOps,
    weights: &W,
    input: &GpuTensor,
    blocks: &[StageBlock],
) -> Result<GpuTensor> {
    let Some((first, rest)) = blocks.split_first() else {
        anyhow::bail!("stage block list cannot be empty");
    };
    let mut current = run_stage_block(
        ops,
        weights,
        input,
        first.point_weight,
        first.point_bias,
        first.depth_weight,
        first.depth_bias,
    )?;
    for block in rest {
        current = run_stage_block(
            ops,
            weights,
            &current,
            block.point_weight,
            block.point_bias,
            block.depth_weight,
            block.depth_bias,
        )?;
    }
    Ok(current)
}

fn encode_stage_blocks<W: WeightProvider>(
    encoder: &mut wgpu::CommandEncoder,
    ops: &GpuInferenceOps,
    weights: &W,
    input: &GpuTensor,
    blocks: &[StageBlock],
) -> Result<GpuTensor> {
    let Some((first, rest)) = blocks.split_first() else {
        anyhow::bail!("stage block list cannot be empty");
    };
    let mut current = encode_stage_block(
        encoder,
        ops,
        weights,
        input,
        first.point_weight,
        first.point_bias,
        first.depth_weight,
        first.depth_bias,
    )?;
    for block in rest {
        current = encode_stage_block(
            encoder,
            ops,
            weights,
            &current,
            block.point_weight,
            block.point_bias,
            block.depth_weight,
            block.depth_bias,
        )?;
    }
    Ok(current)
}

pub fn pool_tensor(ops: &GpuInferenceOps, tensor: &GpuTensor) -> Result<GpuTensor> {
    let cfg = crate::gpu::max_pool::MaxPoolConfig::from_tensor(tensor, 2, 2, 0)?;
    ops.max_pool_tensor(tensor, &cfg)
}

fn encode_pool_tensor(
    encoder: &mut wgpu::CommandEncoder,
    ops: &GpuInferenceOps,
    tensor: &GpuTensor,
) -> Result<GpuTensor> {
    let cfg = crate::gpu::max_pool::MaxPoolConfig::from_tensor(tensor, 2, 2, 0)?;
    ops.encode_max_pool_tensor(encoder, tensor, &cfg)
}

pub fn run_backbone_features<W: WeightProvider>(
    ops: &GpuInferenceOps,
    weights: &W,
    input: &GpuTensor,
    stage_count: usize,
) -> Result<Vec<GpuTensor>> {
    let mut features = Vec::with_capacity(stage_count);
    let mut current = run_stage0_block(ops, weights, input)?;
    for stage in BACKBONE_STAGES.iter().take(stage_count) {
        if stage.pool_before {
            current = pool_tensor(ops, &current)?;
        }
        current = run_stage_blocks(ops, weights, &current, stage.blocks)?;
        features.push(current.clone());
    }
    Ok(features)
}

pub fn encode_backbone_features<W: WeightProvider>(
    encoder: &mut wgpu::CommandEncoder,
    ops: &GpuInferenceOps,
    weights: &W,
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

pub fn encode_neck_and_heads<W: WeightProvider>(
    encoder: &mut wgpu::CommandEncoder,
    ops: &GpuInferenceOps,
    weights: &W,
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

pub fn run_neck_and_heads<W: WeightProvider>(
    ops: &GpuInferenceOps,
    weights: &W,
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

    let p5_raw = run_stage_blocks(ops, weights, &c5, &NECK_BLOCKS[2..3])?;
    let level2 = run_detection_level(ops, weights, p5_raw.clone(), &DETECTION_HEADS[2])?;

    let up_p5 = ops.resize2x_tensor(&p5_raw)?;
    let merged_p4_input = ops.add_tensors(&up_p5, &c4)?;
    let p4_raw = run_stage_blocks(ops, weights, &merged_p4_input, &NECK_BLOCKS[1..2])?;
    let level1 = run_detection_level(ops, weights, p4_raw.clone(), &DETECTION_HEADS[1])?;

    let up_p4 = ops.resize2x_tensor(&p4_raw)?;
    let merged_p3_input = ops.add_tensors(&up_p4, &c3)?;
    let p3_raw = run_stage_blocks(ops, weights, &merged_p3_input, &NECK_BLOCKS[0..1])?;
    let level0 = run_detection_level(ops, weights, p3_raw.clone(), &DETECTION_HEADS[0])?;

    Ok([level0, level1, level2])
}

fn encode_detection_level<W: WeightProvider>(
    encoder: &mut wgpu::CommandEncoder,
    ops: &GpuInferenceOps,
    weights: &W,
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

pub fn run_detection_level<W: WeightProvider>(
    ops: &GpuInferenceOps,
    weights: &W,
    feature: GpuTensor,
    head: &DetectionHeadConfig,
) -> Result<DetectionLevelOutputs> {
    let cls = run_head_branch(ops, weights, &feature, &head.cls)?;
    let obj = run_head_branch(ops, weights, &feature, &head.obj)?;
    let bbox = run_head_branch(ops, weights, &feature, &head.bbox)?;
    let kps = run_head_branch(ops, weights, &feature, &head.kps)?;
    Ok(DetectionLevelOutputs {
        feature,
        cls,
        obj,
        bbox,
        kps,
    })
}

fn encode_head_branch<W: WeightProvider>(
    encoder: &mut wgpu::CommandEncoder,
    ops: &GpuInferenceOps,
    weights: &W,
    input: &GpuTensor,
    branch: &HeadBlock,
) -> Result<GpuTensor> {
    let point_weight = weights.tensor(branch.conv1_weight)?;
    let point_bias = weights.tensor(branch.conv1_bias)?;
    let depth_weight = weights.tensor(branch.conv2_weight)?;
    let depth_bias = weights.tensor(branch.conv2_bias)?;

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

fn run_head_branch<W: WeightProvider>(
    ops: &GpuInferenceOps,
    weights: &W,
    input: &GpuTensor,
    branch: &HeadBlock,
) -> Result<GpuTensor> {
    let point_weight = weights.tensor(branch.conv1_weight)?;
    let point_bias = weights.tensor(branch.conv1_bias)?;
    let depth_weight = weights.tensor(branch.conv2_weight)?;
    let depth_bias = weights.tensor(branch.conv2_bias)?;

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
    let reduced = ops.conv2d_tensor(input, &point_weight, &point_bias, &point_cfg)?;

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
    ops.conv2d_tensor(&reduced, &depth_weight, &depth_bias, &depth_cfg)
}

#[allow(clippy::too_many_arguments)]
fn encode_stage_block<W: WeightProvider>(
    encoder: &mut wgpu::CommandEncoder,
    ops: &GpuInferenceOps,
    weights: &W,
    input: &GpuTensor,
    point_weight: &str,
    point_bias: &str,
    depth_weight: &str,
    depth_bias: &str,
) -> Result<GpuTensor> {
    let pw = weights.tensor(point_weight)?;
    let pb = weights.tensor(point_bias)?;
    let dw = weights.tensor(depth_weight)?;
    let db = weights.tensor(depth_bias)?;
    encode_separable_block(encoder, ops, input, &pw, &pb, &dw, &db)
}

pub fn run_stage_block<W: WeightProvider>(
    ops: &GpuInferenceOps,
    weights: &W,
    input: &GpuTensor,
    point_weight: &str,
    point_bias: &str,
    depth_weight: &str,
    depth_bias: &str,
) -> Result<GpuTensor> {
    let pw = weights.tensor(point_weight)?;
    let pb = weights.tensor(point_bias)?;
    let dw = weights.tensor(depth_weight)?;
    let db = weights.tensor(depth_bias)?;
    run_separable_block(ops, input, &pw, &pb, &dw, &db)
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

pub fn run_separable_block(
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
    let point = ops.conv2d_tensor(input, point_weight, point_bias, &point_cfg)?;

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
    ops.conv2d_tensor(&point, depth_weight, depth_bias, &depth_cfg)
}
