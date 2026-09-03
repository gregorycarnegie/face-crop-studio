//! YuNet's topology, executed on the CPU.
//!
//! This mirrors [`crate::gpu::graph`] step for step. The two are compared
//! against each other and against `tract`, so any divergence here shows up as a
//! parity failure rather than as a subtly worse detection.
//!
//! The topology itself lives in [`crate::yunet`], shared with the WGSL backend
//! so the two cannot describe different networks.

use std::{collections::HashMap, path::Path};

use anyhow::{Context, Result};

use super::{
    conv2d::{Activation, ConvConfig, ConvWeights, conv2d},
    ops,
    tensor::Tensor,
};
use crate::yunet::{
    BACKBONE_STAGES, DETECTION_HEADS, DetectionHeadConfig, HeadBlock, NECK_BLOCKS, StageBlock,
    load_backbone_weights,
};

/// Convolution parameters read out of the ONNX initializers, keyed by the
/// weight tensor's name.
pub struct Weights {
    convs: HashMap<String, ConvWeights>,
}

impl std::fmt::Debug for Weights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Weights")
            .field("convs", &self.convs.len())
            .finish()
    }
}

impl Weights {
    /// Read every initializer the graph needs and pair weights with biases.
    pub fn load(model_path: &Path) -> Result<Self> {
        let initializers = load_backbone_weights(
            model_path,
            BACKBONE_STAGES.len(),
            true,
            DETECTION_HEADS.len(),
        )?;

        let mut convs = HashMap::new();
        let mut add = |weight: &str, bias: &str| -> Result<()> {
            let w = initializers.tensor(weight)?;
            let b = initializers.tensor(bias)?;
            let dims = w.dims();
            anyhow::ensure!(
                dims.len() == 4,
                "conv weight '{weight}' must be 4-D, got {dims:?}"
            );
            convs.insert(
                weight.to_string(),
                ConvWeights::new(
                    dims[0],
                    dims[1],
                    dims[2],
                    dims[3],
                    w.data().to_vec(),
                    b.data().to_vec(),
                )
                .with_context(|| format!("conv '{weight}'"))?,
            );
            Ok(())
        };

        add("420", "421")?;
        add(
            "backbone.model0.conv2.conv1.weight",
            "backbone.model0.conv2.conv1.bias",
        )?;
        add("423", "424")?;
        for stage in BACKBONE_STAGES.iter() {
            for block in stage.blocks.iter() {
                add(block.point_weight, block.point_bias)?;
                add(block.depth_weight, block.depth_bias)?;
            }
        }
        for block in NECK_BLOCKS.iter() {
            add(block.point_weight, block.point_bias)?;
            add(block.depth_weight, block.depth_bias)?;
        }
        for head in DETECTION_HEADS.iter() {
            for branch in [&head.cls, &head.obj, &head.bbox, &head.kps] {
                add(branch.conv1_weight, branch.conv1_bias)?;
                add(branch.conv2_weight, branch.conv2_bias)?;
            }
        }

        Ok(Self { convs })
    }

    fn get(&self, name: &str) -> Result<&ConvWeights> {
        self.convs
            .get(name)
            .with_context(|| format!("conv weight '{name}' was not loaded"))
    }
}

/// A pointwise 1x1 followed by a depthwise KxK — the block the whole network is
/// built from. `depth_activation` differs between the backbone (ReLU) and the
/// detection heads (none), which is the only thing that varies.
fn separable(
    input: &Tensor,
    weights: &Weights,
    point_name: &str,
    depth_name: &str,
    depth_activation: Activation,
) -> Result<Tensor> {
    let point_w = weights.get(point_name)?;
    anyhow::ensure!(
        point_w.kernel_h == 1 && point_w.kernel_w == 1,
        "pointwise '{point_name}' must be 1x1, got {}x{}",
        point_w.kernel_h,
        point_w.kernel_w
    );
    let point = conv2d(input, point_w, &ConvConfig::pointwise(Activation::None))
        .with_context(|| format!("pointwise '{point_name}'"))?;

    let depth_w = weights.get(depth_name)?;
    anyhow::ensure!(
        depth_w.out_channels == point_w.out_channels,
        "depthwise '{depth_name}' expects {} channels, pointwise produced {}",
        depth_w.out_channels,
        point_w.out_channels
    );
    anyhow::ensure!(
        depth_w.in_channels_per_group == 1,
        "depthwise '{depth_name}' expects a channel multiplier of 1, got {}",
        depth_w.in_channels_per_group
    );
    anyhow::ensure!(
        depth_w.kernel_h == depth_w.kernel_w,
        "depthwise '{depth_name}' must be square, got {}x{}",
        depth_w.kernel_h,
        depth_w.kernel_w
    );
    let pad = depth_w.kernel_w / 2;
    conv2d(
        &point,
        depth_w,
        &ConvConfig::depthwise(depth_w.out_channels, pad, depth_activation),
    )
    .with_context(|| format!("depthwise '{depth_name}'"))
}

fn stage_block(input: &Tensor, weights: &Weights, block: &StageBlock) -> Result<Tensor> {
    separable(
        input,
        weights,
        block.point_weight,
        block.depth_weight,
        Activation::Relu,
    )
}

fn stage_blocks(input: &Tensor, weights: &Weights, blocks: &[StageBlock]) -> Result<Tensor> {
    let Some((first, rest)) = blocks.split_first() else {
        anyhow::bail!("stage block list cannot be empty");
    };
    let mut current = stage_block(input, weights, first)?;
    for block in rest {
        current = stage_block(&current, weights, block)?;
    }
    Ok(current)
}

/// The stem: a strided 3x3 over RGB, then a separable block.
fn stage0(input: &Tensor, weights: &Weights) -> Result<Tensor> {
    let stem = conv2d(
        input,
        weights.get("420")?,
        &ConvConfig {
            stride: 2,
            padding: 1,
            groups: 1,
            activation: Activation::Relu,
        },
    )
    .context("stage0 stem conv")?;

    separable(
        &stem,
        weights,
        "backbone.model0.conv2.conv1.weight",
        "423",
        Activation::Relu,
    )
    .context("stage0 separable")
}

/// Run the backbone, returning the stem output followed by one feature map per
/// stage.
///
/// Feature maps are kept in the vector and the next stage reads them back from
/// there, rather than each stage cloning its output so a copy can be retained.
/// Those clones were five full activation buffers per forward pass.
fn backbone(input: &Tensor, weights: &Weights) -> Result<Vec<Tensor>> {
    let mut features = Vec::with_capacity(BACKBONE_STAGES.len() + 1);
    features.push(stage0(input, weights)?);

    for (i, stage) in BACKBONE_STAGES.iter().enumerate() {
        let previous = features.last().expect("stage0 output is always present");
        let pooled;
        let source = if stage.pool_before {
            pooled = ops::max_pool(previous, 2, 2, 0).with_context(|| format!("stage {i} pool"))?;
            &pooled
        } else {
            previous
        };
        let output = stage_blocks(source, weights, stage.blocks)
            .with_context(|| format!("stage {i} blocks"))?;
        features.push(output);
    }
    Ok(features)
}

/// One detection level's four branch outputs.
struct LevelOutputs {
    cls: Tensor,
    obj: Tensor,
    bbox: Tensor,
    kps: Tensor,
}

fn head_branch(input: &Tensor, weights: &Weights, branch: &HeadBlock) -> Result<Tensor> {
    // No activation on either convolution here — the heads emit raw logits and
    // box regressions, and sigmoid is applied only to cls/obj at the very end.
    separable(
        input,
        weights,
        branch.conv1_weight,
        branch.conv2_weight,
        Activation::None,
    )
}

fn detection_level(
    feature: &Tensor,
    weights: &Weights,
    head: &DetectionHeadConfig,
) -> Result<LevelOutputs> {
    Ok(LevelOutputs {
        cls: head_branch(feature, weights, &head.cls).context("cls branch")?,
        obj: head_branch(feature, weights, &head.obj).context("obj branch")?,
        bbox: head_branch(feature, weights, &head.bbox).context("bbox branch")?,
        kps: head_branch(feature, weights, &head.kps).context("kps branch")?,
    })
}

/// The feature-pyramid neck and the three detection heads.
fn neck_and_heads(features: &[Tensor], weights: &Weights) -> Result<[LevelOutputs; 3]> {
    // features[0] is the stem; the five stage outputs follow it.
    anyhow::ensure!(
        features.len() >= 6,
        "need the stem plus five stage outputs, got {}",
        features.len()
    );
    let (c3, c4, c5) = (&features[3], &features[4], &features[5]);

    let p5 = stage_blocks(c5, weights, &NECK_BLOCKS[2..3]).context("neck p5")?;
    let level2 = detection_level(&p5, weights, &DETECTION_HEADS[2])?;

    let merged_p4 = ops::add(&ops::resize2x(&p5)?, c4).context("neck p4 merge")?;
    let p4 = stage_blocks(&merged_p4, weights, &NECK_BLOCKS[1..2]).context("neck p4")?;
    let level1 = detection_level(&p4, weights, &DETECTION_HEADS[1])?;

    let merged_p3 = ops::add(&ops::resize2x(&p4)?, c3).context("neck p3 merge")?;
    let p3 = stage_blocks(&merged_p3, weights, &NECK_BLOCKS[0..1]).context("neck p3")?;
    let level0 = detection_level(&p3, weights, &DETECTION_HEADS[0])?;

    Ok([level0, level1, level2])
}

/// Flatten a CHW branch output into the HWC row-major layout the decoder wants,
/// applying sigmoid on the way for the branches that need it.
fn to_hwc(tensor: &Tensor, apply_sigmoid: bool) -> Vec<f32> {
    let (c, h, w) = (tensor.channels(), tensor.height(), tensor.width());
    let src = tensor.data();
    let mut out = vec![0.0f32; c * h * w];
    for ch in 0..c {
        for y in 0..h {
            for x in 0..w {
                let value = src[(ch * h + y) * w + x];
                out[(y * w + x) * c + ch] = if apply_sigmoid {
                    ops::sigmoid(value)
                } else {
                    value
                };
            }
        }
    }
    out
}

/// One head output, ready to become a `[rows, channels]` tensor.
pub struct HeadOutput {
    pub rows: usize,
    pub channels: usize,
    pub data: Vec<f32>,
}

/// Run the whole network and return the twelve head outputs in the order the
/// decoder expects: cls x3, obj x3, bbox x3, kps x3.
pub fn forward(input: &Tensor, weights: &Weights) -> Result<Vec<HeadOutput>> {
    let features = backbone(input, weights)?;
    let levels = neck_and_heads(&features, weights)?;

    // Branches come out grouped by level; the decoder wants them grouped by
    // branch, so collect interleaved and then regroup — the same two-step the
    // GPU runtime does, kept identical so the orders cannot drift apart.
    let mut interleaved: Vec<HeadOutput> = Vec::with_capacity(12);
    for level in levels.iter() {
        for (tensor, apply_sigmoid) in [
            (&level.cls, true),
            (&level.obj, true),
            (&level.bbox, false),
            (&level.kps, false),
        ] {
            interleaved.push(HeadOutput {
                rows: tensor.height() * tensor.width(),
                channels: tensor.channels(),
                data: to_hwc(tensor, apply_sigmoid),
            });
        }
    }

    let mut grouped: [Vec<HeadOutput>; 4] = std::array::from_fn(|_| Vec::with_capacity(3));
    for (idx, output) in interleaved.into_iter().enumerate() {
        grouped[idx % 4].push(output);
    }
    Ok(grouped.into_iter().flatten().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_hwc_interleaves_channels_and_can_apply_sigmoid() {
        // Two channels of a 1x2 plane: [[1,2],[3,4]] -> HWC [1,3, 2,4].
        let t = Tensor::new(1, 2, 1, 2, vec![1.0, 2.0, 3.0, 4.0]).expect("tensor");
        assert_eq!(to_hwc(&t, false), vec![1.0, 3.0, 2.0, 4.0]);

        let with_sigmoid = to_hwc(&t, true);
        for (got, raw) in with_sigmoid.iter().zip([1.0, 3.0, 2.0, 4.0]) {
            assert!((got - ops::sigmoid(raw)).abs() < 1e-7);
        }
    }
}
