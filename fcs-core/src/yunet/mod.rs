//! YuNet's architecture, independent of how it is executed.
//!
//! The model file supplies weights only; the topology itself lives here, shared
//! by the WGSL graph in [`crate::gpu`] and the pure-Rust one in [`crate::cpu`].
//! It sat under `gpu` until a second backend needed it.

/// Macros for declaring YuNet initializer names.
#[macro_use]
pub mod macros;
/// Loading named float tensors from ONNX model files.
pub mod onnx;
pub mod proto;

use std::path::Path;

use anyhow::Result;

use onnx::OnnxInitializerMap;

/// Initializer names for a pointwise convolution followed by a depthwise convolution.
#[derive(Clone, Copy)]
pub struct StageBlock {
    /// ONNX initializer name of the pointwise weights.
    pub point_weight: &'static str,
    /// ONNX initializer name of the pointwise biases.
    pub point_bias: &'static str,
    /// ONNX initializer name of the depthwise weights.
    pub depth_weight: &'static str,
    /// ONNX initializer name of the depthwise biases.
    pub depth_bias: &'static str,
}

/// An ordered group of separable convolutions in the backbone.
pub struct BackboneStage {
    /// Convolution blocks in execution order.
    pub blocks: &'static [StageBlock],
    /// Whether to apply 2x2 stride-2 max pooling before the first block.
    pub pool_before: bool,
}

/// Initializer names for the two blocks in backbone stage 1.
pub const STAGE1_BLOCKS: [StageBlock; 2] = [
    crate::backbone_block!("1", "1", "426", "427"),
    crate::backbone_block!("1", "2", "429", "430"),
];

/// Initializer names for the two blocks in backbone stage 2.
pub const STAGE2_BLOCKS: [StageBlock; 2] = [
    crate::backbone_block!("2", "1", "432", "433"),
    crate::backbone_block!("2", "2", "435", "436"),
];

/// Initializer names for the two blocks in backbone stage 3.
pub const STAGE3_BLOCKS: [StageBlock; 2] = [
    crate::backbone_block!("3", "1", "438", "439"),
    crate::backbone_block!("3", "2", "441", "442"),
];

/// Initializer names for the two blocks in backbone stage 4.
pub const STAGE4_BLOCKS: [StageBlock; 2] = [
    crate::backbone_block!("4", "1", "444", "445"),
    crate::backbone_block!("4", "2", "447", "448"),
];

/// Initializer names for the two blocks in backbone stage 5.
pub const STAGE5_BLOCKS: [StageBlock; 2] = [
    crate::backbone_block!("5", "1", "450", "451"),
    crate::backbone_block!("5", "2", "453", "454"),
];

/// Backbone stages in execution order, excluding the stage-0 stem.
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

/// Weight/bias pairs for the stem convolution and its separable block, in execution order.
pub const STAGE0_WEIGHT_NAMES: [&str; 6] = [
    "420",
    "421",
    "backbone.model0.conv2.conv1.weight",
    "backbone.model0.conv2.conv1.bias",
    "423",
    "424",
];

/// Neck blocks ordered from the finest to the coarsest feature level.
pub const NECK_BLOCKS: [StageBlock; 3] = [
    crate::neck_block!("0", "462", "463"),
    crate::neck_block!("1", "459", "460"),
    crate::neck_block!("2", "456", "457"),
];

/// Initializer names for one detection branch at one feature level.
#[derive(Clone, Copy)]
pub struct HeadBlock {
    /// ONNX initializer name of the first (pointwise) convolution weights.
    pub conv1_weight: &'static str,
    /// ONNX initializer name of the first convolution biases.
    pub conv1_bias: &'static str,
    /// ONNX initializer name of the second (depthwise) convolution weights.
    pub conv2_weight: &'static str,
    /// ONNX initializer name of the second convolution biases.
    pub conv2_bias: &'static str,
}

impl HeadBlock {
    /// Return first-layer weights/biases followed by second-layer weights/biases.
    pub fn names(&self) -> [&'static str; 4] {
        [
            self.conv1_weight,
            self.conv1_bias,
            self.conv2_weight,
            self.conv2_bias,
        ]
    }
}

/// The four prediction branches at one feature-pyramid level.
pub struct DetectionHeadConfig {
    /// Face classification branch (one channel).
    pub cls: HeadBlock,
    /// Objectness branch (one channel).
    pub obj: HeadBlock,
    /// Bounding-box regression branch (four channels).
    pub bbox: HeadBlock,
    /// Five-landmark regression branch (ten coordinate channels).
    pub kps: HeadBlock,
}

/// Detection heads for strides 8, 16, and 32, in that order.
pub const DETECTION_HEADS: [DetectionHeadConfig; 3] = [
    crate::detection_head!("0"),
    crate::detection_head!("1"),
    crate::detection_head!("2"),
];

/// Load the stem and the requested subset of YuNet weights.
///
/// `stage_count` selects the first 0..=5 backbone stages; `include_neck` adds
/// all three neck blocks; `head_levels` selects the first 0..=3 detection heads.
///
/// # Errors
///
/// Returns an error for out-of-range counts, unreadable or invalid ONNX files,
/// or missing or unsupported initializer tensors.
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
