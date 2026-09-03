//! YuNet's architecture, independent of how it is executed.
//!
//! The model file supplies weights only; the topology itself lives here, shared
//! by the WGSL graph in [`crate::gpu`] and the pure-Rust one in [`crate::cpu`].
//! It sat under `gpu` until a second backend needed it.

#[macro_use]
pub mod macros;
pub mod onnx;
pub mod proto;

use std::path::Path;

use anyhow::Result;

use onnx::OnnxInitializerMap;

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
