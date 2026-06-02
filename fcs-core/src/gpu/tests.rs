use super::*;
use anyhow::{Context, Result};
use fcs_utils::gpu::{GpuAvailability, GpuContext, GpuContextOptions};
use std::path::{Path, PathBuf};
use tract_onnx::prelude::*;

use crate::gpu::{
    OnnxInitializerMap, OnnxTensor,
    conv2d::{Conv2dChannels, Conv2dOptions, SpatialDims},
    max_pool::MaxPoolConfig,
};

macro_rules! test_backbone_stage {
    ($test_name:ident, $stage:expr, $ref_node:expr, $input_name:expr, $pool:expr) => {
        #[test]
        fn $test_name() {
            let Some(ops) = gpu_ops() else {
                eprintln!("Skipping {} (no adapter)", stringify!($test_name));
                return;
            };

            let Some(model_path) = model_file_path() else {
                eprintln!(
                    "Skipping {} (model file {} missing)",
                    stringify!($test_name),
                    MODEL_REL_PATH
                );
                return;
            };

            let loader = load_backbone_weights(&model_path, $stage, false, 0)
                .expect("load backbone weights");

            let input_data = synthetic_input();
            let cpu_vec = reference_output(&model_path, $ref_node, &input_data);

            let input_gpu = ops
                .upload_tensor([1usize, 3, 640, 640], &input_data, Some($input_name))
                .expect("upload input");
            let output_gpu =
                run_backbone_to_stage(&ops, &loader, &input_gpu, $stage).expect("run backbone");

            let final_gpu = if $pool {
                pool_tensor(&ops, &output_gpu).expect("pool tensor")
            } else {
                output_gpu
            };

            let gpu_vec = final_gpu.to_vec().expect("download output");

            assert_eq!(gpu_vec.len(), cpu_vec.len(), "shape mismatch");
            let max_diff = gpu_vec
                .iter()
                .zip(cpu_vec.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff < 1e-3,
                "{} mismatch (max diff {max_diff})",
                stringify!($test_name)
            );
        }
    };
}

fn gpu_ops() -> Option<GpuInferenceOps> {
    match GpuContext::init_with_fallback(&GpuContextOptions::default()) {
        GpuAvailability::Available(ctx) => {
            Some(GpuInferenceOps::new(ctx, None).expect("build ops"))
        }
        _ => None,
    }
}

fn synthetic_input() -> Vec<f32> {
    let input_shape = 3 * 640 * 640;
    (0..input_shape)
        .map(|i| ((i % 257) as f32) / 256.0)
        .collect()
}

fn upload_onx_tensor(ops: &GpuInferenceOps, tensor: &OnnxTensor, label: &str) -> Result<GpuTensor> {
    ops.upload_tensor(tensor.dims().to_vec(), tensor.data(), Some(label))
}

const MODEL_REL_PATH: &str = "models/face_detection_yunet_2023mar_640.onnx";

fn model_file_path() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    candidates.push(PathBuf::from(MODEL_REL_PATH));

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if let Some(workspace_root) = manifest_dir.parent() {
        candidates.push(workspace_root.join(MODEL_REL_PATH));
    }

    candidates.into_iter().find(|path| path.exists())
}

fn reference_tensor(model_path: &Path, node: &str, input: &[f32]) -> (Vec<f32>, Vec<usize>) {
    let mut model = tract_onnx::onnx()
        .model_for_path(model_path)
        .expect("load reference ONNX");
    model
        .select_outputs_by_name([node])
        .expect("set reference output");
    let plan = model
        .into_optimized()
        .expect("optimize reference")
        .into_runnable()
        .expect("plan reference graph");
    let arr = tract_ndarray::Array4::from_shape_vec((1, 3, 640, 640), input.to_vec())
        .expect("reference input tensor shape should be valid");
    let tensor = plan
        .run(tvec!(arr.into_tensor().into()))
        .expect("run reference graph")
        .remove(0)
        .into_tensor()
        .into_plain_array::<f32>()
        .expect("convert reference output");
    let shape = tensor.shape().to_vec();
    (tensor.into_raw_vec_and_offset().0, shape)
}

fn reference_output(model_path: &Path, node: &str, input: &[f32]) -> Vec<f32> {
    reference_tensor(model_path, node, input).0
}

fn assert_tensor_matches(model_path: &Path, node: &str, tensor: &GpuTensor, input: &[f32]) {
    let cpu_vec = reference_output(model_path, node, input);
    let gpu_vec = tensor.to_vec().expect("download tensor");
    assert_eq!(
        gpu_vec.len(),
        cpu_vec.len(),
        "node {node} produced mismatched length"
    );
    let max_diff = gpu_vec
        .iter()
        .zip(cpu_vec.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-3,
        "node {node} mismatch (max diff {max_diff})"
    );
}

fn upload_named(
    ops: &GpuInferenceOps,
    loader: &OnnxInitializerMap,
    name: &str,
) -> Result<GpuTensor> {
    let tensor = loader
        .tensor(name)
        .with_context(|| format!("initializer '{name}' missing"))?;
    upload_onx_tensor(ops, tensor, name)
}

use crate::gpu::graph::{
    BACKBONE_STAGES, DETECTION_HEADS, DetectionHeadConfig, HeadBlock, NECK_BLOCKS,
    STAGE0_WEIGHT_NAMES, STAGE1_BLOCKS, StageBlock,
};

fn load_backbone_weights(
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

fn run_stage_blocks(
    ops: &GpuInferenceOps,
    loader: &OnnxInitializerMap,
    input: &GpuTensor,
    blocks: &[StageBlock],
) -> Result<GpuTensor> {
    let Some((first, rest)) = blocks.split_first() else {
        anyhow::bail!("stage block list cannot be empty");
    };
    let mut current = run_stage_block(
        ops,
        loader,
        input,
        first.point_weight,
        first.point_bias,
        first.depth_weight,
        first.depth_bias,
    )?;
    for block in rest {
        current = run_stage_block(
            ops,
            loader,
            &current,
            block.point_weight,
            block.point_bias,
            block.depth_weight,
            block.depth_bias,
        )?;
    }
    Ok(current)
}

fn run_backbone_to_stage(
    ops: &GpuInferenceOps,
    loader: &OnnxInitializerMap,
    input: &GpuTensor,
    stage_count: usize,
) -> Result<GpuTensor> {
    anyhow::ensure!(stage_count > 0, "stage_count must be > 0");
    anyhow::ensure!(
        stage_count <= BACKBONE_STAGES.len(),
        "requested stage {} exceeds available {}",
        stage_count,
        BACKBONE_STAGES.len()
    );
    let mut current = run_stage0_block(ops, loader, input)?;
    for stage in BACKBONE_STAGES.iter().take(stage_count) {
        if stage.pool_before {
            current = pool_tensor(ops, &current)?;
        }
        current = run_stage_blocks(ops, loader, &current, stage.blocks)?;
    }
    Ok(current)
}

fn run_stage0_block(
    ops: &GpuInferenceOps,
    loader: &OnnxInitializerMap,
    input: &GpuTensor,
) -> Result<GpuTensor> {
    let conv0_weight = upload_named(ops, loader, "420")?;
    let conv0_bias = upload_named(ops, loader, "421")?;
    let pw_weight = upload_named(ops, loader, "backbone.model0.conv2.conv1.weight")?;
    let pw_bias = upload_named(ops, loader, "backbone.model0.conv2.conv1.bias")?;
    let dw_weight = upload_named(ops, loader, "423")?;
    let dw_bias = upload_named(ops, loader, "424")?;

    let conv_cfg = Conv2dConfig::new(
        1,
        Conv2dChannels::new(3, 16),
        SpatialDims::new(640, 640),
        SpatialDims::new(3, 3),
        SpatialDims::new(2, 2),
        SpatialDims::new(1, 1),
        Conv2dOptions::new(1, Some(ActivationKind::Relu)),
    )
    .expect("conv config");
    let relu0 = ops
        .conv2d_tensor(input, &conv0_weight, &conv0_bias, &conv_cfg)
        .expect("stage0 conv");

    let point_cfg = Conv2dConfig::new(
        1,
        Conv2dChannels::new(16, 16),
        SpatialDims::new(320, 320),
        SpatialDims::new(1, 1),
        SpatialDims::new(1, 1),
        SpatialDims::new(0, 0),
        Conv2dOptions::new(1, None),
    )
    .expect("stage0 pointwise conv config should be valid");
    let point = ops
        .conv2d_tensor(&relu0, &pw_weight, &pw_bias, &point_cfg)
        .expect("stage0 pw");

    let depth_cfg = Conv2dConfig::new(
        1,
        Conv2dChannels::new(16, 16),
        SpatialDims::new(320, 320),
        SpatialDims::new(3, 3),
        SpatialDims::new(1, 1),
        SpatialDims::new(1, 1),
        Conv2dOptions::new(16, Some(ActivationKind::Relu)),
    )
    .expect("stage0 depthwise conv config should be valid");
    ops.conv2d_tensor(&point, &dw_weight, &dw_bias, &depth_cfg)
}

fn run_backbone_features(
    ops: &GpuInferenceOps,
    loader: &OnnxInitializerMap,
    input: &GpuTensor,
    stage_count: usize,
) -> Result<Vec<GpuTensor>> {
    anyhow::ensure!(
        stage_count <= BACKBONE_STAGES.len(),
        "requested {stage_count} backbone outputs but only {} exist",
        BACKBONE_STAGES.len()
    );
    let mut features = Vec::with_capacity(stage_count);
    let mut current = run_stage0_block(ops, loader, input)?;
    for stage in BACKBONE_STAGES.iter().take(stage_count) {
        if stage.pool_before {
            current = pool_tensor(ops, &current)?;
        }
        current = run_stage_blocks(ops, loader, &current, stage.blocks)?;
        features.push(current.clone());
    }
    Ok(features)
}

fn pool_tensor(ops: &GpuInferenceOps, tensor: &GpuTensor) -> Result<GpuTensor> {
    let cfg = MaxPoolConfig::from_tensor(tensor, 2, 2, 0)?;
    ops.max_pool_tensor(tensor, &cfg)
}

struct DetectionLevelOutputs {
    feature: GpuTensor,
    cls: GpuTensor,
    obj: GpuTensor,
    bbox: GpuTensor,
    kps: GpuTensor,
}

fn run_neck_and_heads(
    ops: &GpuInferenceOps,
    loader: &OnnxInitializerMap,
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

    let p5_raw = run_stage_blocks(ops, loader, &c5, &NECK_BLOCKS[2..3])?;
    let level2 = run_detection_level(ops, loader, p5_raw.clone(), &DETECTION_HEADS[2])?;

    let up_p5 = ops.resize2x_tensor(&p5_raw)?;
    let merged_p4_input = ops.add_tensors(&up_p5, &c4)?;
    let p4_raw = run_stage_blocks(ops, loader, &merged_p4_input, &NECK_BLOCKS[1..2])?;
    let level1 = run_detection_level(ops, loader, p4_raw.clone(), &DETECTION_HEADS[1])?;

    let up_p4 = ops.resize2x_tensor(&p4_raw)?;
    let merged_p3_input = ops.add_tensors(&up_p4, &c3)?;
    let p3_raw = run_stage_blocks(ops, loader, &merged_p3_input, &NECK_BLOCKS[0..1])?;
    let level0 = run_detection_level(ops, loader, p3_raw.clone(), &DETECTION_HEADS[0])?;

    Ok([level0, level1, level2])
}

fn run_detection_level(
    ops: &GpuInferenceOps,
    loader: &OnnxInitializerMap,
    feature: GpuTensor,
    head: &DetectionHeadConfig,
) -> Result<DetectionLevelOutputs> {
    let cls = run_head_branch(ops, loader, &feature, &head.cls)?;
    let obj = run_head_branch(ops, loader, &feature, &head.obj)?;
    let bbox = run_head_branch(ops, loader, &feature, &head.bbox)?;
    let kps = run_head_branch(ops, loader, &feature, &head.kps)?;
    Ok(DetectionLevelOutputs {
        feature,
        cls,
        obj,
        bbox,
        kps,
    })
}

fn run_head_branch(
    ops: &GpuInferenceOps,
    loader: &OnnxInitializerMap,
    input: &GpuTensor,
    branch: &HeadBlock,
) -> Result<GpuTensor> {
    let point_weight = upload_named(ops, loader, branch.conv1_weight)?;
    let point_bias = upload_named(ops, loader, branch.conv1_bias)?;
    let depth_weight = upload_named(ops, loader, branch.conv2_weight)?;
    let depth_bias = upload_named(ops, loader, branch.conv2_bias)?;

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

fn run_separable_block(
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

fn run_stage_block(
    ops: &GpuInferenceOps,
    loader: &OnnxInitializerMap,
    input: &GpuTensor,
    point_weight: &str,
    point_bias: &str,
    depth_weight: &str,
    depth_bias: &str,
) -> Result<GpuTensor> {
    let pw = upload_named(ops, loader, point_weight)?;
    let pb = upload_named(ops, loader, point_bias)?;
    let dw = upload_named(ops, loader, depth_weight)?;
    let db = upload_named(ops, loader, depth_bias)?;
    run_separable_block(ops, input, &pw, &pb, &dw, &db)
}

#[test]
fn activation_matches_cpu() {
    let Some(ops) = gpu_ops() else {
        eprintln!("Skipping activation GPU test (no adapter)");
        return;
    };
    let tensor: Vec<f32> = (0..32).map(|i| i as f32 - 16.0).collect();
    let relu_gpu = ops
        .activation(&tensor, ActivationKind::Relu)
        .expect("ReLU activation should run on GPU");
    let relu_cpu: Vec<f32> = tensor.iter().map(|v| v.max(0.0)).collect();
    assert_eq!(relu_gpu, relu_cpu);

    let sigmoid_gpu = ops
        .activation(&tensor, ActivationKind::Sigmoid)
        .expect("sigmoid activation should run on GPU");
    let sigmoid_cpu: Vec<f32> = tensor.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
    assert!(
        sigmoid_gpu
            .iter()
            .zip(sigmoid_cpu.iter())
            .all(|(a, b)| (a - b).abs() < 1e-4),
        "sigmoid mismatch"
    );
}

#[test]
fn batch_norm_matches_cpu() {
    let Some(ops) = gpu_ops() else {
        eprintln!("Skipping batch-norm GPU test (no adapter)");
        return;
    };
    let config =
        BatchNormConfig::new(4, 2, 3, 1e-5).expect("batch norm test config should be valid");
    let tensor: Vec<f32> = (0..24).map(|i| (i % 7) as f32 * 0.25).collect();
    let gamma: Vec<f32> = vec![1.0, 0.75, 1.25];
    let beta: Vec<f32> = vec![0.0, 0.1, -0.1];
    let mean: Vec<f32> = vec![0.5, 0.4, 0.3];
    let variance: Vec<f32> = vec![0.2, 0.3, 0.1];
    let gpu = ops
        .batch_norm(&tensor, &gamma, &beta, &mean, &variance, &config)
        .expect("batch norm should run on GPU");
    let cpu = batch_norm_cpu(&tensor, &gamma, &beta, &mean, &variance, &config);
    assert!(
        gpu.iter()
            .zip(cpu.iter())
            .all(|(a, b)| (a - b).abs() < 1e-4),
        "batch-norm mismatch"
    );
}

#[test]
fn conv2d_matches_cpu_groups() {
    let Some(ops) = gpu_ops() else {
        eprintln!("Skipping conv2d GPU test (no adapter)");
        return;
    };
    let config = Conv2dConfig::new(
        1,
        Conv2dChannels::new(4, 4),
        SpatialDims::new(4, 4),
        SpatialDims::new(3, 3),
        SpatialDims::new(1, 1),
        SpatialDims::new(1, 1),
        Conv2dOptions::new(2, None),
    )
    .expect("grouped conv2d test config should be valid");
    let input: Vec<f32> = (0..64).map(|i| ((i * 13 % 17) as f32) * 0.1).collect();
    let weights_len = (config.output_channels as usize)
        * ((config.input_channels / config.groups) as usize)
        * config.kernel_width as usize
        * config.kernel_height as usize;
    let weights: Vec<f32> = (0..weights_len)
        .map(|i| ((i * 7 % 19) as f32) * 0.05)
        .collect();
    let bias: Vec<f32> = (0..config.output_channels)
        .map(|i| i as f32 * 0.1 - 0.2)
        .collect();
    let gpu = ops
        .conv2d(&input, &weights, &bias, &config)
        .expect("grouped conv2d should run on GPU");
    let cpu = conv2d_cpu(&input, &weights, &bias, &config);
    assert!(
        gpu.iter()
            .zip(cpu.iter())
            .all(|(a, b)| (a - b).abs() < 1e-3),
        "conv2d mismatch"
    );
}

fn conv2d_cpu(input: &[f32], weights: &[f32], bias: &[f32], cfg: &Conv2dConfig) -> Vec<f32> {
    let mut output = vec![0.0; cfg.output_element_count()];
    let in_c = cfg.input_channels as usize;
    let out_c = cfg.output_channels as usize;
    let in_w = cfg.input_width as usize;
    let in_h = cfg.input_height as usize;
    let k_w = cfg.kernel_width as usize;
    let k_h = cfg.kernel_height as usize;
    let stride_x = cfg.stride_x as usize;
    let stride_y = cfg.stride_y as usize;
    let pad_x = cfg.pad_x as isize;
    let pad_y = cfg.pad_y as isize;
    let groups = cfg.groups as usize;
    let group_in = in_c / groups;
    let group_out = out_c / groups;
    let weights_per_out = group_in * k_w * k_h;
    let out_w = cfg.output_width as usize;
    let out_h = cfg.output_height as usize;

    for (oc, &bias_val) in bias.iter().enumerate().take(out_c) {
        let group_idx = oc / group_out;
        let in_start = group_idx * group_in;
        for oy in 0..out_h {
            for ox in 0..out_w {
                let mut acc = bias_val;
                for ic_local in 0..group_in {
                    let ic = in_start + ic_local;
                    for ky in 0..k_h {
                        for kx in 0..k_w {
                            let ix = ox * stride_x + kx;
                            let iy = oy * stride_y + ky;
                            let ix = ix as isize - pad_x;
                            let iy = iy as isize - pad_y;
                            if ix < 0 || iy < 0 || ix >= in_w as isize || iy >= in_h as isize {
                                continue;
                            }
                            let input_index = (ic * in_h + iy as usize) * in_w + ix as usize;
                            let weight_index =
                                oc * weights_per_out + ic_local * k_h * k_w + ky * k_w + kx;
                            acc = input[input_index].mul_add(weights[weight_index], acc);
                        }
                    }
                }
                let out_index = (oc * out_h + oy) * out_w + ox;
                output[out_index] = acc;
            }
        }
    }
    output
}

fn batch_norm_cpu(
    tensor: &[f32],
    gamma: &[f32],
    beta: &[f32],
    mean: &[f32],
    variance: &[f32],
    cfg: &BatchNormConfig,
) -> Vec<f32> {
    let mut out = tensor.to_vec();
    let width = cfg.width as usize;
    let height = cfg.height as usize;
    let channels = cfg.channels as usize;
    let plane = width * height;
    for c in 0..channels {
        let gamma_c = gamma[c];
        let beta_c = beta[c];
        let mean_c = mean[c];
        let var_c = variance[c];
        let inv_std = 1.0 / (var_c + cfg.epsilon).sqrt();
        for idx in 0..plane {
            let offset = c * plane + idx;
            let gain = inv_std * gamma_c;
            out[offset] = gain.mul_add(out[offset] - mean_c, beta_c);
        }
    }
    out
}

#[test]
fn gpu_tensor_chain_remains_on_device() {
    let Some(ops) = gpu_ops() else {
        eprintln!("Skipping GPU tensor chain test (no adapter available)");
        return;
    };

    let config = Conv2dConfig::new(
        1,
        Conv2dChannels::new(4, 4),
        SpatialDims::new(4, 4),
        SpatialDims::new(3, 3),
        SpatialDims::new(1, 1),
        SpatialDims::new(1, 1),
        Conv2dOptions::new(2, None),
    )
    .expect("chained conv2d test config should be valid");
    let input: Vec<f32> = (0..(4 * 4 * 4))
        .map(|i| ((i * 17 % 23) as f32) * 0.05)
        .collect();
    let weights_len = (config.output_channels as usize)
        * ((config.input_channels / config.groups) as usize)
        * config.kernel_width as usize
        * config.kernel_height as usize;
    let weights: Vec<f32> = (0..weights_len)
        .map(|i| ((i * 11 % 29) as f32) * 0.03)
        .collect();
    let bias: Vec<f32> = (0..config.output_channels)
        .map(|i| i as f32 * 0.02 - 0.1)
        .collect();

    let input_tensor = ops
        .upload_tensor(config.input_shape_dims(), &input, Some("chain_input"))
        .expect("chain input tensor should upload");
    let weight_tensor = ops
        .upload_tensor(config.weight_shape_dims(), &weights, Some("chain_weights"))
        .expect("chain weight tensor should upload");
    let bias_tensor = ops
        .upload_tensor(config.bias_shape_dims(), &bias, Some("chain_bias"))
        .expect("chain bias tensor should upload");

    let conv_gpu = ops
        .conv2d_tensor(&input_tensor, &weight_tensor, &bias_tensor, &config)
        .expect("chained conv2d should run on GPU");
    let relu_gpu = ops
        .activation_tensor(&conv_gpu, ActivationKind::Relu)
        .expect("chained ReLU should run on GPU");
    let gpu_output = relu_gpu
        .to_vec()
        .expect("chained GPU output should download");

    let cpu_conv = conv2d_cpu(&input, &weights, &bias, &config);
    let relu_cpu: Vec<f32> = cpu_conv.into_iter().map(|v| v.max(0.0)).collect();

    assert!(
        gpu_output
            .iter()
            .zip(relu_cpu.iter())
            .all(|(a, b)| (a - b).abs() < 1e-3),
        "GPU tensor chain diverged from CPU reference"
    );
}

#[test]
fn first_conv_relu_matches_onnx() {
    let Some(ops) = gpu_ops() else {
        eprintln!("Skipping ONNX block parity test (no adapter available)");
        return;
    };

    let Some(model_path) = model_file_path() else {
        eprintln!("Skipping ONNX block parity test (model file {MODEL_REL_PATH} missing)");
        return;
    };

    let loader = OnnxInitializerMap::load(
        &model_path,
        &[
            "420",
            "421",
            "backbone.model0.conv2.conv1.weight",
            "backbone.model0.conv2.conv1.bias",
            "423",
            "424",
        ],
    )
    .expect("load stage0 weights");

    let input_data = synthetic_input();
    let cpu_vec = reference_output(&model_path, "188", &input_data);

    let input_gpu = ops
        .upload_tensor([1usize, 3, 640, 640], &input_data, Some("stage0_input"))
        .expect("upload input");
    let relu_gpu = run_stage0_block(&ops, &loader, &input_gpu).expect("stage0 output");
    let gpu_vec = relu_gpu.to_vec().expect("download gpu relu");

    assert_eq!(gpu_vec.len(), cpu_vec.len(), "shape mismatch");
    let max_diff = gpu_vec
        .iter()
        .zip(cpu_vec.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-3,
        "first conv+relu mismatch (max diff {max_diff})"
    );
}

#[test]
fn conv_depthwise_block_matches_onnx() {
    let Some(ops) = gpu_ops() else {
        eprintln!("Skipping block parity test (no adapter available)");
        return;
    };

    let Some(model_path) = model_file_path() else {
        eprintln!("Skipping block parity test (model file {MODEL_REL_PATH} missing)");
        return;
    };

    let loader = OnnxInitializerMap::load(
        &model_path,
        &[
            "420",
            "421",
            "backbone.model0.conv2.conv1.weight",
            "backbone.model0.conv2.conv1.bias",
            "423",
            "424",
        ],
    )
    .expect("load block weights");

    let input_data = synthetic_input();
    let cpu_vec = reference_output(&model_path, "188", &input_data);

    let input_gpu = ops
        .upload_tensor([1usize, 3, 640, 640], &input_data, Some("stage0_input"))
        .expect("upload input");
    let relu = run_stage0_block(&ops, &loader, &input_gpu).expect("stage0 block");
    let gpu_vec = relu.to_vec().expect("download block output");

    assert_eq!(gpu_vec.len(), cpu_vec.len(), "shape mismatch");
    let max_diff = gpu_vec
        .iter()
        .zip(cpu_vec.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-3,
        "conv-depthwise block mismatch (max diff {max_diff})"
    );
}

#[test]
fn pooled_stage_conv_matches_onnx() {
    let Some(ops) = gpu_ops() else {
        eprintln!("Skipping pooled-stage test (no adapter)");
        return;
    };

    let Some(model_path) = model_file_path() else {
        eprintln!("Skipping pooled-stage test (model file {MODEL_REL_PATH} missing)");
        return;
    };

    let loader = load_backbone_weights(&model_path, 1, false, 0).expect("load stage1 weights");

    let input_data = synthetic_input();
    let cpu_vec = reference_output(&model_path, "Relu_8", &input_data);

    let input_gpu = ops
        .upload_tensor([1usize, 3, 640, 640], &input_data, Some("stage1_input"))
        .expect("upload input");
    let stage0 = run_stage0_block(&ops, &loader, &input_gpu).expect("stage0");
    let pooled = pool_tensor(&ops, &stage0).expect("max pool");
    let relu =
        run_stage_blocks(&ops, &loader, &pooled, &STAGE1_BLOCKS[..1]).expect("stage1 block0");
    let gpu_vec = relu.to_vec().expect("download stage1 output");

    assert_eq!(gpu_vec.len(), cpu_vec.len(), "shape mismatch");
    let max_diff = gpu_vec
        .iter()
        .zip(cpu_vec.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-3,
        "stage1 pooled block mismatch (max diff {max_diff})"
    );
}

test_backbone_stage!(
    pooled_stage_two_block_matches_onnx,
    1,
    "Relu_11",
    "stage1_input",
    false
);

test_backbone_stage!(
    stage2_blocks_match_onnx,
    2,
    "Relu_17",
    "stage2_input",
    false
);

test_backbone_stage!(
    stage2_pooled_matches_onnx,
    2,
    "MaxPool_18",
    "stage2_pool_input",
    true
);

test_backbone_stage!(
    stage3_blocks_match_onnx,
    3,
    "Relu_24",
    "stage3_input",
    false
);

test_backbone_stage!(
    stage3_pooled_matches_onnx,
    3,
    "MaxPool_25",
    "stage3_pool_input",
    true
);

test_backbone_stage!(
    stage4_blocks_match_onnx,
    4,
    "Relu_31",
    "stage4_input",
    false
);

test_backbone_stage!(
    stage4_pooled_matches_onnx,
    4,
    "MaxPool_32",
    "stage4_pool_input",
    true
);

test_backbone_stage!(
    stage5_blocks_match_onnx,
    5,
    "Relu_38",
    "stage5_input",
    false
);

#[test]
fn neck_and_detection_heads_match_onnx() {
    let Some(ops) = gpu_ops() else {
        eprintln!("Skipping neck/head test (no adapter)");
        return;
    };

    let Some(model_path) = model_file_path() else {
        eprintln!("Skipping neck/head test (model file {MODEL_REL_PATH} missing)");
        return;
    };

    let loader = load_backbone_weights(
        &model_path,
        BACKBONE_STAGES.len(),
        true,
        DETECTION_HEADS.len(),
    )
    .expect("load detection weights");

    let input_data = synthetic_input();
    let input_gpu = ops
        .upload_tensor([1usize, 3, 640, 640], &input_data, Some("neck_input"))
        .expect("upload input");
    let backbone_features = run_backbone_features(&ops, &loader, &input_gpu, BACKBONE_STAGES.len())
        .expect("backbone outputs");
    let levels =
        run_neck_and_heads(&ops, &loader, &backbone_features).expect("neck + head outputs");

    let feature_checks = [
        ("Relu_53", &levels[0].feature),
        ("Relu_47", &levels[1].feature),
        ("Relu_41", &levels[2].feature),
    ];
    for (node, tensor) in feature_checks {
        assert_tensor_matches(&model_path, node, tensor, &input_data);
    }

    let branch_checks = [
        ("Conv_55", &levels[0].cls),
        ("Conv_57", &levels[1].cls),
        ("Conv_59", &levels[2].cls),
        ("Conv_61", &levels[0].bbox),
        ("Conv_63", &levels[1].bbox),
        ("Conv_65", &levels[2].bbox),
        ("Conv_67", &levels[0].obj),
        ("Conv_69", &levels[1].obj),
        ("Conv_71", &levels[2].obj),
        ("Conv_73", &levels[0].kps),
        ("Conv_75", &levels[1].kps),
        ("Conv_77", &levels[2].kps),
    ];
    for (node, tensor) in branch_checks {
        assert_tensor_matches(&model_path, node, tensor, &input_data);
    }
}

#[test]
fn conv2d_vec4_matches_standard() {
    println!("Starting conv2d_vec4_matches_standard test");
    let Some(ops) = gpu_ops() else {
        eprintln!("Skipping conv2d_vec4 test (no adapter)");
        return;
    };

    let batch = 1;
    let input_channels = 16;
    let output_channels = 32;
    let width = 64;
    let height = 64;
    let kernel = 3;
    let stride = 1;
    let pad = 1;

    let input_len = (batch * input_channels * width * height) as usize;
    let input: Vec<f32> = (0..input_len).map(|i| (i % 100) as f32 / 100.0).collect();

    let weight_len = (output_channels * input_channels * kernel * kernel) as usize;
    let weights: Vec<f32> = (0..weight_len).map(|i| (i % 100) as f32 / 100.0).collect();

    let bias_len = output_channels as usize;
    let bias: Vec<f32> = (0..bias_len).map(|i| (i % 100) as f32 / 100.0).collect();

    let config = Conv2dConfig::new(
        batch,
        Conv2dChannels::new(input_channels, output_channels),
        SpatialDims::new(width, height),
        SpatialDims::new(kernel, kernel),
        SpatialDims::new(stride, stride),
        SpatialDims::new(pad, pad),
        Conv2dOptions::new(1, Some(ActivationKind::Relu)),
    )
    .expect("vec4 comparison conv2d config should be valid");

    let input_gpu = ops
        .upload_tensor(config.input_shape_dims(), &input, Some("input"))
        .expect("vec4 comparison input tensor should upload");
    let weight_gpu = ops
        .upload_tensor(config.weight_shape_dims(), &weights, Some("weights"))
        .expect("vec4 comparison weight tensor should upload");
    let bias_gpu = ops
        .upload_tensor(config.bias_shape_dims(), &bias, Some("bias"))
        .expect("vec4 comparison bias tensor should upload");

    let standard_out = ops
        .conv2d_tensor(&input_gpu, &weight_gpu, &bias_gpu, &config)
        .expect("standard conv2d should run on GPU");
    let vec4_out = ops
        .conv2d_vec4_tensor(&input_gpu, &weight_gpu, &bias_gpu, &config)
        .expect("vec4 conv2d should run on GPU");

    let standard_vec = standard_out
        .to_vec()
        .expect("standard conv2d output should download");
    let vec4_vec = vec4_out
        .to_vec()
        .expect("vec4 conv2d output should download");

    assert_eq!(standard_vec.len(), vec4_vec.len(), "Output length mismatch");

    let mut max_diff = 0.0f32;
    let mut mismatch_count = 0;
    for (i, (a, b)) in standard_vec.iter().zip(vec4_vec.iter()).enumerate() {
        let diff = (a - b).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > 1e-4 {
            if mismatch_count < 10 {
                eprintln!(
                    "Mismatch at index {}: standard={}, vec4={}, diff={}",
                    i, a, b, diff
                );
            }
            mismatch_count += 1;
        }
    }

    eprintln!("Total mismatches: {}", mismatch_count);
    eprintln!("Max diff between standard and vec4: {}", max_diff);
    assert!(
        max_diff < 1e-4,
        "Vectorized implementation output mismatch (max diff {})",
        max_diff
    );
}

#[test]
#[ignore] // Run with: cargo test -p fcs-core benchmark_conv2d_performance -- --ignored --nocapture
fn benchmark_conv2d_performance() {
    use std::time::Instant;

    let Some(ops) = gpu_ops() else {
        eprintln!("Skipping benchmark (no GPU adapter)");
        return;
    };

    let configs = [
        ("stage0_conv", 1, 3, 16, 640, 640, 3, 2, 1, 1),
        ("stage1_depth", 1, 32, 32, 160, 160, 3, 1, 1, 32),
        ("stage2_point", 1, 32, 64, 160, 160, 1, 1, 0, 1),
        ("head_depth", 1, 64, 64, 40, 40, 3, 1, 1, 64),
    ];

    println!("\n========== Conv2D Performance: Standard vs Vec4 ==========");

    for (name, batch, in_ch, out_ch, width, height, kernel, stride, pad, groups) in configs {
        let input_len = (batch * in_ch * width * height) as usize;
        let input: Vec<f32> = (0..input_len).map(|i| (i % 100) as f32 / 100.0).collect();

        let weight_len = if groups == 1 {
            (out_ch * in_ch * kernel * kernel) as usize
        } else {
            (out_ch * kernel * kernel) as usize
        };
        let weights: Vec<f32> = (0..weight_len).map(|i| (i % 100) as f32 / 100.0).collect();
        let bias: Vec<f32> = (0..out_ch).map(|i| (i % 100) as f32 / 100.0).collect();

        let config = Conv2dConfig::new(
            batch,
            Conv2dChannels::new(in_ch, out_ch),
            SpatialDims::new(width, height),
            SpatialDims::new(kernel, kernel),
            SpatialDims::new(stride, stride),
            SpatialDims::new(pad, pad),
            Conv2dOptions::new(groups, Some(ActivationKind::Relu)),
        )
        .expect("benchmark conv2d config should be valid");

        let input_gpu = ops
            .upload_tensor(config.input_shape_dims(), &input, None)
            .expect("benchmark input tensor should upload");
        let weight_gpu = ops
            .upload_tensor(config.weight_shape_dims(), &weights, None)
            .expect("benchmark weight tensor should upload");
        let bias_gpu = ops
            .upload_tensor(config.bias_shape_dims(), &bias, None)
            .expect("benchmark bias tensor should upload");

        // Warmup
        for _ in 0..5 {
            let _ = ops
                .conv2d_tensor(&input_gpu, &weight_gpu, &bias_gpu, &config)
                .expect("standard conv2d warmup should run on GPU");
        }

        // Benchmark standard
        let iterations = 50;
        let start = Instant::now();
        for _ in 0..iterations {
            let output = ops
                .conv2d_tensor(&input_gpu, &weight_gpu, &bias_gpu, &config)
                .expect("standard conv2d benchmark should run on GPU");
            let _ = output
                .to_vec()
                .expect("standard conv2d benchmark output should download");
        }
        let standard_avg = start.elapsed().as_micros() as f64 / iterations as f64;

        // Warmup vec4
        for _ in 0..5 {
            let _ = ops
                .conv2d_vec4_tensor(&input_gpu, &weight_gpu, &bias_gpu, &config)
                .expect("vec4 conv2d warmup should run on GPU");
        }

        // Benchmark vec4
        let start = Instant::now();
        for _ in 0..iterations {
            let output = ops
                .conv2d_vec4_tensor(&input_gpu, &weight_gpu, &bias_gpu, &config)
                .expect("vec4 conv2d benchmark should run on GPU");
            let _ = output
                .to_vec()
                .expect("vec4 conv2d benchmark output should download");
        }
        let vec4_avg = start.elapsed().as_micros() as f64 / iterations as f64;

        let speedup_pct = (standard_avg / vec4_avg - 1.0) * 100.0;

        println!(
            "{:<15} Standard: {:>7.1}μs | Vec4: {:>7.1}μs | Speedup: {:>+5.1}%",
            name, standard_avg, vec4_avg, speedup_pct
        );
    }

    println!("==========================================================\n");
}
