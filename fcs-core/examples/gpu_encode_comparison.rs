//! Compare separate and merged compute passes in one process, with profiling off.
//! cargo run --release -p fcs-core --example gpu_encode_comparison
use std::time::Instant;

use anyhow::{Context, Result};
use fcs_core::{
    gpu::{
        GpuInferenceOps, GpuTensor,
        graph::{self, DetectionLevelOutputs, GpuWeights},
        utils::ComputeDispatch,
    },
    yunet::{BACKBONE_STAGES, DETECTION_HEADS, load_backbone_weights},
};
use fcs_utils::gpu::{GpuAvailability, GpuContext, GpuContextOptions};

#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn forward(
    encoder: &mut impl ComputeDispatch,
    ops: &GpuInferenceOps,
    weights: &GpuWeights,
    input: &GpuTensor,
) -> Result<[DetectionLevelOutputs; 3]> {
    let features =
        graph::encode_backbone_features(encoder, ops, weights, input, BACKBONE_STAGES.len())?;
    graph::encode_neck_and_heads(encoder, ops, weights, features)
}

// Output readback is deliberately outside these timings. The last phase waits for
// GPU completion; total measures encode + finish + submit + wait, not detect_image.
fn run(
    ops: &GpuInferenceOps,
    weights: &GpuWeights,
    input: &GpuTensor,
    merged: bool,
) -> Result<([DetectionLevelOutputs; 3], [f64; 5])> {
    let _scope = ops.buffer_pool().execution_scope();
    let ctx = ops.context();
    let start = Instant::now();
    let mut encoder = ctx.device().create_command_encoder(&Default::default());
    let outputs = if merged {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        forward(&mut pass, ops, weights, input)?
    } else {
        forward(&mut encoder, ops, weights, input)?
    };
    let recorded = Instant::now();
    let commands = encoder.finish();
    let finished = Instant::now();
    let submission = ctx.queue().submit(Some(commands));
    let submitted = Instant::now();
    ctx.device().poll(wgpu::PollType::Wait {
        submission_index: Some(submission),
        timeout: None,
    })?;
    let done = Instant::now();
    Ok((
        outputs,
        [
            recorded.duration_since(start).as_secs_f64() * 1e3,
            finished.duration_since(recorded).as_secs_f64() * 1e3,
            submitted.duration_since(finished).as_secs_f64() * 1e3,
            done.duration_since(submitted).as_secs_f64() * 1e3,
            done.duration_since(start).as_secs_f64() * 1e3,
        ],
    ))
}

fn main() -> Result<()> {
    let context = match GpuContext::init_with_fallback(&GpuContextOptions::default()) {
        GpuAvailability::Available(ctx) => ctx,
        other => anyhow::bail!("GPU unavailable: {other:?}"),
    };
    println!("adapter: {:?}", context.adapter_info());
    let ops = GpuInferenceOps::new(context, None)?;
    let path = fcs_utils::model_path("models/face_detection_yunet_2023mar_640.onnx")?
        .context("YuNet model missing")?;
    let loader = load_backbone_weights(&path, BACKBONE_STAGES.len(), true, DETECTION_HEADS.len())?;
    let weights: GpuWeights = loader
        .into_map()
        .into_iter()
        .map(|(name, tensor)| Ok((name, ops.upload_tensor(tensor.dims(), tensor.data(), None)?)))
        .collect::<Result<_>>()?;
    let data: Vec<f32> = (0..3 * 640 * 640)
        .map(|i| (i % 257) as f32 / 256.0)
        .collect();
    let input = ops.upload_tensor([1, 3, 640, 640], &data, None)?;

    let (separate, _) = run(&ops, &weights, &input, false)?;
    let (merged, _) = run(&ops, &weights, &input, true)?;
    for (a, b) in separate.iter().zip(&merged) {
        // Dispatch order and arithmetic are unchanged, so every head should match exactly.
        anyhow::ensure!(
            a.heads.to_vec()? == b.heads.to_vec()?,
            "merged pass changed a detection head"
        );
    }
    drop((separate, merged));
    println!("all 12 raw detection heads match exactly");

    let mut samples: [Vec<[f64; 5]>; 2] = Default::default();
    for pair in 0..110 {
        // Alternate order as well as mode so clock ramp/drift does not favour one path.
        for mode in [pair % 2, 1 - pair % 2] {
            let (_, timing) = run(&ops, &weights, &input, mode == 1)?;
            if pair >= 10 {
                samples[mode].push(timing);
            }
        }
    }
    println!("100 paired runs; medians in ms (record includes ending the compute pass)");
    println!("phase             separate     merged");
    for (i, label) in ["record", "finish", "submit", "wait", "total"]
        .iter()
        .enumerate()
    {
        let medians = samples.each_ref().map(|s| {
            let mut values: Vec<f64> = s.iter().map(|v| v[i]).collect();
            values.sort_by(f64::total_cmp);
            values[values.len() / 2]
        });
        println!("{label:18} {:8.3}   {:8.3}", medians[0], medians[1]);
    }
    Ok(())
}
