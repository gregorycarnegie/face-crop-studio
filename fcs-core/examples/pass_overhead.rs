//! Whether a per-pass GPU timestamp measures the dispatch or the pass around it.
//!
//! `gpu_pass_breakdown` sums 61 separately timed compute passes and reports ~0.538 ms of
//! "GPU compute". Two things in that number look wrong. A 20x20 64->64 pointwise dispatch
//! and a 160x160 one differ by 64x in arithmetic but only 2.2x in measured time, and the
//! summed total is larger than the entire `readback_wait` of the unprofiled path, which
//! contains the same forward pass plus the head copies.
//!
//! Experiment 8 asks the question directly: time N identical dispatches as N timestamped
//! passes, then as one timestamped pass containing all N, and compare both against the
//! wall time of submitting and waiting. If the merged figure is far below the sum, the
//! per-pass numbers are measuring the profiler, and the shader backlog is aimed at an
//! artefact.
//!
//! Run with: cargo run --release -p fcs-core --example pass_overhead

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use fcs_core::gpu::{
    GpuInferenceOps, GpuTensor,
    conv2d::{Conv2dChannels, Conv2dConfig, Conv2dOptions, SpatialDims},
};
use fcs_utils::gpu::{GpuAvailability, GpuContext, GpuContextOptions};
use wgpu::util::DeviceExt;

#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// How many dispatches to record, matching the 26 pointwise passes of one forward pass.
const DISPATCHES: usize = 26;
const WARMUP: usize = 10;
const RUNS: usize = 30;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

struct Case {
    label: &'static str,
    cfg: Conv2dConfig,
}

fn main() -> Result<()> {
    let options = GpuContextOptions {
        profiling: true,
        ..GpuContextOptions::default()
    };
    let context = match GpuContext::init_with_fallback(&options) {
        GpuAvailability::Available(ctx) => ctx,
        other => anyhow::bail!("GPU unavailable: {other:?}"),
    };
    anyhow::ensure!(context.profiler().is_some(), "GPU timestamps unavailable");
    println!("adapter: {}\n", context.adapter_info().name);

    let source = std::fs::read_to_string("fcs-core/src/gpu/conv2d.wgsl")
        .context("read fcs-core/src/gpu/conv2d.wgsl")?;
    let shader = context
        .device()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("conv2d"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let pipeline = context
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("conv2d"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let pointwise = |w, h, ic, oc| {
        Conv2dConfig::new(
            1,
            Conv2dChannels::new(ic, oc),
            SpatialDims::new(w, h),
            SpatialDims::new(1, 1),
            SpatialDims::new(1, 1),
            SpatialDims::new(0, 0),
            Conv2dOptions::new(1, None),
        )
    };
    let cases = [
        Case {
            label: "160x160 64->64 (the expensive pointwise layer)",
            cfg: pointwise(160, 160, 64, 64)?,
        },
        Case {
            label: "80x80 64->64 (1/4 the arithmetic)",
            cfg: pointwise(80, 80, 64, 64)?,
        },
        Case {
            label: "20x20 64->64 (1/64 the arithmetic)",
            cfg: pointwise(20, 20, 64, 64)?,
        },
        Case {
            label: "1x1 4->4 (one workgroup, no arithmetic to speak of)",
            cfg: pointwise(1, 1, 4, 4)?,
        },
    ];

    println!(
        "{DISPATCHES} dispatches per run, {RUNS} runs after {WARMUP} warm-ups, medians in us\n"
    );
    println!(
        "{:<46} {:>10} {:>10} {:>10} {:>9}",
        "case", "summed", "merged", "wall", "per-pass"
    );

    for case in &cases {
        let (summed, merged, wall) = measure(&context, &pipeline, &case.cfg)?;
        println!(
            "{:<46} {summed:>10.3} {merged:>10.3} {wall:>10.3} {:>9.3}",
            case.label,
            (summed - merged) / DISPATCHES as f64
        );
    }

    println!(
        "\n`summed` is what gpu_pass_breakdown reports: one timestamp pair per pass.\n\
         `merged` is one timestamp pair around all {DISPATCHES} dispatches in a single pass.\n\
         `wall` is submit-to-poll-complete on the host, which bounds both from above.\n\
         `per-pass` is (summed - merged) / {DISPATCHES}: the cost of the pass boundary itself."
    );
    Ok(())
}

fn measure(
    context: &Arc<GpuContext>,
    pipeline: &wgpu::ComputePipeline,
    cfg: &Conv2dConfig,
) -> Result<(f64, f64, f64)> {
    let ops = GpuInferenceOps::new(context.clone(), None)?;
    let data = |len: usize| {
        (0..len)
            .map(|i| ((i * 17 % 101) as f32 - 50.0) / 100.0)
            .collect::<Vec<_>>()
    };
    let input = ops.upload_tensor(
        cfg.input_shape_dims(),
        &data(cfg.input_shape_dims().iter().product()),
        None,
    )?;
    let weights = ops.upload_tensor(
        cfg.weight_shape_dims(),
        &data(cfg.weight_shape_dims().iter().product()),
        None,
    )?;
    let bias = ops.upload_tensor(
        cfg.bias_shape_dims(),
        &data(cfg.output_channels as usize),
        None,
    )?;
    let output = GpuTensor::uninitialized(context.clone(), cfg.output_shape_dims(), None)?;

    let uniforms = [
        cfg.input_width,
        cfg.input_height,
        cfg.input_channels,
        cfg.output_width,
        cfg.output_height,
        cfg.output_channels,
        cfg.kernel_width,
        cfg.kernel_height,
        cfg.stride_x,
        cfg.stride_y,
        cfg.pad_x,
        cfg.pad_y,
        cfg.groups,
        0,
    ];
    let uniform = context
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pass_overhead uniforms"),
            contents: bytemuck::cast_slice(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let buffers = [
        input.buffer(),
        weights.buffer(),
        bias.buffer(),
        output.buffer(),
        &uniform,
    ];
    let entries: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: b.as_entire_binding(),
        })
        .collect();
    let group = context
        .device()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pass_overhead bindings"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });

    // The production pointwise grid: four pixels and four output channels per thread.
    let groups = [
        cfg.output_width.div_ceil(32),
        cfg.output_height.div_ceil(8),
        cfg.output_channels.div_ceil(4),
    ];

    let mut summed = Vec::with_capacity(RUNS);
    let mut merged = Vec::with_capacity(RUNS);
    let mut wall = Vec::with_capacity(RUNS);

    for run in 0..WARMUP + RUNS {
        // One timestamped pass per dispatch, exactly as the profiled runtime records it.
        let mut encoder = context.device().create_command_encoder(&Default::default());
        for i in 0..DISPATCHES {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("separate"),
                timestamp_writes: context.timestamp_writes(if i == 0 { "sep" } else { "sep_n" }),
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
        }
        context.queue().submit(Some(encoder.finish()));
        let separate: f64 = context
            .take_pass_timings()?
            .iter()
            .map(|t| t.duration_ns / 1e3)
            .sum();

        // All of them inside one pass, with a single timestamp pair around the lot.
        let mut encoder = context.device().create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("merged"),
                timestamp_writes: context.timestamp_writes("merged"),
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            for _ in 0..DISPATCHES {
                pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
            }
        }
        let started = Instant::now();
        context.queue().submit(Some(encoder.finish()));
        context
            .device()
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| anyhow::anyhow!("poll failed: {e}"))?;
        let elapsed = started.elapsed().as_secs_f64() * 1e6;
        let one: f64 = context
            .take_pass_timings()?
            .iter()
            .map(|t| t.duration_ns / 1e3)
            .sum();

        if run >= WARMUP {
            summed.push(separate);
            merged.push(one);
            wall.push(elapsed);
        }
    }

    Ok((median(summed), median(merged), median(wall)))
}
