//! GPU timestamp A/B comparison of two WGSL files with the existing conv2d bindings.
//! cargo run --release -p fcs-core --example conv2d_experiment -- baseline.wgsl candidate.wgsl
//! Uploads, compilation, validation readback and timestamp resolution are outside timing.
//! See experimentation.md for tile coverage arguments and optional-feature probes.
use std::{path::Path, sync::Arc};

use anyhow::{Context, Result};
use fcs_core::gpu::{
    GpuInferenceOps, GpuTensor,
    conv2d::{Conv2dChannels, Conv2dConfig, Conv2dOptions, SpatialDims},
};
use fcs_utils::gpu::{GpuAvailability, GpuContext, GpuContextOptions};
use wgpu::util::DeviceExt;

#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn pipeline(context: &GpuContext, path: &Path) -> Result<wgpu::ComputePipeline> {
    let source =
        std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let shader = context
        .device()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: path.to_str(),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    Ok(context
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: path.to_str(),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        }))
}

fn compare(
    context: &Arc<GpuContext>,
    pipelines: &[wgpu::ComputePipeline; 2],
    cfg: &Conv2dConfig,
    coverage: [[u32; 3]; 2],
    half_storage: bool,
) -> Result<bool> {
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
    let outputs: Vec<GpuTensor> = (0..2)
        .map(|_| GpuTensor::uninitialized(context.clone(), cfg.output_shape_dims(), None))
        .collect::<Result<_>>()?;
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
            label: Some("experiment uniforms"),
            contents: bytemuck::cast_slice(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    // `FCS_CONV_PACK_WEIGHTS` hands side B pointwise weights prepacked as one vec4 per (output
    // tile, input channel), zero-padded past the last channel: the layout a packed kernel would
    // be given once at model load (experiment 30).
    let packed_weights = (std::env::var_os("FCS_CONV_PACK_WEIGHTS").is_some()).then(|| {
        let (ic, oc) = (cfg.input_channels as usize, cfg.output_channels as usize);
        let flat = data(cfg.weight_shape_dims().iter().product());
        let mut packed = vec![0.0f32; oc.div_ceil(4) * ic * 4];
        for o in 0..oc {
            for i in 0..ic {
                packed[(o / 4) * ic * 4 + i * 4 + o % 4] = flat[o * ic + i];
            }
        }
        context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("packed weights"),
                contents: bytemuck::cast_slice(&packed),
                usage: wgpu::BufferUsages::STORAGE,
            })
    });
    let packed: Vec<_> = if half_storage {
        [&input, &weights, &bias]
            .into_iter()
            .map(|t| pack_half(context, t))
            .collect()
    } else {
        Vec::new()
    };
    let groups: Vec<_> = pipelines
        .iter()
        .zip(&outputs)
        .enumerate()
        .map(|(mode, (pipeline, output))| {
            let buffers = [
                if mode == 1 && half_storage {
                    &packed[0]
                } else {
                    input.buffer()
                },
                if mode == 1 && half_storage {
                    &packed[1]
                } else if let (1, Some(buffer)) = (mode, packed_weights.as_ref()) {
                    buffer
                } else {
                    weights.buffer()
                },
                if mode == 1 && half_storage {
                    &packed[2]
                } else {
                    bias.buffer()
                },
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
            context
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("experiment bindings"),
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &entries,
                })
        })
        .collect();

    let mut samples: [Vec<f64>; 2] = Default::default();
    let mut max_error = 0.0f32;
    for pair in 0..70 {
        let order = [pair % 2, 1 - pair % 2];
        let mut encoder = context.device().create_command_encoder(&Default::default());
        for mode in order {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("conv experiment"),
                timestamp_writes: context.timestamp_writes(if mode == 0 { "A" } else { "B" }),
            });
            pass.set_pipeline(&pipelines[mode]);
            pass.set_bind_group(0, &groups[mode], &[]);
            let [x, y, z] = coverage[mode];
            pass.dispatch_workgroups(
                cfg.output_width.div_ceil(x),
                cfg.output_height.div_ceil(y),
                cfg.output_channels.div_ceil(z),
            );
        }
        context.queue().submit(Some(encoder.finish()));
        let timings = context.take_pass_timings()?;
        anyhow::ensure!(timings.len() == 2, "expected two timestamp pairs");
        if pair == 0 {
            let a = outputs[0].to_vec()?;
            let b = outputs[1].to_vec()?;
            for (index, (a, b)) in a.iter().zip(&b).enumerate() {
                max_error = max_error.max((a - b).abs());
                anyhow::ensure!(
                    a.is_finite()
                        && b.is_finite()
                        && (half_storage || (a - b).abs() <= 1e-4 + a.abs() * 1e-4),
                    "output mismatch at {index}: {a} vs {b}"
                );
            }
        }
        if pair >= 20 {
            for (mode, timing) in order.into_iter().zip(timings) {
                samples[mode].push(timing.duration_ns / 1e3);
            }
        }
    }
    let medians = samples.map(|mut values| {
        values.sort_by(f64::total_cmp);
        values[values.len() / 2]
    });
    println!(
        "{:3}x{:<3} {:3}->{:<3} k{} g{:<3} {:9.3} {:9.3} {:7.1}%",
        cfg.input_width,
        cfg.input_height,
        cfg.input_channels,
        cfg.output_channels,
        cfg.kernel_width,
        cfg.groups,
        medians[0],
        medians[1],
        100.0 * (medians[1] / medians[0] - 1.0)
    );
    if half_storage {
        println!(
            "  max absolute error: {max_error:.6} (raw 1e-3 budget: {})",
            max_error <= 1e-3
        );
    }
    Ok(!half_storage || max_error <= 1e-3)
}

// One-time f32 -> f16 upload conversion, deliberately outside shader timings.
fn pack_half(context: &GpuContext, input: &GpuTensor) -> wgpu::Buffer {
    let device = context.device();
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("half storage"),
        size: (input.size_bytes() / 2).next_multiple_of(4),
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("pack f16"),
        source: wgpu::ShaderSource::Wgsl(
            r#"
enable f16;
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f16>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < arrayLength(&dst) && id.x < arrayLength(&src) {
        dst[id.x] = f16(src[id.x]);
    }
}
"#
            .into(),
        ),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pack f16"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((input.shape().elements() as u32).div_ceil(64), 1, 1);
    }
    context.queue().submit(Some(encoder.finish()));
    output
}

fn main() -> Result<()> {
    let mut args: Vec<_> = std::env::args_os().skip(1).collect();
    let half_storage = args.last().is_some_and(|a| a == "--f16-storage");
    if half_storage {
        args.pop();
    }
    anyhow::ensure!(
        args.len() == 2 || args.len() == 5 || args.len() == 8,
        "usage: conv2d_experiment baseline.wgsl candidate.wgsl [B-x B-y B-z | A-x A-y A-z B-x B-y B-z] [--f16-storage]"
    );
    // Optional output coverage selects pointwise-only cases for kernels with a different grid.
    let parse_tile = |start: usize| -> Result<[u32; 3]> {
        Ok([
            args[start].to_string_lossy().parse()?,
            args[start + 1].to_string_lossy().parse()?,
            args[start + 2].to_string_lossy().parse()?,
        ])
    };
    let coverage = match args.len() {
        5 => [[32, 8, 1], parse_tile(2)?],
        8 => [parse_tile(2)?, parse_tile(5)?],
        _ => [[32, 8, 1]; 2],
    };
    anyhow::ensure!(
        coverage.iter().flatten().all(|n| *n > 0),
        "tile dimensions must be positive"
    );
    let options = GpuContextOptions {
        profiling: true,
        optional_features: wgpu::Features::SHADER_F16 | wgpu::Features::SUBGROUP,
        ..Default::default()
    };
    let context = match GpuContext::init_with_fallback(&options) {
        GpuAvailability::Available(ctx) => ctx,
        other => anyhow::bail!("GPU unavailable: {other:?}"),
    };
    anyhow::ensure!(context.profiler().is_some(), "GPU timestamps unavailable");
    println!("{:?}", context.adapter_info());
    println!("optional shader features enabled: {:?}", context.features());
    anyhow::ensure!(
        !half_storage || context.features().contains(wgpu::Features::SHADER_F16),
        "FP16 not supported by this context"
    );
    let pipelines = [
        pipeline(&context, Path::new(&args[0]))?,
        pipeline(&context, Path::new(&args[1]))?,
    ];
    println!(
        "50 alternating pairs after 20 warm-ups; GPU microseconds; f16 storage: {half_storage}"
    );
    let mut all_passed = true;
    println!("shape       channels kernel/group         A         B    change");
    for (width, height, ic, oc, kernel, stride, pad, groups) in [
        (320, 320, 16, 16, 1, 1, 0, 1),
        (160, 160, 16, 64, 1, 1, 0, 1),
        (160, 160, 64, 64, 1, 1, 0, 1),
        (80, 80, 64, 64, 1, 1, 0, 1),
        (40, 40, 64, 64, 1, 1, 0, 1),
        (20, 20, 64, 64, 1, 1, 0, 1),
        (80, 80, 64, 1, 1, 1, 0, 1),
        (40, 40, 64, 10, 1, 1, 0, 1),
        (17, 5, 3, 7, 1, 1, 0, 1),
        (1, 1, 3, 1, 1, 1, 0, 1),
        (320, 320, 16, 16, 3, 1, 1, 16),
        (160, 160, 64, 64, 3, 1, 1, 64),
        (80, 80, 64, 64, 3, 1, 1, 64),
        (40, 40, 64, 64, 3, 1, 1, 64),
        (20, 20, 64, 64, 3, 1, 1, 64),
        (17, 5, 3, 3, 3, 1, 1, 3),
        (1, 1, 3, 3, 3, 1, 1, 3),
        (640, 640, 3, 16, 3, 2, 1, 1),
        (17, 5, 4, 8, 1, 2, 1, 2),
    ] {
        // `FCS_CONV_ONLY_DEPTHWISE` keeps the depthwise cases, for single-kernel candidate files
        // whose `main` would compute nonsense on the others (experiments 32 and 33). Otherwise
        // explicit coverage selects the pointwise cases.
        let depthwise = kernel == 3 && stride == 1 && pad == 1 && groups == ic && ic == oc;
        if std::env::var_os("FCS_CONV_ONLY_DEPTHWISE").is_some() {
            if !depthwise {
                continue;
            }
        } else if args.len() > 2 && (kernel != 1 || stride != 1 || pad != 0 || groups != 1) {
            continue;
        }
        // `FCS_CONV_CASE=WxHxICxOC` keeps one shape, for kernels specialised to it (experiment 28).
        if std::env::var("FCS_CONV_CASE")
            .is_ok_and(|case| case != format!("{width}x{height}x{ic}x{oc}"))
        {
            continue;
        }
        let cfg = Conv2dConfig::new(
            1,
            Conv2dChannels::new(ic, oc),
            SpatialDims::new(width, height),
            SpatialDims::new(kernel, kernel),
            SpatialDims::new(stride, stride),
            SpatialDims::new(pad, pad),
            Conv2dOptions::new(groups, None),
        )?;
        all_passed &= compare(&context, &pipelines, &cfg, coverage, half_storage)?;
    }
    anyhow::ensure!(
        all_passed,
        "FP16 raw-output error exceeded 1e-3; not accepted"
    );
    Ok(())
}
