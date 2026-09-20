//! Tests for the WGSL compute ops.
//!
//! These check the op library itself: convolution against a CPU reference, activations, tensor
//! residency, buffer reuse. The node-by-node comparisons that used to live here ran YuNet's
//! graph against tract, which interpreted the ONNX file as an independent oracle; both are
//! gone with YuNet. The whole-network check that replaced them is
//! `tests/scrfd_parity.rs`, which runs SCRFD on these ops and compares all nine head outputs
//! against ONNX Runtime -- end to end rather than per node, and on the model that ships.

use super::*;
use fcs_utils::gpu::{GpuAvailability, GpuContext, GpuContextOptions};
use std::sync::Arc;

use crate::gpu::conv2d::{Conv2dChannels, Conv2dOptions, SpatialDims};

fn gpu_ops() -> Option<GpuInferenceOps> {
    test_context().map(|ctx| GpuInferenceOps::new(ctx, None).expect("build ops"))
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

/// Overwriting a tensor in place and reading it back returns exactly what was written. Six
/// distinct values, none of them 0, 1 or -1, so no constant stands in for the download.
#[test]
fn a_tensor_overwritten_in_place_downloads_what_was_written() {
    let Some(ops) = gpu_ops() else {
        return;
    };
    let tensor = ops
        .upload_tensor([1usize, 1, 2, 3], &[0.0; 6], Some("overwrite"))
        .expect("upload");
    let data = [2.5, -3.25, 4.0, 7.5, -8.0, 9.75];
    ops.upload_to_tensor(&tensor, &data).expect("overwrite");
    assert_eq!(ops.download_tensor(&tensor).expect("download"), data);
    assert!(format!("{tensor:?}").contains("dims"), "{tensor:?}");
}

/// A tensor from another device cannot be written through these ops, even though its own queue
/// would accept it. One deliberate second device, not the per-test pile-up.
#[test]
fn ops_reject_a_tensor_from_a_different_gpu_context() {
    let Some(ops) = gpu_ops() else {
        return;
    };
    let GpuAvailability::Available(other) =
        GpuContext::init_with_fallback(&GpuContextOptions::default())
    else {
        return;
    };
    let foreign = GpuTensor::from_slice(other, [1usize, 1, 1, 2], &[1.5, 2.5], Some("foreign"))
        .expect("foreign tensor");
    let err = ops
        .upload_to_tensor(&foreign, &[3.5, 4.5])
        .expect_err("a foreign tensor must be refused");
    assert!(format!("{err}").contains("different GPU context"), "{err}");
}

/// Grouped on purpose: two groups halve the weights per output, so a count that ignored groups
/// would differ.
#[test]
fn conv2d_config_validate_checks_every_buffer_length() {
    let config = Conv2dConfig::new(
        1,
        Conv2dChannels::new(4, 4),
        SpatialDims::new(4, 4),
        SpatialDims::new(3, 3),
        SpatialDims::new(1, 1),
        SpatialDims::new(1, 1),
        Conv2dOptions::new(2, None),
    )
    .expect("valid grouped config");
    // input 1*4*4*4 = 64, weights 4 * (4/2) * 3*3 = 72, bias 4.
    assert!(config.validate(64, 72, 4).is_ok());
    assert!(config.validate(63, 72, 4).is_err(), "input length");
    assert!(config.validate(64, 71, 4).is_err(), "weight length");
    assert!(config.validate(64, 72, 3).is_err(), "bias length");
}

#[test]
fn specialized_convolutions_match_cpu_for_tails_activations_and_fallbacks() {
    let Some(ops) = gpu_ops() else {
        eprintln!("Skipping specialized convolution test (no adapter)");
        return;
    };
    for (width, stride, pad, groups, kernel) in [
        (1, 1, 0, 1, 1),
        (3, 1, 0, 1, 1),
        (4, 1, 0, 1, 1),
        (5, 1, 0, 1, 1),
        (37, 1, 0, 1, 1),
        (17, 2, 0, 1, 1),
        (17, 1, 1, 1, 1),
        (17, 1, 0, 2, 1),
        (1, 1, 1, 6, 3),
        (3, 1, 1, 6, 3),
        (37, 1, 1, 6, 3),
        (17, 2, 1, 6, 3),
        (17, 1, 0, 6, 3),
    ] {
        for activation in [
            None,
            Some(ActivationKind::Relu),
            Some(ActivationKind::Sigmoid),
        ] {
            let output_channels = if groups == 6 { 6 } else { 10 };
            let cfg = Conv2dConfig::new(
                1,
                Conv2dChannels::new(6, output_channels),
                SpatialDims::new(width, 5),
                SpatialDims::new(kernel, kernel),
                SpatialDims::new(stride, stride),
                SpatialDims::new(pad, pad),
                Conv2dOptions::new(groups, activation),
            )
            .expect("convolution config");
            let data = |n| {
                (0..n)
                    .map(|i| ((i * 17 % 101) as f32 - 50.0) / 100.0)
                    .collect::<Vec<_>>()
            };
            let input = data(cfg.input_shape_dims().iter().product());
            let weights = data(cfg.weight_shape_dims().iter().product());
            let bias = data(output_channels as usize);
            let expected = conv2d_cpu(&input, &weights, &bias, &cfg);
            let actual = ops
                .conv2d(&input, &weights, &bias, &cfg)
                .expect("convolution dispatch");
            assert_eq!(actual.len(), expected.len());
            for (actual, raw) in actual.iter().zip(expected) {
                let expected = match activation {
                    None => raw,
                    Some(ActivationKind::Relu) => raw.max(0.0),
                    Some(ActivationKind::Sigmoid) => 1.0 / (1.0 + (-raw).exp()),
                };
                assert!(
                    (actual - expected).abs() < 1e-4,
                    "width={width} stride={stride} pad={pad} groups={groups} activation={activation:?}: {actual} vs {expected}"
                );
            }
        }
    }
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

/// Build a minimal ONNX model holding only float initializers, so the loader can be
/// exercised without the real 640x640 model on disk.

#[test]
fn tensor_shape_converts_into_its_dimensions() {
    let shape = TensorShape::new([2usize, 3, 4]).expect("shape");
    assert_eq!(shape.elements(), 24);
    assert_eq!(shape.dims(), &[2, 3, 4]);

    let dims: Vec<usize> = shape.into();
    assert_eq!(dims, vec![2, 3, 4]);
}

#[test]
fn div_ceil_uniform_rounds_up_and_keeps_zero_at_zero() {
    use super::utils::div_ceil_uniform;

    assert_eq!(
        div_ceil_uniform(0, 8),
        0,
        "an empty dispatch needs no groups"
    );
    assert_eq!(div_ceil_uniform(1, 8), 1);
    assert_eq!(div_ceil_uniform(9, 8), 2);
    assert_eq!(div_ceil_uniform(16, 8), 2);
}

#[test]
fn conv2d_reuses_one_uniform_buffer_per_distinct_config() {
    use crate::gpu::conv2d::Conv2dPipeline;

    let Some(ctx) = test_context() else {
        eprintln!("Skipping conv2d uniform cache test (no adapter)");
        return;
    };
    let pipeline = Conv2dPipeline::new(ctx.device(), 4).expect("build conv2d pipeline");

    let cfg = |width: u32| {
        Conv2dConfig::new(
            1,
            Conv2dChannels::new(16, 16),
            SpatialDims::new(width, 320),
            SpatialDims::new(3, 3),
            SpatialDims::new(1, 1),
            SpatialDims::new(1, 1),
            Conv2dOptions::new(1, None),
        )
        .expect("uniform cache test config should be valid")
    };

    // Two dispatches of the same layer must land on one buffer -- that reuse is the whole
    // point: creating 53 of these per forward pass cost 0.445 ms of CPU time.
    let first = pipeline.uniform_buffer_for_test(ctx.device(), &cfg(320));
    let again = pipeline.uniform_buffer_for_test(ctx.device(), &cfg(320));
    assert!(
        Arc::ptr_eq(&first, &again),
        "an identical config should hit the cache"
    );

    // A different shape must not: sharing a buffer across configs would feed one layer's
    // geometry to another, which the ONNX parity tests would catch only by luck.
    let other = pipeline.uniform_buffer_for_test(ctx.device(), &cfg(160));
    assert!(
        !Arc::ptr_eq(&first, &other),
        "a different config must get its own buffer"
    );
}

/// The bind-group cache has to keep paying for itself across repeated dispatches.
///
/// This replaces a test that ran YuNet ten times and read the hit rate off the whole graph.
/// The property it guarded is a property of the op, not of that graph: the same config over
/// tensors the buffer pool keeps handing back must hit rather than rebuild. If the pool stops
/// reusing buffers, or the cache key starts including something per-dispatch, the rate
/// collapses and the cache becomes overhead.
#[test]
fn conv2d_bind_groups_are_reused_across_dispatches() {
    let Some(ops) = gpu_ops() else {
        eprintln!("Skipping bind cache test (no adapter)");
        return;
    };
    let config = Conv2dConfig::new(
        1,
        Conv2dChannels::new(8, 8),
        SpatialDims::new(16, 16),
        SpatialDims::new(3, 3),
        SpatialDims::new(1, 1),
        SpatialDims::new(1, 1),
        Conv2dOptions::new(1, None),
    )
    .expect("config");
    let input: Vec<f32> = (0..8 * 16 * 16).map(|i| (i % 29) as f32 * 0.03).collect();
    let weights: Vec<f32> = (0..8 * 8 * 3 * 3).map(|i| (i % 13) as f32 * 0.02).collect();
    let bias: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();

    let run = || {
        ops.conv2d(&input, &weights, &bias, &config)
            .expect("conv2d runs");
    };
    // Warm first: the first dispatches are misses by definition, and it is the steady state
    // that matters.
    (0..5).for_each(|_| run());
    let (warm_hits, warm_misses) = ops.bind_cache_stats();
    (0..10).for_each(|_| run());
    let (hits, misses) = ops.bind_cache_stats();
    let (hits, misses) = (hits - warm_hits, misses - warm_misses);

    let total = hits + misses;
    assert!(total > 0, "no convolution dispatches were recorded");
    assert!(
        hits * 4 > total * 3,
        "bind group cache hit rate collapsed: {hits} hits, {misses} misses"
    );
}
