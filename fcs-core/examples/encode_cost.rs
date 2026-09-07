//! What the per-dispatch host objects actually cost on this device.
//!
//! `phase_timings` puts `gpu_record` at ~0.19 ms for a 640x640 forward pass, and that pass
//! records 61 dispatches: 53 convolutions (uniforms already cached, bind group per dispatch)
//! plus 4 max-pools, 2 resizes and 2 adds (both created per dispatch). Experiments 20 and 22
//! propose caching the two remaining per-dispatch objects, so the first question is how much
//! either one is worth before writing a cache for it.
//!
//! Measures the wgpu calls in isolation, on a real device, against the cache lookup that
//! would replace them. Nothing here is production code; it exists to size the candidates.
//!
//! Run with: cargo run --release -p fcs-core --example encode_cost

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::Result;
use bytemuck::{Pod, Zeroable, bytes_of};
use fcs_utils::gpu::{GpuAvailability, GpuContext, GpuContextOptions};
use wgpu::util::DeviceExt;

#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Same 16 bytes and same derives as the real small-op uniform structs.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Pod, Zeroable)]
struct Uniforms {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
}

const ITERS: usize = 20_000;

/// Median of `iters` repetitions of `f`, in microseconds.
fn time_each<T>(iters: usize, mut f: impl FnMut(usize) -> T) -> f64 {
    let mut samples = Vec::with_capacity(iters);
    for i in 0..iters {
        let started = Instant::now();
        let value = f(i);
        samples.push(started.elapsed().as_secs_f64() * 1e6);
        drop(value);
    }
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn entries<'a>(
    input: &'a wgpu::Buffer,
    output: &'a wgpu::Buffer,
    uniform: &'a wgpu::Buffer,
) -> [wgpu::BindGroupEntry<'a>; 3] {
    [
        wgpu::BindGroupEntry {
            binding: 0,
            resource: input.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: output.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 2,
            resource: uniform.as_entire_binding(),
        },
    ]
}

fn main() -> Result<()> {
    let context = match GpuContext::init_with_fallback(&GpuContextOptions::default()) {
        GpuAvailability::Available(ctx) => ctx,
        GpuAvailability::Disabled { reason } => anyhow::bail!("GPU disabled: {reason}"),
        GpuAvailability::Unavailable { error } => anyhow::bail!("GPU unavailable: {error}"),
    };
    println!(
        "adapter: {} ({:?})\n",
        context.adapter_info().name,
        context.adapter_info().backend
    );
    let device = context.device();

    let storage_entry = |binding, read_only| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("encode_cost_layout"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, false),
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let storage = |label| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: 4096,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    };
    let input = storage("input");
    let output = storage("output");

    let make_uniform = |u: &Uniforms, label: &'static str| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytes_of(u),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    };
    let key = Uniforms {
        a: 1,
        b: 2,
        c: 3,
        d: 4,
    };
    let uniform = make_uniform(&key, "uniform");

    // Warm the driver allocators before the first timed call.
    for i in 0..500u32 {
        drop(make_uniform(
            &Uniforms {
                a: i,
                b: 0,
                c: 0,
                d: 0,
            },
            "warm",
        ));
    }

    let create_uniform = time_each(ITERS, |i| {
        make_uniform(
            &Uniforms {
                a: i as u32,
                b: 2,
                c: 3,
                d: 4,
            },
            "yunet_add_uniforms",
        )
    });

    let cache: Mutex<HashMap<Uniforms, Arc<wgpu::Buffer>>> = Mutex::new(HashMap::new());
    cache
        .lock()
        .unwrap()
        .insert(key, Arc::new(make_uniform(&key, "cached")));
    let cached_uniform = time_each(ITERS, |_| {
        let mut guard = cache.lock().unwrap();
        guard
            .entry(key)
            .or_insert_with(|| unreachable!("seeded above"))
            .clone()
    });

    let create_bind_group = time_each(ITERS, |_| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("conv2d_bg"),
            layout: &layout,
            entries: &entries(&input, &output, &uniform),
        })
    });

    let held = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("held"),
        layout: &layout,
        entries: &entries(&input, &output, &uniform),
    });
    let bg_cache: Mutex<HashMap<u64, Arc<wgpu::BindGroup>>> = Mutex::new(HashMap::new());
    bg_cache.lock().unwrap().insert(7, Arc::new(held));
    let cached_bind_group = time_each(ITERS, |_| {
        bg_cache.lock().unwrap().get(&7).expect("seeded").clone()
    });

    println!("median per call, {ITERS} iterations");
    println!("{:<34} {:>10}", "operation", "us");
    for (name, us) in [
        ("create_buffer_init (uniform)", create_uniform),
        ("cached uniform lookup", cached_uniform),
        ("create_bind_group", create_bind_group),
        ("cached bind group lookup", cached_bind_group),
    ] {
        println!("{name:<34} {us:>10.4}");
    }

    println!(
        "\nper forward pass (61 dispatches: 53 conv + 4 pool + 2 resize + 2 add)\n\
         20. uniforms for the 8 small ops: {:.4} ms -> {:.4} ms\n\
         22. bind groups for all 61:       {:.4} ms -> {:.4} ms\n\
         against a measured gpu_record of ~0.19 ms",
        create_uniform * 8.0 / 1000.0,
        cached_uniform * 8.0 / 1000.0,
        create_bind_group * 61.0 / 1000.0,
        cached_bind_group * 61.0 / 1000.0,
    );
    Ok(())
}
