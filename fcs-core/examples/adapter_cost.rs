//! How much of a cold start is spent looking for a GPU we already know we will pick.
//!
//! Experiment 80 puts adapter and device creation at roughly three quarters of the time from
//! process launch to the first face -- 620-910 ms, more than everything else together. The
//! default backend set is `Backends::PRIMARY`, so on Windows wgpu loads and enumerates
//! Vulkan as well as D3D12 before returning the adapter it was always going to return.
//!
//! Loaders are cached inside a process, so a second `Instance` in the same run measures
//! nothing. One process therefore measures one configuration and exits; the caller runs it
//! once per configuration.
//!
//!   cargo run --release -p fcs-core --example adapter_cost -- [primary|dx12|vulkan]

use std::time::Instant;

use anyhow::Result;
use fcs_utils::gpu::{GpuAvailability, GpuContext, GpuContextOptions};

#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() -> Result<()> {
    let which = std::env::args().nth(1).unwrap_or_else(|| "primary".into());
    let backends = match which.as_str() {
        "primary" => wgpu::Backends::PRIMARY,
        "dx12" => wgpu::Backends::DX12,
        "vulkan" => wgpu::Backends::VULKAN,
        other => anyhow::bail!("unknown backend set {other}"),
    };

    // `respect_env` off: an environment override would silently answer a different question
    // than the one being asked.
    let options = GpuContextOptions {
        backends,
        respect_env: false,
        // Asked for opportunistically so the report below is about what the adapter
        // supports, not about what the default options happened to enable.
        optional_features: wgpu::Features::PIPELINE_CACHE,
        ..GpuContextOptions::default()
    };

    let started = Instant::now();
    let context = match GpuContext::init_with_fallback(&options) {
        GpuAvailability::Available(ctx) => ctx,
        other => {
            println!("{which:<8} unavailable: {other:?}");
            return Ok(());
        }
    };
    let ms = started.elapsed().as_secs_f64() * 1e3;
    // Experiment 82 wants to know whether compiled pipelines can be persisted. wgpu exposes
    // that as an adapter feature, so the answer is a capability question before it is a
    // design question.
    println!(
        "{which:<8} PIPELINE_CACHE supported: {}",
        context.features().contains(wgpu::Features::PIPELINE_CACHE)
    );
    println!(
        "{which:<8} {ms:>8.1} ms   {} ({:?})",
        context.adapter_info().name,
        context.adapter_info().backend
    );
    Ok(())
}
