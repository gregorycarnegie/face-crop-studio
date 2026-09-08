//! How long each WGSL file takes to become a compute pipeline.
//!
//! Experiment 80 put `conv2d.wgsl` at 197 ms of a ~900 ms cold start -- 90% of all shader
//! compilation, and the other four shaders together are 20 ms. So the question for
//! experiment 82 is not how to compile five shaders in parallel; it is what to do about one.
//! `PIPELINE_CACHE` would be the answer, and `examples/adapter_cost.rs` shows D3D12 does not
//! expose it.
//!
//! That leaves splitting. `conv2d.wgsl` carries four kernels behind one entry point and the
//! host picks between them from the uniforms, so FXC compiles all four every launch. If
//! compilation is superlinear in shader size, per-path pipelines would compile to less in
//! total; if it is linear, splitting buys nothing and the cost is irreducible.
//!
//! Each file is compiled in a fresh process-local module, timed separately, and the first
//! measurement of each is the one that counts -- there is no warm-up, because a cold start
//! has none.
//!
//!   cargo run --release -p fcs-core --example shader_compile_cost -- a.wgsl b.wgsl ...

use std::time::Instant;

use anyhow::{Context, Result};
use fcs_utils::gpu::{GpuAvailability, GpuContext, GpuContextOptions};

#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() -> Result<()> {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    anyhow::ensure!(
        !paths.is_empty(),
        "usage: shader_compile_cost <file.wgsl>..."
    );

    let context = match GpuContext::init_with_fallback(&GpuContextOptions::default()) {
        GpuAvailability::Available(ctx) => ctx,
        other => anyhow::bail!("GPU unavailable: {other:?}"),
    };
    let device = context.device();
    println!(
        "adapter: {} ({:?})\n",
        context.adapter_info().name,
        context.adapter_info().backend
    );

    println!("{:<34} {:>7} {:>11}", "shader", "lines", "compile ms");
    let mut total = 0.0;
    for spec in &paths {
        // `file.wgsl:entry` compiles one entry point of a module, which is how a multi-entry
        // module is priced: if naga and FXC only emit the reachable half, each entry should
        // cost what its own small file costs.
        let (path, entry) = match spec.rsplit_once(":main") {
            Some((file, rest)) => (file.to_string(), format!("main{rest}")),
            None => (spec.clone(), "main".to_string()),
        };
        let path = &path;
        let source = std::fs::read_to_string(path).with_context(|| format!("read {path}"))?;
        let lines = source.lines().count();
        let started = Instant::now();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(path),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(path),
            layout: None,
            module: &module,
            entry_point: Some(&entry),
            compilation_options: Default::default(),
            cache: None,
        });
        let ms = started.elapsed().as_secs_f64() * 1e3;
        total += ms;
        std::hint::black_box(pipeline);
        let name = format!(
            "{}:{entry}",
            std::path::Path::new(path)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.clone())
        );
        println!("{name:<34} {lines:>7} {ms:>11.1}");
    }
    println!("{:<34} {:>7} {total:>11.1}", "= total", "");
    Ok(())
}
