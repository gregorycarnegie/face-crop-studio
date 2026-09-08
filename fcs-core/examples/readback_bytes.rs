//! What the detection-head readback costs per byte (experiment 57).
//!
//! Compacting candidates on the GPU before downloading them is only worth building if the
//! bytes are what the readback is paying for. In production the head copies hide inside
//! `readback_wait`, which also contains the forward pass, so the phase table cannot answer
//! this. Here there is no inference in flight: the same allocate / copy / map / wait /
//! collect sequence `batch_download` runs is timed on buffers of decreasing size, so the
//! difference between the rows is the byte cost and nothing else.
//!
//! Production downloads three fused head buffers, one per level: 6400, 1600 and 400 cells
//! of 16 channels, 537.6 KB in total. A compacted download would be one buffer of
//! `capacity x 16` floats plus a counter, so the candidate rows below are single buffers.
//!
//! Run with: cargo run --release -p fcs-core --example readback_bytes

use std::sync::mpsc;

use anyhow::{Result, anyhow};
use bytemuck::cast_slice;
use fcs_core::gpu::GpuTensor;
use fcs_utils::gpu::{GpuAvailability, GpuContext, GpuContextOptions};

#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Cells per level in the 640x640 graph, and the fused channel count per cell.
const LEVEL_CELLS: [usize; 3] = [6400, 1600, 400];
const HEAD_CHANNELS: usize = 16;
const RUNS: usize = 200;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// One `batch_download`, phase for phase, returning milliseconds per phase.
fn download(context: &GpuContext, tensors: &[&GpuTensor]) -> Result<[f64; 5]> {
    let device = context.device();
    let mut t = std::time::Instant::now();
    let mut phase = [0.0f64; 5];
    let mut lap = |phase: &mut f64| {
        *phase = t.elapsed().as_secs_f64() * 1e3;
        t = std::time::Instant::now();
    };

    let bufs: Vec<wgpu::Buffer> = tensors
        .iter()
        .map(|tensor| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("probe_readback"),
                size: tensor.size_bytes(),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        })
        .collect();
    lap(&mut phase[0]);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("probe_readback_encoder"),
    });
    for (tensor, readback) in tensors.iter().zip(bufs.iter()) {
        encoder.copy_buffer_to_buffer(tensor.buffer(), 0, readback, 0, tensor.size_bytes());
    }
    context.queue().submit(Some(encoder.finish()));
    lap(&mut phase[1]);

    let receivers: Vec<_> = bufs
        .iter()
        .map(|buf| {
            let (tx, rx) = mpsc::channel();
            buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            rx
        })
        .collect();
    lap(&mut phase[2]);

    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .map_err(|e| anyhow!("probe poll failed: {e}"))?;
    lap(&mut phase[3]);

    let mut total = 0usize;
    for (buf, rx) in bufs.iter().zip(receivers.iter()) {
        rx.recv()??;
        let mapped = buf.slice(..).get_mapped_range()?;
        let floats: Vec<f32> = cast_slice(&mapped).to_vec();
        total += floats.len();
        drop(mapped);
        buf.unmap();
    }
    lap(&mut phase[4]);
    std::hint::black_box(total);
    Ok(phase)
}

fn main() -> Result<()> {
    let context = match GpuContext::init_with_fallback(&GpuContextOptions::default()) {
        GpuAvailability::Available(ctx) => ctx,
        GpuAvailability::Disabled { reason } => anyhow::bail!("GPU disabled: {reason}"),
        GpuAvailability::Unavailable { error } => anyhow::bail!("GPU unavailable: {error}"),
    };
    println!("adapter: {}\n", context.adapter_info().name);

    // Production: three buffers. Candidates: one compacted buffer of that many rows.
    let mut cases: Vec<(String, Vec<GpuTensor>)> = vec![(
        "production (3 heads, 8400 cells)".to_string(),
        LEVEL_CELLS
            .iter()
            .map(|cells| {
                GpuTensor::uninitialized(context.clone(), vec![*cells, HEAD_CHANNELS], None)
            })
            .collect::<Result<_>>()?,
    )];
    for rows in [2048usize, 512, 128, 16] {
        cases.push((
            format!("compacted to {rows} cells"),
            vec![GpuTensor::uninitialized(
                context.clone(),
                vec![rows, HEAD_CHANNELS],
                None,
            )?],
        ));
    }

    println!(
        "{:<34} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "case", "KB", "alloc", "copy", "map", "wait", "collect", "total"
    );
    for (label, tensors) in &cases {
        let refs: Vec<&GpuTensor> = tensors.iter().collect();
        let kb = tensors.iter().map(|t| t.size_bytes()).sum::<u64>() as f64 / 1024.0;
        let mut samples: Vec<[f64; 5]> = Vec::with_capacity(RUNS);
        for run in 0..RUNS + 10 {
            let phase = download(&context, &refs)?;
            if run >= 10 {
                samples.push(phase);
            }
        }
        let p: Vec<f64> = (0..5)
            .map(|i| median(samples.iter().map(|s| s[i]).collect()))
            .collect();
        println!(
            "{label:<34} {kb:>8.1} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
            p[0],
            p[1],
            p[2],
            p[3],
            p[4],
            p.iter().sum::<f64>()
        );
    }
    Ok(())
}
