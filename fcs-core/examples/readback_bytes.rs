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
//! `--reads` answers the other half of experiment 17 instead: the decode currently walks
//! a `Vec` that was copied out of the mapped range, and the copy is only worth keeping if
//! reading the mapped range directly is slower than copying it and reading that. Both
//! access patterns are timed over the same production buffers.
//!
//! Run with: cargo run --release -p fcs-core --example readback_bytes [--reads]

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

/// Sum the head buffers the way the decode walks them: the two score channels for every
/// cell, then the remaining fourteen for the cells a score gate would keep (experiment 93
/// leaves roughly one cell in six). Channel-major, so each channel is contiguous.
fn decode_shaped_sum(flat: &[f32], rows: usize) -> f32 {
    let mut acc = 0.0f32;
    for v in &flat[0..2 * rows] {
        acc += *v;
    }
    for cell in (0..rows).step_by(6) {
        for ch in 2..HEAD_CHANNELS {
            acc += flat[ch * rows + cell];
        }
    }
    acc
}

/// `--reads`: copy-then-read against read-in-place, over the mapped production buffers.
///
/// Returns milliseconds for (copy, read the copy, read the mapping in place).
fn read_costs(context: &GpuContext, tensors: &[&GpuTensor]) -> Result<[f64; 3]> {
    let device = context.device();
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
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("probe_readback_encoder"),
    });
    for (tensor, readback) in tensors.iter().zip(bufs.iter()) {
        encoder.copy_buffer_to_buffer(tensor.buffer(), 0, readback, 0, tensor.size_bytes());
    }
    context.queue().submit(Some(encoder.finish()));
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
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .map_err(|e| anyhow!("probe poll failed: {e}"))?;

    let mut out = [0.0f64; 3];
    for ((buf, rx), tensor) in bufs.iter().zip(receivers.iter()).zip(tensors.iter()) {
        rx.recv()??;
        let rows = tensor.shape().dims()[0];
        let mapped = buf.slice(..).get_mapped_range()?;
        let flat: &[f32] = cast_slice(&mapped);

        // In place first, so the copy cannot be the thing that warms the cache for it.
        let t = std::time::Instant::now();
        std::hint::black_box(decode_shaped_sum(flat, rows));
        out[2] += t.elapsed().as_secs_f64() * 1e3;

        let t = std::time::Instant::now();
        let owned: Vec<f32> = flat.to_vec();
        out[0] += t.elapsed().as_secs_f64() * 1e3;

        let t = std::time::Instant::now();
        std::hint::black_box(decode_shaped_sum(&owned, rows));
        out[1] += t.elapsed().as_secs_f64() * 1e3;

        drop(mapped);
        buf.unmap();
    }
    Ok(out)
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

    if std::env::args().any(|a| a == "--reads") {
        let refs: Vec<&GpuTensor> = cases[0].1.iter().collect();
        let mut samples: Vec<[f64; 3]> = Vec::with_capacity(RUNS);
        for run in 0..RUNS + 10 {
            let costs = read_costs(&context, &refs)?;
            if run >= 10 {
                samples.push(costs);
            }
        }
        let p: Vec<f64> = (0..3)
            .map(|i| median(samples.iter().map(|s| s[i]).collect()))
            .collect();
        println!(
            "production heads, 525 KB, {RUNS} runs, medians in ms
"
        );
        println!("  copy out of the mapping      {:>8.3}", p[0]);
        println!("  decode-shaped read of a Vec  {:>8.3}", p[1]);
        println!("  same read, in the mapping    {:>8.3}", p[2]);
        println!(
            "
  copy + read {:>8.3}   read in place {:>8.3}   in-place saves {:+.3}",
            p[0] + p[1],
            p[2],
            (p[0] + p[1]) - p[2]
        );
        return Ok(());
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
