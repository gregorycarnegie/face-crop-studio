//! Is `decode_yunet_outputs_with` short of arithmetic or short of memory bandwidth?
//!
//! It costs 0.077 ms, about 9% of a warm small-image detection, and it decodes all 8400
//! cells even though `apply_postprocess` then discards every one below the score threshold.
//! That invites an early-out: both sigmoids and the sqrt are monotonic, so a cell can be
//! rejected on the raw logit without a single transcendental.
//!
//! Whether that is worth anything depends on what the 0.077 ms is made of. The decode reads
//! 126000 floats through 14 strided gathers per cell and writes 126000 back, which is a
//! megabyte of traffic; if that dominates, skipping the arithmetic saves nothing because the
//! rejected rows still have to be written.
//!
//! Three variants over the same synthetic heads:
//!   full      the production decode
//!   no-math   the same gathers and writes, with sigmoid/exp/sqrt replaced by identity
//!   write     writes alone, no gathers at all
//!
//! Run with: cargo run --release -p fcs-core --example decode_cost

use std::time::Instant;

use anyhow::Result;
use fcs_core::model::{HeadLayout, decode_yunet_outputs_with};
use fcs_core::{InputSize, tensor::Tensor};

#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

const INPUT: InputSize = InputSize::new(640, 640);
const STRIDES: [usize; 3] = [8, 16, 32];
const COLS: usize = 15;
const WARMUP: usize = 20;
const RUNS: usize = 200;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn time<T>(runs: usize, mut f: impl FnMut() -> T) -> f64 {
    for _ in 0..WARMUP {
        std::hint::black_box(f());
    }
    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        let started = Instant::now();
        let out = f();
        samples.push(started.elapsed().as_secs_f64() * 1e3);
        drop(std::hint::black_box(out));
    }
    median(samples)
}

/// Logits spread either side of zero, so the early-out question is not answered by an input
/// where every cell happens to be far below threshold.
fn logits(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 37 + seed * 11) % 211) as f32 - 105.0) / 40.0)
        .collect()
}

fn main() -> Result<()> {
    let cells: Vec<usize> = STRIDES
        .iter()
        .map(|s| (INPUT.width as usize / s) * (INPUT.height as usize / s))
        .collect();
    let total: usize = cells.iter().sum();
    println!("cells per stride {cells:?}, total {total}, {COLS} columns\n");

    // Grouped the way `build_decode_tensors` hands them over: cls x3, obj x3, bbox x3, kps x3.
    let mut outputs: Vec<Tensor> = Vec::with_capacity(12);
    for (channels, group) in [(1usize, 0usize), (1, 1), (4, 2), (10, 3)] {
        for (index, &n) in cells.iter().enumerate() {
            outputs.push(Tensor::from_vec(
                &[channels, n],
                logits(channels * n, group * 3 + index),
            )?);
        }
    }

    let full = time(RUNS, || {
        decode_yunet_outputs_with(&outputs, INPUT, HeadLayout::ChannelMajorLogits, None)
            .expect("decode should succeed")
    });

    // The production path knows the score threshold, so it can skip the arithmetic for
    // cells that cannot reach it. This is the variant the detector runs.
    let gated = time(RUNS, || {
        decode_yunet_outputs_with(&outputs, INPUT, HeadLayout::ChannelMajorLogits, Some(0.6))
            .expect("decode should succeed")
    });

    // Same gathers, same writes, no transcendentals: what the decode would cost if every
    // sigmoid, exp and sqrt were free.
    let no_math = time(RUNS, || {
        let mut dst = vec![0f32; total * COLS];
        let mut write = 0usize;
        for (index, &n) in cells.iter().enumerate() {
            let cls = outputs[index].as_slice();
            let obj = outputs[index + 3].as_slice();
            let bbox = outputs[index + 6].as_slice();
            let kps = outputs[index + 9].as_slice();
            for cell in 0..n {
                let mut row = [0f32; COLS];
                row[0] = bbox[cell];
                row[1] = bbox[n + cell];
                row[2] = bbox[2 * n + cell];
                row[3] = bbox[3 * n + cell];
                for k in 0..10 {
                    row[4 + k] = kps[k * n + cell];
                }
                row[14] = cls[cell] + obj[cell];
                dst[write..write + COLS].copy_from_slice(&row);
                write += COLS;
            }
        }
        dst
    });

    // Writes alone: the floor nothing that still produces 8400x15 can go below.
    let write_only = time(RUNS, || {
        let mut dst = vec![0f32; total * COLS];
        let mut write = 0usize;
        let row = [1.0f32; COLS];
        for _ in 0..total {
            dst[write..write + COLS].copy_from_slice(&row);
            write += COLS;
        }
        dst
    });

    println!("{:<38} {:>10}", "variant", "ms");
    for (name, ms) in [
        ("full decode, no threshold", full),
        ("gated at 0.6, what the detector runs", gated),
        ("gathers + writes, no transcendentals", no_math),
        ("writes only", write_only),
    ] {
        println!("{name:<38} {ms:>10.4}");
    }

    println!(
        "\nAn early-out can only remove what is above `no-math`: {:.4} ms of {:.4}.\n\
         Everything at or below it survives, because a rejected cell still occupies a row.",
        full - no_math,
        full
    );
    Ok(())
}
