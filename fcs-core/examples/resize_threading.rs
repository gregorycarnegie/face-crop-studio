//! Does threading the source resize pay for itself? (experiment 48/49)
//!
//! Resizing the full-resolution source to 640x640 is the largest single cost in detecting
//! a large image, and `fast_image_resize` runs it on one core unless its `rayon` feature is
//! on. That feature is a build-time switch, and this machine's CPU throughput moved by 1.6x
//! between two builds of identical code, so comparing two binaries cannot decide it.
//!
//! `fast_image_resize` picks its thread count from `rayon::current_num_threads()`, so both
//! variants can run in one process: a one-thread pool is the single-threaded build, the
//! default pool is the threaded one. Blocks alternate so any drift lands on both.
//!
//! The resize is timed on its own rather than through `preprocess_dynamic_image`, because
//! the BGR/CHW conversion after it is already rayon-parallel and reacts to the same pool --
//! measuring them together cannot say which one the pool is helping.
//!
//! Without the `rayon` feature on `fast_image_resize` the crate ignores the pool, and every
//! row should read about 1.00x. That is the A/A control for this probe.
//!
//! Run with: cargo run --release -p fcs-core --example resize_threading [image...]

use std::time::Instant;

use anyhow::{Context, Result};
use fcs_utils::resize_image;
use image::imageops::FilterType;

#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

const BLOCKS: usize = 8;
const BLOCK_RUNS: usize = 10;
const OUT: u32 = 640;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() -> Result<()> {
    let paths: Vec<String> = {
        let args: Vec<String> = std::env::args().skip(1).collect();
        if args.is_empty() {
            vec![
                "fixtures/images/249_o.jpg".into(),
                "fixtures/images/271_n.jpg".into(),
                "fixtures/images/006.jpg".into(),
                "fixtures/images/189_g.jpg".into(),
            ]
        } else {
            args
        }
    };

    let single = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .context("build single-thread pool")?;
    println!(
        "default rayon pool: {} threads; comparison pool: 1 thread",
        rayon::current_num_threads()
    );
    println!("{BLOCKS} blocks per variant, {BLOCK_RUNS} runs each, alternating\n");
    println!(
        "{:<30} {:>10} {:>10} {:>10} {:>9}",
        "image (filter)", "1 thread", "all cores", "delta", "speedup"
    );

    for (filter, label) in [
        (FilterType::Triangle, "Quality"),
        (FilterType::Nearest, "Speed"),
    ] {
        for path in &paths {
            let opened = image::open(path).with_context(|| format!("open {path}"))?;
            // Downscaling the source first sweeps megapixels continuously, which is what
            // locating the crossover needs; the fixtures alone jump from 0.2 to 10 MP.
            let image = match std::env::var("FCS_SCALE_DIV")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
            {
                Some(div) if div > 1 => opened.resize_exact(
                    opened.width() / div,
                    opened.height() / div,
                    FilterType::Triangle,
                ),
                _ => opened,
            };
            let mp = f64::from(image.width()) * f64::from(image.height()) / 1e6;
            for _ in 0..5 {
                std::hint::black_box(resize_image(&image, OUT, OUT, filter));
            }

            let mut one = Vec::with_capacity(BLOCKS);
            let mut many = Vec::with_capacity(BLOCKS);
            for block in 0..BLOCKS * 2 {
                let mut samples = Vec::with_capacity(BLOCK_RUNS);
                for _ in 0..BLOCK_RUNS {
                    let started = Instant::now();
                    // The pool is what the resizer reads its thread count from, so running
                    // inside it is what selects the variant.
                    if block % 2 == 0 {
                        single.install(|| {
                            std::hint::black_box(resize_image(&image, OUT, OUT, filter))
                        });
                    } else {
                        std::hint::black_box(resize_image(&image, OUT, OUT, filter));
                    }
                    samples.push(started.elapsed().as_secs_f64() * 1e3);
                }
                if block % 2 == 0 {
                    one.push(median(samples));
                } else {
                    many.push(median(samples));
                }
            }
            let (a, b) = (median(one), median(many));
            let name = format!(
                "{} {:.1} MP ({label})",
                path.rsplit(['/', '\\']).next().unwrap_or(path),
                mp
            );
            println!(
                "{name:<30} {a:>7.3} ms {b:>7.3} ms {:>+7.3} ms {:>8.2}x",
                b - a,
                a / b
            );
        }
    }
    Ok(())
}
