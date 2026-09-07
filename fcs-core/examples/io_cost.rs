//! How much of a folder job is reading the files?
//!
//! Experiment 68 says to skip prefetch work if I/O is already hidden. This measures the read
//! on its own, at the same concurrency the batch uses, so the answer is a number rather than
//! an assumption.
//!
//!   cargo run --release -p fcs-core --example io_cost -- <dir> [passes]

use rayon::prelude::*;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("usage: io_cost <dir> [passes]");
    let passes: usize = args.next().map_or(3, |v| v.parse().unwrap_or(3));

    let mut paths: Vec<_> = std::fs::read_dir(&dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file())
        .collect();
    paths.sort();
    println!("{} files in {dir}", paths.len());

    for pass in 1..=passes {
        // Serial first, then at batch concurrency, so the gap shows what parallelism buys.
        let start = Instant::now();
        let serial: u64 = paths
            .iter()
            .filter_map(|p| std::fs::read(p).ok())
            .map(|b| b.len() as u64)
            .sum();
        let serial_s = start.elapsed().as_secs_f64();

        let start = Instant::now();
        let parallel: u64 = paths
            .par_iter()
            .filter_map(|p| std::fs::read(p).ok())
            .map(|b| b.len() as u64)
            .sum();
        let parallel_s = start.elapsed().as_secs_f64();

        assert_eq!(serial, parallel);
        println!(
            "pass {pass}: {:.1} MB | serial {:.2} s ({:.0} MB/s) | {} threads {:.2} s ({:.0} MB/s)",
            serial as f64 / 1e6,
            serial_s,
            serial as f64 / 1e6 / serial_s,
            rayon::current_num_threads(),
            parallel_s,
            serial as f64 / 1e6 / parallel_s,
        );
    }
    Ok(())
}
