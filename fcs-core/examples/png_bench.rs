//! Compare PNG encoder settings on real exported crops.
//!
//! PNG is lossless, so neither the compression level nor the row filter changes a single
//! decoded pixel: the only things that move are encode time and file size. That makes this a
//! straight time-against-bytes trade rather than a quality question (experiment 72).
//!
//!   cargo run --release -p fcs-core --example png_bench -- <dir-of-pngs> [limit]

use image::{
    ExtendedColorType, ImageEncoder,
    codecs::png::{CompressionType, FilterType, PngEncoder},
};
use rayon::prelude::*;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("usage: png_bench <dir> [limit]");
    let limit: usize = args
        .next()
        .map_or(usize::MAX, |v| v.parse().unwrap_or(usize::MAX));

    let mut paths: Vec<_> = std::fs::read_dir(&dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("png")))
        .collect();
    paths.sort();
    paths.truncate(limit);
    println!("decoding {} crops from {dir}", paths.len());

    // Decoded once and held, so the loop below times encoding alone.
    let images: Vec<_> = paths
        .par_iter()
        .map(|p| image::open(p).map(|i| i.to_rgba8()))
        .collect::<Result<_, _>>()?;
    let raw_bytes: u64 = images.iter().map(|i| i.as_raw().len() as u64).sum();
    println!(
        "{} images, {:.1} MB raw\n",
        images.len(),
        raw_bytes as f64 / 1e6
    );

    let combos = [
        ("fast", CompressionType::Fast, FilterType::Adaptive),
        ("fast/nofilter", CompressionType::Fast, FilterType::NoFilter),
        ("fast/sub", CompressionType::Fast, FilterType::Sub),
        ("fast/paeth", CompressionType::Fast, FilterType::Paeth),
        ("default", CompressionType::Default, FilterType::Adaptive),
        (
            "default/nofilter",
            CompressionType::Default,
            FilterType::NoFilter,
        ),
        ("default/sub", CompressionType::Default, FilterType::Sub),
        ("default/up", CompressionType::Default, FilterType::Up),
        ("default/paeth", CompressionType::Default, FilterType::Paeth),
        ("best", CompressionType::Best, FilterType::Adaptive),
    ];

    println!(
        "{:<18} {:>9} {:>12} {:>10}",
        "setting", "time (s)", "bytes (MB)", "vs default"
    );
    let mut baseline_bytes = 0f64;

    // Two passes in opposite orders, so a warming or drifting machine cannot rank them.
    for pass in 0..2 {
        let order: Vec<usize> = if pass == 0 {
            (0..combos.len()).collect()
        } else {
            (0..combos.len()).rev().collect()
        };
        println!("--- pass {} ---", pass + 1);
        let mut results = vec![(0f64, 0u64); combos.len()];
        for &i in &order {
            let (name, compression, filter) = combos[i];
            let start = Instant::now();
            let total: u64 = images
                .par_iter()
                .map(|img| {
                    let mut buf = Vec::new();
                    PngEncoder::new_with_quality(&mut buf, compression, filter)
                        .write_image(
                            img.as_raw(),
                            img.width(),
                            img.height(),
                            ExtendedColorType::Rgba8,
                        )
                        .expect("encode");
                    buf.len() as u64
                })
                .sum();
            let secs = start.elapsed().as_secs_f64();
            results[i] = (secs, total);
            if name == "default" {
                baseline_bytes = total as f64;
            }
        }
        for (i, (name, _, _)) in combos.iter().enumerate() {
            let (secs, total) = results[i];
            println!(
                "{:<18} {:>9.2} {:>12.1} {:>9.1}%",
                name,
                secs,
                total as f64 / 1e6,
                100.0 * (total as f64 - baseline_bytes) / baseline_bytes
            );
        }
        println!();
    }
    Ok(())
}
