//! Is a second, reduced-scale decode cheaper than resizing the full one?
//!
//! Experiment 51 closed the resize itself: at a fixed source resolution it is bounded below by
//! reading the source once. The only lever left is fewer source pixels, and libjpeg can supply
//! those directly by scaling during the IDCT.
//!
//! Experiment 86 dismissed this because 82% of the reference folder contains a face and so
//! needs the full-resolution decode anyway for its crop -- the reduced decode is *added* work,
//! not replacement work. True, but it does not settle whether the added work costs less than
//! the resize it removes. That is arithmetic, and this measures it (experiment 90).
//!
//!   cargo run --release -p fcs-core --example scaled_decode -- <dir-of-jpegs> [limit]

use fcs_utils::resize_image;
use image::{DynamicImage, ImageReader, RgbImage, imageops::FilterType};
use rayon::prelude::*;
use std::time::Instant;

const TARGET: u32 = 640;

/// Decode a JPEG at `num`/8 scale. `num == 8` is the full-resolution path in production.
fn decode_scaled(bytes: &[u8], num: u8) -> Option<DynamicImage> {
    let mut decompress = mozjpeg::Decompress::new_mem(bytes).ok()?;
    if num < 8 {
        decompress.scale(num);
    }
    let mut started = decompress.rgb().ok()?;
    let (w, h) = (started.width(), started.height());
    let pixels: Vec<u8> = started.read_scanlines().ok()?;
    started.finish().ok()?;
    Some(DynamicImage::ImageRgb8(RgbImage::from_raw(
        w as u32, h as u32, pixels,
    )?))
}

/// The largest reduction whose output still covers the detector input, so nothing is upscaled.
///
/// Production letterboxes into TARGET x TARGET, so the requirement is that the shorter route to
/// that box does not go below it: scaled dimensions must stay >= the size production would
/// itself resize to.
fn best_scale(w: u32, h: u32) -> u8 {
    for num in [1u8, 2, 4] {
        let (sw, sh) = (w * num as u32 / 8, h * num as u32 / 8);
        // Keep a margin: the scaled image must still be at least the target box in the
        // dimension that drives the letterbox fit.
        if sw.min(sh) >= TARGET && sw.max(sh) >= TARGET {
            return num;
        }
    }
    8
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("usage: scaled_decode <dir> [limit]");
    let limit: usize = args
        .next()
        .map_or(usize::MAX, |v| v.parse().unwrap_or(usize::MAX));

    let mut paths: Vec<_> = std::fs::read_dir(&dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("jpg") || e.eq_ignore_ascii_case("jpeg"))
        })
        .collect();
    paths.sort();
    paths.truncate(limit);
    println!("{} JPEGs from {dir}", paths.len());

    // Read once, so file I/O is not part of any measurement below.
    let files: Vec<Vec<u8>> = paths
        .par_iter()
        .filter_map(|p: &std::path::PathBuf| std::fs::read(p).ok())
        .collect();
    println!("{} read into memory\n", files.len());

    let dims: Vec<(u32, u32, u8)> = files
        .par_iter()
        .filter_map(|b| {
            let d = ImageReader::new(std::io::Cursor::new(b))
                .with_guessed_format()
                .ok()?
                .into_dimensions()
                .ok()?;
            Some((d.0, d.1, best_scale(d.0, d.1)))
        })
        .collect();
    let mut counts = [0usize; 9];
    for (_, _, s) in &dims {
        counts[*s as usize] += 1;
    }
    println!(
        "chosen scale: 1/8 {}  1/4 {}  1/2 {}  full {}",
        counts[1], counts[2], counts[4], counts[8]
    );
    let mp: f64 = dims
        .iter()
        .map(|(w, h, _)| f64::from(*w) * f64::from(*h))
        .sum::<f64>()
        / 1e6;
    println!("total source {mp:.0} MP\n");

    let time = |label: &str, f: &(dyn Fn() -> u64 + Sync)| {
        let start = Instant::now();
        let px = f();
        let secs = start.elapsed().as_secs_f64();
        println!("{label:<34} {secs:>7.2} s   ({px} px out)");
        secs
    };

    // Alternated across two passes so a warming machine cannot rank the two paths.
    for pass in 0..2 {
        println!("--- pass {} ---", pass + 1);
        let mut full_decode = 0.0;
        let mut full_resize = 0.0;
        let mut scaled_decode = 0.0;
        let mut scaled_resize = 0.0;

        let run_current = |full_decode: &mut f64, full_resize: &mut f64| {
            // Production today: decode at full resolution, then resize that to the input.
            *full_decode += time("full decode (needed for crops)", &|| {
                files
                    .par_iter()
                    .filter_map(|b| decode_scaled(b, 8))
                    .map(|i| u64::from(i.width()))
                    .sum()
            });
            *full_resize += time("  + resize full -> 640", &|| {
                files
                    .par_iter()
                    .filter_map(|b| decode_scaled(b, 8))
                    .map(|i| {
                        u64::from(resize_image(&i, TARGET, TARGET, FilterType::Triangle).width())
                    })
                    .sum()
            });
        };
        let run_candidate = |scaled_decode: &mut f64, scaled_resize: &mut f64| {
            *scaled_decode += time("scaled decode (extra pass)", &|| {
                files
                    .par_iter()
                    .zip(&dims)
                    .filter_map(|(b, (_, _, s))| decode_scaled(b, *s))
                    .map(|i| u64::from(i.width()))
                    .sum()
            });
            *scaled_resize += time("  + resize scaled -> 640", &|| {
                files
                    .par_iter()
                    .zip(&dims)
                    .filter_map(|(b, (_, _, s))| decode_scaled(b, *s))
                    .map(|i| {
                        u64::from(resize_image(&i, TARGET, TARGET, FilterType::Triangle).width())
                    })
                    .sum()
            });
        };

        if pass == 0 {
            run_current(&mut full_decode, &mut full_resize);
            run_candidate(&mut scaled_decode, &mut scaled_resize);
        } else {
            run_candidate(&mut scaled_decode, &mut scaled_resize);
            run_current(&mut full_decode, &mut full_resize);
        }

        // The timed loops each re-decode, so subtract the decode to isolate the resize.
        let resize_only = full_resize - full_decode;
        let scaled_only = scaled_resize - scaled_decode;
        println!();
        println!("  resize alone, from full   : {:.2} s", resize_only);
        println!(
            "  scaled decode + its resize: {:.2} s  ({:.2} + {:.2})",
            scaled_decode + scaled_only,
            scaled_decode,
            scaled_only
        );
        println!(
            "  => candidate is {:+.2} s per folder\n",
            (scaled_decode + scaled_only) - resize_only
        );
    }
    Ok(())
}
