//! How much of a detection job is actually JPEG decode? (experiment 67)
//!
//! Detection is now about 3 ms for a 10 MP photo. Decoding that photo is several times
//! more, so the decoder -- not the detector -- sets what a folder of images costs. This
//! compares the shipped path (`image`, which uses zune-jpeg) against libjpeg-turbo, which
//! is already linked into every binary through `nokhwa` and so is free to try.
//!
//! Two decoders will not agree bit for bit: the JPEG standard leaves IDCT precision open,
//! and each picks its own. The comparison below reports how far apart they land, because
//! that difference reaches exported crop pixels and is the thing to argue about -- speed
//! alone does not decide this.
//!
//! Cold I/O is deliberately excluded: every file is read into memory first, so this times
//! decoding rather than the disk.
//!
//! Run with: cargo run --release -p fcs-core --example decode_bench [image-count]

use std::time::Instant;

use anyhow::{Context, Result};

#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

const DEFAULT_IMAGES: usize = 25;
const RUNS: usize = 5;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() -> Result<()> {
    let limit: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_IMAGES);

    let mut paths: Vec<_> = std::fs::read_dir("fixtures/images")
        .context("read fixtures/images")?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("jpg")))
        .collect();
    paths.sort();
    // Largest first: decode cost scales with pixels, and the large files are where a
    // folder job actually spends its time.
    paths.sort_by_key(|p| std::cmp::Reverse(std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)));
    paths.truncate(limit);
    anyhow::ensure!(!paths.is_empty(), "no .jpg fixtures under fixtures/images");

    println!(
        "{:<16} {:>9} {:>10} {:>10} {:>8} {:>9} {:>7}",
        "image", "MP", "image ms", "turbo ms", "speedup", "mean |d|", "max |d|"
    );

    let (mut total_image, mut total_turbo) = (0.0f64, 0.0f64);
    let mut worst_max = 0u8;
    let mut all_mean = Vec::new();

    for path in &paths {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;

        let baseline = median(
            (0..RUNS)
                .map(|_| {
                    let t = Instant::now();
                    let img = image::load_from_memory(&bytes).expect("image decode");
                    std::hint::black_box(img.width());
                    t.elapsed().as_secs_f64() * 1e3
                })
                .collect(),
        );

        let mut turbo_pixels = Vec::new();
        let turbo = median(
            (0..RUNS)
                .map(|_| {
                    let t = Instant::now();
                    let decompress = mozjpeg::Decompress::new_mem(&bytes).expect("turbo header");
                    let mut started = decompress.rgb().expect("turbo rgb");
                    let pixels: Vec<[u8; 3]> = started.read_scanlines().expect("turbo scanlines");
                    started.finish().expect("turbo finish");
                    let ms = t.elapsed().as_secs_f64() * 1e3;
                    turbo_pixels = pixels;
                    ms
                })
                .collect(),
        );

        // Same image, two IDCTs: report the gap rather than asserting equality.
        let reference = image::load_from_memory(&bytes)?.to_rgb8();
        let (mut sum, mut count, mut max) = (0u64, 0u64, 0u8);
        for (a, b) in reference.pixels().zip(turbo_pixels.iter()) {
            for (left, right) in a.0.iter().zip(b.iter()) {
                let d = left.abs_diff(*right);
                sum += u64::from(d);
                count += 1;
                max = max.max(d);
            }
        }
        let mean = if count == 0 {
            0.0
        } else {
            sum as f64 / count as f64
        };
        all_mean.push(mean);
        worst_max = worst_max.max(max);
        total_image += baseline;
        total_turbo += turbo;

        let mp = f64::from(reference.width()) * f64::from(reference.height()) / 1e6;
        println!(
            "{:<16} {mp:>9.1} {baseline:>10.2} {turbo:>10.2} {:>7.2}x {mean:>9.4} {max:>7}",
            path.file_name().unwrap_or_default().to_string_lossy(),
            baseline / turbo
        );
    }

    println!(
        "\ntotal over {} images: image {total_image:.0} ms, libjpeg-turbo {total_turbo:.0} ms \
         ({:.2}x)",
        paths.len(),
        total_image / total_turbo
    );
    println!(
        "pixel difference: mean of per-image means {:.4}, worst single channel {worst_max}",
        all_mean.iter().sum::<f64>() / all_mean.len() as f64
    );

    // The production entry point, not the raw decoders: this is where EXIF orientation is
    // applied, and a turbo path that ignored it would return a correctly coloured image of
    // the wrong shape. Dimensions are checked first for exactly that reason.
    println!(
        "
via load_image, against the decoder it replaced"
    );
    let mut mismatched_dims = 0usize;
    let mut worst_load = 0u8;
    for path in &paths {
        let plain = reference_load(path)?.to_rgb8();
        let turbo = fcs_utils::load_image(path)?.to_rgb8();

        if plain.dimensions() != turbo.dimensions() {
            mismatched_dims += 1;
            println!(
                "  {} DIMENSION MISMATCH {:?} vs {:?}",
                path.display(),
                plain.dimensions(),
                turbo.dimensions()
            );
            continue;
        }
        let max = plain
            .pixels()
            .zip(turbo.pixels())
            .flat_map(|(a, b)| a.0.iter().zip(b.0.iter()).map(|(x, y)| x.abs_diff(*y)))
            .max()
            .unwrap_or(0);
        worst_load = worst_load.max(max);
    }
    println!(
        "  {} images, {mismatched_dims} dimension mismatches, worst channel difference {worst_load}",
        paths.len()
    );
    anyhow::ensure!(
        mismatched_dims == 0,
        "orientation handling differs between the two decode paths"
    );
    Ok(())
}

/// What `load_image` did before libjpeg-turbo: `image`'s decoder plus EXIF orientation.
///
/// Kept here rather than behind a runtime switch in the library. Only this probe wants the
/// comparison, and a branch in the shipped loader to serve it would be a production cost
/// paid for a benchmark.
fn reference_load(path: &std::path::Path) -> Result<image::DynamicImage> {
    use image::ImageDecoder;
    let mut decoder = image::ImageReader::open(path)?
        .with_guessed_format()?
        .into_decoder()?;
    let orientation = decoder
        .orientation()
        .unwrap_or(image::metadata::Orientation::NoTransforms);
    let mut image = image::DynamicImage::from_decoder(decoder)?;
    image.apply_orientation(orientation);
    Ok(image)
}
