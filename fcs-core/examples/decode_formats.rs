//! What does each container format cost to decode, on identical pixels?
//!
//! The workload matrix (experiment 6) suggested WebP and PNG decode far slower per megapixel
//! than JPEG, but bucketed a corpus where each format held different images at different sizes,
//! so the comparison was confounded. This re-encodes one source into every format at the same
//! dimensions and times the decode, which is the same content and the same pixel count.
//!
//! Encoding time is reported too, because export uses these paths as well.
//!
//!   cargo run --release -p fcs-core --example decode_formats -- <image> [reps]

use std::time::Instant;

use anyhow::Result;
use image::ImageFormat;

fn pct(sorted: &[f64], f: f64) -> f64 {
    sorted[((sorted.len() as f64 * f) as usize).min(sorted.len() - 1)]
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: decode_formats <image> [reps]");
    let reps: usize = args.next().map_or(7, |v| v.parse().unwrap_or(7));

    let source = image::open(&path)?;
    let mp = f64::from(source.width()) * f64::from(source.height()) / 1e6;
    println!(
        "{}x{} ({mp:.1} MP), {reps} reps\n",
        source.width(),
        source.height()
    );

    // JPEG and WebP here are lossy at these settings and PNG is not, so the encoded sizes are
    // not comparable as quality-for-bytes. The decode timing is the point.
    let formats = [
        ("jpeg", ImageFormat::Jpeg),
        ("png", ImageFormat::Png),
        ("webp", ImageFormat::WebP),
        ("bmp", ImageFormat::Bmp),
        ("tiff", ImageFormat::Tiff),
    ];

    println!(
        "{:<8} {:>10} {:>12} {:>12} {:>12} {:>10}",
        "format", "MB", "encode ms", "decode p50", "decode p95", "ms/MP"
    );

    for (name, format) in formats {
        let mut encoded = std::io::Cursor::new(Vec::new());
        let start = Instant::now();
        if source.write_to(&mut encoded, format).is_err() {
            println!("{name:<8} {:>10} encoder unavailable", "-");
            continue;
        }
        let encode_ms = start.elapsed().as_secs_f64() * 1e3;
        let bytes = encoded.into_inner();

        let mut times = Vec::with_capacity(reps);
        for _ in 0..reps {
            let start = Instant::now();
            let decoded = image::load_from_memory_with_format(&bytes, format)?;
            times.push(start.elapsed().as_secs_f64() * 1e3);
            std::hint::black_box(&decoded);
        }
        times.sort_by(f64::total_cmp);
        let p50 = pct(&times, 0.5);
        println!(
            "{name:<8} {:>10.1} {encode_ms:>12.1} {p50:>12.1} {:>12.1} {:>10.2}",
            bytes.len() as f64 / 1e6,
            pct(&times, 0.95),
            p50 / mp,
        );
    }

    println!("\nJPEG here is the `image` crate's decoder, not the libjpeg-turbo path the");
    println!("application uses for files on disk, which experiment 67 measured at 1.17-1.23x.");
    Ok(())
}
