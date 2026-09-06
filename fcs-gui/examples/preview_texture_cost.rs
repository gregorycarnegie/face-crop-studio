//! What does building the preview texture cost?
//!
//! The GUI clamps a preview texture at 8192 per side, so an ordinary 12 MP photo goes up
//! whole and the clamp never runs. Past that -- camera RAWs and panoramas, which the clamp
//! exists for -- it did run through `DynamicImage::resize_exact`, which samples pixel by pixel
//! through `GenericImageView`. This compares that against the SIMD resize now used, and shows
//! what the `ColorImage` construction costs on its own (experiment 91).
//!
//!   cargo run --release -p fcs-gui --example preview_texture_cost -- <image> [reps]

use std::time::Instant;

use image::DynamicImage;

fn fit(w: u32, h: u32, limit: u32) -> (u32, u32) {
    let scale = limit as f32 / w.max(h) as f32;
    (
        ((w as f32 * scale).floor() as u32).max(1),
        ((h as f32 * scale).floor() as u32).max(1),
    )
}

/// The clamp as it was: `DynamicImage::resize_exact`.
fn clamp_slow(image: &DynamicImage, limit: u32) -> std::borrow::Cow<'_, DynamicImage> {
    let (w, h) = (image.width(), image.height());
    if w <= limit && h <= limit {
        return std::borrow::Cow::Borrowed(image);
    }
    let (nw, nh) = fit(w, h, limit);
    std::borrow::Cow::Owned(image.resize_exact(nw, nh, image::imageops::FilterType::Triangle))
}

/// The clamp as it is now: through `fast_image_resize`.
fn clamp_fast(image: &DynamicImage, limit: u32) -> std::borrow::Cow<'_, DynamicImage> {
    let (w, h) = (image.width(), image.height());
    if w <= limit && h <= limit {
        return std::borrow::Cow::Borrowed(image);
    }
    let (nw, nh) = fit(w, h, limit);
    let filter = image::imageops::FilterType::Triangle;
    let resized = if image.color().has_alpha() {
        let rgba = match image {
            DynamicImage::ImageRgba8(rgba) => std::borrow::Cow::Borrowed(rgba),
            other => std::borrow::Cow::Owned(other.to_rgba8()),
        };
        fcs_utils::resize_rgba_fast(&rgba, nw, nh, filter).map(DynamicImage::ImageRgba8)
    } else {
        Some(DynamicImage::ImageRgb8(fcs_utils::resize_image(
            image, nw, nh, filter,
        )))
    };
    std::borrow::Cow::Owned(resized.unwrap_or_else(|| image.resize_exact(nw, nh, filter)))
}

/// Mirrors `core::detection::color_image_from_dynamic`.
fn build(image: &DynamicImage, limit: u32, slow: bool) -> (egui::ColorImage, [usize; 2]) {
    let clamped = if slow {
        clamp_slow(image, limit)
    } else {
        clamp_fast(image, limit)
    };
    let image = clamped.as_ref();
    let size = [image.width() as usize, image.height() as usize];
    let ci = match image {
        DynamicImage::ImageRgb8(rgb) => egui::ColorImage::from_rgb(size, rgb.as_raw()),
        DynamicImage::ImageRgba8(rgba) => {
            egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_raw())
        }
        other => {
            let rgba = other.to_rgba8();
            egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_raw())
        }
    };
    (ci, size)
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: preview_texture_cost <image> [reps]");
    let reps: usize = args.next().map_or(9, |v| v.parse().unwrap_or(9));

    let bytes = std::fs::read(&path)?;
    let start = Instant::now();
    let image = image::load_from_memory(&bytes)?;
    let decode_ms = start.elapsed().as_secs_f64() * 1e3;
    println!(
        "{}x{} ({:.1} MP), decode {decode_ms:.1} ms, {reps} reps per cell\n",
        image.width(),
        image.height(),
        f64::from(image.width()) * f64::from(image.height()) / 1e6
    );

    println!(
        "{:<8} {:>13} {:>8} {:>14} {:>10} {:>9}",
        "clamp", "texture", "MB", "resize_exact", "fir", "speedup"
    );

    let run = |limit: u32, slow: bool| {
        let mut times = Vec::with_capacity(reps);
        let mut size = [0usize; 2];
        for _ in 0..reps {
            let start = Instant::now();
            let (ci, s) = build(&image, limit, slow);
            times.push(start.elapsed().as_secs_f64() * 1e3);
            size = s;
            std::hint::black_box(&ci);
        }
        times.sort_by(f64::total_cmp);
        (times[times.len() / 2], size)
    };

    // Alternated per row so a warming machine cannot favour whichever variant runs first.
    for (round, limit) in [8192u32, 4096, 3072, 2048, 1536, 1024]
        .into_iter()
        .enumerate()
    {
        let (slow_ms, fast_ms, size) = if round % 2 == 0 {
            let (s, sz) = run(limit, true);
            let (f, _) = run(limit, false);
            (s, f, sz)
        } else {
            let (f, sz) = run(limit, false);
            let (s, _) = run(limit, true);
            (s, f, sz)
        };
        let mb = (size[0] * size[1] * 4) as f64 / 1e6;
        println!(
            "{limit:<8} {:>6}x{:<6} {mb:>8.1} {slow_ms:>14.2} {fast_ms:>10.2} {:>8.1}x",
            size[0],
            size[1],
            slow_ms / fast_ms
        );
    }
    println!("\nRows where the texture equals the source are no-ops on both sides:");
    println!("the clamp does not fire, and the time shown is the ColorImage build alone.");
    Ok(())
}
