//! Where background blur time goes: shader passes vs the upload/readback around them.
//!
//! Run with: cargo run --release -p fcs-core --example blur_cost

use std::time::Instant;

use anyhow::Result;
use fcs_utils::gpu::{
    GpuAvailability, GpuBackgroundBlur, GpuContext, GpuContextOptions, GpuGaussianBlur,
};
use image::{DynamicImage, RgbaImage};

#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

const WARMUP: usize = 5;
const RUNS: usize = 30;
const RADIUS: f32 = 15.0;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn noise(side: u32) -> DynamicImage {
    let mut s = 0x9E37_79B9u32;
    DynamicImage::ImageRgba8(RgbaImage::from_fn(side, side, |_, _| {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        image::Rgba([s as u8, (s >> 8) as u8, (s >> 16) as u8, 255])
    }))
}

fn main() -> Result<()> {
    let options = GpuContextOptions {
        profiling: true,
        ..GpuContextOptions::default()
    };
    let context = match GpuContext::init_with_fallback(&options) {
        GpuAvailability::Available(ctx) => ctx,
        other => anyhow::bail!("GPU unavailable: {other:?}"),
    };
    anyhow::ensure!(context.profiler().is_some(), "GPU timestamps unavailable");
    println!("adapter: {}\n", context.adapter_info().name);

    let gaussian = GpuGaussianBlur::new(context.clone())?;
    let blend = GpuBackgroundBlur::new(context.clone())?;

    println!(
        "{:>6}  {:>9} {:>9} {:>9}  {:>9} {:>9}  {:>9}",
        "side", "blur_h", "blur_v", "blur wall", "blend gpu", "blend wall", "cpu fast"
    );
    for side in [512u32, 1024, 2048] {
        let img = noise(side);
        let (mut h, mut v, mut bw, mut bg, mut blw, mut cpu) =
            (vec![], vec![], vec![], vec![], vec![], vec![]);
        for run in 0..WARMUP + RUNS {
            context.take_pass_timings()?;
            let t = Instant::now();
            let blurred = gaussian.blur(&img, RADIUS)?;
            let blur_wall = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            blend.blend(&img, &blurred, 0.6)?;
            let blend_wall = t.elapsed().as_secs_f64() * 1e3;
            let timings = context.take_pass_timings()?;
            let t = Instant::now();
            std::hint::black_box(image::imageops::fast_blur(&img.to_rgba8(), RADIUS));
            let cpu_wall = t.elapsed().as_secs_f64() * 1e3;
            if run < WARMUP {
                continue;
            }
            let ms = |label: &str| {
                timings
                    .iter()
                    .filter(|p| p.label == label)
                    .map(|p| p.duration_ns / 1e6)
                    .sum::<f64>()
            };
            h.push(ms("gaussian_blur_h"));
            v.push(ms("gaussian_blur_v"));
            bg.push(ms("background_blur"));
            bw.push(blur_wall);
            blw.push(blend_wall);
            cpu.push(cpu_wall);
        }
        println!(
            "{side:>6}  {:>9.3} {:>9.3} {:>9.3}  {:>9.3} {:>9.3}  {:>9.3}",
            median(h),
            median(v),
            median(bw),
            median(bg),
            median(blw),
            median(cpu)
        );
    }
    println!("\nms, median of {RUNS}; radius {RADIUS} (GPU caps at 12)");
    Ok(())
}
