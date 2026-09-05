//! Wall-clock timings for the GPU image shaders that carry the most per-pixel work.
//!
//! Each case submits and waits, so the number covers dispatch plus readback rather than
//! shader time alone — enough to compare a shader change against itself, not a profiler.

use criterion::{Criterion, criterion_group, criterion_main};
use fcs_utils::{
    color::RgbaColor,
    gpu::{
        GpuAvailability, GpuBilateralFilter, GpuContext, GpuContextOptions, GpuHistogramEqualizer,
        GpuShapeMask,
    },
    shape::CropShape,
};
use image::{DynamicImage, Rgba, RgbaImage};
use std::{hint::black_box, sync::Arc};

const WIDTH: u32 = 1024;
const HEIGHT: u32 = 1024;

fn gpu_context() -> Option<Arc<GpuContext>> {
    match GpuContext::init_with_fallback(&GpuContextOptions::default()) {
        GpuAvailability::Available(ctx) => Some(ctx),
        _ => None,
    }
}

/// A varied image: a flat fill would give the histogram a single hot bin and the
/// bilateral filter no colour edges to weigh, flattering both.
fn test_image() -> DynamicImage {
    let mut img = RgbaImage::new(WIDTH, HEIGHT);
    for (x, y, px) in img.enumerate_pixels_mut() {
        let r = ((x * 7 + y * 3) % 256) as u8;
        let g = ((x * 3 + y * 11) % 256) as u8;
        let b = ((x ^ y) % 256) as u8;
        *px = Rgba([r, g, b, 255]);
    }
    DynamicImage::ImageRgba8(img)
}

fn gpu_shader_benchmark(c: &mut Criterion) {
    let Some(ctx) = gpu_context() else {
        eprintln!("Skipping GPU shader benchmarks (no adapter available)");
        return;
    };
    let image = test_image();

    let equalizer = GpuHistogramEqualizer::new(ctx.clone()).expect("histogram equalizer");
    c.bench_function("gpu_hist_equalize_1024", |b| {
        b.iter(|| black_box(equalizer.equalize(black_box(&image)).expect("equalize")));
    });

    let bilateral = GpuBilateralFilter::new(ctx.clone()).expect("bilateral filter");
    // sigma_space 4.0 lands on radius 8, the widest window the shader supports.
    c.bench_function("gpu_bilateral_r8_1024", |b| {
        b.iter(|| {
            black_box(
                bilateral
                    .smooth(black_box(&image), 1.0, 4.0, 30.0)
                    .expect("smooth"),
            )
        });
    });

    let shape_mask = GpuShapeMask::new(ctx).expect("shape mask");
    // Ellipse outlines to many polygon points, so the per-pixel polygon walk dominates.
    c.bench_function("gpu_shape_mask_ellipse_1024", |b| {
        b.iter(|| {
            black_box(
                shape_mask
                    .apply(
                        black_box(&image),
                        &CropShape::Ellipse,
                        0.0,
                        0.0,
                        RgbaColor::opaque(0, 0, 0),
                    )
                    .expect("apply"),
            )
        });
    });

    // Softness > 0 was the only path that previously ran sd_polygon; keep it measured
    // separately so a change to either coverage path shows up on its own.
    let shape_mask_soft = GpuShapeMask::new(gpu_context().expect("adapter")).expect("shape mask");
    c.bench_function("gpu_shape_mask_ellipse_vignette_1024", |b| {
        b.iter(|| {
            black_box(
                shape_mask_soft
                    .apply(
                        black_box(&image),
                        &CropShape::Ellipse,
                        0.35,
                        1.0,
                        RgbaColor::opaque(0, 0, 0),
                    )
                    .expect("apply"),
            )
        });
    });
}

criterion_group!(benches, gpu_shader_benchmark);
criterion_main!(benches);
