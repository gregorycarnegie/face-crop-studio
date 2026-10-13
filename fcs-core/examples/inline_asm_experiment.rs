//! Historical pixel loop vs production row slices and experimental SIMD conversion.
//! cargo run --release -p fcs-core --example inline_asm_experiment -- [image ...]
// Off x86_64 `main` only bails, so the rest is unreachable; CI lints ARM with -D warnings.
#![cfg_attr(not(target_arch = "x86_64"), allow(unused))]

use std::{hint::black_box, time::Instant};

use anyhow::Result;
use fcs_core::{cpu::tensor::Tensor, scrfd};
use image::{DynamicImage, RgbImage, imageops::FilterType};

#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

const SIZE: u32 = 640;
const PAD: f32 = -127.5 / 128.0;

#[derive(Clone, Copy, Debug)]
enum Mode {
    Legacy,
    Rows,
    Intrinsics,
    Asm,
}

fn scalar_row(rgb: &[u8], r: &mut [f32], g: &mut [f32], b: &mut [f32]) {
    for (((pixel, r), g), b) in rgb.as_chunks::<3>().0.iter().zip(r).zip(g).zip(b) {
        *r = (pixel[0] as f32 - 127.5) / 128.0;
        *g = (pixel[1] as f32 - 127.5) / 128.0;
        *b = (pixel[2] as f32 - 127.5) / 128.0;
    }
}

// Each pair extracts one channel from the first 16 and final 8 bytes of eight RGB pixels.
#[cfg(target_arch = "x86_64")]
const MASKS: [[u8; 16]; 6] = [
    [
        0, 3, 6, 9, 12, 15, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    ],
    [
        128, 128, 128, 128, 128, 128, 2, 5, 128, 128, 128, 128, 128, 128, 128, 128,
    ],
    [
        1, 4, 7, 10, 13, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    ],
    [
        128, 128, 128, 128, 128, 0, 3, 6, 128, 128, 128, 128, 128, 128, 128, 128,
    ],
    [
        2, 5, 8, 11, 14, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    ],
    [
        128, 128, 128, 128, 128, 1, 4, 7, 128, 128, 128, 128, 128, 128, 128, 128,
    ],
];

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn intrinsic_bulk(src: *const u8, outputs: [*mut f32; 3], blocks: usize) {
    use std::arch::x86_64::*;
    // SAFETY: the checked caller provides blocks*24 readable bytes, blocks*8 writable
    // floats per disjoint channel, and has checked AVX2 support. Loads are unaligned.
    unsafe {
        let masks: [__m128i; 6] =
            std::array::from_fn(|i| _mm_loadu_si128(MASKS[i].as_ptr().cast()));
        let bias = _mm256_set1_ps(127.5);
        let scale = _mm256_set1_ps(1.0 / 128.0);
        for block in 0..blocks {
            let lo = _mm_loadu_si128(src.add(block * 24).cast());
            let hi = _mm_loadl_epi64(src.add(block * 24 + 16).cast());
            for c in 0..3 {
                let bytes = _mm_or_si128(
                    _mm_shuffle_epi8(lo, masks[2 * c]),
                    _mm_shuffle_epi8(hi, masks[2 * c + 1]),
                );
                let floats = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes));
                let result = _mm256_mul_ps(_mm256_sub_ps(floats, bias), scale);
                _mm256_storeu_ps(outputs[c].add(block * 8), result);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn asm_bulk(src: *const u8, outputs: [*mut f32; 3], blocks: usize) {
    let [r, g, b] = outputs;
    let bias = 127.5_f32;
    let scale = 1.0_f32 / 128.0;
    // SAFETY: same bounds and AVX2 contract as intrinsic_bulk, with blocks > 0.
    // All modified registers are declared. Memory is read AND written; no nomem/readonly.
    unsafe {
        std::arch::asm!(
            "vmovdqu xmm0, [{masks}]",
            "vmovdqu xmm1, [{masks} + 16]",
            "vmovdqu xmm2, [{masks} + 32]",
            "vmovdqu xmm3, [{masks} + 48]",
            "vmovdqu xmm4, [{masks} + 64]",
            "vmovdqu xmm5, [{masks} + 80]",
            "vbroadcastss ymm6, [{bias}]",
            "vbroadcastss ymm7, [{scale}]",
            "2:",
            "vmovdqu xmm8, [{src}]",
            "vmovq xmm9, [{src} + 16]",
            "vpshufb xmm10, xmm8, xmm0",
            "vpshufb xmm11, xmm9, xmm1",
            "vpor xmm10, xmm10, xmm11",
            "vpmovzxbd ymm10, xmm10",
            "vcvtdq2ps ymm10, ymm10",
            "vsubps ymm10, ymm10, ymm6",
            "vmulps ymm10, ymm10, ymm7",
            "vmovups [{r}], ymm10",
            "vpshufb xmm10, xmm8, xmm2",
            "vpshufb xmm11, xmm9, xmm3",
            "vpor xmm10, xmm10, xmm11",
            "vpmovzxbd ymm10, xmm10",
            "vcvtdq2ps ymm10, ymm10",
            "vsubps ymm10, ymm10, ymm6",
            "vmulps ymm10, ymm10, ymm7",
            "vmovups [{g}], ymm10",
            "vpshufb xmm10, xmm8, xmm4",
            "vpshufb xmm11, xmm9, xmm5",
            "vpor xmm10, xmm10, xmm11",
            "vpmovzxbd ymm10, xmm10",
            "vcvtdq2ps ymm10, ymm10",
            "vsubps ymm10, ymm10, ymm6",
            "vmulps ymm10, ymm10, ymm7",
            "vmovups [{b}], ymm10",
            "add {src}, 24",
            "add {r}, 32",
            "add {g}, 32",
            "add {b}, 32",
            "dec {blocks}",
            "jnz 2b",
            src = inout(reg) src => _,
            r = inout(reg) r => _, g = inout(reg) g => _, b = inout(reg) b => _,
            blocks = inout(reg) blocks => _,
            masks = in(reg) MASKS.as_ptr(), bias = in(reg) &bias, scale = in(reg) &scale,
            out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
            out("ymm4") _, out("ymm5") _, out("ymm6") _, out("ymm7") _,
            out("ymm8") _, out("ymm9") _, out("ymm10") _, out("ymm11") _,
            options(nostack),
        );
    }
}

fn row(rgb: &[u8], r: &mut [f32], g: &mut [f32], b: &mut [f32], mode: Mode) {
    assert_eq!(rgb.len(), r.len() * 3);
    assert_eq!(r.len(), g.len());
    assert_eq!(r.len(), b.len());
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    if matches!(mode, Mode::Intrinsics | Mode::Asm) && std::is_x86_feature_detected!("avx2") {
        let blocks = r.len() / 8;
        if blocks > 0 {
            // SAFETY: lengths above cover the bulk loads/stores; slices are disjoint and
            // AVX2 was checked. The final load reads exactly 8 bytes, never past the row.
            unsafe {
                let outputs = [r.as_mut_ptr(), g.as_mut_ptr(), b.as_mut_ptr()];
                match mode {
                    Mode::Intrinsics => intrinsic_bulk(rgb.as_ptr(), outputs, blocks),
                    Mode::Asm => asm_bulk(rgb.as_ptr(), outputs, blocks),
                    _ => unreachable!(),
                }
            }
            done = blocks * 8;
        }
    }
    scalar_row(
        &rgb[done * 3..],
        &mut r[done..],
        &mut g[done..],
        &mut b[done..],
    );
}

// Use the same compiled baseline in the A/A and A/B closures.
#[inline(never)]
fn normalise(resized: &RgbImage, size: u32, mode: Mode) -> Vec<f32> {
    let plane = (size * size) as usize;
    let mut tensor = vec![PAD; 3 * plane];
    if matches!(mode, Mode::Legacy) {
        // Original pixel loop, retained as the independent historical baseline.
        for y in 0..resized.height().min(size) {
            for x in 0..resized.width().min(size) {
                let pixel = resized.get_pixel(x, y);
                let index = (y * size + x) as usize;
                for channel in 0..3 {
                    tensor[channel * plane + index] = (pixel[channel] as f32 - 127.5) / 128.0;
                }
            }
        }
    } else {
        let (r, gb) = tensor.split_at_mut(plane);
        let (g, b) = gb.split_at_mut(plane);
        let width = resized.width().min(size) as usize;
        for (y, rgb) in resized
            .as_raw()
            .chunks_exact(resized.width() as usize * 3)
            .take(resized.height().min(size) as usize)
            .enumerate()
        {
            let start = y * size as usize;
            let end = start + width;
            row(
                &rgb[..width * 3],
                &mut r[start..end],
                &mut g[start..end],
                &mut b[start..end],
                mode,
            );
        }
    }
    tensor
}

fn resize(image: &DynamicImage, size: u32) -> RgbImage {
    let ratio = image.height().max(1) as f32 / image.width().max(1) as f32;
    let (w, h) = if ratio > 1.0 {
        (((size as f32 / ratio) as u32).max(1), size)
    } else {
        (size, ((size as f32 * ratio) as u32).max(1))
    };
    fcs_utils::resize_image(image, w, h, FilterType::Triangle)
}

#[inline(never)]
fn preprocess(image: &DynamicImage, size: u32, mode: Mode) -> Vec<f32> {
    if matches!(mode, Mode::Rows) {
        scrfd::preprocess(image, size).0
    } else {
        normalise(&resize(image, size), size, mode)
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn compare(label: &str, modes: &[Mode], repeats: usize, mut run: impl FnMut(Mode)) {
    for _ in 0..3 {
        for &mode in modes {
            run(mode);
        }
    }
    let mut samples = vec![Vec::new(); modes.len()];
    for round in 0..31 {
        // Reverse order every pair, and rotate the first entry to spread drift.
        for offset in 0..modes.len() {
            let i = if round % 2 == 0 {
                offset
            } else {
                modes.len() - 1 - offset
            };
            let i = (i + round / 2) % modes.len();
            let start = Instant::now();
            for _ in 0..repeats {
                run(modes[i]);
            }
            samples[i].push(start.elapsed().as_secs_f64() * 1e6 / repeats as f64);
        }
    }
    println!("  {label} (31 alternating samples, {repeats} calls/sample):");
    for (i, mode) in modes.iter().enumerate() {
        let mut ratios: Vec<_> = samples[0]
            .iter()
            .zip(&samples[i])
            .map(|(a, b)| a / b)
            .collect();
        let speedup = median(&mut ratios);
        let micros = median(&mut samples[i].clone());
        println!(
            "    {mode:?}: {micros:9.2} us, {speedup:.3}x, ratio IQR {:.3}..{:.3}",
            ratios[7], ratios[23]
        );
    }
}

fn check() {
    for width in 0..65 {
        for offset in 0..4 {
            let bytes: Vec<u8> = (0..offset + width * 3).map(|i| (i * 37) as u8).collect();
            let rgb = &bytes[offset..];
            let mut expected = vec![vec![1234.0; width + 2]; 3];
            for (c, channel) in expected.iter_mut().enumerate() {
                for x in 0..width {
                    channel[x + 1] = (rgb[x * 3 + c] as f32 - 127.5) / 128.0;
                }
            }
            for mode in [Mode::Rows, Mode::Intrinsics, Mode::Asm] {
                let [mut r, mut g, mut b] = std::array::from_fn(|_| vec![1234.0; width + 2]);
                row(
                    rgb,
                    &mut r[1..width + 1],
                    &mut g[1..width + 1],
                    &mut b[1..width + 1],
                    mode,
                );
                assert_eq!(
                    expected,
                    [r, g, b],
                    "row {mode:?}, width {width}, offset {offset}"
                );
            }
        }
    }
    // Every byte value, both vector and scalar tails, and untouched letterbox padding.
    for size in [1_u32, 7, 8, 9, 15, 16, 17, 33, 256] {
        for (w, h) in [
            (size, size),
            (size, 1),
            (1, size),
            (size.saturating_sub(1).max(1), size),
        ] {
            let rgb = RgbImage::from_fn(w, h, |x, y| {
                image::Rgb([
                    (x + y * 17) as u8,
                    (x * 31 + y) as u8,
                    (x * 7 + y * 43) as u8,
                ])
            });
            let expected = normalise(&rgb, size, Mode::Legacy);
            for mode in [Mode::Rows, Mode::Intrinsics, Mode::Asm] {
                assert_eq!(
                    expected,
                    normalise(&rgb, size, mode),
                    "{mode:?}, {w}x{h}/{size}"
                );
            }
            let image = DynamicImage::ImageRgb8(rgb);
            let expected = scrfd::preprocess(&image, size).0;
            for mode in [Mode::Rows, Mode::Intrinsics, Mode::Asm] {
                assert_eq!(
                    expected,
                    preprocess(&image, size, mode),
                    "preprocess {mode:?}"
                );
            }
        }
    }
    println!("Correctness: exact equality, all byte values, short rows, tails and padding passed.");
}

fn main() -> Result<()> {
    #[cfg(not(target_arch = "x86_64"))]
    anyhow::bail!("This probe requires x86_64; production remains portable.");
    #[cfg(target_arch = "x86_64")]
    anyhow::ensure!(
        std::is_x86_feature_detected!("avx2"),
        "This probe requires AVX2"
    );
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::__cpuid;
        if __cpuid(0x8000_0000).eax >= 0x8000_0004 {
            let brand: Vec<_> = (0x8000_0002..=0x8000_0004)
                .flat_map(|leaf| {
                    let r = __cpuid(leaf);
                    [r.eax, r.ebx, r.ecx, r.edx].map(u32::to_le_bytes).concat()
                })
                .collect();
            println!(
                "CPU: {}",
                String::from_utf8_lossy(&brand)
                    .trim_matches(char::from(0))
                    .trim()
            );
        }
    }
    check();
    println!("Rayon threads: {}", rayon::current_num_threads());
    let modes = [Mode::Legacy, Mode::Rows, Mode::Intrinsics, Mode::Asm];
    let mut images = vec![(
        "synthetic 640x640".to_owned(),
        DynamicImage::ImageRgb8(RgbImage::from_fn(640, 640, |x, y| {
            image::Rgb([x as u8, y as u8, (x + y) as u8])
        })),
    )];
    for path in std::env::args().skip(1) {
        images.push((path.clone(), fcs_utils::load_image(path)?));
    }
    let model =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../models/scrfd80k_500m_640.onnx");
    let weights = scrfd::plan::ScrfdWeights::load(&model)?;
    for (name, image) in images {
        println!("\n{name} ({}x{}):", image.width(), image.height());
        let resized = resize(&image, SIZE);
        let expected = scrfd::preprocess(&image, SIZE).0;
        for mode in modes {
            assert_eq!(expected, preprocess(&image, SIZE, mode), "{name}, {mode:?}");
        }
        compare(
            "A/A conversion control",
            &[Mode::Legacy, Mode::Legacy],
            8,
            |mode| {
                black_box(normalise(black_box(&resized), SIZE, mode));
            },
        );
        compare("conversion + allocation + padding", &modes, 8, |mode| {
            black_box(normalise(black_box(&resized), SIZE, mode));
        });
        compare("full preprocessing", &modes, 3, |mode| {
            black_box(preprocess(black_box(&image), SIZE, mode));
        });
        // Exact tensor equality makes the graph input identical; also check its nine heads.
        let infer = |mode| {
            let input =
                Tensor::new(1, 3, 640, 640, preprocess(black_box(&image), SIZE, mode)).unwrap();
            scrfd::plan::run(input, &weights).unwrap()
        };
        let reference = infer(Mode::Legacy);
        for mode in &modes[1..] {
            let got = infer(*mode);
            for (a, b) in reference.iter().zip(&got) {
                assert_eq!(a.data(), b.data(), "network heads: {name}, {mode:?}");
            }
        }
        compare(
            "preprocessing + full CPU graph (no decode/export)",
            &modes,
            1,
            |mode| {
                black_box(infer(mode));
            },
        );
    }
    Ok(())
}
