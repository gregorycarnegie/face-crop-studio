//! What does a detection actually cost, and on which engine?
//!
//! The GPU number was never measured honestly. The old figures timed the call that *records*
//! GPU work, not the work: wgpu queues dispatches and returns, so a timer stopped before the
//! results are read measures the driver accepting commands. Anything downstream of that is an
//! artefact, which is why this waits for the readback inside the timed region -- the same thing
//! `ScrfdDetector::detect` has to do before it can decode.
//!
//! Three questions, in order of how much they change what to do next:
//!
//! 1. **Which engine is fastest for the network alone?** ONNX Runtime, the WGSL kernels, or the
//!    built-in CPU graph, on identical input.
//! 2. **How much of a detection is not the network?** Preprocessing resizes the source down to
//!    640x640 on the CPU, single-threaded, and a 36 MP camera RAW is a lot of pixels to walk.
//!    If that dominates, engine choice barely matters and the resize is the thing to fix.
//! 3. **Does image size change the answer?** The network is fixed-size, so its cost is constant;
//!    preprocessing is not. The two should cross somewhere.
//!
//! Run with a real photograph, because question 2 depends on its dimensions:
//!
//! ```text
//! ORT_DYLIB_PATH=.../onnxruntime.dll cargo run --release -p fcs-core --example engine_speed -- <image>
//! ```

use std::time::Instant;

use anyhow::{Context, Result};
use fcs_core::{
    cpu::tensor::Tensor,
    gpu::{GpuInferenceOps, GpuTensor},
    scrfd::{
        self,
        gpu::{self as scrfd_gpu, ScrfdGpuWeights},
        plan::{self, ScrfdWeights},
    },
};
use fcs_utils::gpu::{GpuAvailability, GpuContext, GpuContextOptions};

#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

const WARMUP: usize = 3;
const RUNS: usize = 15;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Time `f` `RUNS` times after `WARMUP` untimed runs, and report the median in milliseconds.
///
/// Median rather than mean: a single scheduling hiccup or a shader cache miss skews a mean over
/// 15 runs badly, and the question here is what a typical detection costs.
fn time_ms(mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    for _ in 0..WARMUP {
        f()?;
    }
    let mut samples = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let started = Instant::now();
        f()?;
        samples.push(started.elapsed().as_secs_f64() * 1e3);
    }
    Ok(median(samples))
}

fn main() -> Result<()> {
    let model = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .context("fcs-core sits in the workspace root")?
        .join("models/scrfd80k_500m_640.onnx");
    anyhow::ensure!(model.exists(), "no model at {}", model.display());

    let image_path = std::env::args().nth(1);
    let side = scrfd::INPUT_SIZE as usize;

    // --- Question 2: preprocessing, on the image the caller supplied ---
    let input = match &image_path {
        Some(path) => {
            let image = fcs_utils::load_image(path).with_context(|| format!("loading {path}"))?;
            let megapixels = (image.width() as f64 * image.height() as f64) / 1e6;
            let preprocess_ms = time_ms(|| {
                let _ = scrfd::preprocess(&image, scrfd::INPUT_SIZE);
                Ok(())
            })?;
            println!(
                "source {}x{} ({megapixels:.1} MP)\n  preprocess (CPU resize + normalise)  {preprocess_ms:8.2} ms",
                image.width(),
                image.height()
            );
            scrfd::preprocess(&image, scrfd::INPUT_SIZE).0
        }
        None => {
            println!(
                "no image given; timing the network only (pass a path for the preprocess figure)"
            );
            (0..3 * side * side)
                .map(|i| (i as f64 * 0.017).sin() as f32)
                .collect()
        }
    };

    println!("\nnetwork only, {side}x{side} input, median of {RUNS} runs after {WARMUP} warmups:");

    // --- ONNX Runtime ---
    match fcs_ort::Environment::shared() {
        Some(environment) => {
            let session =
                fcs_ort::Session::new(&environment, &model, fcs_ort::SessionOptions::default())?;
            let ms = time_ms(|| {
                session.run(&input, &[1, 3, side, side])?;
                Ok(())
            })?;
            println!("  onnxruntime                          {ms:8.2} ms");
        }
        None => println!("  onnxruntime                            skipped (set ORT_DYLIB_PATH)"),
    }

    // --- WGSL ---
    match GpuContext::init_with_fallback(&GpuContextOptions::default()) {
        GpuAvailability::Available(context) => {
            let ops = GpuInferenceOps::new(context, None)?;
            let weights = ScrfdGpuWeights::load(&ops, &model)?;
            let uploaded = GpuTensor::from_slice(
                ops.context().clone(),
                vec![1, 3, side, side],
                &input,
                Some("input"),
            )?;

            // Recording alone, for the comparison that matters: this is the number the old
            // timings reported, and the gap between it and the next line is how much of the
            // work was still queued when they stopped the clock.
            let record_ms = time_ms(|| {
                scrfd_gpu::run(&ops, &uploaded, &weights)?;
                Ok(())
            })?;

            // The honest figure: read the outputs back, which blocks until the GPU is done.
            // `detect` must do this before it can decode, so this is what a detection pays.
            let ms = time_ms(|| {
                let outputs = scrfd_gpu::run(&ops, &uploaded, &weights)?;
                for tensor in &outputs {
                    tensor.to_vec()?;
                }
                Ok(())
            })?;
            println!("  wgsl, submit only (not a detection)  {record_ms:8.2} ms");
            println!("  wgsl, including readback             {ms:8.2} ms");
        }
        other => println!("  wgsl                                   skipped ({other:?})"),
    }

    // --- Built-in CPU graph ---
    let weights = ScrfdWeights::load(&model)?;
    let ms = time_ms(|| {
        let tensor = Tensor::new(1, 3, side, side, input.clone())?;
        plan::run(tensor, &weights)?;
        Ok(())
    })?;
    println!("  built-in CPU graph                   {ms:8.2} ms");

    // --- End to end, through the detector that ships ---
    if let Some(path) = &image_path {
        let image = fcs_utils::load_image(path)?;
        println!(
            "\nend to end through ScrfdDetector::detect (preprocess + network + decode + NMS):"
        );
        match scrfd::ScrfdDetector::load_from(&model) {
            Some(detector) => {
                let ms = time_ms(|| {
                    detector.detect(&image, 0.5, 0.4)?;
                    Ok(())
                })?;
                println!("  application on {:<12} {ms:8.2} ms", detector.engine());
            }
            None => println!("  application unavailable"),
        }
    }

    // --- Eye refiner: one 112x112 forward pass per face, after detection ---
    let refiner_model = model.with_file_name("eye_refiner.onnx");
    if refiner_model.exists() {
        const SIZE: usize = 112;
        println!(
            "
eye refiner, one face ({SIZE}x{SIZE} input):"
        );
        if let Some(environment) = fcs_ort::Environment::shared() {
            let face: Vec<f32> = (0..3 * SIZE * SIZE)
                .map(|i| (i as f64 * 0.01).sin() as f32)
                .collect();
            let session = fcs_ort::Session::new(
                &environment,
                &refiner_model,
                fcs_ort::SessionOptions::default(),
            )?;
            let ms = time_ms(|| {
                session.run(&face, &[1, 3, SIZE, SIZE])?;
                Ok(())
            })?;
            println!("  onnxruntime, network only            {ms:8.2} ms");
        }
        // Through the shipping API, so this includes the crop resample the app pays too.
        let faces = match (&image_path, scrfd::ScrfdDetector::load_from(&model)) {
            (Some(path), Some(detector)) => {
                let image = fcs_utils::load_image(path)?;
                let found = detector.detect(&image, 0.5, 0.4)?;
                found.first().cloned().map(|face| (image, face))
            }
            _ => None,
        };
        match faces {
            Some((image, face)) => {
                let refiner = fcs_core::eye_refiner::EyeRefiner::load_from(&refiner_model)
                    .context("eye refiner would not load")?;
                let ms = time_ms(|| {
                    refiner.refine(&image, &mut [face.clone()]);
                    Ok(())
                })?;
                println!("  built-in CPU graph, refine()         {ms:8.2} ms");
            }
            None => println!(
                "  built-in CPU graph                     skipped (needs an image with a face)"
            ),
        }
    }

    Ok(())
}
