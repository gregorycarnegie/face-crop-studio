//! CPU inference: the built-in graph against ONNX Runtime, at latency and at throughput.
//!
//! Experiment 69. The backlog calls these "the shipped tract and ONNX Runtime paths", but
//! tract is gone: the built-in path is `CpuGraph`, a pure-Rust YuNet implementation that needs
//! nothing installed. `InferenceBackend::Auto` prefers ONNX Runtime whenever a compatible
//! library is present, and this measures whether that preference is right.
//!
//! Two numbers, because they disagree. **Latency** is one inference at a time, which is what a
//! GUI preview on a CPU-only machine gets. **Throughput** is many images through rayon, which
//! is a folder export. ONNX Runtime's intra-op pool belongs to the session and the session is
//! shared, so raising `intra_threads` adds threads to one pool rather than to each concurrent
//! run: it moves latency and leaves throughput alone. `FCS_ORT_INTRA_THREADS` sweeps it.
//!
//!   cargo run --release -p fcs-core --example cpu_backends -- <image> [reps]

use std::time::Instant;

use anyhow::{Context, Result};
use fcs_core::{InferenceBackend, InputSize, PreprocessConfig, YuNetModel, preprocess_image};
use rayon::prelude::*;

const MODEL: &str = "models/face_detection_yunet_2023mar_640.onnx";

fn percentile(sorted: &[f64], f: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[((sorted.len() as f64 * f) as usize).min(sorted.len() - 1)]
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let image = args.next().expect("usage: cpu_backends <image> [reps]");
    let reps: usize = args.next().map_or(30, |v| v.parse().unwrap_or(30));

    let size = InputSize::new(640, 640);
    let config = PreprocessConfig {
        input_size: size,
        resize_quality: fcs_utils::config::ResizeQuality::Quality,
    };

    // Preprocessed once and reused, so this times inference and nothing around it.
    let prep = preprocess_image(&image, &config).context("preprocess")?;
    println!(
        "{image}\n{}x{} source, {reps} reps, {} rayon threads\n",
        prep.original_size.0,
        prep.original_size.1,
        rayon::current_num_threads()
    );

    // Reports the override only. The default is computed from core count in
    // `fcs_ort::SessionOptions`, so "unset" does not mean 1.
    match std::env::var("FCS_ORT_INTRA_THREADS") {
        Ok(n) => println!("ONNX Runtime intra_threads: {n} (FCS_ORT_INTRA_THREADS)"),
        Err(_) => println!("ONNX Runtime intra_threads: unset, using the built-in default"),
    }

    let mut rows: Vec<(String, f64, f64, f64)> = Vec::new();
    let mut reference: Option<Vec<f32>> = None;

    for (label, backend) in [
        ("cpu-graph", InferenceBackend::CpuGraph),
        ("onnxruntime", InferenceBackend::OnnxRuntime),
    ] {
        let model = match YuNetModel::load_with(MODEL, size, backend) {
            Ok(m) => m,
            Err(e) => {
                println!("{label}: unavailable ({e})");
                continue;
            }
        };

        // Warm: first run pays for lazy allocation and, for ORT, arena setup.
        for _ in 0..3 {
            model.run(prep.tensor.clone())?;
        }

        let mut times = Vec::with_capacity(reps);
        let mut last = None;
        for _ in 0..reps {
            let start = Instant::now();
            let out = model.run(prep.tensor.clone())?;
            times.push(start.elapsed().as_secs_f64() * 1e3);
            last = Some(out);
        }
        times.sort_by(f64::total_cmp);

        // Both backends feed the same decoder, so their raw output must agree.
        if let Some(out) = last {
            let flat: Vec<f32> = out.as_slice().to_vec();
            match &reference {
                None => reference = Some(flat),
                Some(r) => {
                    let worst = r
                        .iter()
                        .zip(&flat)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    println!("  {label} vs cpu-graph: worst output difference {worst:.6}");
                }
            }
        }

        // Throughput: the same inference from every rayon worker at once, which is the shape
        // of a folder export. Reported as images per second so it can be read against latency.
        let jobs = reps.max(rayon::current_num_threads() * 4);
        let start = Instant::now();
        (0..jobs)
            .into_par_iter()
            .try_for_each(|_| model.run(prep.tensor.clone()).map(|_| ()))?;
        let throughput = jobs as f64 / start.elapsed().as_secs_f64();

        rows.push((
            label.to_owned(),
            percentile(&times, 0.5),
            percentile(&times, 0.95),
            throughput,
        ));
    }

    println!(
        "\n{:<14} {:>12} {:>12} {:>16}",
        "backend", "p50 ms", "p95 ms", "parallel img/s"
    );
    for (label, p50, p95, tp) in &rows {
        println!("{label:<14} {p50:>12.2} {p95:>12.2} {tp:>16.1}");
    }
    Ok(())
}
