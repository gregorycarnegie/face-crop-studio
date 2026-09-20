//! Do all three engines compute the same SCRFD?
//!
//! ```text
//! ORT_DYLIB_PATH=.../onnxruntime.dll cargo run --release -p fcs-core --example scrfd_gpu_parity
//! ```
//!
//! The WGSL port is the last thing standing between the packages and dropping YuNet, whose
//! weights come from WIDER FACE under a non-commercial research licence. It only counts if the
//! GPU path is the same network: a wrong stride or group count still runs and is quietly worse.
//!
//! ONNX Runtime is the oracle, and the CPU graph -- already matched to 1.2e-05 -- is the
//! second opinion. Timings are printed because the trade this decides is licence against GPU
//! speed, but they are single runs and not a benchmark.

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

/// Wider than the CPU tolerance: the GPU sums in a different order, and 60 layers of f32
/// accumulate that. A wrong topology is off by far more.
const TOLERANCE: f32 = 5e-3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::args()
        .nth(1)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("fcs-core sits in the workspace root")
                .join("models/scrfd80k_500m_640_named.onnx")
        });
    if !model.exists() {
        return Err(format!("no model at {}", model.display()).into());
    }

    let side = scrfd::INPUT_SIZE as usize;
    let input: Vec<f32> = (0..3 * side * side)
        .map(|i| (i as f64 * 0.017).sin() as f32)
        .collect();

    // --- GPU ---
    let context = match GpuContext::init_with_fallback(&GpuContextOptions::default()) {
        GpuAvailability::Available(context) => context,
        GpuAvailability::Disabled { reason } => {
            return Err(format!("GPU disabled: {reason}").into());
        }
        GpuAvailability::Unavailable { error } => return Err(format!("no GPU: {error}").into()),
    };
    let ops = GpuInferenceOps::new(context, None)?;
    let weights = ScrfdGpuWeights::load(&ops, &model)?;
    let uploaded = GpuTensor::from_slice(
        ops.context().clone(),
        vec![1, 3, side, side],
        &input,
        Some("input"),
    )?;

    // Once to warm up (pipelines compile on first use), then the one that is timed.
    let _ = scrfd_gpu::run(&ops, &uploaded, &weights)?;
    let started = std::time::Instant::now();
    let gpu_outputs = scrfd_gpu::run(&ops, &uploaded, &weights)?;
    let gpu_ms = started.elapsed().as_secs_f64() * 1e3;

    // --- CPU ---
    let cpu_weights = ScrfdWeights::load(&model)?;
    let started = std::time::Instant::now();
    let cpu_outputs = plan::run(Tensor::new(1, 3, side, side, input.clone())?, &cpu_weights)?;
    let cpu_ms = started.elapsed().as_secs_f64() * 1e3;

    // --- ONNX Runtime ---
    let environment =
        fcs_ort::Environment::shared().ok_or("no ONNX Runtime; set ORT_DYLIB_PATH")?;
    let session = fcs_ort::Session::new(&environment, &model, fcs_ort::SessionOptions::default())?;
    let started = std::time::Instant::now();
    let ort_outputs = session.run(&input, &[1, 3, side, side])?;
    let ort_ms = started.elapsed().as_secs_f64() * 1e3;

    println!("WGSL {gpu_ms:.0} ms | built-in CPU {cpu_ms:.0} ms | ONNX Runtime {ort_ms:.0} ms");

    let mut worst_vs_ort = 0.0f32;
    let mut worst_vs_cpu = 0.0f32;
    for (index, ort) in ort_outputs.iter().enumerate() {
        let channels = *ort.shape.last().unwrap_or(&1);
        let mine = gpu_outputs[index].to_vec()?;
        let dims = gpu_outputs[index].shape().dims().to_vec();
        let (height, width) = (dims[2], dims[3]);
        let anchors = dims[1] / channels;
        let cpu = cpu_outputs[index].data();

        for y in 0..height {
            for x in 0..width {
                for anchor in 0..anchors {
                    for channel in 0..channels {
                        let plane = anchor * channels + channel;
                        let at = plane * height * width + y * width + x;
                        // The export sigmoids the class maps; the raw maps here do not.
                        let ours = if channels == 1 {
                            1.0 / (1.0 + (-mine[at]).exp())
                        } else {
                            mine[at]
                        };
                        let row = (y * width + x) * anchors + anchor;
                        worst_vs_ort =
                            worst_vs_ort.max((ours - ort.data[row * channels + channel]).abs());
                        worst_vs_cpu = worst_vs_cpu.max((mine[at] - cpu[at]).abs());
                    }
                }
            }
        }
    }

    println!("worst difference against ONNX Runtime {worst_vs_ort:.3e}");
    println!("worst difference against the CPU graph {worst_vs_cpu:.3e}");
    if worst_vs_ort > TOLERANCE {
        return Err(
            format!("the WGSL graph disagrees with ONNX Runtime by {worst_vs_ort:.3e}").into(),
        );
    }
    println!("GPU PARITY OK");
    Ok(())
}
