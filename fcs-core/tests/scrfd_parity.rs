//! Do all three engines compute the same SCRFD?
//!
//! This is what replaced the YuNet parity suite. Those tests had a better oracle: tract
//! *interpreted* the ONNX file, so it could not share a misreading of the architecture with
//! code that re-encodes the topology by hand. Nothing like that is left -- tract went with
//! YuNet -- so the oracle here is ONNX Runtime, which is a different implementation of the
//! same graph but not an independent reading of it. What that still catches is the thing the
//! port is actually prone to: a stride, group count or wiring that is wrong in the hand-written
//! topology, which changes the numbers by far more than rounding.
//!
//! ```text
//! ORT_DYLIB_PATH=.../onnxruntime.dll cargo test -p fcs-core --test scrfd_parity
//! ```
//!
//! Two kinds of "cannot run here", deliberately treated differently:
//!
//! * **The model or ONNX Runtime is missing.** CI provides both, so this is a misconfiguration
//!   rather than a fact about the machine, and `FCS_STRICT_TESTS=1` turns the skip into a
//!   failure so a leg cannot report a pass having checked nothing.
//! * **There is no GPU adapter.** That is a fact about the machine, and CI's Linux runners do
//!   not have one, so it always skips -- strict or not. Making it strict-failable turned the
//!   Linux legs red for having no graphics card, which is not a defect in this code.

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

/// Both paths are f32 over the same weights in the same order, so the gap should be rounding.
const CPU_TOLERANCE: f32 = 2e-3;

/// Wider: the GPU sums in a different order, and 60 layers of f32 accumulate that.
const GPU_TOLERANCE: f32 = 5e-3;

/// Fail instead of skipping, for CI.
fn strict() -> bool {
    std::env::var("FCS_STRICT_TESTS").is_ok_and(|v| v != "0" && !v.is_empty())
}

/// Skip, or fail when strict. For things CI supplies: the model, and ONNX Runtime.
macro_rules! skip_unless_strict {
    ($($arg:tt)*) => {{
        if strict() {
            panic!("FCS_STRICT_TESTS: {}", format!($($arg)*));
        }
        eprintln!("skipped: {}", format!($($arg)*));
        return;
    }};
}

/// Skip regardless of strictness. For hardware the machine either has or does not.
macro_rules! skip_without_gpu {
    ($($arg:tt)*) => {{
        eprintln!("skipped (no GPU, not a failure): {}", format!($($arg)*));
        return;
    }};
}

/// One GPU context for this whole test binary, or `None` without a usable adapter.
///
/// **Cached, and that is not an optimisation.** A test that opens its own device costs one
/// device per test, `cargo test` starts them together, and cargo-mutants multiplies that by its
/// job count -- which is how the NVIDIA driver wedges (experiment 94; it once produced 86
/// meaningless timeouts in four hours). `fcs_core::gpu::test_context` does the same thing for
/// the in-crate tests but is `cfg(test)` and so unreachable from here.
///
/// Also keeps both tests answering "is there an adapter" the same way, and reports the reason
/// rather than swallowing it.
fn gpu_context() -> Option<std::sync::Arc<GpuContext>> {
    static CONTEXT: std::sync::OnceLock<Option<std::sync::Arc<GpuContext>>> =
        std::sync::OnceLock::new();
    CONTEXT
        .get_or_init(
            || match GpuContext::init_with_fallback(&GpuContextOptions::default()) {
                GpuAvailability::Available(context) => Some(context),
                GpuAvailability::Disabled { reason } => {
                    eprintln!("no GPU: disabled ({reason})");
                    None
                }
                GpuAvailability::Unavailable { error } => {
                    eprintln!("no GPU: unavailable ({error})");
                    None
                }
            },
        )
        .clone()
}

fn model_path() -> Option<std::path::PathBuf> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("fcs-core sits in the workspace root")
        .join("models/scrfd80k_500m_640.onnx");
    path.exists().then_some(path)
}

/// Any input exercises every weight; the comparison is between executions of one graph, not
/// against ground truth.
fn synthetic_input(side: usize) -> Vec<f32> {
    (0..3 * side * side)
        .map(|i| (i as f64 * 0.017).sin() as f32)
        .collect()
}

/// Walk two head outputs together and return the worst absolute difference.
///
/// ONNX Runtime returns the deployment layout -- `(1, H*W*A, C)`, class maps already sigmoided
/// -- while both built-in engines stop at the raw `(1, A*C, H, W)` convolution outputs, which is
/// what the decoder wants. Comparing means putting ours through the same two steps; getting
/// this transpose wrong was one of the traps during the port.
fn worst_against_ort(ours: &[f32], dims: [usize; 4], ort: &fcs_ort::OutputTensor) -> f32 {
    let channels = *ort.shape.last().unwrap_or(&1);
    let (height, width) = (dims[2], dims[3]);
    let anchors = dims[1] / channels;
    let mut worst = 0.0f32;
    for y in 0..height {
        for x in 0..width {
            for anchor in 0..anchors {
                for channel in 0..channels {
                    let plane = anchor * channels + channel;
                    let at = plane * height * width + y * width + x;
                    let mine = if channels == 1 {
                        1.0 / (1.0 + (-ours[at]).exp()) // the class maps are sigmoided on export
                    } else {
                        ours[at]
                    };
                    let row = (y * width + x) * anchors + anchor;
                    worst = worst.max((mine - ort.data[row * channels + channel]).abs());
                }
            }
        }
    }
    worst
}

fn ort_outputs(
    model: &std::path::Path,
    input: &[f32],
    side: usize,
) -> Option<Vec<fcs_ort::OutputTensor>> {
    let environment = fcs_ort::Environment::shared()?;
    let session =
        fcs_ort::Session::new(&environment, model, fcs_ort::SessionOptions::default()).ok()?;
    session.run(input, &[1, 3, side, side]).ok()
}

#[test]
fn the_builtin_cpu_graph_matches_onnx_runtime() {
    let Some(model) = model_path() else {
        skip_unless_strict!("no model at models/scrfd80k_500m_640.onnx");
    };
    let side = scrfd::INPUT_SIZE as usize;
    let input = synthetic_input(side);

    let Some(theirs) = ort_outputs(&model, &input, side) else {
        skip_unless_strict!("no ONNX Runtime; set ORT_DYLIB_PATH");
    };

    let weights = ScrfdWeights::load(&model).expect("weights load by name");
    let run = || {
        plan::run(
            Tensor::new(1, 3, side, side, input.clone()).expect("input tensor"),
            &weights,
        )
        .expect("the CPU graph runs")
    };
    let ours = run();
    // The second run computes into the first run's recycled buffers, so stale contents or a
    // buffer handed out twice would show here and not on a fresh start.
    let again = run();
    for (index, (first, second)) in ours.iter().zip(&again).enumerate() {
        assert_eq!(
            first.data(),
            second.data(),
            "output {index} changed on a rerun"
        );
    }

    assert_eq!(
        ours.len(),
        theirs.len(),
        "{} outputs against {}",
        ours.len(),
        theirs.len()
    );

    let mut worst = 0.0f32;
    for (index, (mine, theirs)) in ours.iter().zip(&theirs).enumerate() {
        assert_eq!(
            mine.data().len(),
            theirs.data.len(),
            "output {index}: {} values against {}",
            mine.data().len(),
            theirs.data.len()
        );
        let dims = [mine.batch(), mine.channels(), mine.height(), mine.width()];
        worst = worst.max(worst_against_ort(mine.data(), dims, theirs));
    }

    assert!(
        worst <= CPU_TOLERANCE,
        "the built-in graph is not computing the same network: {worst:.3e} apart"
    );
    println!("CPU worst difference {worst:.3e} (tolerance {CPU_TOLERANCE:.0e})");
}

#[test]
fn the_wgsl_engine_matches_onnx_runtime() {
    let Some(model) = model_path() else {
        skip_unless_strict!("no model at models/scrfd80k_500m_640.onnx");
    };
    let side = scrfd::INPUT_SIZE as usize;
    let input = synthetic_input(side);

    let Some(theirs) = ort_outputs(&model, &input, side) else {
        skip_unless_strict!("no ONNX Runtime; set ORT_DYLIB_PATH");
    };

    let Some(context) = gpu_context() else {
        skip_without_gpu!("the WGSL engine needs an adapter");
    };
    let ops = GpuInferenceOps::new(context, None).expect("build ops");
    let weights = ScrfdGpuWeights::load(&ops, &model).expect("weights upload by name");
    let uploaded = GpuTensor::from_slice(
        ops.context().clone(),
        vec![1, 3, side, side],
        &input,
        Some("input"),
    )
    .expect("upload input");
    let ours = scrfd_gpu::run(&ops, &uploaded, &weights).expect("the WGSL graph runs");

    assert_eq!(ours.len(), theirs.len(), "output count");

    let mut worst = 0.0f32;
    for (mine, theirs) in ours.iter().zip(&theirs) {
        let dims = mine.shape().dims();
        let dims = [dims[0], dims[1], dims[2], dims[3]];
        let data = mine.to_vec().expect("read back");
        worst = worst.max(worst_against_ort(&data, dims, theirs));
    }

    assert!(
        worst <= GPU_TOLERANCE,
        "the WGSL graph disagrees with ONNX Runtime by {worst:.3e}"
    );
    println!("GPU worst difference {worst:.3e} (tolerance {GPU_TOLERANCE:.0e})");
}

/// Concurrent inference on shared GPU ops must match sequential inference, bit for bit.
///
/// This replaces `concurrent_inference_matches_sequential`, which ran YuNet's WGSL graph and went
/// with it. The race it guards is in shared infrastructure, not in either topology: buffers
/// released while a command encoder was still being built went straight back to `GpuBufferPool`,
/// so another rayon worker could encode into memory the first submission still referenced. That
/// shipped once, varying about 1% of a 1239-image batch, and `GpuBufferPool::execution_scope` is
/// what fixes it. Batch export really does share one set of ops across rayon workers.
///
/// It compares the nine head tensors rather than decoded detections deliberately: a synthetic
/// input finds no faces, so comparing detections would be comparing two empty lists and passing
/// whatever the engine did. Every value in every output is checked here instead.
#[test]
fn concurrent_gpu_inference_matches_sequential() {
    use rayon::prelude::*;

    let Some(model) = model_path() else {
        skip_unless_strict!("no model at models/scrfd80k_500m_640.onnx");
    };
    let Some(context) = gpu_context() else {
        skip_without_gpu!("the race this guards is in the GPU buffer pool");
    };
    let ops = GpuInferenceOps::new(context, None).expect("build ops");
    let weights = ScrfdGpuWeights::load(&ops, &model).expect("weights upload by name");

    let side = scrfd::INPUT_SIZE as usize;
    let input = synthetic_input(side);
    let run_once = || {
        // A fresh upload per run, as batch export does: it is the pool churn of inputs and
        // intermediates being allocated and released concurrently that exposed the race.
        let uploaded = GpuTensor::from_slice(
            ops.context().clone(),
            vec![1, 3, side, side],
            &input,
            Some("input"),
        )
        .expect("upload input");
        scrfd_gpu::run(&ops, &uploaded, &weights)
            .expect("the WGSL graph runs")
            .iter()
            .map(|tensor| tensor.to_vec().expect("read back"))
            .collect::<Vec<Vec<f32>>>()
    };

    let sequential = run_once();
    let values: usize = sequential.iter().map(Vec::len).sum();
    assert!(values > 0, "no head values were produced");

    let runs: Vec<_> = (0..8).into_par_iter().map(|_| run_once()).collect();

    for (worker, run) in runs.iter().enumerate() {
        assert_eq!(run.len(), sequential.len(), "worker {worker}: output count");
        for (index, (mine, baseline)) in run.iter().zip(&sequential).enumerate() {
            // Bit-identical, not approximate: each shader invocation evaluates its loops in a
            // fixed order, so there is no wobble to absorb and any difference is the race.
            assert_eq!(
                mine, baseline,
                "worker {worker}, output {index}: differs from the sequential run"
            );
        }
    }
    println!("8 concurrent runs matched the sequential one across {values} head values");
}
