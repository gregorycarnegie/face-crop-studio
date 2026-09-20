//! Does the built-in CPU graph compute what ONNX Runtime computes?
//!
//! ```text
//! ORT_DYLIB_PATH=.../onnxruntime.dll cargo run -p fcs-core --example scrfd_cpu_parity
//! ```
//!
//! The point of the port is to drop YuNet, whose weights come from WIDER FACE and its
//! non-commercial research licence. That only works if the built-in graph is the same detector,
//! not merely a detector: a topology with one stride or one group count wrong still runs, still
//! produces boxes, and is quietly worse.
//!
//! So this runs the same tensor through both and compares the nine head outputs elementwise.
//! ONNX Runtime is the oracle because it is the path already measured against Python.

use fcs_core::{
    cpu::tensor::Tensor,
    scrfd::{
        self,
        plan::{self, ScrfdWeights},
    },
};

/// Both paths are f32 over the same weights in the same order, so the gap should be rounding.
/// A wrong stride, group count or wiring is off by far more than this.
const TOLERANCE: f32 = 2e-3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Defaults to the named export the topology was generated from. The model shipped in
    // 1.8.0 predates that change and still carries torch's generated names, so loading it here
    // fails by name rather than silently: `models/scrfd80k_500m_640_named.onnx`.
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

    // A real photograph would be better, but any input exercises every weight; the comparison
    // is between two executions of the same graph, not against ground truth.
    let side = scrfd::INPUT_SIZE as usize;
    let input: Vec<f32> = (0..3 * side * side)
        .map(|i| (i as f64 * 0.017).sin() as f32)
        .collect();

    let weights = ScrfdWeights::load(&model)?;
    let started = std::time::Instant::now();
    let ours = plan::run(Tensor::new(1, 3, side, side, input.clone())?, &weights)?;
    let cpu_ms = started.elapsed().as_secs_f64() * 1e3;

    let environment = fcs_ort::Environment::shared()
        .ok_or("no ONNX Runtime; set ORT_DYLIB_PATH so there is something to compare against")?;
    let session = fcs_ort::Session::new(&environment, &model, fcs_ort::SessionOptions::default())?;
    let started = std::time::Instant::now();
    let theirs = session.run(&input, &[1, 3, side, side])?;
    let ort_ms = started.elapsed().as_secs_f64() * 1e3;

    println!("built-in CPU graph {cpu_ms:.0} ms, ONNX Runtime {ort_ms:.0} ms");
    if ours.len() != theirs.len() {
        return Err(format!("{} outputs against {}", ours.len(), theirs.len()).into());
    }

    // ORT returns the deployment layout -- (1, H*W*A, C), sigmoided for the class maps -- while
    // the plan stops at the raw (1, A*C, H, W) convolution outputs, which is what the decoder
    // wants. Comparing means putting ours through the same two steps.
    let mut worst = 0.0f32;
    let mut worst_output = 0;
    for (index, (mine_tensor, theirs)) in ours.iter().zip(&theirs).enumerate() {
        let channels = *theirs.shape.last().unwrap_or(&1);
        let (height, width) = (mine_tensor.height(), mine_tensor.width());
        let anchors = mine_tensor.channels() / channels;
        if mine_tensor.data().len() != theirs.data.len() {
            return Err(format!(
                "output {index}: {} values against {}",
                mine_tensor.data().len(),
                theirs.data.len()
            )
            .into());
        }

        for y in 0..height {
            for x in 0..width {
                for anchor in 0..anchors {
                    for channel in 0..channels {
                        let plane = anchor * channels + channel;
                        let mine = mine_tensor.data()[plane * height * width + y * width + x];
                        let mine = if channels == 1 {
                            1.0 / (1.0 + (-mine).exp()) // the class maps are sigmoided on export
                        } else {
                            mine
                        };
                        let row = (y * width + x) * anchors + anchor;
                        let gap = (mine - theirs.data[row * channels + channel]).abs();
                        if gap > worst {
                            worst = gap;
                            worst_output = index;
                        }
                    }
                }
            }
        }
    }

    println!("worst difference {worst:.3e} (output {worst_output}, tolerance {TOLERANCE:.0e})");
    if worst > TOLERANCE {
        return Err(format!(
            "the built-in graph is not computing the same network: {worst:.3e} apart"
        )
        .into());
    }
    println!("CPU PARITY OK");
    Ok(())
}
