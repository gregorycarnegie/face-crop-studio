//! Bit-exact fingerprint of the raw inference output, for the readback experiments (11-18).
//!
//! Every candidate in that group changes when the host looks at head memory, not what the
//! GPU writes into it, so the check that matters is whether the bytes come back identical.
//! A deterministic input goes in and a fingerprint of every output float comes out; run the
//! probe under each candidate and diff the lines. A partially-mapped or recycled buffer
//! changes the fingerprint, where final detections might survive it through NMS.
//!
//! Run with: cargo run --release -p fcs-core --example readback_parity

use anyhow::{Context, Result};
use fcs_core::{InputSize, gpu::GpuYuNet};
use fcs_utils::gpu::{GpuAvailability, GpuContext, GpuContextOptions};

const MODEL: &str = "models/face_detection_yunet_2023mar_640.onnx";
const INPUT: InputSize = InputSize::new(640, 640);
const RUNS: usize = 5;

fn main() -> Result<()> {
    let context = match GpuContext::init_with_fallback(&GpuContextOptions::default()) {
        GpuAvailability::Available(ctx) => ctx,
        GpuAvailability::Disabled { reason } => anyhow::bail!("GPU disabled: {reason}"),
        GpuAvailability::Unavailable { error } => anyhow::bail!("GPU unavailable: {error}"),
    };
    let model_path = fcs_utils::model_path(MODEL)
        .context("resolve YuNet model")?
        .with_context(|| format!("model {MODEL} not found"))?;
    let model = GpuYuNet::with_context(context, &model_path, INPUT).context("build GPU YuNet")?;

    let input = model.allocate_input(INPUT).context("allocate input")?;
    let pixels = INPUT.width as usize * INPUT.height as usize;
    let data: Vec<f32> = (0..pixels * 3).map(|i| (i % 257) as f32 / 256.0).collect();
    input.write(&data).context("upload input")?;

    // Repeated so an intermittent race shows up as a differing line rather than a clean pass.
    for run in 0..RUNS {
        let raw = model.run_on_device(&input).context("inference")?;
        let values = raw.as_slice();
        // FNV-1a over the raw bits: order-sensitive, and NaN-safe in a way summing is not.
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for v in values {
            for byte in v.to_bits().to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        let finite = values.iter().filter(|v| v.is_finite()).count();
        println!(
            "run {run}: shape {:?} elements {} finite {} fnv1a {hash:#018x}",
            raw.shape(),
            values.len(),
            finite
        );
    }
    Ok(())
}
