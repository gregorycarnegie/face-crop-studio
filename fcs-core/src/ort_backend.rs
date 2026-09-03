//! ONNX Runtime inference backend.
//!
//! `tract` lowers YuNet's 3x3 depthwise convolutions to a scalar fallback — it
//! only unrolls zones with at most 4 taps — which is 59-61% of a CPU detection
//! (measured in 1.5.3). ONNX Runtime vectorises them, which is worth roughly
//! 10x on the inference stage.
//!
//! Finding and validating the runtime, and the FFI itself, are `fcs-ort`'s job.
//! This module only adapts between tract tensors and that binding.

use anyhow::{Context, Result};

use crate::tensor::Tensor;

use crate::{
    model_config::{OUTPUTS_PER_STRIDE, STRIDES},
    preprocess::InputSize,
};

/// A loaded model running on ONNX Runtime.
///
/// No lock and no session pool: ONNX Runtime permits concurrent runs on one
/// session, which `fcs-ort`'s `concurrent_runs_match_sequential` checks rather
/// than assumes. Batch export therefore shares a single session across rayon
/// workers, with the runtime's own threadpool kept to one thread per run so it
/// does not fight rayon for cores.
#[derive(Debug)]
pub(crate) struct OrtBackend {
    session: fcs_ort::Session,
}

impl OrtBackend {
    pub(crate) fn load(
        environment: &std::sync::Arc<fcs_ort::Environment>,
        model_path: &std::path::Path,
    ) -> Result<Self> {
        let session =
            fcs_ort::Session::new(environment, model_path, fcs_ort::SessionOptions::default())
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(Self { session })
    }

    /// Run the graph and return the head outputs as tract tensors, so both
    /// backends share `decode_yunet_outputs` rather than reimplementing it.
    pub(crate) fn run(&self, input: &Tensor, input_size: InputSize) -> Result<Vec<Tensor>> {
        let data = input.as_slice();
        let shape = [
            1usize,
            3,
            input_size.height as usize,
            input_size.width as usize,
        ];

        let outputs = self
            .session
            .run(data, &shape)
            .map_err(|e| anyhow::anyhow!("ONNX Runtime inference failed: {e}"))?;

        let expected = STRIDES.len() * OUTPUTS_PER_STRIDE;
        anyhow::ensure!(
            outputs.len() == expected,
            "ONNX Runtime returned {} outputs, expected {expected}",
            outputs.len()
        );

        outputs
            .into_iter()
            .enumerate()
            .map(|(i, out)| {
                Tensor::from_vec(&out.shape, out.data)
                    .with_context(|| format!("output {i} to tensor"))
            })
            .collect()
    }
}
