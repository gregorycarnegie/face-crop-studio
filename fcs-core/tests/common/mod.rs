//! An independent YuNet implementation to check the shipped backends against.
//!
//! `tract` used to be one of the inference backends, so parity tests could ask
//! `YuNetModel` for it. It is no longer in the product — the shipped backends
//! are ONNX Runtime and the built-in graph — but it is still the only
//! implementation available that *interprets* the ONNX file rather than
//! re-encoding YuNet's topology by hand, which is exactly what makes it a
//! useful oracle. Both shipped backends could share a mistake about the
//! architecture; tract reads the architecture from the model.
//!
//! Kept as a dev-dependency so none of it reaches a released binary.

#![allow(dead_code)] // Each integration test binary uses a different subset.

use std::sync::Arc;

use anyhow::Result;
use fcs_core::{Detection, PostprocessConfig, apply_postprocess};
use tract_onnx::prelude::*;

/// YuNet running under tract, producing decoded detections.
pub struct TractOracle {
    plan: Arc<TypedRunnableModel>,
    input_size: (usize, usize),
}

impl TractOracle {
    pub fn load(model_path: &std::path::Path, width: usize, height: usize) -> Result<Self> {
        let plan = tract_onnx::onnx()
            .model_for_path(model_path)
            .map_err(|e| anyhow::anyhow!("parse ONNX: {e}"))?
            .into_optimized()
            .map_err(|e| anyhow::anyhow!("optimize: {e}"))?
            .into_runnable()
            .map_err(|e| anyhow::anyhow!("plan: {e}"))?;
        Ok(Self {
            plan,
            input_size: (width, height),
        })
    }

    /// Run the graph on a preprocessed BGR CHW buffer and decode the result.
    ///
    /// The decode and NMS come from `fcs-core` deliberately: the point of
    /// comparison is the inference, and sharing everything downstream keeps a
    /// difference attributable to the backend rather than to the decoder.
    pub fn detect(
        &self,
        input: &[f32],
        scale_x: f32,
        scale_y: f32,
        post: &PostprocessConfig,
    ) -> Result<Vec<Detection>> {
        let (w, h) = self.input_size;
        let array = tract_ndarray::Array4::from_shape_vec((1, 3, h, w), input.to_vec())
            .map_err(|e| anyhow::anyhow!("input shape: {e}"))?;
        let outputs = self
            .plan
            .run(tvec!(array.into_tensor().into()))
            .map_err(|e| anyhow::anyhow!("tract run: {e}"))?;

        let tensors: Vec<fcs_core::tensor::Tensor> = outputs
            .into_iter()
            .map(|value| {
                let t = value.into_tensor();
                let shape = t.shape().to_vec();
                let data = t
                    .into_plain_array::<f32>()
                    .map_err(|e| anyhow::anyhow!("output is not f32: {e}"))?
                    .into_raw_vec_and_offset()
                    .0;
                fcs_core::tensor::Tensor::from_vec(&shape, data)
            })
            .collect::<Result<_>>()?;

        let decoded =
            fcs_core::decode_yunet_outputs(&tensors, fcs_core::InputSize::new(w as u32, h as u32))?;
        apply_postprocess(&decoded, scale_x, scale_y, post)
    }
}
