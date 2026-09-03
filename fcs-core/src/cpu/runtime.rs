//! Pure-Rust YuNet inference: load the model once, run it many times.

use std::path::Path;

use crate::tensor::Tensor as CoreTensor;
use anyhow::{Context, Result};

use super::{graph, tensor::Tensor};
use crate::{model::decode_yunet_outputs, preprocess::InputSize};

/// YuNet running entirely on the CPU, with no external runtime and no GPU.
#[derive(Debug)]
pub struct CpuYuNet {
    weights: graph::Weights,
    input_size: InputSize,
}

impl CpuYuNet {
    /// Read the model's weights. The topology itself is compiled in, so only
    /// the initializers are taken from the file.
    pub fn load(model_path: &Path, input_size: InputSize) -> Result<Self> {
        anyhow::ensure!(
            model_path.exists(),
            "model file not found: {}",
            model_path.display()
        );
        Ok(Self {
            weights: graph::Weights::load(model_path).context("loading YuNet weights")?,
            input_size,
        })
    }

    pub fn input_size(&self) -> InputSize {
        self.input_size
    }

    /// Run the network and return the twelve raw head tensors, in the order
    /// `decode_yunet_outputs` expects. `YuNetModel` decodes them itself, the
    /// same way it does for the other backends.
    pub fn head_tensors(&self, input: &CoreTensor) -> Result<Vec<CoreTensor>> {
        let data = input.as_slice();

        let height = self.input_size.height as usize;
        let width = self.input_size.width as usize;
        let tensor = Tensor::new(1, 3, height, width, data.to_vec())
            .context("input tensor does not match the configured input size")?;

        let outputs = graph::forward(&tensor, &self.weights).context("YuNet forward pass")?;
        outputs
            .into_iter()
            .enumerate()
            .map(|(i, out)| {
                CoreTensor::from_vec(&[out.rows, out.channels], out.data)
                    .with_context(|| format!("head output {i} to tensor"))
            })
            .collect()
    }

    /// Run detection and decode, returning the `[N, 15]` rows. Convenience for
    /// callers that are not going through [`crate::YuNetModel`].
    pub fn run(&self, input: CoreTensor) -> Result<CoreTensor> {
        let tensors = self.head_tensors(&input)?;
        decode_yunet_outputs(&tensors, self.input_size)
    }
}
