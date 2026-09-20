//! Running SCRFD's topology on the built-in CPU graph, with no ONNX Runtime.
//!
//! This is what lets the better detector ship without a runtime, and so what lets YuNet --
//! whose weights come from WIDER FACE, "non-commercial academic research only" (DATA_CARD.md)
//! -- be removed from the packages entirely. Until it exists, a machine without ONNX Runtime
//! falls back to YuNet and the licence question comes back with it.
//!
//! The steps come from [`super::topology`], generated from the exported graph rather than
//! transcribed: 60 convolutions, 4 adds and 2 upsamples. Each step writes one slot and reads
//! earlier ones by index, so execution is a single pass over an array with no graph walking.
//!
//! What it deliberately does not do is interpret ONNX. The weights are read by name through
//! the same reader `crate::yunet` uses; everything about the shape of the network is compiled
//! in. A model with a different architecture will fail to load rather than half-run.

use anyhow::{Context, Result};

use crate::{
    cpu::{
        conv2d::{Activation, ConvConfig, ConvWeights, conv2d},
        ops::{add, resize2x},
        tensor::Tensor,
    },
    yunet::onnx::OnnxInitializerMap,
};

/// One operation, writing one slot.
#[derive(Debug, Clone, Copy)]
pub enum Step {
    /// Convolution, with the ReLU that follows it folded in when there is one.
    Conv {
        /// Slot holding the input tensor.
        input: usize,
        /// Initializer name of the weights.
        weight: &'static str,
        /// Initializer name of the bias, empty when the convolution has none.
        bias: &'static str,
        /// Output channels.
        out_channels: usize,
        /// Input channels per group: equal to `out_channels / groups` for depthwise.
        in_per_group: usize,
        /// Square kernel side.
        kernel: usize,
        /// Stride, 1 or 2 in this network.
        stride: usize,
        /// Symmetric padding.
        pad: usize,
        /// 1 for dense, `channels` for depthwise.
        groups: usize,
        /// Whether a ReLU follows.
        relu: bool,
    },
    /// Elementwise sum of two slots, as the neck's lateral joins.
    Add {
        /// First operand's slot.
        a: usize,
        /// Second operand's slot.
        b: usize,
    },
    /// Nearest-neighbour 2x upsample, which is what the neck's `Resize` nodes are here.
    Upsample2x {
        /// Slot holding the input tensor.
        input: usize,
    },
}

/// The network's weights, read once by name and held in step order.
#[derive(Debug)]
pub struct ScrfdWeights {
    convs: Vec<Option<ConvWeights>>,
}

impl ScrfdWeights {
    /// Read every weight the topology names out of an exported model.
    ///
    /// Fails with the missing name rather than a shape error twenty layers later, which is the
    /// symptom of an export whose names moved -- the reason `export_scrfd.py` folds BatchNorm
    /// itself instead of letting torch rename everything.
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let mut names = Vec::new();
        for step in super::topology::STEPS {
            if let Step::Conv { weight, bias, .. } = step {
                names.push(weight);
                if !bias.is_empty() {
                    names.push(bias);
                }
            }
        }
        let map = OnnxInitializerMap::load(path, &names)
            .with_context(|| format!("reading SCRFD weights from {}", path.display()))?;

        let mut convs = Vec::with_capacity(super::topology::STEPS.len());
        for step in super::topology::STEPS {
            convs.push(match step {
                Step::Conv {
                    weight,
                    bias,
                    out_channels,
                    in_per_group,
                    kernel,
                    ..
                } => {
                    let tensor = map.tensor(weight)?;
                    let expected = out_channels * in_per_group * kernel * kernel;
                    anyhow::ensure!(
                        tensor.data().len() == expected,
                        "{weight} holds {} values, the topology expects {expected}",
                        tensor.data().len()
                    );
                    let bias = if bias.is_empty() {
                        vec![0.0; out_channels]
                    } else {
                        map.tensor(bias)?.data().to_vec()
                    };
                    Some(ConvWeights {
                        out_channels,
                        in_channels_per_group: in_per_group,
                        kernel_h: kernel,
                        kernel_w: kernel,
                        data: tensor.data().to_vec(),
                        bias,
                    })
                }
                _ => None,
            });
        }
        Ok(Self { convs })
    }
}

/// Run every step and return the nine head outputs, in the topology's output order.
///
/// Slots are kept for the whole pass rather than freed once consumed: the neck reads backbone
/// levels produced long before it, so liveness is not simply "the previous step".
pub fn run(input: Tensor, weights: &ScrfdWeights) -> Result<Vec<Tensor>> {
    let mut slots: Vec<Option<Tensor>> = Vec::with_capacity(super::topology::STEPS.len() + 1);
    slots.push(Some(input));

    for (index, step) in super::topology::STEPS.iter().enumerate() {
        let produced = match *step {
            Step::Conv {
                input,
                stride,
                pad,
                groups,
                relu,
                ..
            } => {
                let source = slot(&slots, input, index)?;
                let config = ConvConfig {
                    stride,
                    padding: pad,
                    groups,
                    activation: if relu {
                        Activation::Relu
                    } else {
                        Activation::None
                    },
                };
                let weights = weights.convs[index]
                    .as_ref()
                    .context("a Conv step without weights: the plan and the weights disagree")?;
                conv2d(source, weights, &config).with_context(|| format!("step {index}: conv"))?
            }
            Step::Add { a, b } => add(slot(&slots, a, index)?, slot(&slots, b, index)?)
                .with_context(|| format!("step {index}: add"))?,
            Step::Upsample2x { input } => resize2x(slot(&slots, input, index)?)
                .with_context(|| format!("step {index}: upsample"))?,
        };
        slots.push(Some(produced));
    }

    super::topology::OUTPUTS
        .iter()
        .map(|&at| {
            slots
                .get(at)
                .and_then(|s| s.clone())
                .with_context(|| format!("output slot {at} was never written"))
        })
        .collect()
}

fn slot(slots: &[Option<Tensor>], at: usize, step: usize) -> Result<&Tensor> {
    slots
        .get(at)
        .and_then(|s| s.as_ref())
        .with_context(|| format!("step {step} reads slot {at}, which does not exist yet"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_step_reads_a_slot_that_already_exists() {
        // The generated table is only as good as its indices: a step reading ahead of itself,
        // or an output pointing past the end, would fail deep inside execution with a shape
        // error rather than here.
        for (index, step) in super::super::topology::STEPS.iter().enumerate() {
            let reads: Vec<usize> = match *step {
                Step::Conv { input, .. } | Step::Upsample2x { input } => vec![input],
                Step::Add { a, b } => vec![a, b],
            };
            for read in reads {
                // Step `index` writes slot `index + 1`, so anything it reads must be at most
                // its own index: slot 0 is the network input.
                assert!(
                    read <= index,
                    "step {index} reads slot {read}, which is written later"
                );
            }
        }
        for output in super::super::topology::OUTPUTS {
            assert!(
                output <= super::super::topology::STEPS.len(),
                "output slot {output} is past the end of the plan"
            );
        }
    }

    #[test]
    fn the_plan_is_the_shape_the_exporter_produces() {
        let steps = super::super::topology::STEPS;
        let convs = steps
            .iter()
            .filter(|s| matches!(s, Step::Conv { .. }))
            .count();
        // 60 convolutions, 4 lateral adds and 2 upsamples: if a re-export changes this, the
        // decoder's assumptions about strides and anchors deserve re-checking too.
        assert_eq!(convs, 60, "expected SCRFD-500M's 60 convolutions");
        assert_eq!(steps.len(), 66);
        assert_eq!(super::super::topology::OUTPUTS.len(), 9);
    }
}
