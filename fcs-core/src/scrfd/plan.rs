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
//! the shared ONNX initializer reader; everything about the shape of the network is compiled
//! in. A model with a different architecture will fail to load rather than half-run.

use anyhow::{Context, Result};

use crate::{
    cpu::{
        nchwc::{self, BLOCK, Blocked, DenseWeights, DepthwiseWeights, Source},
        tensor::Tensor,
    },
    onnx::OnnxInitializerMap,
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
    convs: Vec<Prepared>,
}

/// What a step runs with, once sibling convolutions are merged.
#[derive(Debug)]
enum Prepared {
    /// An add or an upsample: no weights.
    NoWeights,
    /// A dense convolution covering this step and the `split.len() - 1` steps after it,
    /// which read the same input with the same geometry. `split` is each step's share of the
    /// output channels, in step order.
    Dense {
        weights: DenseWeights,
        split: Vec<usize>,
    },
    /// A depthwise convolution.
    Depthwise(DepthwiseWeights),
    /// Written by the merged convolution of an earlier step.
    Merged,
}

/// How many steps from `at` onwards are dense convolutions of one input with one geometry,
/// and so can run as a single wider convolution.
///
/// Each level's class, box and keypoint heads are three such steps (64->2, 64->8, 64->20).
/// Run apart, each reads the same input: merged, it is read once, and the 2-channel class head
/// stops wasting six of its block's eight lanes.
fn siblings(steps: &[Step], at: usize) -> usize {
    let key = |step: &Step| match *step {
        Step::Conv {
            input,
            in_per_group,
            kernel,
            stride,
            pad,
            groups: 1,
            relu,
            ..
        } => Some((input, in_per_group, kernel, stride, pad, relu)),
        _ => None,
    };
    let Some(first) = key(&steps[at]) else {
        return 1;
    };
    steps[at..]
        .iter()
        .take_while(|step| key(step) == Some(first))
        .count()
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

        let steps = &super::topology::STEPS;
        let mut convs = Vec::with_capacity(steps.len());
        while convs.len() < steps.len() {
            let at = convs.len();
            if !matches!(steps[at], Step::Conv { .. }) {
                convs.push(Prepared::NoWeights);
                continue;
            }
            if let Step::Conv {
                weight,
                bias,
                out_channels,
                in_per_group: 1,
                kernel,
                groups,
                ..
            } = steps[at]
                && groups == out_channels
                && groups > 1
            {
                let bias = if bias.is_empty() {
                    vec![0.0; out_channels]
                } else {
                    map.tensor(bias)?.data().to_vec()
                };
                convs.push(Prepared::Depthwise(
                    DepthwiseWeights::new(out_channels, kernel, map.tensor(weight)?.data(), &bias)
                        .with_context(|| weight.to_string())?,
                ));
                continue;
            }
            let count = siblings(steps, at);
            let (mut data, mut biases, mut split) = (Vec::new(), Vec::new(), Vec::new());
            let (mut in_per, mut side) = (0, 0);
            for step in &steps[at..at + count] {
                let Step::Conv {
                    weight,
                    bias,
                    out_channels,
                    in_per_group,
                    kernel,
                    groups,
                    ..
                } = *step
                else {
                    unreachable!("siblings counts only convolutions");
                };
                anyhow::ensure!(
                    groups == 1,
                    "{weight}: grouped convolution that is not depthwise; the plan has no kernel for it"
                );
                let tensor = map.tensor(weight)?;
                let expected = out_channels * in_per_group * kernel * kernel;
                anyhow::ensure!(
                    tensor.data().len() == expected,
                    "{weight} holds {} values, the topology expects {expected}",
                    tensor.data().len()
                );
                data.extend_from_slice(tensor.data());
                if bias.is_empty() {
                    biases.extend(std::iter::repeat_n(0.0, out_channels));
                } else {
                    biases.extend_from_slice(map.tensor(bias)?.data());
                }
                split.push(out_channels);
                (in_per, side) = (in_per_group, kernel);
            }
            // Slot 0 is the NCHW network input, read as blocks of one channel.
            let Step::Conv { input, .. } = steps[at] else {
                unreachable!("checked above");
            };
            let block_in = if input == 0 { 1 } else { BLOCK };
            let weights =
                DenseWeights::new(split.iter().sum(), in_per, side, block_in, &data, &biases)?;
            convs.push(Prepared::Dense { weights, split });
            convs.extend((1..count).map(|_| Prepared::Merged));
        }
        Ok(Self { convs })
    }
}

/// What a slot holds during a run.
enum Slot {
    /// Slot 0: the NCHW network input, held apart.
    Input,
    /// A whole activation tensor.
    Data(Blocked),
    /// Channels `first..first + count` of a merged convolution's output, `merged[of]`.
    Part {
        of: usize,
        first: usize,
        count: usize,
    },
}

/// Run every step and return the nine head outputs, in the topology's output order.
///
/// Activations stay in the blocked layout from the first convolution, which reads the NCHW
/// input directly, to the outputs, which are the only tensors converted back.
///
/// Slots are kept for the whole pass rather than freed once consumed: the neck reads backbone
/// levels produced long before it, so liveness is not simply "the previous step".
pub fn run(input: Tensor, weights: &ScrfdWeights) -> Result<Vec<Tensor>> {
    let mut slots = Vec::with_capacity(super::topology::STEPS.len() + 1);
    slots.push(Slot::Input);
    let mut merged: Vec<Blocked> = Vec::new();

    for (index, step) in super::topology::STEPS.iter().enumerate() {
        let produced = match *step {
            Step::Conv {
                input: from,
                stride,
                pad,
                relu,
                ..
            } => match &weights.convs[index] {
                Prepared::Dense { weights, split } => {
                    let source = match slots.get(from) {
                        Some(Slot::Input) => Source::from(&input),
                        _ => Source::from(blocked(&slots, from, index)?),
                    };
                    let output = nchwc::conv(source, weights, stride, pad, relu)
                        .with_context(|| format!("step {index}: conv"))?;
                    if split.len() == 1 {
                        output
                    } else {
                        let mut first = 0;
                        for &count in split {
                            slots.push(Slot::Part {
                                of: merged.len(),
                                first,
                                count,
                            });
                            first += count;
                        }
                        merged.push(output);
                        continue;
                    }
                }
                Prepared::Depthwise(weights) => {
                    nchwc::depthwise(blocked(&slots, from, index)?, weights, stride, pad, relu)
                        .with_context(|| format!("step {index}: depthwise conv"))?
                }
                // Its slot was pushed by the merged convolution, in step order.
                Prepared::Merged => continue,
                Prepared::NoWeights => anyhow::bail!(
                    "step {index}: a Conv step without weights: the plan and the weights disagree"
                ),
            },
            Step::Add { a, b } => {
                nchwc::add(blocked(&slots, a, index)?, blocked(&slots, b, index)?)
                    .with_context(|| format!("step {index}: add"))?
            }
            Step::Upsample2x { input: from } => nchwc::upsample2x(blocked(&slots, from, index)?)
                .with_context(|| format!("step {index}: upsample"))?,
        };
        slots.push(Slot::Data(produced));
    }

    super::topology::OUTPUTS
        .iter()
        .map(|&at| match slots.get(at) {
            Some(Slot::Data(tensor)) => tensor.to_nchw(0, tensor.channels()),
            Some(&Slot::Part { of, first, count }) => merged[of].to_nchw(first, count),
            _ => anyhow::bail!("output slot {at} was never written"),
        })
        .collect()
}

/// The whole tensor in slot `at`, for step `step` to read.
fn blocked(slots: &[Slot], at: usize, step: usize) -> Result<&Blocked> {
    match slots.get(at) {
        Some(Slot::Data(tensor)) => Ok(tensor),
        Some(Slot::Input) => {
            anyhow::bail!("step {step} reads the NCHW input with a kernel that needs blocks")
        }
        Some(Slot::Part { .. }) => {
            anyhow::bail!("step {step} reads slot {at}, part of a merged convolution")
        }
        None => anyhow::bail!("step {step} reads slot {at}, which does not exist yet"),
    }
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
    fn each_level_merges_exactly_its_three_heads() {
        let steps = &super::super::topology::STEPS;
        let mut groups = Vec::new();
        let mut at = 0;
        while at < steps.len() {
            let count = siblings(steps, at);
            if count > 1 {
                groups.push((at, count));
            }
            at += count;
        }
        // The class, box and keypoint heads of strides 8, 16 and 32, writing slots 50-52,
        // 57-59 and 64-66 -- every one of `OUTPUTS`. Nothing else shares an input and a
        // geometry.
        assert_eq!(groups, [(49, 3), (56, 3), (63, 3)]);
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
