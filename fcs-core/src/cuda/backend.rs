use std::{path::Path, sync::Arc};

use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use crate::{
    onnx::OnnxInitializerMap,
    scrfd::{
        INPUT_SIZE,
        plan::{Step, siblings},
        topology::{OUTPUTS, STEPS},
    },
};

use fcs_utils::cuda::runtime::Runtime;

/// Weights are shared read-only; each inference owns a stream and its temporary buffers.
#[derive(Debug)]
pub(crate) struct CudaDetector {
    runtime: Arc<Runtime>,
    weights: Vec<Option<(CudaSlice<f32>, CudaSlice<f32>)>>,
    shapes: Vec<[usize; 4]>,
    /// How many consecutive steps each step's launch writes: sibling heads run as one wider
    /// convolution on the first of them (n), and the rest launch nothing (0).
    covers: Vec<usize>,
}

/// Input channels per thread in split-K convolutions. Each detection head (64 inputs) then
/// runs four threads per output, which beat 4, 8 and 32 on an RTX 4090 (2026-10-21) and asks
/// least of smaller GPUs; layers with fewer than two chunks of input keep the plain kernel.
const SPLIT_CHUNK: usize = 16;

impl CudaDetector {
    pub(crate) fn load(path: &Path) -> Result<Self> {
        let names: Vec<_> = STEPS
            .iter()
            .flat_map(|step| match step {
                Step::Conv { weight, bias, .. } => vec![*weight, *bias],
                _ => Vec::new(),
            })
            .filter(|name| !name.is_empty())
            .collect();
        let map = OnnxInitializerMap::load(path, &names)?;
        let runtime = Runtime::new()?;
        let stream = runtime.context.default_stream();
        let mut host = Vec::with_capacity(STEPS.len());
        let mut shapes = vec![[1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize]];
        for (index, step) in STEPS.iter().enumerate() {
            let (shape, conv) = match *step {
                Step::Conv {
                    input,
                    weight,
                    bias,
                    out_channels,
                    in_per_group,
                    kernel,
                    stride,
                    pad,
                    groups,
                    ..
                } => {
                    let [_, channels, height, width] =
                        *shapes.get(input).context("invalid CUDA topology input")?;
                    ensure!(
                        channels == in_per_group * groups && out_channels % groups == 0,
                        "step {index}: invalid convolution channels"
                    );
                    let shape = [
                        1,
                        out_channels,
                        (height + 2 * pad - kernel) / stride + 1,
                        (width + 2 * pad - kernel) / stride + 1,
                    ];
                    let data = map.tensor(weight)?.data();
                    ensure!(
                        data.len() == out_channels * in_per_group * kernel * kernel,
                        "step {index}: invalid weights"
                    );
                    let biases = if bias.is_empty() {
                        vec![0.0; out_channels]
                    } else {
                        map.tensor(bias)?.data().to_vec()
                    };
                    ensure!(biases.len() == out_channels, "step {index}: invalid bias");
                    (shape, Some((data.to_vec(), biases)))
                }
                Step::Add { a, b } => {
                    ensure!(shapes[a] == shapes[b], "step {index}: add shapes differ");
                    (shapes[a], None)
                }
                Step::Upsample2x { input } => {
                    let [n, c, h, w] = shapes[input];
                    ([n, c, h * 2, w * 2], None)
                }
            };
            ensure!(
                shape.iter().product::<usize>() <= u32::MAX as usize,
                "CUDA tensor is too large"
            );
            shapes.push(shape);
            host.push(conv);
        }
        // Merge each run of siblings into its first step, as the CPU plan does: one launch
        // instead of three, with three times the threads for the narrow heads.
        let mut covers = vec![1; STEPS.len()];
        let mut at = 0;
        while at < STEPS.len() {
            let count = siblings(&STEPS, at);
            for follower in at + 1..at + count {
                let (data, biases) = host[follower].take().context("sibling without weights")?;
                let lead = host[at].as_mut().context("sibling without weights")?;
                lead.0.extend(data);
                lead.1.extend(biases);
                covers[follower] = 0;
            }
            covers[at] = count;
            at += count;
        }
        // Merged slots are views into the lead's buffer, read back only as outputs.
        let whole = |slot: usize| slot == 0 || covers[slot - 1] == 1;
        for (index, step) in STEPS.iter().enumerate() {
            let reads = match *step {
                Step::Conv { input, .. } | Step::Upsample2x { input } => whole(input),
                Step::Add { a, b } => whole(a) && whole(b),
            };
            ensure!(reads, "step {index} reads part of a merged convolution");
        }
        let weights = host
            .into_iter()
            .map(|conv| {
                conv.map(|(data, biases)| {
                    Ok((stream.clone_htod(&data)?, stream.clone_htod(&biases)?))
                })
                .transpose()
            })
            .collect::<Result<_>>()?;
        stream.synchronize()?;
        Ok(Self {
            runtime,
            weights,
            shapes,
            covers,
        })
    }

    pub(crate) fn infer(&self, input: &[f32]) -> Result<Vec<(Vec<f32>, [usize; 4])>> {
        ensure!(
            input.len() == self.shapes[0].iter().product::<usize>(),
            "invalid CUDA input length"
        );
        let stream = self.runtime.context.new_stream()?;
        let mut buffers = Vec::with_capacity(STEPS.len() + 1);
        buffers.push(stream.clone_htod(input)?);
        // Per slot: which buffer holds it, and where.
        let mut views = vec![(0, 0, input.len())];
        // shortcut: per-inference buffers; pool only after profiling establishes an allocation bottleneck.
        for (index, step) in STEPS.iter().enumerate() {
            let covers = self.covers[index];
            if covers == 0 {
                continue;
            }
            let shape = self.shapes[index + 1];
            let written = index + 1..index + 1 + covers;
            let len = written
                .clone()
                .map(|slot| self.shapes[slot].iter().product::<usize>())
                .sum::<usize>() as u32;
            let channels: usize = written.clone().map(|slot| self.shapes[slot][1]).sum();
            // SAFETY: each kernel below writes all `len` elements before any step reads them.
            let mut output = unsafe { stream.alloc::<f32>(len as usize)? };
            let config = LaunchConfig {
                grid_dim: (len.div_ceil(128), 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: load validates the fixed topology and all buffer lengths; outputs are
            // disjoint, retained until readback, and every operation uses this run's stream.
            unsafe {
                match *step {
                    Step::Conv {
                        input,
                        in_per_group,
                        out_channels,
                        kernel,
                        stride,
                        pad,
                        groups,
                        relu,
                        ..
                    } => {
                        let [_, _, h, w] = self.shapes[input];
                        let params = [
                            h as u32,
                            w as u32,
                            shape[2] as u32,
                            shape[3] as u32,
                            in_per_group as u32,
                            channels as u32,
                            kernel as u32,
                            stride as u32,
                            pad as u32,
                            groups as u32,
                            u32::from(relu),
                        ];
                        let (weights, bias) = self.weights[index]
                            .as_ref()
                            .context("missing CUDA weights")?;
                        if groups == 1 && kernel > 1 && in_per_group >= 2 * SPLIT_CHUNK {
                            let chunks = in_per_group.div_ceil(SPLIT_CHUNK);
                            let total = u32::try_from(len as usize * chunks)?;
                            // fcs_conv_partial writes every one of the `total` partial sums.
                            let mut partial = stream.alloc::<f32>(total as usize)?;
                            stream
                                .launch_builder(&self.runtime.conv_partial)
                                .arg(&buffers[views[input].0])
                                .arg(weights)
                                .arg(&mut partial)
                                .arg(&(h as u32))
                                .arg(&(w as u32))
                                .arg(&(shape[2] as u32))
                                .arg(&(shape[3] as u32))
                                .arg(&(in_per_group as u32))
                                .arg(&(channels as u32))
                                .arg(&(kernel as u32))
                                .arg(&(stride as u32))
                                .arg(&(pad as u32))
                                .arg(&(SPLIT_CHUNK as u32))
                                .launch(LaunchConfig {
                                    grid_dim: (total.div_ceil(128), 1, 1),
                                    block_dim: (128, 1, 1),
                                    shared_mem_bytes: 0,
                                })?;
                            stream
                                .launch_builder(&self.runtime.conv_reduce)
                                .arg(&partial)
                                .arg(bias)
                                .arg(&mut output)
                                .arg(&len)
                                .arg(&((shape[2] * shape[3]) as u32))
                                .arg(&(chunks as u32))
                                .arg(&u32::from(relu))
                                .launch(config)?;
                        } else {
                            let function = if kernel == 1 && groups == 1 {
                                &self.runtime.pointwise
                            } else if in_per_group == 1 && groups == out_channels {
                                &self.runtime.depthwise
                            } else {
                                &self.runtime.conv
                            };
                            let mut launch = stream.launch_builder(function);
                            launch
                                .arg(&buffers[views[input].0])
                                .arg(weights)
                                .arg(bias)
                                .arg(&mut output);
                            for param in &params {
                                launch.arg(param);
                            }
                            launch.launch(config)?;
                        }
                    }
                    Step::Add { a, b } => {
                        stream
                            .launch_builder(&self.runtime.add)
                            .arg(&buffers[views[a].0])
                            .arg(&buffers[views[b].0])
                            .arg(&mut output)
                            .arg(&len)
                            .launch(config)?;
                    }
                    Step::Upsample2x { input } => {
                        let [_, c, h, w] = self.shapes[input];
                        stream
                            .launch_builder(&self.runtime.upsample)
                            .arg(&buffers[views[input].0])
                            .arg(&mut output)
                            .arg(&(c as u32))
                            .arg(&(h as u32))
                            .arg(&(w as u32))
                            .launch(config)?;
                    }
                }
            }
            let mut offset = 0;
            for slot in written {
                let n = self.shapes[slot].iter().product::<usize>();
                views.push((buffers.len(), offset, n));
                offset += n;
            }
            buffers.push(output);
        }
        OUTPUTS
            .iter()
            .map(|&at| {
                let (buffer, offset, n) = views[at];
                let values = stream.clone_dtoh(&buffers[buffer].slice(offset..offset + n))?;
                Ok((values, self.shapes[at]))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a compatible NVIDIA GPU; run explicitly before shipping CUDA changes"]
    fn cuda_matches_cpu_and_is_safe_for_concurrent_inference() -> Result<()> {
        let path =
            fcs_utils::model_path("models/scrfd80k_500m_640.onnx")?.context("model required")?;
        let mut cuda = CudaDetector::load(&path)?;
        let cpu = crate::scrfd::plan::ScrfdWeights::load(&path)?;
        let side = INPUT_SIZE as usize;
        let input: Vec<_> = (0..3 * side * side)
            .map(|i| (i % 251) as f32 / 125.0 - 1.0)
            .collect();
        let expected = crate::scrfd::plan::run(
            crate::cpu::tensor::Tensor::new(1, 3, side, side, input.clone())?,
            &cpu,
        )?;
        let check = |cuda: &CudaDetector| -> Result<()> {
            let actual = cuda.infer(&input)?;
            assert_eq!(actual.len(), expected.len());
            for ((values, shape), expected) in actual.iter().zip(&expected) {
                assert_eq!(*shape, expected.dims());
                assert_eq!(values.len(), expected.data().len());
                let error = values
                    .iter()
                    .zip(expected.data())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(error < 2e-4, "CUDA/CPU error {error}");
                assert!(values.iter().all(|x| x.is_finite()));
            }
            Ok(())
        };
        for ptx in [
            include_str!("../../../fcs-utils/src/cuda/windows.ptx"),
            include_str!("../../../fcs-utils/src/cuda/linux.ptx"),
        ] {
            cuda.runtime = Arc::new(Runtime::from_ptx(ptx)?);
            check(&cuda)?;
            std::thread::scope(|scope| -> Result<()> {
                let first = scope.spawn(|| check(&cuda));
                let second = scope.spawn(|| check(&cuda));
                first.join().expect("CUDA worker panic")?;
                second.join().expect("CUDA worker panic")
            })?;
        }
        Ok(())
    }
}
