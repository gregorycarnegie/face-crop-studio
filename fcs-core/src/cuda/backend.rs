use std::{path::Path, sync::Arc};

use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use crate::{
    onnx::OnnxInitializerMap,
    scrfd::{
        INPUT_SIZE,
        plan::Step,
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
}

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
        let mut weights = Vec::with_capacity(STEPS.len());
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
                    (
                        shape,
                        Some((stream.clone_htod(data)?, stream.clone_htod(&biases)?)),
                    )
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
            weights.push(conv);
        }
        stream.synchronize()?;
        Ok(Self {
            runtime,
            weights,
            shapes,
        })
    }

    pub(crate) fn infer(&self, input: &[f32]) -> Result<Vec<(Vec<f32>, [usize; 4])>> {
        ensure!(
            input.len() == self.shapes[0].iter().product::<usize>(),
            "invalid CUDA input length"
        );
        let stream = self.runtime.context.new_stream()?;
        let mut slots = Vec::with_capacity(STEPS.len() + 1);
        slots.push(stream.clone_htod(input)?);
        // shortcut: per-inference buffers; pool only after profiling establishes an allocation bottleneck.
        for (index, step) in STEPS.iter().enumerate() {
            let shape = self.shapes[index + 1];
            let len = shape.iter().product::<usize>() as u32;
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
                            out_channels as u32,
                            kernel as u32,
                            stride as u32,
                            pad as u32,
                            groups as u32,
                            u32::from(relu),
                        ];
                        let (weights, bias) = self.weights[index]
                            .as_ref()
                            .context("missing CUDA weights")?;
                        let function = if kernel == 1 && groups == 1 {
                            &self.runtime.pointwise
                        } else if in_per_group == 1 && groups == out_channels {
                            &self.runtime.depthwise
                        } else {
                            &self.runtime.conv
                        };
                        let mut launch = stream.launch_builder(function);
                        launch
                            .arg(&slots[input])
                            .arg(weights)
                            .arg(bias)
                            .arg(&mut output);
                        for param in &params {
                            launch.arg(param);
                        }
                        launch.launch(config)?;
                    }
                    Step::Add { a, b } => {
                        stream
                            .launch_builder(&self.runtime.add)
                            .arg(&slots[a])
                            .arg(&slots[b])
                            .arg(&mut output)
                            .arg(&len)
                            .launch(config)?;
                    }
                    Step::Upsample2x { input } => {
                        let [_, c, h, w] = self.shapes[input];
                        stream
                            .launch_builder(&self.runtime.upsample)
                            .arg(&slots[input])
                            .arg(&mut output)
                            .arg(&(c as u32))
                            .arg(&(h as u32))
                            .arg(&(w as u32))
                            .launch(config)?;
                    }
                }
            }
            slots.push(output);
        }
        OUTPUTS
            .iter()
            .map(|&at| Ok((stream.clone_dtoh(&slots[at])?, self.shapes[at])))
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
