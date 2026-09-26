//! The fixed graph exported by `tools/dataset/export_eye_refiner.py`.
//!
//! Conv/ReLU run on the detector's channel-blocked CPU kernels; the first reads the NCHW crop
//! directly. GlobalAveragePool averages each channel; Flatten then changes only the shape. The
//! export's Gemm has transB=1 and alpha=beta=1, so its [out, in] weights are a plain
//! matrix-vector product -- 33K multiply-adds, too few to be worth a kernel.
//! Semantics checked against ONNX Runtime v1.24.4's cpu/nn/pool.cc and cpu/math/gemm.cc.

use anyhow::{Context, Result, ensure};

use crate::{
    cpu::{
        nchwc::{self, BLOCK, Blocked, DenseWeights},
        tensor::Tensor,
    },
    onnx::OnnxInitializerMap,
};

use super::SIZE;

// ponytail: this is the shipped export's topology, not a general ONNX interpreter.
// Re-exporting with different names/topology requires updating this table and parity checks.
const LAYERS: [(&str, &str, usize, usize, usize); 12] = [
    ("onnx::Conv_101", "onnx::Conv_102", 32, 3, 3),
    ("onnx::Conv_104", "onnx::Conv_105", 32, 32, 3),
    ("onnx::Conv_107", "onnx::Conv_108", 64, 32, 3),
    ("onnx::Conv_110", "onnx::Conv_111", 64, 64, 3),
    ("onnx::Conv_113", "onnx::Conv_114", 128, 64, 3),
    ("onnx::Conv_116", "onnx::Conv_117", 128, 128, 3),
    ("onnx::Conv_119", "onnx::Conv_120", 128, 128, 3),
    ("onnx::Conv_122", "onnx::Conv_123", 128, 128, 3),
    ("onnx::Conv_125", "onnx::Conv_126", 256, 128, 3),
    ("onnx::Conv_128", "onnx::Conv_129", 256, 256, 3),
    ("head.2.weight", "head.2.bias", 128, 256, 1),
    ("head.4.weight", "head.4.bias", 4, 128, 1),
];

#[derive(Debug)]
pub(super) struct Weights {
    convs: Vec<DenseWeights>,
    /// The two Gemm layers after pooling: `[out][in]` weights, bias, output size.
    head: Vec<(Vec<f32>, Vec<f32>, usize)>,
}

impl Weights {
    pub(super) fn load(path: &std::path::Path) -> Result<Self> {
        let names: Vec<_> = LAYERS.iter().flat_map(|l| [l.0, l.1]).collect();
        let map = OnnxInitializerMap::load(path, &names)?;
        let mut convs = Vec::new();
        let mut head = Vec::new();
        for (index, &(weight, bias, out, input, kernel)) in LAYERS.iter().enumerate() {
            let w = map.tensor(weight)?;
            let b = map.tensor(bias)?;
            let expected = if kernel == 1 {
                vec![out, input]
            } else {
                vec![out, input, kernel, kernel]
            };
            ensure!(
                w.dims() == expected,
                "{weight}: expected {expected:?}, got {:?}",
                w.dims()
            );
            ensure!(
                b.dims() == [out],
                "{bias}: expected [{out}], got {:?}",
                b.dims()
            );
            ensure!(
                w.data().iter().chain(b.data()).all(|v| v.is_finite()),
                "{weight}/{bias}: non-finite weights"
            );
            if kernel == 1 {
                head.push((w.data().to_vec(), b.data().to_vec(), out));
            } else {
                // Layer 0 reads the NCHW crop as blocks of one channel.
                let block_in = if index == 0 { 1 } else { BLOCK };
                convs.push(DenseWeights::new(
                    out,
                    input,
                    kernel,
                    block_in,
                    w.data(),
                    b.data(),
                )?);
            }
        }
        Ok(Self { convs, head })
    }

    pub(super) fn run(&self, input: Vec<f32>) -> Result<[f32; 4]> {
        let crop = Tensor::new(1, 3, SIZE, SIZE, input)?;
        let mut x: Option<Blocked> = None;
        for (index, weights) in self.convs.iter().enumerate() {
            let source = match &x {
                Some(blocked) => blocked.into(),
                None => (&crop).into(),
            };
            let stride = if index % 2 == 0 { 2 } else { 1 };
            x = Some(
                nchwc::conv(source, weights, stride, 1, true)
                    .with_context(|| format!("eye refiner convolution {index}"))?,
            );
        }
        let mut v = x
            .context("eye refiner has no convolutions")?
            .global_average();
        for (layer, (weights, bias, out)) in self.head.iter().enumerate() {
            let relu = layer + 1 < self.head.len();
            v = (0..*out)
                .map(|o| {
                    let row = &weights[o * v.len()..(o + 1) * v.len()];
                    let sum = bias[o] + row.iter().zip(&v).map(|(w, x)| w * x).sum::<f32>();
                    if relu { sum.max(0.0) } else { sum }
                })
                .collect();
        }
        v.try_into()
            .ok()
            .context("eye refiner must return four coordinates")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_weights_from_an_incompatible_export() {
        use crate::onnx::proto::{DATA_TYPE_FLOAT, GraphProto, ModelProto, TensorProto};
        use prost::Message;

        let model = ModelProto {
            graph: Some(GraphProto {
                initializer: LAYERS
                    .iter()
                    .flat_map(|l| [l.0, l.1])
                    .map(|name| TensorProto {
                        name: name.into(),
                        dims: vec![1],
                        data_type: DATA_TYPE_FLOAT,
                        float_data: vec![0.0],
                        ..Default::default()
                    })
                    .collect(),
            }),
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wrong-shape.onnx");
        std::fs::write(&path, model.encode_to_vec()).unwrap();
        let error = Weights::load(&path).unwrap_err().to_string();
        assert!(error.contains("onnx::Conv_101: expected"), "{error}");
        assert!(super::super::EyeRefiner::load_from(&path).is_none());
    }

    #[test]
    fn native_graph_matches_onnxruntime() {
        let path = super::super::tests::refiner_model();
        if !path.exists() {
            assert!(
                !super::super::tests::strict_tests(),
                "missing model: {}",
                path.display()
            );
            eprintln!("skipped: eye-refiner model missing");
            return;
        }
        let weights = Weights::load(&path).expect("native model loads without a runtime");
        assert!(weights.run(vec![0.0; 3]).is_err());
        let input: Vec<_> = (0..3 * SIZE * SIZE)
            .map(|i| (i as f64 * 0.01).sin() as f32)
            .collect();
        let got = weights.run(input).unwrap();
        // Independent Python ORT reference, also pinned in examples/eye_refiner_parity.rs.
        let reference = [0.323_281_94, 0.433_649_63, 0.522_323_97, 0.449_758_32];
        compare(got, &reference);

        let Some(environment) = fcs_ort::Environment::shared() else {
            assert!(
                !super::super::tests::strict_tests(),
                "ONNX Runtime required for live parity"
            );
            eprintln!("live parity skipped: no runtime; fixed reference passed");
            return;
        };
        let session = fcs_ort::Session::new(&environment, &path, Default::default()).unwrap();
        let mut worst = 0.0f32;
        for seed in 0..5 {
            let input: Vec<_> = (0..3 * SIZE * SIZE)
                .map(|i| match seed {
                    0 => 0.0,
                    1 => -127.5 / 128.0,
                    2 => 127.5 / 128.0,
                    _ => (((i * 7919 + seed * 104729) % 65521) as f32 / 32760.0) - 1.0,
                })
                .collect();
            let reference = session.run(&input, &[1, 3, SIZE, SIZE]).unwrap();
            let got = weights.run(input).unwrap();
            worst = worst.max(compare(got, &reference[0].data));
        }
        eprintln!("native eye-refiner maximum absolute error: {worst:.3e}");
    }

    fn compare(got: [f32; 4], reference: &[f32]) -> f32 {
        assert_eq!(reference.len(), 4);
        let mut worst = 0.0f32;
        for (got, want) in got.into_iter().zip(reference) {
            assert!(
                got.is_finite() && (got - want).abs() < 1e-5,
                "native {got}, ORT {want}"
            );
            worst = worst.max((got - want).abs());
        }
        worst
    }
}
