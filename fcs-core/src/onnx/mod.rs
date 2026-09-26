//! Reading named float initializers out of an ONNX file.
//!
//! The topology of every network here is compiled in -- `scrfd::topology` for the
//! detector -- so the model file is consulted for weights alone, by name. That is why the
//! export folds BatchNorm itself rather than letting torch's constant folding rename
//! things: a renamed initializer fails here instead of part-way through a frame.

/// The slice of the ONNX protobuf schema this needs.
pub mod proto;

use std::{
    collections::{HashMap, HashSet},
    fs,
    path::Path,
};

use anyhow::{Context, Result, anyhow};
use prost::Message;
use proto::{self as pb, DATA_TYPE_FLOAT};

/// Thin wrapper around a subset of ONNX initializers (float tensors only).
#[derive(Debug)]
pub struct OnnxInitializerMap {
    tensors: HashMap<String, OnnxTensor>,
}

impl OnnxInitializerMap {
    /// Load the ONNX model at `model_path` and retain only the requested initializer names.
    pub fn load<P: AsRef<Path>>(model_path: P, names: &[&str]) -> Result<Self> {
        let bytes = fs::read(model_path.as_ref())
            .with_context(|| format!("failed to read {}", model_path.as_ref().display()))?;
        let proto = pb::ModelProto::decode(&*bytes).context("failed to decode ONNX protobuf")?;
        let graph = proto.graph.context("ONNX model missing GraphProto")?;

        let wanted: HashSet<&str> = names.iter().copied().collect();
        let mut tensors = HashMap::with_capacity(wanted.len());

        for tensor in &graph.initializer {
            if wanted.contains(tensor.name.as_str()) {
                let parsed = OnnxTensor::from_proto(tensor)?;
                tensors.insert(tensor.name.clone(), parsed);
            }
        }

        for name in wanted {
            anyhow::ensure!(
                tensors.contains_key(name),
                "initializer '{name}' not found in model {}",
                model_path.as_ref().display()
            );
        }

        Ok(Self { tensors })
    }

    /// Borrow an initializer tensor by name.
    pub fn tensor(&self, name: &str) -> Result<&OnnxTensor> {
        self.tensors
            .get(name)
            .with_context(|| format!("initializer '{name}' not loaded"))
    }

    /// Return the number of loaded initializer tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Return whether no initializer tensors were loaded.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Iterate over loaded tensors in unspecified order.
    pub fn values(&self) -> impl Iterator<Item = &OnnxTensor> {
        self.tensors.values()
    }

    /// Consume the loader and return its initializer-name-to-tensor map.
    pub fn into_map(self) -> HashMap<String, OnnxTensor> {
        self.tensors
    }
}

/// Float tensor extracted from an ONNX initializer.
#[derive(Debug, Clone)]
pub struct OnnxTensor {
    dims: Vec<usize>,
    data: Vec<f32>,
}

impl OnnxTensor {
    fn from_proto(proto: &pb::TensorProto) -> Result<Self> {
        anyhow::ensure!(
            proto.data_type == DATA_TYPE_FLOAT,
            "only float initializers are supported (found {})",
            proto.data_type
        );

        let dims = proto
            .dims
            .iter()
            .map(|&d| usize::try_from(d).with_context(|| format!("invalid dimension value {d}")))
            .collect::<Result<Vec<_>>>()?;

        let data = if !proto.raw_data.is_empty() {
            let (floats, remainder) = proto.raw_data.as_chunks::<4>();
            anyhow::ensure!(
                remainder.is_empty(),
                "initializer '{}' has an incomplete float payload",
                proto.name
            );
            floats
                .iter()
                .map(|&bytes| f32::from_le_bytes(bytes))
                .collect()
        } else if !proto.float_data.is_empty() {
            proto.float_data.clone()
        } else {
            return Err(anyhow!("initializer '{}' has no data payload", proto.name));
        };

        let expected = dims
            .iter()
            .try_fold(1usize, |size, &dim| size.checked_mul(dim))
            .context("initializer shape overflows usize")?;
        anyhow::ensure!(
            data.len() == expected,
            "initializer '{}' data length ({}) does not match shape {:?}",
            proto.name,
            data.len(),
            dims
        );

        Ok(Self { dims, data })
    }

    /// Tensor dimensions.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Flattened data buffer.
    pub fn data(&self) -> &[f32] {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn raw_floats_are_little_endian_and_malformed_payloads_return_errors() {
        let mut tensor = pb::TensorProto {
            name: "raw".into(),
            dims: vec![1],
            data_type: DATA_TYPE_FLOAT,
            raw_data: 1.25f32.to_le_bytes().to_vec(),
            ..Default::default()
        };
        assert_eq!(OnnxTensor::from_proto(&tensor).unwrap().data(), &[1.25]);
        tensor.raw_data.pop();
        assert!(OnnxTensor::from_proto(&tensor).is_err());
        tensor.raw_data = 1.25f32.to_le_bytes().to_vec();
        tensor.dims = vec![i64::MAX, i64::MAX, 8];
        assert!(OnnxTensor::from_proto(&tensor).is_err());
    }

    /// Build a minimal ONNX file holding just the named float initializers.
    ///
    /// Written with our own [`proto`] declarations, which the tract-built fixture this replaced
    /// did not do: an encoder sharing the reader's tags cannot catch a tag that is wrong in both
    /// directions. `proto`'s own tests say the same. What does prove the tags match ONNX is
    /// loading the real export by name, which `scrfd::plan::ScrfdWeights::load` does and
    /// `tests/scrfd_parity.rs` then checks the values of against ONNX Runtime.
    fn synthetic_onnx_model(tensors: &[(&str, Vec<i64>, Vec<f32>)]) -> Vec<u8> {
        pb::ModelProto {
            graph: Some(pb::GraphProto {
                initializer: tensors
                    .iter()
                    .map(|(name, dims, data)| pb::TensorProto {
                        name: (*name).to_string(),
                        dims: dims.clone(),
                        data_type: DATA_TYPE_FLOAT,
                        float_data: data.clone(),
                        raw_data: Vec::new(),
                    })
                    .collect(),
            }),
        }
        .encode_to_vec()
    }

    fn write_synthetic_model(
        dir: &tempfile::TempDir,
        tensors: &[(&str, Vec<i64>, Vec<f32>)],
    ) -> PathBuf {
        let path = dir.path().join("synthetic.onnx");
        std::fs::write(&path, synthetic_onnx_model(tensors)).expect("write synthetic model");
        path
    }

    #[test]
    fn keeps_only_the_requested_float_initializers() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = write_synthetic_model(
            &dir,
            &[
                ("w", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]),
                ("b", vec![2], vec![0.5, -0.5]),
                ("unused", vec![1], vec![9.0]),
            ],
        );

        let loader = OnnxInitializerMap::load(&path, &["w", "b"]).expect("load initializers");
        assert_eq!(loader.len(), 2, "only the requested initializers are kept");
        assert!(!loader.is_empty());
        assert_eq!(loader.values().count(), 2);

        let w = loader.tensor("w").expect("weight tensor");
        assert_eq!(w.dims(), &[2, 2]);
        assert_eq!(w.data(), &[1.0, 2.0, 3.0, 4.0]);
        assert!(loader.tensor("unused").is_err());

        let map = loader.into_map();
        assert_eq!(map.len(), 2);
        assert_eq!(map["b"].data(), &[0.5, -0.5]);

        // Asking for nothing yields an empty map rather than the whole model.
        let none = OnnxInitializerMap::load(&path, &[]).expect("load no initializers");
        assert!(none.is_empty());
        assert_eq!(none.len(), 0);
    }

    #[test]
    fn rejects_missing_names_and_bad_shapes() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = write_synthetic_model(&dir, &[("w", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])]);

        let err = OnnxInitializerMap::load(&path, &["w", "absent"])
            .expect_err("a missing initializer must fail the load");
        assert!(
            err.to_string().contains("absent"),
            "error should name the missing initializer: {err}"
        );

        let short = write_synthetic_model(&dir, &[("w", vec![2, 2], vec![1.0, 2.0])]);
        let err = OnnxInitializerMap::load(&short, &["w"])
            .expect_err("a data/shape mismatch must fail the load");
        assert!(
            err.to_string().contains("does not match shape"),
            "error should report the shape mismatch: {err}"
        );
    }
}
