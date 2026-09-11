use std::{
    collections::{HashMap, HashSet},
    fs,
    path::Path,
};

use super::proto::{self as pb, DATA_TYPE_FLOAT};
use anyhow::{Context, Result, anyhow};
use bytemuck::cast_slice;
use prost::Message;

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
            let floats = cast_slice::<u8, f32>(&proto.raw_data);
            floats.to_vec()
        } else if !proto.float_data.is_empty() {
            proto.float_data.clone()
        } else {
            return Err(anyhow!("initializer '{}' has no data payload", proto.name));
        };

        anyhow::ensure!(
            data.len() == dims.iter().product::<usize>(),
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
