//! GPU inference building blocks for the YuNet model.
//!
//! Phase 13.3 Option C starts by reimplementing the fundamental layers (conv,
//! pooling, activations) as WGSL compute shaders. These building blocks
//! will power the end-to-end YuNet port in subsequent increments.

pub mod activation;
pub mod add;
pub mod conv2d;
pub mod max_pool;
pub mod ops;
pub mod upsample2x;
pub mod utils;

#[cfg(test)]
mod tests;

pub use activation::ActivationKind;
pub use conv2d::{Conv2dConfig, Conv2dOptions};
pub use ops::GpuInferenceOps;
pub mod tensor;
pub use tensor::{GpuTensor, TensorShape};

const MAX_POOL_WGSL: &str = include_str!("pool.wgsl");
const ADD_WGSL: &str = include_str!("add.wgsl");
const UPSAMPLE2X_WGSL: &str = include_str!("resize2x.wgsl");
const RESIZE2X_ADD_WGSL: &str = include_str!("resize2x_add.wgsl");
pub mod graph;
pub mod runtime;
pub use runtime::GpuYuNet;
