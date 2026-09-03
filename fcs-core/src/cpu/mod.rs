//! Pure-Rust CPU inference for YuNet.
//!
//! A CPU counterpart to [`crate::gpu`], running the same hand-encoded YuNet
//! topology so the app needs neither an external runtime nor a GPU. The op set
//! is small because the model is: convolution (dense, depthwise or grouped,
//! optionally fusing ReLU), 2x2 max pooling, elementwise add, and a nearest 2x
//! upsample. BatchNorm is already folded into the exported weights.

pub mod conv2d;
pub mod graph;
pub mod ops;
pub mod runtime;
pub mod tensor;
