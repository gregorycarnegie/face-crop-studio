//! Pure-Rust CPU inference building blocks.
//!
//! A CPU counterpart to [`crate::gpu`]: the same small op set, on the CPU, so the app needs
//! neither an external runtime nor a GPU. It is small because the networks are: convolution
//! (dense, depthwise or grouped, optionally fusing ReLU), 2x2 max pooling, elementwise add,
//! and a nearest 2x upsample. BatchNorm is already folded into the exported weights.
//!
//! The topology that drives these ops lives with its model; see [`crate::scrfd::plan`].

pub mod conv2d;
pub mod ops;
pub mod tensor;
