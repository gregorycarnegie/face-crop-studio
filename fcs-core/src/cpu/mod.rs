//! Pure-Rust CPU inference building blocks.
//!
//! A CPU counterpart to [`crate::gpu`]: the same small op set, on the CPU, so the app needs
//! neither an external runtime nor a GPU. It is small because the networks are: dense and
//! depthwise convolution (optionally fusing ReLU), elementwise add, a nearest 2x upsample and
//! global average pooling, all in [`nchwc`]'s channel-blocked layout. BatchNorm is already
//! folded into the exported weights.
//!
//! The topology that drives these ops lives with its model; see [`crate::scrfd::plan`].

pub mod nchwc;
pub mod tensor;
