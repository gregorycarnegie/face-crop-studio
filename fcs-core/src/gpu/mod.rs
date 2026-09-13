//! GPU inference building blocks for the YuNet model.
//!
//! [`GpuYuNet`](crate::gpu::GpuYuNet) runs the complete network using WGSL compute shaders.
//! [`GpuInferenceOps`](crate::gpu::GpuInferenceOps) exposes individual tensor operations, including variants
//! that record work into a caller-owned encoder for a single queue submission.

/// Elementwise activation functions for GPU tensors.
pub mod activation;
/// Elementwise tensor addition shader support.
pub mod add;
/// Grouped convolution geometry and shader support.
pub mod conv2d;
/// Max-pooling geometry and shader support.
pub mod max_pool;
/// Reusable inference operations and tensor uploads on a shared GPU context.
pub mod ops;
/// Nearest-neighbour spatial upsampling shader support.
pub mod upsample2x;
/// Recording compute dispatches in encoders or existing passes.
pub mod utils;

#[cfg(test)]
mod tests;

/// One GPU context for the whole test binary, or `None` when there is no usable adapter.
///
/// Every GPU test here used to build its own, and `cargo test` starts dozens at once, which is
/// the many-devices pile-up that wedges the NVIDIA driver (experiment 94; `fcs-utils` shares one
/// through its `test_support` for the same reason). Under cargo-mutants it showed up as roughly
/// one `fcs-core` test run in five hanging until the timeout with only GPU tests still running.
#[cfg(test)]
pub(crate) fn test_context() -> Option<std::sync::Arc<fcs_utils::gpu::GpuContext>> {
    use fcs_utils::gpu::{GpuAvailability, GpuContext, GpuContextOptions};
    static CONTEXT: std::sync::OnceLock<Option<std::sync::Arc<GpuContext>>> =
        std::sync::OnceLock::new();
    CONTEXT
        .get_or_init(
            || match GpuContext::init_with_fallback(&GpuContextOptions::default()) {
                GpuAvailability::Available(ctx) => Some(ctx),
                _ => None,
            },
        )
        .clone()
}

pub use activation::ActivationKind;
pub use conv2d::{Conv2dConfig, Conv2dOptions};
pub use ops::GpuInferenceOps;
/// Device-resident float tensors and validated shapes.
pub mod tensor;
pub use tensor::{GpuTensor, TensorShape};

const MAX_POOL_WGSL: &str = include_str!("pool.wgsl");
const ADD_WGSL: &str = include_str!("add.wgsl");
const UPSAMPLE2X_WGSL: &str = include_str!("resize2x.wgsl");
const RESIZE2X_ADD_WGSL: &str = include_str!("resize2x_add.wgsl");
/// Encoding the YuNet backbone, neck, and detection heads.
pub mod graph;
/// Loading and running the complete YuNet GPU graph.
pub mod runtime;
pub use runtime::GpuYuNet;
