//! Experimental CUDA Rust detection and availability.
pub use fcs_utils::cuda::{WARNING, availability};

#[cfg(any(target_os = "windows", target_os = "linux"))]
mod backend;
#[cfg(any(target_os = "windows", target_os = "linux"))]
pub(crate) use backend::CudaDetector;
