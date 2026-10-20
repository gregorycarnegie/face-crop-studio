use anyhow::{Context, Result, ensure};
use cudarc::{
    driver::{CudaContext, CudaFunction, sys},
    nvrtc::Ptx,
};
use std::sync::Arc;

#[cfg(target_os = "windows")]
pub(crate) const PTX: &str = include_str!("windows.ptx");
#[cfg(target_os = "linux")]
pub(crate) const PTX: &str = include_str!("linux.ptx");

/// Every kernel, resolved once; loading them all is also the module's validation.
#[derive(Debug)]
pub struct Runtime {
    pub context: Arc<CudaContext>,
    pub conv: CudaFunction,
    pub pointwise: CudaFunction,
    pub depthwise: CudaFunction,
    pub add: CudaFunction,
    pub upsample: CudaFunction,
    pub pixel: CudaFunction,
    pub blur: CudaFunction,
    pub bilateral: CudaFunction,
    pub background: CudaFunction,
    pub histogram: CudaFunction,
    pub lut: CudaFunction,
    pub equalize: CudaFunction,
    pub shape: CudaFunction,
}

impl Runtime {
    pub fn new() -> Result<Arc<Self>> {
        static RUNTIME: std::sync::OnceLock<Result<Arc<Runtime>, String>> =
            std::sync::OnceLock::new();
        RUNTIME
            .get_or_init(|| {
                Self::from_ptx(PTX)
                    .map(Arc::new)
                    .map_err(|e| format!("{e:#}"))
            })
            .clone()
            .map_err(anyhow::Error::msg)
    }

    pub fn from_ptx(ptx: &str) -> Result<Self> {
        // SAFETY: only asks the CUDA wrapper to probe for its driver library.
        ensure!(
            unsafe { sys::is_culib_present() },
            "No NVIDIA CUDA driver is installed"
        );
        let mut version = 0;
        // SAFETY: this entry point predates all supported CUDA versions; version is writable.
        unsafe {
            sys::cuDriverGetVersion(&mut version).result()?;
        }
        // Both PTX files declare ISA 7.0 (CUDA 11.0); 12.0 is the floor of cudarc's bindings.
        ensure!(
            version >= 12000,
            "NVIDIA driver is too old for CUDA Rust (CUDA driver API 12.0 or newer required)"
        );
        let count = CudaContext::device_count().context("initializing the NVIDIA CUDA driver")?;
        for ordinal in 0..count as usize {
            let context = CudaContext::new(ordinal)?;
            if context.compute_capability()?.0 < 8 {
                continue;
            }
            // Loading the actual module validates the driver's PTX support, too.
            let module = context
                .load_module(Ptx::from_src(ptx))
                .context("loading experimental CUDA Rust kernels")?;
            let load = |name: &str| {
                module
                    .load_function(name)
                    .with_context(|| format!("loading CUDA kernel {name}"))
            };
            return Ok(Self {
                conv: load("fcs_conv")?,
                pointwise: load("fcs_pointwise")?,
                depthwise: load("fcs_depthwise")?,
                add: load("fcs_add")?,
                upsample: load("fcs_upsample")?,
                pixel: load("fcs_pixel")?,
                blur: load("fcs_blur")?,
                bilateral: load("fcs_bilateral")?,
                background: load("fcs_background")?,
                histogram: load("fcs_histogram")?,
                lut: load("fcs_lut")?,
                equalize: load("fcs_equalize")?,
                shape: load("fcs_shape")?,
                context,
            });
        }
        anyhow::bail!(
            "CUDA Rust requires a compatible NVIDIA GPU with compute capability 8.0 or newer"
        )
    }
}
