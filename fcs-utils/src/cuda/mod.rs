//! Experimental NVIDIA CUDA runtime shared by detection and image processing.

use std::sync::{
    OnceLock,
    atomic::{AtomicBool, Ordering},
};

/// Shown before opting in, and printed for CLI/config-based activation.
pub const WARNING: &str = "CUDA Rust is experimental. It may crash, run slower, or produce different results. Requires a compatible NVIDIA GPU and driver; detection, enhancement and masking use CUDA.";

#[cfg(any(target_os = "windows", target_os = "linux"))]
#[doc(hidden)]
#[allow(missing_docs)]
pub mod runtime;

/// Probe asynchronously once, so opening Settings never blocks on a driver or PTX compilation.
/// `None` means the probe is pending; `Err` explains why activation is disabled.
/// The probe's context is dropped afterwards, so looking at Settings holds no GPU memory.
pub fn availability() -> Option<&'static Result<String, String>> {
    static RESULT: OnceLock<Result<String, String>> = OnceLock::new();
    static STARTED: AtomicBool = AtomicBool::new(false);
    if !STARTED.swap(true, Ordering::Relaxed) {
        std::thread::spawn(|| {
            let _ = RESULT.set(probe());
        });
    }
    RESULT.get()
}

fn probe() -> Result<String, String> {
    #[cfg(any(target_os = "windows", target_os = "linux"))]
    {
        runtime::Runtime::from_ptx(runtime::PTX)
            .and_then(|runtime| Ok(runtime.context.name()?))
            .map_err(|e| format!("{e:#}"))
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        Err("CUDA Rust is available only on supported NVIDIA GPUs on Windows and Linux".into())
    }
}
