//! GPU context management helpers built on top of `wgpu`.
//!
//! This module keeps GPU initialization code in one place so the CLI and GUI
//! can share the same device/queue plumbing while still offering a CPU
//! fallback when no compatible adapter is present.

/// Combined resize + RGB→BGR + HWC→CHW WGSL compute shader.
pub const PREPROCESS_WGSL: &str = include_str!("preprocess.wgsl");
/// Packed 8-bit RGB → f32 BGR CHW, for sources resized on the CPU first.
pub const RGB_TO_CHW_WGSL: &str = include_str!("rgb_to_chw.wgsl");
/// Per-pixel exposure/brightness/contrast/saturation adjust shader.
pub const PIXEL_ADJUST_WGSL: &str = include_str!("pixel_adjust.wgsl");
/// Gaussian blur shader (horizontal/vertical).
pub const GAUSSIAN_BLUR_WGSL: &str = include_str!("gaussian_blur.wgsl");
/// Bilateral filter shader for skin smoothing.
pub const BILATERAL_FILTER_WGSL: &str = include_str!("bilateral_filter.wgsl");
/// Background blur shader for elliptical blending.
pub const BACKGROUND_BLUR_WGSL: &str = include_str!("background_blur.wgsl");
/// Red-eye removal shader.
pub const RED_EYE_WGSL: &str = include_str!("red_eye.wgsl");
/// Shape mask shader.
pub const SHAPE_MASK_WGSL: &str = include_str!("shape_mask.wgsl");
/// Histogram equalization shader module.
pub const HIST_EQUALIZE_WGSL: &str = include_str!("hist_equalize.wgsl");

/// GPU exposure, brightness, contrast, and saturation adjustments.
pub mod pixel_adjust;
pub use pixel_adjust::GpuPixelAdjust;
/// Separable Gaussian blur on the GPU.
pub mod gaussian_blur;
pub use gaussian_blur::GpuGaussianBlur;
/// Edge-preserving bilateral smoothing on the GPU.
pub mod bilateral_filter;
pub use bilateral_filter::GpuBilateralFilter;
/// Blending sharp and blurred images with a central elliptical mask.
pub mod background_blur;
pub use background_blur::GpuBackgroundBlur;
/// GPU red-eye correction and optional eye regions.
pub mod red_eye;
pub use red_eye::{GpuRedEyeRemoval, RedEye};
/// GPU crop-shape masking and edge vignettes.
pub mod shape_mask;
pub use shape_mask::GpuShapeMask;
/// Per-channel histogram equalization on the GPU.
pub mod hist_equalize;
pub use hist_equalize::GpuHistogramEqualizer;
/// Reusable GPU buffer allocations and execution scopes.
pub mod buffer_pool;
pub use buffer_pool::{ExecutionScope, GpuBufferPool};
pub mod profiler;
pub use profiler::{GpuProfiler, PassTiming, total_by_label};
/// Platform-specific estimates of available video memory.
pub mod memory;
pub use memory::get_available_vram;
#[cfg(test)]
pub(crate) mod test_support;

use std::sync::Arc;

/// Compute passes one profiler run can hold. A YuNet forward pass is ~62.
const DEFAULT_PROFILER_PASSES: u32 = 256;

/// How long a blocking wait for the GPU may take before it is treated as a fault.
///
/// Every wait in this codebase covers one submission -- a forward pass, a preprocessing
/// dispatch, a filter or a timestamp resolve -- and those are single-digit milliseconds
/// even on the slowest hardware the app runs on. Windows resets a GPU that stops
/// responding for two seconds, so a wait still running after thirty is not slow work; it
/// is work that is never going to finish, and the caller is better off being told than
/// blocked. Observed once during experiment 82's validation: a test binary sat for ten
/// hours on 35 seconds of CPU and had to be killed, because every `poll` in the pipeline
/// passed `timeout: None` (experiment 94).
pub const GPU_WAIT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Block until the device's submitted work completes, or fail after [`GPU_WAIT_TIMEOUT`].
///
/// The one place any of this code waits on the GPU. A timed-out wait leaves the buffer
/// unmapped and its map callback outstanding, so every caller must propagate the error
/// rather than reading the buffer or returning it to a pool -- which is what `?` on this
/// already does.
pub fn wait_for_gpu(device: &Device, operation: &str) -> anyhow::Result<()> {
    wait_for_gpu_until(device, operation, GPU_WAIT_TIMEOUT)
}

/// [`wait_for_gpu`] with the deadline supplied, so a test can force the timeout branch
/// without waiting thirty seconds for a device that is working perfectly well.
fn wait_for_gpu_until(
    device: &Device,
    operation: &str,
    timeout: std::time::Duration,
) -> anyhow::Result<()> {
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(timeout),
        })
        .map_err(|err| wait_error(&err, operation, timeout))?;
    Ok(())
}

/// What a failed wait tells the caller.
///
/// Split out because a missed deadline is the one poll failure the caller may want to
/// retry rather than give up on, and it has to be distinguishable from a wgpu error to
/// be acted on. Nothing else in the crate can construct a `PollError`, so this is also
/// the only place the message can be checked.
fn wait_error(
    err: &wgpu::PollError,
    operation: &str,
    timeout: std::time::Duration,
) -> anyhow::Error {
    match err {
        wgpu::PollError::Timeout => anyhow::anyhow!(
            "GPU wait during {operation} timed out after {timeout:?}:              the device has not completed its submitted work"
        ),
        other => anyhow::anyhow!("device poll failed during {operation}: {other}"),
    }
}

/// Human-readable summary of resources tracked by the GPU instance.
#[derive(Debug, Clone)]
pub struct GpuReport {
    /// Formatted resource report for diagnostics.
    pub summary: String,
}

use crate::telemetry::telemetry_allows;
use log::{Level, debug, info, warn};
use pollster::block_on;
use serde::Serialize;
use serde_json;
use thiserror::Error;
use wgpu::{
    Adapter, AdapterInfo, Backends, Device, DeviceDescriptor, Dx12Compiler, ExperimentalFeatures,
    Features, Instance, InstanceDescriptor, InstanceFlags, Limits, MemoryHints, PowerPreference,
    Queue, RequestAdapterError, RequestAdapterOptions, RequestDeviceError, Trace,
};

/// Removes backends that are known to be unsafe to even enumerate on the
/// current platform, leaving `base` untouched elsewhere.
///
/// On Windows that means Vulkan. Intel's Vulkan ICD (`igvk64.dll`, driver
/// branch 30.0.101.x) dies with an access violation while wgpu is bringing up
/// the adapter, which crashed the GUI at launch on two different Intel laptops
/// during Microsoft Store certification:
///
/// ```text
/// Faulting application name: fcs-gui.exe
/// Faulting module name: igvk64.dll, version: 30.0.101.1960
/// Exception code: 0xc0000005
/// ```
///
/// The fault is inside the driver, so there is nothing to fix on this side
/// beyond not walking into it. DX12 is the native Windows backend, is present
/// on every machine this app targets, and is what the shipped GPU paths are
/// tested against. Callers that respect the environment still honour
/// `WGPU_BACKEND=vulkan`, so the backend stays reachable for debugging.
pub fn platform_safe_backends(base: Backends) -> Backends {
    if cfg!(target_os = "windows") {
        base - Backends::VULKAN
    } else {
        base
    }
}

/// High-level configuration for creating a [`GpuContext`].
#[derive(Clone, Debug)]
pub struct GpuContextOptions {
    /// Whether GPU support is enabled.
    pub enabled: bool,
    /// Allow environment variables (e.g. `WGPU_BACKEND`) to override defaults.
    pub respect_env: bool,
    /// Which backends should be considered.
    pub backends: Backends,
    /// Instance flags (debug/validation toggles).
    pub flags: InstanceFlags,
    /// Adapter preference (high-performance vs low-power).
    pub power_preference: PowerPreference,
    /// Force wgpu to pick its fallback adapter implementation.
    pub force_fallback_adapter: bool,
    /// Features that must be present on the selected adapter.
    pub required_features: Features,
    /// Optional features that will be enabled when supported.
    pub optional_features: Features,
    /// Limits that must be available. Defaults to the adapter limits.
    pub required_limits: Option<Limits>,
    /// DX12 shader compiler selection for Windows targets.
    pub dx12_shader_compiler: Dx12Compiler,
    /// Optional debug label for the logical device.
    pub label: Option<String>,
    /// Memory allocation hints forwarded to `wgpu`.
    pub memory_hints: Option<MemoryHints>,
    /// Record GPU-side timings for each compute pass. Requires adapter support for
    /// `TIMESTAMP_QUERY`; silently inactive without it.
    pub profiling: bool,
}

impl Default for GpuContextOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            respect_env: true,
            backends: platform_safe_backends(Backends::PRIMARY),
            flags: InstanceFlags::from_build_config(),
            power_preference: PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            required_features: Features::empty(),
            optional_features: Features::empty(),
            required_limits: None,
            dx12_shader_compiler: Dx12Compiler::default(),
            label: Some("YuNet GPU context".to_string()),
            memory_hints: None,
            profiling: false,
        }
    }
}

impl GpuContextOptions {
    /// Convenience helper for explicitly disabling GPU usage.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}

/// Result of attempting to initialize a GPU context while supporting CPU fallback.
#[derive(Debug)]
pub enum GpuAvailability {
    /// GPU resources are ready to use.
    Available(Arc<GpuContext>),
    /// GPU code path has been disabled by configuration (CLI flag, user choice, etc.).
    Disabled {
        /// Human-readable explanation of why GPU use was disabled.
        reason: String,
    },
    /// GPU initialization failed; callers should fall back to CPU.
    Unavailable {
        /// Initialization failure that prompted CPU fallback.
        error: GpuInitError,
    },
}

impl GpuAvailability {
    /// Returns `true` when a GPU context was created successfully.
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Available(_))
    }

    /// Returns a reference to the underlying GPU context when it exists.
    pub fn context(&self) -> Option<&Arc<GpuContext>> {
        match self {
            Self::Available(ctx) => Some(ctx),
            _ => None,
        }
    }
}

/// High-level GPU availability categories mirrored in UI/CLI messaging.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum GpuStatusMode {
    /// GPU status has not been resolved yet.
    Pending,
    /// GPU preprocessing is active.
    Available,
    /// GPU has been explicitly disabled by configuration.
    Disabled,
    /// GPU initialization failed and the app fell back to CPU.
    Fallback,
    /// GPU resources are unavailable due to driver/runtime errors.
    Error,
}

impl GpuStatusMode {
    /// Returns a stable identifier for telemetry output.
    pub fn as_str(self) -> &'static str {
        match self {
            GpuStatusMode::Pending => "pending",
            GpuStatusMode::Available => "available",
            GpuStatusMode::Disabled => "disabled",
            GpuStatusMode::Fallback => "fallback",
            GpuStatusMode::Error => "error",
        }
    }
}

/// User-facing snapshot of GPU availability and adapter metadata.
#[derive(Debug, Clone, Serialize)]
pub struct GpuStatusIndicator {
    /// Mode used for coloring/status badges.
    pub mode: GpuStatusMode,
    /// Short summary string.
    pub summary: String,
    /// Optional detail/failure reason.
    pub detail: Option<String>,
    /// Adapter name when available.
    pub adapter_name: Option<String>,
    /// Backend label (Vulkan, Metal, Dx12, etc.).
    pub backend: Option<String>,
    /// Driver description string.
    pub driver: Option<String>,
    /// Vendor ID reported by wgpu.
    pub vendor_id: Option<u32>,
    /// Device ID reported by wgpu.
    pub device_id: Option<u32>,
}

impl Default for GpuStatusIndicator {
    fn default() -> Self {
        Self::pending()
    }
}

impl GpuStatusIndicator {
    /// Pending/unknown status used during initialization.
    pub fn pending() -> Self {
        Self {
            mode: GpuStatusMode::Pending,
            summary: "Awaiting GPU check".to_string(),
            detail: None,
            adapter_name: None,
            backend: None,
            driver: None,
            vendor_id: None,
            device_id: None,
        }
    }

    /// Successful GPU activation with adapter metadata.
    pub fn available(
        adapter_name: impl Into<String>,
        backend: impl Into<String>,
        driver: Option<String>,
        vendor_id: Option<u32>,
        device_id: Option<u32>,
    ) -> Self {
        let adapter_name = adapter_name.into();
        Self {
            mode: GpuStatusMode::Available,
            summary: format!("Using {}", adapter_name),
            detail: None,
            adapter_name: Some(adapter_name),
            backend: Some(backend.into()),
            driver,
            vendor_id,
            device_id,
        }
    }

    /// GPU explicitly disabled.
    pub fn disabled(reason: impl Into<String>) -> Self {
        Self {
            mode: GpuStatusMode::Disabled,
            summary: "GPU disabled".to_string(),
            detail: Some(reason.into()),
            ..Self::pending()
        }
    }

    /// GPU fallback to CPU path due to runtime failure.
    pub fn fallback(
        reason: impl Into<String>,
        adapter_name: Option<String>,
        backend: Option<String>,
    ) -> Self {
        Self {
            mode: GpuStatusMode::Fallback,
            summary: "GPU fallback to CPU".to_string(),
            detail: Some(reason.into()),
            adapter_name,
            backend,
            driver: None,
            vendor_id: None,
            device_id: None,
        }
    }

    /// GPU entirely unavailable.
    pub fn error(reason: impl Into<String>) -> Self {
        Self {
            mode: GpuStatusMode::Error,
            summary: "GPU unavailable".to_string(),
            detail: Some(reason.into()),
            ..Self::pending()
        }
    }

    /// Emit a telemetry payload describing this status when runtime telemetry is enabled.
    pub fn emit_telemetry(&self) {
        emit_gpu_status_event(self);
    }
}

#[derive(Serialize)]
struct GpuStatusTelemetryPayload {
    event: &'static str,
    mode: &'static str,
    summary: String,
    detail: Option<String>,
    adapter_name: Option<String>,
    backend: Option<String>,
    driver: Option<String>,
    vendor_id: Option<u32>,
    device_id: Option<u32>,
}

fn emit_gpu_status_event(status: &GpuStatusIndicator) {
    use log::log;

    if !telemetry_allows(Level::Info) {
        return;
    }

    let payload = GpuStatusTelemetryPayload {
        event: "gpu_status",
        mode: status.mode.as_str(),
        summary: status.summary.clone(),
        detail: status.detail.clone(),
        adapter_name: status.adapter_name.clone(),
        backend: status.backend.clone(),
        driver: status.driver.clone(),
        vendor_id: status.vendor_id,
        device_id: status.device_id,
    };

    match serde_json::to_string(&payload) {
        Ok(json) => {
            log!(target: "fcs::telemetry", Level::Info, "{json}");
        }
        Err(err) => {
            warn!(
                target: "fcs::telemetry",
                "failed to serialize GPU telemetry payload: {err}"
            );
        }
    }
}

/// Shared GPU device/queue wrapper with a little bit of metadata.
#[derive(Debug)]
pub struct GpuContext {
    instance: Option<Instance>,
    adapter: Option<Adapter>,
    device: Device,
    queue: Queue,
    info: AdapterInfo,
    features: Features,
    limits: Limits,
    profiler: Option<GpuProfiler>,
}

impl GpuContext {
    /// Initialize a new GPU context with the provided options.
    pub fn initialize(options: &GpuContextOptions) -> Result<Self, GpuInitError> {
        if !options.enabled {
            return Err(GpuInitError::Disabled);
        }

        let mut instance_desc = if options.respect_env {
            InstanceDescriptor::new_without_display_handle_from_env()
        } else {
            InstanceDescriptor::new_without_display_handle()
        };

        let backends = if options.respect_env {
            options.backends.with_env()
        } else {
            options.backends
        };

        instance_desc.backends = backends;
        instance_desc.flags = if options.respect_env {
            options.flags.with_env()
        } else {
            options.flags
        };
        instance_desc.backend_options.dx12.shader_compiler = options.dx12_shader_compiler.clone();

        let instance = {
            let _g = crate::telemetry::timing_guard("fcs_utils::gpu_instance", log::Level::Trace);
            Instance::new(instance_desc)
        };
        let _adapter_guard =
            crate::telemetry::timing_guard("fcs_utils::gpu_request_adapter", log::Level::Trace);
        // `WGPU_POWER_PREF=low` reaches the integrated adapter on a machine that has both,
        // which is how the hardware baselines (experiment 10) run the same probes there.
        let power_preference = options
            .respect_env
            .then(PowerPreference::from_env)
            .flatten()
            .unwrap_or(options.power_preference);
        let adapter = block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference,
            force_fallback_adapter: options.force_fallback_adapter,
            // apply_limit_buckets defaults to false: bucketing rounds adapter limits
            // down to anti-fingerprinting presets, which only matters when wgpu is
            // exposed to untrusted content. A desktop app wants the real limits.
            ..Default::default()
        }))
        .map_err(|source| GpuInitError::Adapter { backends, source })?;

        drop(_adapter_guard);
        let info = adapter.get_info();
        let supported_features = adapter.features();

        if !supported_features.contains(options.required_features) {
            return Err(GpuInitError::MissingFeatures {
                requested: options.required_features,
                supported: supported_features,
            });
        }

        // Asked for opportunistically: an adapter without it still builds a context, the
        // profiler just stays absent.
        let mut wanted_optional = options.optional_features;
        if options.profiling {
            wanted_optional.insert(Features::TIMESTAMP_QUERY);
        }
        let optional = wanted_optional.intersection(supported_features);
        let features = options.required_features.union(optional);
        debug!(
            target: "fcs::gpu",
            "Optional GPU features enabled {optional:?}, unsupported {:?}",
            wanted_optional.difference(supported_features)
        );

        let limits = options
            .required_limits
            .clone()
            .unwrap_or_else(|| adapter.limits());

        let device_desc = DeviceDescriptor {
            label: options.label.as_deref(),
            required_features: features,
            required_limits: limits.clone(),
            experimental_features: ExperimentalFeatures::default(),
            memory_hints: options.memory_hints.clone().unwrap_or_default(),
            trace: Trace::default(),
        };

        let (device, queue) = {
            let _g =
                crate::telemetry::timing_guard("fcs_utils::gpu_request_device", log::Level::Trace);
            block_on(adapter.request_device(&device_desc)).map_err(GpuInitError::from)?
        };

        info!(
            target: "fcs::gpu",
            "Using GPU adapter '{}' ({:?}/{:?}) with features {:?}",
            info.name, info.backend, info.device_type, features
        );

        let profiler = options
            .profiling
            .then(|| {
                let profiler = GpuProfiler::new(&device, &queue, features, DEFAULT_PROFILER_PASSES);
                if profiler.is_none() {
                    warn!(
                        target: "fcs::gpu",
                        "GPU profiling requested but adapter '{}' has no TIMESTAMP_QUERY support",
                        info.name
                    );
                }
                profiler
            })
            .flatten();

        Ok(Self {
            instance: Some(instance),
            adapter: Some(adapter),
            device,
            queue,
            info,
            features,
            limits,
            profiler,
        })
    }

    /// Attempt to create a GPU context and gracefully fall back to CPU if that fails.
    pub fn init_with_fallback(options: &GpuContextOptions) -> GpuAvailability {
        // `initialize` refuses a disabled configuration first thing, and the arm below turns
        // that into `Disabled`.
        match Self::initialize(options) {
            Ok(ctx) => GpuAvailability::Available(Arc::new(ctx)),
            Err(GpuInitError::Disabled) => GpuAvailability::Disabled {
                reason: "GPU acceleration disabled via configuration".to_string(),
            },
            Err(err) => {
                warn!(
                    target: "fcs::gpu",
                    "GPU initialization failed ({err}); falling back to CPU."
                );
                GpuAvailability::Unavailable { error: err }
            }
        }
    }

    /// Wrap an existing device/queue pair created by an external renderer (e.g. egui/eframe).
    pub fn from_existing(
        instance: Option<Instance>,
        adapter: Option<Adapter>,
        device: Device,
        queue: Queue,
        info: AdapterInfo,
    ) -> Self {
        let features = device.features();
        let limits = device.limits();
        Self {
            instance,
            adapter,
            device,
            queue,
            info,
            features,
            limits,
            profiler: None,
        }
    }

    /// Returns the shared `wgpu::Device`.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns the shared `wgpu::Queue`.
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// The compute-pass profiler, present only when profiling was enabled and supported.
    pub fn profiler(&self) -> Option<&GpuProfiler> {
        self.profiler.as_ref()
    }

    /// Timestamp writes for one compute pass, or `None` when profiling is off.
    ///
    /// Pass this straight to `ComputePassDescriptor::timestamp_writes` so instrumenting
    /// a pass stays a one-line change.
    pub fn timestamp_writes(&self, label: &str) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        self.profiler.as_ref()?.timestamp_writes(label)
    }

    /// Drains the recorded pass timings, if profiling is on.
    pub fn take_pass_timings(&self) -> anyhow::Result<Vec<PassTiming>> {
        match self.profiler.as_ref() {
            Some(profiler) => profiler.take(&self.device, &self.queue),
            None => Ok(Vec::new()),
        }
    }

    /// Adapter metadata handy for GUI display/logging.
    pub fn adapter_info(&self) -> &AdapterInfo {
        &self.info
    }

    /// `wgpu::Features` enabled on this context.
    pub fn features(&self) -> Features {
        self.features
    }

    /// `wgpu::Limits` negotiated for this context.
    pub fn limits(&self) -> &Limits {
        &self.limits
    }

    /// Returns the underlying `wgpu::Instance` if this context owns one.
    pub fn instance(&self) -> Option<&Instance> {
        self.instance.as_ref()
    }

    /// Returns the underlying adapter when available.
    pub fn adapter(&self) -> Option<&Adapter> {
        self.adapter.as_ref()
    }

    /// Generates a report of global internal counters.
    pub fn generate_report(&self) -> Option<GpuReport> {
        self.instance.as_ref().map(|i| {
            let report = i.generate_report();
            GpuReport {
                summary: format!("{:#?}", report),
            }
        })
    }
}

/// Pack little-endian RGBA bytes into a single `u32` per pixel.
///
/// Each returned element stores the four 8-bit color channels in the order
/// `R | G << 8 | B << 16 | A << 24`.
///
/// ```
/// use fcs_utils::gpu::{pack_rgba_pixels, unpack_rgba_pixels};
///
/// let rgba = [0x11, 0x22, 0x33, 0xff];
/// let packed = pack_rgba_pixels(&rgba);
/// assert_eq!(packed, [0xff33_2211]);
/// assert_eq!(unpack_rgba_pixels(&packed), rgba);
/// ```
pub fn pack_rgba_pixels(bytes: &[u8]) -> Vec<u32> {
    debug_assert!(
        bytes.len().is_multiple_of(4),
        "RGBA buffer must have a multiple of 4 elements"
    );
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| u32::from_le_bytes(*chunk))
        .collect()
}

/// Expand packed RGBA pixels back into a `Vec<u8>` buffer.
pub fn unpack_rgba_pixels(packed: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(packed.len() * 4);
    for value in packed {
        bytes.extend(value.to_le_bytes());
    }
    bytes
}

/// Tracks GPU initialization failures and reasons for CPU fallback.
#[derive(Debug, Error)]
pub enum GpuInitError {
    /// No adapter could be obtained from the requested backends.
    #[error("GPU adapter request failed for {backends:?}: {source}")]
    Adapter {
        /// Backends searched for an adapter.
        backends: Backends,
        /// Underlying adapter-request failure.
        #[source]
        source: RequestAdapterError,
    },
    /// The selected adapter does not support every required feature.
    #[error(
        "GPU adapter missing required features (requested={requested:?}, supported={supported:?})"
    )]
    MissingFeatures {
        /// Features required by the caller.
        requested: Features,
        /// Features advertised by the adapter.
        supported: Features,
    },
    /// Creating the logical GPU device failed.
    #[error("GPU device creation failed: {0}")]
    Device(#[from] RequestDeviceError),
    /// GPU use was disabled by configuration or the environment.
    #[error("GPU acceleration disabled")]
    Disabled,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_options_skip_gpu_setup() {
        let options = GpuContextOptions::disabled();
        match GpuContext::init_with_fallback(&options) {
            GpuAvailability::Disabled { .. } => {}
            other => panic!("expected GPU to be disabled, got {other:?}"),
        }
    }

    #[test]
    fn pack_unpack_rgba_roundtrip() {
        let original: Vec<u8> = vec![0xFF, 0x80, 0x40, 0xCC, 0x00, 0x01, 0x02, 0xFF];
        let packed = pack_rgba_pixels(&original);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_rgba_pixels(&packed);
        assert_eq!(unpacked, original);
    }

    #[test]
    fn pack_rgba_preserves_channel_order() {
        // Single pixel: R=1, G=2, B=3, A=4
        let bytes = vec![1u8, 2, 3, 4];
        let packed = pack_rgba_pixels(&bytes);
        assert_eq!(packed.len(), 1);
        // Little-endian: R | G<<8 | B<<16 | A<<24
        let expected = 1u32 | (2 << 8) | (3 << 16) | (4 << 24);
        assert_eq!(packed[0], expected);
    }

    #[test]
    fn windows_never_offers_vulkan_to_wgpu() {
        // Regression guard for the Store certification crash: Intel's Vulkan ICD
        // faults during adapter bring-up, so Windows builds must not enumerate
        // Vulkan at all. Everything else in the base set has to survive, or the
        // filter would be silently disabling working backends.
        let filtered = platform_safe_backends(Backends::all());

        if cfg!(target_os = "windows") {
            assert!(
                !filtered.contains(Backends::VULKAN),
                "Windows must not enumerate Vulkan: {filtered:?}"
            );
            assert!(
                filtered.contains(Backends::DX12),
                "DX12 is the Windows backend and must remain: {filtered:?}"
            );
            assert_eq!(
                filtered,
                Backends::all() - Backends::VULKAN,
                "only Vulkan should be removed"
            );
        } else {
            assert_eq!(
                filtered,
                Backends::all(),
                "non-Windows platforms must be untouched — Vulkan is the primary \
                 backend on Linux"
            );
        }

        // The shipped default is what actually reaches wgpu, so assert on it too
        // rather than only on the helper.
        assert_eq!(
            GpuContextOptions::default()
                .backends
                .contains(Backends::VULKAN),
            !cfg!(target_os = "windows")
        );
    }

    #[test]
    fn gpu_status_mode_as_str() {
        assert_eq!(GpuStatusMode::Pending.as_str(), "pending");
        assert_eq!(GpuStatusMode::Available.as_str(), "available");
        assert_eq!(GpuStatusMode::Disabled.as_str(), "disabled");
        assert_eq!(GpuStatusMode::Fallback.as_str(), "fallback");
        assert_eq!(GpuStatusMode::Error.as_str(), "error");
    }

    #[test]
    fn gpu_status_indicator_constructors() {
        let pending = GpuStatusIndicator::pending();
        assert_eq!(pending.mode, GpuStatusMode::Pending);
        assert!(pending.adapter_name.is_none());

        let avail =
            GpuStatusIndicator::available("RTX 4090", "Vulkan", None, Some(0x10DE), Some(0x2684));
        assert_eq!(avail.mode, GpuStatusMode::Available);
        assert!(avail.summary.contains("RTX 4090"));
        assert_eq!(avail.adapter_name.as_deref(), Some("RTX 4090"));
        assert_eq!(avail.vendor_id, Some(0x10DE));

        let disabled = GpuStatusIndicator::disabled("user flag");
        assert_eq!(disabled.mode, GpuStatusMode::Disabled);
        assert_eq!(disabled.detail.as_deref(), Some("user flag"));

        let fallback =
            GpuStatusIndicator::fallback("oom", Some("Intel".into()), Some("DX12".into()));
        assert_eq!(fallback.mode, GpuStatusMode::Fallback);
        assert_eq!(fallback.adapter_name.as_deref(), Some("Intel"));

        let error = GpuStatusIndicator::error("driver crash");
        assert_eq!(error.mode, GpuStatusMode::Error);
        assert_eq!(error.detail.as_deref(), Some("driver crash"));
    }

    #[test]
    fn gpu_availability_helpers() {
        let disabled = GpuAvailability::Disabled {
            reason: "test".to_string(),
        };
        assert!(!disabled.is_available());
        assert!(disabled.context().is_none());

        let err = GpuAvailability::Unavailable {
            error: GpuInitError::Disabled,
        };
        assert!(!err.is_available());
        assert!(err.context().is_none());
    }

    // ------------------------------------------------------------------
    // The context accessors and the Available arm of GpuAvailability. The
    // existing coverage above only ever builds the Disabled/Unavailable
    // variants, so `is_available` could return a constant `false` and
    // `context()` a constant `None` without any test noticing — and every
    // accessor on a live context was unreachable.
    // ------------------------------------------------------------------

    #[test]
    fn available_variant_reports_itself_available_and_yields_the_context() {
        let Some(ctx) = test_support::test_context() else {
            eprintln!("Skipping GpuContext accessor test: no adapter");
            return;
        };
        let availability = GpuAvailability::Available(ctx.clone());

        assert!(
            availability.is_available(),
            "the Available variant must report itself available"
        );
        let returned = availability
            .context()
            .expect("the Available variant must hand back its context");
        assert!(
            Arc::ptr_eq(returned, &ctx),
            "context() must return the very context it was given"
        );
    }

    #[test]
    fn context_accessors_describe_the_real_adapter() {
        let Some(ctx) = test_support::test_context() else {
            eprintln!("Skipping GpuContext accessor test: no adapter");
            return;
        };

        // A default-constructed Limits would report the downlevel minimums, so
        // requiring a real texture dimension distinguishes the accessor from a
        // fabricated default.
        let limits = ctx.limits();
        assert!(
            limits.max_texture_dimension_2d >= 2048,
            "limits look defaulted rather than negotiated: {}",
            limits.max_texture_dimension_2d
        );
        assert!(limits.max_buffer_size > 0);

        // features() is allowed to be empty (nothing optional is requested), so
        // assert it agrees with the adapter rather than that it is non-empty.
        let adapter = ctx.adapter().expect("context should own its adapter");
        assert!(
            adapter.features().contains(ctx.features()),
            "reported features must be a subset of what the adapter supports"
        );

        assert!(
            ctx.instance().is_some(),
            "a context built by initialize() owns its instance"
        );

        let info = ctx.adapter_info();
        assert!(
            !info.name.is_empty(),
            "adapter_info should carry a real adapter name"
        );

        let report = ctx
            .generate_report()
            .expect("a context with an instance can report counters");
        assert!(
            !report.summary.is_empty(),
            "generate_report must produce a non-empty summary"
        );

        // The downlevel defaults also clear the 2048 floor above, so compare with the device.
        assert_eq!(ctx.limits(), &ctx.device().limits());
        // `Instance` compares by identity: a freshly made one would not equal this.
        assert_eq!(ctx.instance(), ctx.instance());
    }

    /// Every GPU test here skips itself when the shared context fails to build, so a broken
    /// `initialize` would look like a machine without a GPU. Enumerate adapters independently.
    #[test]
    fn initialize_succeeds_whenever_an_adapter_exists() {
        let options = GpuContextOptions::default();
        let instance = Instance::new(InstanceDescriptor::new_without_display_handle());
        if block_on(instance.enumerate_adapters(options.backends)).is_empty() {
            return;
        }
        assert!(
            test_support::test_context().is_some(),
            "an adapter exists but the GPU context failed to initialize"
        );
    }

    #[test]
    fn optional_features_are_enabled_only_when_asked_for_and_supported() {
        let Some(plain) = test_support::test_context() else {
            return;
        };
        assert_eq!(
            plain.features(),
            Features::empty(),
            "default options ask for nothing optional, so nothing else may be enabled"
        );
        let adapter = plain.adapter().expect("context should own its adapter");
        if !adapter.features().contains(Features::TIMESTAMP_QUERY) {
            return;
        }
        let Some(profiled) = test_support::profiling_context() else {
            return;
        };
        assert!(profiled.features().contains(Features::TIMESTAMP_QUERY));
        assert!(profiled.profiler().is_some());
    }

    /// Only meaningful on a machine with both an integrated and a discrete adapter.
    #[test]
    fn power_preference_picks_the_matching_adapter_kind() {
        let options = GpuContextOptions::default();
        let instance = Instance::new(InstanceDescriptor::new_without_display_handle());
        let adapters = block_on(instance.enumerate_adapters(options.backends));
        let has = |kind| adapters.iter().any(|a| a.get_info().device_type == kind);
        if !(has(wgpu::DeviceType::IntegratedGpu) && has(wgpu::DeviceType::DiscreteGpu)) {
            return;
        }
        for (preference, kind) in [
            (PowerPreference::LowPower, wgpu::DeviceType::IntegratedGpu),
            (PowerPreference::HighPerformance, wgpu::DeviceType::DiscreteGpu),
        ] {
            let ctx = GpuContext::initialize(&GpuContextOptions {
                respect_env: false,
                power_preference: preference,
                ..GpuContextOptions::default()
            })
            .expect("an adapter of the preferred kind exists");
            assert_eq!(ctx.adapter_info().device_type, kind, "{preference:?}");
        }
    }

    #[test]
    fn a_forced_fallback_adapter_is_the_software_one() {
        let options = GpuContextOptions {
            respect_env: false,
            force_fallback_adapter: true,
            ..GpuContextOptions::default()
        };
        // Not every platform ships a software adapter; where one exists it must be the one used.
        if let Ok(ctx) = GpuContext::initialize(&options) {
            assert_eq!(ctx.adapter_info().device_type, wgpu::DeviceType::Cpu);
        }
    }

    #[test]
    fn disabled_and_error_indicators_carry_distinct_summaries_and_the_reason() {
        // Both constructors fill `summary` from a literal and `detail` from the
        // caller. Dropping the summary field would leave them indistinguishable
        // in the UI, since the mode alone is not rendered as text.
        let disabled = GpuStatusIndicator::disabled("switched off in settings");
        assert_eq!(disabled.mode, GpuStatusMode::Disabled);
        assert_eq!(disabled.summary, "GPU disabled");
        assert_eq!(disabled.detail.as_deref(), Some("switched off in settings"));

        let error = GpuStatusIndicator::error("no adapter found");
        assert_eq!(error.mode, GpuStatusMode::Error);
        assert_eq!(error.summary, "GPU unavailable");
        assert_eq!(error.detail.as_deref(), Some("no adapter found"));

        assert_ne!(
            disabled.summary, error.summary,
            "disabled and error must not read the same"
        );

        // Neither claims an adapter it does not have.
        for indicator in [&disabled, &error] {
            assert!(indicator.adapter_name.is_none());
            assert!(indicator.backend.is_none());
        }
    }

    /// Experiment 94's other half: that a wait which does not get what it wanted leaves
    /// the device usable. A zero-length deadline is as close as a test can get to a
    /// missed one on a working GPU -- on this hardware the copy has usually already
    /// landed, so the poll returns `Ok` and the branch is exercised by
    /// [`a_missed_deadline_is_named_as_one`] instead. What this pins is that whatever
    /// the short wait returned, the data still arrives on the next one.
    #[test]
    fn a_short_wait_leaves_the_device_usable() {
        let Some(ctx) = test_support::test_context() else {
            return;
        };
        let (device, queue) = (ctx.device(), ctx.queue());
        let payload: Vec<u32> = (0..4096u32).collect();
        let src = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("wait_probe_src"),
                contents: bytemuck::cast_slice(&payload),
                usage: wgpu::BufferUsages::COPY_SRC,
            },
        );
        let bytes = std::mem::size_of_val(payload.as_slice()) as wgpu::BufferAddress;
        let dst = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wait_probe_dst"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wait_probe"),
        });
        encoder.copy_buffer_to_buffer(&src, 0, &dst, 0, bytes);
        queue.submit(Some(encoder.finish()));

        let _ = wait_for_gpu_until(device, "wait probe", std::time::Duration::ZERO);

        // The point of the experiment: a wait that returned early must not poison the
        // device, and the work it was waiting for must still arrive.
        let (tx, rx) = std::sync::mpsc::channel();
        dst.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        wait_for_gpu(device, "wait probe retry").expect("the device is still usable");
        rx.recv().expect("callback").expect("map");
        let mapped = dst.slice(..).get_mapped_range().expect("mapped range");
        let read: Vec<u32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        dst.unmap();
        assert_eq!(
            read, payload,
            "the copy still lands after a missed deadline"
        );
    }

    /// A missed deadline has to be distinguishable from any other poll failure, because
    /// it is the only one that says "still running" rather than "broken".
    #[test]
    fn a_missed_deadline_is_named_as_one() {
        let timed_out = wait_error(
            &wgpu::PollError::Timeout,
            "batch readback",
            std::time::Duration::from_secs(30),
        )
        .to_string();
        assert!(
            timed_out.contains("timed out") && timed_out.contains("batch readback"),
            "{timed_out}"
        );
        assert!(
            timed_out.contains("30s"),
            "the deadline belongs in the message: {timed_out}"
        );

        let other = wait_error(
            &wgpu::PollError::WrongSubmissionIndex(4, 2),
            "batch readback",
            std::time::Duration::from_secs(30),
        )
        .to_string();
        assert!(
            !other.contains("timed out"),
            "a wgpu fault must not be reported as a slow device: {other}"
        );
    }
}
