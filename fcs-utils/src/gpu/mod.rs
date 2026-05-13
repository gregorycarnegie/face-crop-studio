//! GPU context management helpers built on top of `wgpu`.
//!
//! This module keeps GPU initialization code in one place so the CLI and GUI
//! can share the same device/queue plumbing while still offering a CPU
//! fallback when no compatible adapter is present.

/// Combined resize + RGB→BGR + HWC→CHW WGSL compute shader.
pub const PREPROCESS_WGSL: &str = include_str!("preprocess.wgsl");
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
/// Batched crop-and-resize shader.
pub const CROP_WGSL: &str = include_str!("crop.wgsl");

pub mod pixel_adjust;
pub use pixel_adjust::GpuPixelAdjust;
pub mod gaussian_blur;
pub use gaussian_blur::GpuGaussianBlur;
pub mod bilateral_filter;
pub use bilateral_filter::GpuBilateralFilter;
pub mod background_blur;
pub use background_blur::GpuBackgroundBlur;
pub mod red_eye;
pub use red_eye::{GpuRedEyeRemoval, RedEye};
pub mod shape_mask;
pub use shape_mask::GpuShapeMask;
pub mod hist_equalize;
pub use hist_equalize::GpuHistogramEqualizer;
pub mod crop_batch;
pub use crop_batch::{BatchCropRequest, GpuBatchCropper};
pub mod buffer_pool;
pub use buffer_pool::GpuBufferPool;
pub mod memory;
pub use memory::get_available_vram;

use std::{fmt, num::NonZeroUsize, sync::Arc};

#[derive(Debug, Clone)]
pub struct GpuReport {
    pub summary: String,
}

use crate::telemetry::telemetry_allows;
use async_channel::{Receiver, Sender, TryRecvError, TrySendError, bounded};
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
}

impl Default for GpuContextOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            respect_env: true,
            backends: Backends::PRIMARY,
            flags: InstanceFlags::from_build_config(),
            power_preference: PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            required_features: Features::empty(),
            optional_features: Features::empty(),
            required_limits: None,
            dx12_shader_compiler: Dx12Compiler::default(),
            label: Some("YuNet GPU context".to_string()),
            memory_hints: None,
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
    Disabled { reason: String },
    /// GPU initialization failed; callers should fall back to CPU.
    Unavailable { error: GpuInitError },
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
    pub fn emit_telemetry(&self, pool_capacity: Option<usize>, pool_available: Option<usize>) {
        emit_gpu_status_event(self, pool_capacity, pool_available);
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
    pool_capacity: Option<usize>,
    pool_available: Option<usize>,
}

fn emit_gpu_status_event(
    status: &GpuStatusIndicator,
    pool_capacity: Option<usize>,
    pool_available: Option<usize>,
) {
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
        pool_capacity,
        pool_available,
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

        let instance = Instance::new(instance_desc);
        let adapter = block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: options.power_preference,
            force_fallback_adapter: options.force_fallback_adapter,
            compatible_surface: None,
        }))
        .map_err(|source| GpuInitError::Adapter { backends, source })?;

        let info = adapter.get_info();
        let supported_features = adapter.features();

        if !supported_features.contains(options.required_features) {
            return Err(GpuInitError::MissingFeatures {
                requested: options.required_features,
                supported: supported_features,
            });
        }

        let mut features = options.required_features;
        let optional = options.optional_features & supported_features;
        if !optional.is_empty() {
            debug!(
                target: "fcs::gpu",
                "Enabling optional GPU features: {:?}", optional
            );
            features |= optional;
        }

        let missing_optional = options.optional_features & !supported_features;
        if !missing_optional.is_empty() {
            debug!(
                target: "fcs::gpu",
                "Skipping unsupported optional GPU features: {:?}", missing_optional
            );
        }

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

        let (device, queue) =
            block_on(adapter.request_device(&device_desc)).map_err(GpuInitError::from)?;

        info!(
            target: "fcs::gpu",
            "Using GPU adapter '{}' ({:?}/{:?}) with features {:?}",
            info.name, info.backend, info.device_type, features
        );

        Ok(Self {
            instance: Some(instance),
            adapter: Some(adapter),
            device,
            queue,
            info,
            features,
            limits,
        })
    }

    /// Attempt to create a GPU context and gracefully fall back to CPU if that fails.
    pub fn init_with_fallback(options: &GpuContextOptions) -> GpuAvailability {
        if !options.enabled {
            return GpuAvailability::Disabled {
                reason: "GPU acceleration disabled via configuration".to_string(),
            };
        }

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
pub fn pack_rgba_pixels(bytes: &[u8]) -> Vec<u32> {
    debug_assert!(
        bytes.len().is_multiple_of(4),
        "RGBA buffer must have a multiple of 4 elements"
    );
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
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
    #[error("GPU adapter request failed for {backends:?}: {source}")]
    Adapter {
        backends: Backends,
        #[source]
        source: RequestAdapterError,
    },
    #[error(
        "GPU adapter missing required features (requested={requested:?}, supported={supported:?})"
    )]
    MissingFeatures {
        requested: Features,
        supported: Features,
    },
    #[error("GPU device creation failed: {0}")]
    Device(#[from] RequestDeviceError),
    #[error("GPU acceleration disabled")]
    Disabled,
}

/// Concurrency guard that keeps GPU contexts balanced across worker threads.
#[derive(Clone)]
pub struct GpuContextPool {
    size: NonZeroUsize,
    sender: Sender<Arc<GpuContext>>,
    receiver: Receiver<Arc<GpuContext>>,
}

impl GpuContextPool {
    /// Create a bounded pool backed by the provided GPU context.
    pub fn new(context: Arc<GpuContext>, size: NonZeroUsize) -> Result<Self, GpuPoolError> {
        let (sender, receiver) = bounded(size.get());
        for _ in 0..size.get() {
            sender
                .send_blocking(context.clone())
                .map_err(|_| GpuPoolError::Closed)?;
        }

        Ok(Self {
            size,
            sender,
            receiver,
        })
    }

    /// Acquire a GPU context using a blocking call (handy for rayon/CLI threads).
    pub fn acquire(&self) -> Result<GpuContextGuard, GpuPoolError> {
        let ctx = self
            .receiver
            .recv_blocking()
            .map_err(|_| GpuPoolError::Closed)?;
        Ok(GpuContextGuard::new(ctx, &self.sender))
    }

    /// Acquire a GPU context asynchronously.
    pub async fn acquire_async(&self) -> Result<GpuContextGuard, GpuPoolError> {
        let ctx = self
            .receiver
            .recv()
            .await
            .map_err(|_| GpuPoolError::Closed)?;
        Ok(GpuContextGuard::new(ctx, &self.sender))
    }

    /// Try acquiring without blocking, returning `None` when the pool is currently empty.
    pub fn try_acquire(&self) -> Result<Option<GpuContextGuard>, GpuPoolError> {
        match self.receiver.try_recv() {
            Ok(ctx) => Ok(Some(GpuContextGuard::new(ctx, &self.sender))),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Closed) => Err(GpuPoolError::Closed),
        }
    }

    /// Total capacity enforced by this pool.
    pub fn capacity(&self) -> usize {
        self.size.get()
    }

    /// Current number of idle contexts.
    pub fn available(&self) -> usize {
        self.receiver.len()
    }
}

impl fmt::Debug for GpuContextPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuContextPool")
            .field("capacity", &self.capacity())
            .field("available", &self.available())
            .finish()
    }
}

/// RAII guard for a borrowed GPU context from a [`GpuContextPool`].
pub struct GpuContextGuard {
    context: Option<Arc<GpuContext>>,
    sender: Sender<Arc<GpuContext>>,
}

impl GpuContextGuard {
    fn new(context: Arc<GpuContext>, sender: &Sender<Arc<GpuContext>>) -> Self {
        Self {
            context: Some(context),
            sender: sender.clone(),
        }
    }

    /// Returns a reference to the inner GPU context.
    pub fn context(&self) -> &GpuContext {
        self
    }
}

impl std::ops::Deref for GpuContextGuard {
    type Target = GpuContext;

    fn deref(&self) -> &Self::Target {
        self.context
            .as_deref()
            .expect("GPU context guard should always hold a context")
    }
}

impl Drop for GpuContextGuard {
    fn drop(&mut self) {
        if let Some(ctx) = self.context.take()
            && let Err(err) = self.sender.try_send(ctx)
        {
            match err {
                TrySendError::Full(ctx) => {
                    if let Err(send_err) = self.sender.send_blocking(ctx) {
                        debug!(
                            target: "fcs::gpu",
                            "GPU pool closed while returning guard: {send_err}"
                        );
                    }
                }
                TrySendError::Closed(_) => debug!(
                    target: "fcs::gpu",
                    "GPU pool closed; dropping context guard."
                ),
            }
        }
    }
}

/// Errors that can occur while interacting with a [`GpuContextPool`].
#[derive(Debug, Error)]
pub enum GpuPoolError {
    #[error("GPU context pool has been closed")]
    Closed,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;

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

    #[test]
    fn gpu_context_pool_acquire_and_release() {
        let ctx = match GpuContext::init_with_fallback(&GpuContextOptions::default()) {
            GpuAvailability::Available(ctx) => ctx,
            _ => {
                eprintln!("Skipping GpuContextPool test: no GPU");
                return;
            }
        };

        let pool = GpuContextPool::new(ctx, NonZeroUsize::new(2).unwrap()).expect("pool creation");

        assert_eq!(pool.capacity(), 2);
        assert_eq!(pool.available(), 2);

        let guard1 = pool.acquire().expect("acquire 1");
        assert_eq!(pool.available(), 1);

        let guard2 = pool.acquire().expect("acquire 2");
        assert_eq!(pool.available(), 0);

        // try_acquire on empty pool returns None
        assert!(pool.try_acquire().expect("try_acquire ok").is_none());

        drop(guard1);
        assert_eq!(pool.available(), 1);
        drop(guard2);
        assert_eq!(pool.available(), 2);
    }

    #[test]
    fn gpu_context_guard_deref() {
        let ctx = match GpuContext::init_with_fallback(&GpuContextOptions::default()) {
            GpuAvailability::Available(ctx) => ctx,
            _ => {
                eprintln!("Skipping GpuContextGuard test: no GPU");
                return;
            }
        };
        let pool = GpuContextPool::new(ctx, NonZeroUsize::new(1).unwrap()).unwrap();
        let guard = pool.acquire().unwrap();
        // Deref should give us access to GpuContext methods
        let _ = guard.adapter_info();
    }
}
