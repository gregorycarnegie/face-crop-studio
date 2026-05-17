//! Common helpers shared across YuNet crates.

/// Shared color utilities.
pub mod color;
/// Application configuration and settings management.
pub mod config;
/// Image enhancement utilities (unsharp mask, contrast, exposure, etc.)
pub mod enhance;
/// Test fixture loading and path resolution.
pub mod fixtures;
/// Shared GPU context initialization and pooling helpers.
pub mod gpu;
/// Image loading, resizing, and tensor conversion.
pub mod image_utils;
/// Macros for reducing GPU shader boilerplate.
#[macro_use]
pub mod macros;

/// Data-driven mapping utilities (CSV/Excel/Parquet/SQLite ingestion).
#[cfg(feature = "mapping")]
pub mod mapping;
/// Image output helpers (encoding, metadata preservation).
pub mod output;
/// 2D point geometry primitives.
pub mod point;
/// Image quality analysis (Laplacian variance blur detection).
pub mod quality;
/// Shape metadata and masking helpers for custom crop geometry.
pub mod shape;
#[cfg(test)]
mod shape_tests;
/// Instrumentation helpers for optional performance tracing.
pub mod telemetry;
/// Webcam capture utilities for real-time face detection.
#[cfg(feature = "webcam")]
pub mod webcam;

use anyhow::Result;
pub use color::{
    RgbaColor, cmyk_to_rgb, hsl_to_rgb, hsv_to_rgb, parse_hex_color, rgb_to_cmyk, rgb_to_hsl,
    rgb_to_hsv,
};
pub use config::PositioningMode;
pub use enhance::{EnhancementSettings, WgpuEnhancer, apply_enhancements};
pub use fixtures::{
    fixture_path, fixtures_dir, load_fixture_bytes, load_fixture_image, load_fixture_json,
};
pub use gpu::{
    BatchCropRequest, GpuAvailability, GpuBatchCropper, GpuContext, GpuContextGuard,
    GpuContextOptions, GpuContextPool, GpuInitError, GpuPoolError, RedEye,
};
pub use image_utils::{
    SUPPORTED_IMAGE_EXTENSIONS, compute_resize_scales, dynamic_to_bgr_chw, is_supported_image_path,
    load_image, load_image_raw, resize_image, rgb_to_bgr_chw,
};
use log::LevelFilter;
#[cfg(feature = "mapping")]
pub use mapping::{
    ColumnSelector, MappingCatalog, MappingEntry, MappingFormat, MappingPreview,
    MappingReadOptions, detect_format as detect_mapping_format, inspect_mapping_sources,
    list_sqlite_tables, load_mapping_entries, load_mapping_preview,
};
pub use output::{
    ImageFormatHint, MetadataContext, OutputOptions, PngCompression, append_suffix_to_filename,
    save_dynamic_image,
};
pub use quality::{Quality, QualityFilter, estimate_sharpness, laplacian_variance};
pub use shape::{
    CropShape, PolygonCornerStyle, apply_shape_mask, apply_shape_mask_dynamic,
    outline_points_for_rect,
};
use std::path::{Path, PathBuf};
pub use telemetry::{
    TimingGuard, configure as configure_telemetry, telemetry_allows, telemetry_enabled,
    telemetry_level, timing_guard, timing_guard_if,
};
#[cfg(feature = "webcam")]
pub use webcam::{WebcamCapture, list_webcam_devices};

/// Initialize logging once for CLI and GUI environments.
///
/// This function respects the `RUST_LOG` environment variable if it is set.
/// Otherwise, it falls back to the provided default filter level.
///
/// # Arguments
///
/// * `default_filter` - The `LevelFilter` to use if `RUST_LOG` is not set.
pub fn init_logging(default_filter: LevelFilter) -> Result<()> {
    let mut builder = env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or(default_filter.as_str()),
    );
    builder.filter_module("fcs::telemetry", LevelFilter::Trace);

    if builder.try_init().is_err() {
        // Logger already initialized; nothing to do.
    }
    Ok(())
}

/// Validate that a path exists and resolve it to an absolute path.
///
/// # Arguments
///
/// * `path` - The path to validate and normalize.
pub fn normalize_path<P: AsRef<Path>>(path: P) -> Result<PathBuf> {
    let path = path.as_ref();
    anyhow::ensure!(path.exists(), "path does not exist: {}", path.display());
    Ok(path.canonicalize()?)
}

/// Subdirectory under `/usr/share/` (or AppImage `usr/share/`) where Linux
/// packages install data files alongside the binary in `/usr/bin/`.
const LINUX_SHARE_DIR: &str = "face-crop-studio";

/// Resolve a (possibly relative) data file path to a usable filesystem path.
///
/// Launchers across the supported platforms set CWD inconsistently — macOS
/// `.app` bundles run with CWD=`/`, AppImages mount under `/tmp/.mount_*`, and
/// `.deb` installs put the binary in `/usr/bin/` while data goes to
/// `/usr/share/<app>/`. This helper walks a small list of fallback locations
/// so a configured relative path resolves the same way on every platform.
///
/// Lookup order:
/// 1. The path as given (relative to CWD).
/// 2. `<exe_dir>/<path>` — Windows portable zip and macOS `Contents/MacOS/`.
/// 3. `<exe_dir>/../share/face-crop-studio/<path>` — Linux FHS layout used by
///    both AppImage and `.deb` installs.
///
/// Caller owns error reporting if none of the candidates exist; the original
/// path is returned so error messages stay informative.
pub fn resolve_data_path<P: AsRef<Path>>(path: P) -> PathBuf {
    let path = path.as_ref();
    if path.exists() {
        if path.is_relative() {
            return path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        }
        return path.to_path_buf();
    }
    if path.is_relative()
        && let Ok(exe) = std::env::current_exe()
        && let Some(exe_dir) = exe.parent()
    {
        let next_to_exe = exe_dir.join(path);
        if next_to_exe.exists() {
            return next_to_exe;
        }
        let share = exe_dir.join("../share").join(LINUX_SHARE_DIR).join(path);
        if share.exists() {
            return share;
        }
    }
    path.to_path_buf()
}

#[cfg(test)]
mod path_tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn resolve_data_path_returns_existing_absolute_path_unchanged() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("real.onnx");
        fs::write(&file, b"").unwrap();
        assert_eq!(resolve_data_path(&file), file);
    }

    #[test]
    fn resolve_data_path_returns_existing_relative_path_unchanged() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("present.onnx");
        fs::write(&file, b"").unwrap();
        let prev = std::env::current_dir().unwrap();
        std::env::set_current_dir(&dir).unwrap();
        let resolved = resolve_data_path(Path::new("present.onnx"));
        std::env::set_current_dir(prev).unwrap();
        assert!(resolved.exists());
        assert_eq!(resolved.file_name().unwrap(), "present.onnx");
    }

    #[test]
    fn resolve_data_path_returns_missing_path_unchanged_for_error_messaging() {
        let path = Path::new("definitely/not/here.onnx");
        let resolved = resolve_data_path(path);
        assert_eq!(resolved, path);
    }
}
