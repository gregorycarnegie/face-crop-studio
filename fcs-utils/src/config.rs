//! Shared configuration types consumed across the YuNet workspace.
//!
//! These structures provide a common representation for inference, detection, cropping, and
//! enhancement settings that can be serialized to disk and reused by CLI and GUI front-ends.

use crate::{
    color::RgbaColor,
    gpu::GpuContextOptions,
    output::{ImageFormatHint, PngCompression},
    quality::Quality,
    shape::CropShape,
};

use anyhow::{Context, Result};
use log::LevelFilter;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    env, fmt, fs,
    path::{Path, PathBuf},
    str::FromStr,
};

/// Default input width in pixels.
pub const DEFAULT_INPUT_WIDTH: u32 = 640;
/// Default input height in pixels.
pub const DEFAULT_INPUT_HEIGHT: u32 = 640;
/// Default minimum confidence score for a detection to be considered valid.
pub const DEFAULT_SCORE_THRESHOLD: f32 = 0.9;
/// Default threshold for non-maximum suppression.
pub const DEFAULT_NMS_THRESHOLD: f32 = 0.3;
/// Default maximum number of detections to return.
pub const DEFAULT_TOP_K: usize = 5_000;
/// Default path to the YuNet ONNX model.
pub const DEFAULT_MODEL_PATH: &str = "models/face_detection_yunet_2023mar_640.onnx";
/// Default path for persisted GUI settings.
pub const DEFAULT_SETTINGS_PATH: &str = "config/gui_settings.json";

/// Shared detection parameters that should mirror YuNet defaults.
///
/// These settings directly control the behavior of the post-processing steps,
/// such as non-maximum suppression (NMS) and score filtering.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DetectionSettings {
    /// Minimum confidence score for a detection to be considered valid.
    pub score_threshold: f32,
    /// Threshold for non-maximum suppression to merge overlapping bounding boxes.
    pub nms_threshold: f32,
    /// The maximum number of detections to return.
    pub top_k: usize,
}

impl Default for DetectionSettings {
    fn default() -> Self {
        Self {
            score_threshold: DEFAULT_SCORE_THRESHOLD,
            nms_threshold: DEFAULT_NMS_THRESHOLD,
            top_k: DEFAULT_TOP_K,
        }
    }
}

impl DetectionSettings {
    /// Clamp values to valid ranges, replacing NaN/infinity with defaults.
    pub fn sanitize(&mut self) {
        if !self.score_threshold.is_finite() {
            self.score_threshold = DEFAULT_SCORE_THRESHOLD;
        }
        self.score_threshold = self.score_threshold.clamp(0.0, 1.0);

        if !self.nms_threshold.is_finite() {
            self.nms_threshold = DEFAULT_NMS_THRESHOLD;
        }
        self.nms_threshold = self.nms_threshold.clamp(0.0, 1.0);
    }
}

/// Inference input resolution in pixels (width x height).
///
/// The input image will be resized to these dimensions before being passed to the model.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "snake_case")]
pub enum ResizeQuality {
    /// Preserve visual quality when resizing (default, Triangle filter).
    #[default]
    Quality,
    /// Prioritize throughput for batch inference (Nearest filter).
    Speed,
}

impl ResizeQuality {
    pub const fn as_label(self) -> &'static str {
        match self {
            ResizeQuality::Quality => "Quality",
            ResizeQuality::Speed => "Speed",
        }
    }
}

impl fmt::Display for ResizeQuality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                ResizeQuality::Quality => "quality",
                ResizeQuality::Speed => "speed",
            }
        )
    }
}

impl FromStr for ResizeQuality {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "quality" => Ok(ResizeQuality::Quality),
            "speed" => Ok(ResizeQuality::Speed),
            other => Err(format!(
                "invalid resize quality '{other}'; expected 'quality' or 'speed'"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct InputDimensions {
    pub width: u32,
    pub height: u32,
    /// Choose between quality-focused or speed-focused resizing.
    pub resize_quality: ResizeQuality,
}

impl Default for InputDimensions {
    fn default() -> Self {
        Self {
            width: DEFAULT_INPUT_WIDTH,
            height: DEFAULT_INPUT_HEIGHT,
            resize_quality: ResizeQuality::Speed,
        }
    }
}

impl InputDimensions {
    /// Replace zero dimensions with defaults.
    pub fn sanitize(&mut self) {
        if self.width == 0 {
            self.width = DEFAULT_INPUT_WIDTH;
        }
        if self.height == 0 {
            self.height = DEFAULT_INPUT_HEIGHT;
        }
    }
}

/// How to position the face within the crop region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PositioningMode {
    /// Center the face in the crop region.
    #[default]
    Center,
    /// Place the face roughly at the upper third (rule of thirds).
    #[serde(alias = "rule_of_thirds", alias = "ruleofthirds", alias = "thirds")]
    RuleOfThirds,
    /// Use custom offsets (fractions) to nudge the face relative to crop center.
    Custom,
}

/// Settings for face cropping operations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct CropSettings {
    /// Crop preset name (e.g., "linkedin", "passport", "custom")
    pub preset: String,
    /// Output width in pixels (used when preset is "custom")
    pub output_width: u32,
    /// Output height in pixels (used when preset is "custom")
    pub output_height: u32,
    /// Face height as percentage of output height (0-100)
    pub face_height_pct: f32,
    /// Where to place the face within the crop region.
    pub positioning_mode: PositioningMode,
    /// Vertical offset for custom positioning (-1.0 to 1.0)
    pub vertical_offset: f32,
    /// Horizontal offset for custom positioning (-1.0 to 1.0)
    pub horizontal_offset: f32,
    /// Background color applied when the crop extends outside the source image bounds.
    pub fill_color: RgbaColor,
    /// Output image format.
    pub output_format: ImageFormatHint,
    /// JPEG quality (1-100, only used when format is jpeg)
    pub jpeg_quality: u8,
    /// PNG compression strategy.
    pub png_compression: PngCompression,
    /// WebP quality (0-100, lossy encoding)
    pub webp_quality: u8,
    /// Automatically detect output format from the file extension.
    pub auto_detect_format: bool,
    /// Metadata behavior for exported crops.
    pub metadata: MetadataSettings,
    /// Quality-based automation options.
    pub quality_rules: QualityAutomationSettings,
    /// Geometric shape applied to the exported crop.
    pub shape: CropShape,
    /// Softness of the vignette effect (0.0 to 1.0).
    pub vignette_softness: f32,
    /// Maximum intensity of the vignette effect (0.0 to 1.0).
    pub vignette_intensity: f32,
    /// Color of the vignette effect.
    pub vignette_color: RgbaColor,
    /// Rotate the crop so the eye line is horizontal.
    pub eye_line_align: bool,
    /// Apply EXIF orientation tag when loading the source image.
    pub auto_orient_exif: bool,
}

impl CropSettings {
    /// Clamp values to sensible ranges.
    pub fn sanitize(&mut self) {
        self.shape = self.shape.sanitized();
        self.vignette_softness = self.vignette_softness.clamp(0.0, 1.0);
        self.vignette_intensity = self.vignette_intensity.clamp(0.0, 1.0);
        self.face_height_pct = self.face_height_pct.clamp(1.0, 100.0);
        self.vertical_offset = self.vertical_offset.clamp(-1.0, 1.0);
        self.horizontal_offset = self.horizontal_offset.clamp(-1.0, 1.0);
        if self.output_width == 0 {
            self.output_width = 512;
        }
        if self.output_height == 0 {
            self.output_height = 512;
        }
    }
}

/// Settings for image enhancement operations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct EnhanceSettings {
    /// Enable enhancements
    pub enabled: bool,
    /// Apply histogram-equalization based auto color correction
    pub auto_color: bool,
    /// Exposure adjustment in stops (-2.0 to 2.0)
    pub exposure_stops: f32,
    /// Additional brightness offset (-100 to 100)
    pub brightness: i32,
    /// Contrast multiplier (0.5 to 2.0)
    pub contrast: f32,
    /// Saturation multiplier (0.0 to 2.5)
    pub saturation: f32,
    /// Sharpness (0.0 to 2.0)
    pub sharpness: f32,
    /// Skin smoothing strength (0.0 to 1.0)
    pub skin_smooth: f32,
    /// Enable automated red-eye removal
    pub red_eye_removal: bool,
    /// Enable background blur (portrait mode effect)
    pub background_blur: bool,
}

impl Default for CropSettings {
    fn default() -> Self {
        Self {
            preset: "linkedin".to_string(),
            output_width: 400,
            output_height: 400,
            face_height_pct: 70.0,
            positioning_mode: PositioningMode::default(),
            vertical_offset: 0.0,
            horizontal_offset: 0.0,
            fill_color: RgbaColor::default(),
            output_format: ImageFormatHint::default(),
            jpeg_quality: 90,
            png_compression: PngCompression::default(),
            webp_quality: 90,
            auto_detect_format: true,
            metadata: MetadataSettings::default(),
            quality_rules: QualityAutomationSettings::default(),
            shape: CropShape::Rectangle,
            vignette_softness: 0.0,
            vignette_intensity: 1.0,
            vignette_color: RgbaColor::opaque(0, 0, 0),
            eye_line_align: false,
            auto_orient_exif: true,
        }
    }
}

/// How metadata should be handled for exported crops.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum MetadataMode {
    #[default]
    Preserve,
    Strip,
    Custom,
}

impl std::str::FromStr for MetadataMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "preserve" => Ok(MetadataMode::Preserve),
            "strip" => Ok(MetadataMode::Strip),
            "custom" => Ok(MetadataMode::Custom),
            other => Err(format!("unknown metadata mode '{}'", other)),
        }
    }
}

/// Metadata configuration for exported crops.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct MetadataSettings {
    /// Desired metadata strategy.
    pub mode: MetadataMode,
    /// Include crop settings (size, offsets, preset) as custom metadata.
    pub include_crop_settings: bool,
    /// Include detection quality metrics as custom metadata.
    pub include_quality_metrics: bool,
    /// Arbitrary user-defined metadata key/value pairs.
    pub custom_tags: BTreeMap<String, String>,
}

impl Default for MetadataSettings {
    fn default() -> Self {
        Self {
            mode: MetadataMode::Preserve,
            include_crop_settings: true,
            include_quality_metrics: true,
            custom_tags: BTreeMap::new(),
        }
    }
}

/// Automation options driven by quality analysis.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct QualityAutomationSettings {
    /// Automatically select the highest quality face when multiple are detected.
    pub auto_select_best_face: bool,
    /// Minimum quality required to keep a crop.
    pub min_quality: Option<Quality>,
    /// Skip exporting entirely when no face meets `Quality::High`.
    pub auto_skip_no_high_quality: bool,
    /// Append a quality suffix (e.g., `_highq`) to exported filenames.
    pub quality_suffix: bool,
}

impl EnhanceSettings {
    /// Map to the processing struct used by `apply_enhancements`.
    pub fn to_enhancement_settings(&self) -> crate::enhance::EnhancementSettings {
        crate::enhance::EnhancementSettings {
            auto_color: self.auto_color,
            exposure_stops: self.exposure_stops,
            brightness: self.brightness,
            contrast: self.contrast,
            saturation: self.saturation,
            unsharp_amount: 0.0,
            unsharp_radius: 1.0,
            sharpness: self.sharpness,
            skin_smooth_amount: self.skin_smooth,
            skin_smooth_sigma_space: 3.0,
            skin_smooth_sigma_color: 25.0,
            red_eye_removal: self.red_eye_removal,
            red_eye_threshold: 1.5,
            background_blur: self.background_blur,
            background_blur_radius: 15.0,
            background_blur_mask_size: 0.6,
        }
    }
}

impl Default for EnhanceSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            auto_color: false,
            exposure_stops: 0.0,
            brightness: 0,
            contrast: 1.0,
            saturation: 1.0,
            sharpness: 0.0,
            skin_smooth: 0.0,
            red_eye_removal: false,
            background_blur: false,
        }
    }
}

/// Settings controlling optional runtime telemetry.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TelemetrySettings {
    /// Whether telemetry timing logs are enabled.
    pub enabled: bool,
    /// Logging level for telemetry output (error, warn, info, debug, trace).
    pub level: String,
}

impl Default for TelemetrySettings {
    fn default() -> Self {
        Self {
            enabled: false,
            level: "debug".to_string(),
        }
    }
}

impl TelemetrySettings {
    /// Resolve the configured level string into a `LevelFilter`.
    pub fn level_filter(&self) -> LevelFilter {
        match self.level.trim().to_ascii_lowercase().as_str() {
            "off" => LevelFilter::Off,
            "error" => LevelFilter::Error,
            "warn" | "warning" => LevelFilter::Warn,
            "info" => LevelFilter::Info,
            "trace" => LevelFilter::Trace,
            _ => LevelFilter::Debug,
        }
    }

    /// Update the level string from a `LevelFilter` value.
    pub fn set_level(&mut self, level: LevelFilter) {
        let label = match level {
            LevelFilter::Off => "off",
            LevelFilter::Error => "error",
            LevelFilter::Warn => "warn",
            LevelFilter::Info => "info",
            LevelFilter::Debug => "debug",
            LevelFilter::Trace => "trace",
        };
        self.level = label.to_string();
    }
}

/// Persistent application settings consumed by CLI and GUI front ends.
///
/// This struct aggregates all user-configurable parameters, allowing them to be
/// loaded from and saved to a JSON file.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AppSettings {
    /// Optional override for the YuNet ONNX model path.
    /// If `None`, a default path is used.
    pub model_path: Option<String>,
    /// The input dimensions for model inference.
    pub input: InputDimensions,
    /// The parameters for detection post-processing.
    pub detection: DetectionSettings,
    /// The parameters for face cropping.
    pub crop: CropSettings,
    /// The parameters for image enhancement.
    pub enhance: EnhanceSettings,
    /// Telemetry and diagnostics preferences.
    pub telemetry: TelemetrySettings,
    /// Batch processing logging settings.
    pub batch_logging: BatchLoggingSettings,
    /// GPU runtime preferences shared across CLI and GUI.
    pub gpu: GpuSettings,
}

impl Default for AppSettings {
    fn default() -> Self {
        let mut settings = Self {
            model_path: Some(DEFAULT_MODEL_PATH.into()),
            input: InputDimensions::default(),
            detection: DetectionSettings::default(),
            crop: CropSettings::default(),
            enhance: EnhanceSettings::default(),
            telemetry: TelemetrySettings::default(),
            batch_logging: BatchLoggingSettings::default(),
            gpu: GpuSettings::default(),
        };
        settings.crop.sanitize();
        settings
    }
}

/// Format for batch failure logs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum BatchLogFormat {
    #[default]
    Json,
    Csv,
}

impl fmt::Display for BatchLogFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BatchLogFormat::Json => write!(f, "JSON"),
            BatchLogFormat::Csv => write!(f, "CSV"),
        }
    }
}

/// Settings for logging failed batch operations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct BatchLoggingSettings {
    /// Whether to log failed items to a file.
    pub enabled: bool,
    /// The format of the log file.
    pub format: BatchLogFormat,
}

impl Default for BatchLoggingSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            format: BatchLogFormat::Json,
        }
    }
}

impl AppSettings {
    /// Load settings from a JSON file.
    ///
    /// If the file does not exist or cannot be parsed, an error is returned.
    /// If the `model_path` is missing from the JSON, it falls back to the default.
    pub fn load_from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let contents = fs::read_to_string(path)
            .with_context(|| format!("failed to read settings file {}", path.display()))?;
        let mut settings: AppSettings = serde_json::from_str(&contents)
            .with_context(|| format!("failed to parse settings JSON at {}", path.display()))?;

        if settings.model_path.is_none() {
            settings.model_path = Some(AppSettings::default().model_path.unwrap());
        }

        settings.input.sanitize();
        settings.detection.sanitize();
        settings.crop.sanitize();

        Ok(settings)
    }

    /// Serialize settings to disk in pretty-printed JSON.
    ///
    /// This will overwrite the file if it already exists.
    pub fn save_to_path<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        let payload =
            serde_json::to_string_pretty(self).context("failed to serialize settings JSON")?;
        fs::write(path, payload)
            .with_context(|| format!("failed to write settings file {}", path.display()))?;
        Ok(())
    }
}

/// Returns the default path for persisted application settings (`config/gui_settings.json`).
pub fn default_settings_path() -> PathBuf {
    env::current_dir()
        .map(|dir| dir.join(DEFAULT_SETTINGS_PATH))
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_SETTINGS_PATH))
}

/// GPU-specific runtime preferences.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GpuSettings {
    /// Whether GPU acceleration should be attempted (auto-detect by default).
    pub enabled: bool,
    /// Respect `WGPU_*` environment overrides when initializing the backend.
    pub respect_env: bool,
    /// Execute YuNet inference on the GPU when supported.
    pub inference: bool,
    /// Use GPU for image preprocessing (resize, color conversion).
    pub preprocessing: bool,
}

impl Default for GpuSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            respect_env: true,
            inference: true,
            preprocessing: true, // Enable GPU preprocessing by default when GPU is available
        }
    }
}

impl From<GpuSettings> for GpuContextOptions {
    fn from(settings: GpuSettings) -> Self {
        GpuContextOptions {
            enabled: settings.enabled,
            respect_env: settings.respect_env,
            ..Default::default()
        }
    }
}

impl From<&GpuSettings> for GpuContextOptions {
    fn from(settings: &GpuSettings) -> Self {
        GpuContextOptions {
            enabled: settings.enabled,
            respect_env: settings.respect_env,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn default_settings_round_trip() {
        let file = NamedTempFile::new().expect("tempfile");
        let settings = AppSettings::default();
        settings.save_to_path(file.path()).expect("save");

        let loaded = AppSettings::load_from_path(file.path()).expect("load");
        assert_eq!(loaded.input, settings.input);
        assert_eq!(loaded.detection.top_k, settings.detection.top_k);
        assert_eq!(loaded.model_path, settings.model_path);
        assert_eq!(loaded.telemetry.enabled, settings.telemetry.enabled);
        assert_eq!(loaded.telemetry.level, settings.telemetry.level);
        assert_eq!(loaded.gpu.enabled, settings.gpu.enabled);
        assert_eq!(loaded.gpu.respect_env, settings.gpu.respect_env);
    }

    #[test]
    fn missing_model_path_uses_default() {
        let file = NamedTempFile::new().expect("tempfile");
        let json = r#"{
            "input": { "width": 640, "height": 640 },
            "detection": { "score_threshold": 0.8, "nms_threshold": 0.25, "top_k": 123 }
        }"#;
        fs::write(file.path(), json).expect("write custom settings");

        let loaded = AppSettings::load_from_path(file.path()).expect("load");
        assert_eq!(
            loaded.input,
            InputDimensions {
                width: 640,
                height: 640,
                resize_quality: ResizeQuality::Speed,
            }
        );
        assert_eq!(loaded.detection.top_k, 123);
        assert!(loaded.model_path.is_some());
        assert!(!loaded.telemetry.enabled);
        assert_eq!(loaded.telemetry.level_filter(), LevelFilter::Debug);
        assert!(loaded.gpu.enabled);
        assert!(loaded.gpu.respect_env);
    }

    #[test]
    fn detection_settings_sanitize_clamps_and_fixes_nan() {
        let mut s = DetectionSettings {
            score_threshold: f32::NAN,
            nms_threshold: 1.5,
            top_k: 1,
        };
        s.sanitize();
        assert!(s.score_threshold.is_finite());
        assert_eq!(s.score_threshold, DEFAULT_SCORE_THRESHOLD);
        assert_eq!(s.nms_threshold, 1.0);

        let mut s2 = DetectionSettings {
            score_threshold: -0.1,
            nms_threshold: f32::INFINITY,
            top_k: 1,
        };
        s2.sanitize();
        assert_eq!(s2.score_threshold, 0.0);
        assert_eq!(s2.nms_threshold, DEFAULT_NMS_THRESHOLD);
    }

    #[test]
    fn input_dimensions_sanitize_replaces_zeros() {
        let mut d = InputDimensions {
            width: 0,
            height: 0,
            resize_quality: ResizeQuality::Quality,
        };
        d.sanitize();
        assert_eq!(d.width, DEFAULT_INPUT_WIDTH);
        assert_eq!(d.height, DEFAULT_INPUT_HEIGHT);

        // Non-zero dimensions should be left untouched
        let mut d2 = InputDimensions {
            width: 320,
            height: 240,
            resize_quality: ResizeQuality::Speed,
        };
        d2.sanitize();
        assert_eq!(d2.width, 320);
        assert_eq!(d2.height, 240);
    }

    #[test]
    fn crop_settings_sanitize_clamps_values() {
        let mut c = CropSettings {
            face_height_pct: 200.0,
            vertical_offset: 5.0,
            horizontal_offset: -5.0,
            output_width: 0,
            output_height: 0,
            vignette_softness: 2.0,
            vignette_intensity: -1.0,
            ..CropSettings::default()
        };
        c.sanitize();
        assert_eq!(c.face_height_pct, 100.0);
        assert_eq!(c.vertical_offset, 1.0);
        assert_eq!(c.horizontal_offset, -1.0);
        assert_eq!(c.output_width, 512);
        assert_eq!(c.output_height, 512);
        assert_eq!(c.vignette_softness, 1.0);
        assert_eq!(c.vignette_intensity, 0.0);
    }

    #[test]
    fn metadata_mode_from_str_all_variants() {
        assert_eq!(
            "preserve".parse::<MetadataMode>().unwrap(),
            MetadataMode::Preserve
        );
        assert_eq!(
            "strip".parse::<MetadataMode>().unwrap(),
            MetadataMode::Strip
        );
        assert_eq!(
            "custom".parse::<MetadataMode>().unwrap(),
            MetadataMode::Custom
        );
        assert!("unknown".parse::<MetadataMode>().is_err());
    }

    #[test]
    fn resize_quality_from_str_and_as_label() {
        assert_eq!(
            "quality".parse::<ResizeQuality>().unwrap(),
            ResizeQuality::Quality
        );
        assert_eq!(
            "SPEED".parse::<ResizeQuality>().unwrap(),
            ResizeQuality::Speed
        );
        assert!("fast".parse::<ResizeQuality>().is_err());
        assert_eq!(ResizeQuality::Quality.as_label(), "Quality");
        assert_eq!(ResizeQuality::Speed.as_label(), "Speed");
    }

    #[test]
    fn resize_quality_display() {
        assert_eq!(ResizeQuality::Quality.to_string(), "quality");
        assert_eq!(ResizeQuality::Speed.to_string(), "speed");
    }

    #[test]
    fn batch_log_format_display() {
        assert_eq!(BatchLogFormat::Json.to_string(), "JSON");
        assert_eq!(BatchLogFormat::Csv.to_string(), "CSV");
    }

    #[test]
    fn gpu_settings_into_context_options() {
        let gs = GpuSettings {
            enabled: false,
            respect_env: false,
            inference: true,
            preprocessing: true,
        };
        let opts: GpuContextOptions = gs.into();
        assert!(!opts.enabled);
        assert!(!opts.respect_env);

        let gs_ref = GpuSettings {
            enabled: true,
            respect_env: true,
            inference: false,
            preprocessing: false,
        };
        let opts_ref: GpuContextOptions = (&gs_ref).into();
        assert!(opts_ref.enabled);
        assert!(opts_ref.respect_env);
    }

    #[test]
    fn telemetry_level_parses_variants() {
        let telemetry = TelemetrySettings {
            level: "TRACE".into(),
            ..TelemetrySettings::default()
        };
        assert_eq!(telemetry.level_filter(), LevelFilter::Trace);

        let telemetry = TelemetrySettings {
            level: "Warn".into(),
            ..TelemetrySettings::default()
        };
        assert_eq!(telemetry.level_filter(), LevelFilter::Warn);

        let mut telemetry = TelemetrySettings::default();
        telemetry.set_level(LevelFilter::Info);
        assert_eq!(telemetry.level, "info");
    }
}
