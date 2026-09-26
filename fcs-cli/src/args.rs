//! Command-line argument definitions for fcs-cli.

use clap::{ArgAction, Parser, ValueEnum};
use std::path::PathBuf;

/// Run face detection over images or directories.
#[derive(Debug, Parser)]
#[command(author, version, about)]
pub struct DetectArgs {
    /// Path to an image file or a directory containing images.
    #[arg(short, long, required_unless_present_any = ["mapping_file", "webcam", "watch"])]
    pub input: Option<PathBuf>,

    /// Watch a directory and process new or changed images as they arrive.
    ///
    /// Files already in the directory are left alone; run with --input to process those.
    /// Runs until interrupted, so there is no final JSON document to write, which is why
    /// it cannot be combined with --json.
    #[arg(long, value_name = "DIR", conflicts_with_all = ["input", "webcam", "mapping_file", "json"])]
    pub watch: Option<PathBuf>,

    /// Enable webcam capture mode (captures frames from the default webcam).
    #[arg(long, conflicts_with = "input", conflicts_with = "mapping_file")]
    pub webcam: bool,

    /// Webcam device index (default: 0 for default camera).
    #[arg(long, default_value_t = 0, requires = "webcam")]
    pub webcam_device: u32,

    /// Webcam capture width (default: 640).
    #[arg(long, default_value_t = 640, requires = "webcam")]
    pub webcam_width: u32,

    /// Webcam capture height (default: 480).
    #[arg(long, default_value_t = 480, requires = "webcam")]
    pub webcam_height: u32,

    /// Webcam frame rate (default: 30 fps).
    #[arg(long, default_value_t = 30, requires = "webcam")]
    pub webcam_fps: u32,

    /// Number of frames to capture in webcam mode (0 = continuous, Ctrl+C to stop).
    #[arg(long, default_value_t = 0, requires = "webcam")]
    pub webcam_frames: u32,

    /// Path to the detector ONNX model.
    #[arg(short, long, default_value = "models/scrfd80k_500m_640.onnx")]
    pub model: PathBuf,

    /// Optional settings JSON. Defaults to `config/gui_settings.json` when present, otherwise built-in parameters.
    #[arg(long)]
    pub config: Option<PathBuf>,

    /// Enable telemetry timing logs (defaults to settings file).
    #[arg(long, action = ArgAction::SetTrue)]
    pub telemetry: bool,

    /// Override telemetry logging level (error, warn, info, debug, trace).
    #[arg(long, value_name = "LEVEL")]
    pub telemetry_level: Option<String>,

    /// Force GPU acceleration (auto-detect by default).
    #[arg(long, action = ArgAction::SetTrue, conflicts_with = "no_gpu")]
    pub gpu: bool,

    /// Disable GPU acceleration entirely, even if supported.
    #[arg(long = "no-gpu", action = ArgAction::SetTrue)]
    pub no_gpu: bool,

    /// Control whether `WGPU_*` env vars influence GPU selection (`auto` or `ignore`).
    #[arg(long = "gpu-env", value_enum)]
    pub gpu_env: Option<GpuEnvMode>,

    /// Override score threshold.
    #[arg(long)]
    pub score_threshold: Option<f32>,

    /// Override NMS threshold.
    #[arg(long)]
    pub nms_threshold: Option<f32>,

    /// Override top_k limit.
    #[arg(long)]
    pub top_k: Option<usize>,

    /// Write detections to a JSON file instead of stdout.
    #[arg(long)]
    pub json: Option<PathBuf>,

    /// Directory to write annotated images with bounding boxes overlaid.
    #[arg(long)]
    pub annotate: Option<PathBuf>,

    /// Enable cropping mode: save cropped face images for each detection.
    #[arg(long)]
    pub crop: bool,

    /// Rotate each crop so the eyes are level.
    ///
    /// Uses the built-in eye refiner when `models/eye_refiner.onnx` is present,
    /// and the detector's own eye points otherwise. There is no negative form: the
    /// setting defaults to off, so omitting this leaves it off.
    #[arg(long)]
    pub eye_line_align: bool,

    /// Output directory for cropped face images (required when --crop is used).
    #[arg(long)]
    pub output_dir: Option<PathBuf>,

    /// Preset name for output size (e.g., LinkedIn, Passport, Instagram). If set, overrides --output-width/--output-height.
    #[arg(long)]
    pub preset: Option<String>,

    /// Output width for crops (pixels).
    #[arg(long)]
    pub output_width: Option<u32>,

    /// Output height for crops (pixels).
    #[arg(long)]
    pub output_height: Option<u32>,

    /// Face height percentage in the output image (default: 70).
    #[arg(long, default_value_t = 70.0)]
    pub face_height_pct: f32,

    /// Positioning mode for crop: center, rule_of_thirds, custom
    #[arg(long, default_value = "center")]
    pub positioning_mode: String,

    /// Horizontal offset for custom positioning (fraction -1.0..1.0).
    #[arg(long, default_value_t = 0.0)]
    pub horizontal_offset: f32,

    /// Vertical offset for custom positioning (fraction -1.0..1.0).
    #[arg(long, default_value_t = 0.0)]
    pub vertical_offset: f32,

    /// Fill color for areas outside the source image when crops extend past the image edges (accepts #RRGGBB, rgb(), hsv()).
    #[arg(long, value_name = "COLOR")]
    pub crop_fill_color: Option<String>,

    /// Output image format for saved crops: png, jpeg, webp
    #[arg(long, default_value = "png")]
    pub output_format: String,

    /// JPEG quality when saving as JPEG (1-100).
    #[arg(long, default_value_t = 90u8)]
    pub jpeg_quality: u8,

    /// PNG compression strategy: fast, default, best, or numeric level 0-9.
    #[arg(long)]
    pub png_compression: Option<String>,

    /// Automatically detect output format from the file extension.
    #[arg(long)]
    pub auto_detect_format: Option<bool>,

    /// Select face index (1-based) to save only a specific face. Default: all faces.
    #[arg(long)]
    pub face_index: Option<usize>,

    /// Minimum quality to save crops (low, medium, high). If set, crops below this level are skipped.
    #[arg(long)]
    pub min_quality: Option<String>,

    /// Shortcut to skip low-quality crops (equivalent to `--min-quality medium`).
    #[arg(long)]
    pub skip_low_quality: Option<bool>,

    /// Automatically select the highest-quality face per image. Use `--auto-select-best=false` to disable.
    #[arg(long)]
    pub auto_select_best: Option<bool>,

    /// Skip exporting when no high-quality faces are detected. Use `--skip-no-high-quality=false` to require manual review.
    #[arg(long)]
    pub skip_no_high_quality: Option<bool>,

    /// Append a quality suffix (e.g., `_highq`) to exported filenames. Use `--quality-suffix=false` to disable.
    #[arg(long)]
    pub quality_suffix: Option<bool>,

    /// Metadata handling mode: preserve, strip, or custom.
    #[arg(long)]
    pub metadata_mode: Option<String>,

    /// Include crop settings metadata in output files.
    #[arg(long)]
    pub metadata_include_crop: Option<bool>,

    /// Include quality scores in output metadata.
    #[arg(long)]
    pub metadata_include_quality: Option<bool>,

    /// Custom metadata tags in KEY=VALUE form (may be repeated).
    #[arg(long = "metadata-tag")]
    pub metadata_tags: Vec<String>,

    /// Apply image enhancement pipeline (unsharp mask, contrast, exposure)
    /// to each crop before quality estimation and saving.
    #[arg(long)]
    pub enhance: Option<bool>,

    /// Unsharp mask amount. If provided, overrides preset/default.
    #[arg(long)]
    pub unsharp_amount: Option<f32>,

    /// Unsharp mask blur radius in pixels.
    #[arg(long)]
    pub unsharp_radius: Option<f32>,

    /// Contrast multiplier (0.5-2.0, 1.0 = unchanged).
    #[arg(long)]
    pub enhance_contrast: Option<f32>,

    /// Exposure adjustment in stops (-2.0..=2.0).
    #[arg(long)]
    pub enhance_exposure: Option<f32>,

    /// Additional brightness offset (integer steps applied after exposure)
    #[arg(long)]
    pub enhance_brightness: Option<i32>,

    /// Saturation multiplier (1.0 = unchanged, <1 desaturate, >1 increase)
    #[arg(long)]
    pub enhance_saturation: Option<f32>,

    /// Apply gray-world auto color correction to crops when --enhance is set
    #[arg(long)]
    pub enhance_auto_color: Option<bool>,

    /// Additional sharpening strength (added to unsharp-amount)
    #[arg(long)]
    pub enhance_sharpness: Option<f32>,

    /// Skin smoothing strength (0.0-1.0, uses bilateral filter)
    #[arg(long)]
    pub enhance_skin_smooth: Option<f32>,

    /// Enable automated red-eye removal
    #[arg(long)]
    pub enhance_red_eye_removal: Option<bool>,

    /// Enable background blur (portrait mode effect)
    #[arg(long)]
    pub enhance_background_blur: Option<bool>,

    /// Naming template for output crop files. Variables: {original}, {index}, {width}, {height}, {ext}, {timestamp}
    #[arg(long)]
    pub naming_template: Option<String>,

    /// Enhancement preset to apply when --enhance is set. Options: natural, vivid, professional
    #[arg(long)]
    pub enhancement_preset: Option<String>,

    /// Optional mapping file that lists source images and desired output names.
    #[arg(long = "mapping-file")]
    pub mapping_file: Option<PathBuf>,

    /// Column containing source paths inside the mapping file (name or zero-based index).
    #[arg(long = "mapping-source-col")]
    pub mapping_source_col: Option<String>,

    /// Column containing output names inside the mapping file (name or zero-based index).
    #[arg(long = "mapping-output-col")]
    pub mapping_output_col: Option<String>,

    /// Whether the mapping file contains a header row (defaults to true for CSV/Excel/Parquet).
    #[arg(long = "mapping-has-headers")]
    pub mapping_has_headers: Option<bool>,

    /// Optional delimiter for CSV/TSV mappings (defaults to comma).
    #[arg(long = "mapping-delimiter")]
    pub mapping_delimiter: Option<char>,

    /// Optional sheet name when loading from Excel.
    #[arg(long = "mapping-sheet")]
    pub mapping_sheet: Option<String>,

    /// Optional explicit mapping format (csv, excel, parquet, sqlite).
    #[arg(long = "mapping-format")]
    pub mapping_format: Option<String>,

    /// SQLite table to read when using .db/.sqlite files (defaults to the first table).
    #[arg(long = "mapping-sql-table")]
    pub mapping_sql_table: Option<String>,

    /// Custom SQL query to run when using SQLite mapping files.
    #[arg(long = "mapping-sql-query")]
    pub mapping_sql_query: Option<String>,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum GpuEnvMode {
    /// Respect environment overrides such as `WGPU_BACKEND`.
    Auto,
    /// Ignore environment overrides and rely solely on CLI/config.
    Ignore,
}

impl GpuEnvMode {
    pub fn respects_env(self) -> bool {
        matches!(self, GpuEnvMode::Auto)
    }
}

impl DetectArgs {
    /// Whether a mapping file was given without `--crop`. The mapping only renames crops, so
    /// such a run has nothing to apply it to yet -- worth a notice rather than silence.
    pub(crate) fn mapping_waits_for_crop(&self) -> bool {
        self.mapping_file.is_some() && !self.crop
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_env_mode_auto_respects_env() {
        assert!(GpuEnvMode::Auto.respects_env());
    }

    #[test]
    fn gpu_env_mode_ignore_does_not_respect_env() {
        assert!(!GpuEnvMode::Ignore.respects_env());
    }

    #[test]
    fn a_mapping_file_waits_for_crop_only_without_it() {
        let parse = |flags: &[&str]| {
            // `--input` is required outside watch mode, whatever the field's type says.
            let mut argv = vec!["fcs-cli", "--input", "photo.jpg"];
            argv.extend_from_slice(flags);
            <DetectArgs as clap::Parser>::try_parse_from(argv).expect("valid arguments")
        };
        assert!(parse(&["--mapping-file", "m.csv"]).mapping_waits_for_crop());
        assert!(!parse(&["--mapping-file", "m.csv", "--crop"]).mapping_waits_for_crop());
        assert!(!parse(&[]).mapping_waits_for_crop());
        assert!(!parse(&["--crop"]).mapping_waits_for_crop());
    }
}
