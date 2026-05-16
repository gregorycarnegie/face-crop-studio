//! Configuration loading and CLI override logic.

use std::{collections::BTreeMap, path::PathBuf};

use anyhow::{Context, Result};
use fcs_core::preset_by_name;
use fcs_core::{CropSettings, PositioningMode};
use fcs_utils::{
    ImageFormatHint, PngCompression, Quality,
    config::{
        AppSettings, CropSettings as ConfigCropSettings, MetadataMode, default_settings_path,
    },
    normalize_path,
};
use log::{info, warn};

use crate::{args::DetectArgs, color::parse_fill_color_spec};

/// Load application settings from a file or use defaults.
pub fn load_settings(config_path: Option<&PathBuf>) -> Result<AppSettings> {
    if let Some(path) = config_path {
        let resolved = normalize_path(path)?;
        let settings = AppSettings::load_from_path(&resolved)?;
        info!("Loaded settings from {}", resolved.display());
        Ok(settings)
    } else {
        let default_path = default_settings_path();
        if default_path.exists() {
            let settings = AppSettings::load_from_path(&default_path).with_context(|| {
                format!(
                    "failed to load default settings from {}",
                    default_path.display()
                )
            })?;
            info!("Loaded settings from {}", default_path.display());
            Ok(settings)
        } else {
            Ok(AppSettings::default())
        }
    }
}

/// Apply command-line arguments to override loaded or default settings.
pub fn apply_cli_overrides(settings: &mut AppSettings, args: &DetectArgs) {
    if args.gpu {
        settings.gpu.enabled = true;
    }
    if args.no_gpu {
        settings.gpu.enabled = false;
        settings.gpu.inference = false;
    }
    if args.gpu_inference {
        settings.gpu.inference = true;
    }
    if let Some(mode) = args.gpu_env {
        settings.gpu.respect_env = mode.respects_env();
    }

    if args.telemetry {
        settings.telemetry.enabled = true;
    }
    if let Some(level) = args.telemetry_level.as_ref() {
        let normalized = level.trim();
        if !normalized.is_empty() {
            let lower = normalized.to_ascii_lowercase();
            let disables_telemetry = lower == "off";
            settings.telemetry.level = lower;
            if disables_telemetry {
                settings.telemetry.enabled = false;
            }
        }
    }

    if let Some(width) = args.width {
        settings.input.width = width;
    }
    if let Some(height) = args.height {
        settings.input.height = height;
    }
    if let Some(mode) = args.resize_quality {
        settings.input.resize_quality = mode;
    }
    if let Some(score) = args.score_threshold {
        settings.detection.score_threshold = score;
    }
    if let Some(nms) = args.nms_threshold {
        settings.detection.nms_threshold = nms;
    }
    if let Some(top_k) = args.top_k {
        settings.detection.top_k = top_k;
    }

    if let Some(preset_name) = args.preset.as_ref() {
        settings.crop.preset = preset_name.to_ascii_lowercase();
        if settings.crop.preset != "custom"
            && let Some(preset) = preset_by_name(preset_name)
            && preset.width > 0
            && preset.height > 0
        {
            settings.crop.output_width = preset.width;
            settings.crop.output_height = preset.height;
        }
    }
    if let Some(width) = args.output_width {
        settings.crop.output_width = width;
        settings.crop.preset = "custom".to_string();
    }
    if let Some(height) = args.output_height {
        settings.crop.output_height = height;
        settings.crop.preset = "custom".to_string();
    }

    settings.crop.face_height_pct = args.face_height_pct;
    settings.crop.horizontal_offset = args.horizontal_offset;
    settings.crop.vertical_offset = args.vertical_offset;
    settings.crop.positioning_mode = parse_positioning_mode(&args.positioning_mode);
    if let Some(ref fill) = args.crop_fill_color {
        match parse_fill_color_spec(fill) {
            Ok(color) => settings.crop.fill_color = color,
            Err(err) => warn!("failed to parse --crop-fill-color '{}': {}", fill, err),
        }
    }
    match args.output_format.parse::<ImageFormatHint>() {
        Ok(fmt) => settings.crop.output_format = fmt,
        Err(_) => warn!(
            "unknown --output-format '{}', keeping {:?}",
            args.output_format, settings.crop.output_format
        ),
    }
    settings.crop.jpeg_quality = args.jpeg_quality;
    if let Some(ref compression) = args.png_compression {
        settings.crop.png_compression = PngCompression::parse(compression);
    }
    if let Some(webp) = args.webp_quality {
        settings.crop.webp_quality = webp;
    }
    if let Some(auto) = args.auto_detect_format {
        settings.crop.auto_detect_format = auto;
    }

    if let Some(auto) = args.auto_select_best {
        settings.crop.quality_rules.auto_select_best_face = auto;
    }
    if let Some(skip) = args.skip_no_high_quality {
        settings.crop.quality_rules.auto_skip_no_high_quality = skip;
    }
    if let Some(flag) = args.quality_suffix {
        settings.crop.quality_rules.quality_suffix = flag;
    }
    if let Some(skip) = args.skip_low_quality {
        if skip {
            settings.crop.quality_rules.min_quality = Some(Quality::Medium);
        } else {
            settings.crop.quality_rules.min_quality = None;
        }
    } else if let Some(ref s) = args.min_quality {
        match s.parse::<Quality>() {
            Ok(q) => {
                settings.crop.quality_rules.min_quality = Some(q);
            }
            Err(_) => warn!("unknown --min-quality value '{}', ignoring", s),
        }
    }

    if let Some(ref mode) = args.metadata_mode {
        match mode.parse::<MetadataMode>() {
            Ok(mode) => settings.crop.metadata.mode = mode,
            Err(err) => warn!("{err}"),
        }
    }
    if let Some(include) = args.metadata_include_crop {
        settings.crop.metadata.include_crop_settings = include;
    }
    if let Some(include) = args.metadata_include_quality {
        settings.crop.metadata.include_quality_metrics = include;
    }
    if !args.metadata_tags.is_empty() {
        settings.crop.metadata.custom_tags = parse_metadata_tags_args(&args.metadata_tags);
    }

    settings.crop.sanitize();
}

pub fn build_core_crop_settings(cfg: &ConfigCropSettings) -> CropSettings {
    CropSettings {
        output_width: cfg.output_width,
        output_height: cfg.output_height,
        face_height_pct: cfg.face_height_pct,
        positioning_mode: cfg.positioning_mode,
        horizontal_offset: cfg.horizontal_offset,
        vertical_offset: cfg.vertical_offset,
        fill_color: cfg.fill_color,
        eye_line_align: cfg.eye_line_align,
    }
}

/// Parse the `--positioning-mode` CLI argument value. Reuses the serde aliases
/// configured on [`PositioningMode`] so the CLI accepts the same spellings the
/// config file does.
fn parse_positioning_mode(value: &str) -> PositioningMode {
    let token = value.to_ascii_lowercase();
    // serde_json's enum string deserializer honors the `#[serde(alias = …)]` attrs,
    // so we can route every accepted spelling through one canonical path.
    serde_json::from_value(serde_json::Value::String(token)).unwrap_or(PositioningMode::Center)
}

fn parse_metadata_tags_args(entries: &[String]) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    for entry in entries {
        if let Some((key, value)) = entry.split_once('=') {
            let key = key.trim();
            if key.is_empty() {
                warn!("Ignoring metadata tag with empty key: '{entry}'");
                continue;
            }
            map.insert(key.to_string(), value.trim().to_string());
        } else {
            warn!("Invalid metadata tag '{entry}', expected key=value");
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use std::{
        env, fs,
        sync::{Mutex, OnceLock},
    };

    use clap::Parser;
    use tempfile::tempdir;

    use super::*;
    use crate::args::DetectArgs;

    fn cwd_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    // Helper: parse DetectArgs from a slice of args (prepend binary name).
    fn parse_args(args: &[&str]) -> DetectArgs {
        let mut full = vec!["fcs-cli"];
        full.extend_from_slice(args);
        DetectArgs::try_parse_from(full).expect("valid args")
    }

    #[test]
    fn load_settings_reads_explicit_config_path() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("settings.json");
        let mut expected = AppSettings::default();
        expected.crop.output_width = 777;
        expected.save_to_path(&path).expect("save settings");

        let loaded = load_settings(Some(&path)).expect("load settings from explicit path");
        assert_eq!(loaded.crop.output_width, 777);
    }

    #[test]
    fn load_settings_returns_defaults_when_default_config_is_missing() {
        let _lock = cwd_lock().lock().expect("cwd lock");
        let original_dir = env::current_dir().expect("get current dir");
        let dir = tempdir().expect("tempdir");
        env::set_current_dir(dir.path()).expect("set current dir");

        let loaded = load_settings(None).expect("load default settings");

        env::set_current_dir(original_dir).expect("restore current dir");
        assert_eq!(
            loaded.crop.output_width,
            AppSettings::default().crop.output_width
        );
    }

    #[test]
    fn load_settings_reads_default_config_from_current_directory() {
        let _lock = cwd_lock().lock().expect("cwd lock");
        let original_dir = env::current_dir().expect("get current dir");
        let dir = tempdir().expect("tempdir");
        let default_path = dir.path().join("config").join("gui_settings.json");
        fs::create_dir_all(default_path.parent().expect("parent dir")).expect("create config dir");

        let mut expected = AppSettings::default();
        expected.crop.output_height = 654;
        expected.save_to_path(&default_path).expect("save settings");

        env::set_current_dir(dir.path()).expect("set current dir");
        let loaded = load_settings(None).expect("load default settings from cwd");
        env::set_current_dir(original_dir).expect("restore current dir");

        assert_eq!(loaded.crop.output_height, 654);
    }

    #[test]
    fn load_settings_surfaces_context_when_default_config_is_invalid() {
        let _lock = cwd_lock().lock().expect("cwd lock");
        let original_dir = env::current_dir().expect("get current dir");
        let dir = tempdir().expect("tempdir");
        let default_path = dir.path().join("config").join("gui_settings.json");
        fs::create_dir_all(default_path.parent().expect("parent dir")).expect("create config dir");
        fs::write(&default_path, "{not-json").expect("write invalid config");

        env::set_current_dir(dir.path()).expect("set current dir");
        let err = load_settings(None).unwrap_err().to_string();
        env::set_current_dir(original_dir).expect("restore current dir");

        assert!(err.contains("failed to load default settings"));
        assert!(err.contains("gui_settings.json"));
    }

    // --- parse_positioning_mode ---

    #[test]
    fn positioning_mode_center_variants() {
        assert!(matches!(
            parse_positioning_mode("center"),
            PositioningMode::Center
        ));
        assert!(matches!(
            parse_positioning_mode("unknown"),
            PositioningMode::Center
        ));
        assert!(matches!(
            parse_positioning_mode(""),
            PositioningMode::Center
        ));
    }

    #[test]
    fn positioning_mode_rule_of_thirds_variants() {
        assert!(matches!(
            parse_positioning_mode("rule_of_thirds"),
            PositioningMode::RuleOfThirds
        ));
        assert!(matches!(
            parse_positioning_mode("rule-of-thirds"),
            PositioningMode::RuleOfThirds
        ));
        assert!(matches!(
            parse_positioning_mode("ruleofthirds"),
            PositioningMode::RuleOfThirds
        ));
        // GUI inspector writes "thirds"; CLI must round-trip GUI configs.
        assert!(matches!(
            parse_positioning_mode("thirds"),
            PositioningMode::RuleOfThirds
        ));
    }

    #[test]
    fn positioning_mode_custom() {
        assert!(matches!(
            parse_positioning_mode("custom"),
            PositioningMode::Custom
        ));
    }

    // --- parse_metadata_tags_args ---

    #[test]
    fn metadata_tags_parses_valid_entries() {
        let entries = vec!["author=Alice".to_string(), "project=YuNet".to_string()];
        let map = parse_metadata_tags_args(&entries);
        assert_eq!(map.get("author").map(String::as_str), Some("Alice"));
        assert_eq!(map.get("project").map(String::as_str), Some("YuNet"));
    }

    #[test]
    fn metadata_tags_trims_whitespace() {
        let entries = vec![" key = value with spaces ".to_string()];
        let map = parse_metadata_tags_args(&entries);
        assert_eq!(
            map.get("key").map(String::as_str),
            Some("value with spaces")
        );
    }

    #[test]
    fn metadata_tags_skips_empty_key() {
        let entries = vec!["=value".to_string()];
        let map = parse_metadata_tags_args(&entries);
        assert!(map.is_empty());
    }

    #[test]
    fn metadata_tags_skips_no_equals() {
        let entries = vec!["noequals".to_string()];
        let map = parse_metadata_tags_args(&entries);
        assert!(map.is_empty());
    }

    #[test]
    fn metadata_tags_empty_input() {
        let map = parse_metadata_tags_args(&[]);
        assert!(map.is_empty());
    }

    // --- apply_cli_overrides ---

    #[test]
    fn override_gpu_flags() {
        let mut settings = AppSettings::default();
        let args = parse_args(&["--input", "x.jpg", "--gpu"]);
        apply_cli_overrides(&mut settings, &args);
        assert!(settings.gpu.enabled);
    }

    #[test]
    fn override_no_gpu_disables_inference() {
        let mut settings = AppSettings::default();
        settings.gpu.enabled = true;
        settings.gpu.inference = true;
        let args = parse_args(&["--input", "x.jpg", "--no-gpu"]);
        apply_cli_overrides(&mut settings, &args);
        assert!(!settings.gpu.enabled);
        assert!(!settings.gpu.inference);
    }

    #[test]
    fn override_telemetry_level_off_disables_telemetry() {
        let mut settings = AppSettings::default();
        settings.telemetry.enabled = true;
        let args = parse_args(&["--input", "x.jpg", "--telemetry-level", "off"]);
        apply_cli_overrides(&mut settings, &args);
        assert!(!settings.telemetry.enabled);
        assert_eq!(settings.telemetry.level, "off");
    }

    #[test]
    fn override_telemetry_level_info() {
        let mut settings = AppSettings::default();
        let args = parse_args(&["--input", "x.jpg", "--telemetry-level", "info"]);
        apply_cli_overrides(&mut settings, &args);
        assert_eq!(settings.telemetry.level, "info");
    }

    #[test]
    fn override_detection_thresholds() {
        let mut settings = AppSettings::default();
        let args = parse_args(&[
            "--input",
            "x.jpg",
            "--score-threshold",
            "0.5",
            "--nms-threshold",
            "0.4",
            "--top-k",
            "100",
        ]);
        apply_cli_overrides(&mut settings, &args);
        assert!((settings.detection.score_threshold - 0.5).abs() < f32::EPSILON);
        assert!((settings.detection.nms_threshold - 0.4).abs() < f32::EPSILON);
        assert_eq!(settings.detection.top_k, 100);
    }

    #[test]
    fn override_output_width_height_sets_preset_custom() {
        let mut settings = AppSettings::default();
        let args = parse_args(&[
            "--input",
            "x.jpg",
            "--output-width",
            "300",
            "--output-height",
            "400",
        ]);
        apply_cli_overrides(&mut settings, &args);
        assert_eq!(settings.crop.output_width, 300);
        assert_eq!(settings.crop.output_height, 400);
        assert_eq!(settings.crop.preset, "custom");
    }

    #[test]
    fn override_skip_low_quality_true_sets_medium() {
        let mut settings = AppSettings::default();
        let args = parse_args(&["--input", "x.jpg", "--skip-low-quality=true"]);
        apply_cli_overrides(&mut settings, &args);
        assert_eq!(
            settings.crop.quality_rules.min_quality,
            Some(fcs_utils::Quality::Medium)
        );
    }

    #[test]
    fn override_skip_low_quality_false_clears_min_quality() {
        let mut settings = AppSettings::default();
        settings.crop.quality_rules.min_quality = Some(fcs_utils::Quality::Medium);
        let args = parse_args(&["--input", "x.jpg", "--skip-low-quality=false"]);
        apply_cli_overrides(&mut settings, &args);
        assert!(settings.crop.quality_rules.min_quality.is_none());
    }

    #[test]
    fn override_min_quality_high() {
        let mut settings = AppSettings::default();
        let args = parse_args(&["--input", "x.jpg", "--min-quality", "high"]);
        apply_cli_overrides(&mut settings, &args);
        assert_eq!(
            settings.crop.quality_rules.min_quality,
            Some(fcs_utils::Quality::High)
        );
    }

    #[test]
    fn override_applies_general_runtime_and_export_settings() {
        let mut settings = AppSettings::default();
        let args = parse_args(&[
            "--input",
            "x.jpg",
            "--gpu-inference",
            "--gpu-env",
            "ignore",
            "--telemetry",
            "--width",
            "320",
            "--height",
            "240",
            "--resize-quality",
            "speed",
            "--preset",
            "linkedin",
            "--crop-fill-color",
            "#112233",
            "--output-format",
            "JPG",
            "--png-compression",
            "best",
            "--webp-quality",
            "77",
            "--auto-detect-format=true",
            "--auto-select-best=true",
            "--skip-no-high-quality=true",
            "--quality-suffix=false",
            "--metadata-mode",
            "strip",
            "--metadata-include-crop=false",
            "--metadata-include-quality=true",
            "--metadata-tag",
            "owner=greg",
        ]);

        apply_cli_overrides(&mut settings, &args);

        assert!(settings.gpu.inference);
        assert!(!settings.gpu.respect_env);
        assert!(settings.telemetry.enabled);
        assert_eq!(settings.input.width, 320);
        assert_eq!(settings.input.height, 240);
        assert_eq!(
            settings.input.resize_quality,
            fcs_utils::config::ResizeQuality::Speed
        );
        assert_eq!(settings.crop.preset, "linkedin");
        assert_eq!(settings.crop.output_width, 400);
        assert_eq!(settings.crop.output_height, 400);
        assert_eq!(settings.crop.fill_color.red, 0x11);
        assert_eq!(settings.crop.output_format, ImageFormatHint::Jpeg);
        assert_eq!(settings.crop.png_compression, PngCompression::Best);
        assert_eq!(settings.crop.webp_quality, 77);
        assert!(settings.crop.auto_detect_format);
        assert!(settings.crop.quality_rules.auto_select_best_face);
        assert!(settings.crop.quality_rules.auto_skip_no_high_quality);
        assert!(!settings.crop.quality_rules.quality_suffix);
        assert_eq!(settings.crop.metadata.mode, MetadataMode::Strip);
        assert!(!settings.crop.metadata.include_crop_settings);
        assert!(settings.crop.metadata.include_quality_metrics);
        assert_eq!(
            settings
                .crop
                .metadata
                .custom_tags
                .get("owner")
                .map(String::as_str),
            Some("greg")
        );
    }

    #[test]
    fn override_invalid_values_leave_existing_settings_unchanged() {
        let mut settings = AppSettings::default();
        settings.crop.fill_color.red = 9;
        settings.crop.metadata.mode = MetadataMode::Preserve;
        settings.crop.quality_rules.min_quality = Some(Quality::Low);
        let args = parse_args(&[
            "--input",
            "x.jpg",
            "--crop-fill-color",
            "not-a-color",
            "--min-quality",
            "bogus",
            "--metadata-mode",
            "bogus",
            "--telemetry-level",
            "   ",
        ]);

        apply_cli_overrides(&mut settings, &args);

        assert_eq!(settings.crop.fill_color.red, 9);
        assert_eq!(settings.crop.metadata.mode, MetadataMode::Preserve);
        assert_eq!(settings.crop.quality_rules.min_quality, Some(Quality::Low));
        assert_eq!(
            settings.telemetry.level,
            AppSettings::default().telemetry.level
        );
    }

    #[test]
    fn override_custom_preset_does_not_replace_output_dimensions() {
        let mut settings = AppSettings::default();
        settings.crop.output_width = 321;
        settings.crop.output_height = 654;
        let args = parse_args(&["--input", "x.jpg", "--preset", "custom"]);

        apply_cli_overrides(&mut settings, &args);

        assert_eq!(settings.crop.preset, "custom");
        assert_eq!(settings.crop.output_width, 321);
        assert_eq!(settings.crop.output_height, 654);
    }

    // --- build_core_crop_settings ---

    #[test]
    fn build_core_crop_settings_center() {
        let cfg = fcs_utils::config::CropSettings {
            output_width: 200,
            output_height: 300,
            positioning_mode: PositioningMode::Center,
            ..Default::default()
        };
        let core = build_core_crop_settings(&cfg);
        assert_eq!(core.output_width, 200);
        assert_eq!(core.output_height, 300);
        assert!(matches!(core.positioning_mode, PositioningMode::Center));
    }

    #[test]
    fn build_core_crop_settings_rule_of_thirds() {
        let cfg = fcs_utils::config::CropSettings {
            positioning_mode: PositioningMode::RuleOfThirds,
            ..Default::default()
        };
        let core = build_core_crop_settings(&cfg);
        assert!(matches!(
            core.positioning_mode,
            PositioningMode::RuleOfThirds
        ));
    }
}
