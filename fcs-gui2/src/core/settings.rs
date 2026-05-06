//! Settings persistence.

use anyhow::{Context as AnyhowContext, Result};
use fcs_utils::config::AppSettings;
use log::warn;
use std::path::Path;

pub fn load_settings(path: &Path) -> AppSettings {
    match AppSettings::load_from_path(path) {
        Ok(s) => s,
        Err(err) => {
            warn!(
                "Failed to load settings from {}: {err}. Using defaults.",
                path.display()
            );
            AppSettings::default()
        }
    }
}

pub fn persist_settings(settings: &AppSettings, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
    }
    settings
        .save_to_path(path)
        .with_context(|| format!("failed to write settings to {}", path.display()))
}

pub fn persist_with_feedback(settings: &AppSettings, path: &Path) -> Result<(), String> {
    persist_settings(settings, path).map_err(|e| {
        let msg = format!("Failed to persist settings: {e}");
        warn!("{msg}");
        msg
    })
}
