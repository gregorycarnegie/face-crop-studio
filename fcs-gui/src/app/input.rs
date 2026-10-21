//! Images arriving by drag and drop or folder, and the batch queue they join.

use std::{
    collections::{HashSet, VecDeque},
    path::{Path, PathBuf},
};

use crate::types::{App2, BatchFile, BatchFileStatus, LogKind, SidebarTab};

impl App2 {
    pub(super) fn handle_dropped_files(&mut self, ctx: &egui::Context) {
        let dropped = ctx.input_mut(|i| std::mem::take(&mut i.raw.dropped_files));
        if dropped.is_empty() {
            return;
        }

        const MAPPING_EXTS: &[&str] = &["csv", "xlsx", "xls", "db", "sqlite", "sqlite3"];
        let mut mapping_loaded = false;
        let mut paths = Vec::new();
        let mut unsupported = 0usize;

        for file in dropped {
            // egui 0.36 made DroppedFile a trait; on native `path()` is always a real
            // absolute path, so the previous "dropped file with no path" case is gone.
            let path = file.path();
            let is_mapping = path.extension().and_then(|e| e.to_str()).is_some_and(|e| {
                let lower = e.to_ascii_lowercase();
                MAPPING_EXTS.contains(&lower.as_str())
            });
            if is_mapping {
                self.mapping.set_file(path.to_path_buf());
                let _ = self.mapping.reload_preview();
                self.sidebar_tab = SidebarTab::Mapping;
                self.show_success(format!(
                    "Loaded mapping: {}",
                    path.file_name().and_then(|n| n.to_str()).unwrap_or("file")
                ));
                mapping_loaded = true;
            } else {
                match expand_input_path(path) {
                    Ok(mut images) => paths.append(&mut images),
                    Err(err) => {
                        unsupported += 1;
                        self.push_log(format!("Drop skipped: {err:#}"), LogKind::Warn);
                    }
                }
            }
        }

        let had_image_paths = !paths.is_empty();
        let first = paths.first().cloned();
        let added = self.enqueue_batch_paths(paths);
        if let Some(path) = first {
            self.load_image_path(path);
        }
        if added > 0 {
            self.show_success(format!(
                "Added {added} image(s) to the queue ({} total)",
                self.batch_files.len()
            ));
        } else if !mapping_loaded {
            if unsupported > 0 {
                self.show_error("Unsupported drop", "No supported files were found.");
            } else if had_image_paths {
                self.show_success("All dropped images were already queued.");
            }
        }
    }

    pub fn enqueue_batch_paths(&mut self, paths: Vec<PathBuf>) -> usize {
        let mut existing: HashSet<PathBuf> =
            self.batch_files.iter().map(|f| f.path.clone()).collect();
        let mut added = 0usize;
        for path in paths {
            if !is_supported_image_path(&path) || !existing.insert(path.clone()) {
                continue;
            }
            self.batch_files.push(BatchFile {
                path,
                status: BatchFileStatus::Pending,
                output_override: None,
            });
            added += 1;
        }
        added
    }
}

use fcs_utils::is_supported_image_path;

fn expand_input_path(path: &Path) -> anyhow::Result<Vec<PathBuf>> {
    if path.is_dir() {
        collect_supported_images_in_dir(path)
    } else if path.is_file() && is_supported_image_path(path) {
        Ok(vec![path.to_path_buf()])
    } else {
        anyhow::bail!("unsupported path {}", path.display())
    }
}

pub(crate) fn collect_folder_images(dir: &Path) -> Vec<PathBuf> {
    collect_supported_images_in_dir(dir).unwrap_or_default()
}

fn collect_supported_images_in_dir(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut images = Vec::new();
    let mut queue = VecDeque::new();
    queue.push_back(dir.to_path_buf());

    while let Some(current) = queue.pop_front() {
        for entry in std::fs::read_dir(&current)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                queue.push_back(path);
            } else if path.is_file() && is_supported_image_path(&path) {
                images.push(path);
            }
        }
    }

    images.sort();
    Ok(images)
}
