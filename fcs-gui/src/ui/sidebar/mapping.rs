//! Mapping tab UI and queue application helpers.

use crate::{theme::P, types::App2};
use egui::{RichText, Sense, Ui, Vec2};
use fcs_mapping::MappingEntry;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use super::queue::draw_dashed_border;

pub(super) fn show_mapping(ui: &mut Ui, app: &mut App2) {
    let pad = 10.0_f32;
    ui.add_space(10.0);

    // ── Drop zones ────────────────────────────────────────────────────────────
    mapping_file_drop_zone(ui, app);
    queue_folder_drop_zone(ui, app);

    // ── Format badge ──────────────────────────────────────────────────────────
    if let Some(fmt) = app.mapping.effective_format() {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.add_space(pad);
            ui.label(
                RichText::new(format!("Format: {:?}", fmt))
                    .size(10.0)
                    .color(P::CYAN)
                    .family(egui::FontFamily::Monospace),
            );
        });
    }

    // ── Error ─────────────────────────────────────────────────────────────────
    if let Some(err) = app.mapping.preview_error.as_deref() {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.add_space(pad);
            ui.add(
                egui::Label::new(RichText::new(format!("⚠ {err}")).size(10.5).color(P::ROSE))
                    .wrap(),
            );
        });
    }

    // ── Column selectors ──────────────────────────────────────────────────────
    let columns: Vec<String> = app
        .mapping
        .preview
        .as_ref()
        .map(|p| p.columns.clone())
        .unwrap_or_default();

    if !columns.is_empty() {
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.add_space(pad);
            ui.label(
                RichText::new("Source column")
                    .size(10.0)
                    .color(P::INK3)
                    .family(egui::FontFamily::Monospace),
            );
        });
        let src_name = app
            .mapping
            .source_column_idx
            .and_then(|i| columns.get(i))
            .cloned()
            .unwrap_or_else(|| "— pick —".to_string());
        ui.horizontal(|ui| {
            ui.add_space(pad);
            let avail = ui.available_width() - pad;
            egui::ComboBox::from_id_salt("mapping_src_col")
                .selected_text(&src_name)
                .width(avail)
                .show_ui(ui, |ui| {
                    for (i, name) in columns.iter().enumerate() {
                        ui.selectable_value(&mut app.mapping.source_column_idx, Some(i), name);
                    }
                });
        });

        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.add_space(pad);
            ui.label(
                RichText::new("Output column")
                    .size(10.0)
                    .color(P::INK3)
                    .family(egui::FontFamily::Monospace),
            );
        });
        let out_name = app
            .mapping
            .output_column_idx
            .and_then(|i| columns.get(i))
            .cloned()
            .unwrap_or_else(|| "— pick —".to_string());
        ui.horizontal(|ui| {
            ui.add_space(pad);
            let avail = ui.available_width() - pad;
            egui::ComboBox::from_id_salt("mapping_out_col")
                .selected_text(&out_name)
                .width(avail)
                .show_ui(ui, |ui| {
                    for (i, name) in columns.iter().enumerate() {
                        ui.selectable_value(&mut app.mapping.output_column_idx, Some(i), name);
                    }
                });
        });

        // ── Apply button ──────────────────────────────────────────────────────
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.add_space(pad);
            let can_apply =
                app.mapping.source_column_idx.is_some() && app.mapping.output_column_idx.is_some();
            ui.add_enabled_ui(can_apply, |ui| {
                if ui.button("Apply mapping").clicked() {
                    match app.mapping.load_entries() {
                        Ok(_) => app.show_success(format!(
                            "Mapping loaded: {} entries",
                            app.mapping.entries.len()
                        )),
                        Err(e) => app.show_error("Mapping error", e.to_string()),
                    }
                }
            });
        });

        // ── Entry count + Apply to queue + Run ───────────────────────────────
        if !app.mapping.entries.is_empty() {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.add_space(pad);
                ui.label(
                    RichText::new(format!("{} entries loaded", app.mapping.entries.len()))
                        .size(10.5)
                        .color(P::LIME)
                        .family(egui::FontFamily::Monospace),
                );
            });
            ui.add_space(6.0);

            let has_queue = !app.batch_files.is_empty();
            let has_detector = app.detector.is_some();

            // Apply to queue
            ui.horizontal(|ui| {
                ui.add_space(pad);
                ui.add_enabled_ui(has_queue, |ui| {
                    if ui.button("Apply to queue").clicked() {
                        apply_mapping_to_queue(app);
                    }
                });
                if !has_queue {
                    ui.label(RichText::new("(add images first)").size(9.5).color(P::INK3));
                }
            });

            // Run with mapping
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.add_space(pad);
                let avail = ui.available_width() - pad;
                let enabled = has_queue && has_detector && !app.is_busy;
                ui.add_enabled_ui(enabled, |ui| {
                    if ui
                        .add_sized(
                            Vec2::new(avail, 30.0),
                            egui::Button::new(
                                RichText::new("Run with mapping →")
                                    .size(11.0)
                                    .family(egui::FontFamily::Monospace)
                                    .color(P::PEACH),
                            ),
                        )
                        .clicked()
                    {
                        apply_mapping_to_queue(app);
                        crate::core::export::start_batch_export(app);
                    }
                });
            });
            if !has_queue {
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.add_space(pad);
                    ui.label(
                        RichText::new("Add images to queue first")
                            .size(9.5)
                            .color(P::INK3),
                    );
                });
            }
        }
    }

    // ── Preview table ─────────────────────────────────────────────────────────
    let preview_rows = app.mapping.preview.as_ref().map(|p| p.rows.as_slice());

    if let Some(preview_rows) = preview_rows
        && !preview_rows.is_empty()
    {
        ui.add_space(10.0);
        ui.horizontal(|ui| {
            ui.add_space(pad);
            ui.label(
                RichText::new("Preview")
                    .size(10.0)
                    .color(P::INK3)
                    .family(egui::FontFamily::Monospace),
            );
        });
        ui.add_space(4.0);
        egui::ScrollArea::horizontal()
            .id_salt("mapping_preview_scroll")
            .show(ui, |ui| {
                egui::Grid::new("mapping_preview_grid")
                    .striped(true)
                    .spacing(Vec2::new(8.0, 2.0))
                    .show(ui, |ui| {
                        // Header
                        ui.label(""); // left-indent cell
                        for col in &columns {
                            ui.label(
                                RichText::new(col)
                                    .size(9.5)
                                    .color(P::CYAN)
                                    .family(egui::FontFamily::Monospace),
                            );
                        }
                        ui.end_row();
                        // Rows
                        for row in preview_rows.iter().take(6) {
                            ui.label(""); // left-indent cell
                            for cell in row {
                                let display = if cell.len() > 20 {
                                    format!("{}…", &cell[..19])
                                } else {
                                    cell.clone()
                                };
                                ui.label(
                                    RichText::new(display)
                                        .size(9.5)
                                        .color(P::INK2)
                                        .family(egui::FontFamily::Monospace),
                                );
                            }
                            ui.end_row();
                        }
                    });
            });
    }
}

fn mapping_file_drop_zone(ui: &mut Ui, app: &mut App2) {
    ui.add_space(8.0);
    let dz_rect = egui::Rect::from_min_size(
        egui::pos2(ui.min_rect().min.x + 8.0, ui.cursor().min.y),
        Vec2::new(ui.available_width() - 16.0, 82.0),
    );
    let resp = ui.allocate_rect(dz_rect, Sense::click());
    let painter = ui.painter();

    let has_file = app.mapping.file_path.is_some();
    let border_color = if resp.hovered() {
        P::PEACH
    } else {
        P::peach_alpha(140)
    };
    let bg_color = if resp.hovered() {
        P::peach_alpha(35)
    } else {
        P::peach_alpha(18)
    };

    painter.rect_filled(dz_rect, 10.0, bg_color);
    draw_dashed_border(painter, dz_rect, border_color);

    let icon_rect = egui::Rect::from_center_size(
        egui::pos2(dz_rect.center().x, dz_rect.min.y + 20.0),
        Vec2::splat(26.0),
    );
    painter.rect_filled(icon_rect, 7.0, P::peach_alpha(50));
    painter.text(
        icon_rect.center(),
        egui::Align2::CENTER_CENTER,
        "↓",
        egui::FontId::proportional(13.0),
        P::PEACH,
    );

    if has_file {
        let file_label = app
            .mapping
            .file_path
            .as_deref()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("file");
        let mut display: String = file_label.chars().take(28).collect();
        if file_label.chars().count() > 28 {
            display.push('…');
        }
        painter.text(
            egui::pos2(dz_rect.center().x, dz_rect.min.y + 46.0),
            egui::Align2::CENTER_CENTER,
            display,
            egui::FontId::monospace(9.5),
            P::PEACH,
        );
        painter.text(
            egui::pos2(dz_rect.center().x, dz_rect.min.y + 63.0),
            egui::Align2::CENTER_CENTER,
            "[ Click to replace ]",
            egui::FontId::monospace(9.0),
            P::INK2,
        );
    } else {
        painter.text(
            egui::pos2(dz_rect.center().x, dz_rect.min.y + 46.0),
            egui::Align2::CENTER_CENTER,
            "Drop mapping file",
            egui::FontId::proportional(12.0),
            P::INK,
        );
        painter.text(
            egui::pos2(dz_rect.center().x, dz_rect.min.y + 63.0),
            egui::Align2::CENTER_CENTER,
            "CSV · XLSX · DB  [ Browse ]",
            egui::FontId::monospace(9.5),
            P::INK3,
        );
    }
    ui.add_space(90.0);

    if resp.clicked()
        && let Some(path) = rfd::FileDialog::new()
            .add_filter(
                "Mapping files",
                &["csv", "xlsx", "xls", "db", "sqlite", "sqlite3"],
            )
            .pick_file()
    {
        app.mapping.set_file(path);
        let _ = app.mapping.reload_preview();
    }

    if app.mapping.file_path.is_some() {
        ui.add_space(2.0);
        ui.horizontal(|ui| {
            ui.add_space(8.0);
            if ui.button("Clear mapping").clicked() {
                app.mapping = crate::types::MappingUiState::new();
            }
        });
        ui.add_space(4.0);
    }
}

fn queue_folder_drop_zone(ui: &mut Ui, app: &mut App2) {
    ui.add_space(6.0);
    let dz_rect = egui::Rect::from_min_size(
        egui::pos2(ui.min_rect().min.x + 8.0, ui.cursor().min.y),
        Vec2::new(ui.available_width() - 16.0, 72.0),
    );
    let resp = ui.allocate_rect(dz_rect, Sense::click());
    let painter = ui.painter();

    let border_color = if resp.hovered() {
        P::LIME
    } else {
        P::lime_alpha(140)
    };
    let bg_color = if resp.hovered() {
        P::lime_alpha(25)
    } else {
        P::lime_alpha(12)
    };
    let queue_count = app.batch_files.len();

    painter.rect_filled(dz_rect, 10.0, bg_color);
    draw_dashed_border(painter, dz_rect, border_color);

    let icon_rect = egui::Rect::from_center_size(
        egui::pos2(dz_rect.center().x, dz_rect.min.y + 18.0),
        Vec2::splat(24.0),
    );
    painter.rect_filled(icon_rect, 6.0, P::lime_alpha(35));
    painter.text(
        icon_rect.center(),
        egui::Align2::CENTER_CENTER,
        "+",
        egui::FontId::proportional(13.0),
        P::LIME,
    );

    painter.text(
        egui::pos2(dz_rect.center().x, dz_rect.min.y + 40.0),
        egui::Align2::CENTER_CENTER,
        "Drop folder → queue",
        egui::FontId::proportional(12.0),
        P::INK,
    );
    let subtitle = if queue_count > 0 {
        format!("{queue_count} in queue  [ Browse ]")
    } else {
        "[ Browse ]".to_string()
    };
    painter.text(
        egui::pos2(dz_rect.center().x, dz_rect.min.y + 57.0),
        egui::Align2::CENTER_CENTER,
        subtitle,
        egui::FontId::monospace(9.5),
        P::INK3,
    );
    ui.add_space(80.0);

    if resp.clicked()
        && let Some(folder) = rfd::FileDialog::new().pick_folder()
    {
        let paths = crate::app::collect_folder_images(&folder);
        let first = paths.first().cloned();
        let added = app.enqueue_batch_paths(paths);
        if let Some(path) = first {
            app.load_image_path(path);
        }
        if added > 0 {
            app.show_success(format!(
                "Added {added} image(s) to the queue ({} total)",
                app.batch_files.len()
            ));
        } else {
            app.show_success("No new images found in folder.");
        }
    }
}

fn apply_mapping_to_queue(app: &mut App2) {
    let outputs = match_mapping(
        &app.mapping.entries,
        app.batch_files.iter().map(|f| f.path.as_path()),
    );
    let matched = outputs.iter().flatten().count();
    for (file, output) in app.batch_files.iter_mut().zip(outputs) {
        // Unmatched files clear, rather than keep a name left by an earlier mapping.
        file.output_override = output.map(PathBuf::from);
    }

    app.show_success(format!(
        "Mapping applied: {matched} / {} queue items matched",
        app.batch_files.len()
    ));
}

/// The mapped output name for each queued path, or `None` where no row matches.
///
/// A row matches by file name, else by stem (so a row may leave the extension off). Each tries
/// the exact spelling before ignoring case: Windows and macOS treat `Photo.JPG` and `photo.jpg`
/// as one file, and spreadsheet exports recase names. The first row wins within a tier.
fn match_mapping<'a, 'p>(
    entries: &'a [MappingEntry],
    paths: impl IntoIterator<Item = &'p Path>,
) -> Vec<Option<&'a str>> {
    let mut index = HashMap::new();
    for entry in entries {
        let keys = match_keys(Path::new(&entry.source_path));
        for (tier, key) in keys.into_iter().enumerate() {
            if let Some(key) = key {
                index
                    .entry((tier, key))
                    .or_insert(entry.output_name.as_str());
            }
        }
    }
    paths
        .into_iter()
        .map(|path| {
            match_keys(path)
                .into_iter()
                .enumerate()
                .find_map(|(tier, key)| index.get(&(tier, key?)).copied())
        })
        .collect()
}

/// Lookup keys, most specific first: name, name ignoring case, stem, stem ignoring case.
fn match_keys(path: &Path) -> [Option<String>; 4] {
    let name = path.file_name().and_then(|s| s.to_str());
    let stem = path.file_stem().and_then(|s| s.to_str());
    [
        name.map(str::to_owned),
        name.map(str::to_lowercase),
        stem.map(str::to_owned),
        stem.map(str::to_lowercase),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapping_matches_by_name_then_stem_preferring_the_exact_spelling() {
        let entry = |source: &str, output: &str| MappingEntry {
            source_path: source.into(),
            output_name: output.into(),
        };
        let entries = [
            // A spreadsheet export recased this one; the file on disk is lower case.
            entry("shoots/FaceCropOTBreak_1.jpg", "recased"),
            // Same stem, other extension: must not beat a name match listed later.
            entry("photo.png", "other-extension"),
            entry("PHOTO.JPG", "photo-folded"),
            entry("photo.jpg", "photo-exact"),
            entry("no-extension", "by-stem"),
        ];
        let queue = [
            "in/facecropotbreak_1.jpg",
            "in/photo.jpg",
            "in/Photo.jpg",
            "in/NO-EXTENSION.webp",
            "in/unlisted.jpg",
        ];
        assert_eq!(
            match_mapping(&entries, queue.iter().map(Path::new)),
            [
                Some("recased"),
                Some("photo-exact"),
                Some("photo-folded"),
                Some("by-stem"),
                None
            ]
        );
    }
}
