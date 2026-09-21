//! Queue tab and batch file tree UI.

use crate::{
    theme::P,
    types::{App2, BatchFileStatus},
};
use egui::{RichText, Sense, Stroke, Ui, Vec2};

pub(super) fn show_queue(ui: &mut Ui, app: &mut App2) {
    // Drop zone
    drop_zone(ui, app);

    // Webcam capture
    webcam_bar(ui, app);

    // Folder browse shortcut
    folder_browse_bar(ui, app);

    // File tree
    file_tree(ui, app);
}

fn webcam_bar(ui: &mut Ui, app: &mut App2) {
    use crate::types::WebcamStatus;
    let is_live = app.webcam_state.status == WebcamStatus::Active;
    let subtitle = if is_live {
        match (
            app.webcam_state.live_detect,
            app.webcam_state.last_detect_ms,
        ) {
            // The detection latency is the number worth surfacing: it is what says whether
            // the overlay is keeping up, and it is a fraction of the frame interval.
            (true, Some(ms)) => format!(
                "Live · {} frames · detect {ms:.1} ms · {} skipped",
                app.webcam_state.frames_captured, app.webcam_state.frames_skipped
            ),
            (true, None) => format!(
                "Live · {} frames · detecting…",
                app.webcam_state.frames_captured
            ),
            _ => format!("Live · {} frames", app.webcam_state.frames_captured),
        }
    } else {
        "Default camera".to_string()
    };

    // Laid out rather than painted at fixed offsets: the live subtitle grows with the
    // frame counters and used to run underneath the buttons.
    egui::Frame::new()
        .outer_margin(egui::Margin::symmetric(8, 4))
        .inner_margin(egui::Margin::symmetric(10, 6))
        .corner_radius(8)
        .fill(P::SURFACE)
        .stroke(Stroke::new(1.0, if is_live { P::RULE2 } else { P::RULE }))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                let (dot, _) = ui.allocate_exact_size(Vec2::splat(10.0), Sense::hover());
                let dot_color = if is_live { P::LIME } else { P::INK3 };
                ui.painter().circle_filled(dot.center(), 5.0, dot_color);
                ui.label(RichText::new("Webcam capture").size(13.0).color(P::INK));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if !is_live {
                        if ui.button("Open camera").clicked() {
                            app.open_webcam();
                        }
                        return;
                    }
                    // Right-to-left, so the close button is added first.
                    if ui.button("×").on_hover_text("Close camera").clicked() {
                        app.close_webcam();
                    }
                    if ui
                        .add_enabled(!app.is_busy, egui::Button::new("Detect faces"))
                        .on_hover_text("Freeze frame and run face detection")
                        .clicked()
                    {
                        app.detect_webcam_faces();
                    }
                    // Live toggle: detect every frame rather than on demand.
                    let live_on = app.webcam_state.live_detect;
                    let live = RichText::new("Live").color(if live_on { P::BG } else { P::INK });
                    if ui
                        .add(egui::Button::new(live).selected(live_on))
                        .on_hover_text(if live_on {
                            "Detecting every frame — click to stop"
                        } else {
                            "Detect faces on every frame"
                        })
                        .clicked()
                    {
                        app.toggle_live_detection();
                    }
                });
            });
            ui.label(RichText::new(subtitle).size(11.5).color(P::INK2));
        });
}

fn folder_browse_bar(ui: &mut Ui, app: &mut App2) {
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.add_space(8.0);
        let avail = ui.available_width() - 8.0;
        if ui
            .add_sized(
                Vec2::new(avail, 24.0),
                egui::Button::new(
                    RichText::new("Add folder…")
                        .size(10.5)
                        .family(egui::FontFamily::Monospace)
                        .color(P::INK2),
                ),
            )
            .clicked()
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
    });
    ui.add_space(4.0);
}

pub(super) fn queue_action_bar(ui: &mut Ui, app: &mut App2) {
    let margin = 8.0_f32;
    ui.add_space(8.0);

    // ── Multi-face mode ──────────────────────────────────────────────────────
    // The inverse of the "Auto-select best face" quality rule: with that rule off the
    // batch already exports every detected face.
    ui.horizontal(|ui| {
        ui.add_space(margin);
        let mut every_face = !app.settings.crop.quality_rules.auto_select_best_face;
        if ui
            .checkbox(&mut every_face, "Export every face")
            .on_hover_text(
                "Off exports only the best face per image. \
                 \"Skip if no high-quality face\" in Settings still applies.",
            )
            .changed()
        {
            app.settings.crop.quality_rules.auto_select_best_face = !every_face;
        }
    });
    ui.add_space(4.0);

    // ── Run batch ────────────────────────────────────────────────────────────
    ui.horizontal(|ui| {
        ui.add_space(margin);
        let avail = ui.available_width() - margin;
        let n = app.batch_files.len();
        let enabled = !app.is_busy && app.detector.is_some();
        ui.add_enabled_ui(enabled, |ui| {
            if ui
                .add_sized(
                    Vec2::new(avail, 36.0),
                    egui::Button::new(
                        RichText::new(format!("Run batch  ({n} images) →"))
                            .size(11.0)
                            .family(egui::FontFamily::Monospace)
                            .color(P::BG),
                    )
                    .fill(P::ACCENT),
                )
                .clicked()
            {
                crate::core::export::start_batch_export(app);
            }
        });
    });

    // ── Export queue list ────────────────────────────────────────────────────
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.add_space(margin);
        let avail = ui.available_width() - margin;
        if ui
            .add_sized(
                Vec2::new(avail, 24.0),
                egui::Button::new(
                    RichText::new("Export queue list…")
                        .size(10.0)
                        .family(egui::FontFamily::Monospace)
                        .color(P::INK2),
                ),
            )
            .clicked()
            && let Some(path) = rfd::FileDialog::new()
                .set_file_name("queue.txt")
                .add_filter("Text", &["txt"])
                .save_file()
        {
            let lines: Vec<String> = app
                .batch_files
                .iter()
                .map(|f| f.path.display().to_string())
                .collect();
            match fcs_utils::write_atomically(&path, lines.join("\n").as_bytes()) {
                Ok(_) => app.show_success(format!(
                    "Queue exported to {}",
                    path.file_name().and_then(|n| n.to_str()).unwrap_or("file")
                )),
                Err(e) => app.show_error("Export failed", e.to_string()),
            }
        }
    });

    // ── Batch report ─────────────────────────────────────────────────────────
    let has_results = !app.is_busy
        && app.batch_files.iter().any(|f| {
            !matches!(
                f.status,
                BatchFileStatus::Pending | BatchFileStatus::Processing
            )
        });
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.add_space(margin);
        let avail = ui.available_width() - margin;
        ui.add_enabled_ui(has_results, |ui| {
            if ui
                .add_sized(
                    Vec2::new(avail, 24.0),
                    egui::Button::new(
                        RichText::new("Export batch report…")
                            .size(10.0)
                            .family(egui::FontFamily::Monospace)
                            .color(P::INK2),
                    ),
                )
                .on_hover_text("Every image with its outcome. Saved as JSON or CSV by extension.")
                .clicked()
                && let Some(path) = rfd::FileDialog::new()
                    .set_file_name("batch_report.csv")
                    .add_filter("CSV", &["csv"])
                    .add_filter("JSON", &["json"])
                    .save_file()
            {
                let csv = !path
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("json"));
                let written = crate::core::export::batch_report(&app.batch_files, csv)
                    .and_then(|bytes| fcs_utils::write_atomically(&path, &bytes));
                match written {
                    Ok(()) => app.show_success(format!("Batch report saved to {}", path.display())),
                    Err(e) => app.show_error("Report export failed", format!("{e:#}")),
                }
            }
        });
    });
    ui.add_space(6.0);
}

fn drop_zone(ui: &mut Ui, app: &mut App2) {
    ui.add_space(8.0);
    let dz_rect = egui::Rect::from_min_size(
        egui::pos2(ui.min_rect().min.x + 8.0, ui.cursor().min.y),
        Vec2::new(ui.available_width() - 16.0, 100.0),
    );
    let resp = ui
        .allocate_rect(dz_rect, Sense::click())
        .on_hover_cursor(egui::CursorIcon::PointingHand);
    let painter = ui.painter();

    let border_color = if resp.hovered() {
        P::CYAN
    } else {
        P::cyan_alpha(89)
    };
    let bg_color = if resp.hovered() {
        P::cyan_alpha(30)
    } else {
        P::cyan_alpha(20)
    };

    // Draw dashed border
    painter.rect_filled(dz_rect, 10.0, bg_color);
    draw_dashed_border(painter, dz_rect, border_color);

    // Icon
    let icon_rect = egui::Rect::from_center_size(
        egui::pos2(dz_rect.center().x, dz_rect.min.y + 28.0),
        Vec2::splat(34.0),
    );
    painter.rect_filled(icon_rect, 9.0, P::cyan_alpha(40));
    painter.text(
        icon_rect.center(),
        egui::Align2::CENTER_CENTER,
        "↑",
        egui::FontId::proportional(16.0),
        P::CYAN,
    );

    painter.text(
        egui::pos2(dz_rect.center().x, dz_rect.min.y + 52.0),
        egui::Align2::CENTER_CENTER,
        "Drop images or folder",
        egui::FontId::proportional(13.0),
        P::INK,
    );
    painter.text(
        egui::pos2(dz_rect.center().x, dz_rect.min.y + 68.0),
        egui::Align2::CENTER_CENTER,
        "Images · folders · clipboard",
        egui::FontId::proportional(11.5),
        P::INK3,
    );
    // Hit-test the paste label area (right half of bottom strip) for hover highlight
    let action_y = dz_rect.min.y + 86.0;
    let paste_area = egui::Rect::from_center_size(
        egui::pos2(dz_rect.center().x + 38.0, action_y),
        egui::Vec2::new(56.0, 14.0),
    );
    let pointer_in_paste = ui.input(|i| {
        i.pointer
            .latest_pos()
            .is_some_and(|p| paste_area.contains(p))
    });
    let browse_color = if resp.hovered() && !pointer_in_paste {
        P::CYAN
    } else {
        P::INK2
    };
    let paste_color = if pointer_in_paste { P::CYAN } else { P::INK2 };

    painter.text(
        egui::pos2(dz_rect.center().x - 34.0, action_y),
        egui::Align2::CENTER_CENTER,
        "[ Browse ]",
        egui::FontId::monospace(10.0),
        browse_color,
    );
    painter.text(
        egui::pos2(dz_rect.center().x + 38.0, action_y),
        egui::Align2::CENTER_CENTER,
        "[ Paste ]",
        egui::FontId::monospace(10.0),
        paste_color,
    );
    ui.add_space(8.0);

    if resp.clicked() {
        if resp
            .interact_pointer_pos()
            .is_some_and(|p| paste_area.contains(p))
        {
            app.paste_clipboard_image();
        } else {
            // Open file dialog
            if let Some(paths) = rfd::FileDialog::new()
                .add_filter("Images", fcs_utils::SUPPORTED_IMAGE_EXTENSIONS)
                .pick_files()
            {
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
                }
            }
        }
    }
}

pub(super) fn draw_dashed_border(painter: &egui::Painter, rect: egui::Rect, color: egui::Color32) {
    let stroke = Stroke::new(1.5, color);
    let dash = 6.0;
    let gap = 4.0;
    let r = rect;
    let mut x = r.min.x;
    while x < r.max.x {
        let ex = (x + dash).min(r.max.x);
        painter.line_segment([egui::pos2(x, r.min.y), egui::pos2(ex, r.min.y)], stroke);
        painter.line_segment([egui::pos2(x, r.max.y), egui::pos2(ex, r.max.y)], stroke);
        x += dash + gap;
    }
    let mut y = r.min.y;
    while y < r.max.y {
        let ey = (y + dash).min(r.max.y);
        painter.line_segment([egui::pos2(r.min.x, y), egui::pos2(r.min.x, ey)], stroke);
        painter.line_segment([egui::pos2(r.max.x, y), egui::pos2(r.max.x, ey)], stroke);
        y += dash + gap;
    }
}

fn file_tree(ui: &mut Ui, app: &mut App2) {
    if app.batch_files.is_empty() {
        return;
    }

    let total = app.batch_files.len();
    let mut action = None;
    let is_in_progress = |status: &BatchFileStatus| {
        matches!(
            status,
            BatchFileStatus::Processing
                | BatchFileStatus::Completed { .. }
                | BatchFileStatus::Failed { .. }
        )
    };
    let in_progress_count = app
        .batch_files
        .iter()
        .filter(|file| is_in_progress(&file.status))
        .count();
    let queued_count = total - in_progress_count;

    if in_progress_count > 0 {
        tree_group_header(ui, "Batch results", in_progress_count, total);
        for idx in 0..total {
            if !is_in_progress(&app.batch_files[idx].status) {
                continue;
            }
            let row_action = tree_row(ui, app, idx);
            if action.is_none() {
                action = row_action;
            }
        }
    }
    if queued_count > 0 {
        tree_group_header(ui, "Queued", queued_count, 0);
        for idx in 0..total {
            if is_in_progress(&app.batch_files[idx].status) {
                continue;
            }
            let row_action = tree_row(ui, app, idx);
            if action.is_none() {
                action = row_action;
            }
        }
    }

    match action {
        Some(TreeAction::Load(path)) => app.load_image_path(path),
        Some(TreeAction::Remove(idx)) if idx < app.batch_files.len() => {
            app.batch_files.remove(idx);
            app.show_success("Removed image from queue.");
        }
        _ => {}
    }
}

enum TreeAction {
    Load(std::path::PathBuf),
    Remove(usize),
}

fn tree_group_header(ui: &mut Ui, label: &str, count: usize, total: usize) {
    ui.horizontal(|ui| {
        ui.set_height(28.0);
        ui.add_space(8.0);
        ui.label(
            egui::RichText::new(label)
                .size(10.0)
                .color(P::INK3)
                .family(egui::FontFamily::Monospace),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add_space(8.0);
            let count_str = if total > 0 {
                format!("{count} / {total}")
            } else {
                count.to_string()
            };
            ui.label(
                egui::RichText::new(count_str)
                    .size(10.0)
                    .color(P::PEACH)
                    .family(egui::FontFamily::Monospace),
            );
        });
    });
}

fn tree_row(ui: &mut Ui, app: &App2, idx: usize) -> Option<TreeAction> {
    let file = &app.batch_files[idx];
    let name = file
        .path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("?");
    let active = app.preview.image_path.as_deref() == Some(file.path.as_path());
    let (status, color) = match &file.status {
        BatchFileStatus::Pending => ("Queued".to_string(), P::INK3),
        BatchFileStatus::Processing => ("Processing…".to_string(), P::PEACH),
        BatchFileStatus::Completed {
            faces_detected: 0, ..
        } => ("! No faces found".to_string(), P::PEACH),
        BatchFileStatus::Completed {
            faces_detected,
            faces_exported: 0,
        } => (
            format!("! {faces_detected} found, none passed quality rules"),
            P::PEACH,
        ),
        BatchFileStatus::Completed {
            faces_detected,
            faces_exported,
        } => (
            format!("✔ Exported {faces_exported} of {faces_detected} face(s)"),
            P::GREEN,
        ),
        BatchFileStatus::Failed { .. } => ("× Failed (hover for why)".to_string(), P::RED),
        BatchFileStatus::Skipped => ("— Skipped".to_string(), P::INK3),
    };
    let mut action = None;
    egui::Frame::new()
        .inner_margin(egui::Margin::symmetric(8, 5))
        .fill(if active {
            P::SURFACE2
        } else {
            egui::Color32::TRANSPARENT
        })
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                let width = (ui.available_width() - 38.0).max(40.0);
                let text = RichText::new(format!("{:02}  {name}", idx + 1))
                    .size(13.0)
                    .color(if active { P::BG } else { P::INK });
                if ui
                    .add_sized(
                        [width, 30.0],
                        egui::Button::new(text).truncate().selected(active),
                    )
                    .on_hover_text(file.path.display().to_string())
                    .clicked()
                {
                    action = Some(TreeAction::Load(file.path.clone()));
                }
                if ui.button("×").on_hover_text("Remove from queue").clicked() {
                    action = Some(TreeAction::Remove(idx));
                }
            });
            let label = if active {
                format!("Editing · {status}")
            } else {
                status
            };
            let response = ui.label(RichText::new(label).size(11.5).color(color));
            if let BatchFileStatus::Failed { error } = &file.status {
                response.on_hover_text(error);
            }
        });
    action
}
