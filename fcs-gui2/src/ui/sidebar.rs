//! Left sidebar: Queue / Mapping / History tabs.

use crate::theme::P;
use crate::types::{App2, BatchFileStatus, SidebarTab};
use egui::{RichText, Sense, Stroke, Ui, Vec2};

pub fn show(ui: &mut Ui, app: &mut App2) {
    ui.set_min_height(ui.available_height());

    // Tab bar
    tab_bar(ui, app);

    // Scrollable content
    egui::ScrollArea::vertical()
        .id_salt("sidebar_scroll")
        .show(ui, |ui| match app.sidebar_tab {
            SidebarTab::Queue => show_queue(ui, app),
            SidebarTab::Mapping => show_mapping(ui, app),
            SidebarTab::History => show_history(ui, app),
        });
}

fn tab_bar(ui: &mut Ui, app: &mut App2) {
    let tabs = [
        ("Queue", SidebarTab::Queue),
        ("Mapping", SidebarTab::Mapping),
        ("History", SidebarTab::History),
    ];
    ui.painter().line_segment(
        [
            egui::pos2(ui.min_rect().min.x, ui.min_rect().min.y + 32.0),
            egui::pos2(ui.min_rect().max.x, ui.min_rect().min.y + 32.0),
        ],
        Stroke::new(1.0, P::RULE),
    );
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        ui.set_height(32.0);
        let w = ui.available_width() / tabs.len() as f32;
        for (label, variant) in &tabs {
            let is_active = app.sidebar_tab == *variant;
            let (resp, painter) = ui.allocate_painter(Vec2::new(w, 32.0), Sense::click());
            let text_color = if is_active { P::PEACH } else { P::INK3 };
            if resp.hovered() && !is_active {
                painter.rect_filled(resp.rect, 0.0, P::white_alpha(5));
            }
            if is_active {
                painter.rect_filled(resp.rect, 0.0, P::peach_alpha(10));
                painter.line_segment(
                    [
                        egui::pos2(resp.rect.min.x, resp.rect.max.y - 2.0),
                        egui::pos2(resp.rect.max.x, resp.rect.max.y - 2.0),
                    ],
                    Stroke::new(2.0, P::PEACH),
                );
            }
            painter.text(
                resp.rect.center(),
                egui::Align2::CENTER_CENTER,
                *label,
                egui::FontId::monospace(10.5),
                text_color,
            );
            if resp.clicked() {
                app.sidebar_tab = *variant;
            }
        }
    });
}

fn show_queue(ui: &mut Ui, app: &mut App2) {
    // Drop zone
    drop_zone(ui, app);

    // Folder browse shortcut
    folder_browse_bar(ui, app);

    // File tree
    file_tree(ui, app);

    // Run batch + export buttons
    if !app.batch_files.is_empty() {
        queue_action_bar(ui, app);
    }
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

fn queue_action_bar(ui: &mut Ui, app: &mut App2) {
    let margin = 8.0_f32;
    ui.add_space(8.0);

    // ── Run batch ────────────────────────────────────────────────────────────
    ui.horizontal(|ui| {
        ui.add_space(margin);
        let avail = ui.available_width() - margin;
        let n = app.batch_files.len();
        let enabled = !app.is_busy && app.detector.is_some();
        ui.add_enabled_ui(enabled, |ui| {
            if ui
                .add_sized(
                    Vec2::new(avail, 30.0),
                    egui::Button::new(
                        RichText::new(format!("Run batch  ({n} images) →"))
                            .size(11.0)
                            .family(egui::FontFamily::Monospace)
                            .color(P::PEACH),
                    ),
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
            match std::fs::write(&path, lines.join("\n")) {
                Ok(_) => app.show_success(format!(
                    "Queue exported to {}",
                    path.file_name().and_then(|n| n.to_str()).unwrap_or("file")
                )),
                Err(e) => app.show_error("Export failed", e.to_string()),
            }
        }
    });
    ui.add_space(6.0);
}

fn drop_zone(ui: &mut Ui, app: &mut App2) {
    ui.add_space(8.0);
    let dz_rect = egui::Rect::from_min_size(
        egui::pos2(ui.min_rect().min.x + 8.0, ui.cursor().min.y),
        Vec2::new(ui.available_width() - 16.0, 100.0),
    );
    let resp = ui.allocate_rect(dz_rect, Sense::click());
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
        "JPG · PNG · WEBP",
        egui::FontId::proportional(11.5),
        P::INK3,
    );
    painter.text(
        egui::pos2(dz_rect.center().x, dz_rect.min.y + 86.0),
        egui::Align2::CENTER_CENTER,
        "[ Browse ]  [ Paste ]",
        egui::FontId::monospace(10.0),
        P::INK2,
    );
    ui.add_space(108.0);

    if resp.clicked() {
        // Open file dialog
        if let Some(paths) = rfd::FileDialog::new()
            .add_filter(
                "Images",
                &["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
            )
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

fn draw_dashed_border(painter: &egui::Painter, rect: egui::Rect, color: egui::Color32) {
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
    let in_progress: Vec<usize> = (0..total)
        .filter(|&i| {
            matches!(
                app.batch_files[i].status,
                BatchFileStatus::Processing
                    | BatchFileStatus::Completed { .. }
                    | BatchFileStatus::Failed { .. }
            )
        })
        .collect();
    let queued: Vec<usize> = (0..total)
        .filter(|&i| {
            !matches!(
                app.batch_files[i].status,
                BatchFileStatus::Processing
                    | BatchFileStatus::Completed { .. }
                    | BatchFileStatus::Failed { .. }
            )
        })
        .collect();

    if !in_progress.is_empty() {
        tree_group_header(ui, "In progress", in_progress.len(), total);
        for idx in in_progress {
            let row_action = tree_row(ui, app, idx);
            if action.is_none() {
                action = row_action;
            }
        }
    }
    if !queued.is_empty() {
        tree_group_header(ui, "Queued", queued.len(), 0);
        for idx in queued {
            let row_action = tree_row(ui, app, idx);
            if action.is_none() {
                action = row_action;
            }
        }
    }

    match action {
        Some(TreeAction::Load(path)) => app.load_image_path(path),
        Some(TreeAction::Remove(idx)) => {
            if idx < app.batch_files.len() {
                app.batch_files.remove(idx);
                app.show_success("Removed image from queue.");
            }
        }
        None => {}
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
    let path = app.batch_files[idx].path.clone();
    let status = app.batch_files[idx].status.clone();
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("?")
        .to_string();
    let status_label = status.badge_label().to_string();
    let face_count = status.face_count();

    let is_active = app.preview.image_path.as_deref() == Some(path.as_path());

    let row_h = 32.0;
    let (resp, painter) =
        ui.allocate_painter(Vec2::new(ui.available_width(), row_h), Sense::click());
    let r = resp.rect;

    let bg = if is_active {
        P::peach_alpha(25)
    } else if resp.hovered() {
        P::white_alpha(10)
    } else {
        egui::Color32::TRANSPARENT
    };
    if bg != egui::Color32::TRANSPARENT {
        painter.rect_filled(r, 5.0, bg);
    }

    let text_color = if is_active { P::PEACH } else { P::INK2 };

    // Index
    painter.text(
        egui::pos2(r.min.x + 20.0, r.center().y),
        egui::Align2::RIGHT_CENTER,
        format!("{:02}", idx + 1),
        egui::FontId::monospace(10.0),
        P::INK3,
    );
    // Person icon
    painter.text(
        egui::pos2(r.min.x + 34.0, r.center().y),
        egui::Align2::CENTER_CENTER,
        "◉",
        egui::FontId::proportional(11.0),
        P::INK3,
    );
    // Filename
    let avail_w = r.width() - 132.0;
    painter.text(
        egui::pos2(r.min.x + 46.0, r.center().y),
        egui::Align2::LEFT_CENTER,
        truncate(&name, avail_w, &egui::FontId::monospace(11.5), &painter),
        egui::FontId::monospace(11.5),
        text_color,
    );

    // Badge on the right
    let badge_label = if let Some(count) = face_count {
        count.to_string()
    } else {
        status_label.clone()
    };
    let (badge_bg, badge_fg) = crate::theme::badge_color(&status_label);
    let badge_font = egui::FontId::monospace(9.5);
    let badge_g = painter.layout_no_wrap(badge_label.clone(), badge_font, badge_fg);
    let bw = badge_g.size().x + 10.0;
    let bh = badge_g.size().y + 4.0;
    let remove_rect = egui::Rect::from_center_size(
        egui::pos2(r.max.x - 18.0, r.center().y),
        Vec2::new(24.0, 24.0),
    );
    let load_rect = egui::Rect::from_center_size(
        egui::pos2(r.max.x - 46.0, r.center().y),
        Vec2::new(24.0, 24.0),
    );
    let badge_rect = egui::Rect::from_min_size(
        egui::pos2(load_rect.min.x - bw - 8.0, r.center().y - bh / 2.0),
        Vec2::new(bw, bh),
    );
    painter.rect_filled(badge_rect, 3.0, badge_bg);
    painter.galley(badge_rect.min + Vec2::new(5.0, 2.0), badge_g, badge_fg);

    let load_clicked = row_icon_button(ui, load_rect, idx, "load", "▶", P::CYAN, "Load image");
    let remove_clicked = row_icon_button(
        ui,
        remove_rect,
        idx,
        "remove",
        "×",
        P::ROSE,
        "Remove from queue",
    );

    if remove_clicked {
        Some(TreeAction::Remove(idx))
    } else if load_clicked || resp.clicked() {
        Some(TreeAction::Load(path))
    } else {
        None
    }
}

fn row_icon_button(
    ui: &mut Ui,
    rect: egui::Rect,
    idx: usize,
    salt: &str,
    label: &str,
    color: egui::Color32,
    tooltip: &str,
) -> bool {
    let resp = ui.interact(rect, ui.id().with((salt, idx)), Sense::click());
    let bg = if resp.hovered() {
        P::white_alpha(20)
    } else {
        P::white_alpha(8)
    };
    ui.painter().rect_filled(rect, 5.0, bg);
    ui.painter().rect_stroke(
        rect,
        5.0,
        Stroke::new(1.0, P::white_alpha(24)),
        egui::StrokeKind::Outside,
    );
    ui.painter().text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        label,
        egui::FontId::proportional(12.0),
        color,
    );
    let clicked = resp.clicked();
    resp.on_hover_text(tooltip);
    clicked
}

fn truncate(s: &str, max_w: f32, _font: &egui::FontId, _painter: &egui::Painter) -> String {
    // Simple approximation — assume ~7px per char for monospace
    let chars_fit = (max_w / 7.0) as usize;
    if s.len() <= chars_fit {
        return s.to_string();
    }
    format!("{}…", &s[..chars_fit.saturating_sub(1)])
}

fn show_mapping(ui: &mut Ui, app: &mut App2) {
    let pad = 10.0_f32;
    ui.add_space(10.0);

    // ── File picker ──────────────────────────────────────────────────────────
    let file_label = app
        .mapping
        .file_path
        .as_deref()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "No file loaded".to_string());

    ui.horizontal(|ui| {
        ui.add_space(pad);
        ui.label(
            RichText::new(&file_label)
                .size(11.0)
                .color(if app.mapping.file_path.is_some() {
                    P::INK
                } else {
                    P::INK3
                })
                .family(egui::FontFamily::Monospace),
        );
    });

    ui.add_space(6.0);
    ui.horizontal(|ui| {
        ui.add_space(pad);
        if ui.button("Browse…").clicked()
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
        if app.mapping.file_path.is_some() && ui.button("Clear").clicked() {
            app.mapping = crate::types::MappingUiState::new();
        }
    });

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
    if let Some(err) = app.mapping.preview_error.clone() {
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
                    let cols = columns.clone();
                    for (i, name) in cols.iter().enumerate() {
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
                    let cols = columns.clone();
                    for (i, name) in cols.iter().enumerate() {
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
                    ui.label(
                        RichText::new("(add images first)")
                            .size(9.5)
                            .color(P::INK3),
                    );
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
    let preview_rows: Vec<Vec<String>> = app
        .mapping
        .preview
        .as_ref()
        .map(|p| p.rows.iter().take(6).cloned().collect())
        .unwrap_or_default();
    let preview_cols = columns.clone();

    if !preview_rows.is_empty() {
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
                        ui.add_space(pad);
                        for col in &preview_cols {
                            ui.label(
                                RichText::new(col)
                                    .size(9.5)
                                    .color(P::CYAN)
                                    .family(egui::FontFamily::Monospace),
                            );
                        }
                        ui.end_row();
                        // Rows
                        for row in &preview_rows {
                            ui.add_space(pad);
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

fn show_history(ui: &mut Ui, app: &mut App2) {
    let pad = 10.0_f32;
    ui.add_space(10.0);

    // Snapshot button
    ui.horizontal(|ui| {
        ui.add_space(pad);
        if ui.button("Take snapshot").clicked() {
            let snap = app.settings.crop.clone();
            // Don't push if identical to the last snapshot
            if app.crop_history.last() != Some(&snap) {
                app.crop_history.push(snap);
                app.crop_history_index = app.crop_history.len() - 1;
                app.show_success(format!("Snapshot {} saved", app.crop_history.len()));
            }
        }
    });

    ui.add_space(8.0);

    if app.crop_history.is_empty() {
        ui.horizontal(|ui| {
            ui.add_space(pad);
            ui.label(RichText::new("No snapshots yet.").size(11.0).color(P::INK3));
        });
        return;
    }

    let current_idx = app.crop_history_index;
    let n = app.crop_history.len();
    let mut restore_idx: Option<usize> = None;

    for i in (0..n).rev() {
        let entry = &app.crop_history[i];
        let is_current = i == current_idx;

        let bg = if is_current {
            P::peach_alpha(20)
        } else {
            egui::Color32::TRANSPARENT
        };

        let row_rect =
            egui::Rect::from_min_size(ui.cursor().min, Vec2::new(ui.available_width(), 46.0));
        let (resp, painter) = ui.allocate_painter(row_rect.size(), Sense::hover());
        let r = resp.rect;

        if bg != egui::Color32::TRANSPARENT {
            painter.rect_filled(r, 4.0, bg);
        }

        // Snapshot number
        painter.text(
            egui::pos2(r.min.x + pad, r.min.y + 8.0),
            egui::Align2::LEFT_TOP,
            format!("Snapshot {}", i + 1),
            egui::FontId::monospace(10.5),
            if is_current { P::PEACH } else { P::INK2 },
        );
        // Dimensions + preset
        let dim_str = format!(
            "{}×{}  ·  {}  ·  {:.0}%",
            entry.output_width, entry.output_height, entry.preset, entry.face_height_pct,
        );
        painter.text(
            egui::pos2(r.min.x + pad, r.min.y + 26.0),
            egui::Align2::LEFT_TOP,
            &dim_str,
            egui::FontId::monospace(9.5),
            P::INK3,
        );

        // Restore button (right side)
        if !is_current {
            let btn_rect = egui::Rect::from_center_size(
                egui::pos2(r.max.x - 36.0, r.center().y),
                Vec2::new(52.0, 22.0),
            );
            let btn_resp = ui.interact(btn_rect, ui.id().with(("hist_restore", i)), Sense::click());
            let btn_bg = if btn_resp.hovered() {
                P::peach_alpha(50)
            } else {
                P::peach_alpha(25)
            };
            painter.rect_filled(btn_rect, 5.0, btn_bg);
            painter.text(
                btn_rect.center(),
                egui::Align2::CENTER_CENTER,
                "Restore",
                egui::FontId::monospace(9.0),
                P::PEACH,
            );
            if btn_resp.clicked() {
                restore_idx = Some(i);
            }
        }

        // Separator
        painter.line_segment(
            [
                egui::pos2(r.min.x + pad, r.max.y - 1.0),
                egui::pos2(r.max.x - pad, r.max.y - 1.0),
            ],
            Stroke::new(1.0, P::RULE),
        );
    }

    if let Some(idx) = restore_idx {
        app.settings.crop = app.crop_history[idx].clone();
        app.crop_history_index = idx;
        app.show_success(format!("Restored snapshot {}", idx + 1));
    }
}

// ── Mapping helpers ───────────────────────────────────────────────────────────

fn apply_mapping_to_queue(app: &mut App2) {
    let entries = app.mapping.entries.clone();
    let mut matched = 0usize;

    for file in &mut app.batch_files {
        let file_name = file
            .path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        let file_stem = file
            .path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();

        let hit = entries.iter().find(|e| {
            let src = std::path::Path::new(&e.source_path);
            let src_name = src.file_name().and_then(|s| s.to_str()).unwrap_or_default();
            let src_stem = src.file_stem().and_then(|s| s.to_str()).unwrap_or_default();
            // Match by full filename, then by stem without extension
            src_name == file_name || src_stem == file_stem
        });

        if let Some(entry) = hit {
            file.output_override = Some(std::path::PathBuf::from(&entry.output_name));
            matched += 1;
        }
    }

    app.show_success(format!(
        "Mapping applied: {matched} / {} queue items matched",
        app.batch_files.len()
    ));
}
