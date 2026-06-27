//! Crop history tab UI.

use crate::{theme::P, types::App2};
use egui::{RichText, Sense, Stroke, Ui, Vec2};

pub(super) fn show_history(ui: &mut Ui, app: &mut App2) {
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
