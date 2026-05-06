//! Bottom status bar.

use crate::theme::P;
use crate::types::App2;
use egui::{Frame, Sense, Stroke, Vec2};

pub fn show(ui: &mut egui::Ui, app: &mut App2) {
    egui::Panel::bottom("statusbar")
        .exact_size(28.0)
        .frame(Frame::new().fill(P::BG).inner_margin(egui::Margin::ZERO))
        .show_inside(ui, |ui| {
            ui.painter().line_segment(
                [
                    egui::pos2(ui.min_rect().min.x, ui.min_rect().min.y),
                    egui::pos2(ui.min_rect().max.x, ui.min_rect().min.y),
                ],
                Stroke::new(1.0, P::RULE),
            );

            ui.horizontal_centered(|ui| {
                // Status dot + label
                let (ready_dot, ready_text) = if app.is_busy {
                    (P::PEACH, format!("  Running"))
                } else if app.last_error.is_some() {
                    (P::ROSE, " Error".to_string())
                } else {
                    (P::LIME, " Ready".to_string())
                };
                status_cell(ui, &ready_text, Some(ready_dot));

                // Model
                let model_name = app
                    .settings
                    .model_path
                    .as_deref()
                    .and_then(|p| std::path::Path::new(p).file_stem())
                    .and_then(|s| s.to_str())
                    .unwrap_or("YuNet 640");
                status_cell(ui, &format!("{model_name} · ONNX"), None);

                // GPU
                let gpu_label = app
                    .gpu_status
                    .adapter_name
                    .as_deref()
                    .map(|n| {
                        let backend = app.gpu_status.backend.as_deref().unwrap_or("wgpu");
                        format!("{backend} · {n}")
                    })
                    .unwrap_or_else(|| "wgpu · CPU".to_string());
                status_cell(ui, &gpu_label, None);

                // Batch progress
                if !app.batch_files.is_empty() {
                    let done = app
                        .batch_files
                        .iter()
                        .filter(|f| {
                            !matches!(
                                f.status,
                                crate::types::BatchFileStatus::Pending
                                    | crate::types::BatchFileStatus::Processing
                            )
                        })
                        .count();
                    let total = app.batch_files.len();
                    status_cell_dot(
                        ui,
                        &format!("Batch {done} / {total}"),
                        P::PEACH,
                        app.is_busy,
                    );
                }

                // Right side
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // Clock
                    let time_str = current_time_str();
                    status_cell(ui, &time_str, None);

                    // GPU %
                    status_cell(ui, "GPU —", None);

                    // RAM
                    status_cell(ui, "RAM —", None);

                    // Progress bar if busy
                    if app.is_busy {
                        let (resp, painter) =
                            ui.allocate_painter(Vec2::new(140.0, 5.0), Sense::hover());
                        painter.rect_filled(resp.rect, 3.0, P::white_alpha(15));
                        let fill = egui::Rect::from_min_max(
                            resp.rect.min,
                            egui::pos2(resp.rect.min.x + resp.rect.width() * 0.6, resp.rect.max.y),
                        );
                        // Animated progress — we use time for a simple looping animation
                        painter.rect_filled(fill, 3.0, P::PEACH);
                        ui.add_space(6.0);
                    }
                });
            });
        });
}

fn status_cell(ui: &mut egui::Ui, text: &str, dot: Option<egui::Color32>) {
    let (_, _painter) = ui.allocate_painter(Vec2::new(1.0, 28.0), Sense::hover());
    // draw separator
    ui.painter().line_segment(
        [
            egui::pos2(ui.cursor().min.x, ui.min_rect().min.y),
            egui::pos2(ui.cursor().min.x, ui.min_rect().max.y),
        ],
        Stroke::new(1.0, P::RULE),
    );
    ui.horizontal_centered(|ui| {
        ui.add_space(10.0);
        if let Some(color) = dot {
            let (resp, painter) = ui.allocate_painter(Vec2::splat(6.0), Sense::hover());
            painter.circle_filled(resp.rect.center(), 3.0, color);
            ui.add_space(4.0);
        }
        ui.label(
            egui::RichText::new(text)
                .size(10.5)
                .color(P::INK3)
                .family(egui::FontFamily::Monospace),
        );
        ui.add_space(10.0);
    });
    // right separator
    ui.painter().line_segment(
        [
            egui::pos2(ui.cursor().min.x - 1.0, ui.min_rect().min.y),
            egui::pos2(ui.cursor().min.x - 1.0, ui.min_rect().max.y),
        ],
        Stroke::new(1.0, P::RULE),
    );
}

fn status_cell_dot(ui: &mut egui::Ui, text: &str, dot_color: egui::Color32, _animate: bool) {
    ui.horizontal_centered(|ui| {
        ui.add_space(10.0);
        let (resp, painter) = ui.allocate_painter(Vec2::splat(6.0), Sense::hover());
        painter.circle_filled(resp.rect.center(), 3.0, dot_color);
        ui.add_space(4.0);
        ui.label(
            egui::RichText::new(text)
                .size(10.5)
                .color(P::INK3)
                .family(egui::FontFamily::Monospace),
        );
        ui.add_space(10.0);
        ui.painter().line_segment(
            [
                egui::pos2(ui.cursor().min.x - 1.0, ui.min_rect().min.y),
                egui::pos2(ui.cursor().min.x - 1.0, ui.min_rect().max.y),
            ],
            Stroke::new(1.0, P::RULE),
        );
    });
}

fn current_time_str() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let h = (secs / 3600) % 24;
    let m = (secs / 60) % 60;
    let s = secs % 60;
    format!("{h:02}:{m:02}:{s:02}")
}
