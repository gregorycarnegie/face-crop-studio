//! Toolbar ribbon.

use crate::theme::P;
use crate::types::App2;
use crate::ui::widgets::{gpu_pill, tb_sep};
use egui::{Color32, Frame, Sense, Stroke, Ui, Vec2};

pub fn show(ui: &mut Ui, app: &mut App2) {
    egui::Panel::top("toolbar")
        .exact_size(52.0)
        .show_separator_line(false)
        .frame(
            Frame::new()
                .fill(P::SURFACE.linear_multiply(0.6))
                .inner_margin(egui::Margin::symmetric(12, 8)),
        )
        .show_inside(ui, |ui| {
            ui.horizontal_centered(|ui| {
                // Primary action: Detect
                if primary_btn(ui, "Detect faces →", P::CYAN, bg_from_cyan())
                    && let Some(path) = app.preview.image_path.clone()
                {
                    app.load_image_path(path);
                }
                ui.add_space(4.0);

                // Secondary action: Export
                if primary_btn(ui, "Export crops", P::PEACH, bg_from_peach()) {
                    if app.selected_faces.is_empty() && !app.batch_files.is_empty() {
                        crate::core::export::start_batch_export(app);
                    } else {
                        crate::core::export::export_selected_faces(app);
                    }
                }
                tb_sep(ui);

                // Icon buttons
                icon_btn(ui, "📂", "Open", || {
                    if let Some(paths) = rfd::FileDialog::new()
                        .add_filter(
                            "Images",
                            &["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
                        )
                        .pick_files()
                    {
                        let first = paths.first().cloned();
                        app.enqueue_batch_paths(paths);
                        if let Some(path) = first {
                            app.load_image_path(path);
                        }
                    }
                });
                icon_btn(ui, "💾", "Save", || {
                    crate::core::export::export_selected_faces(app);
                });
                icon_btn(ui, "↩", "Undo", || {});
                icon_btn(ui, "↪", "Redo", || {});
                tb_sep(ui);

                // Rotation
                if ghost_btn(ui, "↶ 90°") {
                    app.canvas_rotation = (app.canvas_rotation + 270.0) % 360.0;
                }
                if ghost_btn(ui, "90° ↷") {
                    app.canvas_rotation = (app.canvas_rotation + 90.0) % 360.0;
                }
                tb_sep(ui);

                // Selection
                if ghost_btn(ui, "Select all") {
                    let n = app.preview.detections.len();
                    app.selected_faces = (0..n).collect();
                }
                if ghost_btn(ui, "Select none") {
                    app.selected_faces.clear();
                }
                tb_sep(ui);

                // Clear
                danger_btn(ui, "Clear", || {
                    app.preview = Default::default();
                    app.selected_faces.clear();
                    app.batch_files.clear();
                });

                // Right: GPU pill
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let label = app
                        .gpu_status
                        .adapter_name
                        .as_deref()
                        .map(|n| format!("GPU · {n}"))
                        .unwrap_or_else(|| "GPU · wgpu".to_string());
                    gpu_pill(ui, &label);
                });
            });
        });
}

fn primary_btn(ui: &mut egui::Ui, label: &str, fg: Color32, bg: Color32) -> bool {
    let font = egui::FontId::proportional(12.5);
    let galley = ui.painter().layout_no_wrap(label.to_string(), font, fg);
    let w = galley.size().x + 26.0;
    let (resp, painter) = ui.allocate_painter(Vec2::new(w, 34.0), Sense::click());
    let r = resp.rect;
    let fill = if resp.hovered() {
        lighten(bg, 0.08)
    } else {
        bg
    };
    painter.rect_filled(r, 7.0, fill);
    painter.galley(
        r.min + Vec2::new(13.0, (34.0 - galley.size().y) / 2.0),
        galley,
        fg,
    );
    resp.clicked()
}

fn ghost_btn(ui: &mut egui::Ui, label: &str) -> bool {
    let font = egui::FontId::proportional(12.5);
    let galley = ui.painter().layout_no_wrap(label.to_string(), font, P::INK);
    let w = galley.size().x + 26.0;
    let (resp, painter) = ui.allocate_painter(Vec2::new(w, 34.0), Sense::click());
    let r = resp.rect;
    if resp.hovered() {
        painter.rect_filled(r, 7.0, P::white_alpha(15));
        painter.rect_stroke(
            r,
            7.0,
            Stroke::new(1.0, P::RULE2),
            egui::StrokeKind::Outside,
        );
    } else {
        painter.rect_stroke(
            r,
            7.0,
            Stroke::new(1.0, P::RULE2),
            egui::StrokeKind::Outside,
        );
        painter.rect_filled(r, 7.0, P::white_alpha(5));
    }
    painter.galley(
        r.min + Vec2::new(13.0, (34.0 - galley.size().y) / 2.0),
        galley,
        P::INK,
    );
    resp.clicked()
}

fn icon_btn(ui: &mut egui::Ui, icon: &str, tooltip: &str, action: impl FnOnce()) -> bool {
    let (resp, painter) = ui.allocate_painter(Vec2::splat(34.0), Sense::click());
    let r = resp.rect;
    if resp.hovered() {
        painter.rect_filled(r, 7.0, P::white_alpha(15));
    }
    painter.rect_stroke(
        r,
        7.0,
        Stroke::new(1.0, P::RULE2),
        egui::StrokeKind::Outside,
    );
    painter.text(
        r.center(),
        egui::Align2::CENTER_CENTER,
        icon,
        egui::FontId::proportional(14.0),
        P::INK2,
    );
    let clicked = resp.clicked();
    resp.on_hover_text(tooltip);
    if clicked {
        action();
    }
    clicked
}

fn danger_btn(ui: &mut egui::Ui, label: &str, action: impl FnOnce()) -> bool {
    let font = egui::FontId::proportional(12.5);
    let galley = ui
        .painter()
        .layout_no_wrap(label.to_string(), font, P::ROSE);
    let w = galley.size().x + 26.0;
    let (resp, painter) = ui.allocate_painter(Vec2::new(w, 34.0), Sense::click());
    let r = resp.rect;
    let bg = if resp.hovered() {
        P::rose_alpha(30)
    } else {
        P::rose_alpha(12)
    };
    painter.rect_filled(r, 7.0, bg);
    painter.rect_stroke(
        r,
        7.0,
        Stroke::new(1.0, P::rose_alpha(64)),
        egui::StrokeKind::Outside,
    );
    painter.galley(
        r.min + Vec2::new(13.0, (34.0 - galley.size().y) / 2.0),
        galley,
        P::ROSE,
    );
    if resp.clicked() {
        action();
        true
    } else {
        false
    }
}

fn lighten(c: Color32, amt: f32) -> Color32 {
    let f = |v: u8| ((v as f32 + amt * 255.0).min(255.0)) as u8;
    Color32::from_rgba_unmultiplied(f(c.r()), f(c.g()), f(c.b()), c.a())
}

fn bg_from_cyan() -> Color32 {
    Color32::from_rgb(
        (0x7b_u8 as f32 * 0.35) as u8,
        (0xe0_u8 as f32 * 0.35) as u8,
        (0xd6_u8 as f32 * 0.35) as u8,
    )
}
fn bg_from_peach() -> Color32 {
    Color32::from_rgb(
        (0xff_u8 as f32 * 0.40) as u8,
        (0xb8_u8 as f32 * 0.35) as u8,
        (0x9a_u8 as f32 * 0.30) as u8,
    )
}
