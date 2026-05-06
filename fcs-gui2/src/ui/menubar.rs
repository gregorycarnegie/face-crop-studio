//! Menu bar.

use crate::theme::P;
use crate::types::App2;
use egui::{Frame, Sense, Stroke, Ui, Vec2};

const ITEMS: &[&str] = &["File", "Edit", "View", "Detect", "Tools", "Window", "Help"];

pub fn show(ui: &mut Ui, app: &mut App2) {
    egui::Panel::top("menubar")
        .exact_size(32.0)
        .frame(
            Frame::new()
                .fill(P::white_alpha(3))
                .inner_margin(egui::Margin::ZERO),
        )
        .show_inside(ui, |ui| {
            // Bottom border
            ui.painter().line_segment(
                [
                    egui::pos2(ui.min_rect().min.x, ui.min_rect().max.y - 1.0),
                    egui::pos2(ui.min_rect().max.x, ui.min_rect().max.y - 1.0),
                ],
                Stroke::new(1.0, P::RULE),
            );

            ui.horizontal_centered(|ui| {
                ui.add_space(6.0);
                for &item in ITEMS {
                    menu_item(ui, item);
                }

                // Right: model status
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add_space(10.0);
                    let model_ready = app.detector.is_some();
                    let (dot_color, status_text) = if model_ready {
                        (P::LIME, format!("YuNet 640 · ready"))
                    } else {
                        (P::ROSE, "YuNet · no model".to_string())
                    };
                    ui.label(
                        egui::RichText::new(&status_text)
                            .size(10.5)
                            .color(P::INK3)
                            .family(egui::FontFamily::Monospace),
                    );
                    // Dot
                    let (resp, painter) = ui.allocate_painter(Vec2::splat(6.0), Sense::hover());
                    painter.circle_filled(resp.rect.center(), 3.0, dot_color);
                });
            });
        });
}

fn menu_item(ui: &mut egui::Ui, label: &str) {
    let font = egui::FontId::proportional(13.0);
    let galley = ui
        .painter()
        .layout_no_wrap(label.to_string(), font, P::INK2);
    let w = galley.size().x + 20.0;
    let (resp, painter) = ui.allocate_painter(Vec2::new(w, 32.0), Sense::click());
    if resp.hovered() {
        painter.rect_filled(
            resp.rect.shrink2(Vec2::new(2.0, 4.0)),
            5.0,
            P::white_alpha(15),
        );
    }
    let text_color = if resp.hovered() { P::INK } else { P::INK2 };
    painter.galley(
        egui::pos2(
            resp.rect.min.x + 10.0,
            resp.rect.center().y - galley.size().y / 2.0,
        ),
        galley,
        text_color,
    );
}
