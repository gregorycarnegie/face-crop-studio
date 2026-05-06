//! Windows-11-style title bar with drag-to-move and custom chrome buttons.

use crate::theme::P;
use crate::types::App2;
use egui::{Color32, Frame, Sense, Stroke, Ui, Vec2};

const BTN_W: f32 = 46.0;

pub fn show(ui: &mut Ui, app: &mut App2) {
    egui::Panel::top("titlebar")
        .exact_size(36.0)
        .frame(Frame::new()
            .fill(P::BG)
            .inner_margin(egui::Margin::ZERO))
        .show_inside(ui, |ui| {
            let full = ui.max_rect();

            // Bottom border
            ui.painter().line_segment(
                [egui::pos2(full.min.x, full.max.y - 1.0),
                 egui::pos2(full.max.x, full.max.y - 1.0)],
                Stroke::new(1.0, P::RULE),
            );

            // Drag region — everything left of the three chrome buttons
            let btns_x = full.max.x - BTN_W * 3.0;
            let drag_rect = egui::Rect::from_min_max(full.min, egui::pos2(btns_x, full.max.y));
            let drag = ui.interact(drag_rect, ui.id().with("tb_drag"), Sense::click_and_drag());
            if drag.drag_started() {
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::StartDrag);
            }
            if drag.double_clicked() {
                let maximized = ui.ctx().input(|i| i.viewport().maximized.unwrap_or(false));
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Maximized(!maximized));
            }

            // Left content (logo + app name + file name) inside the drag area
            let mut left = ui.new_child(
                egui::UiBuilder::new()
                    .max_rect(drag_rect)
                    .layout(egui::Layout::left_to_right(egui::Align::Center)),
            );
            left.add_space(12.0);
            draw_logo(&mut left);
            left.add_space(8.0);
            left.label(egui::RichText::new("Face Crop Studio").size(12.5).color(P::INK).strong());
            left.add_space(6.0);
            left.label(egui::RichText::new("—").size(12.5).color(P::RULE2));
            left.add_space(6.0);

            if let Some(path) = &app.preview.image_path.clone() {
                let name = path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown");
                left.label(egui::RichText::new(name).size(11.5).color(P::INK3).family(egui::FontFamily::Monospace));
                if app.is_busy {
                    left.label(egui::RichText::new("●").size(11.5).color(P::PEACH));
                }
            } else {
                left.label(egui::RichText::new("No file").size(11.5).color(P::INK3).family(egui::FontFamily::Monospace));
            }

            // Chrome buttons — right to left: Close, Maximize, Minimize
            for (i, icon) in [WinIcon::Close, WinIcon::Maximize, WinIcon::Minimize].iter().enumerate() {
                let btn_rect = egui::Rect::from_min_max(
                    egui::pos2(full.max.x - BTN_W * (i as f32 + 1.0), full.min.y),
                    egui::pos2(full.max.x - BTN_W * i as f32, full.max.y),
                );
                win_btn(ui, btn_rect, *icon);
            }
        });
}

#[derive(Clone, Copy)]
enum WinIcon { Minimize, Maximize, Close }

fn win_btn(ui: &mut egui::Ui, rect: egui::Rect, icon: WinIcon) {
    let id = match icon {
        WinIcon::Minimize => ui.id().with("wb_min"),
        WinIcon::Maximize => ui.id().with("wb_max"),
        WinIcon::Close    => ui.id().with("wb_close"),
    };
    let resp = ui.interact(rect, id, Sense::click());
    let is_close = matches!(icon, WinIcon::Close);

    let bg = if resp.hovered() {
        if is_close { Color32::from_rgb(0xe8, 0x11, 0x23) } else { P::white_alpha(15) }
    } else {
        Color32::TRANSPARENT
    };
    let painter = ui.painter();
    if bg != Color32::TRANSPARENT {
        painter.rect_filled(rect, 0.0, bg);
    }

    let c = rect.center();
    let col = if resp.hovered() && is_close { Color32::WHITE } else { P::INK2 };
    let sw = Stroke::new(1.5, col);
    match icon {
        WinIcon::Minimize => {
            painter.line_segment(
                [c - Vec2::new(5.0, 0.0), c + Vec2::new(5.0, 0.0)],
                sw,
            );
        }
        WinIcon::Maximize => {
            let r = egui::Rect::from_center_size(c, Vec2::splat(10.0));
            painter.rect_stroke(r, 1.0, sw, egui::StrokeKind::Outside);
        }
        WinIcon::Close => {
            let d = 4.5;
            painter.line_segment([c + Vec2::new(-d, -d), c + Vec2::new(d, d)], sw);
            painter.line_segment([c + Vec2::new(d, -d), c + Vec2::new(-d, d)], sw);
        }
    }

    if resp.clicked() {
        match icon {
            WinIcon::Minimize => ui.ctx().send_viewport_cmd(egui::ViewportCommand::Minimized(true)),
            WinIcon::Maximize => {
                let maximized = ui.ctx().input(|i| i.viewport().maximized.unwrap_or(false));
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Maximized(!maximized));
            }
            WinIcon::Close => ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close),
        }
    }
}

fn draw_logo(ui: &mut egui::Ui) {
    let (resp, painter) = ui.allocate_painter(Vec2::splat(18.0), Sense::hover());
    let c = resp.rect.center();
    let r = 9.0;
    painter.circle_filled(c, r, P::PEACH);
    painter.circle_filled(c, r * 0.8, P::ROSE);
    painter.circle_filled(c, r * 0.55, P::BG);
    painter.circle_filled(c, r * 0.28, P::PEACH);
}
