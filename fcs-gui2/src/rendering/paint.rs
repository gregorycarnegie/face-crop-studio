//! egui Painter helpers for detection overlays.

use crate::theme::P;
use egui::{Color32, Painter, Pos2, Rect, Stroke, StrokeKind};

pub fn draw_face_box(painter: &Painter, rect: Rect, color: Color32, selected: bool) {
    let stroke_w = if selected { 2.0 } else { 1.5 };
    let stroke = Stroke::new(stroke_w, color);
    if selected {
        painter.rect_stroke(rect, 4.0, stroke, StrokeKind::Outside);
    } else {
        // dashed for unselected
        draw_dashed_rect(painter, rect, stroke);
    }
}

fn draw_dashed_rect(painter: &Painter, rect: Rect, stroke: Stroke) {
    let dash = 6.0;
    let gap = 4.0;
    draw_dashed_h(
        painter, rect.min.x, rect.max.x, rect.min.y, stroke, dash, gap,
    );
    draw_dashed_h(
        painter, rect.min.x, rect.max.x, rect.max.y, stroke, dash, gap,
    );
    draw_dashed_v(
        painter, rect.min.y, rect.max.y, rect.min.x, stroke, dash, gap,
    );
    draw_dashed_v(
        painter, rect.min.y, rect.max.y, rect.max.x, stroke, dash, gap,
    );
}

fn draw_dashed_h(painter: &Painter, x0: f32, x1: f32, y: f32, stroke: Stroke, dash: f32, gap: f32) {
    let mut x = x0;
    while x < x1 {
        let end = (x + dash).min(x1);
        painter.line_segment([Pos2::new(x, y), Pos2::new(end, y)], stroke);
        x += dash + gap;
    }
}

fn draw_dashed_v(painter: &Painter, y0: f32, y1: f32, x: f32, stroke: Stroke, dash: f32, gap: f32) {
    let mut y = y0;
    while y < y1 {
        let end = (y + dash).min(y1);
        painter.line_segment([Pos2::new(x, y), Pos2::new(x, end)], stroke);
        y += dash + gap;
    }
}

pub fn draw_landmark_dot(painter: &Painter, pos: Pos2) {
    painter.circle_filled(pos, 3.5, P::LIME);
    painter.circle_stroke(pos, 3.5, Stroke::new(0.5, P::black_alpha(100)));
}

pub fn draw_drag_handle(painter: &Painter, center: Pos2, color: Color32) {
    let hs = 4.5;
    let rect = Rect::from_center_size(center, egui::Vec2::splat(hs * 2.0));
    painter.rect_filled(rect, 2.0, color);
    painter.rect_stroke(rect, 2.0, Stroke::new(0.5, P::BG), StrokeKind::Outside);
}

pub fn draw_confidence_badge(painter: &Painter, text: &str, above_rect: Rect, color: Color32) {
    let font_id = egui::FontId::monospace(9.5);
    let galley = painter.layout_no_wrap(text.to_owned(), font_id.clone(), color);
    let pad = egui::Vec2::new(5.0, 2.0);
    let text_rect = Rect::from_min_size(
        Pos2::new(
            above_rect.min.x - 1.0,
            above_rect.min.y - galley.size().y - pad.y * 2.0 - 2.0,
        ),
        galley.size() + pad * 2.0,
    );
    painter.rect_filled(text_rect, 2.0, color);
    painter.galley(text_rect.min + pad, galley, P::BG);
}
