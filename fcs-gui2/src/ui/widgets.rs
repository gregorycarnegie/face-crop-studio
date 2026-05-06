//! Custom widgets matching the HTML mockup design language.

use crate::theme::P;
use egui::{Color32, Response, Sense, Stroke, StrokeKind, Ui, Vec2};

// ── Themed slider ─────────────────────────────────────────────────────────────

const LABEL_W: f32 = 50.0;

/// Draw a slider with the peach-thumb dark-track style and return changed flag.
pub fn themed_slider(ui: &mut Ui, value: &mut f32, min: f32, max: f32) -> bool {
    ui.add(egui::Slider::new(value, min..=max).show_value(false)).changed()
}

/// Slider with inline value label on the right.
pub fn slider_with_label(ui: &mut Ui, _label: &str, value: &mut f32, min: f32, max: f32, fmt: &str) -> bool {
    ui.horizontal(|ui| {
        let slider_w = (ui.available_width() - LABEL_W).max(60.0);
        let changed = ui.add_sized(
            [slider_w, 20.0],
            egui::Slider::new(value, min..=max).show_value(false),
        ).changed();
        let display = match fmt {
            "pct"  => format!("{:.0}%", value),
            "conf" => format!("{:.2}", value),
            "deg"  => format!("{:.0}°", value),
            "px"   => format!("{:.0}px", value),
            _      => format!("{value:.1}"),
        };
        ui.monospace(egui::RichText::new(display).color(P::PEACH).size(11.5));
        changed
    }).inner
}

// ── Segmented control ─────────────────────────────────────────────────────────

pub fn segmented_control(ui: &mut Ui, options: &[&str], selected: &mut usize) -> bool {
    let mut changed = false;
    let n = options.len();
    let total_w = ui.available_width();
    let btn_w = total_w / n as f32;

    let bg = P::black_alpha(76);
    let outer = ui.allocate_rect(
        egui::Rect::from_min_size(ui.cursor().min, Vec2::new(total_w, 28.0)),
        Sense::hover(),
    );
    let painter = ui.painter();
    painter.rect_filled(outer.rect, 7.0, bg);
    painter.rect_stroke(outer.rect, 7.0, Stroke::new(1.0, P::RULE), StrokeKind::Outside);

    ui.horizontal(|ui| {
        for (i, &label) in options.iter().enumerate() {
            let btn_rect = egui::Rect::from_min_size(
                outer.rect.min + Vec2::new(i as f32 * btn_w + 2.0, 2.0),
                Vec2::new(btn_w - 4.0, 24.0),
            );
            let resp = ui.allocate_rect(btn_rect, Sense::click());
            let is_on = i == *selected;
            let bg_fill = if is_on { P::PEACH } else if resp.hovered() { P::white_alpha(10) } else { Color32::TRANSPARENT };
            let text_color = if is_on { P::BG } else if resp.hovered() { P::INK } else { P::INK2 };
            let painter = ui.painter();
            if bg_fill != Color32::TRANSPARENT {
                painter.rect_filled(btn_rect, 5.0, bg_fill);
            }
            painter.text(
                btn_rect.center(),
                egui::Align2::CENTER_CENTER,
                label,
                egui::FontId::monospace(11.0),
                text_color,
            );
            if resp.clicked() { *selected = i; changed = true; }
        }
    });
    changed
}

// ── Toggle switch ─────────────────────────────────────────────────────────────

/// Returns (response, changed).
pub fn toggle_switch(ui: &mut Ui, on: &mut bool) -> (Response, bool) {
    let (resp, painter) = ui.allocate_painter(Vec2::new(30.0, 18.0), Sense::click());
    let rect = resp.rect;
    let bg = if *on { P::cyan_alpha(64) } else { P::RULE };
    painter.rect_filled(rect, 9.0, bg);
    painter.rect_stroke(rect, 9.0, Stroke::new(1.0, if *on { P::cyan_alpha(120) } else { P::RULE2 }), StrokeKind::Outside);
    let cx = if *on { rect.max.x - 9.0 } else { rect.min.x + 9.0 };
    let thumb_color = if *on { P::CYAN } else { P::INK3 };
    painter.circle_filled(egui::pos2(cx, rect.center().y), 5.5, thumb_color);
    if *on {
        painter.circle_stroke(egui::pos2(cx, rect.center().y), 5.5, Stroke::new(0.5, P::CYAN));
    }
    let changed = resp.clicked();
    if changed { *on = !*on; }
    (resp, changed)
}

/// Toggle row: label on left, switch on right.
pub fn toggle_row(ui: &mut Ui, label: &str, on: &mut bool) -> bool {
    ui.horizontal(|ui| {
        ui.set_min_height(30.0);
        ui.label(egui::RichText::new(label).size(12.5));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let (_, changed) = toggle_switch(ui, on);
            changed
        }).inner
    }).inner
}

// ── Color swatch + hex input ──────────────────────────────────────────────────

pub fn color_swatch_input(ui: &mut Ui, hex: &mut String, color: &mut [u8; 3]) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        let (resp, painter) = ui.allocate_painter(Vec2::splat(28.0), Sense::click());
        let fill = Color32::from_rgb(color[0], color[1], color[2]);
        painter.rect_filled(resp.rect, 6.0, fill);
        painter.rect_stroke(resp.rect, 6.0, Stroke::new(1.0, P::RULE), egui::StrokeKind::Outside);

        let text_resp = ui.add(
            egui::TextEdit::singleline(hex)
                .desired_width(ui.available_width())
                .font(egui::FontId::monospace(11.5)),
        );
        if text_resp.changed() {
            if let Some(c) = parse_hex_color(hex) {
                *color = c;
                changed = true;
            }
        }
    });
    changed
}

fn parse_hex_color(hex: &str) -> Option<[u8; 3]> {
    let s = hex.trim_start_matches('#');
    if s.len() == 6 {
        let r = u8::from_str_radix(&s[0..2], 16).ok()?;
        let g = u8::from_str_radix(&s[2..4], 16).ok()?;
        let b = u8::from_str_radix(&s[4..6], 16).ok()?;
        Some([r, g, b])
    } else { None }
}

// ── Panel header ──────────────────────────────────────────────────────────────

/// Collapsible panel header.  Returns whether it was clicked to toggle.
pub fn panel_header(ui: &mut Ui, num: &str, title: &str, open: bool) -> bool {
    let resp = ui.allocate_response(Vec2::new(ui.available_width(), 36.0), Sense::click());
    let rect = resp.rect;
    let painter = ui.painter();
    if resp.hovered() {
        painter.rect_filled(rect, 0.0, P::white_alpha(4));
    }
    // Number badge
    let num_rect = egui::Rect::from_min_size(rect.min + Vec2::new(14.0, 10.0), Vec2::new(28.0, 16.0));
    painter.rect_stroke(num_rect, 4.0, Stroke::new(1.0, P::RULE2), egui::StrokeKind::Outside);
    painter.text(
        num_rect.center(),
        egui::Align2::CENTER_CENTER,
        num,
        egui::FontId::monospace(9.5),
        P::INK3,
    );
    // Title
    painter.text(
        egui::pos2(rect.min.x + 50.0, rect.center().y),
        egui::Align2::LEFT_CENTER,
        title,
        egui::FontId::proportional(12.5),
        P::INK,
    );
    // Chevron
    let chev = if open { "▾" } else { "▸" };
    let chev_color = if open { P::PEACH } else { P::INK3 };
    painter.text(
        egui::pos2(rect.max.x - 16.0, rect.center().y),
        egui::Align2::CENTER_CENTER,
        chev,
        egui::FontId::proportional(10.0),
        chev_color,
    );
    // Separator
    painter.line_segment(
        [egui::pos2(rect.min.x, rect.max.y), egui::pos2(rect.max.x, rect.max.y)],
        Stroke::new(1.0, P::RULE),
    );
    resp.clicked()
}

// ── Status badge ──────────────────────────────────────────────────────────────

pub fn badge(ui: &mut Ui, label: &str) {
    let (bg, fg) = crate::theme::badge_color(label);
    let display = if label.chars().all(|c| c.is_ascii_digit()) {
        label.to_string()
    } else {
        label.to_string()
    };
    let font = egui::FontId::monospace(9.5);
    let galley = ui.painter().layout_no_wrap(display.clone(), font.clone(), fg);
    let pad = Vec2::new(5.0, 1.5);
    let (resp, painter) = ui.allocate_painter(galley.size() + pad * 2.0, Sense::hover());
    painter.rect_filled(resp.rect, 3.0, bg);
    painter.galley(resp.rect.min + pad, galley, fg);
}

// ── Face chip ─────────────────────────────────────────────────────────────────

pub fn face_chip(ui: &mut Ui, label: &str, selected: bool, alt: bool) -> Response {
    let (bg, border, check_bg, text) = if !selected {
        (P::white_alpha(8), P::RULE, Color32::TRANSPARENT, P::INK3)
    } else if alt {
        (P::cyan_alpha(25), P::cyan_alpha(76), P::CYAN, P::CYAN)
    } else {
        (P::peach_alpha(25), P::peach_alpha(76), P::PEACH, P::PEACH)
    };

    let font = egui::FontId::monospace(10.5);
    let galley = ui.painter().layout_no_wrap(label.to_string(), font, text);
    let check_size = Vec2::splat(12.0);
    let total_w = check_size.x + 6.0 + galley.size().x + 20.0;
    let total_h = 24.0_f32.max(galley.size().y + 8.0);

    let (resp, painter) = ui.allocate_painter(Vec2::new(total_w, total_h), Sense::click());
    let r = resp.rect;
    painter.rect_filled(r, 12.0, bg);
    painter.rect_stroke(r, 12.0, Stroke::new(1.0, border), egui::StrokeKind::Outside);

    // Check square
    let check_rect = egui::Rect::from_min_size(r.min + Vec2::new(8.0, (total_h - 12.0) / 2.0), check_size);
    painter.rect_filled(check_rect, 3.0, check_bg);
    painter.rect_stroke(check_rect, 3.0, Stroke::new(0.5, if selected { text } else { P::INK3 }), egui::StrokeKind::Outside);
    if selected {
        painter.text(check_rect.center(), egui::Align2::CENTER_CENTER, "✓", egui::FontId::proportional(9.0), P::BG);
    }
    // Label
    painter.galley(r.min + Vec2::new(22.0, (total_h - galley.size().y) / 2.0), galley, text);
    resp
}

// ── GPU pill ──────────────────────────────────────────────────────────────────

pub fn gpu_pill(ui: &mut Ui, label: &str) {
    let font = egui::FontId::monospace(10.5);
    let galley = ui.painter().layout_no_wrap(label.to_string(), font, P::LIME);
    let total_w = 6.0 + 8.0 + galley.size().x + 22.0;
    let total_h = 24.0_f32.max(galley.size().y + 12.0);
    let (resp, painter) = ui.allocate_painter(Vec2::new(total_w, total_h), Sense::hover());
    let r = resp.rect;
    painter.rect_filled(r, 12.0, P::lime_alpha(20));
    painter.rect_stroke(r, 12.0, Stroke::new(1.0, P::lime_alpha(76)), egui::StrokeKind::Outside);
    let cx = r.min + Vec2::new(14.0, total_h / 2.0);
    painter.circle_filled(egui::pos2(cx.x, cx.y), 3.0, P::LIME);
    painter.galley(r.min + Vec2::new(22.0, (total_h - galley.size().y) / 2.0), galley, P::LIME);
}

// ── Field label ───────────────────────────────────────────────────────────────

pub fn field_label(ui: &mut Ui, text: &str) {
    ui.add_space(2.0);
    ui.label(egui::RichText::new(text).size(10.0).color(P::INK3).family(egui::FontFamily::Monospace));
    ui.add_space(2.0);
}

// ── Separators ────────────────────────────────────────────────────────────────

pub fn tb_sep(ui: &mut Ui) {
    let (resp, painter) = ui.allocate_painter(Vec2::new(1.0, 24.0), Sense::hover());
    painter.line_segment(
        [resp.rect.center_top(), resp.rect.center_bottom()],
        Stroke::new(1.0, P::RULE),
    );
}

// ── Labelled ctrl pill ────────────────────────────────────────────────────────

pub fn ctl_pill(ui: &mut Ui, key: &str, val: &str, accent: Option<Color32>) {
    let key_color = P::INK3;
    let val_color = accent.unwrap_or(P::INK);
    let border    = accent.map(|c| Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), 100)).unwrap_or(P::RULE);
    let bg        = P::white_alpha(10);

    let key_font = egui::FontId::monospace(10.5);
    let val_font = egui::FontId::monospace(10.5);
    let key_g = ui.painter().layout_no_wrap(key.to_string(), key_font, key_color);
    let val_g = ui.painter().layout_no_wrap(val.to_string(), val_font, val_color);
    let key_w = key_g.size().x;
    let w = key_w + val_g.size().x + 20.0;
    let h = 22.0_f32.max(key_g.size().y + 8.0);
    let (resp, painter) = ui.allocate_painter(Vec2::new(w, h), Sense::hover());
    let r = resp.rect;
    painter.rect_filled(r, 6.0, bg);
    painter.rect_stroke(r, 6.0, Stroke::new(1.0, border), egui::StrokeKind::Outside);
    let y = r.min.y + (h - key_g.size().y) / 2.0;
    painter.galley(egui::pos2(r.min.x + 6.0, y), key_g, key_color);
    painter.galley(egui::pos2(r.min.x + 6.0 + key_w + 4.0, y), val_g, val_color);
}
