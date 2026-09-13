//! Custom widgets matching the HTML mockup design language.

use crate::theme::P;
use egui::{Color32, CursorIcon, Response, Sense, Stroke, Ui, Vec2};

const LABEL_W: f32 = 50.0;

/// Slider with inline value label on the right.
pub fn slider_with_label(
    ui: &mut Ui,
    label: &str,
    value: &mut f32,
    min: f32,
    max: f32,
    fmt: &str,
) -> bool {
    ui.horizontal(|ui| {
        let slider_w = (ui.available_width() - LABEL_W - ui.spacing().item_spacing.x).max(60.0);
        ui.spacing_mut().slider_width = slider_w;
        // The rail and thumb need their own contrast against the dark panel.
        ui.visuals_mut().widgets.inactive.bg_fill = P::RULE2;
        ui.visuals_mut().widgets.inactive.fg_stroke = Stroke::new(2.0, P::INK);
        let response = ui.add_sized(
            [slider_w, 20.0],
            egui::Slider::new(value, min..=max).show_value(false),
        );
        response.widget_info(|| egui::WidgetInfo::slider(ui.is_enabled(), *value as f64, label));
        let changed = response.changed();
        let display = match fmt {
            "pct" => format!("{:.0}%", value),
            "conf" => format!("{:.2}", value),
            "deg" => format!("{:.0}°", value),
            "int" => format!("{value:.0}"),
            "px" => format!("{:.0}px", value),
            _ => format!("{value:.1}"),
        };
        ui.monospace(egui::RichText::new(display).color(P::INK).size(12.0));
        changed
    })
    .inner
}

// ── Segmented control ─────────────────────────────────────────────────────────

pub fn segmented_control(ui: &mut Ui, options: &[&str], selected: &mut usize) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        let w = (ui.available_width() - 4.0 * options.len().saturating_sub(1) as f32)
            / options.len().max(1) as f32;
        for (i, label) in options.iter().enumerate() {
            let active = *selected == i;
            let text =
                egui::RichText::new(*label)
                    .size(12.0)
                    .color(if active { P::BG } else { P::INK2 });
            if ui
                .add_sized([w, 30.0], egui::Button::new(text).selected(active))
                .clicked()
            {
                changed = *selected != i;
                *selected = i;
            }
        }
    });
    changed
}

/// Native checkbox: a visible checkmark, a clickable label and keyboard support.
pub fn toggle_row(ui: &mut Ui, label: &str, on: &mut bool) -> bool {
    ui.horizontal(|ui| {
        ui.set_min_height(30.0);
        let widgets = &mut ui.visuals_mut().widgets;
        for widget in [
            &mut widgets.inactive,
            &mut widgets.hovered,
            &mut widgets.active,
        ] {
            widget.corner_radius = egui::CornerRadius::same(2);
        }
        ui.checkbox(on, egui::RichText::new(label).size(13.0))
            .changed()
    })
    .inner
}

// ── Panel header ──────────────────────────────────────────────────────────────

/// Collapsible panel header.  Returns whether it was clicked to toggle.
pub fn panel_header(ui: &mut Ui, num: &str, title: &str, open: bool) -> bool {
    let resp = ui
        .allocate_response(Vec2::new(ui.available_width(), 36.0), Sense::click())
        .on_hover_cursor(CursorIcon::PointingHand);
    resp.widget_info(|| {
        egui::WidgetInfo::selected(
            egui::WidgetType::CollapsingHeader,
            ui.is_enabled(),
            open,
            title,
        )
    });
    let rect = resp.rect;
    let painter = ui.painter();
    if resp.hovered() || resp.has_focus() {
        painter.rect_filled(rect, 0.0, P::white_alpha(4));
    }
    if resp.has_focus() {
        painter.rect_stroke(
            rect.shrink(2.0),
            4.0,
            Stroke::new(2.0, P::CYAN),
            egui::StrokeKind::Inside,
        );
    }
    // Accent bar when open
    let t = ui.ctx().animate_bool(resp.id.with("open"), open);
    if t > 0.0 {
        let bar = egui::Rect::from_min_size(
            rect.min + Vec2::new(0.0, 9.0),
            Vec2::new(3.0, (rect.height() - 18.0) * t),
        );
        painter.rect_filled(bar, 2.0, P::peach_alpha((230.0 * t) as u8));
    }
    // Number badge
    let num_rect =
        egui::Rect::from_min_size(rect.min + Vec2::new(14.0, 10.0), Vec2::new(28.0, 16.0));
    painter.rect_stroke(
        num_rect,
        4.0,
        Stroke::new(1.0, P::RULE2),
        egui::StrokeKind::Outside,
    );
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
        egui::FontId::proportional(14.0),
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
        [
            egui::pos2(rect.min.x, rect.max.y),
            egui::pos2(rect.max.x, rect.max.y),
        ],
        Stroke::new(1.0, P::RULE),
    );
    resp.clicked()
}

// ── Face chip ─────────────────────────────────────────────────────────────────

pub fn face_chip(ui: &mut Ui, label: String, selected: bool, _alt: bool) -> Response {
    let (mut bg, mut border, check_bg, text) = if !selected {
        (P::white_alpha(8), P::RULE, Color32::TRANSPARENT, P::INK3)
    } else {
        (P::peach_alpha(25), P::peach_alpha(76), P::PEACH, P::PEACH)
    };

    let font = egui::FontId::monospace(10.5);
    let galley = ui.painter().layout_no_wrap(label.clone(), font, text);
    let check_size = Vec2::splat(12.0);
    let total_w = check_size.x + 6.0 + galley.size().x + 20.0;
    let total_h = 24.0_f32.max(galley.size().y + 8.0);

    let (resp, painter) = ui.allocate_painter(Vec2::new(total_w, total_h), Sense::click());
    let resp = resp.on_hover_cursor(CursorIcon::PointingHand);
    resp.widget_info(|| {
        egui::WidgetInfo::selected(
            egui::WidgetType::Checkbox,
            ui.is_enabled(),
            selected,
            &label,
        )
    });
    if (resp.hovered() || resp.has_focus()) && !selected {
        bg = P::white_alpha(16);
        border = P::RULE2;
    }
    if resp.has_focus() {
        border = P::CYAN;
    }
    let r = resp.rect;
    painter.rect_filled(r, 12.0, bg);
    painter.rect_stroke(r, 12.0, Stroke::new(1.0, border), egui::StrokeKind::Outside);

    // Check square
    let check_rect =
        egui::Rect::from_min_size(r.min + Vec2::new(8.0, (total_h - 12.0) / 2.0), check_size);
    painter.rect_filled(check_rect, 3.0, check_bg);
    painter.rect_stroke(
        check_rect,
        3.0,
        Stroke::new(0.5, if selected { text } else { P::INK3 }),
        egui::StrokeKind::Outside,
    );
    if selected {
        painter.text(
            check_rect.center(),
            egui::Align2::CENTER_CENTER,
            "✔",
            egui::FontId::proportional(9.0),
            P::BG,
        );
    }
    // Label
    painter.galley(
        r.min + Vec2::new(26.0, (total_h - galley.size().y) / 2.0),
        galley,
        text,
    );
    resp
}

// ── GPU pill ──────────────────────────────────────────────────────────────────

pub fn gpu_pill(ui: &mut Ui, label: &str) {
    let font = egui::FontId::monospace(10.5);
    let galley = ui
        .painter()
        .layout_no_wrap(label.to_string(), font, P::LIME);
    let total_w = 6.0 + 8.0 + galley.size().x + 22.0;
    let total_h = 24.0_f32.max(galley.size().y + 12.0);
    let (resp, painter) = ui.allocate_painter(Vec2::new(total_w, total_h), Sense::hover());
    let r = resp.rect;
    painter.rect_filled(r, 12.0, P::lime_alpha(20));
    painter.rect_stroke(
        r,
        12.0,
        Stroke::new(1.0, P::lime_alpha(76)),
        egui::StrokeKind::Outside,
    );
    let cx = r.min + Vec2::new(14.0, total_h / 2.0);
    painter.circle_filled(egui::pos2(cx.x, cx.y), 3.0, P::LIME);
    painter.galley(
        r.min + Vec2::new(22.0, (total_h - galley.size().y) / 2.0),
        galley,
        P::LIME,
    );
}

// ── Field label ───────────────────────────────────────────────────────────────

pub fn field_label(ui: &mut Ui, text: &str) {
    ui.add_space(2.0);
    ui.label(egui::RichText::new(text).size(12.0).color(P::INK2));
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
    let border = accent
        .map(|c| Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), 100))
        .unwrap_or(P::RULE);
    let bg = P::white_alpha(10);

    let key_font = egui::FontId::monospace(10.5);
    let val_font = egui::FontId::monospace(10.5);
    let key_g = ui
        .painter()
        .layout_no_wrap(key.to_string(), key_font, key_color);
    let val_g = ui
        .painter()
        .layout_no_wrap(val.to_string(), val_font, val_color);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::test_support::click_at;
    use egui_kittest::{Harness, kittest::Queryable};

    /// Captured layout inputs plus whatever the widget under test mutates.
    ///
    /// Click positions are computed in the test from `origin` and `width` —
    /// plain inputs the widget did not derive. The widget derives its own hit
    /// rectangles from those same inputs, so if that geometry is wrong the
    /// click lands elsewhere and the assertion fails, which is the point.
    /// Recomputing positions from the widget's own output would track any
    /// error and assert nothing.
    #[derive(Default)]
    struct Probe {
        origin: egui::Pos2,
        width: f32,
        value: f32,
        selected: usize,
        on: bool,
        clicked: bool,
    }

    fn harness<'a>(app: impl FnMut(&mut Ui, &mut Probe) + 'a) -> Harness<'a, Probe> {
        crate::ui::test_support::harness(Probe::default(), app)
    }

    #[test]
    fn segmented_control_selects_the_segment_under_the_cursor() {
        let options = ["ONE", "TWO", "THREE"];
        let mut h = harness(move |ui, probe| {
            probe.origin = ui.cursor().min;
            probe.width = ui.available_width();
            segmented_control(ui, &options, &mut probe.selected);
        });
        h.run();

        let (origin, width) = (h.state().origin, h.state().width);
        assert!(width > 0.0, "harness gave the control no width");
        // Segment i is inset 2px inside a btn_w-wide cell, and 28px tall.
        let btn_w = width / 3.0;
        let centre = |i: usize| origin + egui::vec2(i as f32 * btn_w + btn_w * 0.5, 14.0);

        assert_eq!(h.state().selected, 0, "starts on the first segment");

        click_at(&mut h, centre(2));
        assert_eq!(h.state().selected, 2, "third segment");

        click_at(&mut h, centre(1));
        assert_eq!(h.state().selected, 1, "second segment");

        click_at(&mut h, centre(0));
        assert_eq!(h.state().selected, 0, "back to the first");
    }

    #[test]
    fn segmented_control_ignores_clicks_below_its_row() {
        // The control allocates 28px of height; a click well past that belongs
        // to whatever comes next, not to a segment.
        let options = ["ONE", "TWO"];
        let mut h = harness(move |ui, probe| {
            probe.origin = ui.cursor().min;
            probe.width = ui.available_width();
            segmented_control(ui, &options, &mut probe.selected);
        });
        h.run();

        let (origin, width) = (h.state().origin, h.state().width);
        click_at(&mut h, origin + egui::vec2(width * 0.75, 60.0));
        assert_eq!(
            h.state().selected,
            0,
            "a click below the row selects nothing"
        );
    }

    #[test]
    fn checkbox_label_and_keyboard_toggle_the_value() {
        let mut h = harness(|ui, probe| {
            if toggle_row(ui, "Auto color", &mut probe.on) {
                probe.clicked = true;
            }
        });
        h.run();
        h.get_by_label("Auto color").click();
        h.run();
        assert!(h.state().on && h.state().clicked);
        h.key_press(egui::Key::Tab);
        h.run();
        h.key_press(egui::Key::Space);
        h.run();
        assert!(!h.state().on);
    }

    #[test]
    fn panel_header_reports_a_click_anywhere_on_its_row() {
        let mut h = harness(|ui, probe| {
            probe.origin = ui.cursor().min;
            probe.width = ui.available_width();
            if panel_header(ui, "01", "SOURCE", true) {
                probe.clicked = true;
            }
        });
        h.run();
        assert!(!h.state().clicked, "no click yet");

        // A full-width, 36px-tall row.
        let pos = h.state().origin + egui::vec2(h.state().width * 0.5, 18.0);
        click_at(&mut h, pos);
        assert!(h.state().clicked);
    }

    #[test]
    fn slider_label_formats_by_kind() {
        // Each `fmt` key renders the value differently, and the label text is
        // the only externally visible difference between the match arms.
        for (fmt, value, expected) in [
            ("pct", 42.0_f32, "42%"),
            ("conf", 0.5, "0.50"),
            ("deg", 90.0, "90\u{b0}"),
            ("px", 12.0, "12px"),
            ("", 3.25, "3.2"),
        ] {
            let mut h = harness(move |ui, probe| {
                probe.value = value;
                slider_with_label(ui, "L", &mut probe.value, 0.0, 100.0, fmt);
            });
            h.run();
            // Panics listing the available labels if the text is absent.
            h.get_by_label(expected);
        }
    }

    #[test]
    fn face_chip_sizes_itself_around_its_label() {
        let sizes = std::cell::RefCell::new(Vec::new());
        {
            let sizes = &sizes;
            let mut h = harness(move |ui, _| {
                let narrow = face_chip(ui, "A".to_string(), true, false).rect;
                let wide = face_chip(ui, "AAAAAAAAAA".to_string(), false, true).rect;
                sizes
                    .borrow_mut()
                    .push((narrow.width(), wide.width(), narrow.height()));
            });
            h.run();
        }

        let (narrow, wide, height) = sizes.borrow()[0];
        assert!(
            wide > narrow,
            "a longer label must widen the chip: {narrow} vs {wide}"
        );
        // 12px check square + 6px gap + 20px padding is fixed overhead.
        assert!(
            narrow >= 38.0,
            "chip narrower than its fixed padding: {narrow}"
        );
        assert!(
            height >= 24.0,
            "chip shorter than its 24px minimum: {height}"
        );
    }
}
