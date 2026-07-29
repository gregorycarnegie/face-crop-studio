//! Custom widgets matching the HTML mockup design language.

use crate::theme::P;
use egui::{Color32, CursorIcon, Response, Sense, Stroke, StrokeKind, Ui, Vec2};

const LABEL_W: f32 = 50.0;

/// Slider with inline value label on the right.
pub fn slider_with_label(
    ui: &mut Ui,
    _label: &str,
    value: &mut f32,
    min: f32,
    max: f32,
    fmt: &str,
) -> bool {
    ui.horizontal(|ui| {
        let slider_w = (ui.available_width() - LABEL_W - ui.spacing().item_spacing.x).max(60.0);
        let changed = ui
            .add_sized(
                [slider_w, 20.0],
                egui::Slider::new(value, min..=max).show_value(false),
            )
            .changed();
        let display = match fmt {
            "pct" => format!("{:.0}%", value),
            "conf" => format!("{:.2}", value),
            "deg" => format!("{:.0}°", value),
            "px" => format!("{:.0}px", value),
            _ => format!("{value:.1}"),
        };
        ui.monospace(egui::RichText::new(display).color(P::PEACH).size(11.5));
        changed
    })
    .inner
}

// ── Segmented control ─────────────────────────────────────────────────────────

pub fn segmented_control(ui: &mut Ui, options: &[&str], selected: &mut usize) -> bool {
    let mut changed = false;
    let n = options.len();
    let total_w = ui.available_width();
    let btn_w = total_w / n as f32;

    // Allocate the entire control as one painter — this correctly advances the
    // cursor and gives us a stable ID.  Per-button interaction is registered
    // via ui.interact() which does NOT allocate extra space.
    let (outer_resp, painter) = ui.allocate_painter(Vec2::new(total_w, 28.0), Sense::hover());
    let outer_rect = outer_resp.rect;

    painter.rect_filled(outer_rect, 7.0, P::black_alpha(76));
    painter.rect_stroke(
        outer_rect,
        7.0,
        Stroke::new(1.0, P::RULE),
        StrokeKind::Outside,
    );

    // Sliding selection pill — animates between segments.
    let anim_i =
        ui.ctx()
            .animate_value_with_time(outer_resp.id.with("sel"), *selected as f32, 0.15);
    let pill_rect = egui::Rect::from_min_size(
        outer_rect.min + Vec2::new(anim_i * btn_w + 2.0, 2.0),
        Vec2::new(btn_w - 4.0, 24.0),
    );
    painter.rect_filled(pill_rect, 5.0, P::PEACH);

    for (i, &label) in options.iter().enumerate() {
        let btn_rect = egui::Rect::from_min_size(
            outer_rect.min + Vec2::new(i as f32 * btn_w + 2.0, 2.0),
            Vec2::new(btn_w - 4.0, 24.0),
        );
        let btn_id = outer_resp.id.with(i);
        let resp = ui
            .interact(btn_rect, btn_id, Sense::click())
            .on_hover_cursor(CursorIcon::PointingHand);
        let is_on = i == *selected;
        if !is_on && resp.hovered() {
            painter.rect_filled(btn_rect, 5.0, P::white_alpha(10));
        }
        let text_color = if is_on {
            P::BG
        } else if resp.hovered() {
            P::INK
        } else {
            P::INK2
        };
        painter.text(
            btn_rect.center(),
            egui::Align2::CENTER_CENTER,
            label,
            egui::FontId::monospace(11.0),
            text_color,
        );
        if resp.clicked() {
            *selected = i;
            changed = true;
        }
    }
    changed
}

// ── Toggle switch ─────────────────────────────────────────────────────────────

/// Returns (response, changed).
pub fn toggle_switch(ui: &mut Ui, on: &mut bool) -> (Response, bool) {
    let (resp, painter) = ui.allocate_painter(Vec2::new(30.0, 18.0), Sense::click());
    let resp = resp.on_hover_cursor(CursorIcon::PointingHand);
    let changed = resp.clicked();
    if changed {
        *on = !*on;
    }
    // Animate thumb position and colors between off/on.
    let t = ui.ctx().animate_bool(resp.id, *on);
    let rect = resp.rect;
    painter.rect_filled(rect, 9.0, P::RULE.lerp_to_gamma(P::cyan_alpha(64), t));
    painter.rect_stroke(
        rect,
        9.0,
        Stroke::new(1.0, P::RULE2.lerp_to_gamma(P::cyan_alpha(120), t)),
        StrokeKind::Outside,
    );
    let cx = egui::lerp((rect.min.x + 9.0)..=(rect.max.x - 9.0), t);
    let thumb_color = P::INK3.lerp_to_gamma(P::CYAN, t);
    painter.circle_filled(egui::pos2(cx, rect.center().y), 5.5, thumb_color);
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
        })
        .inner
    })
    .inner
}

// ── Panel header ──────────────────────────────────────────────────────────────

/// Collapsible panel header.  Returns whether it was clicked to toggle.
pub fn panel_header(ui: &mut Ui, num: &str, title: &str, open: bool) -> bool {
    let resp = ui
        .allocate_response(Vec2::new(ui.available_width(), 36.0), Sense::click())
        .on_hover_cursor(CursorIcon::PointingHand);
    let rect = resp.rect;
    let painter = ui.painter();
    if resp.hovered() {
        painter.rect_filled(rect, 0.0, P::white_alpha(4));
    }
    // Peach accent bar when open
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
        [
            egui::pos2(rect.min.x, rect.max.y),
            egui::pos2(rect.max.x, rect.max.y),
        ],
        Stroke::new(1.0, P::RULE),
    );
    resp.clicked()
}

// ── Face chip ─────────────────────────────────────────────────────────────────

pub fn face_chip(ui: &mut Ui, label: String, selected: bool, alt: bool) -> Response {
    let (mut bg, mut border, check_bg, text) = if !selected {
        (P::white_alpha(8), P::RULE, Color32::TRANSPARENT, P::INK3)
    } else if alt {
        (P::cyan_alpha(25), P::cyan_alpha(76), P::CYAN, P::CYAN)
    } else {
        (P::peach_alpha(25), P::peach_alpha(76), P::PEACH, P::PEACH)
    };

    let font = egui::FontId::monospace(10.5);
    let galley = ui.painter().layout_no_wrap(label, font, text);
    let check_size = Vec2::splat(12.0);
    let total_w = check_size.x + 6.0 + galley.size().x + 20.0;
    let total_h = 24.0_f32.max(galley.size().y + 8.0);

    let (resp, painter) = ui.allocate_painter(Vec2::new(total_w, total_h), Sense::click());
    let resp = resp.on_hover_cursor(CursorIcon::PointingHand);
    if resp.hovered() && !selected {
        bg = P::white_alpha(16);
        border = P::RULE2;
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
            "✓",
            egui::FontId::proportional(9.0),
            P::BG,
        );
    }
    // Label
    painter.galley(
        r.min + Vec2::new(22.0, (total_h - galley.size().y) / 2.0),
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
    ui.label(
        egui::RichText::new(text)
            .size(10.0)
            .color(P::INK3)
            .family(egui::FontFamily::Monospace),
    );
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

    /// A fixed viewport keeps the width-derived geometry deterministic.
    fn harness<'a>(app: impl FnMut(&mut Ui, &mut Probe) + 'a) -> Harness<'a, Probe> {
        Harness::builder()
            .with_size(egui::vec2(240.0, 120.0))
            .build_ui_state(app, Probe::default())
    }

    /// egui needs the pointer over the target on the frame the press lands, so
    /// hover, press, and release each get their own frame.
    fn click_at(h: &mut Harness<'_, Probe>, pos: egui::Pos2) {
        h.hover_at(pos);
        h.run();
        h.drag_at(pos);
        h.run();
        h.drop_at(pos);
        h.run();
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
    fn toggle_switch_flips_on_each_click() {
        let mut h = harness(|ui, probe| {
            probe.origin = ui.cursor().min;
            let (_, changed) = toggle_switch(ui, &mut probe.on);
            if changed {
                probe.clicked = true;
            }
        });
        h.run();

        // The switch allocates a fixed 30x18 box at the cursor.
        let centre = h.state().origin + egui::vec2(15.0, 9.0);
        assert!(!h.state().on);

        click_at(&mut h, centre);
        assert!(h.state().on, "first click turns it on");
        assert!(h.state().clicked, "and reports the change");

        click_at(&mut h, centre);
        assert!(!h.state().on, "second click turns it back off");
    }

    #[test]
    fn toggle_switch_ignores_clicks_well_outside_its_box() {
        let mut h = harness(|ui, probe| {
            probe.origin = ui.cursor().min;
            toggle_switch(ui, &mut probe.on);
        });
        h.run();

        // Clear of the 30x18 box and of egui's interact radius.
        let outside = h.state().origin + egui::vec2(80.0, 9.0);
        click_at(&mut h, outside);
        assert!(!h.state().on);
    }

    #[test]
    fn toggle_row_wires_the_label_and_the_switch_together() {
        let mut h = harness(|ui, probe| {
            probe.width = ui.available_width();
            probe.origin = ui.cursor().min;
            if toggle_row(ui, "AUTO COLOR", &mut probe.on) {
                probe.clicked = true;
            }
        });
        h.run();

        // The label is a real widget, so it appears in the accessibility tree.
        h.get_by_label("AUTO COLOR");

        // The switch is right-aligned within the row, which is 30px tall.
        let (origin, width) = (h.state().origin, h.state().width);
        click_at(&mut h, origin + egui::vec2(width - 15.0, 15.0));
        assert!(
            h.state().on,
            "clicking the right-hand switch toggles the row"
        );
        assert!(h.state().clicked);
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
