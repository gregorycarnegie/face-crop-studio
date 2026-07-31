//! Toolbar ribbon.

use crate::{
    theme::P,
    types::App2,
    ui::widgets::{gpu_pill, tb_sep},
};
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
        .show(ui, |ui| {
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
                icon_btn(ui, "📂", "Open", true, || {
                    if let Some(paths) = rfd::FileDialog::new()
                        .add_filter("Images", fcs_utils::SUPPORTED_IMAGE_EXTENSIONS)
                        .pick_files()
                    {
                        let first = paths.first().cloned();
                        app.enqueue_batch_paths(paths);
                        if let Some(path) = first {
                            app.load_image_path(path);
                        }
                    }
                });
                icon_btn(ui, "💾", "Save", true, || {
                    crate::core::export::export_selected_faces(app);
                });
                let can_undo = !app.undo_stack.is_empty();
                let can_redo = !app.redo_stack.is_empty();
                icon_btn(ui, "↩", "Undo (Ctrl+Z)", can_undo, || app.undo());
                icon_btn(ui, "↪", "Redo (Ctrl+Y)", can_redo, || app.redo());
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

                // Draw tool toggle
                if toggle_btn(ui, "Draw box", app.manual_box_tool_enabled) {
                    app.manual_box_tool_enabled = !app.manual_box_tool_enabled;
                    app.manual_box_draft = None;
                }
                // Remove selected (only enabled when something is selected)
                if !app.selected_faces.is_empty() && ghost_btn(ui, "Remove selected") {
                    app.delete_selected_faces();
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
                        .gpu
                        .status
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
    let resp = resp.on_hover_cursor(egui::CursorIcon::PointingHand);
    let r = resp.rect;
    let fill = if resp.is_pointer_button_down_on() {
        lighten(bg, -0.04)
    } else if resp.hovered() {
        lighten(bg, 0.08)
    } else {
        bg
    };
    painter.rect_filled(r, 7.0, fill);
    // Accent border makes the two primary actions read as primary.
    let border_a = if resp.hovered() { 130 } else { 70 };
    painter.rect_stroke(
        r,
        7.0,
        Stroke::new(
            1.0,
            Color32::from_rgba_unmultiplied(fg.r(), fg.g(), fg.b(), border_a),
        ),
        egui::StrokeKind::Outside,
    );
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
    let resp = resp.on_hover_cursor(egui::CursorIcon::PointingHand);
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

fn icon_btn(
    ui: &mut egui::Ui,
    icon: &str,
    tooltip: &str,
    enabled: bool,
    action: impl FnOnce(),
) -> bool {
    let (resp, painter) = ui.allocate_painter(Vec2::splat(34.0), Sense::click());
    let resp = if enabled {
        resp.on_hover_cursor(egui::CursorIcon::PointingHand)
    } else {
        resp
    };
    let r = resp.rect;
    if enabled && resp.hovered() {
        painter.rect_filled(r, 7.0, P::white_alpha(15));
    }
    painter.rect_stroke(
        r,
        7.0,
        Stroke::new(1.0, if enabled { P::RULE2 } else { P::RULE }),
        egui::StrokeKind::Outside,
    );
    painter.text(
        r.center(),
        egui::Align2::CENTER_CENTER,
        icon,
        egui::FontId::proportional(14.0),
        if enabled { P::INK2 } else { P::white_alpha(50) },
    );
    let clicked = enabled && resp.clicked();
    resp.on_hover_text(tooltip);
    if clicked {
        action();
    }
    clicked
}

fn toggle_btn(ui: &mut egui::Ui, label: &str, active: bool) -> bool {
    let font = egui::FontId::proportional(12.5);
    let color = if active { P::CYAN } else { P::INK };
    let galley = ui.painter().layout_no_wrap(label.to_string(), font, color);
    let w = galley.size().x + 26.0;
    let (resp, painter) = ui.allocate_painter(Vec2::new(w, 34.0), Sense::click());
    let resp = resp.on_hover_cursor(egui::CursorIcon::PointingHand);
    let r = resp.rect;
    if active {
        painter.rect_filled(r, 7.0, P::cyan_alpha(30));
        painter.rect_stroke(r, 7.0, Stroke::new(1.5, P::CYAN), egui::StrokeKind::Outside);
    } else {
        let bg = if resp.hovered() {
            P::white_alpha(15)
        } else {
            P::white_alpha(5)
        };
        painter.rect_filled(r, 7.0, bg);
        painter.rect_stroke(
            r,
            7.0,
            Stroke::new(1.0, P::RULE2),
            egui::StrokeKind::Outside,
        );
    }
    painter.galley(
        r.min + Vec2::new(13.0, (34.0 - galley.size().y) / 2.0),
        galley,
        color,
    );
    resp.clicked()
}

fn danger_btn(ui: &mut egui::Ui, label: &str, action: impl FnOnce()) -> bool {
    let font = egui::FontId::proportional(12.5);
    let galley = ui
        .painter()
        .layout_no_wrap(label.to_string(), font, P::ROSE);
    let w = galley.size().x + 26.0;
    let (resp, painter) = ui.allocate_painter(Vec2::new(w, 34.0), Sense::click());
    let resp = resp.on_hover_cursor(egui::CursorIcon::PointingHand);
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

// ponytail: assumes an opaque input. `Color32` stores premultiplied channels, so
// feeding this a translucent colour premultiplies a second time and darkens
// instead of lightening. Every caller passes `bg_from_*`/`P::` opaque constants;
// switch to `Color32::from_rgba_premultiplied` if that ever stops being true.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::test_support::{click_at, harness};
    use std::cell::Cell;
    use std::rc::Rc;

    /// Layout inputs the test needs plus what the button under test reported.
    ///
    /// `origin` is the cursor the button allocates from; click positions are
    /// derived from it and the button's own fixed 34px height, not from the
    /// response rect — reading the rect back would track any geometry error
    /// instead of catching it.
    #[derive(Default)]
    struct Probe {
        origin: egui::Pos2,
        clicks: usize,
    }

    /// Every text button is `label width + 26` wide and 34 tall, so a point 14px
    /// in and 17px down is inside any of them.
    const INSIDE: egui::Vec2 = egui::vec2(14.0, 17.0);
    /// Well past the 34px row, so it belongs to whatever is laid out next.
    const BELOW: egui::Vec2 = egui::vec2(14.0, 60.0);

    #[test]
    fn primary_btn_reports_clicks_on_its_row_only() {
        let mut h = harness(Probe::default(), |ui, probe| {
            probe.origin = ui.cursor().min;
            if primary_btn(ui, "EXPORT", P::INK, P::CYAN) {
                probe.clicks += 1;
            }
        });
        h.run();
        let origin = h.state().origin;

        click_at(&mut h, origin + INSIDE);
        assert_eq!(h.state().clicks, 1);

        click_at(&mut h, origin + BELOW);
        assert_eq!(h.state().clicks, 1, "a click below the row is not a press");
    }

    #[test]
    fn ghost_btn_reports_clicks_on_its_row_only() {
        let mut h = harness(Probe::default(), |ui, probe| {
            probe.origin = ui.cursor().min;
            if ghost_btn(ui, "CANCEL") {
                probe.clicks += 1;
            }
        });
        h.run();
        let origin = h.state().origin;

        click_at(&mut h, origin + INSIDE);
        assert_eq!(h.state().clicks, 1);

        click_at(&mut h, origin + BELOW);
        assert_eq!(h.state().clicks, 1);
    }

    #[test]
    fn toggle_btn_reports_clicks_in_both_states() {
        // `active` only changes how the button paints; it must not gate the
        // click, or a toggle could be switched on but never off.
        for active in [false, true] {
            let mut h = harness(Probe::default(), move |ui, probe| {
                probe.origin = ui.cursor().min;
                if toggle_btn(ui, "GRID", active) {
                    probe.clicks += 1;
                }
            });
            h.run();
            let origin = h.state().origin;

            click_at(&mut h, origin + INSIDE);
            assert_eq!(h.state().clicks, 1, "active = {active}");
        }
    }

    #[test]
    fn danger_btn_runs_its_action_once_per_click() {
        let fired = Rc::new(Cell::new(0usize));
        let counter = Rc::clone(&fired);
        let mut h = harness(Probe::default(), move |ui, probe| {
            probe.origin = ui.cursor().min;
            if danger_btn(ui, "CLEAR", || counter.set(counter.get() + 1)) {
                probe.clicks += 1;
            }
        });
        h.run();
        let origin = h.state().origin;

        assert_eq!(fired.get(), 0, "merely painting must not fire the action");

        click_at(&mut h, origin + INSIDE);
        assert_eq!(h.state().clicks, 1);
        assert_eq!(fired.get(), 1);

        click_at(&mut h, origin + INSIDE);
        assert_eq!(fired.get(), 2, "a second click fires again");
    }

    #[test]
    fn icon_btn_when_disabled_swallows_the_click_and_the_action() {
        // The disabled path is the one that matters: `show` wires undo/redo and
        // the destructive queue actions through it, so a disabled button that
        // still ran its action would fire an operation the app has said is
        // unavailable.
        let fired = Rc::new(Cell::new(0usize));

        for (enabled, expected) in [(false, 0), (true, 1)] {
            let counter = Rc::clone(&fired);
            counter.set(0);
            let mut h = harness(Probe::default(), move |ui, probe| {
                probe.origin = ui.cursor().min;
                if icon_btn(ui, "↶", "Undo", enabled, || {
                    counter.set(counter.get() + 1)
                }) {
                    probe.clicks += 1;
                }
            });
            h.run();
            let origin = h.state().origin;

            // icon_btn is a fixed 34x34 square.
            click_at(&mut h, origin + egui::vec2(17.0, 17.0));
            assert_eq!(h.state().clicks, expected, "enabled = {enabled}");
            assert_eq!(fired.get(), expected, "action, enabled = {enabled}");
        }
    }

    #[test]
    fn lighten_brightens_darkens_and_saturates_without_wrapping() {
        // Opaque, matching every real caller — see the note on `lighten`.
        let c = Color32::from_rgb(100, 150, 200);

        let up = lighten(c, 0.1);
        assert!(up.r() > c.r() && up.g() > c.g() && up.b() > c.b());

        // The two hover/press states in `primary_btn` are a +0.08 and a -0.04,
        // so the negative direction is not hypothetical.
        let down = lighten(c, -0.1);
        assert!(down.r() < c.r() && down.g() < c.g() && down.b() < c.b());

        // Both ends saturate. The `as u8` cast on a negative float is the only
        // thing standing between -25.5 and a wrapped-around bright colour.
        assert_eq!(lighten(c, 10.0), Color32::from_rgb(255, 255, 255));
        assert_eq!(lighten(c, -10.0), Color32::from_rgb(0, 0, 0));

        assert_eq!(lighten(c, 0.1).a(), 255, "alpha is preserved");
    }

    #[test]
    fn the_two_primary_button_backgrounds_are_distinct_and_dark() {
        // They tint the two primary actions apart; identical values would make
        // the toolbar's colour coding meaningless.
        let (cyan, peach) = (bg_from_cyan(), bg_from_peach());
        assert_ne!(cyan, peach);
        for c in [cyan, peach] {
            assert!(
                c.r() < 128 && c.g() < 128 && c.b() < 128,
                "{c:?} must stay dark enough for P::INK text to read on it",
            );
        }
    }
}
