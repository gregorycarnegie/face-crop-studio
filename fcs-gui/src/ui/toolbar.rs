//! Toolbar ribbon.

use crate::{
    theme::P,
    types::App2,
    ui::widgets::{gpu_pill, tb_sep},
};
use egui::{Color32, Frame, Ui, Vec2};
use fcs_utils::gpu::{GpuStatusIndicator, GpuStatusMode};

pub fn show(ui: &mut Ui, app: &mut App2) {
    egui::Panel::top("toolbar")
        .min_size(52.0)
        .show_separator_line(false)
        .frame(
            Frame::new()
                .fill(P::BG1)
                .inner_margin(egui::Margin::symmetric(12, 8)),
        )
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                // Primary action: Detect
                if primary_btn(ui, "Detect faces →", P::BG, P::ACCENT)
                    && let Some(path) = app.preview.image_path.clone()
                {
                    app.load_image_path(path);
                }
                ui.add_space(4.0);

                // Secondary action: Export
                if ghost_btn(ui, "Export crops") {
                    if app.selected_faces.is_empty() && !app.batch_files.is_empty() {
                        crate::core::export::start_batch_export(app);
                    } else {
                        crate::core::export::export_selected_faces(app);
                    }
                }
                tb_sep(ui);

                // Icon buttons
                icon_btn(ui, "Open…", "Open images", true, || {
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
                icon_btn(ui, "Save", "Save selected crops", true, || {
                    crate::core::export::export_selected_faces(app);
                });
                let can_undo = !app.undo_stack.is_empty();
                let can_redo = !app.redo_stack.is_empty();
                icon_btn(ui, "Undo", "Undo (Ctrl+Z)", can_undo, || app.undo());
                icon_btn(ui, "Redo", "Redo (Ctrl+Y)", can_redo, || app.redo());
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
                if ui.available_width() > 300.0 {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let backend = app.detector.as_deref().map(|d| d.inference_backend());
                        gpu_pill(ui, &gpu_label(&app.gpu.status, backend));
                    });
                }
            });
        });
}

/// Where detection really runs, for the toolbar pill and the status bar.
///
/// `status` only describes GPU preprocessing: with that off it carries no adapter even
/// while inference is on the GPU, and with the GPU off it must not claim one at all
/// (this used to fall back to "GPU · wgpu").
pub(crate) fn gpu_label(status: &GpuStatusIndicator, inference_backend: Option<&str>) -> String {
    let gpu_inference = inference_backend == Some("wgsl-gpu");
    match status.adapter_name.as_deref() {
        Some(name) if gpu_inference || status.mode == GpuStatusMode::Available => {
            format!("GPU · {name}")
        }
        _ if gpu_inference => "GPU · inference only".to_string(),
        _ => "CPU".to_string(),
    }
}

fn primary_btn(ui: &mut Ui, label: &str, fg: Color32, bg: Color32) -> bool {
    ui.add(
        egui::Button::new(egui::RichText::new(label).color(fg))
            .fill(bg)
            .min_size(Vec2::new(0.0, 34.0)),
    )
    .clicked()
}

fn ghost_btn(ui: &mut Ui, label: &str) -> bool {
    ui.add(egui::Button::new(label).min_size(Vec2::new(0.0, 34.0)))
        .clicked()
}

fn icon_btn(ui: &mut Ui, icon: &str, tooltip: &str, enabled: bool, action: impl FnOnce()) -> bool {
    let clicked = ui
        .add_enabled(enabled, egui::Button::new(icon).min_size(Vec2::splat(34.0)))
        .on_hover_text(tooltip)
        .clicked();
    if clicked {
        action();
    }
    clicked
}

fn toggle_btn(ui: &mut Ui, label: &str, active: bool) -> bool {
    ui.add(
        egui::Button::new(egui::RichText::new(label).color(if active { P::BG } else { P::INK }))
            .selected(active)
            .min_size(Vec2::new(0.0, 34.0)),
    )
    .clicked()
}

fn danger_btn(ui: &mut Ui, label: &str, action: impl FnOnce()) -> bool {
    let clicked = ghost_btn(ui, label);
    if clicked {
        action();
    }
    clicked
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
    fn gpu_label_names_a_gpu_only_when_detection_uses_one() {
        let available = GpuStatusIndicator::available("RTX 4090", "Dx12", None, None, None);
        assert_eq!(gpu_label(&available, Some("onnxruntime")), "GPU · RTX 4090");
        assert_eq!(gpu_label(&available, None), "GPU · RTX 4090");

        // GPU off (or preprocessing off): no adapter, and nothing should say GPU...
        let disabled = GpuStatusIndicator::disabled("GPU preprocessing disabled");
        assert_eq!(gpu_label(&disabled, Some("onnxruntime")), "CPU");
        // ...unless inference is still on the GPU.
        assert_eq!(
            gpu_label(&disabled, Some("wgsl-gpu")),
            "GPU · inference only"
        );

        // The preprocessor failed: its adapter only counts if inference uses it.
        let fallback = GpuStatusIndicator::fallback("boom", Some("RTX 4090".into()), None);
        assert_eq!(gpu_label(&fallback, Some("cpu-graph")), "CPU");
        assert_eq!(gpu_label(&fallback, Some("wgsl-gpu")), "GPU · RTX 4090");
    }

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
}
