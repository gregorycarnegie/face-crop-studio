//! Menu bar.

use crate::app::collect_folder_images;
use crate::core::settings::persist_with_feedback;
use crate::theme::P;
use crate::types::App2;
use egui::{Frame, Popup, RichText, Sense, Ui, Vec2};
use fcs_utils::{
    config::{MetadataMode, ResizeQuality},
    configure_telemetry,
};

pub fn show(ui: &mut Ui, app: &mut App2) {
    egui::Panel::top("menubar")
        .exact_size(32.0)
        .show_separator_line(false)
        .frame(
            Frame::new()
                .fill(P::white_alpha(3))
                .inner_margin(egui::Margin::ZERO),
        )
        .show(ui, |ui| {
            ui.horizontal_centered(|ui| {
                ui.add_space(6.0);

                menu_item(ui, "File", 180.0, |ui| {
                    if ui.button("Open Images…").clicked()
                        && let Some(paths) = rfd::FileDialog::new()
                            .add_filter("Images", fcs_utils::SUPPORTED_IMAGE_EXTENSIONS)
                            .pick_files()
                    {
                        let first = paths.first().cloned();
                        app.enqueue_batch_paths(paths);
                        if let Some(path) = first {
                            app.load_image_path(path);
                        }
                    }
                    if ui.button("Open Folder…").clicked()
                        && let Some(dir) = rfd::FileDialog::new().pick_folder()
                    {
                        let paths = collect_folder_images(&dir);
                        let first = paths.first().cloned();
                        app.enqueue_batch_paths(paths);
                        if let Some(path) = first {
                            app.load_image_path(path);
                        }
                    }
                    ui.separator();
                    if ui.button("Export Selected").clicked() {
                        crate::core::export::export_selected_faces(app);
                    }
                    if ui.button("Export All").clicked() {
                        crate::core::export::start_batch_export(app);
                    }
                    ui.separator();
                    if ui.button("Save Settings").clicked() {
                        let path = app.settings_path.clone();
                        if let Err(msg) = persist_with_feedback(&app.settings, &path) {
                            app.last_error = Some(msg);
                        }
                    }
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });

                menu_item(ui, "Edit", 180.0, |ui| {
                    if ui.button("Select All").clicked() {
                        let n = app.preview.detections.len();
                        app.selected_faces = (0..n).collect();
                    }
                    if ui.button("Deselect All").clicked() {
                        app.selected_faces.clear();
                    }
                    ui.separator();
                    if ui.button("Clear Queue").clicked() {
                        app.batch_files.clear();
                    }
                });

                menu_item(ui, "View", 180.0, |ui| {
                    ui.checkbox(&mut app.show_crop_overlay, "Show Crop Overlay");
                    ui.separator();
                    if ui.button("Zoom In").clicked() {
                        app.zoom = (app.zoom * 1.25).min(8.0);
                    }
                    if ui.button("Zoom Out").clicked() {
                        app.zoom = (app.zoom / 1.25).max(0.1);
                    }
                    if ui.button("Reset Zoom").clicked() {
                        app.zoom = 1.0;
                        app.pan = egui::Vec2::ZERO;
                    }
                });

                menu_item(ui, "Detect", 180.0, |ui| {
                    if ui.button("Run Detection").clicked()
                        && let Some(path) = app.preview.image_path.clone()
                    {
                        app.load_image_path(path);
                    }
                    ui.separator();
                    if ui.button("Clear Detections").clicked() {
                        app.preview.detections.clear();
                        app.selected_faces.clear();
                    }
                });

                menu_item(ui, "Tools", 180.0, |ui| {
                    ui.checkbox(&mut app.manual_box_tool_enabled, "Draw Box Tool");
                });

                menu_item(ui, "Settings", 240.0, |ui| {
                    if ui.button("Restore Defaults").clicked() {
                        app.settings = app.default_settings.clone();
                        app.needs_detector_rebuild = true;
                        configure_telemetry(
                            app.settings.telemetry.enabled,
                            app.settings.telemetry.level_filter(),
                        );
                    }
                    if ui.button("Set as Default").clicked() {
                        app.default_settings = app.settings.clone();
                    }
                    ui.separator();

                    // ── Detection ──────────────────────────────────────────────
                    section_label(ui, "Detection");
                    let r = ui.add(
                        egui::Slider::new(&mut app.settings.detection.nms_threshold, 0.0..=1.0)
                            .text("NMS threshold")
                            .step_by(0.01),
                    );
                    if r.drag_stopped() || (r.changed() && !r.dragged()) {
                        app.needs_detector_rebuild = true;
                    }
                    let r = ui.add(
                        egui::DragValue::new(&mut app.settings.detection.top_k)
                            .prefix("Top K: ")
                            .range(1..=10_000)
                            .speed(10),
                    );
                    if r.drag_stopped() || (r.changed() && !r.dragged()) {
                        app.needs_detector_rebuild = true;
                    }

                    // ── Input ──────────────────────────────────────────────────
                    ui.separator();
                    section_label(ui, "Input");
                    let old_rq = app.settings.input.resize_quality;
                    ui.horizontal(|ui| {
                        ui.label("Resize quality");
                        ui.radio_value(
                            &mut app.settings.input.resize_quality,
                            ResizeQuality::Quality,
                            "Quality",
                        );
                        ui.radio_value(
                            &mut app.settings.input.resize_quality,
                            ResizeQuality::Speed,
                            "Speed",
                        );
                    });
                    if app.settings.input.resize_quality != old_rq {
                        app.needs_detector_rebuild = true;
                    }

                    // ── GPU ────────────────────────────────────────────────────
                    ui.separator();
                    section_label(ui, "GPU");
                    if ui
                        .checkbox(&mut app.settings.gpu.enabled, "Enable GPU")
                        .changed()
                    {
                        app.needs_detector_rebuild = true;
                    }
                    ui.add_enabled_ui(app.settings.gpu.enabled, |ui| {
                        if ui
                            .checkbox(&mut app.settings.gpu.inference, "GPU inference")
                            .changed()
                        {
                            app.needs_detector_rebuild = true;
                        }
                        if ui
                            .checkbox(&mut app.settings.gpu.preprocessing, "GPU preprocessing")
                            .changed()
                        {
                            app.needs_detector_rebuild = true;
                        }
                        if ui
                            .checkbox(&mut app.settings.gpu.respect_env, "Respect env overrides")
                            .changed()
                        {
                            app.needs_detector_rebuild = true;
                        }
                    });

                    // ── Telemetry ──────────────────────────────────────────────
                    ui.separator();
                    section_label(ui, "Telemetry");
                    let mut tel_dirty = false;
                    if ui
                        .checkbox(&mut app.settings.telemetry.enabled, "Enable telemetry")
                        .changed()
                    {
                        tel_dirty = true;
                    }
                    let old_level = app.settings.telemetry.level.clone();
                    egui::ComboBox::new("telemetry_level", "Log level")
                        .selected_text(app.settings.telemetry.level.as_str())
                        .show_ui(ui, |ui| {
                            for lvl in ["off", "error", "warn", "info", "debug", "trace"] {
                                let selected = app.settings.telemetry.level == lvl;
                                if ui.selectable_label(selected, lvl).clicked() {
                                    app.settings.telemetry.level = lvl.to_string();
                                }
                            }
                        });
                    if app.settings.telemetry.level != old_level {
                        tel_dirty = true;
                    }
                    if tel_dirty {
                        configure_telemetry(
                            app.settings.telemetry.enabled,
                            app.settings.telemetry.level_filter(),
                        );
                    }

                    // ── Quality Rules ──────────────────────────────────────────
                    ui.separator();
                    section_label(ui, "Quality Rules");
                    ui.checkbox(
                        &mut app.settings.crop.quality_rules.auto_select_best_face,
                        "Auto-select best face",
                    );
                    ui.checkbox(
                        &mut app.settings.crop.quality_rules.auto_skip_no_high_quality,
                        "Skip if no high-quality face",
                    );
                    ui.checkbox(
                        &mut app.settings.crop.quality_rules.quality_suffix,
                        "Append quality suffix",
                    );

                    // ── Metadata ───────────────────────────────────────────────
                    ui.separator();
                    section_label(ui, "Metadata");
                    egui::ComboBox::new("metadata_mode", "Mode")
                        .selected_text(metadata_mode_label(&app.settings.crop.metadata.mode))
                        .show_ui(ui, |ui| {
                            for (mode, label) in [
                                (MetadataMode::Preserve, "Preserve"),
                                (MetadataMode::Strip, "Strip"),
                                (MetadataMode::Custom, "Custom"),
                            ] {
                                let selected = app.settings.crop.metadata.mode == mode;
                                if ui.selectable_label(selected, label).clicked() {
                                    app.settings.crop.metadata.mode = mode;
                                }
                            }
                        });
                    ui.checkbox(
                        &mut app.settings.crop.metadata.include_crop_settings,
                        "Include crop settings",
                    );
                    ui.checkbox(
                        &mut app.settings.crop.metadata.include_quality_metrics,
                        "Include quality metrics",
                    );
                });

                menu_item(ui, "Help", 180.0, |ui| {
                    if ui.button("About Face Crop Studio").clicked() {
                        app.show_about = true;
                    }
                });

                // Right: model status
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add_space(10.0);
                    let model_ready = app.detector.is_some();
                    let (dot_color, status_text) = if model_ready {
                        (P::LIME, "YuNet 640 · ready".to_string())
                    } else {
                        (P::ROSE, "YuNet · no model".to_string())
                    };
                    ui.label(
                        egui::RichText::new(&status_text)
                            .size(10.5)
                            .color(P::INK3)
                            .family(egui::FontFamily::Monospace),
                    );
                    let (resp, painter) = ui.allocate_painter(Vec2::splat(6.0), Sense::hover());
                    painter.circle_filled(resp.rect.center(), 3.0, dot_color);
                });
            });
        });
}

fn section_label(ui: &mut Ui, text: &str) {
    ui.label(
        RichText::new(text)
            .size(10.0)
            .color(P::INK3)
            .family(egui::FontFamily::Monospace),
    );
}

fn metadata_mode_label(mode: &MetadataMode) -> &'static str {
    match mode {
        MetadataMode::Preserve => "Preserve",
        MetadataMode::Strip => "Strip",
        MetadataMode::Custom => "Custom",
    }
}

fn menu_item(
    ui: &mut egui::Ui,
    label: &str,
    popup_width: f32,
    add_contents: impl FnOnce(&mut egui::Ui),
) {
    let font = egui::FontId::proportional(13.0);
    let galley = ui
        .painter()
        .layout_no_wrap(label.to_string(), font, P::INK2);
    let w = galley.size().x + 20.0;
    let (resp, painter) = ui.allocate_painter(Vec2::new(w, 32.0), Sense::click());

    let popup_id = Popup::default_response_id(&resp);
    let is_open = Popup::is_id_open(ui.ctx(), popup_id);

    let highlighted = resp.hovered() || is_open;
    if highlighted {
        painter.rect_filled(
            resp.rect.shrink2(Vec2::new(2.0, 4.0)),
            5.0,
            P::white_alpha(if is_open { 20 } else { 15 }),
        );
    }
    let text_color = if highlighted { P::INK } else { P::INK2 };
    painter.galley(
        egui::pos2(
            resp.rect.min.x + 10.0,
            resp.rect.center().y - galley.size().y / 2.0,
        ),
        galley,
        text_color,
    );

    Popup::menu(&resp).width(popup_width).show(|ui| {
        add_contents(ui);
    });
}
