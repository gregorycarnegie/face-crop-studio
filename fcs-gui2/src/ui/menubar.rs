//! Menu bar.

use crate::app::collect_folder_images;
use crate::theme::P;
use crate::types::App2;
use egui::{Frame, Popup, Sense, Stroke, Ui, Vec2};

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

                menu_item(ui, "File", |ui| {
                    if ui.button("Open Images…").clicked() {
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
                    }
                    if ui.button("Open Folder…").clicked() {
                        if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                            let paths = collect_folder_images(&dir);
                            let first = paths.first().cloned();
                            app.enqueue_batch_paths(paths);
                            if let Some(path) = first {
                                app.load_image_path(path);
                            }
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
                    if ui.button("Quit").clicked() {
                        ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });

                menu_item(ui, "Edit", |ui| {
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

                menu_item(ui, "View", |ui| {
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

                menu_item(ui, "Detect", |ui| {
                    if ui.button("Run Detection").clicked() {
                        if let Some(path) = app.preview.image_path.clone() {
                            app.load_image_path(path);
                        }
                    }
                    ui.separator();
                    if ui.button("Clear Detections").clicked() {
                        app.preview.detections.clear();
                        app.selected_faces.clear();
                    }
                });

                menu_item(ui, "Tools", |ui| {
                    ui.checkbox(&mut app.manual_box_tool_enabled, "Draw Box Tool");
                });

                menu_item(ui, "Window", |ui| {
                    if ui.button("Minimize").clicked() {
                        ui.ctx()
                            .send_viewport_cmd(egui::ViewportCommand::Minimized(true));
                    }
                    if ui.button("Maximize").clicked() {
                        ui.ctx()
                            .send_viewport_cmd(egui::ViewportCommand::Maximized(true));
                    }
                });

                menu_item(ui, "Help", |ui| {
                    if ui.button("About Face Crop Studio").clicked() {
                        app.status_line =
                            "Face Crop Studio — YuNet face detection".to_string();
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

fn menu_item(ui: &mut egui::Ui, label: &str, add_contents: impl FnOnce(&mut egui::Ui)) {
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

    Popup::menu(&resp).width(180.0).show(|ui| {
        add_contents(ui);
    });
}
