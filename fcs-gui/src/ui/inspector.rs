//! Right inspector panel — Crop / Output / Enhance tabs.

use crate::theme::P;
use crate::types::{App2, InspectorTab};
use crate::ui::shape::shape_controls;
use crate::ui::widgets::{
    field_label, panel_header, segmented_control, slider_with_label, toggle_row,
};
use egui::{Sense, Stroke, Ui, Vec2};
use fcs_core::preset_by_name;
use fcs_utils::CropShape;

const BODY_MARGIN_X: i8 = 14;

pub fn show(ui: &mut Ui, app: &mut App2) {
    ui.set_min_height(ui.available_height());

    // Tab bar
    inspector_tab_bar(ui, app);

    // Mini stats
    mini_stats(ui, app);

    // Scrollable content
    egui::ScrollArea::vertical()
        .id_salt("inspector_scroll")
        .show(ui, |ui| match app.inspector_tab {
            InspectorTab::Crop => crop_tab(ui, app),
            InspectorTab::Output => output_tab(ui, app),
            InspectorTab::Enhance => enhance_tab(ui, app),
        });
}

fn inspector_tab_bar(ui: &mut Ui, app: &mut App2) {
    let tabs = [
        ("Crop", InspectorTab::Crop),
        ("Output", InspectorTab::Output),
        ("Enhance", InspectorTab::Enhance),
    ];
    ui.painter().line_segment(
        [
            egui::pos2(ui.min_rect().min.x, ui.min_rect().min.y + 32.0),
            egui::pos2(ui.min_rect().max.x, ui.min_rect().min.y + 32.0),
        ],
        Stroke::new(1.0, P::RULE),
    );
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        ui.set_height(32.0);
        let w = ui.available_width() / tabs.len() as f32;
        for (label, variant) in &tabs {
            let is_active = app.inspector_tab == *variant;
            let (resp, painter) = ui.allocate_painter(Vec2::new(w, 32.0), Sense::click());
            let text_color = if is_active { P::CYAN } else { P::INK3 };
            if resp.hovered() && !is_active {
                painter.rect_filled(resp.rect, 0.0, P::white_alpha(5));
            }
            if is_active {
                painter.rect_filled(resp.rect, 0.0, P::cyan_alpha(10));
                painter.line_segment(
                    [
                        egui::pos2(resp.rect.min.x, resp.rect.max.y - 2.0),
                        egui::pos2(resp.rect.max.x, resp.rect.max.y - 2.0),
                    ],
                    Stroke::new(2.0, P::CYAN),
                );
            }
            painter.text(
                resp.rect.center(),
                egui::Align2::CENTER_CENTER,
                *label,
                egui::FontId::monospace(10.5),
                text_color,
            );
            if resp.clicked() {
                app.inspector_tab = *variant;
            }
        }
    });
}

fn mini_stats(ui: &mut Ui, app: &App2) {
    let total = app.preview.detections.len();
    let selected = app.selected_faces.len();
    let detect_ms = app.last_detect_ms.unwrap_or(0);
    let (src_w, src_h) = app.preview.image_size.unwrap_or((0, 0));

    ui.painter().line_segment(
        [
            egui::pos2(ui.min_rect().min.x, ui.cursor().min.y + 116.0),
            egui::pos2(ui.min_rect().max.x, ui.cursor().min.y + 116.0),
        ],
        Stroke::new(1.0, P::RULE),
    );

    let w = ui.available_width() / 2.0;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        stat_cell(ui, "Faces", &total.to_string(), P::PEACH, w);
        stat_cell(ui, "Selected", &selected.to_string(), P::CYAN, w);
    });
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        stat_cell(ui, "Detect", &format!("{detect_ms}ms"), P::LIME, w);
        stat_cell(ui, "Source", &format!("{src_w}×{src_h}"), P::ROSE, w);
    });
}

fn stat_cell(ui: &mut egui::Ui, label: &str, value: &str, color: egui::Color32, w: f32) {
    let (resp, painter) = ui.allocate_painter(Vec2::new(w, 56.0), Sense::hover());
    let rect = resp.rect;
    painter.text(
        egui::pos2(rect.center().x, rect.min.y + 16.0),
        egui::Align2::CENTER_CENTER,
        label,
        egui::FontId::monospace(9.5),
        P::INK3,
    );
    painter.text(
        egui::pos2(rect.center().x, rect.min.y + 36.0),
        egui::Align2::CENTER_CENTER,
        value,
        egui::FontId::proportional(18.0),
        color,
    );
}

// ── Crop tab ──────────────────────────────────────────────────────────────────

fn crop_tab(ui: &mut Ui, app: &mut App2) {
    panel_01_crop_framing(ui, app);
    panel_02_crop_shape(ui, app);
    panel_03_positioning(ui, app);
    panel_04_crops_ready(ui, app);
    panel_05_enhancement(ui, app);
}

fn panel_01_crop_framing(ui: &mut Ui, app: &mut App2) {
    let open = &mut app.panel_state.crop_framing;
    let toggled = panel_header(ui, "01", "Crop framing", *open);
    if toggled {
        *open = !*open;
    }
    if !*open {
        return;
    }

    inspector_body(ui, |ui| {
        ui.add_space(4.0);
        ui.spacing_mut().item_spacing.y = 10.0;

        // Preset dropdown
        field_label(ui, "Preset");
        const PRESETS: &[(&str, &str)] = &[
            ("LinkedIn", "LinkedIn (400×400)"),
            ("Passport", "Passport (413×531)"),
            ("Instagram", "Instagram (1080×1080)"),
            ("Headshot", "Headshot (600×800)"),
            ("ID Card", "ID Card (332×498)"),
            ("Avatar", "Avatar (512×512)"),
            ("Custom", "Custom…"),
        ];
        let current_label = PRESETS
            .iter()
            .find(|(key, _)| *key == app.settings.crop.preset)
            .map(|(_, label)| *label)
            .unwrap_or("Custom…");
        let mut selected_key = app.settings.crop.preset.clone();
        egui::ComboBox::from_id_salt("preset_combo")
            .selected_text(current_label)
            .width(ui.available_width())
            .show_ui(ui, |ui| {
                for (key, label) in PRESETS {
                    ui.selectable_value(&mut selected_key, key.to_string(), *label);
                }
            });
        if selected_key != app.settings.crop.preset {
            app.settings.crop.preset = selected_key.clone();
            if let Some(p) = preset_by_name(&selected_key) {
                app.settings.crop.output_width = p.width;
                app.settings.crop.output_height = p.height;
            }
            app.settings.crop.shape = CropShape::Rectangle;
            let c = &mut app.settings.crop;
            match selected_key.as_str() {
                "LinkedIn" => {
                    c.face_height_pct = 55.0;
                    c.positioning_mode = "center".into();
                    c.horizontal_offset = 0.0;
                    c.vertical_offset = 0.0;
                }
                "Passport" => {
                    c.face_height_pct = 75.0;
                    c.positioning_mode = "center".into();
                    c.horizontal_offset = 0.0;
                    c.vertical_offset = 0.0;
                }
                "Instagram" => {
                    c.face_height_pct = 55.0;
                    c.positioning_mode = "center".into();
                    c.horizontal_offset = 0.0;
                    c.vertical_offset = 0.0;
                }
                "Headshot" => {
                    c.face_height_pct = 55.0;
                    c.positioning_mode = "rule-of-thirds".into();
                    c.horizontal_offset = 0.0;
                    c.vertical_offset = 0.0;
                }
                "ID Card" => {
                    c.face_height_pct = 62.0;
                    c.positioning_mode = "center".into();
                    c.horizontal_offset = 0.0;
                    c.vertical_offset = 0.0;
                }
                "Avatar" => {
                    c.face_height_pct = 70.0;
                    c.positioning_mode = "center".into();
                    c.horizontal_offset = 0.0;
                    c.vertical_offset = 0.0;
                }
                _ => {}
            }
            app.crop_preview_cache.clear();
        }

        // Aspect ratio
        field_label(ui, "Aspect ratio");
        let aspect_options = ["Free", "1:1", "4:5", "3:4"];
        let prev_asp = app.aspect_ratio_idx;
        segmented_control(ui, &aspect_options, &mut app.aspect_ratio_idx);
        if app.aspect_ratio_idx != prev_asp {
            match app.aspect_ratio_idx {
                1 => {
                    app.settings.crop.output_width = 1024;
                    app.settings.crop.output_height = 1024;
                }
                2 => {
                    app.settings.crop.output_width = 1024;
                    app.settings.crop.output_height = 1280;
                }
                3 => {
                    app.settings.crop.output_width = 768;
                    app.settings.crop.output_height = 1024;
                }
                _ => {} // Free — keep current dimensions
            }
        }

        // Face height (stored as 0-100)
        let pct_label = format!("Face height · {:.0}%", app.settings.crop.face_height_pct);
        field_label(ui, &pct_label);
        slider_with_label(
            ui,
            "",
            &mut app.settings.crop.face_height_pct,
            10.0,
            100.0,
            "pct",
        );

        // Width / Height
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.set_width((ui.available_width() - 6.0) / 2.0);
                field_label(ui, "Width");
                let mut w_str = app.settings.crop.output_width.to_string();
                if ui
                    .add(egui::TextEdit::singleline(&mut w_str).desired_width(ui.available_width()))
                    .changed()
                    && let Ok(v) = w_str.parse::<u32>()
                {
                    app.settings.crop.output_width = v;
                }
            });
            ui.add_space(6.0);
            ui.vertical(|ui| {
                field_label(ui, "Height");
                let mut h_str = app.settings.crop.output_height.to_string();
                if ui
                    .add(egui::TextEdit::singleline(&mut h_str).desired_width(ui.available_width()))
                    .changed()
                    && let Ok(v) = h_str.parse::<u32>()
                {
                    app.settings.crop.output_height = v;
                }
            });
        });

        // Crop overlay toggle
        toggle_row(ui, "Show crop overlay", &mut app.show_crop_overlay);

        // Confidence floor
        field_label(
            ui,
            &format!(
                "Confidence floor · {:.2}",
                app.settings.detection.score_threshold
            ),
        );
        slider_with_label(
            ui,
            "",
            &mut app.settings.detection.score_threshold,
            0.0,
            1.0,
            "conf",
        );

        // Fill color
        field_label(ui, "Padding fill color");
        let fc = &mut app.settings.crop.fill_color;
        let mut color = [fc.red, fc.green, fc.blue, fc.alpha];
        if ui
            .color_edit_button_srgba_unmultiplied(&mut color)
            .changed()
        {
            fc.red = color[0];
            fc.green = color[1];
            fc.blue = color[2];
            fc.alpha = color[3];
        }

        ui.add_space(6.0);
    });
    separator_line(ui);
}

fn panel_02_crop_shape(ui: &mut Ui, app: &mut App2) {
    let open = &mut app.panel_state.crop_shape;
    let toggled = panel_header(ui, "02", "Crop shape", *open);
    if toggled {
        *open = !*open;
    }
    if !*open {
        return;
    }

    inspector_body(ui, |ui| {
        ui.add_space(4.0);
        ui.spacing_mut().item_spacing.y = 6.0;
        shape_controls(ui, app);
        ui.add_space(6.0);
    });
    separator_line(ui);
}

fn panel_03_positioning(ui: &mut Ui, app: &mut App2) {
    let open = &mut app.panel_state.positioning;
    let toggled = panel_header(ui, "03", "Positioning", *open);
    if toggled {
        *open = !*open;
    }
    if !*open {
        return;
    }

    inspector_body(ui, |ui| {
        ui.spacing_mut().item_spacing.y = 10.0;

        field_label(ui, "Mode");
        let modes = ["Center", "Thirds", "Custom"];
        let mut mode_idx: usize = match app.settings.crop.positioning_mode.as_str() {
            "center" => 0,
            "thirds" => 1,
            _ => 2,
        };
        let prev = mode_idx;
        segmented_control(ui, &modes, &mut mode_idx);
        if mode_idx != prev {
            app.settings.crop.positioning_mode = match mode_idx {
                0 => "center",
                1 => "thirds",
                _ => "custom",
            }
            .to_string();
        }

        // Offsets
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.set_width((ui.available_width() - 6.0) / 2.0);
                field_label(ui, "Offset X");
                let mut x_str = format!("{:.0}", app.settings.crop.horizontal_offset);
                if ui
                    .add(egui::TextEdit::singleline(&mut x_str).desired_width(ui.available_width()))
                    .changed()
                    && let Ok(v) = x_str.parse::<f32>()
                {
                    app.settings.crop.horizontal_offset = v;
                }
            });
            ui.add_space(6.0);
            ui.vertical(|ui| {
                field_label(ui, "Offset Y");
                let mut y_str = format!("{:.0}", app.settings.crop.vertical_offset);
                if ui
                    .add(egui::TextEdit::singleline(&mut y_str).desired_width(ui.available_width()))
                    .changed()
                    && let Ok(v) = y_str.parse::<f32>()
                {
                    app.settings.crop.vertical_offset = v;
                }
            });
        });

        toggle_row(ui, "Eye-line align", &mut app.settings.crop.eye_line_align);
        toggle_row(
            ui,
            "Auto-orient via EXIF",
            &mut app.settings.crop.auto_orient_exif,
        );
        ui.add_space(6.0);
    });
    separator_line(ui);
}

fn panel_04_crops_ready(ui: &mut Ui, app: &mut App2) {
    let n_ready = app.selected_faces.len();
    let label = format!("Crops  {} ready", n_ready);
    let open = &mut app.panel_state.crops_ready;
    let toggled = panel_header(ui, "04", &label, *open);
    if toggled {
        *open = !*open;
    }
    if !*open {
        return;
    }

    let selected: Vec<usize> = app.selected_faces.iter().copied().collect();
    let mut save_face = None;
    inspector_body(ui, |ui| {
        for &i in &selected {
            if let Some(det) = app.preview.detections.get(i) {
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.set_height(64.0);

                    // Thumbnail
                    let (thumb_resp, painter) =
                        ui.allocate_painter(Vec2::splat(60.0), Sense::hover());
                    let tr = thumb_resp.rect;
                    painter.rect_filled(tr, 7.0, P::BG2);
                    painter.rect_stroke(
                        tr,
                        7.0,
                        Stroke::new(1.0, P::RULE),
                        egui::StrokeKind::Outside,
                    );
                    if let Some(tex) = &det.thumbnail {
                        painter.image(
                            tex.id(),
                            tr,
                            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                            egui::Color32::WHITE,
                        );
                    } else {
                        painter.text(
                            tr.center(),
                            egui::Align2::CENTER_CENTER,
                            "◉",
                            egui::FontId::proportional(20.0),
                            P::INK3,
                        );
                    }

                    ui.add_space(8.0);
                    ui.vertical(|ui| {
                        let name = app
                            .preview
                            .image_path
                            .as_deref()
                            .and_then(|p| p.file_stem())
                            .and_then(|s| s.to_str())
                            .unwrap_or("crop");
                        ui.label(
                            egui::RichText::new(format!("{name}_face{}.jpg", i + 1))
                                .size(11.0)
                                .color(P::INK)
                                .family(egui::FontFamily::Monospace),
                        );
                        let w = app.settings.crop.output_width;
                        let h = app.settings.crop.output_height;
                        ui.label(
                            egui::RichText::new(format!("{w}×{h} · q92"))
                                .size(10.0)
                                .color(P::INK3)
                                .family(egui::FontFamily::Monospace),
                        );
                        ui.label(
                            egui::RichText::new("● ready")
                                .size(10.0)
                                .color(P::LIME)
                                .family(egui::FontFamily::Monospace),
                        );
                    });

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let (resp, painter) =
                            ui.allocate_painter(Vec2::new(46.0, 26.0), Sense::click());
                        let bg = if resp.hovered() {
                            P::lime_alpha(50)
                        } else {
                            P::lime_alpha(30)
                        };
                        painter.rect_filled(resp.rect, 6.0, bg);
                        painter.rect_stroke(
                            resp.rect,
                            6.0,
                            Stroke::new(1.0, P::lime_alpha(76)),
                            egui::StrokeKind::Outside,
                        );
                        painter.text(
                            resp.rect.center(),
                            egui::Align2::CENTER_CENTER,
                            "Save",
                            egui::FontId::monospace(10.0),
                            P::LIME,
                        );
                        if resp.clicked() {
                            save_face = Some(i);
                        }
                    });
                });
            }
        }
        ui.add_space(8.0);
    });
    if let Some(face_index) = save_face {
        crate::core::export::export_one_face(app, face_index);
    }
    separator_line(ui);
}

fn panel_05_enhancement(ui: &mut Ui, app: &mut App2) {
    let open = &mut app.panel_state.enhancement;
    let toggled = panel_header(ui, "05", "Enhancement", *open);
    if toggled {
        *open = !*open;
    }
    if !*open {
        return;
    }

    inspector_body(ui, |ui| {
        ui.spacing_mut().item_spacing.y = 2.0;
        toggle_row(ui, "Auto color", &mut app.settings.enhance.auto_color);
        let mut skin_on = app.settings.enhance.skin_smooth > 0.0;
        if toggle_row(ui, "Skin smoothing", &mut skin_on) {
            app.settings.enhance.skin_smooth = if skin_on { 0.5 } else { 0.0 };
        }
        let mut sharpen_on = app.settings.enhance.sharpness > 0.0;
        if toggle_row(ui, "Sharpen", &mut sharpen_on) {
            app.settings.enhance.sharpness = if sharpen_on { 1.0 } else { 0.0 };
        }
        toggle_row(
            ui,
            "Red-eye removal",
            &mut app.settings.enhance.red_eye_removal,
        );
        ui.add_space(6.0);
    });
    separator_line(ui);
}

// ── Output tab ────────────────────────────────────────────────────────────────

fn output_tab(ui: &mut Ui, app: &mut App2) {
    inspector_body(ui, |ui| {
        ui.add_space(12.0);
        field_label(ui, "Format");
        let formats = ["JPEG", "PNG", "WEBP", "AVIF"];
        let mut fmt_idx = match app
            .settings
            .crop
            .output_format
            .to_ascii_uppercase()
            .as_str()
        {
            "PNG" => 1,
            "WEBP" => 2,
            "AVIF" => 3,
            _ => 0,
        };
        let fmt_idx_bak = fmt_idx;
        segmented_control(ui, &formats, &mut fmt_idx);
        if fmt_idx != fmt_idx_bak {
            app.settings.crop.output_format = match fmt_idx {
                1 => "png",
                2 => "webp",
                3 => "avif",
                _ => "jpeg",
            }
            .to_string();
        }

        ui.add_space(8.0);
        field_label(ui, "JPEG quality");
        let mut q = app.settings.crop.jpeg_quality as f32;
        slider_with_label(ui, "", &mut q, 50.0, 100.0, "px");
        app.settings.crop.jpeg_quality = q as u8;

        ui.add_space(8.0);
        field_label(ui, "Output directory");
        let mut dir_str = String::new(); // output_dir not in AppSettings; use rfd directly
        ui.add(egui::TextEdit::singleline(&mut dir_str).desired_width(ui.available_width()));

        ui.add_space(4.0);
        if ui.button("Browse…").clicked()
            && let Some(_dir) = rfd::FileDialog::new().pick_folder()
        {
            // would persist to settings if field exists
        }
    });
}

// ── Enhance tab ───────────────────────────────────────────────────────────────

fn enhance_tab(ui: &mut Ui, app: &mut App2) {
    inspector_body(ui, |ui| {
        ui.add_space(12.0);

        field_label(
            ui,
            &format!("Exposure · {:.2}", app.settings.enhance.exposure_stops),
        );
        slider_with_label(
            ui,
            "",
            &mut app.settings.enhance.exposure_stops,
            -3.0,
            3.0,
            "",
        );
        ui.add_space(4.0);

        field_label(
            ui,
            &format!("Brightness · {}", app.settings.enhance.brightness),
        );
        let mut bright = app.settings.enhance.brightness as f32;
        slider_with_label(ui, "", &mut bright, -100.0, 100.0, "px");
        app.settings.enhance.brightness = bright as i32;
        ui.add_space(4.0);

        field_label(
            ui,
            &format!("Contrast · {:.2}", app.settings.enhance.contrast),
        );
        slider_with_label(ui, "", &mut app.settings.enhance.contrast, -2.0, 2.0, "");
        ui.add_space(4.0);

        field_label(
            ui,
            &format!("Saturation · {:.2}", app.settings.enhance.saturation),
        );
        slider_with_label(ui, "", &mut app.settings.enhance.saturation, 0.0, 3.0, "");
        ui.add_space(4.0);

        field_label(
            ui,
            &format!("Sharpness · {:.2}", app.settings.enhance.sharpness),
        );
        slider_with_label(ui, "", &mut app.settings.enhance.sharpness, 0.0, 4.0, "");
        ui.add_space(8.0);

        toggle_row(
            ui,
            "Background blur",
            &mut app.settings.enhance.background_blur,
        );
    });
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn inspector_body<R>(ui: &mut Ui, add_contents: impl FnOnce(&mut Ui) -> R) -> R {
    let outer_width = ui.available_width();
    egui::Frame::new()
        .inner_margin(egui::Margin::symmetric(BODY_MARGIN_X, 0))
        .show(ui, |ui| {
            let margin = f32::from(BODY_MARGIN_X) * 2.0;
            ui.set_width((outer_width - margin).max(0.0));
            add_contents(ui)
        })
        .inner
}

fn separator_line(ui: &mut Ui) {
    let (_, painter) = ui.allocate_painter(Vec2::new(ui.available_width(), 1.0), Sense::hover());
    let _y = painter.clip_rect().min.y;
    let _rect = ui.min_rect();
    // Draw via ui scope
    ui.painter().line_segment(
        [
            egui::pos2(ui.min_rect().min.x, ui.cursor().min.y),
            egui::pos2(ui.min_rect().max.x, ui.cursor().min.y),
        ],
        Stroke::new(1.0, P::RULE),
    );
}
