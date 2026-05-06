//! Centre canvas column — image viewport + face overlays + mini-log.

use crate::interaction::coords::{bbox_to_norm_rect, norm_to_screen};
use crate::rendering::paint::{
    draw_confidence_badge, draw_drag_handle, draw_face_box, draw_landmark_dot,
};
use crate::theme::P;
use crate::types::{App2, LogKind};
use crate::ui::widgets::{ctl_pill, face_chip};
use egui::{Color32, Frame, Rect, Sense, Stroke, Ui, Vec2};

pub fn show(ui: &mut Ui, app: &mut App2) {
    ui.set_min_size(ui.available_size());

    // Canvas header
    canvas_header(ui, app);

    // Stage (image + overlays) — takes remaining space minus the bottom bar
    let avail = ui.available_size();
    let bottom_bar_h = 44.0;
    let _stage_h = (avail.y - bottom_bar_h).max(100.0);

    egui::Panel::bottom("canvas_bottom_bar")
        .exact_size(bottom_bar_h)
        .frame(Frame::new().fill(P::black_alpha(50)))
        .show_inside(ui, |ui| {
            canvas_bottom_bar(ui, app);
        });

    egui::CentralPanel::default()
        .frame(Frame::new().fill(P::BG1))
        .show_inside(ui, |ui| {
            stage(ui, app);
        });
}

fn canvas_header(ui: &mut Ui, app: &mut App2) {
    let h = 42.0;
    let (resp, painter) = ui.allocate_painter(Vec2::new(ui.available_width(), h), Sense::hover());
    let r = resp.rect;
    painter.rect_filled(r, 0.0, P::black_alpha(38));
    painter.line_segment(
        [egui::pos2(r.min.x, r.max.y), egui::pos2(r.max.x, r.max.y)],
        Stroke::new(1.0, P::RULE),
    );

    // Filename + dimensions
    let name = app
        .preview
        .image_path
        .as_deref()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .unwrap_or("No image");
    let dims = app
        .preview
        .image_size
        .map(|(w, h)| format!("{w} × {h}"))
        .unwrap_or_default();

    painter.text(
        egui::pos2(r.min.x + 14.0, r.center().y),
        egui::Align2::LEFT_CENTER,
        name,
        egui::FontId::proportional(13.0),
        P::INK,
    );
    if !dims.is_empty() {
        let name_w = painter
            .layout_no_wrap(name.to_string(), egui::FontId::proportional(13.0), P::INK)
            .size()
            .x;
        painter.text(
            egui::pos2(r.min.x + 14.0 + name_w + 12.0, r.center().y),
            egui::Align2::LEFT_CENTER,
            &dims,
            egui::FontId::monospace(11.0),
            P::INK3,
        );
    }

    // Control chips on the right
    // (These use a child ui positioned on the right)
    let chips_x = r.max.x - 360.0;
    let clip = egui::Rect::from_min_max(egui::pos2(chips_x, r.min.y), r.max);
    // We draw them inline via a separate child ui
    let mut child = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(clip)
            .layout(egui::Layout::right_to_left(egui::Align::Center)),
    );
    child.add_space(12.0);
    let conf = app.settings.detection.score_threshold;
    let preset = app.settings.crop.preset.clone();
    let face_h = app.settings.crop.face_height_pct;
    let aspect = format!(
        "{}:{}",
        app.settings.crop.output_width, app.settings.crop.output_height
    );

    ctl_pill(&mut child, "preset ", &preset, Some(P::CYAN));
    child.add_space(4.0);
    ctl_pill(&mut child, "conf ", &format!("{conf:.2}"), Some(P::PEACH));
    child.add_space(4.0);
    ctl_pill(&mut child, "aspect ", &aspect, None);
    child.add_space(4.0);
    ctl_pill(&mut child, "face-h ", &format!("{:.0}%", face_h), None);
}

fn stage(ui: &mut Ui, app: &mut App2) {
    let avail = ui.available_rect_before_wrap();
    let pad = 18.0;
    let stage_outer = avail.shrink(pad);

    // Centre the image maintaining aspect ratio
    let image_rect = if let Some((iw, ih)) = app.preview.image_size {
        let img_ar = iw as f32 / ih as f32;
        let slot_ar = stage_outer.width() / stage_outer.height();
        if img_ar > slot_ar {
            let w = stage_outer.width();
            let h = w / img_ar;
            egui::Rect::from_center_size(stage_outer.center(), Vec2::new(w, h))
        } else {
            let h = stage_outer.height();
            let w = h * img_ar;
            egui::Rect::from_center_size(stage_outer.center(), Vec2::new(w, h))
        }
    } else {
        stage_outer
    };

    // Stage background
    let painter = ui.painter();
    painter.rect_filled(image_rect, 12.0, P::BG);
    painter.rect_stroke(
        image_rect,
        12.0,
        Stroke::new(1.0, P::RULE),
        egui::StrokeKind::Outside,
    );

    // Draw image texture if available
    if let Some(texture) = &app.preview.texture {
        painter.image(
            texture.id(),
            image_rect,
            Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            Color32::WHITE,
        );
    } else if app.is_busy {
        // Loading placeholder
        painter.text(
            image_rect.center(),
            egui::Align2::CENTER_CENTER,
            "Detecting faces…",
            egui::FontId::proportional(14.0),
            P::INK3,
        );
    } else {
        // Empty state
        painter.text(
            image_rect.center() - Vec2::new(0.0, 12.0),
            egui::Align2::CENTER_CENTER,
            "Drop an image to begin",
            egui::FontId::proportional(14.0),
            P::INK3,
        );
        painter.text(
            image_rect.center() + Vec2::new(0.0, 12.0),
            egui::Align2::CENTER_CENTER,
            "or click Detect faces →",
            egui::FontId::monospace(11.0),
            P::INK3,
        );
    }

    // Face detection overlays
    if let Some((img_w, img_h)) = app.preview.image_size {
        let iw = img_w as f32;
        let ih = img_h as f32;
        let dets: Vec<_> = app.preview.detections.iter().enumerate().collect();

        for (i, det) in &dets {
            let bbox = det.active_bbox();
            let norm = bbox_to_norm_rect(bbox.x, bbox.y, bbox.width, bbox.height, iw, ih);
            let screen_rect = norm_to_screen(norm, image_rect);

            let selected = app.selected_faces.contains(i);
            let color = if !selected {
                P::INK3
            } else if i % 2 == 0 {
                P::PEACH
            } else {
                P::CYAN
            };

            draw_face_box(painter, screen_rect, color, selected);

            // Confidence badge
            draw_confidence_badge(
                painter,
                &format!("{:.2}", det.detection.score),
                screen_rect,
                color,
            );

            // Landmarks
            for lm in &det.detection.landmarks {
                let nx = lm.x / iw;
                let ny = lm.y / ih;
                let lm_pos = egui::pos2(
                    image_rect.min.x + nx * image_rect.width(),
                    image_rect.min.y + ny * image_rect.height(),
                );
                draw_landmark_dot(painter, lm_pos);
            }

            // Drag handles on selected
            if selected {
                let corners = [
                    screen_rect.min,
                    egui::pos2(screen_rect.max.x, screen_rect.min.y),
                    egui::pos2(screen_rect.min.x, screen_rect.max.y),
                    screen_rect.max,
                ];
                for corner in corners {
                    draw_drag_handle(painter, corner, color);
                }
            }
        }
    }

    // Face click interaction
    let resp = ui.allocate_rect(image_rect, Sense::click());
    if resp.clicked() {
        if let Some(pos) = resp.interact_pointer_pos() {
            if let Some((img_w, img_h)) = app.preview.image_size {
                let iw = img_w as f32;
                let ih = img_h as f32;
                let mut clicked_any = false;
                for (i, det) in app.preview.detections.iter().enumerate() {
                    let bbox = det.active_bbox();
                    let norm = bbox_to_norm_rect(bbox.x, bbox.y, bbox.width, bbox.height, iw, ih);
                    let sr = norm_to_screen(norm, image_rect);
                    if sr.expand(4.0).contains(pos) {
                        if app.selected_faces.contains(&i) {
                            app.selected_faces.remove(&i);
                        } else {
                            app.selected_faces.insert(i);
                        }
                        clicked_any = true;
                        break;
                    }
                }
                if !clicked_any {
                    app.selected_faces.clear();
                }
            }
        }
    }

    // Mini-log overlay
    mini_log_overlay(ui, app, image_rect);
}

fn mini_log_overlay(ui: &mut Ui, app: &App2, image_rect: egui::Rect) {
    let log_w = 300.0;
    let log_max_lines = 5;
    let line_h = 16.0;
    let pad = 10.0;
    let header_h = 24.0;
    let n = app.log_lines.len().min(log_max_lines);
    let log_h = header_h + n as f32 * line_h + pad * 2.0;

    let log_rect = egui::Rect::from_min_size(
        egui::pos2(image_rect.min.x + 18.0, image_rect.max.y - log_h - 18.0),
        Vec2::new(log_w, log_h),
    );
    let painter = ui.painter();
    painter.rect_filled(log_rect, 9.0, P::SURFACE.linear_multiply(0.85));
    painter.rect_stroke(
        log_rect,
        9.0,
        Stroke::new(1.0, P::RULE),
        egui::StrokeKind::Outside,
    );

    painter.text(
        egui::pos2(log_rect.min.x + pad, log_rect.min.y + 8.0),
        egui::Align2::LEFT_TOP,
        "Pipeline log",
        egui::FontId::proportional(11.0),
        P::INK,
    );
    painter.text(
        egui::pos2(log_rect.max.x - pad, log_rect.min.y + 8.0),
        egui::Align2::RIGHT_TOP,
        "single.run",
        egui::FontId::monospace(9.5),
        P::CYAN,
    );

    let start = app.log_lines.len().saturating_sub(log_max_lines);
    for (i, line) in app.log_lines[start..].iter().enumerate() {
        let y = log_rect.min.y + header_h + i as f32 * line_h;
        let msg_color = match line.kind {
            LogKind::Ok => P::LIME,
            LogKind::Warn => P::PEACH,
            LogKind::Info => P::INK2,
        };
        painter.text(
            egui::pos2(log_rect.min.x + pad, y),
            egui::Align2::LEFT_TOP,
            &line.timestamp,
            egui::FontId::monospace(9.5),
            P::INK3,
        );
        painter.text(
            egui::pos2(log_rect.min.x + 62.0, y),
            egui::Align2::LEFT_TOP,
            &line.message,
            egui::FontId::monospace(9.5),
            msg_color,
        );
    }
}

fn canvas_bottom_bar(ui: &mut Ui, app: &mut App2) {
    ui.painter().line_segment(
        [
            egui::pos2(ui.min_rect().min.x, ui.min_rect().min.y),
            egui::pos2(ui.min_rect().max.x, ui.min_rect().min.y),
        ],
        Stroke::new(1.0, P::RULE),
    );

    ui.horizontal_centered(|ui| {
        ui.set_height(44.0);
        ui.add_space(8.0);

        // Face chips
        let _n = app.preview.detections.len();
        let _selected_faces: Vec<usize> = app.selected_faces.iter().cloned().collect();
        let mut to_toggle: Option<usize> = None;
        for (i, det) in app.preview.detections.iter().enumerate() {
            let label = format!("face_{:03} · {:.2}", i + 1, det.detection.score);
            let selected = app.selected_faces.contains(&i);
            let alt = i % 2 == 1;
            let resp = face_chip(ui, &label, selected, alt);
            if resp.clicked() {
                to_toggle = Some(i);
            }
            ui.add_space(3.0);
        }
        if let Some(i) = to_toggle {
            if app.selected_faces.contains(&i) {
                app.selected_faces.remove(&i);
            } else {
                app.selected_faces.insert(i);
            }
        }

        // Zoom controls on right
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add_space(8.0);
            zoom_btn(ui, "⛶", "Fit");
            ui.add_space(2.0);
            if zoom_btn(ui, "+", "Zoom in") {
                app.zoom = (app.zoom * 1.2).min(8.0);
            }
            ui.add_space(2.0);
            zoom_btn(ui, &format!("{:.0}%", app.zoom * 100.0), "Reset");
            ui.add_space(2.0);
            if zoom_btn(ui, "−", "Zoom out") {
                app.zoom = (app.zoom / 1.2).max(0.1);
            }
        });
    });
}

fn zoom_btn(ui: &mut egui::Ui, label: &str, _tip: &str) -> bool {
    let font = egui::FontId::monospace(10.5);
    let galley = ui
        .painter()
        .layout_no_wrap(label.to_string(), font, P::INK2);
    let w = (galley.size().x + 14.0).max(26.0);
    let (resp, painter) = ui.allocate_painter(Vec2::new(w, 26.0), Sense::click());
    let r = resp.rect;
    let bg = if resp.hovered() {
        P::white_alpha(20)
    } else {
        P::white_alpha(10)
    };
    painter.rect_filled(r, 6.0, bg);
    painter.rect_stroke(r, 6.0, Stroke::new(1.0, P::RULE), egui::StrokeKind::Outside);
    painter.galley(
        r.min + Vec2::new((w - galley.size().x) / 2.0, (26.0 - galley.size().y) / 2.0),
        galley,
        P::INK2,
    );
    resp.clicked()
}
