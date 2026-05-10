//! Centre canvas column — image viewport + face overlays + mini-log.

use crate::rendering::paint::{
    draw_confidence_badge, draw_drag_handle, draw_face_box, draw_landmark_dot,
};
use crate::theme::P;
use crate::types::{App2, LogKind, RotationDragState};
use crate::ui::widgets::{ctl_pill, face_chip};
use egui::{Color32, Frame, Sense, Stroke, Ui, Vec2};
use egui::epaint::{Mesh, Vertex};
use fcs_core::calculate_crop_region;
use fcs_utils::outline_points_for_rect;

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
        .show_separator_line(false)
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

    // Control chips on the right
    let chips_x = r.max.x - 360.0;

    // Filename + dimensions — clipped so they never overlap with the pills
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

    let text_clip = egui::Rect::from_min_max(
        egui::pos2(r.min.x, r.min.y),
        egui::pos2(chips_x - 10.0, r.max.y),
    );
    let left_painter = painter.with_clip_rect(text_clip);

    left_painter.text(
        egui::pos2(r.min.x + 14.0, r.center().y),
        egui::Align2::LEFT_CENTER,
        name,
        egui::FontId::proportional(13.0),
        P::INK,
    );
    if !dims.is_empty() {
        let name_w = left_painter
            .layout_no_wrap(name.to_string(), egui::FontId::proportional(13.0), P::INK)
            .size()
            .x;
        left_painter.text(
            egui::pos2(r.min.x + 14.0 + name_w + 12.0, r.center().y),
            egui::Align2::LEFT_CENTER,
            &dims,
            egui::FontId::monospace(11.0),
            P::INK3,
        );
    }
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

/// Maps a normalized image coordinate (0..1) to screen space under a CW rotation (degrees).
/// `rect` is sized to the *original* (unrotated) image proportions.
fn norm_to_screen_rotated(nx: f32, ny: f32, rect: egui::Rect, rotation_deg: f32) -> egui::Pos2 {
    let dx = nx - 0.5;
    let dy = ny - 0.5;
    let theta = rotation_deg.to_radians();
    let (sin, cos) = theta.sin_cos();
    let rx = dx * cos - dy * sin + 0.5;
    let ry = dx * sin + dy * cos + 0.5;
    egui::pos2(rect.min.x + rx * rect.width(), rect.min.y + ry * rect.height())
}

/// Converts a BoundingBox in image pixel coords to an axis-aligned screen rect under rotation.
fn rotated_bbox_screen_rect(
    bx: f32, by: f32, bw: f32, bh: f32,
    iw: f32, ih: f32,
    rect: egui::Rect,
    rotation_deg: f32,
) -> egui::Rect {
    let corners = [
        (bx / iw,        by / ih),
        ((bx + bw) / iw, by / ih),
        ((bx + bw) / iw, (by + bh) / ih),
        (bx / iw,        (by + bh) / ih),
    ];
    let pts: Vec<egui::Pos2> = corners
        .iter()
        .map(|(nx, ny)| norm_to_screen_rotated(*nx, *ny, rect, rotation_deg))
        .collect();
    let min_x = pts.iter().map(|p| p.x).fold(f32::INFINITY,     f32::min);
    let min_y = pts.iter().map(|p| p.y).fold(f32::INFINITY,     f32::min);
    let max_x = pts.iter().map(|p| p.x).fold(f32::NEG_INFINITY, f32::max);
    let max_y = pts.iter().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max);
    egui::Rect::from_min_max(egui::pos2(min_x, min_y), egui::pos2(max_x, max_y))
}

/// Builds a textured mesh quad for `dest` (original image proportions) rotated CW by `rotation_deg`.
/// Vertices are rotated around the rect center; UVs are always the standard 0→1 mapping.
fn image_shape(texture_id: egui::TextureId, dest: egui::Rect, rotation_deg: f32) -> egui::Shape {
    let center = dest.center();
    let hw = dest.width() / 2.0;
    let hh = dest.height() / 2.0;
    let theta = rotation_deg.to_radians();
    let (sin, cos) = theta.sin_cos();
    let rotate = |dx: f32, dy: f32| egui::pos2(
        center.x + dx * cos - dy * sin,
        center.y + dx * sin + dy * cos,
    );
    let pos = [
        rotate(-hw, -hh), // TL
        rotate( hw, -hh), // TR
        rotate( hw,  hh), // BR
        rotate(-hw,  hh), // BL
    ];
    let uvs = [[0.0f32, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let mut mesh = Mesh::with_texture(texture_id);
    for (p, uv) in pos.iter().zip(uvs.iter()) {
        mesh.vertices.push(Vertex { pos: *p, uv: egui::pos2(uv[0], uv[1]), color: Color32::WHITE });
    }
    mesh.indices = vec![0, 1, 2, 0, 2, 3];
    egui::Shape::mesh(mesh)
}

/// Position of the rotation handle above the image's current top edge.
fn rotation_handle_pos(draw_rect: egui::Rect, rotation_deg: f32) -> egui::Pos2 {
    let hh = draw_rect.height() / 2.0;
    let theta = rotation_deg.to_radians();
    let (sin, cos) = theta.sin_cos();
    // unit vector pointing "up" through the rotated frame
    draw_rect.center() + egui::vec2(sin * (hh + 32.0), -cos * (hh + 32.0))
}

/// Angle (degrees) from image center to mouse position, measuring CW from "up".
fn angle_from_center(center: egui::Pos2, pos: egui::Pos2) -> f32 {
    let dx = pos.x - center.x;
    let dy = pos.y - center.y;
    dx.atan2(-dy).to_degrees()
}

fn draw_rotation_handle(
    painter: &egui::Painter,
    draw_rect: egui::Rect,
    rotation_deg: f32,
    hovered: bool,
    dragging: bool,
) {
    let hh = draw_rect.height() / 2.0;
    let theta = rotation_deg.to_radians();
    let (sin, cos) = theta.sin_cos();
    let center = draw_rect.center();
    let top_edge = center + egui::vec2(sin * hh, -cos * hh);
    let handle   = center + egui::vec2(sin * (hh + 32.0), -cos * (hh + 32.0));

    let line_color = P::white_alpha(60);
    let fill = if dragging { P::CYAN } else if hovered { P::INK } else { P::INK3 };
    painter.line_segment([top_edge, handle], Stroke::new(1.0, line_color));
    painter.circle(handle, 7.0, fill, Stroke::new(1.5, P::white_alpha(120)));

    if rotation_deg.abs() > 0.5 {
        painter.text(
            handle + egui::vec2(12.0, 0.0),
            egui::Align2::LEFT_CENTER,
            format!("{:.1}°", rotation_deg),
            egui::FontId::proportional(10.0),
            P::INK2,
        );
    }
}

fn stage(ui: &mut Ui, app: &mut App2) {
    let avail = ui.available_rect_before_wrap();
    let pad = 18.0;
    let stage_outer = avail.shrink(pad);

    // Scroll-to-zoom
    let scroll = ui.ctx().input(|i| i.smooth_scroll_delta.y);
    if scroll.abs() > 0.1 {
        let factor = (1.0 + scroll * 0.003).clamp(0.85, 1.20_f32);
        app.zoom = (app.zoom * factor).clamp(0.25, 8.0);
    }

    // Compute fit_rect (zoom=1, pan=0) from the AABB of the rotated image.
    // draw_rect uses the *original* image proportions at the same scale — vertices are
    // then rotated around its center, so the visual result fills fit_rect exactly.
    let (fit_rect, draw_rect) = if let Some((iw, ih)) = app.preview.image_size {
        let theta = app.canvas_rotation.to_radians();
        let (sin, cos) = theta.sin_cos();
        let (asin, acos) = (sin.abs(), cos.abs());
        let iw = iw as f32;
        let ih = ih as f32;
        // AABB of rotated image
        let aabb_w = iw * acos + ih * asin;
        let aabb_h = iw * asin + ih * acos;
        let slot_ar = stage_outer.width() / stage_outer.height();
        let scale = if aabb_w / aabb_h > slot_ar {
            stage_outer.width() / aabb_w
        } else {
            stage_outer.height() / aabb_h
        };
        let fr = egui::Rect::from_center_size(
            stage_outer.center(),
            Vec2::new(aabb_w * scale, aabb_h * scale),
        );
        let dr = egui::Rect::from_center_size(
            fr.center() + app.pan,
            Vec2::new(iw * scale * app.zoom, ih * scale * app.zoom),
        );
        (fr, dr)
    } else {
        let fr = stage_outer;
        let dr = egui::Rect::from_center_size(fr.center() + app.pan, fr.size() * app.zoom);
        (fr, dr)
    };

    // --- Rotation handle (drawn + interacted before stage area so it gets event priority) ---
    let unclipped = ui.painter().clone();
    if app.preview.texture.is_some() || app.preview.image_size.is_some() {
        let handle_pos = rotation_handle_pos(draw_rect, app.canvas_rotation);
        let handle_rect = egui::Rect::from_center_size(handle_pos, Vec2::splat(20.0));
        let h_resp = ui.allocate_rect(handle_rect, Sense::drag());

        let center = draw_rect.center();
        if h_resp.drag_started() {
            if let Some(pos) = h_resp.interact_pointer_pos() {
                app.rotation_drag = Some(RotationDragState {
                    start_mouse_angle: angle_from_center(center, pos),
                    start_rotation: app.canvas_rotation,
                });
            }
        }
        if h_resp.dragged() {
            if let (Some(drag), Some(pos)) = (&app.rotation_drag.clone(), h_resp.interact_pointer_pos()) {
                let delta = angle_from_center(center, pos) - drag.start_mouse_angle;
                app.canvas_rotation = drag.start_rotation + delta;
            }
        }
        if h_resp.drag_stopped() {
            app.rotation_drag = None;
        }

        let dragging = app.rotation_drag.is_some();
        draw_rotation_handle(&unclipped, draw_rect, app.canvas_rotation, h_resp.hovered(), dragging);
        if h_resp.hovered() || dragging {
            ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
        }
    }

    // Stage background + border
    let painter = ui.painter();
    painter.rect_filled(fit_rect, 12.0, P::BG);
    painter.rect_stroke(
        fit_rect,
        12.0,
        Stroke::new(1.0, P::RULE),
        egui::StrokeKind::Outside,
    );

    // Clip image/overlay painting to fit_rect
    let painter = painter.with_clip_rect(fit_rect);

    if let Some(texture) = &app.preview.texture {
        painter.add(image_shape(texture.id(), draw_rect, app.canvas_rotation));
    } else if app.is_busy {
        painter.text(
            fit_rect.center(),
            egui::Align2::CENTER_CENTER,
            "Detecting faces…",
            egui::FontId::proportional(14.0),
            P::INK3,
        );
    } else {
        painter.text(
            fit_rect.center() - Vec2::new(0.0, 12.0),
            egui::Align2::CENTER_CENTER,
            "Drop an image to begin",
            egui::FontId::proportional(14.0),
            P::INK3,
        );
        painter.text(
            fit_rect.center() + Vec2::new(0.0, 12.0),
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
        let rot = app.canvas_rotation;
        let dets: Vec<_> = app.preview.detections.iter().enumerate().collect();

        for (i, det) in &dets {
            let bbox = det.active_bbox();
            let screen_rect = rotated_bbox_screen_rect(
                bbox.x, bbox.y, bbox.width, bbox.height, iw, ih, draw_rect, rot,
            );

            let selected = app.selected_faces.contains(i);
            let color = if !selected {
                P::INK3
            } else if i % 2 == 0 {
                P::PEACH
            } else {
                P::CYAN
            };

            draw_face_box(&painter, screen_rect, color, selected);
            draw_confidence_badge(
                &painter,
                &format!("{:.2}", det.detection.score),
                screen_rect,
                color,
            );

            for lm in &det.detection.landmarks {
                let lm_pos = norm_to_screen_rotated(lm.x / iw, lm.y / ih, draw_rect, rot);
                draw_landmark_dot(&painter, lm_pos);
            }

            if selected {
                let corners = [
                    screen_rect.min,
                    egui::pos2(screen_rect.max.x, screen_rect.min.y),
                    egui::pos2(screen_rect.min.x, screen_rect.max.y),
                    screen_rect.max,
                ];
                for corner in corners {
                    draw_drag_handle(&painter, corner, color);
                }
            }
        }
    }

    // Crop region + shape outline overlay
    if app.show_crop_overlay
        && let Some((img_w, img_h)) = app.preview.image_size
    {
        let crop_settings = app.build_crop_settings();
        let crop_stroke = Stroke::new(2.0, P::LIME);
        let iw = img_w as f32;
        let ih = img_h as f32;
        let rot = app.canvas_rotation;
        for det in &app.preview.detections {
            let bbox = det.active_bbox();
            let region = calculate_crop_region(img_w, img_h, bbox, &crop_settings);
            let rx = region.x as f32;
            let ry = region.y as f32;
            let rw = region.width as f32;
            let rh = region.height as f32;

            if rot.abs() < 0.5 {
                // Full shape-outline support at 0°
                let sx = draw_rect.min.x + (rx / iw) * draw_rect.width();
                let sy = draw_rect.min.y + (ry / ih) * draw_rect.height();
                let sw = (rw / iw) * draw_rect.width();
                let sh = (rh / ih) * draw_rect.height();
                let crop_rect = egui::Rect::from_min_size(egui::pos2(sx, sy), Vec2::new(sw, sh));
                let shape_pts = outline_points_for_rect(sw, sh, &app.settings.crop.shape);
                if shape_pts.len() >= 2 {
                    let outline: Vec<egui::Pos2> = shape_pts
                        .iter()
                        .map(|(x, y)| egui::pos2(crop_rect.min.x + x, crop_rect.min.y + y))
                        .collect();
                    painter.add(egui::Shape::closed_line(outline, crop_stroke));
                } else {
                    painter.rect_stroke(crop_rect, 4.0, crop_stroke, egui::StrokeKind::Inside);
                }
            } else {
                let screen_rect = rotated_bbox_screen_rect(rx, ry, rw, rh, iw, ih, draw_rect, rot);
                painter.rect_stroke(screen_rect, 4.0, crop_stroke, egui::StrokeKind::Inside);
            }
        }
    }

    // Allocate stage area for click (face selection) and drag (pan)
    let resp = ui.allocate_rect(fit_rect, Sense::click_and_drag());

    if resp.dragged() && app.rotation_drag.is_none() {
        app.pan += resp.drag_delta();
    }

    if resp.clicked()
        && let Some(pos) = resp.interact_pointer_pos()
        && let Some((img_w, img_h)) = app.preview.image_size
    {
        let iw = img_w as f32;
        let ih = img_h as f32;
        let rot = app.canvas_rotation;
        let mut clicked_any = false;
        for (i, det) in app.preview.detections.iter().enumerate() {
            let bbox = det.active_bbox();
            let sr = rotated_bbox_screen_rect(
                bbox.x, bbox.y, bbox.width, bbox.height, iw, ih, draw_rect, rot,
            );
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

    // Mini-log overlay
    mini_log_overlay(ui, app, fit_rect);
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

    let msg_x = log_rect.min.x + 62.0;
    let msg_clip = egui::Rect::from_min_max(
        egui::pos2(msg_x, log_rect.min.y),
        egui::pos2(log_rect.max.x - pad, log_rect.max.y),
    );
    let msg_painter = painter.with_clip_rect(msg_clip);

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
        msg_painter.text(
            egui::pos2(msg_x, y),
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
            if zoom_btn(ui, "⛶", "Fit") {
                app.zoom = 1.0;
                app.pan = egui::Vec2::ZERO;
            }
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
