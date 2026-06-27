//! Centre canvas column — image viewport + face overlays + mini-log.

use crate::{
    interaction::bbox_drag::{apply_drag, hit_test_handle},
    rendering::paint::{draw_confidence_badge, draw_drag_handle, draw_face_box, draw_landmark_dot},
    theme::P,
    types::{ActiveBoxDrag, App2, DragHandle, LogKind, ManualBoxDraft, RotationDragState},
    ui::widgets::{ctl_pill, face_chip},
};
use egui::{
    Color32, CursorIcon, Frame, Pos2, Sense, Stroke, Ui, Vec2,
    epaint::{Mesh, Vertex},
};
use fcs_core::{BoundingBox, calculate_crop_region};
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
        .show(ui, |ui| {
            canvas_bottom_bar(ui, app);
        });

    egui::CentralPanel::default()
        .frame(Frame::new().fill(P::BG1))
        .show(ui, |ui| {
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

    let conf = app.settings.detection.score_threshold;
    let preset = app.settings.crop.preset.as_str();
    let face_h = app.settings.crop.face_height_pct;
    let aspect = format!(
        "{}:{}",
        app.settings.crop.output_width, app.settings.crop.output_height
    );

    // Measure the exact width the pills will need so the text clip is always tight enough.
    // ctl_pill width formula: key_w + val_w + 20  (matches the widget implementation)
    let pill_font = egui::FontId::monospace(10.5);
    let measure_pill = |key: &str, val: &str| -> f32 {
        let kw = painter
            .layout_no_wrap(key.to_string(), pill_font.clone(), P::INK3)
            .size()
            .x;
        let vw = painter
            .layout_no_wrap(val.to_string(), pill_font.clone(), P::INK)
            .size()
            .x;
        kw + vw + 20.0
    };
    let pills_total_w = measure_pill("preset ",  preset)
        + measure_pill("conf ",   &format!("{conf:.2}"))
        + measure_pill("aspect ", &aspect)
        + measure_pill("face-h ", &format!("{:.0}%", face_h))
        + 3.0 * 4.0   // three add_space(4) between pills
        + 12.0         // trailing add_space(12) on the right
        + 20.0; // extra breathing room

    // chips_x = left edge of the pills container; text is clipped before it.
    let chips_x = (r.max.x - pills_total_w).max(r.min.x + 80.0);

    // Filename + dimensions — hard clip at chips_x so they never bleed into the pills.
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
        egui::pos2(chips_x - 12.0, r.max.y),
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

    // Pills — drawn in a child UI whose max_rect exactly matches what we measured.
    let clip = egui::Rect::from_min_max(egui::pos2(chips_x, r.min.y), r.max);
    let mut child = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(clip)
            .layout(egui::Layout::right_to_left(egui::Align::Center)),
    );
    child.add_space(12.0);
    ctl_pill(&mut child, "preset ", preset, Some(P::CYAN));
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
    egui::pos2(
        rect.min.x + rx * rect.width(),
        rect.min.y + ry * rect.height(),
    )
}

/// Converts a BoundingBox in image pixel coords to an axis-aligned screen rect under rotation.
fn rotated_bbox_screen_rect(
    bx: f32,
    by: f32,
    bw: f32,
    bh: f32,
    img_size: Vec2,
    rect: egui::Rect,
    rotation_deg: f32,
) -> egui::Rect {
    let (iw, ih) = (img_size.x, img_size.y);
    let corners = [
        (bx / iw, by / ih),
        ((bx + bw) / iw, by / ih),
        ((bx + bw) / iw, (by + bh) / ih),
        (bx / iw, (by + bh) / ih),
    ];
    let pts = corners.map(|(nx, ny)| norm_to_screen_rotated(nx, ny, rect, rotation_deg));
    let min_x = pts.iter().map(|p| p.x).fold(f32::INFINITY, f32::min);
    let min_y = pts.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
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
    let rotate = |dx: f32, dy: f32| {
        egui::pos2(
            center.x + dx * cos - dy * sin,
            center.y + dx * sin + dy * cos,
        )
    };
    let pos = [
        rotate(-hw, -hh), // TL
        rotate(hw, -hh),  // TR
        rotate(hw, hh),   // BR
        rotate(-hw, hh),  // BL
    ];
    let uvs = [[0.0f32, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let mut mesh = Mesh::with_texture(texture_id);
    for (p, uv) in pos.iter().zip(uvs.iter()) {
        mesh.vertices.push(Vertex {
            pos: *p,
            uv: egui::pos2(uv[0], uv[1]),
            color: Color32::WHITE,
        });
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

/// Converts a screen-space delta to image-pixel-space delta, accounting for scale and rotation.
fn screen_delta_to_image_delta(
    screen_delta: Vec2,
    draw_rect: egui::Rect,
    img_w: f32,
    img_h: f32,
    rotation_deg: f32,
) -> Vec2 {
    let sx = screen_delta.x * img_w / draw_rect.width();
    let sy = screen_delta.y * img_h / draw_rect.height();
    let theta = rotation_deg.to_radians();
    let (sin, cos) = theta.sin_cos();
    Vec2::new(sx * cos + sy * sin, -sx * sin + sy * cos)
}

fn handle_cursor(handle: DragHandle) -> CursorIcon {
    match handle {
        DragHandle::Move => CursorIcon::Grab,
        DragHandle::NorthWest | DragHandle::SouthEast => CursorIcon::ResizeNwSe,
        DragHandle::NorthEast | DragHandle::SouthWest => CursorIcon::ResizeNeSw,
    }
}

/// Converts a screen position back to image-pixel coordinates, inverting the rotation.
fn screen_to_image_px(
    screen_pos: Pos2,
    draw_rect: egui::Rect,
    img_w: f32,
    img_h: f32,
    rotation_deg: f32,
) -> (f32, f32) {
    let rx = (screen_pos.x - draw_rect.min.x) / draw_rect.width() - 0.5;
    let ry = (screen_pos.y - draw_rect.min.y) / draw_rect.height() - 0.5;
    let theta = rotation_deg.to_radians();
    let (sin, cos) = theta.sin_cos();
    let dx = rx * cos + ry * sin;
    let dy = -rx * sin + ry * cos;
    ((dx + 0.5) * img_w, (dy + 0.5) * img_h)
}

/// Converts a ManualBoxDraft (screen positions) to a BoundingBox in image pixel coords.
fn draft_to_image_bbox(
    draft: ManualBoxDraft,
    draw_rect: egui::Rect,
    img_w: f32,
    img_h: f32,
    rotation_deg: f32,
) -> BoundingBox {
    let (ax, ay) = screen_to_image_px(draft.start, draw_rect, img_w, img_h, rotation_deg);
    let (bx, by) = screen_to_image_px(draft.current, draw_rect, img_w, img_h, rotation_deg);
    let x = ax.min(bx).max(0.0);
    let y = ay.min(by).max(0.0);
    let w = (ax - bx).abs().min(img_w - x);
    let h = (ay - by).abs().min(img_h - y);
    BoundingBox {
        x,
        y,
        width: w,
        height: h,
    }
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
    let handle = center + egui::vec2(sin * (hh + 32.0), -cos * (hh + 32.0));

    let line_color = P::white_alpha(60);
    let fill = if dragging {
        P::CYAN
    } else if hovered {
        P::INK
    } else {
        P::INK3
    };
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
    // Reserve space above the image for the rotation handle (line + circle + label clearance).
    // This keeps the handle inside the CentralPanel so interaction and painting both work.
    let handle_reserve = 48.0;
    let stage_outer = avail.shrink(pad);
    // The image is fitted into a slot that leaves handle_reserve pixels at the top of stage_outer.
    let image_slot = egui::Rect::from_min_max(
        egui::pos2(stage_outer.min.x, stage_outer.min.y + handle_reserve),
        stage_outer.max,
    );

    // fit_rect: canvas border frame, sized to the AABB of the rotated image, centred in image_slot.
    // draw_rect: original-proportion rect at the same scale — vertices are rotated around its
    //            centre, so the rotated mesh exactly fills fit_rect.
    let (fit_rect, draw_rect) = if let Some((iw, ih)) = app.preview.image_size {
        let theta = app.canvas_rotation.to_radians();
        let (sin, cos) = theta.sin_cos();
        let (asin, acos) = (sin.abs(), cos.abs());
        let iw = iw as f32;
        let ih = ih as f32;
        let aabb_w = iw * acos + ih * asin;
        let aabb_h = iw * asin + ih * acos;
        let slot_ar = image_slot.width() / image_slot.height();
        let scale = if aabb_w / aabb_h > slot_ar {
            image_slot.width() / aabb_w
        } else {
            image_slot.height() / aabb_h
        };
        let fr = egui::Rect::from_center_size(
            image_slot.center(),
            Vec2::new(aabb_w * scale, aabb_h * scale),
        );
        let dr = egui::Rect::from_center_size(
            fr.center() + app.pan,
            Vec2::new(iw * scale * app.zoom, ih * scale * app.zoom),
        );
        (fr, dr)
    } else {
        let fr = image_slot;
        let dr = egui::Rect::from_center_size(fr.center() + app.pan, fr.size() * app.zoom);
        (fr, dr)
    };

    // Scroll-to-zoom — only when the pointer is over the canvas
    let pointer_over_canvas = ui
        .ctx()
        .input(|i| i.pointer.hover_pos().is_some_and(|p| fit_rect.contains(p)));
    if pointer_over_canvas {
        let scroll = ui.ctx().input(|i| i.smooth_scroll_delta.y);
        if scroll.abs() > 0.1 {
            let factor = (1.0 + scroll * 0.003).clamp(0.85, 1.20_f32);
            app.zoom = (app.zoom * factor).clamp(0.25, 8.0);
        }
    }

    // --- Rotation handle ---
    // Allocate the hit-rect first (beats the stage pan allocation below).
    // The handle lives in the handle_reserve zone above fit_rect — always inside stage_outer,
    // so both ui.allocate_rect interaction and ui.painter drawing work normally.
    let h_resp_opt = if app.preview.texture.is_some() || app.preview.image_size.is_some() {
        let handle_pos = rotation_handle_pos(draw_rect, app.canvas_rotation);
        let handle_rect = egui::Rect::from_center_size(handle_pos, Vec2::splat(20.0));
        let h_resp = ui.allocate_rect(handle_rect, Sense::drag());

        let center = draw_rect.center();
        if h_resp.drag_started()
            && let Some(pos) = h_resp.interact_pointer_pos()
        {
            app.rotation_drag = Some(RotationDragState {
                start_mouse_angle: angle_from_center(center, pos),
                start_rotation: app.canvas_rotation,
            });
        }
        if h_resp.dragged()
            && let (Some(drag), Some(pos)) = (app.rotation_drag, h_resp.interact_pointer_pos())
        {
            let delta = angle_from_center(center, pos) - drag.start_mouse_angle;
            app.canvas_rotation = drag.start_rotation + delta;
        }
        if h_resp.drag_stopped() {
            app.rotation_drag = None;
        }
        if h_resp.hovered() || app.rotation_drag.is_some() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
        }
        Some(h_resp)
    } else {
        None
    };

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
        for (i, det) in app.preview.detections.iter().enumerate() {
            let bbox = det.active_bbox();
            let screen_rect = rotated_bbox_screen_rect(
                bbox.x,
                bbox.y,
                bbox.width,
                bbox.height,
                Vec2::new(iw, ih),
                draw_rect,
                rot,
            );

            let selected = app.selected_faces.contains(&i);
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
                format!("{:.2}", det.detection.score),
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

    // Manual box draft overlay
    if let Some(draft) = app.manual_box_draft {
        let draft_rect = egui::Rect::from_two_pos(draft.start, draft.current);
        if draft_rect.width() > 4.0 || draft_rect.height() > 4.0 {
            painter.rect_stroke(
                draft_rect,
                2.0,
                Stroke::new(2.0, P::CYAN),
                egui::StrokeKind::Outside,
            );
            painter.rect_filled(draft_rect, 2.0, P::cyan_alpha(20));
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
                let screen_rect =
                    rotated_bbox_screen_rect(rx, ry, rw, rh, Vec2::new(iw, ih), draw_rect, rot);
                painter.rect_stroke(screen_rect, 4.0, crop_stroke, egui::StrokeKind::Inside);
            }
        }
    }

    // Delete key removes selected faces
    let delete_pressed = ui
        .ctx()
        .input(|i| i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace));
    if delete_pressed && !app.selected_faces.is_empty() {
        app.delete_selected_faces();
    }

    // Allocate stage area for click (face selection/draw) and drag (pan/draw)
    let resp = ui.allocate_rect(fit_rect, Sense::click_and_drag());

    // Draw-box tool: crosshair cursor + drag to draw
    if app.manual_box_tool_enabled && app.preview.image_size.is_some() {
        if resp.hovered() || app.manual_box_draft.is_some() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
        }
        if let Some(pos) = resp.interact_pointer_pos() {
            if resp.drag_started() {
                app.manual_box_draft = Some(ManualBoxDraft {
                    start: pos,
                    current: pos,
                });
            }
            if resp.dragged()
                && app.manual_box_draft.is_some()
                && let Some(draft) = app.manual_box_draft.as_mut()
            {
                draft.current = pos;
            }
        }
        if resp.drag_stopped()
            && let Some(draft) = app.manual_box_draft.take()
        {
            let w = (draft.start.x - draft.current.x).abs();
            let h = (draft.start.y - draft.current.y).abs();
            if w > 8.0
                && h > 8.0
                && let Some((img_w, img_h)) = app.preview.image_size
            {
                let bbox = draft_to_image_bbox(
                    draft,
                    draw_rect,
                    img_w as f32,
                    img_h as f32,
                    app.canvas_rotation,
                );
                if bbox.width > 1.0 && bbox.height > 1.0 {
                    app.commit_manual_box(bbox);
                }
            }
        }
    } else {
        // Normal mode: bbox resize on handle drag, pan otherwise, select on click.
        let (iw, ih) = app
            .preview
            .image_size
            .map(|(w, h)| (w as f32, h as f32))
            .unwrap_or((1.0, 1.0));
        let rot = app.canvas_rotation;

        // Hover cursor: show resize/grab over handles of selected faces
        if app.active_bbox_drag.is_none() {
            if let Some(hover_pos) = ui.ctx().input(|i| i.pointer.hover_pos()) {
                let mut found_handle = false;
                for &sel_i in &app.selected_faces {
                    if let Some(det) = app.preview.detections.get(sel_i) {
                        let bbox = det.active_bbox();
                        let sr = rotated_bbox_screen_rect(
                            bbox.x,
                            bbox.y,
                            bbox.width,
                            bbox.height,
                            Vec2::new(iw, ih),
                            draw_rect,
                            rot,
                        );
                        if let Some(h) = hit_test_handle(sr, hover_pos) {
                            ui.ctx().set_cursor_icon(handle_cursor(h));
                            found_handle = true;
                            break;
                        }
                    }
                }
                let _ = found_handle;
            }
        } else {
            if let Some(drag) = &app.active_bbox_drag {
                ui.ctx().set_cursor_icon(handle_cursor(drag.handle));
            }
        }

        // Start a bbox drag when the user presses on a handle of a selected face
        if resp.drag_started()
            && app.rotation_drag.is_none()
            && let Some(press_pos) = resp.interact_pointer_pos()
        {
            let mut started = false;
            for &sel_i in &app.selected_faces {
                if let Some(det) = app.preview.detections.get(sel_i) {
                    let bbox = det.active_bbox();
                    let sr = rotated_bbox_screen_rect(
                        bbox.x,
                        bbox.y,
                        bbox.width,
                        bbox.height,
                        Vec2::new(iw, ih),
                        draw_rect,
                        rot,
                    );
                    if let Some(handle) = hit_test_handle(sr, press_pos) {
                        app.active_bbox_drag = Some(ActiveBoxDrag {
                            index: sel_i,
                            handle,
                            start_bbox: bbox,
                            drag_start_screen: press_pos,
                        });
                        started = true;
                        break;
                    }
                }
            }
            let _ = started;
        }

        // Apply bbox drag or pan
        if resp.dragged() && app.rotation_drag.is_none() {
            if let Some(drag) = app.active_bbox_drag {
                if let Some(cur_pos) = resp.interact_pointer_pos() {
                    let screen_delta = cur_pos - drag.drag_start_screen;
                    let image_delta =
                        screen_delta_to_image_delta(screen_delta, draw_rect, iw, ih, rot);
                    let new_bbox = apply_drag(&drag, image_delta, iw, ih);
                    if let Some(det) = app.preview.detections.get_mut(drag.index) {
                        det.set_bbox(new_bbox);
                    }
                }
            } else {
                app.pan += resp.drag_delta();
            }
        }

        if resp.drag_stopped() {
            app.active_bbox_drag = None;
        }

        // Click: select / deselect faces
        if resp.clicked()
            && let Some(pos) = resp.interact_pointer_pos()
            && app.preview.image_size.is_some()
        {
            let mut clicked_any = false;
            for (i, det) in app.preview.detections.iter().enumerate() {
                let bbox = det.active_bbox();
                let sr = rotated_bbox_screen_rect(
                    bbox.x,
                    bbox.y,
                    bbox.width,
                    bbox.height,
                    Vec2::new(iw, ih),
                    draw_rect,
                    rot,
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

        // Right-click on a face box to remove it
        if ui.ctx().input(|i| i.pointer.secondary_clicked())
            && let Some(pos) = ui.ctx().input(|i| i.pointer.interact_pos())
            && app.preview.image_size.is_some()
        {
            let mut hit_idx: Option<usize> = None;
            for (i, det) in app.preview.detections.iter().enumerate() {
                let bbox = det.active_bbox();
                let sr = rotated_bbox_screen_rect(
                    bbox.x,
                    bbox.y,
                    bbox.width,
                    bbox.height,
                    Vec2::new(iw, ih),
                    draw_rect,
                    rot,
                );
                if sr.expand(4.0).contains(pos) {
                    hit_idx = Some(i);
                    break;
                }
            }
            if let Some(i) = hit_idx {
                app.selected_faces.clear();
                app.selected_faces.insert(i);
                app.delete_selected_faces();
            }
        }
    }

    // Mini-log overlay
    mini_log_overlay(ui, app, fit_rect);

    // Draw the rotation handle last (on top of image + overlays).
    // The handle lives inside stage_outer, so the panel painter clips it correctly.
    if let Some(h_resp) = h_resp_opt {
        let handle_painter = ui.painter().with_clip_rect(stage_outer);
        let dragging = app.rotation_drag.is_some();
        draw_rotation_handle(
            &handle_painter,
            draw_rect,
            app.canvas_rotation,
            h_resp.hovered(),
            dragging,
        );
    }
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
    for (i, line) in app.log_lines.iter().skip(start).enumerate() {
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
        let mut to_toggle: Option<usize> = None;
        for (i, det) in app.preview.detections.iter().enumerate() {
            let label = format!("face_{:03} · {:.2}", i + 1, det.detection.score);
            let selected = app.selected_faces.contains(&i);
            let alt = i % 2 == 1;
            let resp = face_chip(ui, label, selected, alt);
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
            zoom_btn(ui, format!("{:.0}%", app.zoom * 100.0), "Reset");
            ui.add_space(2.0);
            if zoom_btn(ui, "−", "Zoom out") {
                app.zoom = (app.zoom / 1.2).max(0.1);
            }
        });
    });
}

fn zoom_btn(ui: &mut egui::Ui, label: impl Into<String>, _tip: &str) -> bool {
    let font = egui::FontId::monospace(10.5);
    let galley = ui.painter().layout_no_wrap(label.into(), font, P::INK2);
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
