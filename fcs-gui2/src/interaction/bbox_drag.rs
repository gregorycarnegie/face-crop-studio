//! Bounding box drag/resize interaction.

use crate::types::{ActiveBoxDrag, DragHandle};
use egui::{Pos2, Rect};
use fcs_core::BoundingBox;

pub const HANDLE_SIZE: f32 = 9.0;

/// Returns which drag handle (if any) is under `pos` for a bounding box `rect`.
pub fn hit_test_handle(rect: Rect, pos: Pos2) -> Option<DragHandle> {
    let _hs = HANDLE_SIZE / 2.0;
    let corners = [
        (DragHandle::NorthWest, Pos2::new(rect.min.x, rect.min.y)),
        (DragHandle::NorthEast, Pos2::new(rect.max.x, rect.min.y)),
        (DragHandle::SouthWest, Pos2::new(rect.min.x, rect.max.y)),
        (DragHandle::SouthEast, Pos2::new(rect.max.x, rect.max.y)),
    ];
    for (handle, corner) in corners {
        let h_rect = Rect::from_center_size(corner, egui::Vec2::splat(HANDLE_SIZE + 4.0));
        if h_rect.contains(pos) {
            return Some(handle);
        }
    }
    if rect.shrink(4.0).contains(pos) {
        return Some(DragHandle::Move);
    }
    None
}

/// Apply a drag delta to a bounding box in image pixel space.
pub fn apply_drag(
    drag: &ActiveBoxDrag,
    delta_px: egui::Vec2,
    img_w: f32,
    img_h: f32,
) -> BoundingBox {
    let mut bbox = drag.start_bbox;
    match drag.handle {
        DragHandle::Move => {
            bbox.x = (bbox.x + delta_px.x).clamp(0.0, img_w - bbox.width);
            bbox.y = (bbox.y + delta_px.y).clamp(0.0, img_h - bbox.height);
        }
        DragHandle::NorthWest => {
            let new_x = (bbox.x + delta_px.x).clamp(0.0, bbox.x + bbox.width - 10.0);
            let new_y = (bbox.y + delta_px.y).clamp(0.0, bbox.y + bbox.height - 10.0);
            bbox.width += bbox.x - new_x;
            bbox.height += bbox.y - new_y;
            bbox.x = new_x;
            bbox.y = new_y;
        }
        DragHandle::NorthEast => {
            let new_y = (bbox.y + delta_px.y).clamp(0.0, bbox.y + bbox.height - 10.0);
            bbox.height += bbox.y - new_y;
            bbox.y = new_y;
            bbox.width = (bbox.width + delta_px.x).clamp(10.0, img_w - bbox.x);
        }
        DragHandle::SouthWest => {
            let new_x = (bbox.x + delta_px.x).clamp(0.0, bbox.x + bbox.width - 10.0);
            bbox.width += bbox.x - new_x;
            bbox.x = new_x;
            bbox.height = (bbox.height + delta_px.y).clamp(10.0, img_h - bbox.y);
        }
        DragHandle::SouthEast => {
            bbox.width = (bbox.width + delta_px.x).clamp(10.0, img_w - bbox.x);
            bbox.height = (bbox.height + delta_px.y).clamp(10.0, img_h - bbox.y);
        }
    }
    bbox
}
