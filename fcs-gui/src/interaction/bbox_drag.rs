//! Bounding box drag/resize interaction.

use crate::types::{ActiveBoxDrag, DragHandle};
use egui::{Pos2, Rect};
use fcs_core::BoundingBox;

pub const HANDLE_SIZE: f32 = 9.0;

/// Returns which drag handle (if any) is under `pos` for a bounding box `rect`.
pub fn hit_test_handle(rect: Rect, pos: Pos2) -> Option<DragHandle> {
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
            move_left_edge(&mut bbox, delta_px.x);
            move_top_edge(&mut bbox, delta_px.y);
        }
        DragHandle::NorthEast => {
            move_top_edge(&mut bbox, delta_px.y);
            extend_right_edge(&mut bbox, delta_px.x, img_w);
        }
        DragHandle::SouthWest => {
            move_left_edge(&mut bbox, delta_px.x);
            extend_bottom_edge(&mut bbox, delta_px.y, img_h);
        }
        DragHandle::SouthEast => {
            extend_right_edge(&mut bbox, delta_px.x, img_w);
            extend_bottom_edge(&mut bbox, delta_px.y, img_h);
        }
    }
    bbox
}

/// Minimum width/height a corner drag is allowed to produce; corners cannot
/// cross over to invert the box.
const MIN_BBOX_EXTENT: f32 = 10.0;

fn move_left_edge(bbox: &mut BoundingBox, delta_x: f32) {
    let new_x = (bbox.x + delta_x).clamp(0.0, bbox.x + bbox.width - MIN_BBOX_EXTENT);
    bbox.width += bbox.x - new_x;
    bbox.x = new_x;
}

fn move_top_edge(bbox: &mut BoundingBox, delta_y: f32) {
    let new_y = (bbox.y + delta_y).clamp(0.0, bbox.y + bbox.height - MIN_BBOX_EXTENT);
    bbox.height += bbox.y - new_y;
    bbox.y = new_y;
}

fn extend_right_edge(bbox: &mut BoundingBox, delta_x: f32, img_w: f32) {
    bbox.width = (bbox.width + delta_x).clamp(MIN_BBOX_EXTENT, img_w - bbox.x);
}

fn extend_bottom_edge(bbox: &mut BoundingBox, delta_y: f32, img_h: f32) {
    bbox.height = (bbox.height + delta_y).clamp(MIN_BBOX_EXTENT, img_h - bbox.y);
}

#[cfg(test)]
mod tests {
    use super::*;

    const IMG_W: f32 = 200.0;
    const IMG_H: f32 = 100.0;

    /// A box with room to move in every direction, so a clamp firing in a test
    /// means the code clamped, not that the fixture was already at the edge.
    fn bbox() -> BoundingBox {
        BoundingBox {
            x: 50.0,
            y: 30.0,
            width: 40.0,
            height: 20.0,
        }
    }

    fn drag(handle: DragHandle) -> ActiveBoxDrag {
        ActiveBoxDrag {
            index: 0,
            handle,
            start_bbox: bbox(),
            drag_start_screen: Pos2::ZERO,
        }
    }

    fn dragged(handle: DragHandle, dx: f32, dy: f32) -> BoundingBox {
        apply_drag(&drag(handle), egui::vec2(dx, dy), IMG_W, IMG_H)
    }

    fn rect() -> Rect {
        Rect::from_min_max(Pos2::new(20.0, 20.0), Pos2::new(120.0, 80.0))
    }

    // ── hit_test_handle ──────────────────────────────────────────────────

    #[test]
    fn each_corner_hits_its_own_handle() {
        let r = rect();
        for (pos, expected) in [
            (r.min, DragHandle::NorthWest),
            (Pos2::new(r.max.x, r.min.y), DragHandle::NorthEast),
            (Pos2::new(r.min.x, r.max.y), DragHandle::SouthWest),
            (r.max, DragHandle::SouthEast),
        ] {
            assert_eq!(hit_test_handle(r, pos), Some(expected), "at {pos:?}");
        }
    }

    #[test]
    fn corner_handles_extend_past_the_rect_edge() {
        // The handle box is HANDLE_SIZE + 4 wide and centred on the corner, so
        // half of it hangs outside the rect and must still be grabbable.
        let r = rect();
        let reach = (HANDLE_SIZE + 4.0) / 2.0 - 0.5;
        assert_eq!(
            hit_test_handle(r, r.min - egui::vec2(reach, reach)),
            Some(DragHandle::NorthWest),
        );
        assert_eq!(
            hit_test_handle(r, r.min - egui::vec2(reach + 2.0, reach + 2.0)),
            None,
            "past the handle box is a miss, not a corner",
        );
    }

    #[test]
    fn the_interior_moves_and_the_outside_hits_nothing() {
        let r = rect();
        assert_eq!(hit_test_handle(r, r.center()), Some(DragHandle::Move));
        assert_eq!(hit_test_handle(r, Pos2::new(500.0, 500.0)), None);
    }

    #[test]
    fn corners_win_over_move_where_they_overlap() {
        // Just inside the top-left corner is both within the shrunken interior
        // test and within the corner handle; the corner has to be checked first
        // or the corners become unusable.
        let r = rect();
        assert_eq!(
            hit_test_handle(r, r.min + egui::vec2(4.5, 4.5)),
            Some(DragHandle::NorthWest),
        );
    }

    // ── apply_drag ───────────────────────────────────────────────────────

    #[test]
    fn move_translates_without_resizing() {
        let moved = dragged(DragHandle::Move, 10.0, -5.0);
        assert_eq!((moved.x, moved.y), (60.0, 25.0));
        assert_eq!((moved.width, moved.height), (40.0, 20.0));
    }

    #[test]
    fn move_clamps_to_the_image_and_keeps_the_box_size() {
        let far = dragged(DragHandle::Move, 1000.0, 1000.0);
        assert_eq!((far.x, far.y), (IMG_W - 40.0, IMG_H - 20.0));
        assert_eq!((far.width, far.height), (40.0, 20.0));

        let near = dragged(DragHandle::Move, -1000.0, -1000.0);
        assert_eq!((near.x, near.y), (0.0, 0.0));
        assert_eq!((near.width, near.height), (40.0, 20.0));
    }

    #[test]
    fn dragging_a_corner_moves_only_the_edges_it_owns() {
        let nw = dragged(DragHandle::NorthWest, -10.0, -5.0);
        assert_eq!((nw.x, nw.y), (40.0, 25.0), "origin follows the corner");
        assert_eq!((nw.width, nw.height), (50.0, 25.0), "and the box grows");
        assert_eq!(nw.x + nw.width, 90.0, "the opposite edge stays put");

        let se = dragged(DragHandle::SouthEast, 10.0, 5.0);
        assert_eq!((se.x, se.y), (50.0, 30.0), "origin is untouched");
        assert_eq!((se.width, se.height), (50.0, 25.0));

        let ne = dragged(DragHandle::NorthEast, 10.0, -5.0);
        assert_eq!((ne.x, ne.y), (50.0, 25.0), "x fixed, y follows");
        assert_eq!((ne.width, ne.height), (50.0, 25.0));

        let sw = dragged(DragHandle::SouthWest, -10.0, 5.0);
        assert_eq!((sw.x, sw.y), (40.0, 30.0), "x follows, y fixed");
        assert_eq!((sw.width, sw.height), (50.0, 25.0));
    }

    #[test]
    fn corners_cannot_be_dragged_past_the_opposite_edge() {
        // Every corner collapses to MIN_BBOX_EXTENT rather than inverting.
        for handle in [
            DragHandle::NorthWest,
            DragHandle::NorthEast,
            DragHandle::SouthWest,
            DragHandle::SouthEast,
        ] {
            let sign = match handle {
                DragHandle::NorthWest => (1.0, 1.0),
                DragHandle::NorthEast => (-1.0, 1.0),
                DragHandle::SouthWest => (1.0, -1.0),
                _ => (-1.0, -1.0),
            };
            let out = apply_drag(
                &drag(handle),
                egui::vec2(sign.0 * 1000.0, sign.1 * 1000.0),
                IMG_W,
                IMG_H,
            );
            assert_eq!(out.width, MIN_BBOX_EXTENT, "{handle:?} width collapsed");
            assert_eq!(out.height, MIN_BBOX_EXTENT, "{handle:?} height collapsed");
            assert!(out.x >= 0.0 && out.y >= 0.0, "{handle:?} left the image");
        }
    }

    #[test]
    fn growing_a_corner_stops_at_the_image_bounds() {
        let se = dragged(DragHandle::SouthEast, 1000.0, 1000.0);
        assert_eq!(se.x + se.width, IMG_W);
        assert_eq!(se.y + se.height, IMG_H);

        let nw = dragged(DragHandle::NorthWest, -1000.0, -1000.0);
        assert_eq!((nw.x, nw.y), (0.0, 0.0));
        assert_eq!(nw.x + nw.width, 90.0, "the anchored edge does not move");
        assert_eq!(nw.y + nw.height, 50.0);
    }
}
