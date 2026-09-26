//! What the pointer is doing on the canvas.
//!
//! One gesture at a time, by construction. This replaced three `Option` fields -- a rotation
//! drag, a box drag and a draw-tool draft -- whose exclusivity rested on `is_none()` guards
//! scattered through the canvas, with panning implied by all three being empty. New gestures
//! are new variants: the compiler then points at every `match` that has to decide about them.

use egui::{Pos2, Rect};

use super::bbox_drag::hit_test_handle;
use crate::types::{ActiveBoxDrag, ManualBoxDraft, RotationDragState};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum CanvasGesture {
    #[default]
    Idle,
    /// Dragging the image around the stage.
    Panning,
    /// Moving or resizing a selected face's box by a handle.
    ResizingBox(ActiveBoxDrag),
    /// Turning the image by the rotation handle.
    Rotating(RotationDragState),
    /// Drawing a new face box with the draw tool.
    DrawingBox(ManualBoxDraft),
}

/// A selected face, as the canvas hit-tests it: its index and its box on screen.
pub struct FaceTarget {
    pub index: usize,
    pub screen_rect: Rect,
    pub bbox: fcs_core::BoundingBox,
}

impl CanvasGesture {
    /// The gesture a drag on the stage (not the rotation handle) starts at `pos`.
    ///
    /// With the draw tool on it always draws. Otherwise a handle of a selected face wins --
    /// the first face in `selected` that has one under the pointer -- and anything else pans.
    pub fn begin_stage_drag(
        pos: Pos2,
        draw_tool: bool,
        selected: impl IntoIterator<Item = FaceTarget>,
    ) -> Self {
        if draw_tool {
            return Self::DrawingBox(ManualBoxDraft {
                start: pos,
                current: pos,
            });
        }
        selected
            .into_iter()
            .find_map(|face| {
                hit_test_handle(face.screen_rect, pos).map(|handle| {
                    Self::ResizingBox(ActiveBoxDrag {
                        index: face.index,
                        handle,
                        start_bbox: face.bbox,
                        drag_start_screen: pos,
                    })
                })
            })
            .unwrap_or(Self::Panning)
    }

    /// End the gesture once no button is held.
    ///
    /// Run after the frame's input is handled, so a release the canvas did see still reaches its
    /// `drag_stopped` branch first. This catches the ones it did not: a release while the result
    /// preview was showing, or with the pointer outside the window. Without it the stale gesture
    /// hijacked the next drag -- a pan moved the old box from where the old drag began.
    pub fn settle(&mut self, any_button_down: bool) {
        if !any_button_down {
            *self = Self::Idle;
        }
    }

    pub fn is_idle(&self) -> bool {
        matches!(self, Self::Idle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DragHandle;
    use fcs_core::BoundingBox;

    fn face(index: usize, rect: Rect) -> FaceTarget {
        FaceTarget {
            index,
            screen_rect: rect,
            bbox: BoundingBox {
                x: 0.0,
                y: 0.0,
                width: 10.0,
                height: 10.0,
            },
        }
    }

    const BOX: Rect = Rect {
        min: Pos2::new(100.0, 100.0),
        max: Pos2::new(200.0, 200.0),
    };

    #[test]
    fn a_handle_of_a_selected_face_starts_a_resize() {
        let gesture =
            CanvasGesture::begin_stage_drag(Pos2::new(200.0, 200.0), false, [face(3, BOX)]);
        let CanvasGesture::ResizingBox(drag) = gesture else {
            panic!("expected a resize, got {gesture:?}");
        };
        assert_eq!((drag.index, drag.handle), (3, DragHandle::SouthEast));
    }

    #[test]
    fn inside_a_selected_box_moves_it_and_elsewhere_pans() {
        let inside =
            CanvasGesture::begin_stage_drag(Pos2::new(150.0, 150.0), false, [face(0, BOX)]);
        assert!(matches!(
            inside,
            CanvasGesture::ResizingBox(ActiveBoxDrag {
                handle: DragHandle::Move,
                ..
            })
        ));
        let outside = CanvasGesture::begin_stage_drag(Pos2::new(20.0, 20.0), false, [face(0, BOX)]);
        assert_eq!(outside, CanvasGesture::Panning);
        let nothing_selected = CanvasGesture::begin_stage_drag(Pos2::new(150.0, 150.0), false, []);
        assert_eq!(nothing_selected, CanvasGesture::Panning);
    }

    #[test]
    fn the_draw_tool_draws_even_over_a_selected_box() {
        let pos = Pos2::new(150.0, 150.0);
        assert_eq!(
            CanvasGesture::begin_stage_drag(pos, true, [face(0, BOX)]),
            CanvasGesture::DrawingBox(ManualBoxDraft {
                start: pos,
                current: pos
            })
        );
    }

    #[test]
    fn a_gesture_ends_when_no_button_is_held() {
        let mut gesture = CanvasGesture::Rotating(RotationDragState {
            start_mouse_angle: 0.0,
            start_rotation: 90.0,
        });
        gesture.settle(true);
        assert!(!gesture.is_idle(), "still held, still rotating");
        gesture.settle(false);
        assert!(gesture.is_idle());
    }
}
