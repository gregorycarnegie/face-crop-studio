//! Undo/redo of face edits, and the edits themselves: delete and draw.

use crate::types::{App2, EditSnapshot};

impl App2 {
    fn edit_snapshot(&self) -> EditSnapshot {
        EditSnapshot {
            detections: self.preview.detections.clone(),
            selected: self.selected_faces.clone(),
        }
    }

    fn restore_snapshot(&mut self, snap: EditSnapshot) {
        self.preview.detections = snap.detections;
        self.selected_faces = snap.selected;
        // A resize in progress names its face by index, which the restored list may not share.
        self.gesture = Default::default();
    }

    /// Capture the current face state before a mutation. Clears the redo stack.
    pub fn push_undo(&mut self) {
        self.undo_stack.push(self.edit_snapshot());
        if self.undo_stack.len() > 50 {
            self.undo_stack.remove(0);
        }
        self.redo_stack.clear();
    }

    pub fn undo(&mut self) {
        if let Some(snap) = self.undo_stack.pop() {
            let current = self.edit_snapshot();
            self.redo_stack.push(current);
            self.restore_snapshot(snap);
        }
    }

    pub fn redo(&mut self) {
        if let Some(snap) = self.redo_stack.pop() {
            let current = self.edit_snapshot();
            self.undo_stack.push(current);
            self.restore_snapshot(snap);
        }
    }

    /// Drop edit history — called when a new image's detections arrive so undo
    /// can't restore boxes from a previous image.
    pub fn clear_edit_history(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
    }

    pub fn delete_selected_faces(&mut self) {
        if self.selected_faces.is_empty() {
            return;
        }
        self.push_undo();
        let mut i = 0;
        self.preview.detections.retain(|_| {
            let keep = !self.selected_faces.contains(&i);
            i += 1;
            keep
        });
        self.selected_faces.clear();
        // Indices after the removed faces shift, so a resize in progress would move another box.
        self.gesture = Default::default();
    }

    pub fn commit_manual_box(&mut self, bbox: fcs_core::BoundingBox) {
        use fcs_core::Detection;
        use fcs_utils::quality::Quality;
        self.push_undo();
        let det = crate::types::DetectionWithQuality {
            detection: Detection {
                bbox,
                landmarks: [None; 5],
                score: 1.0,
            },
            quality_score: 1.0,
            quality: Quality::High,
            thumbnail: None,
            current_bbox: bbox,
            original_bbox: bbox,
            origin: crate::types::DetectionOrigin::Manual,
        };
        let idx = self.preview.detections.len();
        self.preview.detections.push(det);
        self.selected_faces.insert(idx);
    }
}
