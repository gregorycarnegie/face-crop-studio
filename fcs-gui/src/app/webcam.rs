//! The live camera: opening, closing, frame polling and per-frame detection.

use std::{
    path::PathBuf,
    sync::{Arc, mpsc},
};

use image::DynamicImage;

use crate::types::{App2, LogKind, WebcamStatus};

impl App2 {
    pub(super) fn poll_webcam_frames(&mut self, ctx: &egui::Context) {
        if self.webcam_state.frame_rx.is_none() {
            return;
        }
        // Drain the channel, keeping only the most recent frame
        let mut latest: Option<(egui::ColorImage, Arc<DynamicImage>)> = None;
        if let Some(rx) = &self.webcam_state.frame_rx {
            while let Ok(frame) = rx.try_recv() {
                latest = Some(frame);
            }
        }
        if let Some((color_image, raw)) = latest {
            self.webcam_state.frames_captured += 1;
            let tex_name = format!("webcam_{}", self.texture_seq);
            self.texture_seq += 1;
            let w = color_image.size[0] as u32;
            let h = color_image.size[1] as u32;
            {
                let _guard = fcs_utils::telemetry::timing_guard(
                    "fcs_gui::webcam_texture",
                    log::Level::Trace,
                );
                let texture =
                    ctx.load_texture(&tex_name, color_image, egui::TextureOptions::default());
                self.preview.texture = Some(texture);
            }
            self.preview.image_size = Some((w, h));
            self.preview.source_image = Some(raw.clone());

            // Live detection: one in flight at a time, and only while the camera is open.
            // A frame arriving during a detection is counted and dropped rather than queued
            // -- the overlay should track the picture, not trail it.
            if self.webcam_state.live_detect
                && self.webcam_state.status == WebcamStatus::Active
                && let Some(detector) = self.detector.clone()
            {
                if self.webcam_state.detect_inflight {
                    self.webcam_state.frames_skipped += 1;
                } else {
                    self.webcam_state.detect_inflight = true;
                    crate::core::detection::spawn_webcam_detection(
                        self.webcam_state.frames_captured,
                        raw,
                        detector,
                        self.job_tx.clone(),
                    );
                }
            }
        }
    }

    pub fn open_webcam(&mut self) {
        use crate::core::detection::spawn_webcam_stream;
        use std::sync::atomic::Ordering;
        // Stop any existing stream first
        if let Some(flag) = &self.webcam_state.stop_flag {
            flag.store(true, Ordering::Relaxed);
        }
        let stop_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (frame_tx, frame_rx) = mpsc::channel();
        self.webcam_state.stop_flag = Some(stop_flag.clone());
        self.webcam_state.frame_rx = Some(frame_rx);
        self.webcam_state.status = WebcamStatus::Active;
        self.webcam_state.frames_captured = 0;
        self.webcam_state.error_message = None;
        // Show webcam in canvas
        self.preview.image_path = Some(PathBuf::from("webcam://live"));
        self.preview.texture = None;
        self.preview.image_size = None;
        self.preview.detections.clear();
        self.preview.source_image = None;
        self.preview.is_loading = false;
        self.clear_edit_history();
        let idx = self.webcam_state.device_index;
        let w = self.webcam_state.width;
        let h = self.webcam_state.height;
        let fps = self.webcam_state.fps;
        spawn_webcam_stream(idx, w, h, fps, stop_flag, frame_tx);
        self.push_log("Webcam opened".into(), LogKind::Info);
    }

    /// Turn per-frame detection on or off.
    ///
    /// Turning it off clears the overlay: the boxes belong to a frame that is already gone,
    /// and leaving them on screen would make a frozen overlay look like a live one.
    pub fn toggle_live_detection(&mut self) {
        self.webcam_state.live_detect = !self.webcam_state.live_detect;
        if self.webcam_state.live_detect {
            self.webcam_state.frames_skipped = 0;
            self.webcam_state.last_detect_ms = None;
            self.push_log("Live detection on".into(), LogKind::Info);
        } else {
            self.preview.detections.clear();
            self.selected_faces.clear();
            self.push_log("Live detection off".into(), LogKind::Info);
        }
    }

    pub fn close_webcam(&mut self) {
        use std::sync::atomic::Ordering;
        if let Some(flag) = &self.webcam_state.stop_flag {
            flag.store(true, Ordering::Relaxed);
        }
        self.webcam_state.stop_flag = None;
        self.webcam_state.frame_rx = None;
        self.webcam_state.status = WebcamStatus::Inactive;
        self.webcam_state.live_detect = false;
        self.webcam_state.detect_inflight = false;
        self.preview.detections.clear();
        self.push_log("Webcam closed".into(), LogKind::Info);
    }

    pub fn detect_webcam_faces(&mut self) {
        use crate::core::detection::spawn_detection_job_from_image;
        use std::sync::atomic::Ordering;
        // Freeze stream
        if let Some(flag) = &self.webcam_state.stop_flag {
            flag.store(true, Ordering::Relaxed);
        }
        self.webcam_state.stop_flag = None;
        self.webcam_state.frame_rx = None;
        self.webcam_state.status = WebcamStatus::Inactive;
        let Some(source_image) = self.preview.source_image.clone() else {
            self.show_error(
                "No frame",
                "No webcam frame captured yet — wait a moment after opening the camera",
            );
            return;
        };
        let path = PathBuf::from("webcam://live");
        self.is_busy = true;
        self.preview.is_loading = true;
        let job_id = self.job_counter;
        self.job_counter += 1;
        self.current_job = Some(job_id);
        self.push_log("Detecting faces in webcam frame…".into(), LogKind::Info);
        spawn_detection_job_from_image(
            job_id,
            source_image,
            path,
            self.detector.clone(),
            self.eye_refiner.clone(),
            self.job_tx.clone(),
        );
    }
}
