//! eframe::App implementation for fcs-gui.

mod clipboard;
mod dialogs;
mod history;
mod input;
mod webcam;

pub(crate) use input::collect_folder_images;

use self::dialogs::{show_about_window, show_cuda_warning, show_overwrite_dialog};
use crate::{
    core::{detection::build_detector, settings::load_settings},
    types::*,
    ui,
};

use eframe::{App, CreationContext, Frame};
use egui::{CursorIcon, ResizeDirection, ViewportCommand};
use fcs_core::CropSettings as CoreCropSettings;
use fcs_utils::{
    config::default_settings_path,
    configure_telemetry,
    gpu::{GpuContext, GpuStatusIndicator},
};
use log::info;
use std::{
    collections::{HashSet, VecDeque},
    path::PathBuf,
    sync::{Arc, mpsc},
};

const SIDEBAR_W: f32 = 300.0;
const INSPECTOR_W: f32 = 340.0;

// ── App creation ─────────────────────────────────────────────────────────────

impl App2 {
    pub fn new(cc: &CreationContext<'_>) -> Self {
        crate::theme::apply(&cc.egui_ctx);
        egui_extras::install_image_loaders(&cc.egui_ctx);

        let settings_path = default_settings_path();
        let settings = load_settings(&settings_path);
        configure_telemetry(
            settings.telemetry.enabled,
            settings.telemetry.level_filter(),
        );

        let shared_gpu = share_gpu_from_eframe(cc);

        // Experiment 81: `build_detector` compiles five compute pipelines, which measured
        // 150 ms of an 890 ms launch, and it used to run right here -- so the window stayed
        // empty for it. It is built on a thread instead and picked up by `poll_detector`
        // a few frames later. Nothing needs to wait for it: every consumer of `detector`
        // already gates on it being present, because it can legitimately fail to build.
        let (detector_tx, detector_rx) = mpsc::channel();
        let build_settings = settings.clone();
        std::thread::spawn(move || {
            let started = std::time::Instant::now();
            let built = build_detector(&build_settings, shared_gpu);
            info!(
                "startup: build_detector {:.0} ms, finished {:.0} ms after launch",
                started.elapsed().as_secs_f64() * 1e3,
                crate::since_launch_ms()
            );
            let _ = detector_tx.send(built);
        });

        Self::with_settings(settings, settings_path, Some(detector_rx))
    }

    /// The app without the user's settings file or a detector: default settings, saved to
    /// `settings_path` (a temporary file in tests, since the GUI autosaves), and no detector
    /// thread, so the status stays on the startup line.
    #[cfg(test)]
    pub(crate) fn for_tests(cc: &CreationContext<'_>, settings_path: PathBuf) -> Self {
        crate::theme::apply(&cc.egui_ctx);
        // The About window draws the logo; with no loader egui draws a warning glyph no font has.
        egui_extras::install_image_loaders(&cc.egui_ctx);
        Self::with_settings(
            fcs_utils::config::AppSettings::default(),
            settings_path,
            None,
        )
    }

    fn with_settings(
        settings: fcs_utils::config::AppSettings,
        settings_path: PathBuf,
        detector_rx: Option<mpsc::Receiver<DetectorBuild>>,
    ) -> Self {
        let gpu = GpuPipeline::from_context(
            GpuStatusIndicator::pending(),
            None,
            fcs_utils::EnhancementRuntime::cpu_only(),
        );
        let detector = None;
        let status_line = STATUS_DETECTOR_PENDING.to_owned();

        let (job_tx, job_rx) = mpsc::channel();
        let default_settings = settings.clone();
        let model_path_input = settings.model_path.clone().unwrap_or_default();
        let crop_history = vec![settings.crop.clone()];
        let crop_fill_hex_input = crop_fill_hex(&settings.crop);
        let aspect_ratio_idx = derive_aspect_ratio_idx(&settings.crop);

        Self {
            settings,
            default_settings,
            settings_path,
            gpu,
            detector,
            // Arrives with the detector, on the same channel.
            eye_refiner: None,
            detector_rx,
            job_tx,
            job_rx,
            preview: PreviewState::default(),
            selected_faces: HashSet::new(),
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            show_crop_overlay: true,
            result_preview: Default::default(),
            pending_export: None,
            crop_history,
            crop_history_index: 0,
            crop_fill_hex_input,
            aspect_ratio_locked: false,
            aspect_ratio_idx,
            batch_files: Vec::new(),
            batch_current_index: None,
            mapping: MappingUiState::new(),
            gesture: Default::default(),
            manual_box_tool_enabled: false,
            sidebar_tab: SidebarTab::Queue,
            inspector_tab: InspectorTab::Crop,
            panel_state: PanelState::default(),
            log_lines: VecDeque::from([LogLine {
                timestamp: "—".into(),
                message: "Waiting for image…".into(),
                kind: LogKind::Info,
            }]),
            status_line,
            last_error: None,
            is_busy: false,
            batch_running: false,
            last_detect_ms: None,
            texture_seq: 0,
            job_counter: 0,
            current_job: None,
            model_path_input,
            model_path_dirty: false,
            clipboard_paste_pending: false,
            suppress_image_paste: false,
            first_frame_logged: false,
            webcam_state: WebcamState::default(),
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
            canvas_rotation: 0.0,
            show_about: false,
            confirm_cuda: false,
            needs_detector_rebuild: false,
            needs_postprocess_update: false,
            last_window_title: String::new(),
        }
    }

    fn sync_window_title(&mut self, ctx: &egui::Context) {
        let mut desired = match self
            .preview
            .image_path
            .as_ref()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
        {
            Some(name) => format!("Face Crop Studio — {name}"),
            None => "Face Crop Studio".to_owned(),
        };
        if self.is_busy || self.batch_running {
            desired.push_str(" ●");
        }
        if desired != self.last_window_title {
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(desired.clone()));
            self.last_window_title = desired;
        }
    }
}

/// Reuse eframe's wgpu device/queue so the GUI shares one GPU context with the
/// renderer instead of spinning up a second adapter.
fn share_gpu_from_eframe(cc: &CreationContext<'_>) -> Option<Arc<GpuContext>> {
    cc.wgpu_render_state.as_ref().map(|rs| {
        let adapter = rs.adapter.get_info();
        info!(
            "Sharing GPU context from eframe renderer: '{}' ({:?})",
            adapter.name, adapter.backend
        );
        Arc::new(GpuContext::from_existing(
            None,
            None,
            rs.device.clone(),
            rs.queue.clone(),
            rs.adapter.get_info(),
        ))
    })
}

/// Maps the saved output dimensions onto the sidebar's aspect-ratio selector
/// index: 1 = 1:1, 2 = 4:5, 3 = 3:4, 0 = custom/other.
fn derive_aspect_ratio_idx(crop: &fcs_utils::config::CropSettings) -> usize {
    let w = crop.output_width as f32;
    let h = crop.output_height as f32;
    if w == h {
        1
    } else if h > 0.0 && (w / h - 4.0 / 5.0).abs() < 0.01 {
        2
    } else if h > 0.0 && (w / h - 3.0 / 4.0).abs() < 0.01 {
        3
    } else {
        0
    }
}

/// Initial value for the crop fill-color hex input field.
fn crop_fill_hex(crop: &fcs_utils::config::CropSettings) -> String {
    format!(
        "#{:02X}{:02X}{:02X}",
        crop.fill_color.red, crop.fill_color.green, crop.fill_color.blue
    )
}

/// Status-bar text shown immediately after startup.
/// Shown between the first frame and the detector arriving on it (experiment 81).
const STATUS_DETECTOR_PENDING: &str = "Starting the detector…";

/// Names the detector and the engine it landed on.
///
/// One detector ships, using the WGSL kernels or the built-in CPU graph depending on
/// GPU availability, and that choice affects speed. Until this said
/// so, the only way to find out which was running was the log.
fn initial_status_line(detector: Option<&fcs_core::FaceDetector>) -> String {
    match detector {
        Some(detector) => format!(
            "{} ready on {} — select an image to detect faces.",
            detector.model_name(),
            detector.inference_backend()
        ),
        None => "Model not loaded — configure model path to begin.".to_owned(),
    }
}

// ── eframe::App ───────────────────────────────────────────────────────────────

impl App for App2 {
    /// Detects Cmd/Ctrl+V for image paste on every platform, including Wayland.
    ///
    /// egui_winit swallows the Key::V *press* event when the clipboard holds an
    /// image (not text), so `key_pressed(Key::V)` is always false in that case.
    /// Key *releases* are never swallowed, so we watch for
    /// `Key::V { pressed: false }` with the command modifier instead.
    ///
    /// When the clipboard actually holds text, egui_winit emits `Event::Paste`
    /// in the press frame. We latch that and skip the image-paste trigger on the
    /// following release frame, so a normal text paste is never intercepted.
    fn raw_input_hook(&mut self, _ctx: &egui::Context, raw_input: &mut egui::RawInput) {
        // Latch: text paste succeeded this frame → skip image paste on V release.
        if raw_input
            .events
            .iter()
            .any(|e| matches!(e, egui::Event::Paste(_)))
        {
            self.suppress_image_paste = true;
        }

        for event in &raw_input.events {
            if let egui::Event::Key {
                key: egui::Key::V,
                pressed: false,
                modifiers,
                ..
            } = event
                && modifiers.command
                && !std::mem::take(&mut self.suppress_image_paste)
            {
                self.clipboard_paste_pending = true;
            }
        }
    }

    fn logic(&mut self, ctx: &egui::Context, _frame: &mut Frame) {
        if !self.first_frame_logged {
            self.first_frame_logged = true;
            info!(
                "startup: first frame at {:.0} ms since launch",
                crate::since_launch_ms()
            );
        }
        self.poll_detector(ctx);
        if self.needs_detector_rebuild {
            self.rebuild_detector();
        }
        // Once the pointer is released, so a slider drag re-detects once rather than on every
        // frame of the drag.
        if self.needs_postprocess_update && !ctx.input(|i| i.pointer.any_down()) {
            self.apply_postprocess_settings();
        }
        self.poll_worker(ctx);
        self.poll_webcam_frames(ctx);
        self.handle_dropped_files(ctx);
        self.sync_window_title(ctx);
        if std::mem::take(&mut self.clipboard_paste_pending) {
            self.paste_clipboard_image();
        }
        // Undo/Redo shortcuts — skipped while a text field has focus so
        // TextEdit keeps its own internal undo.
        if ctx.memory(|m| m.focused().is_none()) {
            use egui::{Key, KeyboardShortcut, Modifiers};
            let redo_y = KeyboardShortcut::new(Modifiers::COMMAND, Key::Y);
            let redo_shift_z = KeyboardShortcut::new(Modifiers::COMMAND | Modifiers::SHIFT, Key::Z);
            let undo = KeyboardShortcut::new(Modifiers::COMMAND, Key::Z);
            // Consumed here, before any widget sees it, so a focused button is not also pressed.
            if ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Space)) {
                self.result_preview.enabled = !self.result_preview.enabled;
            }
            if ctx.input_mut(|i| i.consume_shortcut(&redo_y) || i.consume_shortcut(&redo_shift_z)) {
                self.redo();
            } else if ctx.input_mut(|i| i.consume_shortcut(&undo)) {
                self.undo();
            }
        }
        if self.is_busy || self.batch_running || self.webcam_state.status == WebcamStatus::Active {
            ctx.request_repaint();
        }
    }

    fn ui(&mut self, root_ui: &mut egui::Ui, _frame: &mut Frame) {
        ui::menubar::show(root_ui, self);
        ui::toolbar::show(root_ui, self);
        ui::statusbar::show(root_ui, self);

        let side_frame = egui::Frame::new()
            .fill(crate::theme::P::BG2)
            .inner_margin(egui::Margin::ZERO)
            .outer_margin(egui::Margin::ZERO);

        egui::Panel::left("sidebar")
            .exact_size(SIDEBAR_W)
            .resizable(false)
            .show_separator_line(false)
            .frame(side_frame)
            .show(root_ui, |ui| {
                ui::sidebar::show(ui, self);
            });

        egui::Panel::right("inspector")
            .exact_size(INSPECTOR_W)
            .resizable(false)
            .show_separator_line(false)
            .frame(side_frame)
            .show(root_ui, |ui| {
                ui::inspector::show(ui, self);
            });

        egui::CentralPanel::default()
            .frame(
                egui::Frame::new()
                    .fill(crate::theme::P::BG1)
                    .inner_margin(egui::Margin::ZERO)
                    .outer_margin(egui::Margin::ZERO),
            )
            .show(root_ui, |ui| {
                ui::canvas::show(ui, self);
            });

        install_resize_edges(root_ui);

        if self.show_about {
            show_about_window(root_ui.ctx(), &mut self.show_about);
        }
        if self.confirm_cuda {
            let available =
                self.settings.gpu.enabled && matches!(fcs_core::cuda::availability(), Some(Ok(_)));
            if let Some(enable) = show_cuda_warning(root_ui.ctx(), available) {
                if enable {
                    self.settings.gpu.experimental_cuda = true;
                    self.needs_detector_rebuild = true;
                }
                self.confirm_cuda = false;
            }
        }
        if let Some(pending) = &self.pending_export {
            let choice = show_overwrite_dialog(root_ui.ctx(), pending);
            if let Some(choice) = choice {
                crate::core::export::resolve_pending_export(self, choice);
            }
        }
    }
}

fn install_resize_edges(ui: &mut egui::Ui) {
    let full = ui.max_rect();
    let edge = 6.0;
    let corner = 14.0;
    let entries = [
        (
            "resize_n",
            egui::Rect::from_min_max(full.min, egui::pos2(full.max.x, full.min.y + edge)),
            ResizeDirection::North,
            CursorIcon::ResizeNorth,
        ),
        (
            "resize_s",
            egui::Rect::from_min_max(egui::pos2(full.min.x, full.max.y - edge), full.max),
            ResizeDirection::South,
            CursorIcon::ResizeSouth,
        ),
        (
            "resize_w",
            egui::Rect::from_min_max(full.min, egui::pos2(full.min.x + edge, full.max.y)),
            ResizeDirection::West,
            CursorIcon::ResizeWest,
        ),
        (
            "resize_e",
            egui::Rect::from_min_max(egui::pos2(full.max.x - edge, full.min.y), full.max),
            ResizeDirection::East,
            CursorIcon::ResizeEast,
        ),
        (
            "resize_nw",
            egui::Rect::from_min_size(full.min, egui::vec2(corner, corner)),
            ResizeDirection::NorthWest,
            CursorIcon::ResizeNorthWest,
        ),
        (
            "resize_ne",
            egui::Rect::from_min_size(
                egui::pos2(full.max.x - corner, full.min.y),
                egui::vec2(corner, corner),
            ),
            ResizeDirection::NorthEast,
            CursorIcon::ResizeNorthEast,
        ),
        (
            "resize_sw",
            egui::Rect::from_min_size(
                egui::pos2(full.min.x, full.max.y - corner),
                egui::vec2(corner, corner),
            ),
            ResizeDirection::SouthWest,
            CursorIcon::ResizeSouthWest,
        ),
        (
            "resize_se",
            egui::Rect::from_min_size(
                full.max - egui::vec2(corner, corner),
                egui::vec2(corner, corner),
            ),
            ResizeDirection::SouthEast,
            CursorIcon::ResizeSouthEast,
        ),
    ];

    for (id, rect, direction, cursor) in entries {
        let response = ui.interact(rect, ui.scope_id().with(id), egui::Sense::click_and_drag());
        if response.hovered() {
            ui.output_mut(|o| o.cursor_icon = cursor);
        }
        if response.drag_started() {
            ui.ctx()
                .send_viewport_cmd(ViewportCommand::BeginResize(direction));
        }
    }
}

// ── Worker polling / dropped files ───────────────────────────────────────────

impl App2 {
    /// Takes delivery of the detector built off the critical path in `App::new`.
    ///
    /// egui only repaints when something asks it to, so this asks: without the repaint
    /// request the window would paint once and then sit idle until the user moved the
    /// mouse, and the detector would appear whenever that happened rather than when it
    /// was ready.
    fn poll_detector(&mut self, ctx: &egui::Context) {
        let Some(rx) = &self.detector_rx else {
            return;
        };
        match rx.try_recv() {
            Ok((status, context, result, eye_refiner, enhancement)) => {
                self.detector_rx = None;
                self.gpu = GpuPipeline::from_context(status, context, enhancement);
                self.eye_refiner = eye_refiner.map(Arc::new);
                self.detector = match result {
                    Ok(d) => {
                        info!("detector loaded");
                        Some(Arc::new(d))
                    }
                    Err(err) => {
                        log::warn!("Model unavailable: {err}");
                        None
                    }
                };
                self.status_line = initial_status_line(self.detector.as_deref());
                info!(
                    "startup: detector ready at {:.0} ms since launch",
                    crate::since_launch_ms()
                );
                // Anything dropped or pasted while it was building was parked rather than
                // refused; run it now.
                if let Some(path) = self.preview.image_path.clone()
                    && self.preview.is_loading
                {
                    self.load_image_path(path);
                }
            }
            Err(mpsc::TryRecvError::Empty) => ctx.request_repaint(),
            Err(mpsc::TryRecvError::Disconnected) => {
                // The builder thread died without sending. Nothing will arrive, so stop
                // asking and leave the app in the same state a failed build leaves it.
                self.detector_rx = None;
                log::warn!("Detector build thread ended without a result");
                self.status_line = initial_status_line(None);
                // Release anything that was parked waiting for it, or the preview spins
                // on a detector that is never coming.
                if self.preview.is_loading {
                    self.preview.is_loading = false;
                    self.show_error(
                        "Detector unavailable",
                        "The detector failed to start. Check the model path in settings.",
                    );
                }
            }
        }
    }

    pub fn rebuild_detector(&mut self) {
        self.needs_detector_rebuild = false;
        // A rebuild reads the current thresholds and re-detects anyway.
        self.needs_postprocess_update = false;
        let shared = self.gpu.context.clone();
        let (status, new_gpu_ctx, result, eye_refiner, enhancement) =
            build_detector(&self.settings, shared);
        if new_gpu_ctx.is_some() {
            // New context: adopt it (and its status) as the active GPU unit.
            self.gpu = GpuPipeline::from_context(status, new_gpu_ctx, enhancement);
        } else {
            self.gpu.status = status;
        }
        // Replaced along with the detector, or a rebuild would leave the previous one behind.
        self.eye_refiner = eye_refiner.map(Arc::new);
        self.detector = match result {
            Ok(d) => {
                info!("Detector rebuilt");
                Some(Arc::new(d))
            }
            Err(err) => {
                log::warn!("Detector rebuild failed: {err}");
                self.show_error("Rebuild failed", format!("{err:#}"));
                None
            }
        };
        if let Some(path) = self.preview.image_path.clone() {
            self.load_image_path(path);
        }
    }

    fn poll_worker(&mut self, ctx: &egui::Context) {
        let mut updated = false;
        while let Ok(msg) = self.job_rx.try_recv() {
            self.handle_job_message(ctx, msg);
            updated = true;
        }
        if updated {
            ctx.request_repaint();
        }
    }

    fn handle_job_message(&mut self, ctx: &egui::Context, msg: JobMessage) {
        use egui::TextureOptions;
        match msg {
            JobMessage::DetectionFinished { job_id, data } => {
                if Some(job_id) != self.current_job {
                    return;
                }
                self.current_job = None;
                self.is_busy = false;
                self.preview.is_loading = false;

                let tex_name = format!("preview_{}", self.texture_seq);
                self.texture_seq += 1;
                let texture =
                    ctx.load_texture(&tex_name, data.color_image, TextureOptions::default());

                self.preview.texture = Some(texture.clone());
                self.preview.image_size = Some(data.original_size);
                self.preview.detections = data.detections.clone();
                self.preview.source_image = Some(data.original_image.clone());
                for det in &mut self.preview.detections {
                    crate::core::quality::refresh_thumbnail(
                        ctx,
                        det,
                        &data.original_image,
                        &self.settings,
                        &mut self.texture_seq,
                    );
                }
                self.gesture = Default::default();
                self.manual_box_tool_enabled = false;
                self.canvas_rotation = 0.0;
                self.clear_edit_history();

                self.last_detect_ms = Some(data.detect_ms);
                let n = self.preview.detections.len();
                self.push_log(format!("Detected {n} face(s)"), LogKind::Ok);
                self.status_line = format!("Detected {n} face(s)");

                // Auto-select all faces
                self.selected_faces = (0..n).collect();
            }
            JobMessage::WebcamDetections {
                frame_number,
                detections,
                detect_ms,
            } => {
                self.webcam_state.detect_inflight = false;
                // A result that outlived its camera, or arrived after the user turned live
                // detection off, must not paint boxes over whatever is on screen now.
                if self.webcam_state.status != WebcamStatus::Active
                    || !self.webcam_state.live_detect
                {
                    return;
                }
                self.webcam_state.last_detect_ms = Some(detect_ms);
                let _ = frame_number;
                self.preview.detections = detections;
                self.selected_faces.clear();
            }
            JobMessage::DetectionFailed { job_id, error } => {
                if Some(job_id) != self.current_job {
                    return;
                }
                self.current_job = None;
                self.is_busy = false;
                self.preview.is_loading = false;
                self.last_error = Some(error.clone());
                self.push_log(format!("Detection failed: {error}"), LogKind::Warn);
                self.status_line = "Detection failed.".to_owned();
            }
            JobMessage::BatchProgress { path, status } => {
                // A row removed mid-batch simply gets no update.
                if let Some(f) = self.batch_files.iter_mut().find(|f| f.path == path) {
                    f.status = status;
                }
            }
            JobMessage::ResultPreview {
                generation,
                face,
                rendered,
            } => crate::core::result_preview::receive(self, ctx, generation, face, rendered),
            JobMessage::BatchComplete { completed, failed } => {
                self.batch_running = false;
                self.push_log(
                    format!("Batch done: {completed} ok, {failed} failed"),
                    LogKind::Ok,
                );
                self.status_line = format!("Batch complete: {completed} ok, {failed} failed");
            }
            _ => {}
        }
    }

    pub fn push_log(&mut self, message: String, kind: LogKind) {
        use std::time::{SystemTime, UNIX_EPOCH};
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let h = (secs / 3600) % 24;
        let m = (secs / 60) % 60;
        let s = secs % 60;
        self.log_lines.push_back(LogLine {
            timestamp: format!("{h:02}:{m:02}:{s:02}"),
            message,
            kind,
        });
        if self.log_lines.len() > 100 {
            self.log_lines.pop_front();
        }
    }

    /// Apply changed detection thresholds without rebuilding the detector (experiment 66).
    ///
    /// Score threshold, NMS and top-k are applied after inference, so the loaded model and its
    /// compiled pipelines stay valid, and the decoded image is already in memory. Routing these
    /// through `rebuild_detector` recompiled every shader on the UI thread and re-decoded the
    /// file from disk, all to change a comparison.
    fn apply_postprocess_settings(&mut self) {
        use crate::core::detection::spawn_detection_job_from_image;
        // Still building: keep the flag and apply it once the detector arrives.
        let Some(detector) = self.detector.clone() else {
            return;
        };
        self.needs_postprocess_update = false;
        // Re-wraps at the new settings, sharing the loaded model.
        let detector = Arc::new(detector.with_settings(&self.settings.detection));
        self.detector = Some(Arc::clone(&detector));
        let Some(path) = self.preview.image_path.clone() else {
            return;
        };
        // A load still in flight has no decoded image yet -- or the previous one -- so restart
        // it with the new detector rather than re-detecting whatever is in memory.
        let Some(image) = self
            .preview
            .source_image
            .clone()
            .filter(|_| !self.preview.is_loading)
        else {
            self.load_image_path(path);
            return;
        };
        self.is_busy = true;
        let job_id = self.job_counter;
        self.job_counter += 1;
        self.current_job = Some(job_id);
        spawn_detection_job_from_image(
            job_id,
            image,
            path,
            Some(detector),
            self.eye_refiner.clone(),
            self.job_tx.clone(),
        );
    }

    pub fn load_image_path(&mut self, path: PathBuf) {
        use crate::core::detection::spawn_detection_job;
        self.preview.begin_loading(path.clone());
        // A file dropped between the first frame and the detector arriving would otherwise
        // be told "No detector loaded. Configure model path in settings." -- true of a
        // missing model, wrong about one that is 100 ms away (experiment 81). Hold the
        // path instead; `poll_detector` replays it, the same way `rebuild_detector` does.
        if self.detector.is_none() && self.detector_rx.is_some() {
            self.status_line = STATUS_DETECTOR_PENDING.to_owned();
            return;
        }
        self.is_busy = true;
        let job_id = self.job_counter;
        self.job_counter += 1;
        self.current_job = Some(job_id);
        self.push_log(format!("Loading {}", path.display()), LogKind::Info);
        let rotation = self.canvas_rotation;
        let auto_orient_exif = self.settings.crop.auto_orient_exif;
        spawn_detection_job(
            job_id,
            path,
            self.detector.clone(),
            self.eye_refiner.clone(),
            rotation,
            auto_orient_exif,
            self.job_tx.clone(),
        );
    }

    pub fn show_success(&mut self, message: impl Into<String>) {
        let message = message.into();
        self.last_error = None;
        self.status_line = message.clone();
        self.push_log(message, LogKind::Ok);
    }

    pub fn show_error(&mut self, title: impl AsRef<str>, detail: impl AsRef<str>) {
        let message = format!("{}: {}", title.as_ref(), detail.as_ref());
        self.last_error = Some(message.clone());
        self.status_line = message.clone();
        self.push_log(message, LogKind::Warn);
    }

    pub fn resolved_output_dimensions(&self) -> (u32, u32) {
        (
            self.settings.crop.output_width,
            self.settings.crop.output_height,
        )
    }

    pub fn build_crop_settings(&self) -> CoreCropSettings {
        (&self.settings.crop).into()
    }
}

#[cfg(test)]
mod tests;
