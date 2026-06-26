//! eframe::App implementation for fcs-gui.

use crate::core::{detection::build_detector, settings::load_settings};
use crate::types::*;
use crate::ui;

use eframe::{App, CreationContext, Frame};
use egui::{CursorIcon, ResizeDirection, ViewportCommand};
use fcs_core::CropSettings as CoreCropSettings;
use fcs_utils::{config::default_settings_path, configure_telemetry, gpu::GpuContext};
use image::DynamicImage;
use log::info;
use lru::LruCache;
use std::{
    collections::{HashSet, VecDeque},
    num::NonZeroUsize,
    path::{Path, PathBuf},
    sync::{Arc, mpsc},
};

const SIDEBAR_W: f32 = 300.0;
const INSPECTOR_W: f32 = 340.0;

// ── Clipboard image helpers ───────────────────────────────────────────────────

/// Reads an image from the system clipboard.
///
/// Tries arboard (CF_DIB / CF_BITMAP) first. On Windows, falls back to the
/// registered `"PNG"` clipboard format used by browsers and some screenshot
/// tools that only write PNG rather than a Device-Independent Bitmap.
fn read_clipboard_image() -> anyhow::Result<Arc<DynamicImage>> {
    use image::{DynamicImage, RgbaImage};

    if let Ok(mut cb) = arboard::Clipboard::new()
        && let Ok(data) = cb.get_image()
        && let Some(rgba) = RgbaImage::from_raw(
            data.width as u32,
            data.height as u32,
            data.bytes.into_owned(),
        )
    {
        return Ok(Arc::new(DynamicImage::ImageRgba8(rgba)));
    }

    #[cfg(target_os = "windows")]
    if let Ok(img) = windows_png_from_clipboard() {
        return Ok(Arc::new(img));
    }

    Err(anyhow::anyhow!(
        "No image found in clipboard. Copy an image first, then paste."
    ))
}

/// Windows-only fallback: reads the `"PNG"` registered clipboard format.
///
/// Browsers and some screenshot tools (e.g. ShareX, Greenshot) write PNG bytes
/// under this format instead of CF_DIB, which arboard does not handle.
#[cfg(target_os = "windows")]
fn windows_png_from_clipboard() -> anyhow::Result<image::DynamicImage> {
    use windows::Win32::Foundation::HGLOBAL;
    use windows::Win32::System::DataExchange::{
        CloseClipboard, GetClipboardData, IsClipboardFormatAvailable, OpenClipboard,
        RegisterClipboardFormatW,
    };
    use windows::Win32::System::Memory::{GlobalLock, GlobalSize, GlobalUnlock};

    unsafe {
        let format = RegisterClipboardFormatW(windows::core::w!("PNG"));
        anyhow::ensure!(format != 0, "RegisterClipboardFormatW failed");

        if IsClipboardFormatAvailable(format).is_err() {
            return Err(anyhow::anyhow!("PNG format not in clipboard"));
        }

        // None = NULL HWND (clipboard not owned by any window)
        OpenClipboard(None).map_err(|e| anyhow::anyhow!("OpenClipboard: {e}"))?;

        let result = (|| -> anyhow::Result<image::DynamicImage> {
            let handle =
                GetClipboardData(format).map_err(|e| anyhow::anyhow!("GetClipboardData: {e}"))?;
            let hglobal = HGLOBAL(handle.0);
            let size = GlobalSize(hglobal);
            anyhow::ensure!(size > 0, "clipboard PNG data is empty");
            let ptr = GlobalLock(hglobal);
            anyhow::ensure!(!ptr.is_null(), "GlobalLock failed");
            let bytes = std::slice::from_raw_parts(ptr.cast::<u8>(), size);
            let img =
                image::load_from_memory(bytes).map_err(|e| anyhow::anyhow!("PNG decode: {e}"))?;
            let _ = GlobalUnlock(hglobal);
            Ok(img)
        })();

        let _ = CloseClipboard();
        result
    }
}

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

        let (gpu_status, gpu_context, detector_result) = build_detector(&settings, shared_gpu);
        let gpu = GpuPipeline::from_context(gpu_status, gpu_context);

        let detector = match detector_result {
            Ok(d) => {
                info!("YuNet model loaded");
                Some(Arc::new(d))
            }
            Err(err) => {
                log::warn!("Model unavailable: {err}");
                None
            }
        };

        let status_line = initial_status_line(detector.is_some()).to_owned();

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
            job_tx,
            job_rx,
            preview: PreviewState::default(),
            cache: LruCache::new(NonZeroUsize::new(50).unwrap()),
            crop_preview_cache: LruCache::new(NonZeroUsize::new(500).unwrap()),
            image_cache: LruCache::new(NonZeroUsize::new(20).unwrap()),
            selected_faces: HashSet::new(),
            show_crop_overlay: true,
            crop_history,
            crop_history_index: 0,
            crop_fill_hex_input,
            aspect_ratio_locked: false,
            aspect_ratio_idx,
            batch_files: Vec::new(),
            batch_current_index: None,
            mapping: MappingUiState::new(),
            manual_box_draft: None,
            active_bbox_drag: None,
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
            last_detect_ms: None,
            texture_seq: 0,
            job_counter: 0,
            current_job: None,
            model_path_input,
            model_path_dirty: false,
            clipboard_temp_images: Vec::new(),
            clipboard_paste_pending: false,
            suppress_image_paste: false,
            webcam_state: WebcamState::default(),
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
            canvas_rotation: 0.0,
            rotation_drag: None,
            show_about: false,
            needs_detector_rebuild: false,
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
        if self.is_busy {
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
        info!("Sharing GPU context from eframe renderer");
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
fn initial_status_line(has_detector: bool) -> &'static str {
    if has_detector {
        "Model ready — select an image to detect faces."
    } else {
        "Model not loaded — configure model path to begin."
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
        if self.needs_detector_rebuild {
            self.rebuild_detector();
        }
        self.poll_worker(ctx);
        self.poll_webcam_frames(ctx);
        self.handle_dropped_files(ctx);
        self.sync_window_title(ctx);
        if std::mem::take(&mut self.clipboard_paste_pending) {
            self.paste_clipboard_image();
        }
        if self.is_busy || self.webcam_state.status == WebcamStatus::Active {
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
        let response = ui.interact(rect, ui.id().with(id), egui::Sense::click_and_drag());
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
    pub fn rebuild_detector(&mut self) {
        self.needs_detector_rebuild = false;
        let shared = self.gpu.context.clone();
        let (status, new_gpu_ctx, result) = build_detector(&self.settings, shared);
        if new_gpu_ctx.is_some() {
            // New context: adopt it (and its status) as the active GPU unit.
            self.gpu = GpuPipeline::from_context(status, new_gpu_ctx);
        } else {
            self.gpu.status = status;
        }
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
        self.cache.clear();
        self.crop_preview_cache.clear();
        if let Some(path) = self.preview.image_path.clone() {
            self.load_image_path(path);
        }
    }

    fn poll_webcam_frames(&mut self, ctx: &egui::Context) {
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
            let texture = ctx.load_texture(&tex_name, color_image, egui::TextureOptions::default());
            self.preview.texture = Some(texture);
            self.preview.image_size = Some((w, h));
            self.preview.source_image = Some(raw);
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
            JobMessage::DetectionFinished {
                job_id,
                cache_key,
                data,
            } => {
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
                self.active_bbox_drag = None;
                self.manual_box_tool_enabled = false;
                self.canvas_rotation = 0.0;

                self.last_detect_ms = Some(data.detect_ms);
                let n = self.preview.detections.len();
                self.push_log(format!("Detected {n} face(s)"), LogKind::Ok);
                self.status_line = format!("Detected {n} face(s)");

                // Auto-select all faces
                self.selected_faces = (0..n).collect();

                self.cache.put(
                    cache_key,
                    crate::types::DetectionCacheEntry {
                        texture,
                        detections: data.detections,
                        original_size: data.original_size,
                        source_image: data.original_image,
                    },
                );
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
            JobMessage::BatchProgress { index, status } => {
                if let Some(f) = self.batch_files.get_mut(index) {
                    f.status = status;
                }
            }
            JobMessage::BatchComplete { completed, failed } => {
                self.is_busy = false;
                self.push_log(
                    format!("Batch done: {completed} ok, {failed} failed"),
                    LogKind::Ok,
                );
                self.status_line = format!("Batch complete: {completed} ok, {failed} failed");
            }
            _ => {}
        }
    }

    fn handle_dropped_files(&mut self, ctx: &egui::Context) {
        let dropped = ctx.input_mut(|i| std::mem::take(&mut i.raw.dropped_files));
        if dropped.is_empty() {
            return;
        }

        const MAPPING_EXTS: &[&str] = &["csv", "xlsx", "xls", "db", "sqlite", "sqlite3"];
        let mut mapping_loaded = false;
        let mut paths = Vec::new();
        let mut unsupported = 0usize;

        for file in dropped {
            let Some(path) = file.path else {
                unsupported += 1;
                continue;
            };
            let is_mapping = path.extension().and_then(|e| e.to_str()).is_some_and(|e| {
                let lower = e.to_ascii_lowercase();
                MAPPING_EXTS.contains(&lower.as_str())
            });
            if is_mapping {
                self.mapping.set_file(path.clone());
                let _ = self.mapping.reload_preview();
                self.sidebar_tab = SidebarTab::Mapping;
                self.show_success(format!(
                    "Loaded mapping: {}",
                    path.file_name().and_then(|n| n.to_str()).unwrap_or("file")
                ));
                mapping_loaded = true;
            } else {
                match expand_input_path(&path) {
                    Ok(mut images) => paths.append(&mut images),
                    Err(err) => {
                        unsupported += 1;
                        self.push_log(format!("Drop skipped: {err:#}"), LogKind::Warn);
                    }
                }
            }
        }

        let had_image_paths = !paths.is_empty();
        let first = paths.first().cloned();
        let added = self.enqueue_batch_paths(paths);
        if let Some(path) = first {
            self.load_image_path(path);
        }
        if added > 0 {
            self.show_success(format!(
                "Added {added} image(s) to the queue ({} total)",
                self.batch_files.len()
            ));
        } else if !mapping_loaded {
            if unsupported > 0 {
                self.show_error("Unsupported drop", "No supported files were found.");
            } else if had_image_paths {
                self.show_success("All dropped images were already queued.");
            }
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

    pub fn load_image_path(&mut self, path: PathBuf) {
        use crate::core::detection::spawn_detection_job;
        self.preview.begin_loading(path.clone());
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
            rotation,
            auto_orient_exif,
            self.job_tx.clone(),
        );
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
        let idx = self.webcam_state.device_index;
        let w = self.webcam_state.width;
        let h = self.webcam_state.height;
        let fps = self.webcam_state.fps;
        spawn_webcam_stream(idx, w, h, fps, stop_flag, frame_tx);
        self.push_log("Webcam opened".into(), LogKind::Info);
    }

    pub fn close_webcam(&mut self) {
        use std::sync::atomic::Ordering;
        if let Some(flag) = &self.webcam_state.stop_flag {
            flag.store(true, Ordering::Relaxed);
        }
        self.webcam_state.stop_flag = None;
        self.webcam_state.frame_rx = None;
        self.webcam_state.status = WebcamStatus::Inactive;
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
            self.job_tx.clone(),
        );
    }

    pub fn paste_clipboard_image(&mut self) {
        use crate::core::detection::spawn_detection_job_from_image;

        let image = match read_clipboard_image() {
            Ok(img) => img,
            Err(err) => {
                self.show_error("Clipboard", err.to_string());
                return;
            }
        };

        let path = PathBuf::from("clipboard://paste");
        self.preview.begin_loading(path.clone());
        self.is_busy = true;
        let job_id = self.job_counter;
        self.job_counter += 1;
        self.current_job = Some(job_id);
        self.push_log("Detecting faces in clipboard image…".into(), LogKind::Info);

        spawn_detection_job_from_image(
            job_id,
            image,
            path,
            self.detector.clone(),
            self.job_tx.clone(),
        );
    }

    pub fn enqueue_batch_paths(&mut self, paths: Vec<PathBuf>) -> usize {
        let mut existing: HashSet<PathBuf> =
            self.batch_files.iter().map(|f| f.path.clone()).collect();
        let mut added = 0usize;
        for path in paths {
            if !is_supported_image_path(&path) || !existing.insert(path.clone()) {
                continue;
            }
            self.batch_files.push(BatchFile {
                path,
                status: BatchFileStatus::Pending,
                output_override: None,
            });
            added += 1;
        }
        added
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

    pub fn delete_selected_faces(&mut self) {
        let mut i = 0;
        self.preview.detections.retain(|_| {
            let keep = !self.selected_faces.contains(&i);
            i += 1;
            keep
        });
        self.selected_faces.clear();
        self.crop_preview_cache.clear();
    }

    pub fn commit_manual_box(&mut self, bbox: fcs_core::BoundingBox) {
        use fcs_core::{Detection, Landmark};
        use fcs_utils::quality::Quality;
        let det = crate::types::DetectionWithQuality {
            detection: Detection {
                bbox,
                landmarks: [Landmark::new(0.0, 0.0); 5],
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
        self.crop_preview_cache.clear();
    }
}

pub(crate) use fcs_utils::is_supported_image_path;

fn expand_input_path(path: &Path) -> anyhow::Result<Vec<PathBuf>> {
    if path.is_dir() {
        collect_supported_images_in_dir(path)
    } else if path.is_file() && is_supported_image_path(path) {
        Ok(vec![path.to_path_buf()])
    } else {
        anyhow::bail!("unsupported path {}", path.display())
    }
}

pub(crate) fn collect_folder_images(dir: &Path) -> Vec<PathBuf> {
    collect_supported_images_in_dir(dir).unwrap_or_default()
}

fn collect_supported_images_in_dir(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut images = Vec::new();
    let mut queue = VecDeque::new();
    queue.push_back(dir.to_path_buf());

    while let Some(current) = queue.pop_front() {
        for entry in std::fs::read_dir(&current)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                queue.push_back(path);
            } else if path.is_file() && is_supported_image_path(&path) {
                images.push(path);
            }
        }
    }

    images.sort();
    Ok(images)
}

// ── About dialog ──────────────────────────────────────────────────────────────

fn show_about_window(ctx: &egui::Context, open: &mut bool) {
    use crate::theme::P;

    let window_frame = egui::Frame::window(&ctx.global_style())
        .fill(P::BG2)
        .stroke(egui::Stroke::new(1.0, P::RULE2))
        .corner_radius(egui::CornerRadius::same(8))
        .inner_margin(egui::Margin::same(24))
        .shadow(egui::Shadow {
            offset: [0, 8],
            blur: 32,
            spread: 0,
            color: P::black_alpha(120),
        });

    let mut close = false;
    egui::Window::new("About Face Crop Studio")
        .open(open)
        .collapsible(false)
        .resizable(false)
        .fixed_size(egui::vec2(340.0, 0.0))
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .title_bar(false)
        .frame(window_frame)
        .show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add(
                    egui::Image::new(egui::include_image!("../assets/app_logo.svg"))
                        .fit_to_exact_size(egui::Vec2::splat(48.0)),
                );

                ui.add_space(10.0);
                ui.label(
                    egui::RichText::new("Face Crop Studio")
                        .size(17.0)
                        .color(P::INK)
                        .strong(),
                );
                ui.add_space(2.0);
                ui.label(
                    egui::RichText::new(concat!("v", env!("CARGO_PKG_VERSION")))
                        .size(11.0)
                        .color(P::INK3),
                );
                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new("AI-powered face detection and cropping")
                        .size(12.0)
                        .color(P::INK3),
                );
            });

            ui.add_space(16.0);
            ui.separator();
            ui.add_space(12.0);

            ui.label(
                egui::RichText::new(
                    "Face Crop Studio uses the YuNet neural network to detect \
                     faces and automatically crop portraits at scale.",
                )
                .size(12.5)
                .color(P::INK2),
            );

            ui.add_space(14.0);

            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Website").size(12.0).color(P::INK3));
                ui.add_space(4.0);
                ui.hyperlink_to(
                    egui::RichText::new("facecropstudio.com")
                        .size(12.0)
                        .color(P::CYAN),
                    "https://facecropstudio.com/",
                );
            });

            ui.add_space(16.0);

            ui.vertical_centered(|ui| {
                if ui.button(egui::RichText::new("Close").size(13.0)).clicked() {
                    close = true;
                }
            });
        });
    if close {
        *open = false;
    }
}
