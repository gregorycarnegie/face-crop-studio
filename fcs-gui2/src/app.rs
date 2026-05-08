//! eframe::App implementation for fcs-gui2.

use crate::core::{detection::build_detector, settings::load_settings};
use crate::types::*;
use crate::ui;

use eframe::{App, CreationContext, Frame};
use egui::{CursorIcon, ResizeDirection, ViewportCommand};
use fcs_core::{CropSettings as CoreCropSettings, PositioningMode, preset_by_name};
use fcs_utils::{
    WgpuEnhancer,
    config::default_settings_path,
    configure_telemetry,
    gpu::{GpuBatchCropper, GpuContext},
    quality::Quality,
};
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

type GpuPipelineHandles = (
    Option<Arc<GpuContext>>,
    Option<Arc<WgpuEnhancer>>,
    Option<Arc<GpuBatchCropper>>,
);

// ── App creation ─────────────────────────────────────────────────────────────

impl App2 {
    pub fn new(cc: &CreationContext<'_>) -> Self {
        crate::theme::apply(&cc.egui_ctx);

        let settings_path = default_settings_path();
        let settings = load_settings(&settings_path);
        configure_telemetry(
            settings.telemetry.enabled,
            settings.telemetry.level_filter(),
        );

        let shared_gpu = cc.wgpu_render_state.as_ref().map(|rs| {
            info!("Sharing GPU context from eframe renderer");
            Arc::new(GpuContext::from_existing(
                None,
                None,
                rs.device.clone(),
                rs.queue.clone(),
                rs.adapter.get_info(),
            ))
        });

        let (initial_gpu_status, initial_gpu_context, detector_result) =
            build_detector(&settings, shared_gpu);

        let (gpu_context, gpu_enhancer, gpu_batch_cropper) =
            init_gpu_pipelines(initial_gpu_context);

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

        let status_line = if detector.is_some() {
            "Model ready — select an image to detect faces.".to_owned()
        } else {
            "Model not loaded — configure model path to begin.".to_owned()
        };

        let (job_tx, job_rx) = mpsc::channel();
        let model_path_input = settings.model_path.clone().unwrap_or_default();
        let crop_history = vec![settings.crop.clone()];
        let crop_fill_hex_input = format!(
            "#{:02X}{:02X}{:02X}",
            settings.crop.fill_color.red,
            settings.crop.fill_color.green,
            settings.crop.fill_color.blue
        );

        Self {
            settings,
            settings_path,
            gpu_status: initial_gpu_status,
            gpu_context,
            gpu_enhancer,
            gpu_batch_cropper,
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
            batch_files: Vec::new(),
            batch_current_index: None,
            mapping: MappingUiState::new(),
            manual_box_draft: None,
            active_bbox_drag: None,
            manual_box_tool_enabled: false,
            sidebar_tab: SidebarTab::Queue,
            inspector_tab: InspectorTab::Crop,
            panel_state: PanelState::default(),
            log_lines: vec![LogLine {
                timestamp: "—".into(),
                message: "Waiting for image…".into(),
                kind: LogKind::Info,
            }],
            status_line,
            last_error: None,
            is_busy: false,
            texture_seq: 0,
            job_counter: 0,
            current_job: None,
            model_path_input,
            model_path_dirty: false,
            clipboard_temp_images: Vec::new(),
            webcam_state: WebcamState::default(),
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
        }
    }
}

fn init_gpu_pipelines(context: Option<Arc<GpuContext>>) -> GpuPipelineHandles {
    let Some(ctx) = context else {
        return (None, None, None);
    };
    let enhancer = WgpuEnhancer::new(ctx.clone()).ok().map(Arc::new);
    let cropper = GpuBatchCropper::new(ctx.clone()).ok().map(Arc::new);
    (Some(ctx), enhancer, cropper)
}

// ── eframe::App ───────────────────────────────────────────────────────────────

impl App for App2 {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut Frame) {
        self.poll_worker(ctx);
        self.handle_dropped_files(ctx);
        if self.is_busy {
            ctx.request_repaint();
        }
    }

    fn ui(&mut self, root_ui: &mut egui::Ui, _frame: &mut Frame) {
        ui::titlebar::show(root_ui, self);
        ui::menubar::show(root_ui, self);
        ui::toolbar::show(root_ui, self);
        ui::statusbar::show(root_ui, self);

        egui::Panel::left("sidebar")
            .exact_size(SIDEBAR_W)
            .resizable(false)
            .show_inside(root_ui, |ui| {
                ui::sidebar::show(ui, self);
            });

        egui::Panel::right("inspector")
            .exact_size(INSPECTOR_W)
            .resizable(false)
            .show_inside(root_ui, |ui| {
                ui::inspector::show(ui, self);
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(crate::theme::P::BG1))
            .show_inside(root_ui, |ui| {
                ui::canvas::show(ui, self);
            });

        install_resize_edges(root_ui);
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
                self.active_bbox_drag = None;
                self.manual_box_tool_enabled = false;

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
        let dropped: Vec<_> = ctx.input(|i| i.raw.dropped_files.clone());
        if dropped.is_empty() {
            return;
        }

        let mut paths = Vec::new();
        let mut unsupported = 0usize;
        for file in dropped {
            let Some(path) = file.path else {
                unsupported += 1;
                continue;
            };
            match expand_input_path(&path) {
                Ok(mut images) => paths.append(&mut images),
                Err(err) => {
                    unsupported += 1;
                    self.push_log(format!("Drop skipped: {err:#}"), LogKind::Warn);
                }
            }
        }

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
        } else if unsupported > 0 {
            self.show_error("Unsupported drop", "No supported images were found.");
        } else {
            self.show_success("All dropped images were already queued.");
        }

        ctx.input_mut(|i| i.raw.dropped_files.clear());
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
        self.log_lines.push(LogLine {
            timestamp: format!("{h:02}:{m:02}:{s:02}"),
            message,
            kind,
        });
        if self.log_lines.len() > 100 {
            self.log_lines.remove(0);
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
        spawn_detection_job(
            job_id,
            path,
            self.detector.clone(),
            self.settings.clone(),
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
        if self.settings.crop.preset == "custom" {
            (
                self.settings.crop.output_width,
                self.settings.crop.output_height,
            )
        } else if let Some(preset) = preset_by_name(&self.settings.crop.preset) {
            (preset.width, preset.height)
        } else {
            (
                self.settings.crop.output_width,
                self.settings.crop.output_height,
            )
        }
    }

    pub fn build_crop_settings(&self) -> CoreCropSettings {
        build_crop_settings_from_app_settings(&self.settings)
    }

    pub fn quality_suffix(&self, quality: Quality) -> Option<&'static str> {
        if !self.settings.crop.quality_rules.quality_suffix {
            return None;
        }
        match quality {
            Quality::High => Some("_highq"),
            Quality::Medium => Some("_medq"),
            Quality::Low => Some("_lowq"),
        }
    }
}

pub(crate) fn build_crop_settings_from_app_settings(
    settings: &fcs_utils::config::AppSettings,
) -> CoreCropSettings {
    let (output_width, output_height) = if settings.crop.preset == "custom" {
        (settings.crop.output_width, settings.crop.output_height)
    } else if let Some(preset) = preset_by_name(&settings.crop.preset) {
        (preset.width, preset.height)
    } else {
        (settings.crop.output_width, settings.crop.output_height)
    };

    let positioning_mode = match settings.crop.positioning_mode.as_str() {
        "rule-of-thirds" | "rule_of_thirds" | "thirds" => PositioningMode::RuleOfThirds,
        "custom" => PositioningMode::Custom,
        _ => PositioningMode::Center,
    };

    CoreCropSettings {
        output_width,
        output_height,
        face_height_pct: settings.crop.face_height_pct,
        positioning_mode,
        horizontal_offset: settings.crop.horizontal_offset,
        vertical_offset: settings.crop.vertical_offset,
        fill_color: settings.crop.fill_color,
    }
}

pub(crate) fn is_supported_image_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|ext| {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "jpg" | "jpeg" | "png" | "webp" | "bmp" | "tif" | "tiff"
            )
        })
        .unwrap_or(false)
}

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
