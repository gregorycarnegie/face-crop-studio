//! Application state types for fcs-gui.

use egui::TextureHandle;
use fcs_core::{BoundingBox, Detection};
use fcs_mapping::{
    ColumnSelector, MappingCatalog, MappingEntry, MappingFormat, MappingPreview,
    MappingReadOptions, inspect_mapping_sources, load_mapping_entries, load_mapping_preview,
};
use fcs_utils::{
    config::{AppSettings, CropSettings as ConfigCropSettings},
    gpu::{GpuContext, GpuStatusIndicator},
    quality::Quality,
};
use image::DynamicImage;
use std::{
    collections::{HashSet, VecDeque},
    path::PathBuf,
    sync::{Arc, atomic::AtomicBool, mpsc},
};

// ── Sidebar / inspector tab selectors ────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum SidebarTab {
    #[default]
    Queue,
    Mapping,
    History,
}

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum InspectorTab {
    #[default]
    Crop,
    Output,
    Enhance,
}

// ── Batch file status ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchFileStatus {
    Pending,
    Processing,
    Completed {
        faces_detected: usize,
        faces_exported: usize,
    },
    Failed {
        error: String,
    },
    Skipped,
}

impl BatchFileStatus {
    /// Stable outcome name, as written to the batch report.
    pub fn outcome(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Processing => "processing",
            Self::Completed {
                faces_detected: 0, ..
            } => "no_faces",
            // Faces were found, but the quality rules kept every one of them back.
            Self::Completed {
                faces_exported: 0, ..
            } => "filtered",
            Self::Completed { .. } => "succeeded",
            Self::Failed { .. } => "failed",
            Self::Skipped => "skipped",
        }
    }
    pub fn face_count(&self) -> Option<usize> {
        if let Self::Completed { faces_exported, .. } = self {
            Some(*faces_exported)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub struct BatchFile {
    pub path: PathBuf,
    pub status: BatchFileStatus,
    pub output_override: Option<PathBuf>,
}

// ── Detection quality ─────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DetectionOrigin {
    Detector,
    Manual,
}

#[derive(Clone)]
pub struct DetectionWithQuality {
    pub detection: Detection,
    pub quality_score: f64,
    pub quality: Quality,
    pub thumbnail: Option<TextureHandle>,
    pub current_bbox: BoundingBox,
    pub original_bbox: BoundingBox,
    pub origin: DetectionOrigin,
}

impl DetectionWithQuality {
    pub fn active_bbox(&self) -> BoundingBox {
        self.current_bbox
    }
    pub fn reset_bbox(&mut self) {
        self.current_bbox = self.original_bbox;
    }
    pub fn set_bbox(&mut self, bbox: BoundingBox) {
        self.current_bbox = bbox;
    }
    pub fn is_manual(&self) -> bool {
        matches!(self.origin, DetectionOrigin::Manual)
    }
    pub fn is_modified(&self) -> bool {
        self.current_bbox != self.original_bbox
    }
}

// ── Edit history (Undo/Redo) ──────────────────────────────────────────────────

/// Snapshot of the editable face state, captured before each mutation.
pub struct EditSnapshot {
    pub detections: Vec<DetectionWithQuality>,
    pub selected: HashSet<usize>,
}

// ── Preview state ─────────────────────────────────────────────────────────────

#[derive(Default)]
pub struct PreviewState {
    pub image_path: Option<PathBuf>,
    pub texture: Option<TextureHandle>,
    pub image_size: Option<(u32, u32)>,
    pub detections: Vec<DetectionWithQuality>,
    pub is_loading: bool,
    pub source_image: Option<Arc<DynamicImage>>,
}

impl PreviewState {
    pub fn begin_loading(&mut self, path: PathBuf) {
        self.image_path = Some(path);
        self.texture = None;
        self.image_size = None;
        self.detections.clear();
        self.is_loading = true;
        self.source_image = None;
    }
}

// ── Webcam state ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebcamStatus {
    Inactive,
    Starting,
    Active,
    Stopping,
    Error,
}

pub struct WebcamState {
    pub status: WebcamStatus,
    pub device_index: u32,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub frames_captured: u32,
    pub error_message: Option<String>,
    pub stop_flag: Option<Arc<AtomicBool>>,
    pub frame_rx: Option<mpsc::Receiver<(egui::ColorImage, Arc<DynamicImage>)>>,
    /// Detect on every frame rather than only when the button is pressed.
    ///
    /// Affordable because the loop is capture-bound: a frame arrives every ~42 ms and
    /// detection answers in 2.5-3.6 ms, so the pipeline is idle most of each frame
    /// (experiment 95).
    pub live_detect: bool,
    /// One detection in flight at a time. Without this a slow frame would queue work behind
    /// itself and the overlay would fall further behind the picture the longer it ran.
    pub detect_inflight: bool,
    /// Detection latency of the last completed live frame, for the sidebar readout.
    pub last_detect_ms: Option<f64>,
    /// Frames that arrived while a detection was already running, so the cost of live
    /// detection is visible rather than inferred.
    pub frames_skipped: u32,
}
impl Default for WebcamState {
    fn default() -> Self {
        Self {
            status: WebcamStatus::Inactive,
            device_index: 0,
            width: 1280,
            height: 720,
            fps: 30,
            frames_captured: 0,
            error_message: None,
            live_detect: false,
            detect_inflight: false,
            last_detect_ms: None,
            frames_skipped: 0,
            stop_flag: None,
            frame_rx: None,
        }
    }
}

// ── Drag / interaction ────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
pub struct ManualBoxDraft {
    pub start: egui::Pos2,
    pub current: egui::Pos2,
}

#[derive(Clone, Copy)]
pub struct ActiveBoxDrag {
    pub index: usize,
    pub handle: DragHandle,
    pub start_bbox: BoundingBox,
    pub drag_start_screen: egui::Pos2,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DragHandle {
    Move,
    NorthWest,
    NorthEast,
    SouthWest,
    SouthEast,
}

#[derive(Clone, Copy)]
pub struct RotationDragState {
    pub start_mouse_angle: f32,
    pub start_rotation: f32,
}

#[derive(Clone, Copy, Default)]
pub struct PointerSnapshot {
    pub pressed: bool,
    pub released: bool,
    pub down: bool,
    pub press_origin: Option<egui::Pos2>,
    pub pos: Option<egui::Pos2>,
}
impl PointerSnapshot {
    pub fn capture(ctx: &egui::Context) -> Self {
        ctx.input(|i| PointerSnapshot {
            pressed: i.pointer.primary_pressed(),
            released: i.pointer.primary_released(),
            down: i.pointer.primary_down(),
            press_origin: i.pointer.press_origin(),
            pos: i.pointer.interact_pos(),
        })
    }
}

// ── Panel open/close state ────────────────────────────────────────────────────

pub struct PanelState {
    pub crop_framing: bool,
    pub crop_shape: bool,
    pub positioning: bool,
    pub crops_ready: bool,
    pub enhancement: bool,
}
impl Default for PanelState {
    fn default() -> Self {
        Self {
            crop_framing: true,
            crop_shape: true,
            positioning: true,
            crops_ready: true,
            enhancement: false,
        }
    }
}

// ── Job messages ──────────────────────────────────────────────────────────────

pub struct DetectionJobSuccess {
    pub path: PathBuf,
    pub color_image: egui::ColorImage,
    pub detections: Vec<DetectionWithQuality>,
    pub original_size: (u32, u32),
    pub original_image: Arc<DynamicImage>,
    pub detect_ms: u64,
}

pub enum JobMessage {
    DetectionFinished {
        job_id: u64,
        data: DetectionJobSuccess,
    },
    DetectionFailed {
        job_id: u64,
        error: String,
    },
    /// A live webcam frame's detections.
    ///
    /// Deliberately not `DetectionFinished`: that path uploads a preview texture, rebuilds
    /// every thumbnail, clears the edit history and resets the selection, which is right for
    /// one deliberate detection and impossible twenty-four times a second.
    WebcamDetections {
        frame_number: u32,
        detections: Vec<DetectionWithQuality>,
        detect_ms: f64,
    },
    WebcamError(String),
    WebcamStopped,
    BatchProgress {
        index: usize,
        status: BatchFileStatus,
    },
    BatchComplete {
        completed: usize,
        failed: usize,
    },
}

// ── Log line (mini-log overlay) ───────────────────────────────────────────────

#[derive(Clone)]
pub struct LogLine {
    pub timestamp: String,
    pub message: String,
    pub kind: LogKind,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum LogKind {
    Info,
    Ok,
    Warn,
}

// ── Mapping UI state ──────────────────────────────────────────────────────────

pub struct MappingUiState {
    pub file_path: Option<PathBuf>,
    pub base_dir: Option<PathBuf>,
    pub detected_format: Option<MappingFormat>,
    pub format_override: Option<MappingFormat>,
    pub has_headers: bool,
    pub delimiter_input: String,
    pub sheet_name: String,
    pub sql_table: String,
    pub sql_query: String,
    pub catalog: MappingCatalog,
    pub preview: Option<MappingPreview>,
    pub preview_error: Option<String>,
    pub source_column_idx: Option<usize>,
    pub output_column_idx: Option<usize>,
    pub entries: Vec<MappingEntry>,
}
impl Default for MappingUiState {
    fn default() -> Self {
        Self::new()
    }
}
impl MappingUiState {
    pub fn new() -> Self {
        Self {
            file_path: None,
            base_dir: None,
            detected_format: None,
            format_override: None,
            has_headers: true,
            delimiter_input: ",".to_string(),
            sheet_name: String::new(),
            sql_table: String::new(),
            sql_query: String::new(),
            catalog: MappingCatalog::default(),
            preview: None,
            preview_error: None,
            source_column_idx: None,
            output_column_idx: None,
            entries: Vec::new(),
        }
    }
    pub fn set_file(&mut self, path: PathBuf) {
        use fcs_mapping::detect_format;
        self.file_path = Some(path);
        self.base_dir = self
            .file_path
            .as_ref()
            .and_then(|p| p.parent().map(|x| x.to_path_buf()));
        self.detected_format = self.file_path.as_ref().map(|p| detect_format(p));
        self.preview = None;
        self.preview_error = None;
        self.source_column_idx = None;
        self.output_column_idx = None;
        self.entries.clear();
        self.sheet_name.clear();
        self.sql_table.clear();
        self.sql_query.clear();
        self.delimiter_input = ",".to_string();
        self.refresh_catalog();
    }
    pub fn effective_format(&self) -> Option<MappingFormat> {
        self.format_override.or(self.detected_format)
    }
    pub fn refresh_catalog(&mut self) {
        if let Some(path) = &self.file_path {
            match inspect_mapping_sources(path, &self.read_options()) {
                Ok(catalog) => {
                    self.catalog = catalog;
                    if self.sheet_name.is_empty()
                        && let Some(first) = self.catalog.sheets.first()
                    {
                        self.sheet_name = first.clone();
                    }
                    if self.sql_table.is_empty()
                        && let Some(first) = self.catalog.sql_tables.first()
                    {
                        self.sql_table = first.clone();
                    }
                }
                Err(err) => self.preview_error = Some(err.to_string()),
            }
        } else {
            self.catalog = MappingCatalog::default();
        }
    }
    pub fn read_options(&self) -> MappingReadOptions {
        let mut opts = MappingReadOptions {
            format: self.effective_format(),
            has_headers: Some(self.has_headers),
            delimiter: self.delimiter_input.chars().next().map(|c| c as u8),
            ..Default::default()
        };
        if !self.sheet_name.trim().is_empty() {
            opts.sheet_name = Some(self.sheet_name.trim().to_string());
        }
        if !self.sql_table.trim().is_empty() {
            opts.sql_table = Some(self.sql_table.trim().to_string());
        }
        if !self.sql_query.trim().is_empty() {
            opts.sql_query = Some(self.sql_query.trim().to_string());
        }
        opts
    }
    pub fn load_entries(&mut self) -> anyhow::Result<()> {
        let path = self
            .file_path
            .clone()
            .ok_or_else(|| anyhow::anyhow!("No file selected"))?;
        let src = self
            .source_column_idx
            .map(ColumnSelector::Index)
            .ok_or_else(|| anyhow::anyhow!("No source column"))?;
        let out = self
            .output_column_idx
            .map(ColumnSelector::Index)
            .ok_or_else(|| anyhow::anyhow!("No output column"))?;
        match load_mapping_entries(&path, &self.read_options(), &src, &out) {
            Ok(e) => {
                self.entries = e;
                self.preview_error = None;
                Ok(())
            }
            Err(e) => {
                self.preview_error = Some(e.to_string());
                Err(e)
            }
        }
    }
    pub fn reload_preview(&mut self) -> anyhow::Result<()> {
        let path = self
            .file_path
            .clone()
            .ok_or_else(|| anyhow::anyhow!("No file selected"))?;
        match load_mapping_preview(&path, &self.read_options()) {
            Ok(p) => {
                if self.source_column_idx.is_none() && !p.columns.is_empty() {
                    self.source_column_idx = Some(0);
                }
                if self.output_column_idx.is_none() && p.columns.len() > 1 {
                    self.output_column_idx = Some(1);
                }
                self.preview = Some(p);
                self.preview_error = None;
                self.entries.clear();
                Ok(())
            }
            Err(e) => {
                self.preview_error = Some(e.to_string());
                self.preview = None;
                Err(e)
            }
        }
    }
}

/// What `build_detector` produces, as it crosses back from the thread that built it.
///
/// The eye refiner rides along because it opens an ONNX session, and the reason the detector
/// is built off the UI thread (experiment 81) applies equally to it. `None` is ordinary: no
/// runtime, or no model file -- see `fcs_core::EyeRefiner::load`.
pub type DetectorBuild = (
    GpuStatusIndicator,
    Option<Arc<GpuContext>>,
    anyhow::Result<fcs_core::YuNetDetector>,
    Option<fcs_core::EyeRefiner>,
);

// ── GPU pipeline ──────────────────────────────────────────────────────────────

/// Shared GPU context plus the status indicator describing the active adapter.
/// Bundled so the app state and detector rebuilds carry one cohesive GPU unit
/// instead of parallel fields.
pub struct GpuPipeline {
    pub status: GpuStatusIndicator,
    pub context: Option<Arc<GpuContext>>,
}

impl GpuPipeline {
    /// Bundles the GPU status indicator with an optional shared GPU context.
    /// With no context the app runs CPU-only.
    pub fn from_context(status: GpuStatusIndicator, context: Option<Arc<GpuContext>>) -> Self {
        Self { status, context }
    }
}

// ── Main app struct ───────────────────────────────────────────────────────────

pub struct App2 {
    // Backend
    pub settings: AppSettings,
    pub default_settings: AppSettings,
    pub settings_path: PathBuf,
    pub gpu: GpuPipeline,
    pub detector: Option<Arc<fcs_core::YuNetDetector>>,
    /// Better eye points for levelling crops, when a runtime and the model are both present.
    ///
    /// Applied to every detection rather than gated on `settings.crop.eye_line_align`:
    /// detections are computed once when an image loads, but the toggle can be flipped at any
    /// time afterwards, and gating would leave stale unrefined landmarks behind whenever it
    /// was. One small forward pass per face against a detector pass that dominates it.
    pub eye_refiner: Option<Arc<fcs_core::EyeRefiner>>,
    /// Set while the detector is still being built on a background thread.
    ///
    /// Experiment 81: building it took 150 ms of a 890 ms launch and ran before the first
    /// frame, so the window sat empty for it. Everything that uses the detector already
    /// gates on `detector.is_some()`, so "not built yet" needs no new state beyond the
    /// channel the finished one arrives on.
    pub detector_rx: Option<std::sync::mpsc::Receiver<DetectorBuild>>,
    pub job_tx: mpsc::Sender<JobMessage>,
    pub job_rx: mpsc::Receiver<JobMessage>,

    // Preview
    pub preview: PreviewState,

    // Selection & editing
    pub selected_faces: HashSet<usize>,
    pub undo_stack: Vec<EditSnapshot>,
    pub redo_stack: Vec<EditSnapshot>,
    pub show_crop_overlay: bool,
    pub crop_history: Vec<ConfigCropSettings>,
    pub crop_history_index: usize,
    pub crop_fill_hex_input: String,
    pub aspect_ratio_locked: bool,
    pub aspect_ratio_idx: usize,

    // Batch
    pub batch_files: Vec<BatchFile>,
    pub batch_current_index: Option<usize>,

    // Mapping
    pub mapping: MappingUiState,

    // Interaction
    pub manual_box_draft: Option<ManualBoxDraft>,
    pub active_bbox_drag: Option<ActiveBoxDrag>,
    pub manual_box_tool_enabled: bool,

    // UI state
    pub sidebar_tab: SidebarTab,
    pub inspector_tab: InspectorTab,
    pub panel_state: PanelState,
    pub log_lines: VecDeque<LogLine>,

    // Misc
    pub status_line: String,
    pub last_error: Option<String>,
    pub last_detect_ms: Option<u64>,
    pub is_busy: bool,
    pub texture_seq: u64,
    pub job_counter: u64,
    pub current_job: Option<u64>,
    pub model_path_input: String,
    pub model_path_dirty: bool,
    pub clipboard_paste_pending: bool,
    /// Set when egui handled a text paste (clipboard had text). Cleared on the
    /// next V-release so we don't also trigger an image paste in that case.
    pub suppress_image_paste: bool,
    /// One-shot latch for the startup timing line (experiment 81).
    pub first_frame_logged: bool,
    pub webcam_state: WebcamState,

    // Canvas zoom / rotation
    pub zoom: f32,
    pub pan: egui::Vec2,
    pub canvas_rotation: f32,
    pub rotation_drag: Option<RotationDragState>,

    // Dialogs
    pub show_about: bool,

    // Deferred side-effects
    pub needs_detector_rebuild: bool,
    /// A detection threshold changed; the model stays, only postprocessing and the result do.
    pub needs_postprocess_update: bool,

    // Cached OS window title; compared each frame to decide whether to emit
    // ViewportCommand::Title. Drives the taskbar/Alt+Tab label on Windows and
    // the native title bar on macOS/Linux.
    pub last_window_title: String,
}
