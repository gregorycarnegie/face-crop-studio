//! Application state types for fcs-gui.

use egui::TextureHandle;
use fcs_core::{BoundingBox, Detection};
use fcs_utils::{
    CropShape, PolygonCornerStyle,
    config::{AppSettings, CropSettings as ConfigCropSettings, ResizeQuality},
    gpu::{GpuContext, GpuStatusIndicator},
    mapping::{
        ColumnSelector, MappingCatalog, MappingEntry, MappingFormat, MappingPreview,
        MappingReadOptions, inspect_mapping_sources, load_mapping_entries, load_mapping_preview,
    },
    quality::Quality,
};
use image::DynamicImage;
use lru::LruCache;
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
    pub fn badge_label(&self) -> &str {
        match self {
            Self::Pending => "—",
            Self::Processing => "run",
            Self::Completed { faces_exported, .. } => {
                if *faces_exported == 0 {
                    "skip"
                } else {
                    "ok"
                }
            }
            Self::Failed { .. } => "err",
            Self::Skipped => "skip",
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

// ── Cache keys ────────────────────────────────────────────────────────────────

#[derive(Hash, PartialEq, Eq, Clone)]
pub struct CacheKey {
    pub path: PathBuf,
    pub model_path: Option<String>,
    pub input_width: u32,
    pub input_height: u32,
    pub resize_quality: ResizeQuality,
    pub score_bits: u32,
    pub nms_bits: u32,
    pub top_k: usize,
}

pub struct DetectionCacheEntry {
    pub texture: TextureHandle,
    pub detections: Vec<DetectionWithQuality>,
    pub original_size: (u32, u32),
    pub source_image: Arc<DynamicImage>,
}

/// Hashable encoding of `CropShape` + vignette settings for cache keying.
#[derive(Clone, Hash, PartialEq, Eq)]
pub struct ShapeKey {
    pub kind: u8,
    pub param1_bits: u32,
    pub param2_bits: u32,
    pub sides: u8,
    pub rotation_bits: u32,
    pub vignette_softness_bits: u32,
    pub vignette_intensity_bits: u32,
    pub vignette_r: u8,
    pub vignette_g: u8,
    pub vignette_b: u8,
    pub vignette_a: u8,
}

impl ShapeKey {
    pub fn from_crop(crop: &ConfigCropSettings) -> Self {
        let (kind, param1_bits, param2_bits, sides, rotation_bits) = encode_shape(&crop.shape);
        Self {
            kind,
            param1_bits,
            param2_bits,
            sides,
            rotation_bits,
            vignette_softness_bits: crop.vignette_softness.to_bits(),
            vignette_intensity_bits: crop.vignette_intensity.to_bits(),
            vignette_r: crop.vignette_color.red,
            vignette_g: crop.vignette_color.green,
            vignette_b: crop.vignette_color.blue,
            vignette_a: crop.vignette_color.alpha,
        }
    }
}

fn encode_shape(shape: &CropShape) -> (u8, u32, u32, u8, u32) {
    match shape {
        CropShape::Rectangle => (0, 0, 0, 0, 0),
        CropShape::Ellipse => (1, 0, 0, 0, 0),
        CropShape::RoundedRectangle { radius_pct } => (2, radius_pct.to_bits(), 0, 0, 0),
        CropShape::ChamferedRectangle { size_pct } => (3, size_pct.to_bits(), 0, 0, 0),
        CropShape::Polygon {
            sides,
            rotation_deg,
            corner_style,
        } => {
            let (ck, p1) = match corner_style {
                PolygonCornerStyle::Sharp => (0u8, 0u32),
                PolygonCornerStyle::Rounded { radius_pct } => (1, radius_pct.to_bits()),
                PolygonCornerStyle::Chamfered { size_pct } => (2, size_pct.to_bits()),
                PolygonCornerStyle::Bezier { tension } => (3, tension.to_bits()),
            };
            (4 + ck, p1, 0, *sides, rotation_deg.to_bits())
        }
        CropShape::Star {
            points,
            inner_radius_pct,
            rotation_deg,
        } => (
            8,
            inner_radius_pct.to_bits(),
            0,
            *points,
            rotation_deg.to_bits(),
        ),
        CropShape::KochPolygon {
            sides,
            rotation_deg,
            iterations,
        } => (9, *iterations as u32, 0, *sides, rotation_deg.to_bits()),
        CropShape::KochRectangle { iterations } => (10, *iterations as u32, 0, 0, 0),
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct CropPreviewKey {
    pub path: PathBuf,
    pub face_index: usize,
    pub output_width: u32,
    pub output_height: u32,
    pub positioning_mode: u8,
    pub face_height_bits: u32,
    pub horizontal_bits: u32,
    pub vertical_bits: u32,
    pub fill_color_bits: u32,
    pub shape: ShapeKey,
}

pub struct CropPreviewCacheEntry {
    pub image: Arc<DynamicImage>,
    pub texture: Option<TextureHandle>,
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

#[derive(Clone, Copy, PartialEq, Eq)]
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
        cache_key: CacheKey,
        data: DetectionJobSuccess,
    },
    DetectionFailed {
        job_id: u64,
        error: String,
    },
    WebcamFrame {
        image: DynamicImage,
        frame_number: u32,
        detections: Vec<DetectionWithQuality>,
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
        use fcs_utils::mapping::detect_format;
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
    pub job_tx: mpsc::Sender<JobMessage>,
    pub job_rx: mpsc::Receiver<JobMessage>,

    // Preview
    pub preview: PreviewState,
    pub cache: LruCache<CacheKey, DetectionCacheEntry>,
    pub crop_preview_cache: LruCache<CropPreviewKey, CropPreviewCacheEntry>,
    pub image_cache: LruCache<PathBuf, Arc<DynamicImage>>,

    // Selection & editing
    pub selected_faces: HashSet<usize>,
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

    // Cached OS window title; compared each frame to decide whether to emit
    // ViewportCommand::Title. Drives the taskbar/Alt+Tab label on Windows and
    // the native title bar on macOS/Linux.
    pub last_window_title: String,
}
