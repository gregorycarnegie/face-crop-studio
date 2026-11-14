//! The one object QML talks to. It owns the queue, the selection and the settings; the work
//! itself runs on background threads and comes back through `QThreadHandle::invoke_mut`.
//!
//! Photos have two identities. QML knows a photo by its position in the queue (`current`,
//! `select(index)`); everything that outlives a moment (detection jobs, image keys, the held
//! decode) knows it by its `uid`, which survives other photos being removed.

use crate::{
    images::{self, Pixels},
    pipeline::{self, Detected, Engine, Face, PhotoExport, Source, Turn},
};
use fcs_core::BoundingBox;
use fcs_mapping::{ColumnSelector, MappingFormat, MappingReadOptions};
use fcs_utils::{
    CropShape, OutputClaims, OverwritePolicy,
    config::{AppSettings, DetectionSettings, GpuSettings, default_settings_path},
    is_supported_image_path, outline_points_for_rect,
};
use image::{DynamicImage, RgbImage, RgbaImage};
use log::warn;
use qtbridge::{QmlObject, qobject};
use serde::Serialize;
use serde_json::{Value, json};
use std::{
    collections::VecDeque,
    fs,
    path::{Path, PathBuf},
    sync::{
        Arc, RwLock,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::Instant,
};

enum Status {
    Queued,
    Done,
    Failed(String),
}

struct Entry {
    uid: u64,
    source: Source,
    turn: Turn,
    /// Bumped each time the photo is detected again, so its thumbnail gets a new URL and QML's
    /// image cache cannot show the old one.
    rev: u32,
    status: Status,
    size: (u32, u32),
    faces: Vec<Face>,
    selected: Vec<bool>,
    /// The output name a mapping file gives this photo.
    mapped: Option<String>,
}

/// A loaded mapping file: which table, which columns, and what it matched.
struct MappingFile {
    path: PathBuf,
    format: MappingFormat,
    options: MappingReadOptions,
    sheets: Vec<String>,
    tables: Vec<String>,
    columns: Vec<String>,
    preview: Vec<Vec<String>>,
    total: usize,
    source: usize,
    output: usize,
}

/// A running camera stream.
struct Camera {
    stop: Arc<AtomicBool>,
    /// Find faces in every frame.
    live: Arc<AtomicBool>,
    /// A frame is on its way to the screen; the stream drops frames until it arrives, so a busy
    /// UI never builds a queue of them.
    busy: Arc<AtomicBool>,
}

/// What the camera is asked for; `WebcamCapture` picks the nearest format it offers.
const CAMERA_SIZE: (u32, u32) = (1280, 720);
const CAMERA_FPS: u32 = 30;

/// Rows of a mapping file shown in the sidebar.
const MAPPING_PREVIEW_ROWS: usize = 12;
/// Extensions read as mapping files rather than photos when dropped or opened.
const MAPPING_EXTENSIONS: &[&str] = &[
    "csv", "tsv", "xlsx", "xls", "xlsm", "ods", "parquet", "pq", "db", "sqlite", "sqlite3",
];

struct DetectJob {
    uid: u64,
    source: Source,
    orient: bool,
    turn: Turn,
    detection: DetectionSettings,
}

struct ExportJob {
    uid: u64,
    source: Source,
    turn: Turn,
    detected: usize,
    faces: Vec<(usize, Face)>,
    mapped: Option<String>,
}

/// One photo's line in the export report: fcs-gui's columns, so the reports compare.
#[derive(Serialize)]
struct ReportRow {
    image: String,
    outcome: &'static str,
    faces_detected: usize,
    faces_exported: usize,
    error: Option<String>,
}

/// The current photo's faces and their export selection, as undo restores them.
type Snapshot = (Vec<Face>, Vec<bool>);
/// Undo steps kept per photo, as in fcs-gui.
const UNDO_LIMIT: usize = 50;
/// The smallest box an edit leaves, in photo pixels.
const MIN_BOX: f32 = 8.0;
/// Canvas copies kept in memory: up to ~12 MB each at 2048 px, so a few photos back and forth
/// switch instantly without holding a whole batch.
const CANVAS_CACHE: usize = 8;
/// Activity log lines kept.
const LOG_LIMIT: usize = 300;

pub struct Studio {
    settings: AppSettings,
    settings_path: PathBuf,
    engine: Option<Arc<Engine>>,
    /// The engine detection jobs run on, read as each job starts, so a backend switch takes
    /// effect for photos already queued.
    engine_slot: Arc<RwLock<Option<Arc<Engine>>>>,
    jobs: Option<mpsc::Sender<DetectJob>>,
    entries: Vec<Entry>,
    next_uid: u64,
    /// The decoded current photo, kept for previews. Only one: a batch of camera files would
    /// not fit in memory.
    image: Option<(u64, Arc<DynamicImage>)>,
    preview_running: bool,
    preview_dirty: bool,
    preview_seq: u64,
    /// Image-store keys of the previews on screen.
    preview_keys: Vec<String>,
    /// Photos whose canvas copy is in the image store, oldest first.
    canvases: VecDeque<u64>,
    /// Edits of the current photo; both cleared when another photo is selected.
    undo_stack: Vec<Snapshot>,
    redo_stack: Vec<Snapshot>,
    started: Instant,

    ready: bool,
    engine_label: String,
    engine_error: String,
    /// A backend switch is loading the new engine.
    switching: bool,
    /// The CUDA probe's verdict: the device name, why CUDA is unusable, or empty while it runs.
    cuda_status: String,
    cuda_available: bool,
    cuda_warning: String,
    pending: i32,
    queue: Vec<Value>,
    selected_total: i32,
    current: i32,
    image_url: String,
    image_width: i32,
    image_height: i32,
    faces: Vec<Value>,
    can_undo: bool,
    can_redo: bool,
    quarter_turns: i32,
    tilt: f64,
    previews: Vec<Value>,
    presets: Vec<Value>,
    image_filter: String,
    shapes: Vec<Value>,
    /// `settings`, serialized: QML binds to `Studio.settings.crop.face_height_pct` and the like.
    settings_json: Value,
    /// The crop shape's outline for the current output size, in fractions of the crop.
    shape_outline: Vec<Value>,
    /// The preset's canonical name ("LinkedIn"); settings may hold any case ("linkedin", as
    /// fcs-gui saves it). Empty for Custom.
    preset_name: String,
    exporting: bool,
    /// The last export's rows, for saving its report afterwards.
    last_report: Vec<ReportRow>,
    has_report: bool,
    export_progress: f64,
    log: Vec<Value>,
    mapping: Option<MappingFile>,
    /// `mapping`, as the sidebar shows it.
    mapping_json: Value,
    camera: Option<Camera>,
    /// The newest camera frame, kept for a snapshot.
    camera_frame: Option<Arc<DynamicImage>>,
    camera_frames: u64,
    snapshots: u32,
    pastes: u32,
    camera_open: bool,
    camera_live: bool,
    camera_status: String,
    camera_url: String,
    camera_faces: Vec<Value>,
    cameras: Vec<Value>,
    /// For the About box: the app (workspace) version and the Qt it runs on.
    version: String,
    qt_version: String,
}

impl Default for Studio {
    fn default() -> Self {
        let settings_path = default_settings_path();
        // Same file as fcs-gui, so both front ends start from the same crop.
        let settings = AppSettings::load_from_path(&settings_path).unwrap_or_else(|err| {
            warn!(
                "Failed to load settings from {}: {err}",
                settings_path.display()
            );
            AppSettings::default()
        });
        let presets = fcs_core::standard_presets()
            .iter()
            .filter(|p| p.width > 0)
            .map(|p| json!({ "name": p.name, "width": p.width, "height": p.height }))
            .collect();
        let mut studio = Self {
            settings,
            settings_path,
            engine: None,
            engine_slot: Arc::new(RwLock::new(None)),
            jobs: None,
            entries: Vec::new(),
            next_uid: 0,
            image: None,
            preview_running: false,
            preview_dirty: false,
            preview_seq: 0,
            preview_keys: Vec::new(),
            canvases: VecDeque::new(),
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            started: Instant::now(),
            ready: false,
            engine_label: String::new(),
            engine_error: String::new(),
            switching: false,
            cuda_status: String::new(),
            cuda_available: false,
            cuda_warning: fcs_utils::cuda::WARNING.into(),
            pending: 0,
            queue: Vec::new(),
            selected_total: 0,
            current: -1,
            image_url: String::new(),
            image_width: 0,
            image_height: 0,
            faces: Vec::new(),
            can_undo: false,
            can_redo: false,
            quarter_turns: 0,
            tilt: 0.0,
            previews: Vec::new(),
            presets,
            image_filter: format!(
                "Images ({})",
                fcs_utils::SUPPORTED_IMAGE_EXTENSIONS
                    .iter()
                    .map(|e| format!("*.{e}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
            shapes: fcs_utils::shape_catalogue()
                .into_iter()
                .map(|(label, shape)| {
                    json!({ "label": label, "outline": outline(&shape, 1.0, 1.0), "shape": shape })
                })
                .collect(),
            settings_json: Value::Null,
            shape_outline: Vec::new(),
            preset_name: String::new(),
            exporting: false,
            last_report: Vec::new(),
            has_report: false,
            export_progress: 0.0,
            log: Vec::new(),
            mapping: None,
            mapping_json: json!({ "loaded": false }),
            camera: None,
            camera_frame: None,
            camera_frames: 0,
            snapshots: 0,
            pastes: 0,
            camera_open: false,
            camera_live: false,
            camera_status: String::new(),
            camera_url: String::new(),
            camera_faces: Vec::new(),
            cameras: Vec::new(),
            version: fcs_core::version().to_string(),
            qt_version: images::qt_version(),
        };
        studio.sync_settings();
        studio
    }
}

#[qobject(Singleton)]
impl Studio {
    qproperty!("ready", Member = ready, Notify = engine_changed);
    qproperty!(
        "engineLabel",
        Member = engine_label,
        Notify = engine_changed
    );
    qproperty!(
        "engineError",
        Member = engine_error,
        Notify = engine_changed
    );
    qproperty!("switching", Member = switching, Notify = engine_changed);
    qproperty!("cudaStatus", Member = cuda_status, Notify = engine_changed);
    qproperty!(
        "cudaAvailable",
        Member = cuda_available,
        Notify = engine_changed
    );
    qproperty!("cudaWarning", Member = cuda_warning, Constant);
    qproperty!("version", Member = version, Constant);
    qproperty!("qtVersion", Member = qt_version, Constant);
    qproperty!("pending", Member = pending, Notify = queue_changed);
    qproperty!("queue", Member = queue, Notify = queue_changed);
    qproperty!(
        "selectedTotal",
        Member = selected_total,
        Notify = queue_changed
    );
    qproperty!("current", Member = current, Notify = image_changed);
    qproperty!("imageUrl", Member = image_url, Notify = image_changed);
    qproperty!("imageWidth", Member = image_width, Notify = image_changed);
    qproperty!("imageHeight", Member = image_height, Notify = image_changed);
    qproperty!("faces", Member = faces, Notify = image_changed);
    qproperty!("canUndo", Member = can_undo, Notify = image_changed);
    qproperty!("canRedo", Member = can_redo, Notify = image_changed);
    qproperty!(
        "quarterTurns",
        Member = quarter_turns,
        Notify = image_changed
    );
    qproperty!("tilt", Member = tilt, Notify = image_changed);
    qproperty!("previews", Member = previews, Notify = previews_changed);
    qproperty!("presets", Member = presets, Constant);
    qproperty!("imageFilter", Member = image_filter, Constant);
    qproperty!("shapes", Member = shapes, Constant);
    qproperty!(
        "settings",
        Member = settings_json,
        Notify = settings_changed
    );
    qproperty!(
        "presetName",
        Member = preset_name,
        Notify = settings_changed
    );
    qproperty!(
        "shapeOutline",
        Member = shape_outline,
        Notify = settings_changed
    );
    qproperty!("exporting", Member = exporting, Notify = export_changed);
    qproperty!("hasReport", Member = has_report, Notify = export_changed);
    qproperty!(
        "exportProgress",
        Member = export_progress,
        Notify = export_changed
    );
    qproperty!("log", Member = log, Notify = log_changed);
    qproperty!("mapping", Member = mapping_json, Notify = mapping_changed);
    qproperty!("cameraOpen", Member = camera_open, Notify = camera_changed);
    qproperty!("cameraLive", Member = camera_live, Notify = camera_changed);
    qproperty!(
        "cameraStatus",
        Member = camera_status,
        Notify = camera_changed
    );
    qproperty!("cameras", Member = cameras, Notify = camera_changed);
    qproperty!(
        "cameraUrl",
        Member = camera_url,
        Notify = camera_frame_changed
    );
    qproperty!(
        "cameraFaces",
        Member = camera_faces,
        Notify = camera_frame_changed
    );

    #[qsignal(qml_name = "engineChanged")]
    fn engine_changed(&mut self);
    #[qsignal(qml_name = "queueChanged")]
    fn queue_changed(&mut self);
    #[qsignal(qml_name = "imageChanged")]
    fn image_changed(&mut self);
    #[qsignal(qml_name = "previewsChanged")]
    fn previews_changed(&mut self);
    #[qsignal(qml_name = "settingsChanged")]
    fn settings_changed(&mut self);
    #[qsignal(qml_name = "exportChanged")]
    fn export_changed(&mut self);
    #[qsignal(qml_name = "logChanged")]
    fn log_changed(&mut self);
    #[qsignal(qml_name = "mappingChanged")]
    fn mapping_changed(&mut self);
    #[qsignal(qml_name = "cameraChanged")]
    fn camera_changed(&mut self);
    #[qsignal(qml_name = "cameraFrameChanged")]
    fn camera_frame_changed(&mut self);
    /// A transient message; `url` is a folder to offer opening, or empty.
    #[qsignal]
    fn toast(&mut self, message: String, url: String);

    /// Load the models off the UI thread. Called once the window is up, so the user looks at
    /// the app rather than at nothing while shaders compile.
    #[qslot]
    fn start(&mut self) {
        // Photos named on the command line ("Open with", or dropped on the exe).
        let args: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
        if !args.is_empty() {
            self.add_paths(args);
        }
        let handle = self.get_thread_handle();
        let settings = self.settings.clone();
        thread::spawn(move || {
            let engine = Engine::load(&settings).map_err(|err| format!("{err:#}"));
            handle.invoke_mut(move |studio| studio.engine_loaded(engine));
        });
        // The probe compiles PTX and opens a driver context, so it runs once, off this thread,
        // and the backend picker learns the answer when it arrives.
        let handle = self.get_thread_handle();
        thread::spawn(move || {
            let verdict = loop {
                if let Some(verdict) = fcs_utils::cuda::availability() {
                    break verdict.clone();
                }
                thread::sleep(std::time::Duration::from_millis(100));
            };
            handle.invoke_mut(move |studio| {
                studio.cuda_available = verdict.is_ok();
                studio.cuda_status = verdict.unwrap_or_else(|reason| reason);
                studio.engine_changed();
            });
        });
    }

    /// Queue photos, or every supported photo in a dropped folder.
    #[qslot(qml_name = "addFiles")]
    fn add_files(&mut self, urls: Vec<String>) {
        self.add_paths(urls.iter().map(|u| pipeline::url_to_path(u)).collect());
    }

    #[qslot]
    fn select(&mut self, index: i32) {
        if index == self.current || index < 0 || index as usize >= self.entries.len() {
            return;
        }
        self.current = index;
        self.current_changed();
    }

    /// Take photo `index` out of the queue (not off the disk).
    #[qslot]
    fn remove(&mut self, index: i32) {
        let Some(position) = usize::try_from(index)
            .ok()
            .filter(|&i| i < self.entries.len())
        else {
            return;
        };
        let entry = self.entries.remove(position);
        self.forget(&entry);
        self.note(
            "info",
            format!("Removed {}", file_name(entry.source.name())),
        );
        let current = self.current;
        if current > index {
            // The same photo stays current; its position moved up.
            self.current -= 1;
            self.publish_queue();
            self.publish_image();
        } else if current == index {
            self.current = index.min(self.entries.len() as i32 - 1);
            self.publish_queue();
            self.current_changed();
        } else {
            self.publish_queue();
        }
    }

    /// Empty the queue.
    #[qslot(qml_name = "clearQueue")]
    fn clear_queue(&mut self) {
        let removed = std::mem::take(&mut self.entries);
        for entry in &removed {
            self.forget(entry);
        }
        if !removed.is_empty() {
            self.note(
                "info",
                format!("Cleared {} from the queue", photos(removed.len())),
            );
        }
        self.current = -1;
        self.publish_queue();
        self.current_changed();
    }

    #[qslot(qml_name = "toggleFace")]
    fn toggle_face(&mut self, face: i32) {
        let Some(entry) = self.current_entry_mut() else {
            return;
        };
        if let Some(on) = entry.selected.get_mut(face as usize) {
            *on = !*on;
            self.selection_edited();
        }
    }

    #[qslot(qml_name = "selectAllFaces")]
    fn select_all_faces(&mut self, on: bool) {
        if let Some(entry) = self.current_entry_mut() {
            entry.selected.iter_mut().for_each(|s| *s = on);
            self.selection_edited();
        }
    }

    #[qslot(qml_name = "setPreset")]
    fn set_preset(&mut self, name: String) {
        fcs_core::apply_preset(&mut self.settings.crop, &name);
        self.settings_edited();
    }

    /// Move or resize face `index` of the current photo; the box is in fractions of the photo.
    #[qslot(qml_name = "setFaceBox")]
    fn set_face_box(&mut self, index: i32, x: f64, y: f64, w: f64, h: f64) {
        self.edit_faces(|entry| {
            let bbox = clamped_box(entry.size, x, y, w, h);
            let face = entry.faces.get_mut(usize::try_from(index).ok()?)?;
            (face.detection.bbox != bbox).then(|| face.detection.bbox = bbox)
        });
    }

    /// Add a face the detector missed, selected for export.
    #[qslot(qml_name = "addFace")]
    fn add_face(&mut self, x: f64, y: f64, w: f64, h: f64) {
        self.edit_faces(|entry| {
            entry
                .faces
                .push(Face::manual(clamped_box(entry.size, x, y, w, h)));
            entry.selected.push(true);
            Some(())
        });
    }

    #[qslot(qml_name = "deleteFace")]
    fn delete_face(&mut self, index: i32) {
        self.edit_faces(|entry| {
            let index = usize::try_from(index)
                .ok()
                .filter(|&i| i < entry.faces.len())?;
            entry.faces.remove(index);
            entry.selected.remove(index);
            Some(())
        });
    }

    /// Put face `index` back where the detector found it.
    #[qslot(qml_name = "resetFace")]
    fn reset_face(&mut self, index: i32) {
        self.edit_faces(|entry| {
            let face = entry.faces.get_mut(usize::try_from(index).ok()?)?;
            (face.detection.bbox != face.original).then(|| face.detection.bbox = face.original)
        });
    }

    #[qslot]
    fn undo(&mut self) {
        self.step_history(true);
    }

    #[qslot]
    fn redo(&mut self) {
        self.step_history(false);
    }

    /// Turn the current photo by `quarters` clockwise quarter turns and find its faces again.
    #[qslot]
    fn rotate(&mut self, quarters: i32) {
        self.turn_current(|turn| {
            turn.quarters = (turn.quarters as i32 + quarters).rem_euclid(4) as u8
        });
    }

    /// Straighten the current photo by `degrees` and find its faces again.
    #[qslot(qml_name = "setTilt")]
    fn set_tilt(&mut self, degrees: f64) {
        self.turn_current(|turn| turn.tilt_deg = degrees.clamp(-45.0, 45.0) as f32);
    }

    /// Change any settings: `patch` is merged into the serialized `AppSettings`, so
    /// `{"crop": {"face_height_pct": 60}}` changes one field and keeps the rest.
    #[qslot]
    fn update(&mut self, patch: Value) {
        let mut value = match serde_json::to_value(&self.settings) {
            Ok(value) => value,
            Err(err) => return warn!("Failed to serialize settings: {err}"),
        };
        merge(&mut value, patch);
        match serde_json::from_value::<AppSettings>(value) {
            Ok(settings) => self.apply_settings(settings),
            Err(err) => {
                warn!("Rejected a settings change: {err}");
                self.toast(format!("Couldn't apply that setting: {err}"), String::new());
            }
        }
    }

    /// Every setting back to how the app ships (fcs-gui's Restore Defaults returns to the
    /// settings it launched with instead; fcs-qt saves every change, so that point is gone).
    #[qslot(qml_name = "restoreDefaults")]
    fn restore_defaults(&mut self) {
        self.apply_settings(AppSettings::default());
        self.note("info", "Settings restored to their defaults".into());
        self.toast("Settings restored to their defaults".into(), String::new());
    }

    #[qslot(qml_name = "restoreEnhance")]
    fn restore_enhance(&mut self) {
        self.settings.enhance = Default::default();
        self.settings_edited();
    }

    /// How many files an export into `folder_url` would find already there, so the user can
    /// choose to replace or keep them first. Counts every name a face could take, with each
    /// quality suffix when suffixes are on.
    #[qslot]
    fn conflicts(&mut self, folder_url: String) -> i32 {
        let out_dir = pipeline::url_to_path(&folder_url);
        let suffixes: &[&str] = if self.settings.crop.quality_rules.quality_suffix {
            &["_highq", "_medq", "_lowq"]
        } else {
            &[""]
        };
        let mut count = 0;
        for job in self.export_jobs() {
            for (index, _) in &job.faces {
                count += suffixes
                    .iter()
                    .filter(|suffix| {
                        let naming = pipeline::Naming {
                            mapped: job.mapped.as_deref(),
                            multi_face: job.faces.len() > 1,
                            suffix,
                        };
                        pipeline::face_path(
                            job.source.name(),
                            *index,
                            &naming,
                            &self.settings,
                            &out_dir,
                        )
                        .exists()
                    })
                    .count() as i32;
            }
        }
        count
    }

    /// Save every selected face of every finished photo into `folder_url`. `policy` is
    /// "replace" or "keep" (existing files stay, new ones get a `(2)` suffix); `report` is
    /// "none", "csv" or "json".
    #[qslot(qml_name = "exportTo")]
    fn export_to(&mut self, folder_url: String, policy: String, report: String) {
        let Some(engine) = self.engine.clone().filter(|_| !self.exporting) else {
            return;
        };
        let jobs = self.export_jobs();
        let total: usize = jobs.iter().map(|job| job.faces.len()).sum();
        if total == 0 {
            self.toast("Select at least one face to export".into(), String::new());
            return;
        }
        self.exporting = true;
        self.export_progress = 0.0;
        self.export_changed();
        self.note(
            "info",
            format!("Exporting {} from {}", faces(total), photos(jobs.len())),
        );

        let out_dir = pipeline::url_to_path(&folder_url);
        let overwrite = if policy == "replace" {
            OverwritePolicy::Overwrite
        } else {
            OverwritePolicy::KeepBoth
        };
        let settings = self.settings.clone();
        let current = self.image.clone();
        let handle = self.get_thread_handle();
        thread::spawn(move || {
            let claims = OutputClaims::new(overwrite);
            let mut totals = PhotoExport::default();
            let mut rows = Vec::with_capacity(jobs.len());
            let mut done = 0usize;
            for job in jobs {
                let image = match &current {
                    Some((uid, image)) if *uid == job.uid => Ok(image.clone()),
                    _ => pipeline::load(&job.source, settings.crop.auto_orient_exif, job.turn)
                        .map(Arc::new),
                };
                let (result, error) = match image {
                    Ok(image) => (
                        pipeline::export_photo(
                            &engine,
                            &image,
                            &job.source,
                            &job.faces,
                            job.mapped.as_deref(),
                            &settings,
                            &out_dir,
                            &claims,
                        ),
                        None,
                    ),
                    Err(err) => {
                        warn!("{err:#}");
                        let failed = PhotoExport {
                            failed: job.faces.len(),
                            ..PhotoExport::default()
                        };
                        (failed, Some(format!("{err:#}")))
                    }
                };
                rows.push(ReportRow {
                    image: job.source.name().display().to_string(),
                    outcome: if result.failed > 0 {
                        "failed"
                    } else {
                        "succeeded"
                    },
                    faces_detected: job.detected,
                    faces_exported: result.exported,
                    error: error.or_else(|| {
                        (result.failed > 0)
                            .then(|| format!("{} faces failed to save", result.failed))
                    }),
                });
                totals.exported += result.exported;
                totals.skipped += result.skipped;
                totals.failed += result.failed;
                done += job.faces.len();
                let progress = done as f64 / total as f64;
                handle.invoke_mut(move |studio| {
                    studio.export_progress = progress;
                    studio.export_changed();
                });
            }
            let report = write_report(&rows, &report, &out_dir, &claims);
            handle.invoke_mut(move |studio| {
                studio.last_report = rows;
                studio.export_finished(totals, report, folder_url)
            });
        });
    }

    /// Save the last export's report to `url`: JSON for a `.json` name, CSV otherwise.
    #[qslot(qml_name = "saveReport")]
    fn save_report(&mut self, url: String) {
        let path = pipeline::url_to_path(&url);
        let json = path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("json"));
        let written = encode_report(&self.last_report, if json { "json" } else { "csv" })
            .expect("csv and json always encode")
            .and_then(|bytes| fcs_utils::write_atomically(&path, &bytes));
        self.saved(written, "Report", &path);
    }

    /// Save the queued photos' paths to `url`, one per line, as fcs-gui's queue list does.
    /// Snapshots and pastes have no file, so they are left out.
    #[qslot(qml_name = "saveQueueList")]
    fn save_queue_list(&mut self, url: String) {
        let path = pipeline::url_to_path(&url);
        let lines: Vec<String> = self
            .entries
            .iter()
            .filter_map(|e| e.source.file())
            .map(|p| p.display().to_string())
            .collect();
        let written = fcs_utils::write_atomically(&path, lines.join("\n").as_bytes());
        self.saved(written, "Queue list", &path);
    }

    /// Read a mapping file (CSV, Excel, Parquet or SQLite) whose rows name exported faces.
    #[qslot(qml_name = "loadMapping")]
    fn load_mapping(&mut self, url: String) {
        self.open_mapping(pipeline::url_to_path(&url));
    }

    /// Read another sheet of the open workbook.
    #[qslot(qml_name = "setMappingSheet")]
    fn set_mapping_sheet(&mut self, sheet: String) {
        if let Some(mapping) = &mut self.mapping {
            mapping.options.sheet_name = Some(sheet);
            self.refresh_mapping(true);
        }
    }

    /// Read another table of the open database.
    #[qslot(qml_name = "setMappingTable")]
    fn set_mapping_table(&mut self, table: String) {
        if let Some(mapping) = &mut self.mapping {
            mapping.options.sql_table = Some(table);
            self.refresh_mapping(true);
        }
    }

    /// Choose which columns hold the photo and its output name.
    #[qslot(qml_name = "setMappingColumns")]
    fn set_mapping_columns(&mut self, source: i32, output: i32) {
        if let Some(mapping) = &mut self.mapping {
            let last = mapping.columns.len().saturating_sub(1);
            mapping.source = (source.max(0) as usize).min(last);
            mapping.output = (output.max(0) as usize).min(last);
            self.apply_mapping();
        }
    }

    #[qslot(qml_name = "clearMapping")]
    fn clear_mapping(&mut self) {
        if self.mapping.take().is_some() {
            self.note(
                "info",
                "Mapping cleared: exports use their usual names".into(),
            );
        }
        self.entries.iter_mut().for_each(|e| e.mapped = None);
        self.publish_mapping(None);
        self.publish_queue();
    }

    /// Start streaming camera `device` (an index into `cameras`, which this refreshes).
    #[qslot(qml_name = "openCamera")]
    fn open_camera(&mut self, device: i32) {
        self.stop_camera();
        let devices = fcs_utils::list_webcam_devices().unwrap_or_else(|err| {
            warn!("Failed to list cameras: {err:#}");
            Vec::new()
        });
        self.cameras = devices
            .iter()
            .map(|(index, name)| json!({ "index": index, "name": name }))
            .collect();
        self.camera_open = true;
        self.camera_live = false;
        let Some(&(index, ref name)) = devices.get(device.max(0) as usize).or(devices.first())
        else {
            self.camera_status = "No camera found".into();
            return self.camera_changed();
        };
        self.camera_status = format!("Starting {name}…");
        self.camera_changed();

        let camera = Camera {
            stop: Arc::new(AtomicBool::new(false)),
            live: Arc::new(AtomicBool::new(false)),
            busy: Arc::new(AtomicBool::new(false)),
        };
        let (stop, live, busy) = (
            camera.stop.clone(),
            camera.live.clone(),
            camera.busy.clone(),
        );
        self.camera = Some(camera);
        let engine = self.engine.clone();
        let detection = self.settings.detection.clone();
        let handle = self.get_thread_handle();
        let name = name.clone();
        thread::spawn(move || {
            let (width, height) = CAMERA_SIZE;
            let mut cam =
                match fcs_utils::WebcamCapture::with_device_index(index, width, height, CAMERA_FPS)
                {
                    Ok(cam) => cam,
                    Err(err) => {
                        let message = format!("Couldn't open {name}: {err:#}");
                        return handle.invoke_mut(move |studio| studio.camera_failed(message));
                    }
                };
            let (w, h) = cam.resolution();
            handle.invoke_mut(move |studio| studio.camera_started(name, w, h));
            let detector = engine
                .as_ref()
                .map(|e| e.detector.with_settings(&detection));
            while !stop.load(Ordering::Relaxed) {
                let frame = match cam.capture_frame() {
                    Ok(frame) => frame,
                    Err(err) => {
                        let message = format!("The camera stopped: {err:#}");
                        handle.invoke_mut(move |studio| studio.camera_failed(message));
                        break;
                    }
                };
                if busy.swap(true, Ordering::AcqRel) {
                    // The last frame is not on screen yet: skip this one rather than queue it.
                    continue;
                }
                let faces = match (&detector, live.load(Ordering::Relaxed)) {
                    (Some(detector), true) => detector
                        .detect_image(&frame)
                        .map(|out| frame_faces(&out.detections, frame.width(), frame.height()))
                        .unwrap_or_default(),
                    _ => Vec::new(),
                };
                let rgb = frame.to_rgb8();
                let frame = Arc::new(frame);
                handle.invoke_mut(move |studio| studio.camera_frame_ready(frame, rgb, faces));
            }
            let _ = cam.stop();
        });
    }

    /// Find faces in every frame, or stop.
    #[qslot(qml_name = "setCameraLive")]
    fn set_camera_live(&mut self, on: bool) {
        if let Some(camera) = &self.camera {
            camera.live.store(on, Ordering::Relaxed);
            self.camera_live = on;
            if !on {
                // Boxes from a frame already gone would read as live ones.
                self.camera_faces.clear();
                self.camera_frame_changed();
            }
            self.camera_changed();
        }
    }

    /// Add the frame on screen to the photos, held in memory.
    #[qslot]
    fn snapshot(&mut self) {
        let Some(frame) = self.camera_frame.clone() else {
            return self.toast("No camera frame yet".into(), String::new());
        };
        self.snapshots += 1;
        let name = PathBuf::from(format!("Camera {}.png", self.snapshots));
        let label = file_name(&name);
        self.note("info", format!("Snapshot added as {label}"));
        self.add_memory(name, frame);
    }

    /// Add the clipboard's image to the photos, held in memory.
    #[qslot]
    fn paste(&mut self) {
        let Some(image) = images::clipboard() else {
            return self.toast("The clipboard holds no image".into(), String::new());
        };
        self.pastes += 1;
        let name = PathBuf::from(format!("Pasted {}.png", self.pastes));
        let label = file_name(&name);
        self.note("info", format!("Pasted image added as {label}"));
        self.add_memory(name, Arc::new(DynamicImage::ImageRgba8(image)));
    }

    #[qslot(qml_name = "closeCamera")]
    fn close_camera(&mut self) {
        self.stop_camera();
        self.camera_open = false;
        self.camera_changed();
    }
}

impl Studio {
    /// Queue an image that has no file (a snapshot or a paste) and show it.
    fn add_memory(&mut self, name: PathBuf, image: Arc<DynamicImage>) {
        let label = file_name(&name);
        let entry = Entry {
            uid: self.next_uid,
            source: Source::Memory { name, image },
            turn: Turn::default(),
            rev: 0,
            status: Status::Queued,
            size: (0, 0),
            faces: Vec::new(),
            selected: Vec::new(),
            mapped: None,
        };
        self.next_uid += 1;
        self.send_detection(&entry);
        self.entries.push(entry);
        self.pending += 1;
        self.toast(format!("Added {label} to the photos"), String::new());
        self.current = self.entries.len() as i32 - 1;
        if self.mapping.is_some() {
            self.apply_mapping();
        }
        self.publish_queue();
        self.current_changed();
    }

    fn add_paths(&mut self, dropped: Vec<PathBuf>) {
        let mut paths = Vec::new();
        // Absolute, as dialogs and drops give them: command-line paths are relative to wherever
        // the app was started, and the queue list and reports must not depend on that.
        let dropped = dropped
            .into_iter()
            .map(|p| std::path::absolute(&p).unwrap_or(p));
        for path in dropped {
            if path.is_dir() {
                let mut found: Vec<_> = fs::read_dir(&path)
                    .into_iter()
                    .flatten()
                    .flatten()
                    .map(|e| e.path())
                    .filter(|p| is_supported_image_path(p))
                    .collect();
                found.sort();
                paths.extend(found);
            } else if is_supported_image_path(&path) {
                paths.push(path);
            } else if is_mapping_file(&path) {
                // A spreadsheet dropped with the photos names their exports.
                self.open_mapping(path);
            }
        }
        paths.retain(|p| {
            !self
                .entries
                .iter()
                .any(|e| e.source.file() == Some(p.as_path()))
        });
        if paths.is_empty() {
            if self.mapping.is_none() {
                self.toast("No new supported photos there".into(), String::new());
            }
            return;
        }
        self.note("info", format!("Added {}", photos(paths.len())));
        for path in paths {
            let entry = Entry {
                uid: self.next_uid,
                source: Source::File(path),
                turn: Turn::default(),
                rev: 0,
                status: Status::Queued,
                size: (0, 0),
                faces: Vec::new(),
                selected: Vec::new(),
                mapped: None,
            };
            self.next_uid += 1;
            self.send_detection(&entry);
            self.entries.push(entry);
            self.pending += 1;
        }
        if self.current < 0 {
            self.current = 0;
            self.publish_image();
        }
        if self.mapping.is_some() {
            // The new photos may be in the open mapping.
            self.apply_mapping();
        }
        self.publish_queue();
    }

    fn stop_camera(&mut self) {
        if let Some(camera) = self.camera.take() {
            camera.stop.store(true, Ordering::Relaxed);
            self.note("info", "Camera off".into());
        }
        self.camera_frame = None;
        self.camera_live = false;
        self.camera_url.clear();
        self.camera_faces.clear();
        images::remove("camera/0");
        images::remove("camera/1");
        self.camera_frame_changed();
    }

    fn camera_started(&mut self, name: String, width: u32, height: u32) {
        if self.camera.is_some() {
            self.camera_status = format!("{name} · {width}×{height}");
            self.note("ok", format!("Camera on: {name}, {width}×{height}"));
            self.camera_changed();
        }
    }

    fn camera_failed(&mut self, message: String) {
        warn!("{message}");
        self.note("error", message.clone());
        self.camera = None;
        self.camera_status = message;
        self.camera_changed();
    }

    fn camera_frame_ready(&mut self, frame: Arc<DynamicImage>, rgb: RgbImage, faces: Vec<Value>) {
        // Closed while this frame was on its way.
        let Some(camera) = &self.camera else {
            return;
        };
        camera.busy.store(false, Ordering::Release);
        self.camera_frames += 1;
        // Alternating keys: a new URL each frame, without keeping more than two frames.
        let key = format!("camera/{}", self.camera_frames % 2);
        self.camera_url = images::put(&key, Pixels::Rgb(rgb));
        self.camera_faces = faces;
        self.camera_frame = Some(frame);
        self.camera_frame_changed();
    }

    fn open_mapping(&mut self, path: PathBuf) {
        let format = fcs_mapping::detect_format(&path);
        let options = MappingReadOptions {
            preview_rows: MAPPING_PREVIEW_ROWS,
            ..MappingReadOptions::default()
        };
        let catalog = match fcs_mapping::inspect_mapping_sources(&path, &options) {
            Ok(catalog) => catalog,
            Err(err) => return self.mapping_failed(&path, err),
        };
        let options = MappingReadOptions {
            sheet_name: catalog.sheets.first().cloned(),
            sql_table: catalog.sql_tables.first().cloned(),
            ..options
        };
        self.mapping = Some(MappingFile {
            path,
            format,
            options,
            sheets: catalog.sheets,
            tables: catalog.sql_tables,
            columns: Vec::new(),
            preview: Vec::new(),
            total: 0,
            source: 0,
            output: 0,
        });
        self.refresh_mapping(true);
    }

    /// Re-read the preview (another sheet or table, or a new file), guess the columns when
    /// `guess` is set, then apply.
    fn refresh_mapping(&mut self, guess: bool) {
        let Some(mapping) = &mut self.mapping else {
            return;
        };
        let preview = match fcs_mapping::load_mapping_preview(&mapping.path, &mapping.options) {
            Ok(preview) => preview,
            Err(err) => {
                let path = mapping.path.clone();
                return self.mapping_failed(&path, err);
            }
        };
        mapping.columns = preview.columns;
        mapping.preview = preview.rows;
        mapping.total = preview.total_rows;
        if guess {
            let (source, output) = guess_mapping_columns(&mapping.columns);
            mapping.source = source;
            mapping.output = output;
        }
        self.apply_mapping();
    }

    /// Give every queued photo the output name its mapping row holds, or none.
    fn apply_mapping(&mut self) {
        let Some(mapping) = &self.mapping else {
            return;
        };
        let entries = fcs_mapping::load_mapping_entries(
            &mapping.path,
            &mapping.options,
            &ColumnSelector::Index(mapping.source),
            &ColumnSelector::Index(mapping.output),
        );
        let entries = match entries {
            Ok(entries) => entries,
            Err(err) => {
                let path = mapping.path.clone();
                return self.mapping_failed(&path, err);
            }
        };
        let names: Vec<Option<String>> =
            fcs_mapping::match_entries(&entries, self.entries.iter().map(|e| e.source.name()))
                .into_iter()
                .map(|name| name.map(str::to_owned))
                .collect();
        for (entry, name) in self.entries.iter_mut().zip(names) {
            entry.mapped = name;
        }
        let matched = self.entries.iter().filter(|e| e.mapped.is_some()).count();
        let file = file_name(&mapping.path);
        let summary = format!(
            "{file}: {matched} of {} matched",
            photos(self.entries.len())
        );
        self.note("info", summary);
        self.publish_mapping(Some(matched));
        self.publish_queue();
    }

    fn mapping_failed(&mut self, path: &Path, err: anyhow::Error) {
        let message = format!("Couldn't read {}: {err:#}", file_name(path));
        warn!("{message}");
        self.note("error", message.clone());
        self.toast(message, String::new());
    }

    fn publish_mapping(&mut self, matched: Option<usize>) {
        self.mapping_json = match &self.mapping {
            None => json!({ "loaded": false }),
            Some(m) => json!({
                "loaded": true,
                "file": file_name(&m.path),
                "format": m.format.display_name(),
                "sheets": m.sheets,
                "sheet": m.options.sheet_name,
                "tables": m.tables,
                "table": m.options.sql_table,
                "columns": m.columns,
                "preview": m.preview,
                "rows": m.total,
                "source": m.source,
                "output": m.output,
                "matched": matched.unwrap_or(0),
                "photos": self.entries.len(),
            }),
        };
        self.mapping_changed();
    }

    fn engine_loaded(&mut self, engine: Result<Engine, String>) {
        match engine {
            Ok(engine) => {
                self.note("ok", format!("Models ready on {}", engine.label));
                self.install_engine(engine);
                // Photos added while the models loaded.
                for entry in &self.entries {
                    self.send_detection(entry);
                }
            }
            Err(err) => {
                self.note("error", format!("Models failed to load: {err}"));
                self.engine_error = err;
            }
        }
        self.engine_changed();
    }

    /// Make `engine` the one previews, exports and detection jobs use, starting the detection
    /// worker the first time.
    fn install_engine(&mut self, engine: Engine) {
        let engine = Arc::new(engine);
        self.engine_label = engine.label.clone();
        self.engine_error.clear();
        self.ready = true;
        *self.engine_slot.write().unwrap_or_else(|e| e.into_inner()) = Some(engine.clone());
        self.engine = Some(engine);
        if self.jobs.is_some() {
            return;
        }
        let (tx, rx) = mpsc::channel::<DetectJob>();
        let slot = self.engine_slot.clone();
        let handle = self.get_thread_handle();
        thread::spawn(move || {
            for job in rx {
                let engine = slot.read().unwrap_or_else(|e| e.into_inner()).clone();
                let result = match engine {
                    Some(engine) => {
                        pipeline::detect(&engine, &job.source, job.orient, job.turn, &job.detection)
                            .map_err(|err| format!("{err:#}"))
                    }
                    None => Err("no detector".to_string()),
                };
                let uid = job.uid;
                handle.invoke_mut(move |studio| studio.detected(uid, result));
            }
        });
        self.jobs = Some(tx);
    }

    /// Load an engine for the new backend settings in the background. Until it is up, the old
    /// one keeps working; if it fails, the settings go back to `previous` and it stays.
    fn rebuild_engine(&mut self, previous: GpuSettings) {
        if self.switching {
            return;
        }
        self.switching = true;
        self.ready = false;
        let target = backend_name(&self.settings.gpu);
        self.engine_label = format!("Switching to {target}…");
        self.note("info", format!("Switching to {target}"));
        self.engine_changed();
        let settings = self.settings.clone();
        let handle = self.get_thread_handle();
        thread::spawn(move || {
            let engine = Engine::load(&settings).map_err(|err| format!("{err:#}"));
            handle.invoke_mut(move |studio| studio.engine_rebuilt(engine, previous));
        });
    }

    fn engine_rebuilt(&mut self, engine: Result<Engine, String>, previous: GpuSettings) {
        self.switching = false;
        match engine {
            Ok(engine) => {
                let label = engine.label.clone();
                self.install_engine(engine);
                self.note("ok", format!("Now running on {label}"));
                // Boxes found by the old backend can differ a little; find them all again so
                // every photo agrees with what export will use.
                let count = self.redetect_all();
                if count > 0 {
                    self.note("info", format!("Finding faces again in {}", photos(count)));
                }
            }
            Err(err) => {
                let message = format!(
                    "Couldn't switch to {}: {err}",
                    backend_name(&self.settings.gpu)
                );
                warn!("{message}");
                self.note("error", message.clone());
                self.toast(message, String::new());
                self.settings.gpu = previous;
                self.settings_edited();
                self.ready = self.engine.is_some();
                if let Some(engine) = &self.engine {
                    self.engine_label = engine.label.clone();
                }
            }
        }
        self.engine_changed();
    }

    fn send_detection(&self, entry: &Entry) {
        if let Some(jobs) = &self.jobs {
            let _ = jobs.send(DetectJob {
                uid: entry.uid,
                source: entry.source.clone(),
                orient: self.settings.crop.auto_orient_exif,
                turn: entry.turn,
                detection: self.settings.detection.clone(),
            });
        }
    }

    fn detected(&mut self, uid: u64, result: Result<Detected, String>) {
        // Removed while it was being detected: nothing to update, and `forget` already
        // took it off the pending count.
        let Some(position) = self.position(uid) else {
            return;
        };
        self.pending -= 1;
        let is_current = self.current == position as i32;
        let entry = &mut self.entries[position];
        let name = file_name(entry.source.name());
        match result {
            Ok(detected) => {
                let found = detected.faces.len();
                entry.size = (detected.image.width(), detected.image.height());
                entry.selected = vec![true; found];
                entry.faces = detected.faces;
                entry.status = Status::Done;
                images::put(&thumb_key(uid, entry.rev), Pixels::Rgb(detected.thumb));
                self.cache_canvas(uid, detected.canvas);
                self.note(
                    if found == 0 { "warn" } else { "ok" },
                    format!(
                        "{name}: {}",
                        if found == 0 {
                            "no faces found".into()
                        } else {
                            faces(found)
                        }
                    ),
                );
                if is_current {
                    // New boxes: the old edits no longer apply to anything.
                    self.undo_stack.clear();
                    self.redo_stack.clear();
                    self.image = Some((uid, detected.image));
                    self.publish_image();
                    self.request_previews();
                }
            }
            Err(err) => {
                warn!("{err}");
                self.note("error", format!("{name}: {err}"));
                self.entries[position].status = Status::Failed(err);
                if is_current {
                    self.publish_image();
                }
            }
        }
        self.publish_queue();
    }

    /// Drop everything held for a photo leaving the queue.
    fn forget(&mut self, entry: &Entry) {
        if matches!(entry.status, Status::Queued) {
            self.pending -= 1;
        }
        images::remove(&thumb_key(entry.uid, entry.rev));
        images::remove(&canvas_key(entry.uid));
        self.canvases.retain(|&held| held != entry.uid);
        if self
            .image
            .as_ref()
            .is_some_and(|(held, _)| *held == entry.uid)
        {
            self.image = None;
        }
    }

    /// Another photo became current (or none did): start it fitted and fresh.
    fn current_changed(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
        self.previews.clear();
        self.previews_changed();
        self.publish_image();
        self.load_current_image();
    }

    /// Apply `edit` to the current photo's faces as one undo step. `edit` returns `None` when
    /// it changed nothing, so a no-op leaves no step behind.
    fn edit_faces(&mut self, edit: impl FnOnce(&mut Entry) -> Option<()>) {
        let Some(entry) = self
            .current_entry_mut()
            .filter(|e| matches!(e.status, Status::Done))
        else {
            return;
        };
        let before = (entry.faces.clone(), entry.selected.clone());
        if edit(entry).is_none() {
            return;
        }
        self.undo_stack.push(before);
        if self.undo_stack.len() > UNDO_LIMIT {
            self.undo_stack.remove(0);
        }
        self.redo_stack.clear();
        self.selection_edited();
    }

    fn step_history(&mut self, back: bool) {
        let (from, to) = if back {
            (&mut self.undo_stack, &mut self.redo_stack)
        } else {
            (&mut self.redo_stack, &mut self.undo_stack)
        };
        let Some((faces, selected)) = from.pop() else {
            return;
        };
        let Some(entry) = usize::try_from(self.current)
            .ok()
            .and_then(|i| self.entries.get_mut(i))
        else {
            return;
        };
        to.push((
            std::mem::replace(&mut entry.faces, faces),
            std::mem::replace(&mut entry.selected, selected),
        ));
        self.selection_edited();
    }

    /// Change how the current photo is turned and send it back through detection: the old
    /// boxes were found on differently turned pixels. Its edits and history go with them.
    fn turn_current(&mut self, change: impl FnOnce(&mut Turn)) {
        let Some(entry) = self
            .current_entry_mut()
            .filter(|e| matches!(e.status, Status::Done))
        else {
            return;
        };
        let before = entry.turn;
        change(&mut entry.turn);
        if entry.turn != before {
            self.requeue(self.current as usize);
            self.publish_queue();
        }
    }

    /// Send every finished or failed photo back through detection. Returns how many.
    fn redetect_all(&mut self) -> usize {
        let positions: Vec<usize> = (0..self.entries.len())
            .filter(|&i| !matches!(self.entries[i].status, Status::Queued))
            .collect();
        for &position in &positions {
            self.requeue(position);
        }
        self.publish_queue();
        positions.len()
    }

    /// Clear photo `position`'s faces and detect it again with the current settings.
    fn requeue(&mut self, position: usize) {
        let entry = &mut self.entries[position];
        let uid = entry.uid;
        entry.status = Status::Queued;
        entry.faces.clear();
        entry.selected.clear();
        images::remove(&thumb_key(uid, entry.rev));
        entry.rev += 1;
        images::remove(&canvas_key(uid));
        self.canvases.retain(|&held| held != uid);
        self.pending += 1;
        self.send_detection(&self.entries[position]);
        if self.current == position as i32 {
            self.image = None;
            self.undo_stack.clear();
            self.redo_stack.clear();
            self.previews.clear();
            self.previews_changed();
            self.publish_image();
        }
    }

    fn position(&self, uid: u64) -> Option<usize> {
        self.entries.iter().position(|e| e.uid == uid)
    }

    fn current_entry(&self) -> Option<&Entry> {
        usize::try_from(self.current)
            .ok()
            .and_then(|i| self.entries.get(i))
    }

    fn current_entry_mut(&mut self) -> Option<&mut Entry> {
        usize::try_from(self.current)
            .ok()
            .and_then(|i| self.entries.get_mut(i))
    }

    fn current_uid(&self) -> Option<u64> {
        self.current_entry().map(|e| e.uid)
    }

    /// Every finished photo with a selected face, and those faces.
    fn export_jobs(&self) -> Vec<ExportJob> {
        self.entries
            .iter()
            .filter(|e| matches!(e.status, Status::Done))
            .map(|e| ExportJob {
                uid: e.uid,
                source: e.source.clone(),
                turn: e.turn,
                detected: e.faces.len(),
                mapped: e.mapped.clone(),
                faces: e
                    .faces
                    .iter()
                    .zip(&e.selected)
                    .enumerate()
                    .filter(|(_, (_, on))| **on)
                    .map(|(i, (face, _))| (i, face.clone()))
                    .collect(),
            })
            .filter(|job| !job.faces.is_empty())
            .collect()
    }

    fn selection_edited(&mut self) {
        self.publish_image();
        self.publish_queue();
        self.request_previews();
    }

    /// Make `settings` current, save them, and find faces again if the backend or the
    /// confidence floor moved.
    fn apply_settings(&mut self, mut settings: AppSettings) {
        settings.detection.sanitize();
        let floor = settings.detection.confidence;
        let floor_moved = floor != self.settings.detection.confidence;
        let previous_gpu = self.settings.gpu.clone();
        let backend_moved = gpu_key(&settings.gpu) != gpu_key(&previous_gpu);
        self.settings = settings;
        self.settings_edited();
        if backend_moved {
            // Re-detects everything once the new engine is up.
            return self.rebuild_engine(previous_gpu);
        }
        // The boxes on screen were found under the old floor.
        if floor_moved {
            let count = self.redetect_all();
            if count > 0 {
                self.note(
                    "info",
                    format!(
                        "Confidence floor {floor:.2}: finding faces again in {}",
                        photos(count)
                    ),
                );
            }
        }
    }

    fn settings_edited(&mut self) {
        self.settings.crop.sanitize();
        let saved = self
            .settings_path
            .parent()
            .map_or(Ok(()), fs::create_dir_all)
            .map_err(anyhow::Error::from)
            .and_then(|()| self.settings.save_to_path(&self.settings_path));
        if let Err(err) = saved {
            self.toast(format!("Couldn't save settings: {err}"), String::new());
        }
        self.sync_settings();
        self.settings_changed();
        // The crop overlays on the photo follow the framing.
        self.publish_image();
        self.request_previews();
    }

    fn sync_settings(&mut self) {
        // Cheap atomics; covers start-up and every edit, as fcs-gui does.
        fcs_utils::configure_telemetry(
            self.settings.telemetry.enabled,
            self.settings.telemetry.level_filter(),
        );
        self.settings_json = serde_json::to_value(&self.settings).unwrap_or_default();
        let crop = &self.settings.crop;
        self.preset_name = fcs_core::preset_by_name(&crop.preset)
            .filter(|p| p.width > 0)
            .map_or_else(String::new, |p| p.name.to_string());
        self.shape_outline = outline(
            &crop.shape,
            crop.output_width as f32,
            crop.output_height as f32,
        );
    }

    /// Add a line to the activity log: `kind` is "info", "ok", "warn" or "error".
    fn note(&mut self, kind: &str, text: String) {
        let secs = self.started.elapsed().as_secs();
        self.log.push(json!({
            "time": format!("{:02}:{:02}", secs / 60, secs % 60),
            "kind": kind,
            "text": text,
        }));
        if self.log.len() > LOG_LIMIT {
            self.log.remove(0);
        }
        self.log_changed();
    }

    fn publish_queue(&mut self) {
        self.queue = self
            .entries
            .iter()
            .map(|e| {
                let (status, error) = match &e.status {
                    Status::Queued => ("queued", ""),
                    Status::Done => ("done", ""),
                    Status::Failed(err) => ("failed", err.as_str()),
                };
                json!({
                    "name": file_name(e.source.name()),
                    "thumb": if matches!(e.status, Status::Done) { images::url(&thumb_key(e.uid, e.rev)) } else { String::new() },
                    "status": status,
                    "error": error,
                    "faces": e.faces.len(),
                    "selected": e.selected.iter().filter(|s| **s).count(),
                    "mapped": e.mapped.clone().unwrap_or_default(),
                })
            })
            .collect();
        self.selected_total = self
            .entries
            .iter()
            .map(|e| e.selected.iter().filter(|s| **s).count() as i32)
            .sum();
        self.queue_changed();
    }

    fn publish_image(&mut self) {
        self.can_undo = !self.undo_stack.is_empty();
        self.can_redo = !self.redo_stack.is_empty();
        let turn = self.current_entry().map(|e| e.turn).unwrap_or_default();
        self.quarter_turns = i32::from(turn.quarters);
        self.tilt = f64::from(turn.tilt_deg);
        // Shown once its canvas copy is in the store; until then the canvas says it is opening.
        let Some(entry) = self
            .current_entry()
            .filter(|e| matches!(e.status, Status::Done) && self.canvases.contains(&e.uid))
        else {
            self.image_url.clear();
            self.faces.clear();
            self.image_changed();
            return;
        };
        let (w, h) = (entry.size.0.max(1) as f32, entry.size.1.max(1) as f32);
        let core: fcs_core::CropSettings = (&self.settings.crop).into();
        let image_url = images::url(&canvas_key(entry.uid));
        let size = entry.size;
        let faces = entry
            .faces
            .iter()
            .zip(&entry.selected)
            .enumerate()
            .map(|(i, (face, on))| {
                let b = face.detection.bbox;
                let x = (b.x / w).clamp(0.0, 1.0);
                let y = (b.y / h).clamp(0.0, 1.0);
                let crop = fcs_core::calculate_crop_region(size.0, size.1, b, &core);
                json!({
                    "index": i,
                    "x": x,
                    "y": y,
                    "w": (b.width / w).min(1.0 - x),
                    "h": (b.height / h).min(1.0 - y),
                    "score": face.detection.score,
                    "quality": format!("{:?}", face.quality),
                    "selected": on,
                    "manual": face.manual,
                    "modified": face.detection.bbox != face.original,
                    // Where this face's crop falls, in fractions of the photo: the overlay.
                    "crop": {
                        "x": crop.x as f32 / w,
                        "y": crop.y as f32 / h,
                        "w": crop.width as f32 / w,
                        "h": crop.height as f32 / h,
                    },
                })
            })
            .collect();
        self.image_url = image_url;
        self.image_width = size.0 as i32;
        self.image_height = size.1 as i32;
        self.faces = faces;
        self.image_changed();
    }

    /// Decode the current photo for previews, and remake its canvas copy if that was evicted.
    fn load_current_image(&mut self) {
        let Some(uid) = self.current_uid() else {
            return;
        };
        if self.image.as_ref().is_some_and(|(held, _)| *held == uid) {
            self.request_previews();
            return;
        }
        self.image = None;
        let Some(entry) = self
            .current_entry()
            .filter(|e| matches!(e.status, Status::Done))
        else {
            return;
        };
        let source = entry.source.clone();
        let turn = entry.turn;
        let orient = self.settings.crop.auto_orient_exif;
        let handle = self.get_thread_handle();
        thread::spawn(move || {
            let loaded = pipeline::load(&source, orient, turn)
                .map(|image| (pipeline::canvas(&image), image));
            handle.invoke_mut(move |studio| studio.current_image_loaded(uid, loaded));
        });
    }

    fn current_image_loaded(&mut self, uid: u64, loaded: anyhow::Result<(RgbImage, DynamicImage)>) {
        match loaded {
            Ok((canvas, image)) => {
                if self.current_uid() != Some(uid) {
                    return;
                }
                self.image = Some((uid, Arc::new(image)));
                if !self.canvases.contains(&uid) {
                    self.cache_canvas(uid, canvas);
                    self.publish_image();
                }
                self.request_previews();
            }
            // Read once already, so the file changed or went away since.
            Err(err) => {
                let err = format!("{err:#}");
                warn!("{err}");
                self.note("error", err.clone());
                if let Some(position) = self.position(uid) {
                    self.entries[position].status = Status::Failed(err);
                }
                self.publish_image();
                self.publish_queue();
            }
        }
    }

    /// Hold `canvas` as photo `uid`'s copy, evicting the oldest other one past
    /// [`CANVAS_CACHE`]. Never the current photo's: it is on screen.
    fn cache_canvas(&mut self, uid: u64, canvas: RgbImage) {
        images::put(&canvas_key(uid), Pixels::Rgb(canvas));
        self.canvases.retain(|&held| held != uid);
        self.canvases.push_back(uid);
        let current = self.current_uid();
        while self.canvases.len() > CANVAS_CACHE {
            let Some(oldest) = self.canvases.iter().position(|&held| Some(held) != current) else {
                break;
            };
            if let Some(evicted) = self.canvases.remove(oldest) {
                images::remove(&canvas_key(evicted));
            }
        }
    }

    /// Render the selected faces of the current photo with the current settings. One render
    /// at a time: edits that land meanwhile (a dragged slider) coalesce into one more.
    fn request_previews(&mut self) {
        if self.preview_running {
            self.preview_dirty = true;
            return;
        }
        let (Some(engine), Some((uid, image))) = (self.engine.clone(), self.image.clone()) else {
            return;
        };
        let Some(entry) = self.current_entry().filter(|e| e.uid == uid) else {
            return;
        };
        let faces: Vec<(usize, Face)> = entry
            .faces
            .iter()
            .zip(&entry.selected)
            .enumerate()
            .filter(|(_, (_, on))| **on)
            .map(|(i, (face, _))| (i, face.clone()))
            .collect();
        self.preview_running = true;
        self.preview_seq += 1;
        let seq = self.preview_seq;
        let settings = self.settings.clone();
        let handle = self.get_thread_handle();
        thread::spawn(move || {
            let previews: Vec<(usize, RgbaImage)> = faces
                .into_iter()
                .map(|(i, face)| {
                    let crop = pipeline::finish(&engine, &image, &face.detection, &settings);
                    (i, pipeline::preview(&crop.image))
                })
                .collect();
            handle.invoke_mut(move |studio| studio.previews_ready(uid, seq, previews));
        });
    }

    fn previews_ready(&mut self, uid: u64, seq: u64, previews: Vec<(usize, RgbaImage)>) {
        self.preview_running = false;
        // Fresh keys each render, so QML never shows a cached older one; the last set is
        // already on screen and can leave the store.
        for key in self.preview_keys.drain(..) {
            images::remove(&key);
        }
        if self.current_uid() == Some(uid) {
            let mut keys = Vec::with_capacity(previews.len());
            self.previews = previews
                .into_iter()
                .map(|(i, pixels)| {
                    let key = format!("preview/{seq}/{i}");
                    let url = images::put(&key, Pixels::Rgba(pixels));
                    keys.push(key);
                    json!({ "index": i, "url": url })
                })
                .collect();
            self.preview_keys = keys;
            self.previews_changed();
        }
        if std::mem::take(&mut self.preview_dirty) {
            self.request_previews();
        }
    }

    /// Log and toast the outcome of saving `what` to `path`.
    fn saved(&mut self, written: anyhow::Result<()>, what: &str, path: &Path) {
        match written {
            Ok(()) => {
                self.note("ok", format!("{what} saved as {}", path.display()));
                self.toast(format!("{what} saved"), String::new());
            }
            Err(err) => {
                let message = format!("Couldn't save the {}: {err:#}", what.to_lowercase());
                self.note("error", message.clone());
                self.toast(message, String::new());
            }
        }
    }

    fn export_finished(&mut self, totals: PhotoExport, report: Option<String>, folder_url: String) {
        self.exporting = false;
        self.has_report = !self.last_report.is_empty();
        self.export_changed();
        let mut message = format!("Exported {}", faces(totals.exported));
        if totals.skipped > 0 {
            message += &format!(", {} left out by the quality rules", totals.skipped);
        }
        if totals.failed > 0 {
            message += &format!(", {} failed (see Activity)", totals.failed);
        }
        let kind = if totals.failed > 0 { "error" } else { "ok" };
        self.note(kind, message.clone());
        if let Some(report) = report {
            self.note("info", report);
        }
        self.toast(message, folder_url);
    }
}

/// The report as "csv" or "json" (with success/failure totals); None for any other format.
fn encode_report(rows: &[ReportRow], format: &str) -> Option<anyhow::Result<Vec<u8>>> {
    Some(match format {
        "csv" => (|| {
            let mut writer = csv::Writer::from_writer(Vec::new());
            for row in rows {
                writer.serialize(row)?;
            }
            Ok(writer.into_inner().map_err(|e| e.into_error())?)
        })(),
        "json" => {
            let count = |outcome| rows.iter().filter(|r| r.outcome == outcome).count();
            let json = json!({
                "succeeded": count("succeeded"),
                "failed": count("failed"),
                "images": rows,
            });
            serde_json::to_vec_pretty(&json).map_err(Into::into)
        }
        _ => return None,
    })
}

/// Write the export report into `out_dir`; returns a line for the activity log.
fn write_report(
    rows: &[ReportRow],
    format: &str,
    out_dir: &Path,
    claims: &OutputClaims,
) -> Option<String> {
    let bytes = encode_report(rows, format)?;
    let name = format!("face-crop-report.{format}");
    let path = claims.claim(out_dir.join(&name), Path::new(&name));
    let written = bytes.and_then(|bytes| fcs_utils::write_atomically(&path, &bytes));
    Some(match written {
        Ok(()) => format!("Report saved as {}", path.display()),
        Err(err) => format!("Couldn't save the report: {err:#}"),
    })
}

/// Merge `patch` into `base`: objects key by key, anything else replaced. Whole-number floats
/// become integers on the way, since QML sends every number as a double and serde will not
/// read `400.0` into the `u32` an output width is.
fn merge(base: &mut Value, patch: Value) {
    match (base, patch) {
        (Value::Object(base), Value::Object(patch)) => {
            for (key, value) in patch {
                merge(base.entry(key).or_insert(Value::Null), value);
            }
        }
        (base, Value::Number(n)) => {
            *base = match n.as_f64() {
                Some(f) if f.fract() == 0.0 && f.abs() < 2f64.powi(53) => json!(f as i64),
                _ => Value::Number(n),
            };
        }
        (base, patch) => *base = patch,
    }
}

/// Detected faces as boxes in fractions of a `width` x `height` frame.
fn frame_faces(detections: &[fcs_core::Detection], width: u32, height: u32) -> Vec<Value> {
    let (w, h) = (width.max(1) as f32, height.max(1) as f32);
    detections
        .iter()
        .map(|d| {
            let b = d.bbox;
            let (x, y) = ((b.x / w).clamp(0.0, 1.0), (b.y / h).clamp(0.0, 1.0));
            json!({ "x": x, "y": y, "w": (b.width / w).min(1.0 - x), "h": (b.height / h).min(1.0 - y) })
        })
        .collect()
}

/// The backend fields of the GPU settings, for spotting a change (`GpuSettings` has no
/// `PartialEq`).
fn gpu_key(gpu: &GpuSettings) -> (bool, bool, bool) {
    (gpu.enabled, gpu.experimental_cuda, gpu.respect_env)
}

fn backend_name(gpu: &GpuSettings) -> &'static str {
    match (gpu.enabled, gpu.experimental_cuda) {
        (false, _) => "the CPU",
        (true, true) => "CUDA Rust",
        (true, false) => "the GPU (WGSL)",
    }
}

fn is_mapping_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| MAPPING_EXTENSIONS.iter().any(|m| e.eq_ignore_ascii_case(m)))
}

/// Which columns probably hold the photo and its output name, by their headers; else the
/// first two.
fn guess_mapping_columns(columns: &[String]) -> (usize, usize) {
    let find = |words: &[&str], skip: Option<usize>| {
        columns.iter().enumerate().position(|(i, name)| {
            let name = name.to_lowercase();
            Some(i) != skip && words.iter().any(|w| name.contains(w))
        })
    };
    let source = find(
        &["file", "path", "image", "photo", "picture", "source"],
        None,
    )
    .unwrap_or(0);
    let output = find(&["output", "name", "target", "rename", "id"], Some(source))
        .unwrap_or(if source == 0 { 1 } else { 0 })
        .min(columns.len().saturating_sub(1));
    (source, output)
}

fn thumb_key(uid: u64, rev: u32) -> String {
    format!("thumb/{uid}.{rev}")
}

fn canvas_key(uid: u64) -> String {
    format!("canvas/{uid}")
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn faces(n: usize) -> String {
    if n == 1 {
        "1 face".into()
    } else {
        format!("{n} faces")
    }
}

fn photos(n: usize) -> String {
    if n == 1 {
        "1 photo".into()
    } else {
        format!("{n} photos")
    }
}

/// A box given in fractions of a `size` photo, in pixels, kept inside the photo and at
/// least [`MIN_BOX`] on a side.
fn clamped_box(size: (u32, u32), x: f64, y: f64, w: f64, h: f64) -> BoundingBox {
    let (iw, ih) = (size.0 as f32, size.1 as f32);
    let side = |v: f64, full: f32| (v as f32 * full).clamp(MIN_BOX.min(full), full);
    let (width, height) = (side(w, iw), side(h, ih));
    BoundingBox {
        x: (x as f32 * iw).clamp(0.0, iw - width),
        y: (y as f32 * ih).clamp(0.0, ih - height),
        width,
        height,
    }
}

/// `shape`'s outline drawn into a `width` x `height` box, as [x, y] fractions of it.
fn outline(shape: &CropShape, width: f32, height: f32) -> Vec<Value> {
    // Traced at a real pixel size (longest side 200), so curves get enough points to look
    // smooth at any size QML draws them.
    let scale = 200.0 / width.max(height).max(1.0);
    let (w, h) = ((width * scale).max(1.0), (height * scale).max(1.0));
    outline_points_for_rect(w, h, shape)
        .into_iter()
        .map(|(x, y)| json!([x / w, y / h]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_patches_one_field_and_turns_whole_doubles_into_integers() {
        let mut settings = serde_json::to_value(AppSettings::default()).unwrap();
        merge(
            &mut settings,
            json!({ "crop": { "output_width": 640.0, "vertical_offset": 0.25 } }),
        );
        let settings: AppSettings = serde_json::from_value(settings).expect("still valid");
        assert_eq!(settings.crop.output_width, 640);
        assert_eq!(settings.crop.vertical_offset, 0.25);
        assert_eq!(
            settings.crop.output_height,
            AppSettings::default().crop.output_height
        );
    }

    #[test]
    fn edited_boxes_stay_inside_the_photo_and_above_the_minimum() {
        let size = (1000, 500);
        let inside = clamped_box(size, 0.1, 0.2, 0.3, 0.4);
        assert_eq!(
            (inside.x, inside.y, inside.width, inside.height),
            (100.0, 100.0, 300.0, 200.0)
        );
        // Dragged past the right and bottom edges: moved back in, size kept.
        let pushed = clamped_box(size, 0.9, 0.9, 0.3, 0.4);
        assert_eq!((pushed.x, pushed.y), (700.0, 300.0));
        // Shrunk to nothing, or dragged off the top-left: the minimum, at the edge.
        let tiny = clamped_box(size, -0.5, -0.5, 0.0, -1.0);
        assert_eq!(
            (tiny.x, tiny.y, tiny.width, tiny.height),
            (0.0, 0.0, MIN_BOX, MIN_BOX)
        );
    }

    #[test]
    fn every_catalogue_shape_has_an_outline_inside_its_box() {
        for (label, shape) in fcs_utils::shape_catalogue() {
            let points = outline(&shape, 400.0, 300.0);
            assert!(points.len() >= 3, "{label}");
            for p in points {
                let (x, y) = (p[0].as_f64().unwrap(), p[1].as_f64().unwrap());
                assert!(
                    (-0.01..=1.01).contains(&x) && (-0.01..=1.01).contains(&y),
                    "{label}: ({x}, {y})"
                );
            }
        }
    }

    #[test]
    fn mapping_columns_are_guessed_from_their_headers() {
        let cols = |names: &[&str]| names.iter().map(|n| n.to_string()).collect::<Vec<_>>();
        assert_eq!(
            guess_mapping_columns(&cols(&["Employee ID", "Photo file"])),
            (1, 0)
        );
        assert_eq!(
            guess_mapping_columns(&cols(&["filename", "output name"])),
            (0, 1)
        );
        // Nothing recognisable: the first two columns, in order.
        assert_eq!(guess_mapping_columns(&cols(&["a", "b", "c"])), (0, 1));
        // One column only: both point at it rather than past the end.
        assert_eq!(guess_mapping_columns(&cols(&["only"])), (0, 0));
    }

    #[test]
    fn reports_carry_fcs_guis_columns_in_csv_and_json() {
        let out = std::env::temp_dir().join(format!("fcs-qt-report-{}", std::process::id()));
        fs::create_dir_all(&out).unwrap();
        let rows = [
            ReportRow {
                image: "a.jpg".into(),
                outcome: "succeeded",
                faces_detected: 2,
                faces_exported: 1,
                error: None,
            },
            ReportRow {
                image: "b.jpg".into(),
                outcome: "failed",
                faces_detected: 1,
                faces_exported: 0,
                error: Some("unreadable".into()),
            },
        ];
        let claims = OutputClaims::new(OverwritePolicy::Overwrite);
        assert!(write_report(&rows, "none", &out, &claims).is_none());
        write_report(&rows, "csv", &out, &claims).unwrap();
        write_report(&rows, "json", &out, &claims).unwrap();
        let csv = fs::read_to_string(out.join("face-crop-report.csv")).unwrap();
        assert_eq!(
            csv.lines().collect::<Vec<_>>(),
            [
                "image,outcome,faces_detected,faces_exported,error",
                "a.jpg,succeeded,2,1,",
                "b.jpg,failed,1,0,unreadable"
            ]
        );
        let json: Value =
            serde_json::from_str(&fs::read_to_string(out.join("face-crop-report.json")).unwrap())
                .unwrap();
        assert_eq!(
            (json["succeeded"].as_u64(), json["failed"].as_u64()),
            (Some(1), Some(1))
        );
        assert_eq!(json["images"][1]["error"], "unreadable");
        let _ = fs::remove_dir_all(&out);
    }
}
