//! The one object QML talks to. It owns the queue, the selection and the settings; the work
//! itself runs on background threads and comes back through `QThreadHandle::invoke_mut`.

use crate::{
    images::{self, Pixels},
    pipeline::{self, Detected, Engine, Face, Turn},
};
use fcs_core::BoundingBox;
use fcs_utils::{
    CropShape, OutputClaims, OverwritePolicy,
    config::{AppSettings, default_settings_path},
    is_supported_image_path, outline_points_for_rect,
};
use image::{DynamicImage, RgbImage, RgbaImage};
use log::warn;
use qtbridge::{QmlObject, qobject};
use serde_json::{Value, json};
use std::{
    collections::VecDeque,
    fs,
    path::PathBuf,
    sync::{Arc, mpsc},
    thread,
};

enum Status {
    Queued,
    Done,
    Failed(String),
}

struct Entry {
    path: PathBuf,
    turn: Turn,
    /// Bumped each time the photo is detected again, so its thumbnail gets a new URL and QML's
    /// image cache cannot show the old one.
    rev: u32,
    status: Status,
    size: (u32, u32),
    faces: Vec<Face>,
    selected: Vec<bool>,
}

/// Entry id, source path, whether to apply EXIF orientation, and how to turn the photo.
type DetectJob = (usize, PathBuf, bool, Turn);
/// The current photo's faces and their export selection, as undo restores them.
type Snapshot = (Vec<Face>, Vec<bool>);
/// Undo steps kept per photo, as in fcs-gui.
const UNDO_LIMIT: usize = 50;
/// The smallest box an edit leaves, in photo pixels.
const MIN_BOX: f32 = 8.0;

/// Canvas copies kept in memory: up to ~12 MB each at 2048 px, so a few photos back and forth
/// switch instantly without holding a whole batch.
const CANVAS_CACHE: usize = 8;
/// Entry id, source path, its turn, and its selected faces with their indices.
type ExportJob = (usize, PathBuf, Turn, Vec<(usize, Face)>);

pub struct Studio {
    settings: AppSettings,
    settings_path: PathBuf,
    engine: Option<Arc<Engine>>,
    jobs: Option<mpsc::Sender<DetectJob>>,
    entries: Vec<Entry>,
    /// The decoded current image, kept for previews. Only one: a batch of camera files
    /// would not fit in memory.
    image: Option<(usize, Arc<DynamicImage>)>,
    preview_running: bool,
    preview_dirty: bool,
    preview_seq: u64,
    /// Image-store keys of the previews on screen.
    preview_keys: Vec<String>,
    /// Entries whose canvas copy is in the image store, oldest first.
    canvases: VecDeque<usize>,
    /// Edits of the current photo; both cleared when another photo is selected.
    undo_stack: Vec<Snapshot>,
    redo_stack: Vec<Snapshot>,

    ready: bool,
    engine_label: String,
    engine_error: String,
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
    exporting: bool,
    export_progress: f64,
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
            jobs: None,
            entries: Vec::new(),
            image: None,
            preview_running: false,
            preview_dirty: false,
            preview_seq: 0,
            preview_keys: Vec::new(),
            canvases: VecDeque::new(),
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            ready: false,
            engine_label: String::new(),
            engine_error: String::new(),
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
            exporting: false,
            export_progress: 0.0,
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
        "shapeOutline",
        Member = shape_outline,
        Notify = settings_changed
    );
    qproperty!("exporting", Member = exporting, Notify = export_changed);
    qproperty!(
        "exportProgress",
        Member = export_progress,
        Notify = export_changed
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
        self.undo_stack.clear();
        self.redo_stack.clear();
        self.previews.clear();
        self.previews_changed();
        self.publish_image();
        self.load_current_image();
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
            Ok(settings) => {
                self.settings = settings;
                self.settings_edited();
            }
            Err(err) => {
                warn!("Rejected a settings change: {err}");
                self.toast(format!("Couldn't apply that setting: {err}"), String::new());
            }
        }
    }

    #[qslot(qml_name = "restoreEnhance")]
    fn restore_enhance(&mut self) {
        self.settings.enhance = Default::default();
        self.settings_edited();
    }

    /// Save every selected face of every finished photo into `folder_url`.
    #[qslot(qml_name = "exportTo")]
    fn export_to(&mut self, folder_url: String) {
        let Some(engine) = self.engine.clone().filter(|_| !self.exporting) else {
            return;
        };
        let jobs: Vec<ExportJob> = self
            .entries
            .iter()
            .enumerate()
            .map(|(id, e)| {
                let faces = e
                    .faces
                    .iter()
                    .zip(&e.selected)
                    .enumerate()
                    .filter(|(_, (_, on))| **on)
                    .map(|(i, (face, _))| (i, face.clone()))
                    .collect::<Vec<_>>();
                (id, e.path.clone(), e.turn, faces)
            })
            .filter(|(_, _, _, faces)| !faces.is_empty())
            .collect();
        let total: usize = jobs.iter().map(|(_, _, _, f)| f.len()).sum();
        if total == 0 {
            self.toast("Select at least one face to export".into(), String::new());
            return;
        }
        self.exporting = true;
        self.export_progress = 0.0;
        self.export_changed();

        let out_dir = pipeline::url_to_path(&folder_url);
        let settings = self.settings.clone();
        let current = self.image.clone();
        let handle = self.get_thread_handle();
        thread::spawn(move || {
            // Rename rather than replace: an earlier export in the same folder is someone's work.
            let claims = OutputClaims::new(OverwritePolicy::KeepBoth);
            let (mut saved, mut failed, mut done) = (0usize, 0usize, 0usize);
            for (id, path, turn, faces) in jobs {
                let image = match &current {
                    Some((current_id, image)) if *current_id == id => Ok(image.clone()),
                    _ => pipeline::load(&path, settings.crop.auto_orient_exif, turn).map(Arc::new),
                };
                let image = match image {
                    Ok(image) => image,
                    Err(err) => {
                        warn!("Failed to load {}: {err:#}", path.display());
                        failed += faces.len();
                        done += faces.len();
                        continue;
                    }
                };
                for (index, face) in faces {
                    match pipeline::export_face(
                        &engine, &image, &path, &face, index, &settings, &out_dir, &claims,
                    ) {
                        Ok(_) => saved += 1,
                        Err(err) => {
                            warn!(
                                "Failed to export {} face {}: {err:#}",
                                path.display(),
                                index + 1
                            );
                            failed += 1;
                        }
                    }
                    done += 1;
                    let progress = done as f64 / total as f64;
                    handle.invoke_mut(move |studio| {
                        studio.export_progress = progress;
                        studio.export_changed();
                    });
                }
            }
            handle.invoke_mut(move |studio| studio.export_finished(saved, failed, folder_url));
        });
    }
}

impl Studio {
    fn add_paths(&mut self, dropped: Vec<PathBuf>) {
        let mut paths = Vec::new();
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
            }
        }
        paths.retain(|p| !self.entries.iter().any(|e| &e.path == p));
        if paths.is_empty() {
            self.toast("No new supported photos there".into(), String::new());
            return;
        }
        for path in paths {
            let id = self.entries.len();
            if let Some(jobs) = &self.jobs {
                let _ = jobs.send((
                    id,
                    path.clone(),
                    self.settings.crop.auto_orient_exif,
                    Turn::default(),
                ));
            }
            self.entries.push(Entry {
                path,
                turn: Turn::default(),
                rev: 0,
                status: Status::Queued,
                size: (0, 0),
                faces: Vec::new(),
                selected: Vec::new(),
            });
            self.pending += 1;
        }
        if self.current < 0 {
            self.current = 0;
            self.publish_image();
        }
        self.publish_queue();
    }

    fn engine_loaded(&mut self, engine: Result<Engine, String>) {
        match engine {
            Ok(engine) => {
                let engine = Arc::new(engine);
                self.engine_label = engine.label.clone();
                self.ready = true;
                let (tx, rx) = mpsc::channel::<DetectJob>();
                let worker_engine = engine.clone();
                let handle = self.get_thread_handle();
                thread::spawn(move || {
                    for (id, path, orient, turn) in rx {
                        let result = pipeline::detect(&worker_engine, &path, orient, turn)
                            .map_err(|err| format!("{err:#}"));
                        handle.invoke_mut(move |studio| studio.detected(id, result));
                    }
                });
                // Photos added while the models loaded.
                for (id, entry) in self.entries.iter().enumerate() {
                    let orient = self.settings.crop.auto_orient_exif;
                    let _ = tx.send((id, entry.path.clone(), orient, entry.turn));
                }
                self.jobs = Some(tx);
                self.engine = Some(engine);
            }
            Err(err) => self.engine_error = err,
        }
        self.engine_changed();
    }

    fn detected(&mut self, id: usize, result: Result<Detected, String>) {
        self.pending -= 1;
        let Some(entry) = self.entries.get_mut(id) else {
            return;
        };
        match result {
            Ok(detected) => {
                entry.size = (detected.image.width(), detected.image.height());
                entry.selected = vec![true; detected.faces.len()];
                entry.faces = detected.faces;
                entry.status = Status::Done;
                images::put(&thumb_key(id, entry.rev), Pixels::Rgb(detected.thumb));
                self.cache_canvas(id, detected.canvas);
                if self.current == id as i32 {
                    // New boxes: the old edits no longer apply to anything.
                    self.undo_stack.clear();
                    self.redo_stack.clear();
                    self.image = Some((id, detected.image));
                    self.publish_image();
                    self.request_previews();
                }
            }
            Err(err) => {
                warn!("{err}");
                entry.status = Status::Failed(err);
                if self.current == id as i32 {
                    self.publish_image();
                }
            }
        }
        self.publish_queue();
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
        let orient = self.settings.crop.auto_orient_exif;
        let Ok(id) = usize::try_from(self.current) else {
            return;
        };
        let Some(entry) = self
            .entries
            .get_mut(id)
            .filter(|e| matches!(e.status, Status::Done))
        else {
            return;
        };
        let before = entry.turn;
        change(&mut entry.turn);
        if entry.turn == before {
            return;
        }
        entry.status = Status::Queued;
        entry.faces.clear();
        entry.selected.clear();
        images::remove(&thumb_key(id, entry.rev));
        entry.rev += 1;
        if let Some(jobs) = &self.jobs {
            let _ = jobs.send((id, entry.path.clone(), orient, entry.turn));
        }
        self.pending += 1;
        self.canvases.retain(|&held| held != id);
        images::remove(&format!("canvas/{id}"));
        self.image = None;
        self.undo_stack.clear();
        self.redo_stack.clear();
        self.previews.clear();
        self.previews_changed();
        self.publish_image();
        self.publish_queue();
    }

    fn current_entry_mut(&mut self) -> Option<&mut Entry> {
        usize::try_from(self.current)
            .ok()
            .and_then(|i| self.entries.get_mut(i))
    }

    fn selection_edited(&mut self) {
        self.publish_image();
        self.publish_queue();
        self.request_previews();
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
        self.settings_json = serde_json::to_value(&self.settings).unwrap_or_default();
        let crop = &self.settings.crop;
        self.shape_outline = outline(
            &crop.shape,
            crop.output_width as f32,
            crop.output_height as f32,
        );
    }

    fn publish_queue(&mut self) {
        self.queue = self
            .entries
            .iter()
            .enumerate()
            .map(|(id, e)| {
                let (status, error) = match &e.status {
                    Status::Queued => ("queued", ""),
                    Status::Done => ("done", ""),
                    Status::Failed(err) => ("failed", err.as_str()),
                };
                json!({
                    "name": e.path.file_name().map(|n| n.to_string_lossy()).unwrap_or_default(),
                    "thumb": if matches!(e.status, Status::Done) { images::url(&thumb_key(id, e.rev)) } else { String::new() },
                    "status": status,
                    "error": error,
                    "faces": e.faces.len(),
                    "selected": e.selected.iter().filter(|s| **s).count(),
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
        // Shown once its canvas copy is in the store; until then the canvas says it is opening.
        let entry = usize::try_from(self.current)
            .ok()
            .filter(|id| self.canvases.contains(id))
            .and_then(|id| Some(id).zip(self.entries.get(id)))
            .filter(|(_, e)| matches!(e.status, Status::Done));
        self.can_undo = !self.undo_stack.is_empty();
        self.can_redo = !self.redo_stack.is_empty();
        let current = usize::try_from(self.current)
            .ok()
            .and_then(|i| self.entries.get(i));
        let turn = current.map(|e| e.turn).unwrap_or_default();
        self.quarter_turns = i32::from(turn.quarters);
        self.tilt = f64::from(turn.tilt_deg);
        let Some((id, entry)) = entry else {
            self.image_url.clear();
            self.faces.clear();
            self.image_changed();
            return;
        };
        let (w, h) = (entry.size.0.max(1) as f32, entry.size.1.max(1) as f32);
        let core: fcs_core::CropSettings = (&self.settings.crop).into();
        self.image_url = images::url(&format!("canvas/{id}"));
        self.image_width = entry.size.0 as i32;
        self.image_height = entry.size.1 as i32;
        self.faces = entry
            .faces
            .iter()
            .zip(&entry.selected)
            .enumerate()
            .map(|(i, (face, on))| {
                let b = face.detection.bbox;
                let x = (b.x / w).clamp(0.0, 1.0);
                let y = (b.y / h).clamp(0.0, 1.0);
                let crop = fcs_core::calculate_crop_region(entry.size.0, entry.size.1, b, &core);
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
        self.image_changed();
    }

    /// Decode the current photo for previews, and remake its canvas copy if that was evicted.
    fn load_current_image(&mut self) {
        let Ok(id) = usize::try_from(self.current) else {
            return;
        };
        if self.image.as_ref().is_some_and(|(held, _)| *held == id) {
            self.request_previews();
            return;
        }
        self.image = None;
        let Some(entry) = self
            .entries
            .get(id)
            .filter(|e| matches!(e.status, Status::Done))
        else {
            return;
        };
        let path = entry.path.clone();
        let turn = entry.turn;
        let orient = self.settings.crop.auto_orient_exif;
        let handle = self.get_thread_handle();
        thread::spawn(move || {
            let loaded =
                pipeline::load(&path, orient, turn).map(|image| (pipeline::canvas(&image), image));
            handle.invoke_mut(move |studio| studio.current_image_loaded(id, loaded));
        });
    }

    fn current_image_loaded(
        &mut self,
        id: usize,
        loaded: anyhow::Result<(RgbImage, DynamicImage)>,
    ) {
        match loaded {
            Ok((canvas, image)) => {
                if self.current != id as i32 {
                    return;
                }
                self.image = Some((id, Arc::new(image)));
                if !self.canvases.contains(&id) {
                    self.cache_canvas(id, canvas);
                    self.publish_image();
                }
                self.request_previews();
            }
            // Read once already, so the file changed or went away since.
            Err(err) => {
                let err = format!("{err:#}");
                warn!("{err}");
                if let Some(entry) = self.entries.get_mut(id) {
                    entry.status = Status::Failed(err);
                }
                self.publish_image();
                self.publish_queue();
            }
        }
    }

    /// Hold `canvas` as entry `id`'s copy, evicting the oldest other one past
    /// [`CANVAS_CACHE`]. Never the current photo's: it is on screen.
    fn cache_canvas(&mut self, id: usize, canvas: RgbImage) {
        images::put(&format!("canvas/{id}"), Pixels::Rgb(canvas));
        self.canvases.retain(|&held| held != id);
        self.canvases.push_back(id);
        while self.canvases.len() > CANVAS_CACHE {
            let current = self.current;
            let Some(oldest) = self
                .canvases
                .iter()
                .position(|&held| held as i32 != current)
            else {
                break;
            };
            if let Some(evicted) = self.canvases.remove(oldest) {
                images::remove(&format!("canvas/{evicted}"));
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
        let (Some(engine), Some((id, image))) = (self.engine.clone(), self.image.clone()) else {
            return;
        };
        let Some(entry) = self.entries.get(id) else {
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
            handle.invoke_mut(move |studio| studio.previews_ready(id, seq, previews));
        });
    }

    fn previews_ready(&mut self, id: usize, seq: u64, previews: Vec<(usize, RgbaImage)>) {
        self.preview_running = false;
        // Fresh keys each render, so QML never shows a cached older one; the last set is
        // already on screen and can leave the store.
        for key in self.preview_keys.drain(..) {
            images::remove(&key);
        }
        if self.current == id as i32 {
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

    fn export_finished(&mut self, saved: usize, failed: usize, folder_url: String) {
        self.exporting = false;
        self.export_changed();
        let faces = |n| {
            if n == 1 {
                "1 face".to_string()
            } else {
                format!("{n} faces")
            }
        };
        let message = match failed {
            0 => format!("Exported {}", faces(saved)),
            _ => format!("Exported {}, {} failed (see log)", faces(saved), failed),
        };
        self.toast(message, folder_url);
    }
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

fn thumb_key(id: usize, rev: u32) -> String {
    format!("thumb/{id}.{rev}")
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
}
