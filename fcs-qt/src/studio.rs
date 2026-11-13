//! The one object QML talks to. It owns the queue, the selection and the settings; the work
//! itself runs on background threads and comes back through `QThreadHandle::invoke_mut`.

use crate::pipeline::{self, Detected, Engine, Face};
use fcs_utils::{
    CropShape, ImageFormatHint, OutputClaims, OverwritePolicy,
    config::{AppSettings, default_settings_path},
    is_supported_image_path, load_image,
};
use image::DynamicImage;
use log::warn;
use qtbridge::{QmlObject, qobject};
use serde_json::{Value, json};
use std::{
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
    display: PathBuf,
    status: Status,
    size: (u32, u32),
    faces: Vec<Face>,
    selected: Vec<bool>,
}

type DetectJob = (usize, PathBuf, PathBuf);
/// Entry id, source path, and its selected faces with their indices.
type ExportJob = (usize, PathBuf, Vec<(usize, Face)>);

pub struct Studio {
    settings: AppSettings,
    settings_path: PathBuf,
    /// Canvas copies and preview PNGs; removed on drop.
    scratch: PathBuf,
    engine: Option<Arc<Engine>>,
    jobs: Option<mpsc::Sender<DetectJob>>,
    entries: Vec<Entry>,
    /// The decoded current image, kept for previews. Only one: a batch of camera files
    /// would not fit in memory.
    image: Option<(usize, Arc<DynamicImage>)>,
    preview_running: bool,
    preview_dirty: bool,
    preview_seq: u64,
    preview_files: Vec<PathBuf>,

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
    previews: Vec<Value>,
    presets: Vec<Value>,
    image_filter: String,
    preset: String,
    output_width: i32,
    output_height: i32,
    face_height: f64,
    shape: String,
    format: String,
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
        let scratch = std::env::temp_dir().join(format!("fcs-qt-{}", std::process::id()));
        if let Err(err) = fs::create_dir_all(&scratch) {
            warn!("Failed to create {}: {err}", scratch.display());
        }
        let presets = fcs_core::standard_presets()
            .iter()
            .filter(|p| p.width > 0)
            .map(|p| json!({ "name": p.name, "width": p.width, "height": p.height }))
            .collect();
        let mut studio = Self {
            settings,
            settings_path,
            scratch,
            engine: None,
            jobs: None,
            entries: Vec::new(),
            image: None,
            preview_running: false,
            preview_dirty: false,
            preview_seq: 0,
            preview_files: Vec::new(),
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
            preset: String::new(),
            output_width: 0,
            output_height: 0,
            face_height: 0.0,
            shape: String::new(),
            format: String::new(),
            exporting: false,
            export_progress: 0.0,
        };
        studio.sync_settings();
        studio
    }
}

impl Drop for Studio {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.scratch);
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
    qproperty!("previews", Member = previews, Notify = previews_changed);
    qproperty!("presets", Member = presets, Constant);
    qproperty!("imageFilter", Member = image_filter, Constant);
    qproperty!("preset", Member = preset, Notify = settings_changed);
    qproperty!(
        "outputWidth",
        Member = output_width,
        Notify = settings_changed
    );
    qproperty!(
        "outputHeight",
        Member = output_height,
        Notify = settings_changed
    );
    qproperty!(
        "faceHeight",
        Member = face_height,
        Notify = settings_changed
    );
    qproperty!("shape", Member = shape, Notify = settings_changed);
    qproperty!("format", Member = format, Notify = settings_changed);
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

    #[qslot(qml_name = "setFaceHeight")]
    fn set_face_height(&mut self, pct: f64) {
        self.settings.crop.face_height_pct = pct as f32;
        self.settings_edited();
    }

    #[qslot(qml_name = "setShape")]
    fn set_shape(&mut self, name: String) {
        self.settings.crop.shape = match name.as_str() {
            "rounded" => CropShape::RoundedRectangle { radius_pct: 0.12 },
            "ellipse" => CropShape::Ellipse,
            _ => CropShape::Rectangle,
        };
        self.settings_edited();
    }

    #[qslot(qml_name = "setFormat")]
    fn set_format(&mut self, ext: String) {
        if let Some(format) = ImageFormatHint::from_extension(&ext) {
            self.settings.crop.output_format = format;
            self.settings_edited();
        }
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
                (id, e.path.clone(), faces)
            })
            .filter(|(_, _, faces)| !faces.is_empty())
            .collect();
        let total: usize = jobs.iter().map(|(_, _, f)| f.len()).sum();
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
            for (id, path, faces) in jobs {
                let image = match &current {
                    Some((current_id, image)) if *current_id == id => Ok(image.clone()),
                    _ => load_image(&path).map(Arc::new),
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
            let display = self.scratch.join(format!("display-{id}.jpg"));
            if let Some(jobs) = &self.jobs {
                let _ = jobs.send((id, path.clone(), display.clone()));
            }
            self.entries.push(Entry {
                path,
                display,
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
                    for (id, path, display) in rx {
                        let result = pipeline::detect(&worker_engine, &path, &display)
                            .map_err(|err| format!("{err:#}"));
                        handle.invoke_mut(move |studio| studio.detected(id, result));
                    }
                });
                // Photos added while the models loaded.
                for (id, entry) in self.entries.iter().enumerate() {
                    let _ = tx.send((id, entry.path.clone(), entry.display.clone()));
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
                if self.current == id as i32 {
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
        self.request_previews();
    }

    fn sync_settings(&mut self) {
        let crop = &self.settings.crop;
        self.preset = crop.preset.clone();
        self.output_width = crop.output_width as i32;
        self.output_height = crop.output_height as i32;
        self.face_height = crop.face_height_pct as f64;
        self.shape = match crop.shape {
            CropShape::Rectangle => "rectangle",
            CropShape::RoundedRectangle { .. } => "rounded",
            CropShape::Ellipse => "ellipse",
            _ => "custom",
        }
        .into();
        self.format = crop.output_format.extension().into();
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
                    "name": e.path.file_name().map(|n| n.to_string_lossy()).unwrap_or_default(),
                    "thumb": if matches!(e.status, Status::Done) { pipeline::path_to_url(&e.display) } else { String::new() },
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
        let entry = usize::try_from(self.current)
            .ok()
            .and_then(|i| self.entries.get(i))
            .filter(|e| matches!(e.status, Status::Done));
        let Some(entry) = entry else {
            self.image_url.clear();
            self.faces.clear();
            self.image_changed();
            return;
        };
        let (w, h) = (entry.size.0.max(1) as f32, entry.size.1.max(1) as f32);
        self.image_url = pipeline::path_to_url(&entry.display);
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
                json!({
                    "index": i,
                    "x": x,
                    "y": y,
                    "w": (b.width / w).min(1.0 - x),
                    "h": (b.height / h).min(1.0 - y),
                    "score": face.detection.score,
                    "quality": format!("{:?}", face.quality),
                    "selected": on,
                })
            })
            .collect();
        self.image_changed();
    }

    /// Decode the current photo for previews, unless it is the one already held.
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
        let handle = self.get_thread_handle();
        thread::spawn(move || match load_image(&path) {
            Ok(image) => handle.invoke_mut(move |studio| {
                if studio.current == id as i32 {
                    studio.image = Some((id, Arc::new(image)));
                    studio.request_previews();
                }
            }),
            Err(err) => warn!("Failed to load {}: {err:#}", path.display()),
        });
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
        let scratch = self.scratch.clone();
        let settings = self.settings.clone();
        let handle = self.get_thread_handle();
        thread::spawn(move || {
            let previews: Vec<(usize, PathBuf)> = faces
                .into_iter()
                .filter_map(|(i, face)| {
                    let crop = pipeline::finish(&engine, &image, &face.detection, &settings);
                    let path = scratch.join(format!("preview-{seq}-{i}.png"));
                    match pipeline::save_preview(&crop.image, &path) {
                        Ok(()) => Some((i, path)),
                        Err(err) => {
                            warn!("Failed to write preview {}: {err:#}", path.display());
                            None
                        }
                    }
                })
                .collect();
            handle.invoke_mut(move |studio| studio.previews_ready(id, previews));
        });
    }

    fn previews_ready(&mut self, id: usize, previews: Vec<(usize, PathBuf)>) {
        self.preview_running = false;
        // Each render has fresh names, so QML never shows a cached older one; the last set
        // is already decoded on screen and can go.
        remove_files(&mut self.preview_files);
        if self.current == id as i32 {
            self.previews = previews
                .iter()
                .map(|(i, path)| json!({ "index": i, "url": pipeline::path_to_url(path) }))
                .collect();
            self.preview_files = previews.into_iter().map(|(_, p)| p).collect();
            self.previews_changed();
        } else {
            remove_files(&mut previews.into_iter().map(|(_, p)| p).collect());
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

fn remove_files(files: &mut Vec<PathBuf>) {
    for file in files.drain(..) {
        let _ = fs::remove_file(file);
    }
}
