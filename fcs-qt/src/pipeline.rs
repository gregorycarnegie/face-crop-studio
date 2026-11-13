//! Everything below the UI: loading the models, detecting, finishing and saving crops.
//! Mirrors what fcs-gui's core does, minus its egui types.

use anyhow::{Context, Result};
use fcs_core::{BoundingBox, Detection, EyeRefiner, FaceDetector};
use fcs_utils::{
    EnhancementRuntime, FinishedCrop, GpuAvailability, GpuContext, GpuContextOptions,
    MetadataContext, OutputClaims, OutputOptions, Quality, QualityFilter,
    append_suffix_to_filename,
    config::{AppSettings, DetectionSettings},
    estimate_sharpness, load_image, load_image_raw, resolve_data_path, save_dynamic_image,
};
use image::{DynamicImage, RgbImage, RgbaImage};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

/// Longest side of the copy the canvas shows. The face boxes are positioned in fractions of
/// the image, so the canvas never needs the full resolution.
const CANVAS_SIDE: u32 = 2048;
/// Longest side of a queue thumbnail (48 px at up to 4x).
const THUMB_SIDE: u32 = 192;
/// Longest side of a result preview card.
const PREVIEW_SIDE: u32 = 480;

pub struct Engine {
    pub detector: FaceDetector,
    pub refiner: Option<EyeRefiner>,
    pub enhancement: EnhancementRuntime,
    /// What runs the detector, for the status pill: "GPU · <adapter>" or "CPU".
    pub label: String,
}

impl Engine {
    pub fn load(settings: &AppSettings) -> Result<Self> {
        let options: GpuContextOptions = (&settings.gpu).into();
        let context = match GpuContext::init_with_fallback(&options) {
            GpuAvailability::Available(context) if settings.gpu.enabled => Some(context),
            _ => None,
        };
        let adapter = context.as_ref().map(|c| c.adapter_info().name.clone());
        let enhancement = EnhancementRuntime::new(context);
        let model = settings
            .model_path
            .as_deref()
            .map(resolve_data_path)
            .context("no model path configured")?;
        let detector = FaceDetector::load_from_with_gpu(&model, &options)
            .with_context(|| format!("no detector model at {}", model.display()))?
            .with_settings(&settings.detection);
        let refiner = EyeRefiner::load();
        let backend = detector.inference_backend();
        let label = match adapter {
            Some(name) => format!("{backend} · {name}"),
            None => backend.to_string(),
        };
        Ok(Self {
            detector,
            refiner,
            enhancement,
            label,
        })
    }
}

#[derive(Clone)]
pub struct Face {
    /// Its `bbox` is the box as edited; cropping and export read it.
    pub detection: Detection,
    pub quality: Quality,
    /// The box as the detector found it, for "reset".
    pub original: BoundingBox,
    /// Drawn by hand rather than detected.
    pub manual: bool,
}

impl Face {
    /// A face the user drew where the detector found none: no landmarks, full confidence,
    /// and High quality, as fcs-gui's `commit_manual_box` makes them.
    pub fn manual(bbox: BoundingBox) -> Self {
        Self {
            detection: Detection {
                bbox,
                landmarks: [None; 5],
                score: 1.0,
            },
            quality: Quality::High,
            original: bbox,
            manual: true,
        }
    }
}

/// How a photo is turned before anything sees it: whole clockwise quarter turns, which swap
/// width and height, then a small straightening tilt that keeps the size.
#[derive(Clone, Copy, Default, PartialEq)]
pub struct Turn {
    pub quarters: u8,
    pub tilt_deg: f32,
}

pub struct Detected {
    pub image: Arc<DynamicImage>,
    pub faces: Vec<Face>,
    pub canvas: RgbImage,
    pub thumb: RgbImage,
}

/// Decode `path`, turned upright by its EXIF orientation when `orient` (the "Auto-orient via
/// EXIF" setting). Detection, previews and export all load through here, so the boxes found
/// on one decode land on the same pixels in the others.
pub fn load(path: &Path, orient: bool, turn: Turn) -> Result<DynamicImage> {
    let image = if orient {
        load_image(path)
    } else {
        load_image_raw(path)
    }
    .with_context(|| format!("failed to load {}", path.display()))?;
    let image = match turn.quarters % 4 {
        1 => image.rotate90(),
        2 => image.rotate180(),
        3 => image.rotate270(),
        _ => image,
    };
    Ok(fcs_core::rotate_photo(&image, turn.tilt_deg).unwrap_or(image))
}

/// Load `path`, find and refine its faces with `settings` (the confidence floor and the rest),
/// and make the canvas and queue copies.
pub fn detect(
    engine: &Engine,
    path: &Path,
    orient: bool,
    turn: Turn,
    settings: &DetectionSettings,
) -> Result<Detected> {
    let image = load(path, orient, turn)?;
    // Cheap: the same loaded model under other thresholds.
    let mut output = engine
        .detector
        .with_settings(settings)
        .detect_image(&image)
        .with_context(|| format!("detection failed for {}", path.display()))?;
    if let Some(refiner) = &engine.refiner {
        refiner.refine(&image, &mut output.detections);
    }
    let faces = output
        .detections
        .into_iter()
        .map(|detection| Face {
            quality: face_quality(&image, &detection),
            original: detection.bbox,
            manual: false,
            detection,
        })
        .collect();
    Ok(Detected {
        canvas: canvas(&image),
        thumb: scaled(&image, THUMB_SIDE),
        image: Arc::new(image),
        faces,
    })
}

/// Sharpness of the detected box itself, clamped into the image as fcs-gui does.
fn face_quality(image: &DynamicImage, detection: &Detection) -> Quality {
    let b = detection.bbox;
    let x = (b.x.max(0.0) as u32).min(image.width().saturating_sub(1));
    let y = (b.y.max(0.0) as u32).min(image.height().saturating_sub(1));
    let w = (b.width.max(1.0) as u32).min(image.width() - x);
    let h = (b.height.max(1.0) as u32).min(image.height() - y);
    estimate_sharpness(&image.crop_imm(x, y, w, h)).1
}

/// The copy of a photo the canvas shows.
pub fn canvas(image: &DynamicImage) -> RgbImage {
    scaled(image, CANVAS_SIDE)
}

fn scaled(image: &DynamicImage, side: u32) -> RgbImage {
    let (w, h) = fit(image.width(), image.height(), side);
    fcs_utils::resize_image(image, w, h, image::imageops::FilterType::Triangle)
}

fn fit(w: u32, h: u32, side: u32) -> (u32, u32) {
    let scale = (side as f32 / w.max(h) as f32).min(1.0);
    (
        ((w as f32 * scale).round() as u32).max(1),
        ((h as f32 * scale).round() as u32).max(1),
    )
}

/// One face cropped, enhanced, shaped and filled: what export writes. Same steps as
/// fcs-gui's `finish_face`.
pub fn finish(
    engine: &Engine,
    image: &DynamicImage,
    detection: &Detection,
    settings: &AppSettings,
) -> fcs_utils::FinishedCrop {
    let core: fcs_core::CropSettings = (&settings.crop).into();
    let raw = fcs_core::crop_face_from_image(image, detection, &core);
    let eyes = fcs_core::eye_positions(detection, image.width(), image.height(), &core);
    engine.enhancement.finish_crop(
        raw,
        &settings.crop,
        Some(&settings.enhance.to_enhancement_settings()),
        (!eyes.is_empty()).then_some(&eyes[..]),
    )
}

/// A finished crop sized for the preview strip, with alpha kept for a shaped crop's corners.
pub fn preview(crop: &DynamicImage) -> RgbaImage {
    crop.thumbnail(PREVIEW_SIDE, PREVIEW_SIDE).into_rgba8()
}

/// What exporting one photo did with each of its selected faces.
#[derive(Debug, Default, PartialEq)]
pub struct PhotoExport {
    pub exported: usize,
    /// Left out by the quality rules.
    pub skipped: usize,
    pub failed: usize,
}

/// The quality rules from the settings, as the CLI applies them.
pub fn quality_filter(settings: &AppSettings) -> QualityFilter {
    let rules = &settings.crop.quality_rules;
    QualityFilter {
        min_quality: rules.min_quality,
        auto_select: rules.auto_select_best_face,
        auto_skip_no_high: rules.auto_skip_no_high_quality,
        suffix_enabled: rules.quality_suffix,
    }
}

/// Finish the selected `faces` of one photo and save those the quality rules keep, in the
/// CLI's order: skip the photo when no face is sharp enough, keep only the best face when
/// asked, drop faces under the minimum, then name what is left. The rules judge the finished
/// crop, as the CLI does, not the detected box.
#[allow(clippy::too_many_arguments)] // each is a separate input; a struct for one caller adds nothing
pub fn export_photo(
    engine: &Engine,
    image: &DynamicImage,
    source: &Path,
    faces: &[(usize, Face)],
    mapped: Option<&str>,
    settings: &AppSettings,
    out_dir: &Path,
    claims: &OutputClaims,
) -> PhotoExport {
    let filter = quality_filter(settings);
    let finished: Vec<FinishedCrop> = faces
        .iter()
        .map(|(_, face)| finish(engine, image, &face.detection, settings))
        .collect();
    let mut result = PhotoExport::default();
    if filter.should_skip_image(finished.iter().map(|c| c.quality).max()) {
        result.skipped = finished.len();
        return result;
    }
    let best = filter.auto_select.then(|| {
        let scored: Vec<_> = finished
            .iter()
            .map(|c| (c.quality, c.quality_score))
            .collect();
        filter.select_best_index(&scored)
    });
    for (position, ((index, face), crop)) in faces.iter().zip(finished).enumerate() {
        if best.is_some_and(|best| best != Some(position)) || filter.should_skip(crop.quality) {
            result.skipped += 1;
            continue;
        }
        let naming = Naming {
            mapped,
            multi_face: faces.len() > 1,
            suffix: filter.suffix_for(crop.quality).unwrap_or(""),
        };
        match save_face(
            &crop, source, face, *index, &naming, settings, out_dir, claims,
        ) {
            Ok(_) => result.exported += 1,
            Err(err) => {
                log::warn!(
                    "Failed to export {} face {}: {err:#}",
                    source.display(),
                    index + 1
                );
                result.failed += 1;
            }
        }
    }
    result
}

/// How an exported face is named.
pub struct Naming<'a> {
    /// The output name a mapping file gave the photo, if any.
    pub mapped: Option<&'a str>,
    /// Whether the photo exports more than one face, so a mapped name needs numbering.
    pub multi_face: bool,
    /// A quality suffix such as `_highq`, or empty.
    pub suffix: &'a str,
}

/// Where face `index` of `source` is saved: `{stem}_face_NN{suffix}.{ext}` by default, or the
/// mapped name (`{name}` or `{name}_faceN`, without a quality suffix), both as fcs-gui does.
pub fn face_path(
    source: &Path,
    index: usize,
    naming: &Naming,
    settings: &AppSettings,
    out_dir: &Path,
) -> PathBuf {
    let ext = settings.crop.output_format.extension();
    if let Some(mapped) = naming.mapped {
        return fcs_utils::mapped_output_path(
            out_dir,
            Path::new(mapped),
            ext,
            index,
            naming.multi_face,
        );
    }
    let stem = source
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("face");
    let name = format!("{stem}_face_{:02}.{ext}", index + 1);
    out_dir.join(append_suffix_to_filename(&name, naming.suffix))
}

/// Save one finished crop; `claims` decides whether an existing file is replaced or kept.
#[allow(clippy::too_many_arguments)] // each is a separate input; a struct for one caller adds nothing
fn save_face(
    crop: &FinishedCrop,
    source: &Path,
    face: &Face,
    index: usize,
    naming: &Naming,
    settings: &AppSettings,
    out_dir: &Path,
    claims: &OutputClaims,
) -> Result<PathBuf> {
    let path = claims.claim(face_path(source, index, naming, settings, out_dir), source);
    if let Some(parent) = path.parent() {
        // A mapped name may put the crop in a subfolder.
        std::fs::create_dir_all(parent)?;
    }
    save_dynamic_image(
        &crop.image,
        &path,
        &OutputOptions::from_crop_settings(&settings.crop),
        &MetadataContext {
            source_path: Some(source),
            crop_settings: Some(&settings.crop),
            detection_score: Some(face.detection.score),
            quality: Some(crop.quality),
            quality_score: Some(crop.quality_score),
        },
    )?;
    Ok(path)
}

/// A `file:` URL from QML (FileDialog, drag and drop) as a local path.
pub fn url_to_path(url: &str) -> PathBuf {
    let rest = url.strip_prefix("file://").unwrap_or(url);
    // `file:///C:/x` leaves `/C:/x` on Windows.
    let rest = match rest.as_bytes() {
        [b'/', drive, b':', ..] if drive.is_ascii_alphabetic() => &rest[1..],
        _ => rest,
    };
    let bytes = rest.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let hex = (bytes[i] == b'%')
            .then(|| bytes.get(i + 1..i + 3))
            .flatten()
            .and_then(|h| std::str::from_utf8(h).ok())
            .and_then(|h| u8::from_str_radix(h, 16).ok());
        match hex {
            Some(byte) => {
                out.push(byte);
                i += 3;
            }
            None => {
                out.push(bytes[i]);
                i += 1;
            }
        }
    }
    PathBuf::from(String::from_utf8_lossy(&out).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_urls_become_local_paths() {
        assert_eq!(
            url_to_path("file:///C:/Users/a%20b/%231%20100%25.jpg"),
            Path::new("C:/Users/a b/#1 100%.jpg")
        );
        assert_eq!(
            url_to_path("file:///home/a%20b/x.png"),
            Path::new("/home/a b/x.png")
        );
        assert_eq!(
            url_to_path("file:///C:/Photos/caf%C3%A9.jpg"),
            Path::new("C:/Photos/café.jpg")
        );
        // A stray `%` is kept rather than dropped.
        assert_eq!(url_to_path("file:///tmp/50%"), Path::new("/tmp/50%"));
    }

    /// The real models end to end: detect a sample and export it twice into one folder (the
    /// second must be renamed, not replace the first), then again keeping only the best face.
    #[test]
    fn detects_and_exports_a_sample_without_overwriting() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let mut settings = AppSettings {
            model_path: Some(
                root.join("models/scrfd80k_500m_640.onnx")
                    .display()
                    .to_string(),
            ),
            ..AppSettings::default()
        };
        let engine = Engine::load(&settings).expect("engine");
        let out = std::env::temp_dir().join(format!("fcs-qt-test-{}", std::process::id()));
        std::fs::create_dir_all(&out).expect("temp dir");
        let source = root.join("samples/sample_01.jpg");

        let detected =
            detect(&engine, &source, true, Turn::default(), &settings.detection).expect("detect");
        assert_eq!(detected.faces.len(), 1);
        // sample_01 is 1024 square: under both limits, the canvas keeps it whole.
        assert_eq!(detected.canvas.dimensions(), (1024, 1024));
        assert_eq!(detected.thumb.dimensions(), (THUMB_SIDE, THUMB_SIDE));

        // A fresh claims set per export, as each click of Export makes one.
        let export = |faces: &[(usize, Face)], settings: &AppSettings| {
            let claims = OutputClaims::new(fcs_utils::OverwritePolicy::KeepBoth);
            export_photo(
                &engine,
                &detected.image,
                &source,
                faces,
                None,
                settings,
                &out,
                &claims,
            )
        };
        let one = vec![(0, detected.faces[0].clone())];
        assert_eq!(export(&one, &settings).exported, 1);
        assert_eq!(export(&one, &settings).exported, 1);
        let first = out.join("sample_01_face_01.png");
        let saved = image::open(&first).expect("readable crop");
        assert_eq!(
            (saved.width(), saved.height()),
            (settings.crop.output_width, settings.crop.output_height)
        );
        assert!(
            out.join("sample_01_face_01(2).png").is_file(),
            "the second export was renamed"
        );

        // A hand-drawn box over the background, beside the real face: best-face-only keeps one.
        settings.crop.quality_rules.auto_select_best_face = true;
        let mut two = one.clone();
        two.push((
            1,
            Face::manual(BoundingBox {
                x: 10.0,
                y: 10.0,
                width: 60.0,
                height: 60.0,
            }),
        ));
        let result = export(&two, &settings);
        assert_eq!((result.exported, result.skipped, result.failed), (1, 1, 0));
        let _ = std::fs::remove_dir_all(&out);
    }

    /// HEIC (libheif) and AVIF (dav1d) decode in this build, whose C libraries have to come
    /// from vcpkg's dynamic-CRT triplet. Both files are sample_01 at 512 px.
    #[test]
    fn decodes_heic_and_avif() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata");
        for name in ["sample_01.heic", "sample_01.avif"] {
            let image = load_image(dir.join(name)).unwrap_or_else(|e| panic!("{name}: {e:#}"));
            assert_eq!((image.width(), image.height()), (512, 512), "{name}");
        }
    }

    #[test]
    fn fit_only_shrinks() {
        assert_eq!(fit(5120, 2560, 2560), (2560, 1280));
        assert_eq!(fit(800, 600, 2560), (800, 600));
    }

    #[test]
    fn quarter_turns_swap_the_sides_and_a_tilt_keeps_them() {
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("../samples/sample_01.jpg");
        let raw = load(&source, true, Turn::default()).unwrap();
        // A 1024 square would hide a missed swap, so crop it to a portrait first.
        let tmp = std::env::temp_dir().join(format!("fcs-qt-turn-{}.png", std::process::id()));
        raw.crop_imm(0, 0, 600, 1000).save(&tmp).unwrap();
        let turned = load(
            &tmp,
            true,
            Turn {
                quarters: 1,
                tilt_deg: 0.0,
            },
        )
        .unwrap();
        let tilted = load(
            &tmp,
            true,
            Turn {
                quarters: 2,
                tilt_deg: 5.0,
            },
        )
        .unwrap();
        let _ = std::fs::remove_file(&tmp);
        assert_eq!((turned.width(), turned.height()), (1000, 600));
        assert_eq!((tilted.width(), tilted.height()), (600, 1000));
    }
}
