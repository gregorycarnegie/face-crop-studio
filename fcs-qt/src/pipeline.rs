//! Everything below the UI: loading the models, detecting, finishing and saving crops.
//! Mirrors what fcs-gui's core does, minus its egui types.

use anyhow::{Context, Result};
use fcs_core::{Detection, EyeRefiner, FaceDetector};
use fcs_utils::{
    EnhancementRuntime, GpuAvailability, GpuContext, GpuContextOptions, MetadataContext,
    OutputClaims, OutputOptions, Quality, config::AppSettings, estimate_sharpness, load_image,
    resolve_data_path, save_dynamic_image,
};
use image::{DynamicImage, ImageEncoder, codecs::jpeg::JpegEncoder};
use std::{
    fs::File,
    io::BufWriter,
    path::{Path, PathBuf},
    sync::Arc,
};

/// Longest side of the copy the canvas shows. The face boxes are positioned in fractions of
/// the image, so the canvas never needs the full resolution.
const DISPLAY_SIDE: u32 = 2560;
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
    pub detection: Detection,
    pub quality: Quality,
}

pub struct Detected {
    pub image: Arc<DynamicImage>,
    pub faces: Vec<Face>,
}

/// Load `path` (EXIF-oriented, as the detector sees it), find and refine its faces, and write
/// the canvas copy to `display`.
pub fn detect(engine: &Engine, path: &Path, display: &Path) -> Result<Detected> {
    let image = load_image(path).with_context(|| format!("failed to load {}", path.display()))?;
    let mut output = engine
        .detector
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
            detection,
        })
        .collect();
    save_display_copy(&image, display)?;
    Ok(Detected {
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

fn save_display_copy(image: &DynamicImage, path: &Path) -> Result<()> {
    let (w, h) = fit(image.width(), image.height(), DISPLAY_SIDE);
    let rgb = fcs_utils::resize_image(image, w, h, image::imageops::FilterType::Triangle);
    let file = BufWriter::new(File::create(path)?);
    JpegEncoder::new_with_quality(file, 90).write_image(
        rgb.as_raw(),
        w,
        h,
        image::ExtendedColorType::Rgb8,
    )?;
    Ok(())
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

/// Write a small PNG of a finished crop for the preview strip (PNG keeps a shaped crop's
/// transparent corners).
// shortcut: images reach QML as temp files because qtbridge 0.4 has no image provider; costs an
// encode and decode per preview. Replace with a QQuickImageProvider if previews feel slow.
pub fn save_preview(crop: &DynamicImage, path: &Path) -> Result<()> {
    crop.thumbnail(PREVIEW_SIDE, PREVIEW_SIDE)
        .save_with_format(path, image::ImageFormat::Png)?;
    Ok(())
}

/// Finish and save face `index` of `source` into `out_dir` as `{stem}_face_NN.{ext}`, the
/// GUI's naming. `claims` renames rather than replaces an existing file.
#[allow(clippy::too_many_arguments)] // each is a separate input; a struct for one caller adds nothing
pub fn export_face(
    engine: &Engine,
    image: &DynamicImage,
    source: &Path,
    face: &Face,
    index: usize,
    settings: &AppSettings,
    out_dir: &Path,
    claims: &OutputClaims,
) -> Result<PathBuf> {
    let finished = finish(engine, image, &face.detection, settings);
    let stem = source
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("face");
    let ext = settings.crop.output_format.extension();
    let path = claims.claim(
        out_dir.join(format!("{stem}_face_{:02}.{ext}", index + 1)),
        source,
    );
    save_dynamic_image(
        &finished.image,
        &path,
        &OutputOptions::from_crop_settings(&settings.crop),
        &MetadataContext {
            source_path: Some(source),
            crop_settings: Some(&settings.crop),
            detection_score: Some(face.detection.score),
            quality: Some(finished.quality),
            quality_score: Some(finished.quality_score),
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

/// A local path as a `file:` URL QML's `Image` can load.
pub fn path_to_url(path: &Path) -> String {
    let s = path.to_string_lossy().replace('\\', "/");
    let mut url = String::from(if s.starts_with('/') {
        "file://"
    } else {
        "file:///"
    });
    for c in s.chars() {
        match c {
            '%' => url.push_str("%25"),
            '#' => url.push_str("%23"),
            '?' => url.push_str("%3F"),
            ' ' => url.push_str("%20"),
            c => url.push(c),
        }
    }
    url
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_urls_round_trip_through_paths() {
        for path in ["C:/Users/a b/#1 100%.jpg", "/home/a b/ü?.png"] {
            assert_eq!(url_to_path(&path_to_url(Path::new(path))), Path::new(path));
        }
        assert_eq!(
            url_to_path("file:///C:/Photos/caf%C3%A9.jpg"),
            Path::new("C:/Photos/café.jpg")
        );
        // A stray `%` is kept rather than dropped.
        assert_eq!(url_to_path("file:///tmp/50%"), Path::new("/tmp/50%"));
    }

    /// The real models end to end: detect a sample, then export its face twice into one folder.
    /// The second export must be renamed, not replace the first.
    #[test]
    fn detects_and_exports_a_sample_without_overwriting() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let settings = AppSettings {
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

        let detected = detect(&engine, &source, &out.join("display.jpg")).expect("detect");
        assert_eq!(detected.faces.len(), 1);
        assert!(out.join("display.jpg").is_file());

        // A fresh claims set per export, as each click of Export makes one.
        let export = || {
            let claims = OutputClaims::new(fcs_utils::OverwritePolicy::KeepBoth);
            let face = &detected.faces[0];
            export_face(
                &engine,
                &detected.image,
                &source,
                face,
                0,
                &settings,
                &out,
                &claims,
            )
        };
        let (first, second) = (export(), export());
        let (first, second) = (first.expect("first export"), second.expect("second export"));
        assert_eq!(first.file_name().unwrap(), "sample_01_face_01.png");
        assert_ne!(first, second);
        let saved = image::open(&first).expect("readable crop");
        assert_eq!(
            (saved.width(), saved.height()),
            (settings.crop.output_width, settings.crop.output_height)
        );
        let _ = std::fs::remove_dir_all(&out);
    }

    #[test]
    fn fit_only_shrinks() {
        assert_eq!(fit(5120, 2560, 2560), (2560, 1280));
        assert_eq!(fit(800, 600, 2560), (800, 600));
    }
}
