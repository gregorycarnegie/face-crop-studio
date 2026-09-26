//! Export orchestration and filesystem writing.

use super::{
    encoders::{encode_avif, encode_bmp, encode_jpeg, encode_png, encode_tiff, encode_webp},
    metadata::{
        build_custom_metadata_payload, inject_jpeg_metadata, inject_png_metadata, load_jpeg_exif,
        load_png_exif_chunks,
    },
    types::{ImageFormatHint, MetadataContext, OutputOptions},
};
use crate::config::MetadataMode;
use anyhow::{Context, Result};
use image::DynamicImage;
use log::debug;
use std::{fs, path::Path};

/// Save an image using the provided options and metadata context.
///
/// Creates missing parent directories and replaces an existing destination. The replacement
/// goes through a temporary file in the same directory, so a failed write leaves whatever was
/// already there untouched rather than truncating it.
/// A recognized extension selects the format when `options.auto_detect` is true;
/// otherwise `options.format` is used, with PNG as the fallback.
///
/// Source EXIF copying supports PNG-to-PNG and JPEG-to-JPEG. Custom metadata
/// is embedded for PNG and JPEG only. WebP encoding is always lossless -- there was a
/// `webp_quality` setting for years, and it never reached the encoder, which `image` only
/// offers losslessly. Honouring it needs a different encoder, not a config field.
///
/// # Errors
///
/// Returns an error on directory creation, encoding, metadata serialization, or
/// file creation/write failure -- including a failure that only shows up when the buffered
/// bytes reach the disk, which this used to discard and report as success. See
/// [`crate::write_atomically`] for what "replaces" guarantees and what it does not.
pub fn save_dynamic_image(
    image: &DynamicImage,
    destination: &Path,
    options: &OutputOptions,
    metadata: &MetadataContext<'_>,
) -> Result<()> {
    if let Some(parent) = destination.parent().filter(|p| !p.exists()) {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let format = determine_format(destination, options);
    debug!(
        "Saving crop to {} using {:?} format",
        destination.display(),
        format
    );

    let mut encoded = match format {
        ImageFormatHint::Png => encode_png(image, options.png_compression)?,
        ImageFormatHint::Jpeg => encode_jpeg(image, options.jpeg_quality)?,
        ImageFormatHint::Webp => encode_webp(image)?,
        ImageFormatHint::Tiff => encode_tiff(image)?,
        ImageFormatHint::Bmp => encode_bmp(image)?,
        ImageFormatHint::Avif => encode_avif(image)?,
    };

    // Prepare metadata payload if applicable.
    let custom_payload = build_custom_metadata_payload(&options.metadata, metadata)?;

    // Strip mode needs no guard here: the payload is None and no EXIF is loaded, so the
    // injectors hand the encoded bytes straight back.
    let preserve = matches!(options.metadata.mode, MetadataMode::Preserve);
    match format {
        ImageFormatHint::Png => {
            let exif_chunks = if preserve {
                load_png_exif_chunks(metadata.source_path)
            } else {
                Vec::new()
            };
            encoded = inject_png_metadata(encoded, &exif_chunks, custom_payload.as_deref());
        }
        ImageFormatHint::Jpeg => {
            let exif = preserve
                .then(|| load_jpeg_exif(metadata.source_path))
                .flatten();
            encoded = inject_jpeg_metadata(encoded, exif, custom_payload.as_deref());
        }
        ImageFormatHint::Webp
        | ImageFormatHint::Tiff
        | ImageFormatHint::Bmp
        | ImageFormatHint::Avif => {
            // Metadata injection not yet implemented for these formats
        }
    }

    crate::write_atomically(destination, &encoded)
}

pub(super) fn determine_format(path: &Path, options: &OutputOptions) -> ImageFormatHint {
    if !options.auto_detect {
        return options.format.unwrap_or_default();
    }

    if let Some(fmt) = path
        .extension()
        .and_then(|e| e.to_str())
        .and_then(ImageFormatHint::from_extension)
    {
        fmt
    } else {
        options.format.unwrap_or_default()
    }
}

/// Append a suffix to a filename, before its last extension.
///
/// ```
/// use fcs_utils::append_suffix_to_filename;
///
/// assert_eq!(append_suffix_to_filename("face.png", "_lowq"), "face_lowq.png");
/// assert_eq!(append_suffix_to_filename("face", "_lowq"), "face_lowq");
/// assert_eq!(append_suffix_to_filename("scan.tar.gz", "_1"), "scan.tar_1.gz");
/// ```
pub fn append_suffix_to_filename(name: &str, suffix: &str) -> String {
    if suffix.is_empty() {
        return name.to_string();
    }
    if let Some(idx) = name.rfind('.') {
        let (base, ext) = name.split_at(idx);
        format!("{base}{suffix}{ext}")
    } else {
        format!("{name}{suffix}")
    }
}

/// What to do when an export's file name already exists on disk from before the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OverwritePolicy {
    /// Replace the existing file. The CLI's behaviour, and the GUI's once the user agrees.
    #[default]
    Overwrite,
    /// Keep it, and save the new crop as `name(2).ext`, `name(3).ext`...
    KeepBoth,
}

/// Destinations already taken in one batch, so two sources that name the same file get
/// distinct names instead of the later silently replacing the earlier.
///
/// `a/portrait.jpg` and `b/portrait.jpg` both become `portrait_face1.png` under the default
/// naming, and the writer replaces what it finds. A clash within the batch is always renamed;
/// a file left by an earlier run is replaced or kept according to the [`OverwritePolicy`].
///
/// ponytail: first come, first served, so under a parallel batch which clashing source keeps the
/// plain name depends on completion order. Nothing is lost either way; a pre-pass over every
/// planned name would make it deterministic.
#[derive(Debug, Default)]
pub struct OutputClaims {
    taken: std::sync::Mutex<std::collections::HashMap<String, std::path::PathBuf>>,
    policy: OverwritePolicy,
}

impl OutputClaims {
    /// Claims for one batch under `policy`.
    pub fn new(policy: OverwritePolicy) -> Self {
        Self {
            policy,
            ..Self::default()
        }
    }

    /// `path` itself, or `path` with `(2)`, `(3)`... before the extension if a different source
    /// already claimed it -- or, under [`OverwritePolicy::KeepBoth`], if it already exists on
    /// disk. The same source claiming again gets the same path back, so a watched file that is
    /// saved twice still replaces its own output.
    ///
    /// Compared without case: Windows and macOS treat `Portrait_face1.png` and
    /// `portrait_face1.png` as one file. The existence check runs under the lock, so two workers
    /// cannot both settle on the same free name.
    pub fn claim(&self, path: std::path::PathBuf, source: &Path) -> std::path::PathBuf {
        use std::collections::hash_map::Entry;
        let mut taken = self.taken.lock().unwrap_or_else(|e| e.into_inner());
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let mut candidate = path.clone();
        for n in 2.. {
            match taken.entry(candidate.to_string_lossy().to_lowercase()) {
                Entry::Occupied(owner) if owner.get() == source => break,
                Entry::Vacant(slot)
                    if self.policy == OverwritePolicy::Overwrite || !candidate.exists() =>
                {
                    slot.insert(source.to_path_buf());
                    break;
                }
                _ => {
                    candidate =
                        path.with_file_name(append_suffix_to_filename(&name, &format!("({n})")));
                }
            }
        }
        if candidate != path {
            log::info!(
                "{} is taken; saving {}'s crop as {}",
                path.display(),
                source.display(),
                candidate.display()
            );
        }
        candidate
    }
}
