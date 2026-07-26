use std::path::{Component, Path, PathBuf};

use anyhow::Result;
use fcs_utils::{
    ImageFormatHint, MetadataContext, OutputOptions, append_suffix_to_filename, save_dynamic_image,
};

use crate::ProcessedCrop;

pub(crate) fn normalized_output_extension(format: ImageFormatHint) -> &'static str {
    match format {
        ImageFormatHint::Png => "png",
        ImageFormatHint::Jpeg => "jpg",
        ImageFormatHint::Webp => "webp",
        ImageFormatHint::Tiff => "tif",
        ImageFormatHint::Bmp => "bmp",
        ImageFormatHint::Avif => "avif",
    }
}

/// Inputs that determine a crop's output filename and directory placement.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CropFilenameSpec<'a> {
    pub source_stem: &'a str,
    pub crop_index: usize,
    pub output_width: u32,
    pub output_height: u32,
    pub ext: &'a str,
    pub naming_template: Option<&'a str>,
    pub quality_suffix: Option<&'a str>,
    pub timestamp: u64,
}

pub(crate) fn build_crop_filename(spec: CropFilenameSpec<'_>) -> String {
    let mut out_name = if let Some(tmpl) = spec.naming_template {
        let mut name = tmpl.to_string();
        name = name.replace("{original}", spec.source_stem);
        name = name.replace("{index}", &(spec.crop_index + 1).to_string());
        name = name.replace("{width}", &spec.output_width.to_string());
        name = name.replace("{height}", &spec.output_height.to_string());
        name = name.replace("{ext}", spec.ext);
        name = name.replace("{timestamp}", &spec.timestamp.to_string());
        if !tmpl.contains("{ext}") {
            format!("{}.{}", name, spec.ext)
        } else {
            name
        }
    } else {
        format!(
            "{}_face{}.{}",
            spec.source_stem,
            spec.crop_index + 1,
            spec.ext
        )
    };

    if let Some(suffix) = spec.quality_suffix {
        out_name = append_suffix_to_filename(&out_name, suffix);
    }

    out_name
}

pub(crate) fn build_crop_output_path(
    out_dir: &Path,
    source_path: &Path,
    spec: CropFilenameSpec<'_>,
) -> PathBuf {
    let source_stem = source_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("image");
    let spec = CropFilenameSpec {
        source_stem,
        ..spec
    };
    // A naming template like `../{original}` — or a source stem holding separators — must
    // not write outside the chosen output directory. Drop every path component but the
    // last, then neutralise what Windows still re-parses when joined (a `a:` drive
    // prefix replaces the whole path), so the result is always one plain filename.
    let name = build_crop_filename(spec);
    let file = Path::new(&name)
        .components()
        .filter_map(|c| match c {
            Component::Normal(s) => s.to_str(),
            _ => None,
        })
        .next_back()
        .unwrap_or_default()
        .replace([':', '/', '\\'], "_");

    if file.trim_matches('.').is_empty() {
        out_dir.join("image")
    } else {
        out_dir.join(file)
    }
}

pub(crate) fn save_processed_crop(
    crop: &ProcessedCrop,
    out_path: &Path,
    image_path: &Path,
    settings: &fcs_utils::config::AppSettings,
    output_options: &OutputOptions,
) -> Result<()> {
    let metadata_ctx = MetadataContext {
        source_path: Some(image_path),
        crop_settings: Some(&settings.crop),
        detection_score: Some(crop.score),
        quality: Some(crop.quality),
        quality_score: Some(crop.quality_score),
    };

    save_dynamic_image(&crop.image, out_path, output_options, &metadata_ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn spec<'a>(
        source_stem: &'a str,
        crop_index: usize,
        ext: &'a str,
        naming_template: Option<&'a str>,
        quality_suffix: Option<&'a str>,
        timestamp: u64,
    ) -> CropFilenameSpec<'a> {
        CropFilenameSpec {
            source_stem,
            crop_index,
            output_width: 512,
            output_height: 640,
            ext,
            naming_template,
            quality_suffix,
            timestamp,
        }
    }

    #[test]
    fn build_crop_filename_applies_template_and_quality_suffix() {
        let name = build_crop_filename(spec(
            "portrait",
            1,
            "jpg",
            Some("{original}_{index}_{width}x{height}_{timestamp}"),
            Some("_highq"),
            1234,
        ));

        assert_eq!(name, "portrait_2_512x640_1234_highq.jpg");
    }

    #[test]
    fn build_crop_filename_appends_extension_when_template_omits_it() {
        let name = build_crop_filename(spec(
            "portrait",
            0,
            "png",
            Some("{original}_face{index}"),
            None,
            0,
        ));

        assert_eq!(name, "portrait_face1.png");
    }

    #[test]
    fn build_crop_filename_keeps_template_extension_placeholder() {
        let name = build_crop_filename(spec(
            "portrait",
            0,
            "webp",
            Some("{original}_{index}.{ext}"),
            None,
            0,
        ));

        assert_eq!(name, "portrait_1.webp");
    }

    #[test]
    fn normalized_output_extension_maps_known_formats() {
        assert_eq!(normalized_output_extension(ImageFormatHint::Png), "png");
        assert_eq!(normalized_output_extension(ImageFormatHint::Jpeg), "jpg");
        assert_eq!(normalized_output_extension(ImageFormatHint::Webp), "webp");
    }

    #[test]
    fn build_crop_output_path_joins_directory_and_filename() {
        let dir = tempdir().unwrap();
        let out = build_crop_output_path(
            dir.path(),
            Path::new("/tmp/portrait.jpg"),
            spec("", 0, "png", None, Some("_lowq"), 0),
        );

        assert_eq!(
            out.file_name().and_then(|s| s.to_str()),
            Some("portrait_face1_lowq.png")
        );
    }

    #[test]
    fn build_crop_output_path_uses_default_name_when_source_stem_is_missing() {
        let dir = tempdir().unwrap();
        let out = build_crop_output_path(
            dir.path(),
            Path::new(""),
            CropFilenameSpec {
                output_width: 256,
                output_height: 256,
                ..spec("", 0, "png", None, None, 0)
            },
        );

        assert_eq!(
            out.file_name().and_then(|s| s.to_str()),
            Some("image_face1.png")
        );
    }

    proptest::proptest! {
        /// No template or source stem may place a crop outside the requested output
        /// directory — `--naming-template "../{original}"` must stay put.
        #[test]
        fn build_crop_output_path_stays_under_out_dir(
            stem in r"[^\x00]{0,24}",
            template in proptest::option::of(r"[^\x00]{0,24}"),
            suffix in proptest::option::of(r"[^\x00]{0,8}"),
        ) {
            let out_dir = Path::new("out");
            let path = build_crop_output_path(
                out_dir,
                Path::new(&stem),
                spec("", 0, "png", template.as_deref(), suffix.as_deref(), 0),
            );

            proptest::prop_assert!(path.starts_with(out_dir), "escaped out_dir: {path:?}");
            proptest::prop_assert!(
                path.components().all(|c| matches!(c, Component::Normal(_))),
                "non-normal component in {path:?}"
            );
        }
    }

    #[test]
    fn build_crop_output_path_strips_template_traversal() {
        let dir = tempdir().unwrap();
        let out = build_crop_output_path(
            dir.path(),
            Path::new("portrait.jpg"),
            spec("", 0, "png", Some("../../{original}"), None, 0),
        );

        assert_eq!(out, dir.path().join("portrait.png"));
    }

    #[test]
    fn save_processed_crop_writes_image_to_disk() {
        use fcs_utils::{Quality, config::AppSettings};
        use image::{DynamicImage, Rgba, RgbaImage};

        let dir = tempdir().unwrap();
        let out_path = dir.path().join("saved.png");
        let image_path = dir.path().join("source.png");

        // Write a tiny source PNG so metadata context has a valid path.
        let img = DynamicImage::ImageRgba8(RgbaImage::from_pixel(2, 2, Rgba([128u8, 0, 0, 255])));
        img.save(&image_path).unwrap();

        let mut settings = AppSettings::default();
        settings.crop.output_format = ImageFormatHint::Png;
        let options = OutputOptions::from_crop_settings(&settings.crop);

        let crop = crate::ProcessedCrop {
            index: 0,
            image: img,
            quality: Quality::High,
            quality_score: 1234.0,
            score: 0.5,
        };

        save_processed_crop(&crop, &out_path, &image_path, &settings, &options).unwrap();
        assert!(out_path.exists());
    }
}
