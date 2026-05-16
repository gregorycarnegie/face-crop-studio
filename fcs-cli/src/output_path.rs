use std::path::{Path, PathBuf};

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

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_crop_filename(
    source_stem: &str,
    crop_index: usize,
    output_width: u32,
    output_height: u32,
    ext: &str,
    naming_template: Option<&str>,
    quality_suffix: Option<&str>,
    timestamp: u64,
) -> String {
    let mut out_name = if let Some(tmpl) = naming_template {
        let mut name = tmpl.to_string();
        name = name.replace("{original}", source_stem);
        name = name.replace("{index}", &(crop_index + 1).to_string());
        name = name.replace("{width}", &output_width.to_string());
        name = name.replace("{height}", &output_height.to_string());
        name = name.replace("{ext}", ext);
        name = name.replace("{timestamp}", &timestamp.to_string());
        if !tmpl.contains("{ext}") {
            format!("{}.{}", name, ext)
        } else {
            name
        }
    } else {
        format!("{}_face{}.{}", source_stem, crop_index + 1, ext)
    };

    if let Some(suffix) = quality_suffix {
        out_name = append_suffix_to_filename(&out_name, suffix);
    }

    out_name
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_crop_output_path(
    out_dir: &Path,
    source_path: &Path,
    crop_index: usize,
    output_width: u32,
    output_height: u32,
    ext: &str,
    naming_template: Option<&str>,
    quality_suffix: Option<&str>,
    timestamp: u64,
) -> PathBuf {
    let source_stem = source_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("image");
    out_dir.join(build_crop_filename(
        source_stem,
        crop_index,
        output_width,
        output_height,
        ext,
        naming_template,
        quality_suffix,
        timestamp,
    ))
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

    #[test]
    fn build_crop_filename_applies_template_and_quality_suffix() {
        let name = build_crop_filename(
            "portrait",
            1,
            512,
            640,
            "jpg",
            Some("{original}_{index}_{width}x{height}_{timestamp}"),
            Some("_highq"),
            1234,
        );

        assert_eq!(name, "portrait_2_512x640_1234_highq.jpg");
    }

    #[test]
    fn build_crop_filename_appends_extension_when_template_omits_it() {
        let name = build_crop_filename(
            "portrait",
            0,
            512,
            640,
            "png",
            Some("{original}_face{index}"),
            None,
            0,
        );

        assert_eq!(name, "portrait_face1.png");
    }

    #[test]
    fn build_crop_filename_keeps_template_extension_placeholder() {
        let name = build_crop_filename(
            "portrait",
            0,
            512,
            640,
            "webp",
            Some("{original}_{index}.{ext}"),
            None,
            0,
        );

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
            0,
            512,
            640,
            "png",
            None,
            Some("_lowq"),
            0,
        );

        assert_eq!(
            out.file_name().and_then(|s| s.to_str()),
            Some("portrait_face1_lowq.png")
        );
    }

    #[test]
    fn build_crop_output_path_uses_default_name_when_source_stem_is_missing() {
        let dir = tempdir().unwrap();
        let out =
            build_crop_output_path(dir.path(), Path::new(""), 0, 256, 256, "png", None, None, 0);

        assert_eq!(
            out.file_name().and_then(|s| s.to_str()),
            Some("image_face1.png")
        );
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
