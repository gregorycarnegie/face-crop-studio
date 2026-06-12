//! Image format encoders.

use super::types::PngCompression;
use anyhow::{Context, Result};
use image::{
    DynamicImage, ExtendedColorType, ImageEncoder,
    codecs::{
        bmp::BmpEncoder,
        jpeg::JpegEncoder,
        png::{FilterType, PngEncoder},
        tiff::TiffEncoder,
        webp::WebPEncoder,
    },
};
use rgb::FromSlice;
use std::io::Cursor;

fn encode_rgba8<F>(image: &DynamicImage, encode_op: F, context: &str) -> Result<Vec<u8>>
where
    F: FnOnce(&mut Cursor<Vec<u8>>, &[u8], u32, u32) -> image::ImageResult<()>,
{
    let rgba = image.to_rgba8();
    let mut cursor = Cursor::new(Vec::new());
    encode_op(&mut cursor, rgba.as_raw(), rgba.width(), rgba.height())
        .context(context.to_string())?;
    Ok(cursor.into_inner())
}

macro_rules! encode_impl {
    ($image:expr, $context:literal, |$cursor:ident| $encoder:expr) => {
        encode_rgba8(
            $image,
            |$cursor, data, width, height| {
                $encoder.write_image(data, width, height, ExtendedColorType::Rgba8)
            },
            $context,
        )
    };
}

pub(super) fn encode_jpeg(image: &DynamicImage, quality: u8) -> Result<Vec<u8>> {
    let rgb = image.to_rgb8();
    let mut buffer = Vec::new();
    {
        let encoder = JpegEncoder::new_with_quality(&mut buffer, quality);
        encoder
            .write_image(
                rgb.as_raw(),
                rgb.width(),
                rgb.height(),
                ExtendedColorType::Rgb8,
            )
            .context("failed to encode JPEG")?;
    }
    Ok(buffer)
}

pub(super) fn encode_avif(image: &DynamicImage) -> Result<Vec<u8>> {
    let rgba = image.to_rgba8();
    let width = rgba.width() as usize;
    let height = rgba.height() as usize;
    let raw = rgba.as_raw();

    let pixels = raw.as_rgba();
    let img = imgref::Img::new(pixels, width, height);

    // Use UnassociatedDirty to preserve RGB values in transparent pixels,
    // preventing the "green tint" issue often caused by premultiplied alpha or YUV subsampling on zeroed pixels.
    let res = ravif::Encoder::new()
        .with_quality(80.0)
        .with_speed(4)
        .with_alpha_color_mode(ravif::AlphaColorMode::UnassociatedDirty)
        .encode_rgba(img)
        .map_err(|e| anyhow::anyhow!("AVIF encoding failed: {}", e))?;

    Ok(res.avif_file)
}

pub(super) fn encode_bmp(image: &DynamicImage) -> Result<Vec<u8>> {
    encode_impl!(image, "failed to encode BMP", |cursor| {
        BmpEncoder::new(cursor)
    })
}

pub(super) fn encode_png(image: &DynamicImage, compression: PngCompression) -> Result<Vec<u8>> {
    encode_impl!(image, "failed to encode PNG", |cursor| {
        PngEncoder::new_with_quality(cursor, compression.into_image(), FilterType::Adaptive)
    })
}

pub(super) fn encode_tiff(image: &DynamicImage) -> Result<Vec<u8>> {
    encode_impl!(image, "failed to encode TIFF", |cursor| {
        TiffEncoder::new(cursor)
    })
}

pub(super) fn encode_webp(image: &DynamicImage) -> Result<Vec<u8>> {
    encode_impl!(image, "failed to encode WebP", |cursor| {
        WebPEncoder::new_lossless(cursor)
    })
}
