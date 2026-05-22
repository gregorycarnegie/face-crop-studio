//! Image loading and conversion helpers shared across the workspace.
//!
//! This module centralizes routines for reading files, resizing RGB buffers, and converting to
//! tensor-friendly layouts while preserving compatibility with OpenCV glue code.

use anyhow::{Context, Result};
use fast_image_resize::{self as fir, images::Image as FirImage, images::ImageRef as FirImageRef};
use image::metadata::Orientation;
use image::{DynamicImage, ImageDecoder, ImageReader, RgbImage, imageops::FilterType};
use ndarray::Array3;
use rayon::prelude::*;
use std::{borrow::Cow, path::Path};

/// Filename extensions recognized as decodable raster images across CLI and GUI.
/// Kept in lower-case; callers should normalize input before comparison.
pub const SUPPORTED_IMAGE_EXTENSIONS: &[&str] =
    &["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"];

/// Returns `true` if the given path's extension is in [`SUPPORTED_IMAGE_EXTENSIONS`].
pub fn is_supported_image_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|ext| {
            SUPPORTED_IMAGE_EXTENSIONS
                .iter()
                .any(|supported| ext.eq_ignore_ascii_case(supported))
        })
        .unwrap_or(false)
}

/// Load an image from disk into memory.
///
/// # Arguments
///
/// * `path` - The path to the image file.
pub fn load_image<P: AsRef<Path>>(path: P) -> Result<DynamicImage> {
    let path_ref = path.as_ref();
    let reader = ImageReader::open(path_ref)
        .with_context(|| format!("failed to open image {}", path_ref.display()))?
        .with_guessed_format()
        .with_context(|| format!("failed to guess image format for {}", path_ref.display()))?;

    let mut decoder = reader
        .into_decoder()
        .with_context(|| format!("failed to create decoder for {}", path_ref.display()))?;

    let orientation = decoder.orientation().unwrap_or(Orientation::NoTransforms);
    let mut image = DynamicImage::from_decoder(decoder)
        .with_context(|| format!("failed to decode image {}", path_ref.display()))?;
    image.apply_orientation(orientation);
    Ok(image)
}

/// Load an image from disk without applying EXIF orientation.
pub fn load_image_raw<P: AsRef<Path>>(path: P) -> Result<DynamicImage> {
    let path_ref = path.as_ref();
    ImageReader::open(path_ref)
        .with_context(|| format!("failed to open image {}", path_ref.display()))?
        .with_guessed_format()
        .with_context(|| format!("failed to guess image format for {}", path_ref.display()))?
        .decode()
        .with_context(|| format!("failed to decode image {}", path_ref.display()))
}

/// Resize an image to the requested resolution using the provided filter.
///
/// # Arguments
///
/// * `image` - The image to resize.
/// * `width` - The target width.
/// * `height` - The target height.
/// * `filter` - The sampling filter to use for resizing.
pub fn resize_image(image: &DynamicImage, width: u32, height: u32, filter: FilterType) -> RgbImage {
    if filter == FilterType::Nearest
        && let Some(fast) = resize_image_fast(image, width, height)
    {
        return fast;
    }
    image.resize_exact(width, height, filter).to_rgb8()
}

fn resize_image_fast(image: &DynamicImage, width: u32, height: u32) -> Option<RgbImage> {
    let rgb: Cow<'_, RgbImage> = match image.as_rgb8() {
        Some(rgb) => Cow::Borrowed(rgb),
        None => Cow::Owned(image.to_rgb8()),
    };

    let src = FirImageRef::new(
        rgb.width(),
        rgb.height(),
        rgb.as_raw(),
        fir::PixelType::U8x3,
    )
    .ok()?;

    let mut dst = FirImage::new(width, height, fir::PixelType::U8x3);
    let mut resizer = fir::Resizer::new();
    let options = fir::ResizeOptions::new().resize_alg(fir::ResizeAlg::Nearest);
    resizer.resize(&src, &mut dst, &options).ok()?;

    RgbImage::from_raw(width, height, dst.into_vec())
}

/// Convert an RGB image into a BGR CHW array with values matching OpenCV's `blobFromImage`.
///
/// This function rearranges the memory layout from HWC (height, width, channels) to
/// CHW (channels, height, width) and swaps the red and blue channels.
///
/// Uses rayon to process channels in parallel for improved performance.
/// Allocates one contiguous output buffer and fills it directly.
///
/// # Arguments
///
/// * `image` - The RGB image to convert.
pub fn rgb_to_bgr_chw(image: &RgbImage) -> Array3<f32> {
    let (width, height) = image.dimensions();
    let w = width as usize;
    let h = height as usize;
    let channel_len = w * h;
    let row_stride = w * 3; // Keep as multiplication since 3 is not a power of 2
    let pixels = image.as_raw();

    let mut data = vec![0.0f32; 3 * channel_len];

    let (b_slice, rest) = data.split_at_mut(channel_len);
    let (g_slice, r_slice) = rest.split_at_mut(channel_len);

    b_slice
        .par_chunks_mut(w)
        .zip(g_slice.par_chunks_mut(w))
        .zip(r_slice.par_chunks_mut(w))
        .enumerate()
        .for_each(|(y, ((b_row, g_row), r_row))| {
            let src_row = &pixels[y * row_stride..(y + 1) * row_stride];
            for x in 0..w {
                // Optimized: x * 3 = (x << 1) + x
                let src = (x << 1) + x;
                b_row[x] = src_row[src + 2] as f32;
                g_row[x] = src_row[src + 1] as f32;
                r_row[x] = src_row[src] as f32;
            }
        });

    Array3::from_shape_vec((3, h, w), data).expect("shape matches data length")
}

/// Convert any dynamic image into a BGR CHW array by first converting to RGB.
///
/// # Arguments
///
/// * `image` - The dynamic image to convert.
pub fn dynamic_to_bgr_chw(image: &DynamicImage) -> Array3<f32> {
    match image.as_rgb8() {
        Some(rgb) => rgb_to_bgr_chw(rgb),
        None => rgb_to_bgr_chw(&image.to_rgb8()),
    }
}

/// Compute scale factors used to reproject detections from model space to original space.
///
/// This is necessary when the model runs on a resized version of the original image.
///
/// # Arguments
///
/// * `original` - A tuple of the original image's (width, height).
/// * `target` - A tuple of the resized image's (width, height).
pub fn compute_resize_scales(original: (u32, u32), target: (u32, u32)) -> Result<(f32, f32)> {
    let (orig_w, orig_h) = original;
    let (target_w, target_h) = target;
    anyhow::ensure!(
        target_w > 0 && target_h > 0,
        "target dimensions must be non-zero"
    );
    anyhow::ensure!(
        orig_w > 0 && orig_h > 0,
        "original dimensions must be non-zero"
    );
    Ok((
        orig_w as f32 / target_w as f32,
        orig_h as f32 / target_h as f32,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    #[test]
    fn rgb_to_bgr_chw_converts_correctly() {
        let mut image = RgbImage::new(2, 2);
        image.put_pixel(0, 0, image::Rgb([0, 128, 255]));
        image.put_pixel(1, 0, image::Rgb([255, 128, 0]));
        image.put_pixel(0, 1, image::Rgb([64, 64, 64]));
        image.put_pixel(1, 1, image::Rgb([255, 255, 255]));

        let array = rgb_to_bgr_chw(&image);
        assert_eq!(array.shape(), &[3, 2, 2]);

        assert_eq!(array[(0, 0, 0)], 255.0);
        assert_eq!(array[(2, 0, 0)], 0.0);
        assert_eq!(array[(1, 0, 1)], 128.0);
    }

    #[test]
    fn compute_resize_scales_returns_expected_values() {
        let (sx, sy) = compute_resize_scales((640, 480), (320, 240)).unwrap();
        assert_eq!(sx, 2.0);
        assert_eq!(sy, 2.0);
    }

    #[test]
    fn compute_resize_scales_rejects_zero() {
        assert!(compute_resize_scales((0, 480), (320, 240)).is_err());
        assert!(compute_resize_scales((640, 480), (0, 240)).is_err());
    }

    #[test]
    fn dynamic_to_bgr_chw_matches_rgb_input() {
        let mut image = RgbImage::new(2, 1);
        image.put_pixel(0, 0, Rgb([10, 20, 30]));
        image.put_pixel(1, 0, Rgb([100, 150, 200]));
        let dynamic = DynamicImage::ImageRgb8(image.clone());

        let from_dynamic = dynamic_to_bgr_chw(&dynamic);
        let from_rgb = rgb_to_bgr_chw(&image);

        assert_eq!(from_dynamic, from_rgb);
    }

    #[test]
    fn load_image_errors_on_missing_file() {
        let result = super::load_image("/nonexistent/path/image.png");
        assert!(result.is_err());
    }

    #[test]
    fn resize_image_triangle_path() {
        // Triangle filter bypasses the fast path — exercises the fallback branch
        let mut image = ImageBuffer::<Rgb<u8>, _>::new(4, 4);
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            *pixel = Rgb([(x * 40) as u8, (y * 40) as u8, 128]);
        }
        let dynamic = DynamicImage::ImageRgb8(image);
        let result = super::resize_image(&dynamic, 2, 2, FilterType::Triangle);
        assert_eq!(result.dimensions(), (2, 2));
    }

    #[test]
    fn fast_resize_matches_reference_nearest() {
        let mut image = ImageBuffer::<Rgb<u8>, _>::new(5, 3);
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            let r = (x * 50 + y * 30) as u8;
            let g = (x * 20 + y * 60) as u8;
            let b = (x * 90 + y * 10) as u8;
            *pixel = Rgb([r, g, b]);
        }
        let dynamic = DynamicImage::ImageRgb8(image.clone());

        let expected = dynamic.resize_exact(7, 4, FilterType::Nearest).to_rgb8();
        let fast = resize_image(&dynamic, 7, 4, FilterType::Nearest);

        assert_eq!(fast.as_raw(), expected.as_raw());
    }
}

#[cfg(test)]
mod benchmarks {
    use super::*;
    use image::RgbImage;
    use std::time::Instant;

    fn rgb_to_bgr_chw_baseline(image: &RgbImage) -> Array3<f32> {
        let (width, height) = image.dimensions();
        let w = width as usize;
        let h = height as usize;
        let channel_len = w * h;
        let row_stride = w * 3;
        let pixels = image.as_raw();
        let mut data = vec![0f32; 3 * channel_len];

        data.par_chunks_mut(channel_len)
            .enumerate()
            .for_each(|(channel, channel_buf)| {
                let rgb_index = match channel {
                    0 => 2,
                    1 => 1,
                    2 => 0,
                    _ => unreachable!(),
                };

                for (row_idx, dst_row) in channel_buf.chunks_mut(w).enumerate() {
                    let src_row = &pixels[row_idx * row_stride..(row_idx + 1) * row_stride];
                    for x in 0..w {
                        dst_row[x] = src_row[x * 3 + rgb_index] as f32;
                    }
                }
            });

        Array3::from_shape_vec((3, h, w), data).expect("shape matches data length")
    }

    #[test]
    fn bench_rgb_to_bgr_chw() {
        // 640x640 image (typical inference size)
        let img = RgbImage::new(640, 640);

        for _ in 0..5 {
            let _ = rgb_to_bgr_chw_baseline(&img);
            let _ = rgb_to_bgr_chw(&img);
        }

        let start = Instant::now();
        let iterations = 100;
        for _ in 0..iterations {
            let _ = rgb_to_bgr_chw(&img);
        }
        let optimized = start.elapsed();

        let start_baseline = Instant::now();
        for _ in 0..iterations {
            let _ = rgb_to_bgr_chw_baseline(&img);
        }
        let baseline = start_baseline.elapsed();

        println!(
            "rgb_to_bgr_chw optimized avg: {:?}, baseline avg: {:?}",
            optimized / iterations,
            baseline / iterations
        );
    }
}
