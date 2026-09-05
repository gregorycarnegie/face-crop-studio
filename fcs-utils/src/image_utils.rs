//! Image loading and conversion helpers shared across the workspace.
//!
//! This module centralizes routines for reading files, resizing RGB buffers, and converting to
//! tensor-friendly layouts while preserving compatibility with OpenCV glue code.

use anyhow::{Context, Result};
use fast_image_resize::{
    self as fir,
    images::{Image as FirImage, ImageRef as FirImageRef},
};
use image::{
    DynamicImage, ImageDecoder, ImageReader, RgbImage, imageops::FilterType, metadata::Orientation,
};
use rayon::prelude::*;
use std::{borrow::Cow, path::Path};

/// Filename extensions recognized as decodable raster images across CLI and GUI.
/// Kept in lower-case; callers should normalize input before comparison.
///
/// Camera RAW extensions are only present when the `raw` feature is enabled, so file
/// dialogs and batch enqueue logic stay in sync with what [`load_image`] can decode.
pub const SUPPORTED_IMAGE_EXTENSIONS: &[&str] = &[
    "jpg",
    "jpeg",
    "png",
    "webp",
    "bmp",
    "tif",
    "tiff",
    // Unconditional, unlike raw/heic below: the `avif` and `avif-native` features
    // of the `image` dependency are always on, and dav1d is installed on all
    // three CI/release platforms for exactly this.
    "avif",
    #[cfg(feature = "raw")]
    "dng",
    #[cfg(feature = "raw")]
    "cr2",
    #[cfg(feature = "raw")]
    "cr3",
    #[cfg(feature = "raw")]
    "nef",
    #[cfg(feature = "raw")]
    "arw",
    #[cfg(feature = "raw")]
    "rw2",
    #[cfg(feature = "raw")]
    "orf",
    #[cfg(feature = "raw")]
    "raf",
    #[cfg(feature = "raw")]
    "srw",
    #[cfg(feature = "raw")]
    "pef",
    #[cfg(feature = "heic")]
    "heic",
    #[cfg(feature = "heic")]
    "heif",
    #[cfg(feature = "heic")]
    "hif",
];

/// Returns `true` if `ext` (case-insensitive) is one of the camera RAW formats routed
/// through the [`imagepipe`] decoder rather than the `image` crate.
#[cfg(feature = "raw")]
fn is_raw_extension(ext: &str) -> bool {
    const RAW_EXTENSIONS: &[&str] = &[
        "dng", "cr2", "cr3", "nef", "arw", "rw2", "orf", "raf", "srw", "pef",
    ];
    RAW_EXTENSIONS
        .iter()
        .any(|raw| ext.eq_ignore_ascii_case(raw))
}

/// Decode a camera RAW file into an 8-bit sRGB image via `imagepipe` (demosaic, white
/// balance, gamma). Orientation from the RAW metadata is already applied by the pipeline.
///
/// `rawloader` calls `.unwrap()` on DNG variants it does not support (e.g. unusual
/// lossless-JPEG precisions). That panic fires inside its internal `rayon` parallel decode,
/// on many workers at once — simultaneous panics on the shared global pool trip rayon's
/// abort path, which a plain `catch_unwind`/thread join cannot stop. Running the decode in
/// a dedicated single-thread pool means at most one panic occurs, so it propagates cleanly
/// to the `catch_unwind` here and becomes a recoverable `Err` instead of crashing the app.
#[cfg(feature = "raw")]
fn load_raw_image(path: &Path) -> Result<DynamicImage> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .thread_name(|_| "fcs-raw-decode".into())
        .build()
        .context("failed to build RAW decode thread pool")?;

    let decoded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pool.install(|| imagepipe::simple_decode_8bit(path, 0, 0))
    }))
    .map_err(|_| {
        anyhow::anyhow!(
            "RAW decoder panicked on {} (unsupported camera/format variant)",
            path.display()
        )
    })?
    .map_err(|e| anyhow::anyhow!("failed to decode RAW image {}: {e}", path.display()))?;

    let buffer = RgbImage::from_raw(decoded.width as u32, decoded.height as u32, decoded.data)
        .with_context(|| {
            format!(
                "RAW decode returned mismatched buffer for {}",
                path.display()
            )
        })?;
    Ok(DynamicImage::ImageRgb8(buffer))
}

/// Returns `true` if `ext` (case-insensitive) is a HEIC/HEIF container routed through
/// the native `libheif` decoder rather than the `image` crate.
#[cfg(feature = "heic")]
fn is_heic_extension(ext: &str) -> bool {
    const HEIC_EXTENSIONS: &[&str] = &["heic", "heif", "hif"];
    HEIC_EXTENSIONS.iter().any(|h| ext.eq_ignore_ascii_case(h))
}

/// Decode a HEIC/HEIF file into an 8-bit sRGB image via native `libheif`. libheif applies
/// the container's `irot`/`imir` orientation transforms during decode, so the result is
/// already upright (this is why HEIC routes the same way in `load_image_raw`).
#[cfg(feature = "heic")]
fn load_heic_image(path: &Path) -> Result<DynamicImage> {
    use libheif_rs::{ColorSpace, HeifContext, LibHeif, RgbChroma};

    let ctx = HeifContext::read_from_file(&path.to_string_lossy())
        .with_context(|| format!("failed to open HEIC image {}", path.display()))?;
    let handle = ctx
        .primary_image_handle()
        .with_context(|| format!("no primary image in {}", path.display()))?;
    let image = LibHeif::new()
        .decode(&handle, ColorSpace::Rgb(RgbChroma::Rgb), None)
        .with_context(|| format!("failed to decode HEIC image {}", path.display()))?;

    let planes = image.planes();
    let plane = planes
        .interleaved
        .context("HEIC decode returned no interleaved RGB plane")?;
    let width = plane.width as usize;
    let height = plane.height as usize;
    let stride = plane.stride;

    // libheif's row stride is padded and generally exceeds width*3, so copy row by row
    // into a tightly packed RGB buffer.
    let mut buf = Vec::with_capacity(width * height * 3);
    for row in 0..height {
        let start = row * stride;
        buf.extend_from_slice(&plane.data[start..start + width * 3]);
    }

    let rgb = RgbImage::from_raw(width as u32, height as u32, buf)
        .context("HEIC decode produced a mismatched buffer")?;
    Ok(DynamicImage::ImageRgb8(rgb))
}

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

    // Opt-in while the output difference is being judged: libjpeg-turbo decodes the
    // fixture corpus about 1.23x faster than the default path, but the two disagree by up
    // to 5/255 on a channel, and those pixels reach exported crops. Speed alone does not
    // decide that, so it is off unless asked for.
    if jpeg_turbo_enabled()
        && path_ref
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("jpg") || e.eq_ignore_ascii_case("jpeg"))
        && let Some(image) = decode_jpeg_turbo(path_ref)
    {
        return Ok(image);
    }

    #[cfg(feature = "raw")]
    if path_ref
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(is_raw_extension)
    {
        return load_raw_image(path_ref);
    }

    #[cfg(feature = "heic")]
    if path_ref
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(is_heic_extension)
    {
        return load_heic_image(path_ref);
    }

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
/// Whether to decode JPEGs with libjpeg-turbo instead of the `image` crate's decoder.
///
/// A build without NASM compiles libjpeg-turbo's scalar fallback and is *slower* than the
/// default path, so this is not something to turn on blind -- see CONTRIBUTING.md.
fn jpeg_turbo_enabled() -> bool {
    std::env::var_os("FCS_JPEG_TURBO").is_some_and(|v| v != "0")
}

/// Decode a JPEG with libjpeg-turbo, or `None` to fall back to the ordinary path.
///
/// Returns `None` rather than an error for anything unusual -- a CMYK or 16-bit file, a
/// truncated one, a `.jpg` that is not a JPEG at all -- because the caller has a decoder
/// that handles more formats than this one does and should simply use it. Only the common
/// case is taken here.
///
/// Orientation still comes from `image`'s EXIF parsing: building the decoder reads headers
/// and not pixels, so asking it for the orientation costs nothing and avoids a second EXIF
/// implementation disagreeing with the first about which way a photo goes.
fn decode_jpeg_turbo(path: &Path) -> Option<DynamicImage> {
    let bytes = std::fs::read(path).ok()?;

    let mut decoder = ImageReader::new(std::io::Cursor::new(&bytes))
        .with_guessed_format()
        .ok()?
        .into_decoder()
        .ok()?;
    let orientation = decoder.orientation().unwrap_or(Orientation::NoTransforms);

    let decompress = mozjpeg::Decompress::new_mem(&bytes).ok()?;
    let mut started = decompress.rgb().ok()?;
    let (width, height) = (started.width(), started.height());
    let pixels: Vec<u8> = started.read_scanlines().ok()?;
    started.finish().ok()?;

    let buffer = RgbImage::from_raw(
        u32::try_from(width).ok()?,
        u32::try_from(height).ok()?,
        pixels,
    )?;
    let mut image = DynamicImage::ImageRgb8(buffer);
    image.apply_orientation(orientation);
    Some(image)
}

pub fn load_image_raw<P: AsRef<Path>>(path: P) -> Result<DynamicImage> {
    let path_ref = path.as_ref();

    #[cfg(feature = "raw")]
    if path_ref
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(is_raw_extension)
    {
        return load_raw_image(path_ref);
    }

    #[cfg(feature = "heic")]
    if path_ref
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(is_heic_extension)
    {
        return load_heic_image(path_ref);
    }

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
    if let Some(fast) = resize_image_fast(image, width, height, fir_alg(filter)) {
        return fast;
    }
    image.resize_exact(width, height, filter).to_rgb8()
}

/// Map an `image` filter to the SIMD equivalent in `fast_image_resize`.
///
/// `image::imageops::resize` samples pixel-by-pixel through `GenericImageView`, which profiling
/// showed to be ~60% of the whole `Quality` detection pipeline. The kernels below are the same
/// filters, so output is equivalent up to rounding.
fn fir_alg(filter: FilterType) -> fir::ResizeAlg {
    // Experiment 51: FCS_RESIZE_ALG swaps the algorithm used for the `Quality` filter so a
    // candidate can be evaluated against production without a second build. Candidates are
    // quality-changing and none is adopted; `examples/resize_quality.rs` is what decides.
    if filter == FilterType::Triangle
        && let Some(alg) = std::env::var_os("FCS_RESIZE_ALG")
    {
        let alg = alg.to_string_lossy();
        if let Some(multiplicity) = alg.strip_prefix("super") {
            let m: u8 = multiplicity.parse().unwrap_or(2);
            return fir::ResizeAlg::SuperSampling(fir::FilterType::Bilinear, m);
        }
        if alg == "interp" {
            return fir::ResizeAlg::Interpolation(fir::FilterType::Bilinear);
        }
    }
    match filter {
        FilterType::Nearest => fir::ResizeAlg::Nearest,
        FilterType::Triangle => fir::ResizeAlg::Convolution(fir::FilterType::Bilinear),
        FilterType::CatmullRom => fir::ResizeAlg::Convolution(fir::FilterType::CatmullRom),
        FilterType::Gaussian => fir::ResizeAlg::Convolution(fir::FilterType::Gaussian),
        FilterType::Lanczos3 => fir::ResizeAlg::Convolution(fir::FilterType::Lanczos3),
    }
}

fn resize_image_fast(
    image: &DynamicImage,
    width: u32,
    height: u32,
    alg: fir::ResizeAlg,
) -> Option<RgbImage> {
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
    let options = fir::ResizeOptions::new().resize_alg(alg);
    let threaded = threading_pays(alg, rgb.width(), rgb.height());
    let run = |resizer: &mut fir::Resizer| {
        if threaded {
            resizer.resize(&src, &mut dst, &options)
        } else if let Some(pool) = single_thread_pool() {
            pool.install(|| resizer.resize(&src, &mut dst, &options))
        } else {
            resizer.resize(&src, &mut dst, &options)
        }
    };
    RESIZER.with_borrow_mut(run).ok()?;

    RgbImage::from_raw(width, height, dst.into_vec())
}

thread_local! {
    /// One `Resizer` per thread, kept alive between calls for its scratch buffer.
    ///
    /// A separable convolution writes an intermediate image between the horizontal and
    /// vertical passes, and `fast_image_resize` holds that buffer inside the `Resizer`,
    /// growing it on demand and zeroing the new part. Building a `Resizer` per call threw
    /// the buffer away every time, so every resize re-zeroed it: for a 10 MP source the
    /// intermediate is 640x4240x3 = 8.1 MB, and profiling warm detection put that `memset`
    /// at 4.1% of all CPU time -- more than the decode of the model outputs.
    ///
    /// Thread-local rather than shared: `resize` needs `&mut`, and a mutex here would
    /// serialise the batch path. The cost is one retained buffer per thread that has ever
    /// resized, sized to the largest source that thread has seen.
    static RESIZER: std::cell::RefCell<fir::Resizer> =
        std::cell::RefCell::new(fir::Resizer::new());
}

/// Source pixels below which threading the resize costs more than it saves.
///
/// Measured on this workstation (32 rayon threads) by alternating a one-thread pool against
/// the default pool in one process, 640x640 output, bilinear: 0.6 MP 0.79x, 1.1 MP 0.85x,
/// 2.5 MP 0.80-0.89x, 5.5 MP 1.27x, 10.1 MP 1.46-1.58x, 22.1 MP 1.16-1.26x. Fork and join
/// cost a roughly fixed 0.10-0.17 ms, which is most of a small resize and a fraction of a
/// large one, so the crossover sits near 4 MP.
///
/// ponytail: one constant for every machine. It is a function of core count and memory
/// bandwidth, so a smaller or larger box has a different crossover; `examples/resize_threading.rs`
/// in fcs-core re-measures it if this ever looks wrong.
const RESIZE_THREADING_MIN_PIXELS: u32 = 4_000_000;

/// Whether to let `fast_image_resize` spread this resize across the rayon pool.
///
/// The crate reads its thread count from `rayon::current_num_threads()` and has no per-call
/// switch, so the only way to say no is to run it inside a one-thread pool.
fn threading_pays(alg: fir::ResizeAlg, width: u32, height: u32) -> bool {
    // Nearest just gathers one source pixel per output pixel: 0.17-0.22 ms even for a 22 MP
    // source, which is less than the cost of handing it to other threads. Measured 0.54-0.73x
    // at every size.
    if matches!(alg, fir::ResizeAlg::Nearest) {
        return false;
    }
    width.saturating_mul(height) >= RESIZE_THREADING_MIN_PIXELS
}

/// A one-thread rayon pool, built once, used to hold `fast_image_resize` to a single core.
///
/// Returns `None` if the pool cannot be built, which sends the caller down the threaded path
/// rather than failing the resize -- slower than intended is better than no image.
fn single_thread_pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: std::sync::OnceLock<Option<rayon::ThreadPool>> = std::sync::OnceLock::new();
    POOL.get_or_init(|| rayon::ThreadPoolBuilder::new().num_threads(1).build().ok())
        .as_ref()
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
pub fn rgb_to_bgr_chw(image: &RgbImage) -> Vec<f32> {
    let (width, height) = image.dimensions();
    let w = width as usize;
    let h = height as usize;
    let channel_len = w * h;
    let row_stride = w * 3; // Keep as multiplication since 3 is not a power of 2
    let pixels = image.as_raw();

    let total = 3 * channel_len;
    // Deliberately uninitialised. The loop below writes every element, so `vec![0.0; total]`
    // would zero 4.9 MB per 640x640 detection for nothing; profiling put that `memset` at
    // 3.1% of all CPU time in the warm GPU path, and removing it measured 0.09-0.38 ms off
    // preprocessing (experiment 49).
    let mut data: Vec<f32> = Vec::with_capacity(total);
    {
        let spare = &mut data.spare_capacity_mut()[..total];
        let (b_slice, rest) = spare.split_at_mut(channel_len);
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
                    b_row[x].write(f32::from(src_row[src + 2]));
                    g_row[x].write(f32::from(src_row[src + 1]));
                    r_row[x].write(f32::from(src_row[src]));
                }
            });
    }
    // SAFETY: the three planes above partition all `total` elements, `par_chunks_mut(w)`
    // covers each plane's `h` rows exactly, and the inner loop writes all `w` of every row,
    // so every element is initialised. `f32` has no destructor, so an unwind out of the
    // loop leaves a length-0 `Vec` with nothing to drop.
    unsafe { data.set_len(total) };
    data
}

/// Convert any dynamic image into a BGR CHW array by first converting to RGB.
///
/// # Arguments
///
/// * `image` - The dynamic image to convert.
pub fn dynamic_to_bgr_chw(image: &DynamicImage) -> Vec<f32> {
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

    /// Every element of the output must be written, on an awkward size.
    ///
    /// `rgb_to_bgr_chw` fills a `Vec` it allocated uninitialised and then claims the whole
    /// length, so "the loop covers every index" is a safety requirement, not a nicety. A
    /// spot check of three elements cannot see a gap; this compares all of them against an
    /// independently computed reference, at a width and height that share no convenient
    /// factor with any chunking the implementation might use.
    #[test]
    fn rgb_to_bgr_chw_writes_every_element() {
        let (w, h) = (37u32, 23u32);
        let image = RgbImage::from_fn(w, h, |x, y| {
            image::Rgb([
                ((x * 7 + y) % 256) as u8,
                ((y * 13 + x * 3) % 256) as u8,
                ((x * y + 11) % 256) as u8,
            ])
        });

        let out = rgb_to_bgr_chw(&image);
        assert_eq!(out.len() as u32, 3 * w * h);

        let plane = (w * h) as usize;
        for y in 0..h {
            for x in 0..w {
                let px = image.get_pixel(x, y).0;
                let cell = (y * w + x) as usize;
                // BGR channel order, one plane per channel.
                for (channel, source) in [px[2], px[1], px[0]].into_iter().enumerate() {
                    let got = out[channel * plane + cell];
                    assert_eq!(
                        got,
                        f32::from(source),
                        "channel {channel} at ({x}, {y}) was not written correctly"
                    );
                }
            }
        }
    }

    #[test]
    fn rgb_to_bgr_chw_converts_correctly() {
        let mut image = RgbImage::new(2, 2);
        image.put_pixel(0, 0, image::Rgb([0, 128, 255]));
        image.put_pixel(1, 0, image::Rgb([255, 128, 0]));
        image.put_pixel(0, 1, image::Rgb([64, 64, 64]));
        image.put_pixel(1, 1, image::Rgb([255, 255, 255]));

        let array = rgb_to_bgr_chw(&image);
        assert_eq!(array.len(), 3 * 2 * 2);

        // Flat CHW layout: index = channel * h * w + y * w + x
        assert_eq!(array[0], 255.0); // B at (0, 0)
        assert_eq!(array[2 * 4], 0.0); // R at (0, 0)
        assert_eq!(array[4 + 1], 128.0); // G at (1, 0)
    }

    /// Every pixel of a multi-row image, not just the first row.
    ///
    /// The row offset is `y * row_stride`, which is 0 for y = 0 whatever the
    /// operator — so assertions confined to the top row cannot tell a multiply
    /// from a divide, and a mutant that made every row read row 0 survived.
    #[test]
    fn rgb_to_bgr_chw_reads_each_row_from_its_own_offset() {
        let (w, h) = (3u32, 4u32);
        let mut image = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                // Distinct per pixel and per channel, so a row mixup cannot
                // coincidentally produce the expected value.
                let base = (y * w + x) as u8;
                image.put_pixel(x, y, image::Rgb([base, base + 50, base + 100]));
            }
        }

        let array = rgb_to_bgr_chw(&image);
        let channel_len = (w * h) as usize;
        assert_eq!(array.len(), 3 * channel_len);

        for y in 0..h {
            for x in 0..w {
                let base = (y * w + x) as f32;
                let flat = (y * w + x) as usize;
                assert_eq!(array[flat], base + 100.0, "B at ({x}, {y})");
                assert_eq!(array[channel_len + flat], base + 50.0, "G at ({x}, {y})");
                assert_eq!(array[2 * channel_len + flat], base, "R at ({x}, {y})");
            }
        }
    }

    #[test]
    fn is_supported_image_path_rejects_other_extensions_and_bare_names() {
        // The function had only positive cases, so replacing it with `true`
        // wholesale went unnoticed — which would make the app try to decode
        // every file in a directory.
        for name in [
            "notes.txt",
            "video.mp4",
            "archive.zip",
            "model.onnx",
            "settings.json",
        ] {
            assert!(
                !is_supported_image_path(Path::new(name)),
                "{name} must not be treated as an image"
            );
        }

        // No extension at all, and a dotfile whose name is not an extension.
        assert!(!is_supported_image_path(Path::new("README")));
        assert!(!is_supported_image_path(Path::new("archive.tar.gz")));

        // A representative positive case, so the test fails if the list breaks
        // rather than passing because everything returns false.
        assert!(is_supported_image_path(Path::new("photo.png")));
    }

    /// Threading the resize must not change a single pixel.
    ///
    /// `resize_image` hands large sources to the rayon pool and keeps small ones on one
    /// thread, so the same image resized either way has to come out identical -- otherwise
    /// detections would depend on how many cores the machine has.
    #[test]
    fn threaded_and_single_threaded_resize_agree() {
        // Over RESIZE_THREADING_MIN_PIXELS, so the default path threads it.
        let (w, h) = (2400u32, 1800u32);
        assert!(
            w * h >= RESIZE_THREADING_MIN_PIXELS,
            "source must cross the gate"
        );
        let source = RgbImage::from_fn(w, h, |x, y| {
            image::Rgb([(x % 251) as u8, (y % 241) as u8, ((x ^ y) % 233) as u8])
        });
        let dynamic = DynamicImage::ImageRgb8(source);

        let single = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("single-thread pool");

        for filter in [
            FilterType::Triangle,
            FilterType::Lanczos3,
            FilterType::Nearest,
        ] {
            let threaded = resize_image(&dynamic, 640, 640, filter);
            let serial = single.install(|| resize_image(&dynamic, 640, 640, filter));
            assert_eq!(
                threaded.as_raw(),
                serial.as_raw(),
                "{filter:?} resize differs between one thread and many"
            );
        }
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
        // Triangle now maps onto the fast (SIMD) bilinear path rather than image's sampler.
        let mut image = ImageBuffer::<Rgb<u8>, _>::new(4, 4);
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            *pixel = Rgb([(x * 40) as u8, (y * 40) as u8, 128]);
        }
        let dynamic = DynamicImage::ImageRgb8(image);
        let result = super::resize_image(&dynamic, 2, 2, FilterType::Triangle);
        assert_eq!(result.dimensions(), (2, 2));
    }

    #[test]
    fn avif_extension_is_supported_case_insensitively() {
        // Unconditional, unlike the raw/heic cases below — no feature gate.
        assert!(is_supported_image_path(Path::new("portrait.avif")));
        assert!(is_supported_image_path(Path::new("PORTRAIT.AVIF")));
    }

    #[test]
    fn avif_round_trips_through_save_and_load() {
        // AVIF output was already wired up while the extension was missing from
        // SUPPORTED_IMAGE_EXTENSIONS, so the app could write a file it then
        // refused to open. This asserts both halves work, and doubles as the
        // check that dav1d is actually linked for decode — without it the
        // `avif-native` feature is dead weight on all three platforms.
        use crate::output::{
            ImageFormatHint, MetadataContext, OutputOptions, PngCompression, save_dynamic_image,
        };
        use image::GenericImageView as _;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("crop.avif");

        let mut src = image::RgbImage::new(16, 16);
        for (x, y, px) in src.enumerate_pixels_mut() {
            *px = image::Rgb([(x * 16) as u8, (y * 16) as u8, 128]);
        }
        let src = DynamicImage::ImageRgb8(src);

        let options = OutputOptions {
            format: Some(ImageFormatHint::Avif),
            auto_detect: true,
            jpeg_quality: 90,
            png_compression: PngCompression::Default,
            webp_quality: 90,
            metadata: Default::default(),
        };
        save_dynamic_image(&src, &path, &options, &MetadataContext::default())
            .expect("AVIF encode");

        let loaded = super::load_image(&path).expect("AVIF decode");
        assert_eq!(loaded.dimensions(), (16, 16));
    }

    #[cfg(feature = "raw")]
    #[test]
    fn raw_extensions_are_supported_and_classified() {
        assert!(is_supported_image_path(Path::new("photo.DNG")));
        assert!(is_supported_image_path(Path::new("shot.cr2")));
        assert!(is_raw_extension("NeF"));
        assert!(!is_raw_extension("png"));
    }

    #[cfg(feature = "raw")]
    #[test]
    fn load_image_rejects_malformed_raw_without_panicking() {
        let dir = tempfile::tempdir().unwrap();
        let bogus = dir.path().join("not_really.dng");
        std::fs::write(&bogus, b"this is not a raw file").unwrap();
        // Must route to the RAW decoder and return an Err, not panic.
        assert!(super::load_image(&bogus).is_err());
    }

    #[cfg(feature = "heic")]
    #[test]
    fn heic_extensions_are_supported_and_classified() {
        assert!(is_supported_image_path(Path::new("photo.HEIC")));
        assert!(is_supported_image_path(Path::new("shot.heif")));
        assert!(is_heic_extension("Hif"));
        assert!(!is_heic_extension("png"));
    }

    #[cfg(feature = "heic")]
    #[test]
    fn load_image_rejects_malformed_heic_without_panicking() {
        let dir = tempfile::tempdir().unwrap();
        let bogus = dir.path().join("not_really.heic");
        std::fs::write(&bogus, b"this is not a heic file").unwrap();
        // Must route to the HEIC decoder and return an Err, not panic.
        assert!(super::load_image(&bogus).is_err());
    }

    /// Opt-in real-decode check: set `FCS_HEIC_SAMPLE` to a real HEIC file path.
    /// Skips when unset so CI without a sample asset stays green (matches the RAW pattern).
    #[cfg(feature = "heic")]
    #[test]
    fn load_image_decodes_real_heic_when_sample_provided() {
        let Ok(sample) = std::env::var("FCS_HEIC_SAMPLE") else {
            eprintln!("skipping: FCS_HEIC_SAMPLE not set");
            return;
        };
        let img = super::load_image(&sample).expect("real HEIC sample should decode");
        assert!(img.width() > 0 && img.height() > 0);
    }

    /// Opt-in real-decode check: set `FCS_RAW_SAMPLE` to a real camera RAW file path.
    /// Skips when unset so CI without a sample asset stays green (matches the fixture pattern).
    #[cfg(feature = "raw")]
    #[test]
    fn load_image_decodes_real_raw_when_sample_provided() {
        let Ok(sample) = std::env::var("FCS_RAW_SAMPLE") else {
            eprintln!("skipping: FCS_RAW_SAMPLE not set");
            return;
        };
        let img = super::load_image(&sample).expect("real RAW sample should decode");
        assert!(img.width() > 0 && img.height() > 0);
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

    fn rgb_to_bgr_chw_baseline(image: &RgbImage) -> Vec<f32> {
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

        data
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
