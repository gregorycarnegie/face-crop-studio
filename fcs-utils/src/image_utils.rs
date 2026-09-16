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
    DynamicImage, ImageDecoder, ImageReader, RgbImage, RgbaImage, imageops::FilterType,
    metadata::Orientation,
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
    let buf = pack_rows(plane.data, width, height, plane.stride)
        .context("HEIC plane is shorter than its stride and size claim")?;

    let rgb = RgbImage::from_raw(width as u32, height as u32, buf)
        .context("HEIC decode produced a mismatched buffer")?;
    Ok(DynamicImage::ImageRgb8(rgb))
}

/// Copy `height` rows of `width` RGB pixels out of a buffer whose rows are padded to `stride` bytes.
///
/// libheif pads each row, so only the first `width * 3` bytes of every `stride` are pixels. `None`
/// when the buffer is shorter than that claim, which a crafted file can arrange: slicing it
/// unchecked panicked.
#[cfg(any(feature = "heic", test))]
fn pack_rows(data: &[u8], width: usize, height: usize, stride: usize) -> Option<Vec<u8>> {
    let row_bytes = width.checked_mul(3)?;
    (0..height)
        .map(|row| {
            let start = row.checked_mul(stride)?;
            data.get(start..start.checked_add(row_bytes)?)
        })
        .collect::<Option<Vec<_>>>()
        .map(|rows| rows.concat())
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

    // libjpeg-turbo decodes the fixture corpus about 1.23x faster, and decode is the
    // largest cost in processing a folder. It disagrees with the `image` decoder by up to
    // 5/255 on a channel -- the JPEG standard leaves IDCT precision open and both are
    // valid -- which moves detection boxes by about a pixel. That difference was reviewed
    // against real crops before this became the default.
    if path_ref
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("jpg") || e.eq_ignore_ascii_case("jpeg"))
        && let Some(image) = decode_jpeg_turbo(path_ref, true)
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
/// Decode a JPEG with libjpeg-turbo, or `None` to fall back to the ordinary path.
///
/// Needs NASM at build time. Without it `mozjpeg-sys` quietly compiles libjpeg-turbo's
/// scalar C fallback, which is *slower* than the decoder this replaces -- see
/// CONTRIBUTING.md. The release workflow installs NASM and fails if it is missing.
///
/// Returns `None` rather than an error for anything unusual -- a CMYK or 16-bit file, a
/// truncated one, a `.jpg` that is not a JPEG at all -- because the caller has a decoder
/// that handles more formats than this one does and should simply use it. Only the common
/// case is taken here.
///
/// With `apply_exif_orientation`, orientation comes from `image`'s EXIF parsing: building
/// the decoder reads headers and not pixels, so asking it costs nothing and avoids a second
/// EXIF implementation disagreeing with the first about which way a photo goes. Without it,
/// the pixels are returned as stored -- matching [`load_image_raw`].
fn decode_jpeg_turbo(path: &Path, apply_exif_orientation: bool) -> Option<DynamicImage> {
    let bytes = std::fs::read(path).ok()?;

    let orientation = if apply_exif_orientation {
        let mut decoder = ImageReader::new(std::io::Cursor::new(&bytes))
            .with_guessed_format()
            .ok()?
            .into_decoder()
            .ok()?;
        decoder.orientation().unwrap_or(Orientation::NoTransforms)
    } else {
        Orientation::NoTransforms
    };

    let decompress = mozjpeg::Decompress::new_mem(&bytes).ok()?;

    // libjpeg has no CMYK/YCCK -> RGB conversion, so `rgb()` below trips its fatal error
    // handler, and mozjpeg installs that handler as `extern "C-unwind"`: it unwinds out of C
    // instead of returning an error, so the panic escapes this function's `Option` and takes
    // the process down with it -- no message, no output file. One such image killed a batch
    // of 12,416 after 12,387 of them had been processed. The `image` crate decodes these, so
    // leave them to the fallback below.
    if matches!(
        decompress.color_space(),
        mozjpeg::ColorSpace::JCS_CMYK | mozjpeg::ColorSpace::JCS_YCCK
    ) {
        return None;
    }

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

/// Decode an image without applying the normal EXIF-orientation correction.
///
/// Unlike [`load_image`], ordinary image formats retain their stored orientation.
/// This does not mean camera RAW specifically: camera RAW and HEIC files use
/// their dedicated decoders when the `raw` and `heic` features are enabled.
/// Returns an error if the file cannot be opened, identified, or decoded.
pub fn load_image_raw<P: AsRef<Path>>(path: P) -> Result<DynamicImage> {
    let path_ref = path.as_ref();

    // Here as well as in `load_image`: the GUI picks between the two loaders on the
    // auto-orient setting, so covering only one would decode a detection with one decoder
    // and its export with the other inside a single run.
    if path_ref
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("jpg") || e.eq_ignore_ascii_case("jpeg"))
        && let Some(image) = decode_jpeg_turbo(path_ref, false)
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
    let experiment = std::env::var_os("FCS_RESIZE_ALG");
    fir_alg_with(
        filter,
        experiment
            .as_deref()
            .map(|v| v.to_string_lossy())
            .as_deref(),
    )
}

/// [`fir_alg`] with the experiment variable passed in, so it can be tested without setting a
/// process-wide environment variable while other tests resize in parallel.
fn fir_alg_with(filter: FilterType, experiment: Option<&str>) -> fir::ResizeAlg {
    // Experiment 51: FCS_RESIZE_ALG swaps the algorithm used for the `Quality` filter so a
    // candidate can be evaluated against production without a second build. Candidates are
    // quality-changing and none is adopted; `examples/resize_quality.rs` is what decides.
    if filter == FilterType::Triangle
        && let Some(alg) = experiment
    {
        if let Some(multiplicity) = alg.strip_prefix("super") {
            let m: u8 = multiplicity.parse().unwrap_or(2);
            return fir::ResizeAlg::SuperSampling(fir::FilterType::Bilinear, m);
        }
        if alg == "interp" {
            return fir::ResizeAlg::Interpolation(fir::FilterType::Bilinear);
        }
        // The `Speed` setting the application already exposes, reachable from the same
        // harness so experiment 54 could measure what that setting costs a detection.
        if alg == "nearest" {
            return fir::ResizeAlg::Nearest;
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

    let out = resize_pixels_fast(
        rgb.as_raw(),
        rgb.width(),
        rgb.height(),
        fir::PixelType::U8x3,
        width,
        height,
        alg,
    )?;
    RgbImage::from_raw(width, height, out)
}

/// The RGBA counterpart of [`resize_image`], for callers that carry an alpha channel.
///
/// `crop_face_from_image` resizes an RGBA canvas -- the fill colour has an alpha, and rotation
/// needs somewhere to put the corners -- so it could not use the RGB path above and was still
/// going through `image::imageops::resize`. A batch profile put that one call at 10.9% of all
/// CPU over a 1239-image folder, the largest single cost left once the GPU cropper had gone
/// (experiment 88).
///
/// Alpha is convolved as a plain fourth channel (`use_alpha(false)`), which is what
/// `image::imageops::resize` does, so this is a speed change rather than a quality one.
/// Premultiplying would handle a transparent fill colour better -- colour would stop bleeding
/// out of fully transparent pixels -- but that is different output and belongs in its own
/// experiment.
///
/// Returns `None` if `fast_image_resize` rejects the request, leaving the caller to fall back.
pub fn resize_rgba_fast(
    image: &RgbaImage,
    width: u32,
    height: u32,
    filter: FilterType,
) -> Option<RgbaImage> {
    let out = resize_pixels_fast(
        image.as_raw(),
        image.width(),
        image.height(),
        fir::PixelType::U8x4,
        width,
        height,
        fir_alg(filter),
    )?;
    RgbaImage::from_raw(width, height, out)
}

/// Run one `fast_image_resize` convolution over raw interleaved bytes.
fn resize_pixels_fast(
    src_bytes: &[u8],
    src_width: u32,
    src_height: u32,
    pixel_type: fir::PixelType,
    width: u32,
    height: u32,
    alg: fir::ResizeAlg,
) -> Option<Vec<u8>> {
    let src = FirImageRef::new(src_width, src_height, src_bytes, pixel_type).ok()?;

    let mut dst = FirImage::new(width, height, pixel_type);
    // `use_alpha` defaults on and would premultiply; see `resize_rgba_fast`. It is ignored for
    // pixel types without an alpha channel, so setting it here is safe for both callers.
    let options = fir::ResizeOptions::new().resize_alg(alg).use_alpha(false);
    let threaded = threading_pays(alg, src_width, src_height);

    // Taken out of the thread-local for the duration and put back after, rather than held
    // borrowed across the resize. `resize` spreads itself over rayon, and in batch work
    // rayon is already busy with other images, so a thread blocked in here steals another
    // image's task and re-enters this function *on the same thread*. Holding a `RefCell`
    // borrow across that call panicked with "RefCell already borrowed" -- only ever under
    // nested parallelism, so single images and the test suite never saw it.
    //
    // A re-entrant call finds `None` and builds its own `Resizer`; the buffer reuse this
    // cache exists for is lost for that call only, and correctness does not depend on it.
    let mut resizer = RESIZER
        .with(|slot| slot.borrow_mut().take())
        .unwrap_or_default();

    let result = if threaded {
        resizer.resize(&src, &mut dst, &options)
    } else if let Some(pool) = single_thread_pool() {
        pool.install(|| resizer.resize(&src, &mut dst, &options))
    } else {
        resizer.resize(&src, &mut dst, &options)
    };

    RESIZER.with(|slot| *slot.borrow_mut() = Some(resizer));
    result.ok()?;

    Some(dst.into_vec())
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
    ///
    /// `Option` so a caller can take it out for the length of the resize instead of holding
    /// a borrow across it -- see `resize_image_fast` for why that distinction is load-bearing.
    static RESIZER: std::cell::RefCell<Option<fir::Resizer>> =
        const { std::cell::RefCell::new(None) };
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
    // Inside a rayon worker, always. Splitting each resize again within a batch that is
    // already parallel across images looks like oversubscription, and twice now it has
    // measured better anyway:
    //
    //  - Experiment 63: threading throughout ran 14.7-17.1 s against 19.2-24.4 s gated over
    //    four order-alternated pairs of a 1239-image folder at `Quality`. Rayon's work
    //    stealing absorbs the nesting, and the gate leaves cores idle on uneven image sizes
    //    and at the tail of the batch.
    //  - Experiment 88: saying no is not free. It means `single_thread_pool().install()`,
    //    and from a worker that is a cross-registry hop (`Registry::in_worker_cross`, 17.9%
    //    of all CPU in a batch profile) rather than an ordinary join. Yielding here took the
    //    same folder from a median 9.77 s to 8.03 s, ~18%, winning every pair in both orders.
    if rayon::current_thread_index().is_some() {
        return true;
    }
    // Off a worker the hop is cheap and the size gate decides, which is the only place it was
    // ever measured: one image at a time, threading a sub-4 MP resize really is 0.79-0.89x.
    width.saturating_mul(height) >= RESIZE_THREADING_MIN_PIXELS
}

/// A one-thread rayon pool, built once, used to hold `fast_image_resize` to a single core.
///
/// Returns `None` if the pool cannot be built, which sends the caller down the threaded path
/// rather than failing the resize -- slower than intended is better than no image.
///
/// **One pool for the whole process, so concurrent non-worker callers serialise on it.**
/// `threading_pays` returns true on a rayon worker, so nothing in this application reaches
/// here concurrently: the CLI folder job and watch mode are `par_iter`, the GUI's detection
/// entry points are `rayon::spawn`, and GUI batch export is `pool.install`. A caller that
/// detects from plain threads would be a different matter -- experiment 24 profiled exactly
/// that and found eight threads' resizes running one after another on this pool's single
/// core, one thread holding 31.5% of all CPU in the run, and per-process throughput capped
/// at about half what it reaches through rayon. If a plain-threaded caller ever appears,
/// this is what it will hit, and confining per calling thread rather than per process is
/// the fix.
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

/// The same conversion, writing the image into a larger canvas and padding the rest.
///
/// The letterboxed counterpart of [`rgb_to_bgr_chw`]: `image` is the already-resized drawn
/// region, `fit` says where it sits inside the model input. Padding is written here rather
/// than by clearing the buffer first, which keeps the "every element is written exactly
/// once" property the uninitialised allocation depends on.
///
/// Kept bit-identical to what `rgb_to_chw.wgsl` produces for the same inputs -- each output
/// float is either an exact source byte or the pad value, with no arithmetic in between --
/// because the CPU and GPU preprocessors are compared against each other by test.
pub fn rgb_to_bgr_chw_letterboxed(
    image: &RgbImage,
    target: (u32, u32),
    fit: &InputFit,
) -> Vec<f32> {
    let (target_w, target_h) = (target.0 as usize, target.1 as usize);
    let (drawn_w, drawn_h) = (fit.drawn.0 as usize, fit.drawn.1 as usize);
    let (origin_x, origin_y) = (fit.origin.0 as usize, fit.origin.1 as usize);
    debug_assert_eq!(
        (image.width() as usize, image.height() as usize),
        (drawn_w, drawn_h),
        "the drawn region must already be resized to the fit"
    );

    let channel_len = target_w * target_h;
    let row_stride = drawn_w * 3;
    let pixels = image.as_raw();
    let total = 3 * channel_len;

    let mut data: Vec<f32> = Vec::with_capacity(total);
    {
        let spare = &mut data.spare_capacity_mut()[..total];
        let (b_slice, rest) = spare.split_at_mut(channel_len);
        let (g_slice, r_slice) = rest.split_at_mut(channel_len);

        b_slice
            .par_chunks_mut(target_w)
            .zip(g_slice.par_chunks_mut(target_w))
            .zip(r_slice.par_chunks_mut(target_w))
            .enumerate()
            .for_each(|(y, ((b_row, g_row), r_row))| {
                let src_y = y.wrapping_sub(origin_y);
                if src_y >= drawn_h {
                    for x in 0..target_w {
                        b_row[x].write(LETTERBOX_PAD);
                        g_row[x].write(LETTERBOX_PAD);
                        r_row[x].write(LETTERBOX_PAD);
                    }
                    return;
                }
                let src_row = &pixels[src_y * row_stride..(src_y + 1) * row_stride];
                for x in 0..target_w {
                    let src_x = x.wrapping_sub(origin_x);
                    if src_x >= drawn_w {
                        b_row[x].write(LETTERBOX_PAD);
                        g_row[x].write(LETTERBOX_PAD);
                        r_row[x].write(LETTERBOX_PAD);
                        continue;
                    }
                    let src = (src_x << 1) + src_x;
                    b_row[x].write(f32::from(src_row[src + 2]));
                    g_row[x].write(f32::from(src_row[src + 1]));
                    r_row[x].write(f32::from(src_row[src]));
                }
            });
    }
    // SAFETY: as in `rgb_to_bgr_chw` -- the three planes partition all `total` elements,
    // every row is covered once, and each branch above writes all `target_w` of its row.
    unsafe { data.set_len(total) };
    data
}

/// The value the letterbox bars are filled with, on every path.
///
/// Black. Experiment 96 measured black against YOLO's 114 at four faces in 1239, so this is
/// a choice with no measured consequence rather than a tuned constant.
pub const LETTERBOX_PAD: f32 = 0.0;

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

/// How a source image is laid onto the model's fixed input.
///
/// One scale for both axes, centred, with the remainder padded -- letterboxing. The
/// preprocessor used to scale x and y independently, which showed the model a face squashed
/// in proportion to how far the source was from square: over 1239 images that cost 77
/// detections outright and moved every box on a non-square source (experiment 96).
///
/// `scale` is what maps a model coordinate back to a source pixel, and `origin` is what has
/// to come off it first. Both axes share one scale, so a detection cannot be distorted by
/// the mapping alone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InputFit {
    /// Source pixels per model pixel, the same on both axes.
    pub scale: f32,
    /// Size of the drawn region inside the model input, in model pixels.
    pub drawn: (u32, u32),
    /// Where the drawn region starts inside the model input, in model pixels.
    pub origin: (u32, u32),
}

impl InputFit {
    /// The padding offset expressed in source pixels.
    ///
    /// Postprocessing multiplies a model coordinate by `scale`; subtracting this from the
    /// result is the same as subtracting `origin` before the multiply, and does not need the
    /// decode to know about letterboxing at all.
    pub fn source_offset(&self) -> (f32, f32) {
        (
            self.origin.0 as f32 * self.scale,
            self.origin.1 as f32 * self.scale,
        )
    }
}

/// Fit a source image inside the model input without distorting it.
///
/// # Arguments
///
/// * `original` - A tuple of the original image's (width, height).
/// * `target` - A tuple of the model input's (width, height).
///
/// # Example
///
/// ```
/// use fcs_utils::fit_input;
///
/// # fn main() -> anyhow::Result<()> {
/// let fit = fit_input((1000, 500), (640, 640))?;
/// assert_eq!(fit.scale, 1.5625); // width is the tight axis
/// assert_eq!(fit.drawn, (640, 320));
/// assert_eq!(fit.origin, (0, 160)); // bars above and below
/// assert_eq!(fit.source_offset(), (0.0, 250.0));
///
/// assert!(fit_input((0, 500), (640, 640)).is_err());
/// # Ok(())
/// # }
/// ```
pub fn fit_input(original: (u32, u32), target: (u32, u32)) -> Result<InputFit> {
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
    // The larger of the two ratios: whichever axis is tightest decides, and the other one
    // gets bars. `max` rather than `min` because this is source pixels per model pixel.
    let scale = (orig_w as f32 / target_w as f32).max(orig_h as f32 / target_h as f32);
    let drawn = (
        ((orig_w as f32 / scale).round() as u32).clamp(1, target_w),
        ((orig_h as f32 / scale).round() as u32).clamp(1, target_h),
    );
    Ok(InputFit {
        scale,
        drawn,
        origin: ((target_w - drawn.0) / 2, (target_h - drawn.1) / 2),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    /// A four-component JPEG must not take the process down.
    ///
    /// `decode_jpeg_turbo` returns `Option` so an awkward file falls through to the `image`
    /// crate, but a CMYK/YCCK source made libjpeg's fatal error handler unwind through C
    /// before that could happen: the whole run died with no message and wrote no output.
    /// Without the colour-space check this test fails on that panic; in the CLI, where
    /// nothing catches it, the same panic ends the run and writes no output at all.
    #[test]
    fn cmyk_jpeg_falls_back_instead_of_crashing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("cmyk.jpg");
        let (width, height) = (16usize, 8usize);

        let mut encoder = mozjpeg::Compress::new(mozjpeg::ColorSpace::JCS_CMYK);
        encoder.set_size(width, height);
        encoder.set_quality(90.0);
        let mut started = encoder
            .start_compress(std::fs::File::create(&path).expect("create fixture"))
            .expect("start CMYK compress");
        started
            .write_scanlines(&vec![128u8; width * height * 4])
            .expect("write CMYK scanlines");
        started.finish().expect("finish CMYK fixture");

        let image = load_image(&path).expect("a CMYK JPEG must decode via the fallback");
        assert_eq!(
            (image.width(), image.height()),
            (width as u32, height as u32)
        );
    }

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

    /// Resizing from inside a rayon parallel iterator must not panic.
    ///
    /// `fast_image_resize` spreads a large resize over rayon, so a thread blocked in
    /// `resize` steals other queued work -- and in batch processing that work is another
    /// image's resize, re-entering `resize_image_fast` on the same thread. The `Resizer`
    /// cache originally held a `RefCell` borrow across the resize and panicked with
    /// "RefCell already borrowed" the moment two levels of parallelism met. A real folder
    /// of 1239 photos lost two thirds of its output to it while the whole test suite passed.
    ///
    /// The pool is deliberately small and the item count well above it: stealing only
    /// happens when a worker runs out of its own work while blocked inside `resize`, so a
    /// wide pool with one item per thread -- which is what an earlier version of this test
    /// did -- never triggers it and passes against the broken code.
    #[test]
    fn concurrent_resizes_do_not_re_enter_the_resizer_cache() {
        // Under RESIZE_THREADING_MIN_PIXELS on purpose. That is the branch handing the
        // resize to the one-thread pool, and `install` from a rayon worker lets that worker
        // pick up outer work while it waits -- which is how the re-entry happens. The real
        // failure was on small images for exactly this reason.
        let (w, h) = (1200u32, 800u32);
        assert!(
            w * h < RESIZE_THREADING_MIN_PIXELS,
            "must stay under the threading gate"
        );
        let sources: Vec<DynamicImage> = (0..64)
            .map(|i| {
                DynamicImage::ImageRgb8(RgbImage::from_fn(w, h, |x, y| {
                    image::Rgb([(x + i) as u8, (y + i) as u8, ((x ^ y) + i) as u8])
                }))
            })
            .collect();

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("build small pool");

        let sizes: Vec<(u32, u32)> = pool.install(|| {
            sources
                .par_iter()
                .map(|image| resize_image(image, 640, 640, FilterType::Triangle).dimensions())
                .collect()
        });

        assert_eq!(sizes.len(), sources.len());
        assert!(sizes.iter().all(|&d| d == (640, 640)));
    }

    /// Threading the resize must not change a single pixel.
    ///
    /// `resize_image` hands large sources to the rayon pool and keeps small ones on one
    /// thread, so the same image resized either way has to come out identical -- otherwise
    /// detections would depend on how many cores the machine has.
    #[test]
    fn threading_gate_yields_inside_a_rayon_worker() {
        // Off a worker the size gate decides, and a small source is held to one core.
        let alg = fir::ResizeAlg::Convolution(fir::FilterType::Lanczos3);
        assert!(
            !threading_pays(alg, 512, 512),
            "small source gated off-worker"
        );

        // On a worker it must not, because saying no there means a cross-registry
        // `install()` hop that costs more than the threading it avoids (experiment 88).
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("pool");
        assert!(
            pool.install(|| threading_pays(alg, 512, 512)),
            "small source must thread when a worker is already running it"
        );

        // Nearest is still refused everywhere: one source pixel per output pixel is
        // cheaper than handing the work to anyone.
        assert!(!pool.install(|| threading_pays(fir::ResizeAlg::Nearest, 512, 512)));
    }

    #[test]
    fn rgba_fast_resize_matches_imageops_within_rounding() {
        // Smooth, which is what a photograph looks like to a resampler. High-frequency noise
        // is the wrong fixture here: Lanczos3 has negative lobes, so two implementations
        // disagree far more at a hard edge without either being wrong -- across a real
        // 1239-image folder the worst crop still differed by 23 (experiment 88).
        //
        // Alpha runs against the colour ramp rather than with it, so a path that
        // premultiplied would divide colour by a different factor at each end and diverge
        // by much more than rounding.
        let source = image::RgbaImage::from_fn(600, 400, |x, y| {
            image::Rgba([
                (x / 3) as u8,
                (y / 2) as u8,
                ((x + y) / 4) as u8,
                (255 - x / 3) as u8,
            ])
        });

        let fast = resize_rgba_fast(&source, 128, 96, FilterType::Lanczos3).expect("fir resize");
        let reference = image::imageops::resize(&source, 128, 96, FilterType::Lanczos3);

        assert_eq!(fast.dimensions(), reference.dimensions());
        let worst = fast
            .as_raw()
            .iter()
            .zip(reference.as_raw())
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .expect("non-empty");
        assert!(worst <= 2, "max channel difference {worst} is not rounding");
    }

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
    fn a_square_target_letterboxes_the_wider_axis() {
        // 16:9 into a square: full width, bars top and bottom, one scale for both axes.
        let fit = fit_input((1920, 1080), (640, 640)).unwrap();
        assert_eq!(fit.drawn, (640, 360));
        assert_eq!(fit.origin, (0, 140));
        assert!((fit.scale - 3.0).abs() < 1e-6);
        // 1080 / 3.0 = 360, and the bars are (640 - 360) / 2 either side.
        assert_eq!(fit.origin.1 * 2 + fit.drawn.1, 640);

        // Portrait into a landscape target: bars left and right, so origin.x is the computed one.
        let fit = fit_input((1000, 2000), (640, 480)).unwrap();
        assert_eq!(fit.drawn, (240, 480));
        assert_eq!(fit.origin, (200, 0)); // `%` would give 0, and 640 / 240 / 2 would give 1
    }

    #[test]
    fn a_source_matching_the_target_aspect_gets_no_bars() {
        for source in [(640, 640), (1280, 1280), (320, 320)] {
            let fit = fit_input(source, (640, 640)).unwrap();
            assert_eq!(fit.drawn, (640, 640), "{source:?}");
            assert_eq!(fit.origin, (0, 0), "{source:?}");
        }
    }

    #[test]
    fn the_fit_maps_a_model_coordinate_back_to_the_source() {
        for source in [(1920u32, 1080u32), (1080, 1920), (4000, 3000), (640, 640)] {
            let fit = fit_input(source, (640, 640)).unwrap();
            let (ox, oy) = fit.source_offset();
            for corner in [(0.0f32, 0.0f32), (source.0 as f32, source.1 as f32)] {
                // Source -> model, the way every preprocessor lays the image down.
                let model_x = corner.0 / fit.scale + fit.origin.0 as f32;
                let model_y = corner.1 / fit.scale + fit.origin.1 as f32;
                // Model -> source, the way postprocessing reads it back.
                let back_x = model_x * fit.scale - ox;
                let back_y = model_y * fit.scale - oy;
                assert!(
                    (back_x - corner.0).abs() < 0.01 && (back_y - corner.1).abs() < 0.01,
                    "{source:?}: {corner:?} came back as ({back_x}, {back_y})"
                );
            }
        }
    }

    #[test]
    fn fit_input_rejects_zero() {
        assert!(fit_input((0, 480), (320, 240)).is_err());
        assert!(fit_input((640, 480), (0, 240)).is_err());
    }

    #[test]
    fn pack_rows_drops_the_stride_padding_and_rejects_a_short_plane() {
        // Two pixels (6 bytes) per row padded to a stride of 7: every seventh byte is padding.
        let data: Vec<u8> = (0..21).collect();
        let expected: Vec<u8> = [0..6, 7..13, 14..20].into_iter().flatten().collect();
        assert_eq!(pack_rows(&data, 2, 3, 7), Some(expected));
        assert_eq!(
            pack_rows(&data, 2, 4, 7),
            None,
            "a fourth row runs past the end"
        );
    }

    #[test]
    fn the_resize_experiment_only_overrides_the_quality_filter() {
        use fir::{FilterType as Fir, ResizeAlg as Alg};
        let alg = fir_alg_with;
        assert!(matches!(
            alg(FilterType::Triangle, None),
            Alg::Convolution(Fir::Bilinear)
        ));
        assert!(matches!(
            alg(FilterType::Triangle, Some("interp")),
            Alg::Interpolation(Fir::Bilinear)
        ));
        assert!(matches!(
            alg(FilterType::Triangle, Some("nearest")),
            Alg::Nearest
        ));
        assert!(matches!(
            alg(FilterType::Triangle, Some("super3")),
            Alg::SuperSampling(Fir::Bilinear, 3)
        ));
        assert!(matches!(
            alg(FilterType::Triangle, Some("unknown")),
            Alg::Convolution(Fir::Bilinear)
        ));
        // Triangle is the Quality filter; the experiment leaves every other filter alone.
        assert!(matches!(
            alg(FilterType::Nearest, Some("interp")),
            Alg::Nearest
        ));
        assert!(matches!(
            alg(FilterType::CatmullRom, Some("nearest")),
            Alg::Convolution(Fir::CatmullRom)
        ));
    }

    #[test]
    fn jpegs_decode_through_libjpeg_turbo_in_both_loaders() {
        // High-frequency content, where libjpeg-turbo's IDCT and the image crate's differ; the
        // assert_ne at the end proves this fixture can tell the two decoders apart at all.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("noise.jpg");
        let noise = RgbImage::from_fn(64, 48, |x, y| {
            image::Rgb([
                ((x * 37 + y * 91) % 256) as u8,
                ((x * 53 + y * 17) % 256) as u8,
                ((x * 11 + y * 29) % 256) as u8,
            ])
        });
        DynamicImage::ImageRgb8(noise)
            .save(&path)
            .expect("encode jpeg");

        let turbo = decode_jpeg_turbo(&path, false)
            .expect("libjpeg-turbo decodes a plain JPEG")
            .to_rgb8();
        assert_eq!(turbo.dimensions(), (64, 48));
        assert_eq!(load_image(&path).expect("load_image").to_rgb8(), turbo);
        assert_eq!(
            load_image_raw(&path).expect("load_image_raw").to_rgb8(),
            turbo
        );

        let image_crate = ImageReader::open(&path)
            .expect("open")
            .decode()
            .expect("decode")
            .to_rgb8();
        assert_ne!(
            turbo, image_crate,
            "the fixture must separate the two decoders"
        );
    }

    #[test]
    fn the_fast_resizer_and_its_single_thread_pool_are_available() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_fn(37, 23, |x, y| {
            image::Rgb([x as u8, y as u8, (x ^ y) as u8])
        }));
        let out = resize_image_fast(
            &image,
            11,
            7,
            fir::ResizeAlg::Convolution(fir::FilterType::Bilinear),
        )
        .expect("fast resize handles RGB8");
        assert_eq!(out.dimensions(), (11, 7));
        assert_eq!(single_thread_pool().expect("pool").current_num_threads(), 1);
    }

    #[test]
    fn letterboxed_conversion_pads_the_bars_and_keeps_the_pixels() {
        // A 16:9 source drawn into a square: rows outside the drawn band are pad, and rows
        // inside carry the exact source bytes with the channels swapped.
        let fit = fit_input((1920, 1080), (64, 64)).unwrap();
        assert_eq!(fit.drawn, (64, 36));
        let drawn = RgbImage::from_fn(fit.drawn.0, fit.drawn.1, |x, y| {
            image::Rgb([x as u8, y as u8, 200])
        });
        let data = rgb_to_bgr_chw_letterboxed(&drawn, (64, 64), &fit);
        assert_eq!(data.len(), 3 * 64 * 64);

        let plane = 64 * 64;
        let at = |x: usize, y: usize, c: usize| data[c * plane + y * 64 + x];
        let (ox, oy) = (fit.origin.0 as usize, fit.origin.1 as usize);
        // A row above the drawn band.
        assert_eq!(at(10, oy - 1, 0), LETTERBOX_PAD);
        assert_eq!(at(10, oy - 1, 2), LETTERBOX_PAD);
        // A pixel inside it: BGR order, so plane 0 is blue.
        assert_eq!(at(10 + ox, 5 + oy, 0), 200.0);
        assert_eq!(at(10 + ox, 5 + oy, 1), 5.0);
        assert_eq!(at(10 + ox, 5 + oy, 2), 10.0);
        // And a row below.
        assert_eq!(at(10, oy + fit.drawn.1 as usize, 1), LETTERBOX_PAD);
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
