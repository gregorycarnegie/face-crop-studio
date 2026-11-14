//! Images held in memory for QML. `cpp/image_provider.cpp` serves them as `image://fcs/<key>`,
//! calling [`fcs_image_request`] whenever an `Image` element loads one, so nothing touches
//! the disk.

use image::{RgbImage, RgbaImage};
use std::{
    collections::HashMap,
    ffi::c_void,
    sync::{LazyLock, Mutex, MutexGuard},
};

pub enum Pixels {
    Rgb(RgbImage),
    Rgba(RgbaImage),
}

// A process-wide store: there is one QML engine, and Qt calls the provider from its own
// loader threads, outside anything Studio owns.
static STORE: LazyLock<Mutex<HashMap<String, Pixels>>> = LazyLock::new(Default::default);

fn store() -> MutexGuard<'static, HashMap<String, Pixels>> {
    STORE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Hold `pixels` under `key`, replacing what was there, and return the URL QML loads it by.
pub fn put(key: &str, pixels: Pixels) -> String {
    store().insert(key.to_string(), pixels);
    url(key)
}

pub fn url(key: &str) -> String {
    format!("image://fcs/{key}")
}

pub fn remove(key: &str) {
    store().remove(key);
}

/// Called by the provider with a `QImage*` as `ctx`: sizes the image through `alloc`, which
/// returns its pixel buffer and row stride, and copies the pixels in.
pub type Alloc = unsafe extern "C" fn(
    ctx: *mut c_void,
    width: u32,
    height: u32,
    channels: u32,
    stride: *mut usize,
) -> *mut u8;

/// Copy the image stored under the UTF-8 key `id[..id_len]` into the buffer `alloc` provides.
/// Returns false for an unknown key or a failed allocation.
///
/// # Safety
///
/// `id` must point to `id_len` readable bytes, and `alloc` must return null or a buffer of at
/// least `stride * height` writable bytes with `stride` written.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fcs_image_request(
    id: *const u8,
    id_len: usize,
    ctx: *mut c_void,
    alloc: Alloc,
) -> bool {
    if id.is_null() {
        return false;
    }
    // SAFETY: the caller passes a valid key buffer.
    let key = unsafe { std::slice::from_raw_parts(id, id_len) };
    let Ok(key) = std::str::from_utf8(key) else {
        return false;
    };
    let store = store();
    let Some(pixels) = store.get(key) else {
        return false;
    };
    let (width, height, channels, raw) = match pixels {
        Pixels::Rgb(i) => (i.width(), i.height(), 3, i.as_raw()),
        Pixels::Rgba(i) => (i.width(), i.height(), 4, i.as_raw()),
    };
    let row = width as usize * channels as usize;
    let mut stride = 0;
    // SAFETY: the caller's allocator, called as documented.
    let dst = unsafe { alloc(ctx, width, height, channels, &mut stride) };
    if dst.is_null() || stride < row {
        return false;
    }
    // Row by row: QImage pads rows to 4 bytes, so an RGB row can be shorter than the stride.
    for (y, src) in raw.chunks_exact(row).enumerate() {
        // SAFETY: `alloc` gave `stride * height` bytes and `row <= stride`.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst.add(y * stride), row) };
    }
    true
}

unsafe extern "C" {
    /// cpp/image_provider.cpp: the clipboard's image as RGBA, sized through `alloc`.
    fn fcs_clipboard_image(ctx: *mut c_void, alloc: Alloc) -> bool;
    /// cpp/image_provider.cpp: `qVersion()`, a static NUL-terminated string.
    fn fcs_qt_version() -> *const std::ffi::c_char;
}

/// The version of the Qt libraries loaded at run time.
pub fn qt_version() -> String {
    // SAFETY: qVersion returns a static, NUL-terminated string.
    unsafe { std::ffi::CStr::from_ptr(fcs_qt_version()) }
        .to_string_lossy()
        .into_owned()
}

/// A tightly packed buffer for the C++ side to fill, as `(width, height, bytes)`.
type Filled = (u32, u32, Vec<u8>);

unsafe extern "C" fn fill(
    ctx: *mut c_void,
    width: u32,
    height: u32,
    channels: u32,
    stride: *mut usize,
) -> *mut u8 {
    // SAFETY: `clipboard` passes a `Filled` as ctx and a valid stride pointer.
    let filled = unsafe { &mut *ctx.cast::<Filled>() };
    let row = width as usize * channels as usize;
    *filled = (width, height, vec![0; row * height as usize]);
    unsafe { *stride = row };
    filled.2.as_mut_ptr()
}

/// The image on the system clipboard, if it holds one. Call on the GUI thread (QClipboard).
pub fn clipboard() -> Option<RgbaImage> {
    let mut filled: Filled = (0, 0, Vec::new());
    // SAFETY: `fill` matches `Alloc` and `filled` outlives the call.
    let found = unsafe { fcs_clipboard_image(std::ptr::from_mut(&mut filled).cast(), fill) };
    found
        .then(|| RgbaImage::from_raw(filled.0, filled.1, filled.2))
        .flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_sizes_a_packed_rgba_buffer() {
        let mut filled: Filled = (0, 0, Vec::new());
        let mut stride = 0;
        // SAFETY: a `Filled` ctx and a valid stride pointer, as `clipboard` passes.
        let ptr = unsafe { fill(std::ptr::from_mut(&mut filled).cast(), 3, 2, 4, &mut stride) };
        assert_eq!(stride, 12);
        assert_eq!((filled.0, filled.1, filled.2.len()), (3, 2, 24));
        assert_eq!(ptr, filled.2.as_mut_ptr());
        assert!(RgbaImage::from_raw(filled.0, filled.1, filled.2).is_some());
    }

    /// Stands in for a QImage: a buffer whose rows are padded to 4 bytes.
    struct Target {
        stride: usize,
        bytes: Vec<u8>,
        shape: (u32, u32, u32),
    }

    unsafe extern "C" fn alloc(
        ctx: *mut c_void,
        width: u32,
        height: u32,
        channels: u32,
        stride: *mut usize,
    ) -> *mut u8 {
        // SAFETY: the tests pass a `Target` as ctx and a valid stride pointer.
        let target = unsafe { &mut *ctx.cast::<Target>() };
        target.stride = (width as usize * channels as usize).next_multiple_of(4);
        target.bytes = vec![0xEE; target.stride * height as usize];
        target.shape = (width, height, channels);
        unsafe { *stride = target.stride };
        target.bytes.as_mut_ptr()
    }

    fn request(key: &str) -> Option<Target> {
        let mut target = Target {
            stride: 0,
            bytes: Vec::new(),
            shape: (0, 0, 0),
        };
        let ctx = std::ptr::from_mut(&mut target).cast();
        // SAFETY: a valid key buffer and the allocator above.
        let found = unsafe { fcs_image_request(key.as_ptr(), key.len(), ctx, alloc) };
        found.then_some(target)
    }

    #[test]
    fn copies_rows_into_a_padded_buffer_and_leaves_the_padding() {
        // 3 px of RGB is 9 bytes a row, padded to 12.
        let rgb = RgbImage::from_fn(3, 2, |x, y| image::Rgb([x as u8, y as u8, 7]));
        let url = put("test/rgb", Pixels::Rgb(rgb));
        assert_eq!(url, "image://fcs/test/rgb");

        let target = request("test/rgb").expect("stored");
        assert_eq!((target.shape, target.stride), ((3, 2, 3), 12));
        assert_eq!(
            &target.bytes[..12],
            &[0, 0, 7, 1, 0, 7, 2, 0, 7, 0xEE, 0xEE, 0xEE]
        );
        assert_eq!(&target.bytes[12..21], &[0, 1, 7, 1, 1, 7, 2, 1, 7]);

        let rgba = RgbaImage::from_pixel(2, 1, image::Rgba([1, 2, 3, 4]));
        put("test/rgba", Pixels::Rgba(rgba));
        let target = request("test/rgba").expect("stored");
        assert_eq!(target.shape, (2, 1, 4));
        assert_eq!(target.bytes, [1, 2, 3, 4, 1, 2, 3, 4]);

        remove("test/rgb");
        assert!(request("test/rgb").is_none());
        assert!(request("test/never").is_none());
    }
}
