//! Pasting an image from the system clipboard.

use std::{path::PathBuf, sync::Arc};

use image::DynamicImage;

use crate::types::{App2, LogKind};

/// Reads an image from the system clipboard.
///
/// Tries arboard (CF_DIB / CF_BITMAP) first. On Windows, falls back to the
/// registered `"PNG"` clipboard format used by browsers and some screenshot
/// tools that only write PNG rather than a Device-Independent Bitmap.
fn read_clipboard_image() -> anyhow::Result<Arc<DynamicImage>> {
    use image::{DynamicImage, RgbaImage};

    if let Ok(mut cb) = arboard::Clipboard::new()
        && let Ok(data) = cb.get_image()
        && let Some(rgba) = RgbaImage::from_raw(
            data.width as u32,
            data.height as u32,
            data.bytes.into_owned(),
        )
    {
        return Ok(Arc::new(DynamicImage::ImageRgba8(rgba)));
    }

    #[cfg(target_os = "windows")]
    if let Ok(img) = windows_png_from_clipboard() {
        return Ok(Arc::new(img));
    }

    Err(anyhow::anyhow!(
        "No image found in clipboard. Copy an image first, then paste."
    ))
}

/// Windows-only fallback: reads the `"PNG"` registered clipboard format.
///
/// Browsers and some screenshot tools (e.g. ShareX, Greenshot) write PNG bytes
/// under this format instead of CF_DIB, which arboard does not handle.
#[cfg(target_os = "windows")]
fn windows_png_from_clipboard() -> anyhow::Result<image::DynamicImage> {
    use windows::Win32::{
        Foundation::HGLOBAL,
        System::{
            DataExchange::{
                CloseClipboard, GetClipboardData, IsClipboardFormatAvailable, OpenClipboard,
                RegisterClipboardFormatW,
            },
            Memory::{GlobalLock, GlobalSize, GlobalUnlock},
        },
    };

    unsafe {
        let format = RegisterClipboardFormatW(windows::core::w!("PNG"));
        anyhow::ensure!(format != 0, "RegisterClipboardFormatW failed");

        if IsClipboardFormatAvailable(format).is_err() {
            return Err(anyhow::anyhow!("PNG format not in clipboard"));
        }

        // None = NULL HWND (clipboard not owned by any window)
        OpenClipboard(None).map_err(|e| anyhow::anyhow!("OpenClipboard: {e}"))?;

        let result = (|| -> anyhow::Result<image::DynamicImage> {
            let handle =
                GetClipboardData(format).map_err(|e| anyhow::anyhow!("GetClipboardData: {e}"))?;
            let hglobal = HGLOBAL(handle.0);
            let size = GlobalSize(hglobal);
            anyhow::ensure!(size > 0, "clipboard PNG data is empty");
            let ptr = GlobalLock(hglobal);
            anyhow::ensure!(!ptr.is_null(), "GlobalLock failed");
            let bytes = std::slice::from_raw_parts(ptr.cast::<u8>(), size);
            let img =
                image::load_from_memory(bytes).map_err(|e| anyhow::anyhow!("PNG decode: {e}"))?;
            let _ = GlobalUnlock(hglobal);
            Ok(img)
        })();

        let _ = CloseClipboard();
        result
    }
}

impl App2 {
    pub fn paste_clipboard_image(&mut self) {
        use crate::core::detection::spawn_detection_job_from_image;

        let image = match read_clipboard_image() {
            Ok(img) => img,
            Err(err) => {
                self.show_error("Clipboard", err.to_string());
                return;
            }
        };

        let path = PathBuf::from("clipboard://paste");
        self.preview.begin_loading(path.clone());
        self.is_busy = true;
        let job_id = self.job_counter;
        self.job_counter += 1;
        self.current_job = Some(job_id);
        self.push_log("Detecting faces in clipboard image…".into(), LogKind::Info);

        spawn_detection_job_from_image(
            job_id,
            image,
            path,
            self.detector.clone(),
            self.eye_refiner.clone(),
            self.job_tx.clone(),
        );
    }
}
