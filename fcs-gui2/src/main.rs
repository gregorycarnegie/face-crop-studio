#![cfg_attr(
    all(target_os = "windows", not(debug_assertions)),
    windows_subsystem = "windows"
)]

use eframe::{NativeOptions, egui};
use fcs_gui2::App2;
use fcs_utils::init_logging;
use ico::IconDir;
use log::{LevelFilter, warn};
use std::{io::Cursor, sync::Arc};

fn main() -> eframe::Result<()> {
    init_logging(LevelFilter::Info).expect("failed to initialize logging");
    let mut options = NativeOptions::default();
    options.viewport = options
        .viewport
        .with_inner_size([1480.0, 920.0])
        .with_min_inner_size([1000.0, 640.0])
        .with_resizable(true)
        .with_drag_and_drop(true)
        .with_decorations(false);

    if let Some(icon) = load_app_icon() {
        options.viewport = options.viewport.with_icon(Arc::new(icon));
    }

    eframe::run_native(
        "Face Crop Studio",
        options,
        Box::new(|cc| Ok(Box::new(App2::new(cc)))),
    )
}

fn load_app_icon() -> Option<egui::IconData> {
    const ICON_BYTES: &[u8] = include_bytes!("../assets/app_icon.ico");
    let dir = match IconDir::read(Cursor::new(ICON_BYTES)) {
        Ok(d) => d,
        Err(err) => {
            warn!("Failed to read app icon: {err}");
            return None;
        }
    };
    let mut best: Option<(ico::IconImage, u32)> = None;
    for entry in dir.entries() {
        if let Ok(img) = entry.decode() {
            let score = img.width().saturating_mul(img.height());
            if best.as_ref().is_none_or(|(_, s)| score > *s) {
                best = Some((img, score));
            }
        }
    }
    best.map(|(img, _)| egui::IconData {
        rgba: img.rgba_data().to_vec(),
        width: img.width(),
        height: img.height(),
    })
}
