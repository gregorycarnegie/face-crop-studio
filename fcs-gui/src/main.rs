#![cfg_attr(
    all(target_os = "windows", not(debug_assertions)),
    windows_subsystem = "windows"
)]

use eframe::{NativeOptions, egui};
use fcs_gui::App2;
use fcs_utils::init_logging;
use log::{LevelFilter, warn};
use std::sync::Arc;

fn main() -> eframe::Result<()> {
    init_logging(LevelFilter::Info).expect("failed to initialize logging");
    let mut options = NativeOptions::default();
    options.viewport = options
        .viewport
        .with_inner_size([1480.0, 920.0])
        .with_min_inner_size([1000.0, 640.0])
        .with_resizable(true)
        .with_drag_and_drop(true);

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
    // The `image` ico decoder picks the best-quality frame in the file.
    match image::load_from_memory_with_format(ICON_BYTES, image::ImageFormat::Ico) {
        Ok(img) => {
            let rgba = img.to_rgba8();
            let (width, height) = rgba.dimensions();
            Some(egui::IconData {
                rgba: rgba.into_raw(),
                width,
                height,
            })
        }
        Err(err) => {
            warn!("Failed to read app icon: {err}");
            None
        }
    }
}
