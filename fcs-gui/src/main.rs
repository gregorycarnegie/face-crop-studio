#![cfg_attr(
    all(target_os = "windows", not(debug_assertions)),
    windows_subsystem = "windows"
)]

/// See the `mimalloc` note in the workspace Cargo.toml: tract's per-node tensor churn is
/// pathological on the Windows system heap, and swapping the allocator is worth ~35% of a
/// single detection.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use eframe::{NativeOptions, egui};
use fcs_gui::App2;
use fcs_utils::{init_logging, platform_safe_backends};
use log::{LevelFilter, warn};
use std::sync::Arc;

fn main() -> eframe::Result<()> {
    let _ = fcs_gui::LAUNCH.set(std::time::Instant::now());
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

    // eframe defaults to `Backends::PRIMARY | GL`, which includes Vulkan. On
    // Windows that lets wgpu pick Intel's Vulkan driver, which faults during
    // startup and takes the process with it before a window ever appears — see
    // `platform_safe_backends`. Everything the renderer needs is available on
    // DX12, and GL stays as the fallback eframe already relied on.
    if let eframe::egui_wgpu::WgpuSetup::CreateNew(setup) = &mut options.wgpu_options.wgpu_setup {
        setup.instance_descriptor.backends =
            platform_safe_backends(setup.instance_descriptor.backends);
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
