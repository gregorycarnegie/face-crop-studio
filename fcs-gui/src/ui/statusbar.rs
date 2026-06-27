//! Bottom status bar.

use crate::{theme::P, types::App2};
use egui::{Frame, Sense, Stroke, Vec2};

pub fn show(ui: &mut egui::Ui, app: &mut App2) {
    // Always repaint once per second so the clock and stats stay live even when idle.
    ui.ctx()
        .request_repaint_after(std::time::Duration::from_secs(1));

    egui::Panel::bottom("statusbar")
        .exact_size(28.0)
        .show_separator_line(false)
        .frame(Frame::new().fill(P::BG).inner_margin(egui::Margin::ZERO))
        .show(ui, |ui| {
            ui.horizontal_centered(|ui| {
                // Status dot + label
                let (ready_dot, ready_text) = if app.is_busy {
                    (P::PEACH, "  Running".to_string())
                } else if app.last_error.is_some() {
                    (P::ROSE, " Error".to_string())
                } else {
                    (P::LIME, " Ready".to_string())
                };
                status_cell(ui, &ready_text, Some(ready_dot));

                // Model
                let model_name = app
                    .settings
                    .model_path
                    .as_deref()
                    .and_then(|p| std::path::Path::new(p).file_stem())
                    .and_then(|s| s.to_str())
                    .unwrap_or("YuNet 640");
                status_cell(ui, &format!("{model_name} · ONNX"), None);

                // GPU adapter name
                let gpu_label = app
                    .gpu
                    .status
                    .adapter_name
                    .as_deref()
                    .map(|n| {
                        let backend = app.gpu.status.backend.as_deref().unwrap_or("wgpu");
                        format!("{backend} · {n}")
                    })
                    .unwrap_or_else(|| "wgpu · CPU".to_string());
                status_cell(ui, &gpu_label, None);

                // Batch progress
                if !app.batch_files.is_empty() {
                    let done = app
                        .batch_files
                        .iter()
                        .filter(|f| {
                            !matches!(
                                f.status,
                                crate::types::BatchFileStatus::Pending
                                    | crate::types::BatchFileStatus::Processing
                            )
                        })
                        .count();
                    let total = app.batch_files.len();
                    status_cell_dot(
                        ui,
                        &format!("Batch {done} / {total}"),
                        P::PEACH,
                        app.is_busy,
                    );
                }

                // Right side
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // Clock — local time via Win32 GetLocalTime
                    status_cell(ui, &local_time_str(), None);

                    // GPU / RAM stats — only available on macOS and Windows.
                    #[cfg(any(target_os = "macos", target_os = "windows"))]
                    {
                        // GPU — dedicated VRAM usage via DXGI
                        let vram_label = gpu_vram_mb()
                            .map(|mb| format!("VRAM {mb} MB"))
                            .unwrap_or_else(|| "GPU —".to_string());
                        status_cell(ui, &vram_label, None);

                        // RAM — current process working set
                        let ram_label = process_ram_mb()
                            .map(|mb| format!("RAM {mb} MB"))
                            .unwrap_or_else(|| "RAM —".to_string());
                        status_cell(ui, &ram_label, None);
                    }

                    // Animated indeterminate progress bar when busy
                    if app.is_busy {
                        let (resp, painter) =
                            ui.allocate_painter(Vec2::new(140.0, 5.0), Sense::hover());
                        painter.rect_filled(resp.rect, 3.0, P::white_alpha(15));

                        // Bounce a 40%-wide block back and forth using egui's time.
                        let t = ui.ctx().input(|i| i.time) as f32;
                        let phase = (t * 1.2).sin() * 0.5 + 0.5; // 0..1 smooth bounce
                        let block_w = resp.rect.width() * 0.4;
                        let x = resp.rect.min.x + phase * (resp.rect.width() - block_w);
                        let fill = egui::Rect::from_min_max(
                            egui::pos2(x, resp.rect.min.y),
                            egui::pos2(x + block_w, resp.rect.max.y),
                        );
                        painter.rect_filled(fill, 3.0, P::PEACH);
                        ui.add_space(6.0);
                    }
                });
            });
        });
}

fn status_cell(ui: &mut egui::Ui, text: &str, dot: Option<egui::Color32>) {
    let (_, _painter) = ui.allocate_painter(Vec2::new(1.0, 28.0), Sense::hover());
    ui.painter().line_segment(
        [
            egui::pos2(ui.cursor().min.x, ui.min_rect().min.y),
            egui::pos2(ui.cursor().min.x, ui.min_rect().max.y),
        ],
        Stroke::new(1.0, P::RULE),
    );
    ui.horizontal_centered(|ui| {
        ui.add_space(10.0);
        if let Some(color) = dot {
            let (resp, painter) = ui.allocate_painter(Vec2::splat(6.0), Sense::hover());
            painter.circle_filled(resp.rect.center(), 3.0, color);
            ui.add_space(4.0);
        }
        ui.label(
            egui::RichText::new(text)
                .size(10.5)
                .color(P::INK3)
                .family(egui::FontFamily::Monospace),
        );
        ui.add_space(10.0);
    });
    ui.painter().line_segment(
        [
            egui::pos2(ui.cursor().min.x - 1.0, ui.min_rect().min.y),
            egui::pos2(ui.cursor().min.x - 1.0, ui.min_rect().max.y),
        ],
        Stroke::new(1.0, P::RULE),
    );
}

fn status_cell_dot(ui: &mut egui::Ui, text: &str, dot_color: egui::Color32, _animate: bool) {
    ui.horizontal_centered(|ui| {
        ui.add_space(10.0);
        let (resp, painter) = ui.allocate_painter(Vec2::splat(6.0), Sense::hover());
        painter.circle_filled(resp.rect.center(), 3.0, dot_color);
        ui.add_space(4.0);
        ui.label(
            egui::RichText::new(text)
                .size(10.5)
                .color(P::INK3)
                .family(egui::FontFamily::Monospace),
        );
        ui.add_space(10.0);
        ui.painter().line_segment(
            [
                egui::pos2(ui.cursor().min.x - 1.0, ui.min_rect().min.y),
                egui::pos2(ui.cursor().min.x - 1.0, ui.min_rect().max.y),
            ],
            Stroke::new(1.0, P::RULE),
        );
    });
}

/// Returns current local time as HH:MM:SS.
/// Uses Win32 `GetLocalTime` (kernel32, auto-linked) so the result reflects the
/// system timezone rather than UTC.
fn local_time_str() -> String {
    #[cfg(target_os = "windows")]
    {
        #[repr(C)]
        struct SystemTime {
            year: u16,
            month: u16,
            dow: u16,
            day: u16,
            hour: u16,
            minute: u16,
            second: u16,
            ms: u16,
        }
        unsafe extern "system" {
            fn GetLocalTime(lp: *mut SystemTime);
        }
        let mut t = SystemTime {
            year: 0,
            month: 0,
            dow: 0,
            day: 0,
            hour: 0,
            minute: 0,
            second: 0,
            ms: 0,
        };
        unsafe { GetLocalTime(&mut t) }
        return format!("{:02}:{:02}:{:02}", t.hour, t.minute, t.second);
    }
    #[allow(unreachable_code)]
    {
        // UTC fallback for non-Windows
        use std::time::{SystemTime, UNIX_EPOCH};
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        format!(
            "{:02}:{:02}:{:02}",
            (secs / 3600) % 24,
            (secs / 60) % 60,
            secs % 60
        )
    }
}

/// Returns this process's working-set size in MiB, or None if the query fails.
/// Uses `K32GetProcessMemoryInfo` (kernel32, auto-linked on Windows Vista+).
#[cfg(any(target_os = "macos", target_os = "windows"))]
fn process_ram_mb() -> Option<u64> {
    #[cfg(target_os = "windows")]
    {
        #[repr(C)]
        struct Pmc {
            cb: u32,
            page_faults: u32,
            peak_ws: usize,
            working_set: usize,
            qpeak_paged: usize,
            qpaged: usize,
            qpeak_nonpaged: usize,
            qnonpaged: usize,
            pagefile: usize,
            peak_pagefile: usize,
        }
        unsafe extern "system" {
            fn GetCurrentProcess() -> *mut std::ffi::c_void;
            fn K32GetProcessMemoryInfo(proc: *mut std::ffi::c_void, pmc: *mut Pmc, cb: u32) -> i32;
        }
        let mut pmc: Pmc = unsafe { std::mem::zeroed() };
        pmc.cb = std::mem::size_of::<Pmc>() as u32;
        let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut pmc, pmc.cb) };
        if ok != 0 {
            return Some(pmc.working_set as u64 >> 20); // bytes to MiB
        }
        return None;
    }
    #[allow(unreachable_code)]
    None
}

/// Returns the dedicated GPU VRAM currently used by this process in MiB.
/// Uses `IDXGIAdapter3::QueryVideoMemoryInfo` (DXGI, already linked via eframe/wgpu).
#[cfg(any(target_os = "macos", target_os = "windows"))]
fn gpu_vram_mb() -> Option<u64> {
    #[cfg(target_os = "windows")]
    {
        use windows::{
            Win32::Graphics::Dxgi::{
                CreateDXGIFactory1, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, IDXGIAdapter3, IDXGIFactory1,
            },
            core::Interface as _,
        };
        unsafe {
            let factory: IDXGIFactory1 = CreateDXGIFactory1().ok()?;
            let adapter1 = factory.EnumAdapters1(0).ok()?;
            let adapter3: IDXGIAdapter3 = adapter1.cast().ok()?;
            let mut info = windows::Win32::Graphics::Dxgi::DXGI_QUERY_VIDEO_MEMORY_INFO::default();
            adapter3
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info)
                .ok()?;
            if info.Budget == 0 {
                return None;
            }
            return Some(info.CurrentUsage >> 20); //bytes to MiB
        }
    }
    #[allow(unreachable_code)]
    None
}
