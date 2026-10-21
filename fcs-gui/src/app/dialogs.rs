//! Modal dialogs and the About window.

pub(super) fn show_cuda_warning(ctx: &egui::Context, available: bool) -> Option<bool> {
    let mut choice = None;
    let response = egui::Modal::new(egui::Id::unique("confirm_cuda")).show(ctx, |ui| {
        ui.set_width(420.0);
        ui.heading("Enable experimental CUDA Rust?");
        ui.label(fcs_core::cuda::WARNING);
        ui.horizontal(|ui| {
            if ui.button("Cancel").clicked() {
                choice = Some(false);
            }
            if ui
                .add_enabled(available, egui::Button::new("Enable CUDA Rust"))
                .clicked()
            {
                choice = Some(true);
            }
        });
    });
    if choice.is_none() && response.should_close() {
        choice = Some(false);
    }
    choice
}

/// Ask what to do about files an export would replace. `Some(None)` is Cancel; `None` means the
/// user has not answered yet.
pub(super) fn show_overwrite_dialog(
    ctx: &egui::Context,
    pending: &crate::core::export::PendingExport,
) -> Option<Option<fcs_utils::OverwritePolicy>> {
    use crate::{core::export::PendingKind, theme::P};
    use fcs_utils::OverwritePolicy;

    const LISTED: usize = 6;
    let mut choice = None;
    let response = egui::Modal::new(egui::Id::unique("overwrite_dialog")).show(ctx, |ui| {
        ui.set_width(420.0);
        ui.label(
            egui::RichText::new("Some files already exist")
                .size(16.0)
                .color(P::INK)
                .strong(),
        );
        ui.add_space(8.0);
        let count = pending.conflicts.len();
        let folder = pending.output_dir.display();
        ui.label(match pending.kind {
            PendingKind::Faces(_) => {
                format!("{count} crop(s) would replace existing files in {folder}:")
            }
            PendingKind::Batch => {
                format!("This batch may replace {count} existing file(s) in {folder}:")
            }
        });
        ui.add_space(4.0);
        for path in pending.conflicts.iter().take(LISTED) {
            let name = path
                .strip_prefix(&pending.output_dir)
                .unwrap_or(path)
                .display()
                .to_string();
            ui.label(egui::RichText::new(name).monospace().color(P::INK2));
        }
        if count > LISTED {
            ui.label(egui::RichText::new(format!("…and {} more", count - LISTED)).color(P::INK3));
        }
        ui.add_space(8.0);
        ui.label(
            egui::RichText::new(
                "Save as new files keeps them, naming the new crops like photo_face_01(2).jpg.",
            )
            .size(11.5)
            .color(P::INK3),
        );
        ui.add_space(12.0);
        ui.horizontal(|ui| {
            if ui.button("Overwrite").clicked() {
                choice = Some(Some(OverwritePolicy::Overwrite));
            }
            if ui.button("Save as new files").clicked() {
                choice = Some(Some(OverwritePolicy::KeepBoth));
            }
            if ui.button("Cancel").clicked() {
                choice = Some(None);
            }
        });
    });
    // Escape or a click outside is Cancel.
    if choice.is_none() && response.should_close() {
        choice = Some(None);
    }
    choice
}

// ── About dialog ──────────────────────────────────────────────────────────────

pub(super) fn show_about_window(ctx: &egui::Context, open: &mut bool) {
    use crate::theme::P;

    let window_frame = egui::Frame::window(&ctx.global_style())
        .fill(P::BG2)
        .stroke(egui::Stroke::new(1.0, P::RULE2))
        .corner_radius(egui::CornerRadius::same(8))
        .inner_margin(egui::Margin::same(24))
        .shadow(egui::Shadow {
            offset: [0, 8],
            blur: 32,
            spread: 0,
            color: P::black_alpha(120),
        });

    let mut close = false;
    egui::Window::new("About Face Crop Studio")
        .open(open)
        .collapsible(false)
        .resizable(false)
        .fixed_size(egui::vec2(340.0, 0.0))
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .title_bar(false)
        .frame(window_frame)
        .show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add(
                    egui::Image::new(egui::include_image!("../../assets/app_logo.svg"))
                        .fit_to_exact_size(egui::Vec2::splat(48.0)),
                );

                ui.add_space(10.0);
                ui.label(
                    egui::RichText::new("Face Crop Studio")
                        .size(17.0)
                        .color(P::INK)
                        .strong(),
                );
                ui.add_space(2.0);
                ui.label(
                    egui::RichText::new(concat!("v", env!("CARGO_PKG_VERSION")))
                        .size(11.0)
                        .color(P::INK3),
                );
                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new("AI-powered face detection and cropping")
                        .size(12.0)
                        .color(P::INK3),
                );
            });

            ui.add_space(16.0);
            ui.separator();
            ui.add_space(12.0);

            ui.label(
                egui::RichText::new(
                    "Face Crop Studio uses a neural network to detect faces \
                     and automatically crop portraits at scale.",
                )
                .size(12.5)
                .color(P::INK2),
            );

            ui.add_space(14.0);

            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Website").size(12.0).color(P::INK3));
                ui.add_space(4.0);
                ui.hyperlink_to(
                    egui::RichText::new("facecropstudio.com")
                        .size(12.0)
                        .color(P::CYAN),
                    "https://facecropstudio.com/",
                );
            });

            ui.add_space(16.0);

            ui.vertical_centered(|ui| {
                if ui.button(egui::RichText::new("Close").size(13.0)).clicked() {
                    close = true;
                }
            });
        });
    if close {
        *open = false;
    }
}
