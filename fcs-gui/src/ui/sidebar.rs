//! Left sidebar: Queue / Mapping / History tabs.

mod history;
mod mapping;
mod queue;

use crate::theme::P;
use crate::types::{App2, SidebarTab};
use egui::{Sense, Stroke, Ui, Vec2};

use history::show_history;
use mapping::show_mapping;
use queue::{queue_action_bar, show_queue};

pub fn show(ui: &mut Ui, app: &mut App2) {
    ui.set_min_height(ui.available_height());

    tab_bar(ui, app);

    const ACTION_BAR_H: f32 = 82.0;
    let queue_has_files = app.sidebar_tab == SidebarTab::Queue && !app.batch_files.is_empty();
    let scroll_max_h = if queue_has_files {
        (ui.available_height() - ACTION_BAR_H).max(80.0)
    } else {
        f32::INFINITY
    };

    egui::ScrollArea::vertical()
        .id_salt("sidebar_scroll")
        .max_height(scroll_max_h)
        .show(ui, |ui| match app.sidebar_tab {
            SidebarTab::Queue => show_queue(ui, app),
            SidebarTab::Mapping => show_mapping(ui, app),
            SidebarTab::History => show_history(ui, app),
        });

    if queue_has_files {
        queue_action_bar(ui, app);
    }
}

fn tab_bar(ui: &mut Ui, app: &mut App2) {
    let tabs = [
        ("Queue", SidebarTab::Queue),
        ("Mapping", SidebarTab::Mapping),
        ("History", SidebarTab::History),
    ];
    ui.painter().line_segment(
        [
            egui::pos2(ui.min_rect().min.x, ui.min_rect().min.y + 32.0),
            egui::pos2(ui.min_rect().max.x, ui.min_rect().min.y + 32.0),
        ],
        Stroke::new(1.0, P::RULE),
    );
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        ui.set_height(32.0);
        let w = ui.available_width() / tabs.len() as f32;
        for (label, variant) in &tabs {
            let is_active = app.sidebar_tab == *variant;
            let (resp, painter) = ui.allocate_painter(Vec2::new(w, 32.0), Sense::click());
            let text_color = if is_active { P::PEACH } else { P::INK3 };
            if resp.hovered() && !is_active {
                painter.rect_filled(resp.rect, 0.0, P::white_alpha(5));
            }
            if is_active {
                painter.rect_filled(resp.rect, 0.0, P::peach_alpha(10));
                painter.line_segment(
                    [
                        egui::pos2(resp.rect.min.x, resp.rect.max.y - 2.0),
                        egui::pos2(resp.rect.max.x, resp.rect.max.y - 2.0),
                    ],
                    Stroke::new(2.0, P::PEACH),
                );
            }
            painter.text(
                resp.rect.center(),
                egui::Align2::CENTER_CENTER,
                *label,
                egui::FontId::monospace(10.5),
                text_color,
            );
            if resp.clicked() {
                app.sidebar_tab = *variant;
            }
        }
    });
}
