//! Left sidebar: Queue / Mapping / History tabs.

mod history;
mod mapping;
mod queue;

use crate::types::{App2, SidebarTab};
use egui::Ui;

use history::show_history;
use mapping::show_mapping;
use queue::{queue_action_bar, show_queue};

pub fn show(ui: &mut Ui, app: &mut App2) {
    ui.set_min_height(ui.available_height());

    tab_bar(ui, app);

    const ACTION_BAR_H: f32 = 96.0;
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
    let tabs = [SidebarTab::Queue, SidebarTab::Mapping, SidebarTab::History];
    let mut selected = tabs
        .iter()
        .position(|tab| *tab == app.sidebar_tab)
        .unwrap_or(0);
    egui::Frame::new().inner_margin(8).show(ui, |ui| {
        crate::ui::widgets::segmented_control(ui, &["Queue", "Mapping", "History"], &mut selected);
    });
    app.sidebar_tab = tabs[selected];
}
