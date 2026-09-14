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

    // A panel sizes itself to the bar, so the list scrolls in whatever is left. A fixed
    // height reserve went stale as buttons were added and clipped the last one off.
    if app.sidebar_tab == SidebarTab::Queue && !app.batch_files.is_empty() {
        egui::Panel::bottom("queue_action_bar")
            .resizable(false)
            .show_separator_line(false)
            .frame(egui::Frame::new())
            .show(ui, |ui| queue_action_bar(ui, app));
    }

    egui::ScrollArea::vertical()
        .id_salt("sidebar_scroll")
        .show(ui, |ui| match app.sidebar_tab {
            SidebarTab::Queue => show_queue(ui, app),
            SidebarTab::Mapping => show_mapping(ui, app),
            SidebarTab::History => show_history(ui, app),
        });
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
