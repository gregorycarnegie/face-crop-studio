//! Left sidebar: Queue / Mapping / History tabs.

use crate::theme::P;
use crate::types::{App2, BatchFileStatus, SidebarTab};
use egui::{Sense, Stroke, Ui, Vec2};

pub fn show(ui: &mut Ui, app: &mut App2) {
    ui.set_min_height(ui.available_height());

    // Tab bar
    tab_bar(ui, app);

    // Scrollable content
    egui::ScrollArea::vertical()
        .id_salt("sidebar_scroll")
        .show(ui, |ui| match app.sidebar_tab {
            SidebarTab::Queue => show_queue(ui, app),
            SidebarTab::Mapping => show_mapping(ui, app),
            SidebarTab::History => show_history(ui, app),
        });
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

fn show_queue(ui: &mut Ui, app: &mut App2) {
    // Drop zone
    drop_zone(ui, app);

    // File tree
    file_tree(ui, app);
}

fn drop_zone(ui: &mut Ui, app: &mut App2) {
    ui.add_space(8.0);
    let dz_rect = egui::Rect::from_min_size(
        egui::pos2(ui.min_rect().min.x + 8.0, ui.cursor().min.y),
        Vec2::new(ui.available_width() - 16.0, 100.0),
    );
    let resp = ui.allocate_rect(dz_rect, Sense::click());
    let painter = ui.painter();

    let border_color = if resp.hovered() {
        P::CYAN
    } else {
        P::cyan_alpha(89)
    };
    let bg_color = if resp.hovered() {
        P::cyan_alpha(30)
    } else {
        P::cyan_alpha(20)
    };

    // Draw dashed border
    painter.rect_filled(dz_rect, 10.0, bg_color);
    draw_dashed_border(painter, dz_rect, border_color);

    // Icon
    let icon_rect = egui::Rect::from_center_size(
        egui::pos2(dz_rect.center().x, dz_rect.min.y + 28.0),
        Vec2::splat(34.0),
    );
    painter.rect_filled(icon_rect, 9.0, P::cyan_alpha(40));
    painter.text(
        icon_rect.center(),
        egui::Align2::CENTER_CENTER,
        "↑",
        egui::FontId::proportional(16.0),
        P::CYAN,
    );

    painter.text(
        egui::pos2(dz_rect.center().x, dz_rect.min.y + 52.0),
        egui::Align2::CENTER_CENTER,
        "Drop images or folder",
        egui::FontId::proportional(13.0),
        P::INK,
    );
    painter.text(
        egui::pos2(dz_rect.center().x, dz_rect.min.y + 68.0),
        egui::Align2::CENTER_CENTER,
        "JPG · PNG · WEBP",
        egui::FontId::proportional(11.5),
        P::INK3,
    );
    painter.text(
        egui::pos2(dz_rect.center().x, dz_rect.min.y + 86.0),
        egui::Align2::CENTER_CENTER,
        "[ Browse ]  [ Paste ]",
        egui::FontId::monospace(10.0),
        P::INK2,
    );
    ui.add_space(108.0);

    if resp.clicked() {
        // Open file dialog
        if let Some(paths) = rfd::FileDialog::new()
            .add_filter(
                "Images",
                &["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
            )
            .pick_files()
        {
            let first = paths.first().cloned();
            let added = app.enqueue_batch_paths(paths);
            if let Some(path) = first {
                app.load_image_path(path);
            }
            if added > 0 {
                app.show_success(format!(
                    "Added {added} image(s) to the queue ({} total)",
                    app.batch_files.len()
                ));
            }
        }
    }
}

fn draw_dashed_border(painter: &egui::Painter, rect: egui::Rect, color: egui::Color32) {
    let stroke = Stroke::new(1.5, color);
    let dash = 6.0;
    let gap = 4.0;
    let r = rect;
    let mut x = r.min.x;
    while x < r.max.x {
        let ex = (x + dash).min(r.max.x);
        painter.line_segment([egui::pos2(x, r.min.y), egui::pos2(ex, r.min.y)], stroke);
        painter.line_segment([egui::pos2(x, r.max.y), egui::pos2(ex, r.max.y)], stroke);
        x += dash + gap;
    }
    let mut y = r.min.y;
    while y < r.max.y {
        let ey = (y + dash).min(r.max.y);
        painter.line_segment([egui::pos2(r.min.x, y), egui::pos2(r.min.x, ey)], stroke);
        painter.line_segment([egui::pos2(r.max.x, y), egui::pos2(r.max.x, ey)], stroke);
        y += dash + gap;
    }
}

fn file_tree(ui: &mut Ui, app: &mut App2) {
    if app.batch_files.is_empty() {
        return;
    }

    let total = app.batch_files.len();
    let mut action = None;
    let in_progress: Vec<usize> = (0..total)
        .filter(|&i| {
            matches!(
                app.batch_files[i].status,
                BatchFileStatus::Processing
                    | BatchFileStatus::Completed { .. }
                    | BatchFileStatus::Failed { .. }
            )
        })
        .collect();
    let queued: Vec<usize> = (0..total)
        .filter(|&i| {
            !matches!(
                app.batch_files[i].status,
                BatchFileStatus::Processing
                    | BatchFileStatus::Completed { .. }
                    | BatchFileStatus::Failed { .. }
            )
        })
        .collect();

    if !in_progress.is_empty() {
        tree_group_header(ui, "In progress", in_progress.len(), total);
        for idx in in_progress {
            let row_action = tree_row(ui, app, idx);
            if action.is_none() {
                action = row_action;
            }
        }
    }
    if !queued.is_empty() {
        tree_group_header(ui, "Queued", queued.len(), 0);
        for idx in queued {
            let row_action = tree_row(ui, app, idx);
            if action.is_none() {
                action = row_action;
            }
        }
    }

    match action {
        Some(TreeAction::Load(path)) => app.load_image_path(path),
        Some(TreeAction::Remove(idx)) => {
            if idx < app.batch_files.len() {
                app.batch_files.remove(idx);
                app.show_success("Removed image from queue.");
            }
        }
        None => {}
    }
}

enum TreeAction {
    Load(std::path::PathBuf),
    Remove(usize),
}

fn tree_group_header(ui: &mut Ui, label: &str, count: usize, total: usize) {
    ui.horizontal(|ui| {
        ui.set_height(28.0);
        ui.add_space(8.0);
        ui.label(
            egui::RichText::new(label)
                .size(10.0)
                .color(P::INK3)
                .family(egui::FontFamily::Monospace),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add_space(8.0);
            let count_str = if total > 0 {
                format!("{count} / {total}")
            } else {
                count.to_string()
            };
            ui.label(
                egui::RichText::new(count_str)
                    .size(10.0)
                    .color(P::PEACH)
                    .family(egui::FontFamily::Monospace),
            );
        });
    });
}

fn tree_row(ui: &mut Ui, app: &App2, idx: usize) -> Option<TreeAction> {
    let path = app.batch_files[idx].path.clone();
    let status = app.batch_files[idx].status.clone();
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("?")
        .to_string();
    let status_label = status.badge_label().to_string();
    let face_count = status.face_count();

    let is_active = app.preview.image_path.as_deref() == Some(path.as_path());

    let row_h = 32.0;
    let (resp, painter) =
        ui.allocate_painter(Vec2::new(ui.available_width(), row_h), Sense::click());
    let r = resp.rect;

    let bg = if is_active {
        P::peach_alpha(25)
    } else if resp.hovered() {
        P::white_alpha(10)
    } else {
        egui::Color32::TRANSPARENT
    };
    if bg != egui::Color32::TRANSPARENT {
        painter.rect_filled(r, 5.0, bg);
    }

    let text_color = if is_active { P::PEACH } else { P::INK2 };

    // Index
    painter.text(
        egui::pos2(r.min.x + 20.0, r.center().y),
        egui::Align2::RIGHT_CENTER,
        format!("{:02}", idx + 1),
        egui::FontId::monospace(10.0),
        P::INK3,
    );
    // Person icon
    painter.text(
        egui::pos2(r.min.x + 34.0, r.center().y),
        egui::Align2::CENTER_CENTER,
        "◉",
        egui::FontId::proportional(11.0),
        P::INK3,
    );
    // Filename
    let avail_w = r.width() - 132.0;
    painter.text(
        egui::pos2(r.min.x + 46.0, r.center().y),
        egui::Align2::LEFT_CENTER,
        truncate(&name, avail_w, &egui::FontId::monospace(11.5), &painter),
        egui::FontId::monospace(11.5),
        text_color,
    );

    // Badge on the right
    let badge_label = if let Some(count) = face_count {
        count.to_string()
    } else {
        status_label.clone()
    };
    let (badge_bg, badge_fg) = crate::theme::badge_color(&status_label);
    let badge_font = egui::FontId::monospace(9.5);
    let badge_g = painter.layout_no_wrap(badge_label.clone(), badge_font, badge_fg);
    let bw = badge_g.size().x + 10.0;
    let bh = badge_g.size().y + 4.0;
    let remove_rect = egui::Rect::from_center_size(
        egui::pos2(r.max.x - 18.0, r.center().y),
        Vec2::new(24.0, 24.0),
    );
    let load_rect = egui::Rect::from_center_size(
        egui::pos2(r.max.x - 46.0, r.center().y),
        Vec2::new(24.0, 24.0),
    );
    let badge_rect = egui::Rect::from_min_size(
        egui::pos2(load_rect.min.x - bw - 8.0, r.center().y - bh / 2.0),
        Vec2::new(bw, bh),
    );
    painter.rect_filled(badge_rect, 3.0, badge_bg);
    painter.galley(badge_rect.min + Vec2::new(5.0, 2.0), badge_g, badge_fg);

    let load_clicked = row_icon_button(ui, load_rect, idx, "load", "▶", P::CYAN, "Load image");
    let remove_clicked = row_icon_button(
        ui,
        remove_rect,
        idx,
        "remove",
        "×",
        P::ROSE,
        "Remove from queue",
    );

    if remove_clicked {
        Some(TreeAction::Remove(idx))
    } else if load_clicked || resp.clicked() {
        Some(TreeAction::Load(path))
    } else {
        None
    }
}

fn row_icon_button(
    ui: &mut Ui,
    rect: egui::Rect,
    idx: usize,
    salt: &str,
    label: &str,
    color: egui::Color32,
    tooltip: &str,
) -> bool {
    let resp = ui.interact(rect, ui.id().with((salt, idx)), Sense::click());
    let bg = if resp.hovered() {
        P::white_alpha(20)
    } else {
        P::white_alpha(8)
    };
    ui.painter().rect_filled(rect, 5.0, bg);
    ui.painter().rect_stroke(
        rect,
        5.0,
        Stroke::new(1.0, P::white_alpha(24)),
        egui::StrokeKind::Outside,
    );
    ui.painter().text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        label,
        egui::FontId::proportional(12.0),
        color,
    );
    let clicked = resp.clicked();
    resp.on_hover_text(tooltip);
    clicked
}

fn truncate(s: &str, max_w: f32, _font: &egui::FontId, _painter: &egui::Painter) -> String {
    // Simple approximation — assume ~7px per char for monospace
    let chars_fit = (max_w / 7.0) as usize;
    if s.len() <= chars_fit {
        return s.to_string();
    }
    format!("{}…", &s[..chars_fit.saturating_sub(1)])
}

fn show_mapping(ui: &mut Ui, _app: &mut App2) {
    ui.add_space(12.0);
    ui.label(
        egui::RichText::new("Mapping import")
            .size(12.5)
            .color(P::INK2),
    );
    ui.add_space(4.0);
    ui.label(
        egui::RichText::new(
            "Load a CSV, Excel, or SQLite file to map source images to output filenames.",
        )
        .size(11.5)
        .color(P::INK3),
    );
}

fn show_history(ui: &mut Ui, app: &mut App2) {
    ui.add_space(12.0);
    if app.crop_history.is_empty() {
        ui.label(
            egui::RichText::new("No history yet.")
                .size(11.5)
                .color(P::INK3),
        );
        return;
    }
    for (i, _entry) in app.crop_history.iter().enumerate() {
        ui.label(
            egui::RichText::new(format!("State {}", i + 1))
                .size(11.5)
                .color(P::INK2),
        );
    }
}
