//! The whole window, driven through its accessibility tree.
//!
//! `ui/`, `core/` and `rendering/` are excluded from mutation testing because their painting only
//! exists inside a live frame. These tests run the real `App2` frame by frame and check what a
//! person would: every panel is there, the tabs swap their content, and the menus and shortcuts
//! change the state they claim to. They use no images, so they behave the same on WARP, Metal,
//! lavapipe and a real GPU; colours and pixel layout stay untested here.

use super::*;
use egui::accesskit::Role;
use egui_kittest::{Harness, kittest::Queryable};

/// The app on default settings, saving into `dir` rather than the user's own settings file.
fn launch(dir: &tempfile::TempDir) -> Harness<'static, App2> {
    let path = dir.path().join("gui_settings.json");
    let mut harness = Harness::builder()
        .with_size(egui::vec2(1400.0, 900.0))
        .build_eframe(move |cc| App2::for_tests(cc, path.clone()));
    harness.run();
    harness
}

fn click_button(harness: &mut Harness<'_, App2>, label: &str) {
    harness.get_by_role_and_label(Role::Button, label).click();
    harness.run();
}

#[test]
fn every_panel_is_on_screen_at_launch() {
    let dir = tempfile::tempdir().unwrap();
    let harness = launch(&dir);

    for label in ["File", "Edit", "View", "Help"] {
        harness.get_by_role_and_label(Role::Button, label);
    }
    for label in ["Queue", "Mapping", "History", "Crop", "Output", "Enhance"] {
        harness.get_by_role_and_label(Role::Button, label);
    }
}

/// The status bar shows the app's own message for each state. It once said "Ready" whatever
/// happened, while every message the app wrote went nowhere.
#[test]
fn the_status_bar_shows_what_the_app_is_doing() {
    let dir = tempfile::tempdir().unwrap();
    let mut harness = launch(&dir);
    harness.get_by_label(STATUS_DETECTOR_PENDING);
    assert!(harness.query_by_label("Ready").is_none());

    // What `poll_detector` does once the background build reports in.
    harness.state_mut().detector_rx = None;
    harness.state_mut().show_success("Detected 3 face(s)");
    harness.run();
    harness.get_by_label("Detected 3 face(s)");

    harness.state_mut().show_error("Export failed", "disk full");
    harness.run();
    harness.get_by_label("Export failed: disk full");

    harness.state_mut().is_busy = true;
    harness.run_steps(2);
    harness.get_by_label("Running");
    assert!(harness.query_by_label("Export failed: disk full").is_none());
}

#[test]
fn the_status_bar_names_the_model_without_onnx() {
    let dir = tempfile::tempdir().unwrap();
    let harness = launch(&dir);
    harness.get_by_label("scrfd80k_500m_640");
    assert!(harness.query_by_label_contains("ONNX").is_none());
}

#[test]
fn the_inspector_tabs_swap_their_controls() {
    let dir = tempfile::tempdir().unwrap();
    let mut harness = launch(&dir);
    harness.get_by_label("Preset");
    assert!(harness.query_by_label("Brightness").is_none());

    click_button(&mut harness, "Enhance");
    assert!(harness.state().inspector_tab == InspectorTab::Enhance);
    assert!(harness.query_all_by_label("Brightness").next().is_some());
    assert!(harness.query_by_label("Preset").is_none());

    click_button(&mut harness, "Output");
    assert!(harness.state().inspector_tab == InspectorTab::Output);
    harness.get_by_label("Format");
    assert!(harness.query_all_by_label("Brightness").next().is_none());
}

#[test]
fn the_sidebar_tabs_swap_their_panels() {
    let dir = tempfile::tempdir().unwrap();
    let mut harness = launch(&dir);
    assert!(harness.query_by_label("Take snapshot").is_none());

    click_button(&mut harness, "History");
    assert!(harness.state().sidebar_tab == SidebarTab::History);
    harness.get_by_role_and_label(Role::Button, "Take snapshot");

    click_button(&mut harness, "Mapping");
    assert!(harness.state().sidebar_tab == SidebarTab::Mapping);
    assert!(harness.query_by_label("Take snapshot").is_none());
}

#[test]
fn a_history_snapshot_records_the_crop_settings() {
    let dir = tempfile::tempdir().unwrap();
    let mut harness = launch(&dir);
    click_button(&mut harness, "History");
    assert_eq!(harness.state().crop_history.len(), 1);

    // Unchanged settings are not snapshotted twice.
    click_button(&mut harness, "Take snapshot");
    assert_eq!(harness.state().crop_history.len(), 1);

    harness.state_mut().settings.crop.output_width += 10;
    click_button(&mut harness, "Take snapshot");
    // The rows are painted, so the history itself is the check.
    let history = &harness.state().crop_history;
    assert_eq!(history.len(), 2);
    assert_eq!(history[1].output_width, history[0].output_width + 10);
}

#[test]
fn help_opens_the_about_window() {
    let dir = tempfile::tempdir().unwrap();
    let mut harness = launch(&dir);
    assert!(!harness.state().show_about);

    click_button(&mut harness, "Help");
    click_button(&mut harness, "About Face Crop Studio");
    assert!(harness.state().show_about);
    harness.get_by_label(concat!("v", env!("CARGO_PKG_VERSION")));
}

#[test]
fn the_view_menu_hides_the_crop_overlay() {
    let dir = tempfile::tempdir().unwrap();
    let mut harness = launch(&dir);
    assert!(harness.state().show_crop_overlay);

    click_button(&mut harness, "View");
    harness.get_by_label("Show Crop Overlay").click();
    harness.run();
    assert!(!harness.state().show_crop_overlay);
}

#[test]
fn save_settings_writes_the_settings_file() {
    let dir = tempfile::tempdir().unwrap();
    let mut harness = launch(&dir);
    let path = dir.path().join("gui_settings.json");
    assert!(!path.exists());

    click_button(&mut harness, "File");
    click_button(&mut harness, "Save Settings");
    let saved = fcs_utils::config::AppSettings::load_from_path(&path).expect("settings saved");
    assert_eq!(
        saved.crop.output_width,
        harness.state().settings.crop.output_width
    );
}

#[test]
fn space_toggles_the_result_preview() {
    let dir = tempfile::tempdir().unwrap();
    let mut harness = launch(&dir);
    let before = harness.state().result_preview.enabled;

    harness.key_press(egui::Key::Space);
    harness.run();
    assert_eq!(harness.state().result_preview.enabled, !before);
}
