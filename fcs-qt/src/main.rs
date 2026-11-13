//! Face Crop Studio on Qt Quick, through Qt Bridge for Rust. An experiment beside the egui
//! app (fcs-gui): same models, same settings file, a QML front end.

mod pipeline;
mod studio;

use qtbridge::{QApp, include_bytes_qml};

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    // Every control the app styles is built on Basic templates; pin it so a desktop style
    // never mixes in. An explicit QT_QUICK_CONTROLS_STYLE still wins.
    if std::env::var_os("QT_QUICK_CONTROLS_STYLE").is_none() {
        // SAFETY: first thing in main, before Qt or any other thread starts.
        unsafe { std::env::set_var("QT_QUICK_CONTROLS_STYLE", "Basic") };
    }

    // The QML module lives at :/qt/qml/fcs_qt, the URI qtbridge registers Studio under.
    include_bytes_qml!("fcs_qt", "qt/qml");
    let code = QApp::new()
        .application_name("Face Crop Studio")
        .register::<studio::Studio>()
        .add_import_path("qrc:/qt/qml")
        .load_qml_from_file("qrc:/qt/qml/fcs_qt/Main.qml")
        .run();
    std::process::exit(code);
}
