//! Face Crop Studio on Qt Quick, through Qt Bridge for Rust. An experiment beside the egui
//! app (fcs-gui): same models, same settings file, a QML front end.

mod images;
mod pipeline;
mod studio;

use qtbridge::{QApp, include_bytes_qml};
use std::{ffi::c_void, pin::Pin};

unsafe extern "C" {
    /// cpp/image_provider.cpp: registers `image://fcs/` on the engine, which takes ownership.
    fn fcs_install_image_provider(engine: *mut c_void);
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    // Every control the app styles is built on Basic templates; pin it so a desktop style
    // never mixes in. An explicit QT_QUICK_CONTROLS_STYLE still wins.
    set_env_default("QT_QUICK_CONTROLS_STYLE", "Basic".as_ref());
    // vcpkg builds libheif with an empty plugin directory, and on Windows heif_init then loads
    // every DLL in the root of the current drive (C:\*.dll): anything planted there would run
    // in this process. Point it at a folder of our own unless the user chose one.
    if let Some(dir) = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join("libheif-plugins")))
    {
        set_env_default("LIBHEIF_PLUGIN_PATH", dir.as_os_str());
    }

    // The QML module lives at :/qt/qml/fcs_qt, the URI qtbridge registers Studio under.
    include_bytes_qml!("fcs_qt", "qt/qml");
    let mut app = QApp::new();
    app.application_name("Face Crop Studio")
        .register::<studio::Studio>()
        .add_import_path("qrc:/qt/qml");
    // qtbridge has no image provider API, but hands out its engine (`qml_engine`, doc-hidden),
    // which is all the C++ side needs. Installed before any QML loads.
    // SAFETY: the pointer is the live QQmlEngine, used only for the duration of the call.
    unsafe {
        let engine = Pin::into_inner_unchecked(app.qml_engine());
        fcs_install_image_provider(std::ptr::from_mut(engine).cast());
    }
    let code = app.load_qml_from_file("qrc:/qt/qml/fcs_qt/Main.qml").run();
    std::process::exit(code);
}

/// Set `name` for Qt and the C libraries unless it is already set. Call it before any other
/// thread starts.
fn set_env_default(name: &str, value: &std::ffi::OsStr) {
    if std::env::var_os(name).is_some() {
        return;
    }
    // Qt and libheif read variables through the C runtime's getenv, whose copy of the
    // environment `std::env::set_var` (SetEnvironmentVariableW) never updates. _wputenv_s
    // updates both, and the whole process shares the one dynamic UCRT (see build.rs).
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        unsafe extern "C" {
            fn _wputenv_s(name: *const u16, value: *const u16) -> i32;
        }
        let wide = |s: &std::ffi::OsStr| s.encode_wide().chain([0]).collect::<Vec<u16>>();
        let (name, value) = (wide(name.as_ref()), wide(value));
        // SAFETY: both are NUL-terminated, and no other thread is running yet.
        if unsafe { _wputenv_s(name.as_ptr(), value.as_ptr()) } != 0 {
            log::warn!("Failed to set {}", String::from_utf16_lossy(&name));
        }
    }
    #[cfg(not(windows))]
    // SAFETY: no other thread is running yet; glibc's getenv reads the same environment.
    unsafe {
        std::env::set_var(name, value)
    };
}
