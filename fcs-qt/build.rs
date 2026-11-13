fn main() {
    // Qt's DLLs use the dynamic CRT. The workspace config links the static one, and a process
    // with both risks one heap freeing the other's memory (Qt's inline container code
    // allocates on our side). Config files cannot remove the workspace's `+crt-static` (rustc
    // lets it win over a later `-crt-static`); only RUSTFLAGS replaces it, which dev.ps1 sets.
    let msvc = std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc");
    let static_crt = std::env::var("CARGO_CFG_TARGET_FEATURE")
        .unwrap_or_default()
        .split(',')
        .any(|f| f == "crt-static");
    if msvc && static_crt {
        panic!(
            "fcs-qt must link the dynamic CRT, as Qt does. Build through fcs-qt/dev.ps1, or set \
             RUSTFLAGS=\"-C target-cpu=x86-64-v3\" (it replaces the workspace's +crt-static)."
        );
    }

    // The image provider (cpp/image_provider.cpp): qtbridge has none, and QML's Image cannot
    // load from memory without one.
    println!("cargo:rerun-if-changed=cpp/image_provider.cpp");
    let modules = ["Core", "Gui", "Qml", "Quick"].map(String::from).to_vec();
    let qt = qt_build_utils::QtBuild::new(modules).expect("Qt 6 with qmake on PATH (see dev.ps1)");
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .includes(qt.include_paths())
        .file("cpp/image_provider.cpp");
    for framework in qt.framework_paths() {
        build.flag(format!("-F{}", framework.display()));
    }
    if build.get_compiler().is_like_msvc() {
        // Qt 6 headers refuse MSVC's default `__cplusplus` of 199711L.
        build.flag("/Zc:__cplusplus").flag("/permissive-");
    }
    qt.cargo_link_libraries(&mut build);
    build.compile("fcs_qt_image_provider");
}
