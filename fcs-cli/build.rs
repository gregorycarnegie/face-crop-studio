// Mirrors fcs-gui/build.rs — see the note there on why both guards are needed.
// This one previously checked only `cfg(windows)` (the host), so a Windows host
// cross-compiling to Linux would still have tried to embed Windows resources.
fn main() {
    #[cfg(windows)]
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("../fcs-gui/assets/app_icon.ico");
        if let Err(err) = res.compile() {
            panic!("failed to compile Windows resources for fcs-cli: {err}");
        }
    }
}
