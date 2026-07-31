// Two guards, and both are load-bearing:
//
//   `cfg(windows)`      — build scripts compile for the *host*, and winresource
//                         shells out to the Windows resource compiler, so it
//                         cannot run anywhere else.
//   CARGO_CFG_TARGET_OS — resources only belong in a Windows binary, so a
//                         Windows host cross-compiling to Linux must skip it.
//
// Written as one `if` rather than an early return: on a non-Windows host the
// `cfg` strips the whole statement, and a bare trailing `return;` in the
// remaining empty body trips clippy::needless_return. That fires only on the
// macOS and Linux CI legs — a local `cargo clippy` on Windows cannot see it,
// because build scripts are always compiled for the host.
fn main() {
    #[cfg(windows)]
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("assets/app_icon.ico");
        if let Err(err) = res.compile() {
            panic!("failed to compile Windows resources for fcs-gui: {err}");
        }
    }
}
