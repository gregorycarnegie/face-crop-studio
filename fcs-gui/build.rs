fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }

    #[cfg(windows)]
    {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("assets/app_icon.ico");
        if let Err(err) = res.compile() {
            panic!("failed to compile Windows resources for fcs-gui: {err}");
        }
    }
}
