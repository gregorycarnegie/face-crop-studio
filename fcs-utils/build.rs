fn main() {
    let compiler = wesl::Compiler::default();
    for shader in [
        "pixel_adjust",
        "gaussian_blur",
        "bilateral_filter",
        "background_blur",
        "shape_mask",
        "hist_equalize",
    ] {
        let source = format!("src/gpu/{shader}.wesl");
        let compiled = compiler
            .compile(&source)
            .unwrap_or_else(|err| panic!("failed to compile {source}: {err}"));
        compiled.emit_rerun_if_changed();
        compiled.write_artifact(shader);
    }
}
