fn main() {
    let compiler = wesl::Compiler::default();
    for (directory, shaders) in [
        (
            "src/gpu",
            &["activation", "add", "conv2d", "pool", "resize2x"][..],
        ),
        ("examples/shaders", &["encode_noop", "pack_half"][..]),
    ] {
        for shader in shaders {
            let source = format!("{directory}/{shader}.wesl");
            let compiled = compiler
                .compile(&source)
                .unwrap_or_else(|err| panic!("failed to compile {source}: {err}"));
            compiled.emit_rerun_if_changed();
            compiled.write_artifact(shader);
        }
    }
}
