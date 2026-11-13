// Qt's DLLs use the dynamic CRT. The workspace config links the static one, and a process
// with both risks one heap freeing the other's memory (Qt's inline container code allocates
// on our side). Config files cannot remove the workspace's `+crt-static` (rustc lets it win
// over a later `-crt-static`); only RUSTFLAGS replaces it, which dev.ps1 sets.
fn main() {
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
}
