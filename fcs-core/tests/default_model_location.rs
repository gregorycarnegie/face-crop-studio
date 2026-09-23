//! The default model location, which only `load()` resolves.
//!
//! Every other test loads a model by explicit path, so replacing `ScrfdDetector::load` or
//! `EyeRefiner::load` with `None` survived: the one call the app makes at startup was the one
//! call no test made. `load` resolves `models/...` against the working directory first, and
//! under `cargo test` that is the crate root, which has no `models/`. Moving it is
//! process-global -- which is why this is a test binary of its own holding exactly one test.
//! There is nothing else in this process for the change to race with.

use fcs_core::{EyeRefiner, ScrfdDetector};
use std::path::Path;

/// Fail instead of skipping, for CI.
fn strict() -> bool {
    std::env::var("FCS_STRICT_TESTS").is_ok_and(|v| v != "0" && !v.is_empty())
}

#[test]
fn both_models_load_from_the_default_location() {
    // The workspace root, which is where `cargo run` starts the app from.
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("fcs-core sits in the workspace root");
    std::env::set_current_dir(root).expect("move to the workspace root");

    // The prerequisites, checked separately from the loads: skipping because a model or the
    // runtime is absent is fine outside CI, but skipping because `load` itself returned `None`
    // would hide exactly the failure this test is for.
    let models = Path::new("models").is_dir();
    let runtime = fcs_ort::Environment::shared().is_some();
    if !(models && runtime) {
        assert!(
            !strict(),
            "FCS_STRICT_TESTS: models/ present: {models}, ONNX Runtime present: {runtime}"
        );
        eprintln!("skipped: models/ present: {models}, ONNX Runtime present: {runtime}");
        return;
    }

    assert!(
        ScrfdDetector::load().is_some(),
        "SCRFD did not load from the default location"
    );
    assert!(
        EyeRefiner::load().is_some(),
        "the eye refiner did not load from the default location"
    );
}
