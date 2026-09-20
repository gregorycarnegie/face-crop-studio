//! Exercise the whole binding against a real ONNX Runtime.
//!
//! This is the test that matters most in this crate. `sys::OrtApi` is a
//! hand-maintained prefix of a 424-entry function table, and a wrong offset is
//! undefined behaviour rather than a compile error — the const size assertion
//! catches a wrong *type*, but only calling through the table catches a wrong
//! *position*. The calls below span offsets 3 to 100, so a shift anywhere in
//! the declared prefix shows up here as a crash or nonsense rather than in
//! production.
//!
//! Skips when no compatible runtime is installed, unless FCS_STRICT_TESTS is
//! set, matching the convention used for the GPU tests.

use std::path::PathBuf;

use fcs_ort::{Environment, Session, SessionOptions};

const MODEL: &str = "../models/scrfd80k_500m_640.onnx";

fn model_path() -> Option<PathBuf> {
    let path = PathBuf::from(MODEL);
    path.exists().then_some(path)
}

fn environment() -> Option<std::sync::Arc<Environment>> {
    let strict = std::env::var("FCS_STRICT_TESTS").is_ok();
    match Environment::load() {
        Some(env) => Some(env),
        None => {
            assert!(!strict, "no ONNX Runtime available under FCS_STRICT_TESTS");
            eprintln!("skipping: no compatible ONNX Runtime");
            None
        }
    }
}

#[test]
fn loads_a_model_and_reads_its_signature() {
    let (Some(env), Some(model)) = (environment(), model_path()) else {
        return;
    };
    let session =
        Session::new(&env, &model, SessionOptions::default()).expect("session should open");

    // SCRFD takes one input and emits cls/bbox/kps for strides 8/16/32.
    assert_eq!(session.input_names().len(), 1);
    assert_eq!(session.output_names().len(), 9);
    assert!(
        !session.input_names()[0].as_bytes().is_empty(),
        "input name must be read from the model"
    );
    assert!(
        Environment::shared().is_some(),
        "a runtime loaded, so the shared environment must too"
    );
    assert!(env.runtime().version().starts_with("1."));
    // Environment's Debug nests Runtime's, which carries the version.
    assert!(format!("{env:?}").contains("version"), "{env:?}");
    assert!(format!("{session:?}").contains("inputs"), "{session:?}");
    eprintln!(
        "runtime {} at {}",
        env.runtime().version(),
        env.runtime().path().display()
    );
}

#[test]
fn runs_the_graph_and_returns_well_formed_outputs() {
    let (Some(env), Some(model)) = (environment(), model_path()) else {
        return;
    };
    let session = Session::new(&env, &model, SessionOptions::default()).expect("session");

    let shape = [1usize, 3, 640, 640];
    let input = vec![0.5f32; shape.iter().product()];
    let outputs = session.run(&input, &shape).expect("run should succeed");

    assert_eq!(outputs.len(), 9);
    // Grid cells per stride for a 640x640 input -- 80x80, 40x40, 20x20 -- times the two anchors
    // per cell. The nine outputs come grouped by kind rather than by stride: the three class
    // maps first, then the three box maps, then the three keypoint maps.
    let expected_rows = [12800usize, 3200, 800];
    for (i, out) in outputs.iter().enumerate() {
        let rows = expected_rows[i % 3];
        assert_eq!(
            out.data.len(),
            out.shape.iter().product::<usize>(),
            "output {i}: data length disagrees with shape {:?}",
            out.shape
        );
        assert!(
            out.shape.contains(&rows),
            "output {i} shape {:?} has no {rows}-cell dimension",
            out.shape
        );
        assert!(
            out.data.iter().all(|v| v.is_finite()),
            "output {i} contains non-finite values"
        );
    }
}

/// A shape that disagrees with the data length must be refused rather than
/// handed to the runtime, which would read past the end of the buffer.
#[test]
fn a_mismatched_shape_is_rejected() {
    let (Some(env), Some(model)) = (environment(), model_path()) else {
        return;
    };
    let session = Session::new(&env, &model, SessionOptions::default()).expect("session");
    let err = session
        .run(&[0.0f32; 10], &[1, 3, 640, 640])
        .expect_err("a short buffer must be rejected");
    assert!(format!("{err}").contains("needs"), "unhelpful error: {err}");
}

#[test]
fn a_missing_model_is_an_error_not_a_crash() {
    let Some(env) = environment() else { return };
    let err = Session::new(
        &env,
        std::path::Path::new("../models/does-not-exist.onnx"),
        SessionOptions::default(),
    )
    .expect_err("a missing model must not open");
    eprintln!("missing model reported as: {err}");
}

/// ONNX Runtime documents concurrent `Run` on one session as safe, which is why
/// [`Session::run`] takes `&self` and no pool is needed. Documentation is not
/// evidence, so this runs the same input from many threads and requires every
/// result to match the sequential one bit for bit.
#[test]
fn concurrent_runs_match_sequential() {
    let (Some(env), Some(model)) = (environment(), model_path()) else {
        return;
    };
    let session = Session::new(&env, &model, SessionOptions::default()).expect("session");

    let shape = [1usize, 3, 640, 640];
    let input: Vec<f32> = (0..shape.iter().product::<usize>())
        .map(|i| (i % 255) as f32)
        .collect();

    let baseline = session.run(&input, &shape).expect("sequential run");

    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..8)
            .map(|_| scope.spawn(|| session.run(&input, &shape).expect("concurrent run")))
            .collect();
        for handle in handles {
            let got = handle.join().expect("thread panicked");
            assert_eq!(got.len(), baseline.len());
            for (i, (a, b)) in got.iter().zip(baseline.iter()).enumerate() {
                assert_eq!(a.shape, b.shape, "output {i} shape diverged under threads");
                assert_eq!(
                    a.data, b.data,
                    "output {i} values diverged under concurrent runs"
                );
            }
        }
    });
}
