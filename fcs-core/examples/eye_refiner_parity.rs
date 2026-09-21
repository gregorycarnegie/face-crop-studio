//! Does the exported eye refiner run under `fcs_ort` and agree with onnxruntime?
//!
//! The refiner was trained and exported in Python, and every number measured for it came
//! from a Python process. Before any of it is wired into the cropper, the graph has to
//! produce the same four values through the Rust path that will actually ship.
//!
//! The input is generated rather than loaded so both sides can build it from the same
//! formula with no fixture file to drift: `x[i] = sin(i * 0.01)` over a 1x3x112x112 tensor.
//! The reference was produced by onnxruntime 1.x in WSL:
//!
//! ```text
//! python -c "import numpy as np, onnxruntime as rt; ..."
//! output 0.32328194 0.43364963 0.52232397 0.44975832
//! ```
//!
//! These constants are tied to a specific export. When the model is re-exported they must be
//! recomputed -- and that is a feature: the first version of this example carried the previous
//! export's numbers and failed loudly the moment the model was replaced, which is how the
//! swap was noticed rather than shipped.
//!
//! Run with the runtime on hand, which is not discoverable by default on this machine:
//!
//! ```text
//! ORT_DYLIB_PATH=.../onnxruntime.dll cargo run -p fcs-core --example eye_refiner_parity
//! ```

use std::path::PathBuf;

/// Straight from the Python reference above.
const REFERENCE: [f32; 4] = [0.323_281_94, 0.433_649_63, 0.522_323_97, 0.449_758_32];

/// Generous next to the 1.8e-07 the export itself matched torch to; this is checking that
/// the graph is the same graph, not chasing the last ulp of a different BLAS.
const TOLERANCE: f32 = 1e-5;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("fcs-core sits in the workspace root")
        .join("models/eye_refiner.onnx");
    if !model.exists() {
        return Err(format!(
            "model not found at {}; the refiner export has to be in models/",
            model.display()
        )
        .into());
    }

    let environment = fcs_ort::Environment::shared()
        .ok_or("no compatible ONNX Runtime found; set ORT_DYLIB_PATH to an onnxruntime library")?;
    println!(
        "ONNX Runtime {} ({})",
        environment.runtime().version(),
        environment.runtime().path().display()
    );

    let session = fcs_ort::Session::new(&environment, &model, fcs_ort::SessionOptions::default())?;
    println!(
        "inputs {:?} -> outputs {:?}",
        session.input_names(),
        session.output_names()
    );

    let shape = [1usize, 3, 112, 112];
    let input: Vec<f32> = (0..shape.iter().product::<usize>())
        .map(|i| (i as f64 * 0.01).sin() as f32)
        .collect();

    let outputs = session.run(&input, &shape)?;
    let eyes = outputs
        .first()
        .ok_or("the model returned no output tensor")?;
    println!("output shape {:?}", eyes.shape);
    println!(
        "rust      {}",
        eyes.data
            .iter()
            .map(|v| format!("{v:.8}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    println!(
        "reference {}",
        REFERENCE
            .iter()
            .map(|v| format!("{v:.8}"))
            .collect::<Vec<_>>()
            .join(" ")
    );

    if eyes.data.len() != REFERENCE.len() {
        return Err(format!(
            "expected {} values, got {}",
            REFERENCE.len(),
            eyes.data.len()
        )
        .into());
    }

    let worst = eyes
        .data
        .iter()
        .zip(REFERENCE)
        .map(|(got, want)| (got - want).abs())
        .fold(0.0f32, f32::max);
    println!("worst absolute difference {worst:.3e} (tolerance {TOLERANCE:.0e})");

    if worst > TOLERANCE {
        return Err(format!("the Rust path disagrees with onnxruntime by {worst:.3e}").into());
    }
    println!("PARITY OK (graph)");

    check_real_image()?;
    Ok(())
}

/// The second half: does `EyeRefiner`'s own preprocessing match Python's?
///
/// The graph check above feeds a synthetic tensor, so it proves the weights are the same and
/// nothing about how a face becomes a tensor. That is the part with room to be quietly wrong:
/// `cv2.warpAffine` with `INTER_LINEAR`, a constant zero border, BGR to RGB, and the crop box
/// expanded by 1.25 are four independent conventions, and getting any of them wrong shifts
/// every predicted eye without failing anything.
///
/// So this runs a real photograph through the shipping path and compares against the same
/// image and box run through `train_eye_refiner.py`'s own preprocessing:
///
/// ```text
/// eyes in image coords: 920.4756 767.1260  1184.2014 696.8716
/// ```
///
/// Skipped with a printed note when the image is absent, so the example still runs anywhere.
fn check_real_image() -> Result<(), Box<dyn std::error::Error>> {
    use fcs_core::{BoundingBox, Detection, EyeRefiner};

    const IMAGE: &str = r"C:\Users\grego\Downloads\VinaSkyy\1.jpg";
    const BOX: [f32; 4] = [804.6168, 427.279_45, 535.7621, 807.3744];
    /// From the Python run quoted above.
    const PYTHON_EYES: [[f32; 2]; 2] = [[920.4756, 767.126], [1184.2014, 696.8716]];
    /// A pixel of slack in a 1341-pixel-wide face: `cv2` interpolates in fixed point and this
    /// samples in f32, so exact equality is not the claim. A convention mismatch misses by
    /// tens of pixels, not by one.
    const PIXEL_TOLERANCE: f32 = 1.0;

    let image_path = std::path::Path::new(IMAGE);
    if !image_path.exists() {
        println!("skipping the real-image check: {IMAGE} is not on this machine");
        return Ok(());
    }
    let Some(refiner) = EyeRefiner::load() else {
        return Err("EyeRefiner::load() returned None despite the runtime being present".into());
    };

    let image = image::open(image_path)?;
    let mut detections = vec![Detection {
        bbox: BoundingBox {
            x: BOX[0],
            y: BOX[1],
            width: BOX[2],
            height: BOX[3],
        },
        landmarks: [None; 5],
        score: 1.0,
    }];

    let refined = refiner.refine(&image, &mut detections);
    if refined != 1 {
        return Err(format!("expected 1 refined detection, got {refined}").into());
    }

    let mut worst = 0.0f32;
    for (index, expected) in PYTHON_EYES.iter().enumerate() {
        let got = detections[0].landmarks[index]
            .ok_or_else(|| format!("the refiner left eye {index} absent"))?;
        println!(
            "eye {index}: rust {:9.4} {:9.4}   python {:9.4} {:9.4}",
            got.x, got.y, expected[0], expected[1]
        );
        worst = worst.max((got.x - expected[0]).abs());
        worst = worst.max((got.y - expected[1]).abs());
    }
    println!("worst coordinate difference {worst:.4} px (tolerance {PIXEL_TOLERANCE})");

    if worst > PIXEL_TOLERANCE {
        return Err(format!(
            "Rust preprocessing disagrees with Python by {worst:.4} px; \
             the crop is not being built the way the model was trained"
        )
        .into());
    }
    println!("PARITY OK (preprocessing)");
    Ok(())
}
