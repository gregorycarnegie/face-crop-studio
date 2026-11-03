use crate::decode::{LANDMARKS, Letterbox, TRAINED_LANDMARKS, decode_row, sigmoid};

/// Kani leaves `expf` unconstrained, so this states the contract the activation relies on:
/// finite and non-negative, exactly 1 at zero, at most 1 below and at least 1 above. The result is
/// checked against that contract; the libm `expf` meeting it is trusted. Overflow to infinity
/// (logits below about -88.7) is covered by a unit test in `decode.rs`: Kani's NaN check flags
/// `1 / inf`, which IEEE defines as 0.
fn exp_contract(x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= 0.0);
    kani::assume(if x == 0.0 {
        r == 1.0
    } else if x < 0.0 {
        r <= 1.0
    } else {
        r >= 1.0
    });
    r
}

/// The score activation is a probability for every non-NaN logit, above one half only for
/// positive logits and at least one half for every positive one. Not "exactly when": logits in
/// (0, ~1e-7) round to exactly 0.5.
#[kani::proof]
#[kani::stub(f32::exp, exp_contract)]
fn sigmoid_is_a_probability_split_at_zero() {
    let value: f32 = kani::any();
    kani::assume(!value.is_nan());
    let p = sigmoid(value);
    assert!((0.0..=1.0).contains(&p));
    if p > 0.5 {
        assert!(value > 0.0);
    }
    if value > 0.0 {
        assert!(p >= 0.5);
    }
}

/// One head row, at any of the three levels, with raw outputs and letterbox scale in the ranges
/// the detector produces: the box and both eye landmarks are finite, the box is non-negative
/// whenever its distances are, and the untrained landmarks stay absent.
#[kani::proof]
#[kani::solver(cadical)]
fn decode_row_is_finite_and_ordered() {
    let level: usize = kani::any();
    kani::assume(level < 3);
    let (cells, stride) = [(80, 8.0), (40, 16.0), (20, 32.0)][level];
    let row: usize = kani::any();
    kani::assume(row < cells * cells * 2);
    let boxes: [f32; 4] = kani::any();
    let points: [f32; TRAINED_LANDMARKS * 2] = kani::any();
    kani::assume(boxes.iter().chain(&points).all(|v| v.abs() <= 1e4));
    let scale: f32 = kani::any();
    kani::assume((1e-3..=1e3).contains(&scale));

    let d = decode_row(
        0.5,
        &boxes,
        &points,
        row,
        cells,
        stride,
        Letterbox { scale },
    );
    let b = d.bbox;
    assert!(b.x.is_finite() && b.y.is_finite() && b.width.is_finite() && b.height.is_finite());
    if boxes.iter().all(|&v| v >= 0.0) {
        assert!(b.width >= 0.0 && b.height >= 0.0);
    }
    for (index, landmark) in d.landmarks.iter().enumerate().take(LANDMARKS) {
        match landmark {
            Some(p) => assert!(index < TRAINED_LANDMARKS && p.x.is_finite() && p.y.is_finite()),
            None => assert!(index >= TRAINED_LANDMARKS),
        }
    }
}
