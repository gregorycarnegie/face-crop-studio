use super::*;

fn any_box() -> BoundingBox {
    BoundingBox {
        x: kani::any(),
        y: kani::any(),
        width: kani::any(),
        height: kani::any(),
    }
}

/// Every field within 1e18, so no area or edge overflows f32 -- far beyond any image. Past it,
/// overflowing areas make `union` an intermediate NaN that `iou` turns into 0.
fn bounded(b: &BoundingBox) -> bool {
    const LIMIT: f32 = 1e18;
    [b.x, b.y, b.width, b.height]
        .iter()
        .all(|v| v.abs() <= LIMIT)
}

/// The condition under which `iou` computes a ratio rather than returning zero.
fn overlap(a: &BoundingBox, b: &BoundingBox) -> bool {
    a.x.max(b.x) < a.right().min(b.right()) && a.y.max(b.y) < a.bottom().min(b.bottom())
}

/// For every pair of bounded boxes, IoU computes no NaN at any step, is non-negative, and is
/// positive only for overlapping boxes. It is not bounded by 1: rounding `x + width` can inflate
/// the intersection past either area.
#[kani::proof]
#[kani::solver(cadical)]
fn iou_is_nonnegative_and_positive_only_on_overlap() {
    let (a, b) = (any_box(), any_box());
    kani::assume(bounded(&a) && bounded(&b));
    let v = a.iou(&b);
    assert!(!v.is_nan());
    assert!(v >= 0.0);
    if v > 0.0 {
        assert!(overlap(&a, &b));
    }
}

/// Offsets from the scene origin keep the order of finite coordinates (IEEE rounding is
/// monotone).
#[kani::proof]
fn subtracting_the_scene_origin_preserves_order() {
    let (lo, hi, origin): (f32, f32, f32) = (kani::any(), kani::any(), kani::any());
    kani::assume(lo.is_finite() && hi.is_finite() && origin.is_finite() && lo <= hi);
    assert!(lo - origin <= hi - origin);
}

/// A later position never maps to an earlier cell, and every cell is on the grid. With IEEE
/// division being monotone in its numerator for a positive divisor, `grid_cell_index` inherits
/// this; with the order lemma above, overlapping boxes in the scene bounds get intersecting cell
/// ranges, so the grid compares every pair the naive search could suppress.
#[kani::proof]
fn cell_of_is_monotone_and_in_range() {
    let (lo, hi): (f32, f32) = (kani::any(), kani::any());
    kani::assume(!lo.is_nan() && !hi.is_nan() && lo <= hi);
    let grid_size: usize = kani::any();
    kani::assume((1..=NMS_GRID_SIZE).contains(&grid_size));
    let (a, b) = (cell_of(lo, grid_size), cell_of(hi, grid_size));
    assert!(a <= b && b < grid_size);
}
