use crate::postprocess::{BoundingBox, Detection};

/// Spatial grid resolution used by the optimized NMS path.
const NMS_GRID_SIZE: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq)]
struct SceneBounds {
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
}

impl SceneBounds {
    fn width(&self) -> f32 {
        self.max_x - self.min_x
    }

    fn height(&self) -> f32 {
        self.max_y - self.min_y
    }

    fn cell_range_for_bbox(&self, bbox: &BoundingBox, grid_size: usize) -> CellRange {
        let cell_w = self.width() / grid_size as f32;
        let cell_h = self.height() / grid_size as f32;
        CellRange {
            min_col: grid_cell_index(bbox.x - self.min_x, cell_w, grid_size),
            max_col: grid_cell_index(bbox.right() - self.min_x, cell_w, grid_size),
            min_row: grid_cell_index(bbox.y - self.min_y, cell_h, grid_size),
            max_row: grid_cell_index(bbox.bottom() - self.min_y, cell_h, grid_size),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CellRange {
    min_col: usize,
    max_col: usize,
    min_row: usize,
    max_row: usize,
}

struct SpatialGrid {
    grid_size: usize,
    bounds: SceneBounds,
    cells: Vec<Vec<usize>>,
}

fn compute_scene_bounds(detections: &[Detection]) -> Option<SceneBounds> {
    if detections.is_empty() {
        return None;
    }

    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;

    for detection in detections {
        let bbox = detection.bbox;
        min_x = min_x.min(bbox.x);
        min_y = min_y.min(bbox.y);
        max_x = max_x.max(bbox.right());
        max_y = max_y.max(bbox.bottom());
    }

    let bounds = SceneBounds {
        min_x,
        min_y,
        max_x,
        max_y,
    };

    if bounds.width() <= f32::EPSILON || bounds.height() <= f32::EPSILON {
        None
    } else {
        Some(bounds)
    }
}

fn grid_cell_index(offset: f32, cell_d: f32, grid_size: usize) -> usize {
    if cell_d <= f32::EPSILON {
        return 0;
    }

    (offset / cell_d).floor().clamp(0.0, (grid_size - 1) as f32) as usize
}

/// Grid resolution for this particular scene, at most [`NMS_GRID_SIZE`].
///
/// A box is inserted into every cell it touches, so the build costs
/// `N * cells_per_box`. With a fixed resolution that term explodes exactly where NMS matters
/// most: a tight cluster of faces has small scene bounds, so 32x32 cells are far smaller than
/// the boxes, and each box lands in hundreds of them. Measured on 5000 clustered detections
/// the build alone was 23 ms, against 0.2 ms for the same count spread out -- the search was
/// never the problem (experiment 58).
///
/// Sizing cells to the typical box instead keeps `cells_per_box` near 1 either way. The grid
/// is only an acceleration structure, so this changes speed and not results.
fn grid_size_for(detections: &[Detection], bounds: SceneBounds) -> usize {
    // Mean rather than median: one pass, and NMS input is one object class at one rough scale,
    // so the two agree closely enough to size a cell by.
    let mut total = 0.0f32;
    for d in detections {
        total += d.bbox.width.max(d.bbox.height);
    }
    let mean_extent = total / detections.len() as f32;
    // Zero-area or non-finite boxes leave nothing to size a cell by; fall back to the cap.
    if !mean_extent.is_finite() || mean_extent <= 0.0 {
        return NMS_GRID_SIZE;
    }
    let longest = bounds.width().max(bounds.height());
    // 0 and NaN cast to 0 and clamp to one cell; +inf (subnormal boxes) saturates to the cap.
    ((longest / mean_extent).ceil() as usize).clamp(1, NMS_GRID_SIZE)
}

fn build_spatial_grid(detections: &[Detection], bounds: SceneBounds) -> SpatialGrid {
    let grid_size = grid_size_for(detections, bounds);
    let mut cells: Vec<Vec<usize>> = (0..grid_size * grid_size)
        .map(|_| Vec::with_capacity(detections.len() / (grid_size * grid_size / 4).max(1)))
        .collect();

    for (i, detection) in detections.iter().enumerate() {
        let range = bounds.cell_range_for_bbox(&detection.bbox, grid_size);
        for row in range.min_row..=range.max_row {
            let row_offset = row * grid_size;
            for col in range.min_col..=range.max_col {
                cells[row_offset + col].push(i);
            }
        }
    }

    SpatialGrid {
        grid_size,
        bounds,
        cells,
    }
}

fn suppress_overlapping_candidates(
    detections: &[Detection],
    threshold: f32,
    grid: &SpatialGrid,
) -> Vec<bool> {
    let len = detections.len();
    let mut suppressed = vec![false; len];
    let mut check_token = vec![usize::MAX; len];

    for i in 0..len {
        if suppressed[i] {
            continue;
        }

        let bbox = detections[i].bbox;
        let range = grid.bounds.cell_range_for_bbox(&bbox, grid.grid_size);

        for row in range.min_row..=range.max_row {
            let row_offset = row * grid.grid_size;
            for col in range.min_col..=range.max_col {
                let cell = &grid.cells[row_offset + col];
                for &candidate in cell {
                    if candidate > i && !suppressed[candidate] && check_token[candidate] != i {
                        check_token[candidate] = i;
                        if bbox.iou(&detections[candidate].bbox) > threshold {
                            suppressed[candidate] = true;
                        }
                    }
                }
            }
        }
    }

    suppressed
}

fn compact_unsuppressed_detections(detections: &mut Vec<Detection>, suppressed: &[bool]) {
    let mut keep = 0;
    for (i, &is_suppressed) in suppressed.iter().enumerate() {
        if !is_suppressed {
            if i != keep {
                detections.swap(i, keep);
            }
            keep += 1;
        }
    }
    detections.truncate(keep);
}

pub(crate) fn apply_nms_in_place(detections: &mut Vec<Detection>, threshold: f32) {
    let len = detections.len();
    if len <= 1 {
        return;
    }

    // For small datasets, the overhead of building the grid outweighs the benefit.
    // The break-even point is typically around 100-200 items.
    if len < 200 {
        apply_nms_naive(detections, threshold);
        return;
    }

    let Some(bounds) = compute_scene_bounds(detections) else {
        apply_nms_naive(detections, threshold);
        return;
    };

    let grid = build_spatial_grid(detections, bounds);
    let suppressed = suppress_overlapping_candidates(detections, threshold, &grid);
    compact_unsuppressed_detections(detections, &suppressed);
}

fn apply_nms_naive(detections: &mut Vec<Detection>, threshold: f32) {
    let len = detections.len();
    let mut suppressed = vec![false; len];
    let mut keep = 0;

    for i in 0..len {
        if suppressed[i] {
            continue;
        }

        if keep != i {
            detections.swap(keep, i);
            suppressed.swap(keep, i);
        }

        let reference_bbox = detections[keep].bbox;
        for j in (keep + 1)..len {
            if !suppressed[j] && reference_bbox.iou(&detections[j].bbox) > threshold {
                suppressed[j] = true;
            }
        }

        keep += 1;
    }

    detections.truncate(keep);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::postprocess::{BoundingBox, Detection, Landmark};
    use proptest::prelude::*;

    fn detection_with_score(score: f32, bbox: BoundingBox) -> Detection {
        Detection {
            bbox,
            landmarks: [Landmark::new(0.0, 0.0); 5],
            score,
        }
    }

    fn bbox(x: f32, y: f32, width: f32, height: f32) -> BoundingBox {
        BoundingBox {
            x,
            y,
            width,
            height,
        }
    }

    // ------------------------------------------------------------------
    // Grid geometry.
    //
    // The property test below compares the grid path against the naive one,
    // which is a strong check — but it cannot see an error the two paths
    // share, and under 200 detections both *are* the naive path. These pin
    // the grid helpers directly.

    #[test]
    fn scene_bounds_span_is_max_minus_min() {
        let bounds = SceneBounds {
            min_x: 2.0,
            min_y: 5.0,
            max_x: 10.0,
            max_y: 25.0,
        };
        // Deliberately unequal so a width/height swap shows up.
        assert_eq!(bounds.width(), 8.0);
        assert_eq!(bounds.height(), 20.0);
    }

    #[test]
    fn compute_scene_bounds_rejects_either_axis_collapsing() {
        // A row of zero-width boxes still spans vertically, so the guard has
        // to fire on either axis rather than needing both to collapse.
        let flat_x = vec![
            detection_with_score(0.9, bbox(4.0, 0.0, 0.0, 10.0)),
            detection_with_score(0.8, bbox(4.0, 20.0, 0.0, 10.0)),
        ];
        assert!(compute_scene_bounds(&flat_x).is_none(), "zero width");

        let flat_y = vec![
            detection_with_score(0.9, bbox(0.0, 4.0, 10.0, 0.0)),
            detection_with_score(0.8, bbox(20.0, 4.0, 10.0, 0.0)),
        ];
        assert!(compute_scene_bounds(&flat_y).is_none(), "zero height");

        // A real span is accepted.
        let ok = vec![
            detection_with_score(0.9, bbox(0.0, 0.0, 10.0, 10.0)),
            detection_with_score(0.8, bbox(20.0, 20.0, 10.0, 10.0)),
        ];
        assert!(compute_scene_bounds(&ok).is_some());
    }

    #[test]
    fn grid_cell_index_maps_offsets_to_cells() {
        // Ordinary case: 15 units into 10-unit cells is cell 1.
        assert_eq!(grid_cell_index(15.0, 10.0, 10), 1);
        assert_eq!(grid_cell_index(0.0, 10.0, 10), 0);
        // Past the last cell it clamps rather than indexing out of range.
        assert_eq!(grid_cell_index(1000.0, 10.0, 10), 9);
        // Negative offsets clamp to the first cell.
        assert_eq!(grid_cell_index(-5.0, 10.0, 10), 0);
        // A degenerate cell size short-circuits instead of dividing by zero,
        // which would otherwise clamp infinity to the last cell.
        assert_eq!(grid_cell_index(50.0, 0.0, 10), 0);
    }

    #[test]
    fn cell_range_covers_every_cell_the_bbox_touches() {
        // Origin deliberately away from zero so subtracting it matters.
        let bounds = SceneBounds {
            min_x: 5.0,
            min_y: 10.0,
            max_x: 105.0,
            max_y: 110.0,
        };
        // 100 units across 10 cells means 10 units per cell.
        //   x: 20..40 relative to 5 is 15..35  -> cols 1..3
        //   y: 35..65 relative to 10 is 25..55 -> rows 2..5
        let range = bounds.cell_range_for_bbox(&bbox(20.0, 35.0, 20.0, 30.0), 10);
        assert_eq!(
            range,
            CellRange {
                min_col: 1,
                max_col: 3,
                min_row: 2,
                max_row: 5,
            }
        );
    }

    // ------------------------------------------------------------------
    // Suppression thresholds.

    #[test]
    fn apply_nms_naive_suppresses_only_above_the_threshold() {
        // Identical boxes have an IoU of exactly 1.0. At a threshold of 1.0
        // the comparison is strictly greater, so nothing is suppressed.
        let mut same = vec![
            detection_with_score(0.9, bbox(0.0, 0.0, 10.0, 10.0)),
            detection_with_score(0.8, bbox(0.0, 0.0, 10.0, 10.0)),
        ];
        apply_nms_naive(&mut same, 1.0);
        assert_eq!(same.len(), 2, "IoU equal to the threshold is kept");

        // Just below the threshold and the duplicate goes.
        let mut same = vec![
            detection_with_score(0.9, bbox(0.0, 0.0, 10.0, 10.0)),
            detection_with_score(0.8, bbox(0.0, 0.0, 10.0, 10.0)),
        ];
        apply_nms_naive(&mut same, 0.99);
        assert_eq!(same.len(), 1);
        assert_eq!(same[0].score, 0.9, "the higher-scoring box survives");
    }

    #[test]
    fn apply_nms_naive_keeps_disjoint_boxes_and_preserves_order() {
        // Three boxes: two overlapping, one far away. Comparing each kept box
        // against the ones *after* it is what makes this work; starting the
        // inner scan at the box itself would suppress everything.
        let mut dets = vec![
            detection_with_score(0.9, bbox(0.0, 0.0, 10.0, 10.0)),
            detection_with_score(0.8, bbox(1.0, 1.0, 10.0, 10.0)),
            detection_with_score(0.7, bbox(100.0, 100.0, 10.0, 10.0)),
        ];
        apply_nms_naive(&mut dets, 0.5);
        assert_eq!(dets.len(), 2);
        assert_eq!(dets[0].score, 0.9);
        assert_eq!(dets[1].score, 0.7, "the distant box is untouched");
    }

    #[test]
    fn grid_suppression_uses_the_same_strict_threshold() {
        // Over 200 detections takes the spatial-grid path, which has its own
        // copy of the IoU comparison. Identical boxes at a threshold of 1.0
        // must all survive there too.
        let mut dets: Vec<_> = (0..250)
            .map(|i| detection_with_score(1.0 - i as f32 * 0.001, bbox(0.0, 0.0, 10.0, 10.0)))
            .collect();
        apply_nms_in_place(&mut dets, 1.0);
        assert_eq!(dets.len(), 250, "IoU equal to the threshold is kept");
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn grid_nms_matches_naive_reference(
            boxes in prop::collection::vec(
                (0_u16..2_000, 0_u16..2_000, 1_u16..200, 1_u16..200),
                200..240,
            ),
            threshold in 0_u16..=1_000,
        ) {
            let mut optimized: Vec<_> = boxes
                .into_iter()
                .enumerate()
                .map(|(i, (x, y, width, height))| {
                    detection_with_score(
                        1.0 - i as f32 * 0.001,
                        BoundingBox {
                            x: x.into(),
                            y: y.into(),
                            width: width.into(),
                            height: height.into(),
                        },
                    )
                })
                .collect();
            let mut expected = optimized.clone();
            let threshold = threshold as f32 / 1_000.0;

            apply_nms_in_place(&mut optimized, threshold);
            apply_nms_naive(&mut expected, threshold);

            prop_assert_eq!(optimized, expected);
        }
    }

    #[test]
    fn compute_scene_bounds_returns_none_for_degenerate_scene() {
        let detections = vec![
            detection_with_score(
                1.0,
                BoundingBox {
                    x: 0.0,
                    y: 0.0,
                    width: 0.0,
                    height: 0.0,
                },
            ),
            detection_with_score(
                0.9,
                BoundingBox {
                    x: 0.0,
                    y: 0.0,
                    width: 0.0,
                    height: 0.0,
                },
            ),
        ];

        assert!(compute_scene_bounds(&detections).is_none());
    }

    #[test]
    fn build_spatial_grid_covers_all_cells_touched_by_bbox() {
        let detections = vec![
            detection_with_score(
                1.0,
                BoundingBox {
                    x: 0.0,
                    y: 0.0,
                    width: 96.0,
                    height: 96.0,
                },
            ),
            detection_with_score(
                0.9,
                BoundingBox {
                    x: 48.0,
                    y: 48.0,
                    width: 1.0,
                    height: 1.0,
                },
            ),
        ];
        let bounds = compute_scene_bounds(&detections).expect("scene should be valid");
        let grid = build_spatial_grid(&detections, bounds);

        let range = grid
            .bounds
            .cell_range_for_bbox(&detections[0].bbox, grid.grid_size);
        for row in range.min_row..=range.max_row {
            for col in range.min_col..=range.max_col {
                assert!(
                    grid.cells[row * grid.grid_size + col].contains(&0),
                    "cell ({row}, {col}) should contain detection 0"
                );
            }
        }
    }

    #[test]
    fn compact_unsuppressed_detections_preserves_kept_order() {
        let mut detections = vec![
            detection_with_score(
                0.9,
                BoundingBox {
                    x: 0.0,
                    y: 0.0,
                    width: 10.0,
                    height: 10.0,
                },
            ),
            detection_with_score(
                0.8,
                BoundingBox {
                    x: 20.0,
                    y: 0.0,
                    width: 10.0,
                    height: 10.0,
                },
            ),
            detection_with_score(
                0.7,
                BoundingBox {
                    x: 40.0,
                    y: 0.0,
                    width: 10.0,
                    height: 10.0,
                },
            ),
        ];
        let suppressed = vec![false, true, false];

        compact_unsuppressed_detections(&mut detections, &suppressed);

        assert_eq!(detections.len(), 2);
        assert!((detections[0].score - 0.9).abs() < f32::EPSILON);
        assert!((detections[1].score - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn grid_and_naive_agree_across_scene_shapes() {
        // The grid is an acceleration structure, so sizing its cells to the scene
        // (experiment 58) must change speed and nothing else. These four shapes cover the
        // cases that pick different resolutions: a tight cluster where boxes are large
        // against the scene bounds, a spread-out scene where they are tiny, one of each
        // mixed, and identical stacked boxes where the extent collapses to a point.
        let scenes: Vec<(&str, Vec<Detection>)> = vec![
            (
                "clustered",
                (0..600)
                    .map(|i| {
                        detection_with_score(
                            1.0 - i as f32 * 1e-4,
                            BoundingBox {
                                x: 100.0 + i as f32 * 0.1,
                                y: 100.0,
                                width: 80.0,
                                height: 80.0,
                            },
                        )
                    })
                    .collect(),
            ),
            (
                "spread",
                (0..600)
                    .map(|i| {
                        detection_with_score(
                            1.0 - i as f32 * 1e-4,
                            BoundingBox {
                                x: (i % 30) as f32 * 200.0,
                                y: (i / 30) as f32 * 200.0,
                                width: 40.0,
                                height: 40.0,
                            },
                        )
                    })
                    .collect(),
            ),
            (
                "mixed scales",
                (0..600)
                    .map(|i| {
                        let big = i % 3 == 0;
                        detection_with_score(
                            1.0 - i as f32 * 1e-4,
                            BoundingBox {
                                x: (i % 25) as f32 * 37.0,
                                y: (i / 25) as f32 * 41.0,
                                width: if big { 300.0 } else { 12.0 },
                                height: if big { 300.0 } else { 12.0 },
                            },
                        )
                    })
                    .collect(),
            ),
            (
                "identical stack",
                (0..300)
                    .map(|i| {
                        detection_with_score(
                            1.0 - i as f32 * 1e-4,
                            BoundingBox {
                                x: 5.0,
                                y: 5.0,
                                width: 50.0,
                                height: 50.0,
                            },
                        )
                    })
                    .collect(),
            ),
        ];

        for (name, scene) in scenes {
            let mut fast = scene.clone();
            let mut naive = scene;
            apply_nms_in_place(&mut fast, 0.3);
            apply_nms_naive(&mut naive, 0.3);
            assert_eq!(fast.len(), naive.len(), "{name}: kept a different count");
            for (a, b) in fast.iter().zip(&naive) {
                assert_eq!(a.score, b.score, "{name}: different detection kept");
                assert_eq!(a.bbox, b.bbox, "{name}: different box kept");
            }
        }
    }

    #[test]
    fn grid_size_shrinks_for_clustered_scenes_and_stays_valid() {
        // Boxes larger than the scene bounds must not ask for a zero-sized or oversized grid.
        let stacked: Vec<_> = (0..10)
            .map(|i| {
                detection_with_score(
                    1.0 - i as f32 * 0.01,
                    BoundingBox {
                        x: 0.0,
                        y: 0.0,
                        width: 100.0,
                        height: 100.0,
                    },
                )
            })
            .collect();
        let bounds = compute_scene_bounds(&stacked).expect("bounds");
        let size = grid_size_for(&stacked, bounds);
        assert!(
            (1..=NMS_GRID_SIZE).contains(&size),
            "grid size {size} out of range"
        );

        // A scene far wider than its boxes should still reach the cap.
        let spread: Vec<_> = (0..64)
            .map(|i| {
                detection_with_score(
                    1.0 - i as f32 * 0.001,
                    BoundingBox {
                        x: i as f32 * 1000.0,
                        y: 0.0,
                        width: 10.0,
                        height: 10.0,
                    },
                )
            })
            .collect();
        let bounds = compute_scene_bounds(&spread).expect("bounds");
        assert_eq!(grid_size_for(&spread, bounds), NMS_GRID_SIZE);
    }

    /// Cells are sized to the mean box: longest sides 6, 10 and 14 average 10, and a scene 73
    /// long needs ceil(73 / 10) = 8 of them. Not the cap and not 1, so the arithmetic has
    /// nowhere to hide.
    #[test]
    fn grid_size_is_the_scene_span_over_the_mean_box_extent() {
        let boxed = |width, height| detection_with_score(0.9, bbox(0.0, 0.0, width, height));
        let bounds = || SceneBounds {
            min_x: 0.0,
            min_y: 0.0,
            max_x: 73.0,
            max_y: 41.0,
        };
        let boxes = [boxed(6.0, 2.0), boxed(3.0, 10.0), boxed(14.0, 5.0)];
        assert_eq!(grid_size_for(&boxes, bounds()), 8);

        // A non-finite box leaves nothing to size a cell by: the cap, not a NaN cast to one cell.
        assert_eq!(
            grid_size_for(&[boxed(f32::NAN, f32::NAN)], bounds()),
            NMS_GRID_SIZE
        );
    }

    #[test]
    fn apply_nms_in_place_uses_large_grid_path_for_large_inputs() {
        let mut detections: Vec<_> = (0..200)
            .map(|i| {
                detection_with_score(
                    1.0 - i as f32 * 0.001,
                    BoundingBox {
                        x: i as f32 * 20.0,
                        y: 0.0,
                        width: 10.0,
                        height: 10.0,
                    },
                )
            })
            .collect();
        // Already in descending score order; add overlapping low-score box last.
        detections.push(detection_with_score(
            0.0,
            BoundingBox {
                x: 1.0,
                y: 1.0,
                width: 10.0,
                height: 10.0,
            },
        ));

        apply_nms_in_place(&mut detections, 0.3);

        assert_eq!(detections.len(), 200);
        assert!((detections[0].score - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn apply_nms_in_place_handles_zero_and_one_items() {
        let mut empty: Vec<Detection> = vec![];
        apply_nms_in_place(&mut empty, 0.3);
        assert_eq!(empty.len(), 0);

        let single = detection_with_score(
            0.9,
            BoundingBox {
                x: 0.0,
                y: 0.0,
                width: 10.0,
                height: 10.0,
            },
        );
        let mut one = vec![single];
        apply_nms_in_place(&mut one, 0.3);
        assert_eq!(one.len(), 1);
    }

    #[test]
    fn apply_nms_naive_handles_zero_and_one_items() {
        let mut empty: Vec<Detection> = vec![];
        apply_nms_naive(&mut empty, 0.3);
        assert_eq!(empty.len(), 0);

        let mut one = vec![detection_with_score(
            0.9,
            BoundingBox {
                x: 0.0,
                y: 0.0,
                width: 5.0,
                height: 5.0,
            },
        )];
        apply_nms_naive(&mut one, 0.3);
        assert_eq!(one.len(), 1);
    }

    #[test]
    fn apply_nms_in_place_degenerate_scene_falls_back_to_naive() {
        // All boxes at the same point → zero scene extent, triggers naive fallback.
        let mut detections: Vec<_> = (0..5)
            .map(|i| {
                detection_with_score(
                    0.9 - i as f32 * 0.1,
                    BoundingBox {
                        x: 0.0,
                        y: 0.0,
                        width: 0.0,
                        height: 0.0,
                    },
                )
            })
            .collect();
        // Should not panic.
        apply_nms_in_place(&mut detections, 0.3);
    }

    #[test]
    fn grid_nms_degenerate_scene_falls_back_to_naive_for_large_input() {
        // 200+ items all with width=0 and height=0 → scene extent is zero → degenerate path.
        let mut detections: Vec<_> = (0..201)
            .map(|i| {
                detection_with_score(
                    1.0 - i as f32 * 0.004,
                    BoundingBox {
                        x: 0.0,
                        y: i as f32 * 10.0,
                        width: 0.0,
                        height: 0.0,
                    },
                )
            })
            .collect();
        // Should not panic and should survive (all non-overlapping zero-area boxes).
        apply_nms_in_place(&mut detections, 0.3);
        assert_eq!(detections.len(), 201);
    }

    #[test]
    fn grid_nms_compaction_swap_fires_when_suppressed_item_in_middle() {
        // Build 201 detections so the grid path is used (len >= 200).
        // Detection at index 1 overlaps detection at index 0 and is suppressed.
        // When compacting, detection at index 2 ends up at keep=1 → swap fires (line 342).
        let mut detections = Vec::with_capacity(201);

        // Index 0: highest score, box at x=0
        detections.push(detection_with_score(
            1.0,
            BoundingBox {
                x: 0.0,
                y: 0.0,
                width: 50.0,
                height: 50.0,
            },
        ));
        // Index 1: overlaps index 0 heavily (will be suppressed)
        detections.push(detection_with_score(
            0.999,
            BoundingBox {
                x: 5.0,
                y: 5.0,
                width: 50.0,
                height: 50.0,
            },
        ));
        // Indices 2..200: non-overlapping boxes spread across the scene
        for i in 2..201 {
            detections.push(detection_with_score(
                1.0 - i as f32 * 0.004,
                BoundingBox {
                    x: i as f32 * 200.0,
                    y: 0.0,
                    width: 10.0,
                    height: 10.0,
                },
            ));
        }

        // Already sorted by score descending; NMS threshold low enough to suppress index 1.
        apply_nms_in_place(&mut detections, 0.3);

        // Index 1 was suppressed, all others survive.
        assert_eq!(detections.len(), 200);
        // Highest-score detection still first.
        assert!((detections[0].score - 1.0).abs() < f32::EPSILON);
    }
}

#[cfg(test)]
mod benches {
    use super::*;
    use crate::postprocess::{BoundingBox, Detection, Landmark};
    use std::time::{Duration, Instant};

    fn apply_nms_in_place_baseline(detections: &mut Vec<Detection>, threshold: f32) {
        let len = detections.len();
        if len <= 1 {
            return;
        }

        let mut suppressed = vec![false; len];
        let mut keep = 0;

        for i in 0..len {
            if suppressed[i] {
                continue;
            }

            if keep != i {
                detections.swap(keep, i);
                suppressed.swap(keep, i);
            }

            let reference_bbox = detections[keep].bbox;
            for j in (keep + 1)..len {
                if !suppressed[j] && reference_bbox.iou(&detections[j].bbox) > threshold {
                    suppressed[j] = true;
                }
            }

            keep += 1;
        }

        detections.truncate(keep);
    }

    struct SimpleRng(u64);
    impl SimpleRng {
        fn next_f32(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((self.0 >> 32) as u32) as f32 / 4294967296.0
        }
    }

    fn synthetic_detections(count: usize) -> Vec<Detection> {
        let mut out = Vec::with_capacity(count);
        let mut rng = SimpleRng(12345);
        for _ in 0..count {
            out.push(Detection {
                bbox: BoundingBox {
                    x: rng.next_f32() * 2000.0,
                    y: rng.next_f32() * 2000.0,
                    width: rng.next_f32().mul_add(100.0, 20.0),
                    height: rng.next_f32().mul_add(100.0, 20.0),
                },
                landmarks: [Landmark { x: 0.0, y: 0.0 }; 5],
                score: rng.next_f32(),
            });
        }
        // Essential: Sort by score descending to simulate real model output
        out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        out
    }

    #[test]
    #[ignore]
    fn bench_nms_variants() {
        let template = synthetic_detections(5_000);
        let iterations = 20;

        let mut optimized_total = Duration::ZERO;
        let mut baseline_total = Duration::ZERO;

        for i in 0..iterations {
            if i % 2 == 0 {
                let mut data = template.clone();
                let start = Instant::now();
                apply_nms_in_place(&mut data, 0.3);
                optimized_total += start.elapsed();

                let mut baseline = template.clone();
                let start = Instant::now();
                apply_nms_in_place_baseline(&mut baseline, 0.3);
                baseline_total += start.elapsed();
            } else {
                let mut baseline = template.clone();
                let start = Instant::now();
                apply_nms_in_place_baseline(&mut baseline, 0.3);
                baseline_total += start.elapsed();

                let mut data = template.clone();
                let start = Instant::now();
                apply_nms_in_place(&mut data, 0.3);
                optimized_total += start.elapsed();
            }
        }

        let diff = baseline_total.as_secs_f64() / optimized_total.as_secs_f64();
        println!(
            "NMS Benchmark (5k random items, {} iters):\n  Optimized (Grid): {:?}\n  Baseline (Naive): {:?}\n  Speedup:          {:.2}x",
            iterations,
            optimized_total / iterations as u32,
            baseline_total / iterations as u32,
            diff
        );
    }
}

#[cfg(test)]
mod benchmarks {
    use super::*;
    use crate::postprocess::Landmark;
    use std::time::Instant;

    fn detection(x: f32, y: f32, size: f32, score: f32) -> Detection {
        Detection {
            bbox: BoundingBox {
                x,
                y,
                width: size,
                height: size,
            },
            landmarks: [Landmark::new(0.0, 0.0); 5],
            score,
        }
    }

    /// Faces spread far enough apart that nothing is merged: every pair is compared and
    /// every detection survives, which is the pure comparison cost.
    fn separated(n: usize) -> Vec<Detection> {
        (0..n)
            .map(|i| {
                let col = (i % 64) as f32;
                let row = (i / 64) as f32;
                detection(col * 200.0, row * 200.0, 40.0, 1.0 - i as f32 * 1e-6)
            })
            .collect()
    }

    /// Every detection inside one face's radius, so every probe removes. This is the shape
    /// that makes `Vec::remove` inside the inner loop expensive.
    fn clustered(n: usize) -> Vec<Detection> {
        (0..n)
            .map(|i| detection(100.0 + i as f32 * 0.01, 100.0, 80.0, 1.0 - i as f32 * 1e-6))
            .collect()
    }

    #[test]
    fn bench_apply_nms_in_place_scaling() {
        println!("{:>7} {:>14} {:>14}", "n", "separated ms", "clustered ms");
        for n in [100usize, 500, 1000, 2000, 5000] {
            let mut sep = separated(n);
            let start = Instant::now();
            apply_nms_in_place(&mut sep, 0.3);
            let sep_ms = start.elapsed().as_secs_f64() * 1e3;

            let mut clu = clustered(n);
            let start = Instant::now();
            apply_nms_in_place(&mut clu, 0.3);
            let clu_ms = start.elapsed().as_secs_f64() * 1e3;

            println!("{n:>7} {sep_ms:>14.3} {clu_ms:>14.3}");
        }
    }
}
