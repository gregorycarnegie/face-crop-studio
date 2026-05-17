use crate::{
    model_config::NMS_GRID_SIZE,
    postprocess::{BoundingBox, Detection},
};

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
            max_col: grid_cell_index(bbox.x + bbox.width - self.min_x, cell_w, grid_size),
            min_row: grid_cell_index(bbox.y - self.min_y, cell_h, grid_size),
            max_row: grid_cell_index(bbox.y + bbox.height - self.min_y, cell_h, grid_size),
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

    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;

    for detection in detections {
        let bbox = detection.bbox;
        if bbox.x < min_x {
            min_x = bbox.x;
        }
        if bbox.y < min_y {
            min_y = bbox.y;
        }

        let right = bbox.x + bbox.width;
        let bottom = bbox.y + bbox.height;
        if right > max_x {
            max_x = right;
        }
        if bottom > max_y {
            max_y = bottom;
        }
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

fn build_spatial_grid(detections: &[Detection], bounds: SceneBounds) -> SpatialGrid {
    let mut cells: Vec<Vec<usize>> = (0..NMS_GRID_SIZE * NMS_GRID_SIZE)
        .map(|_| Vec::with_capacity(detections.len() / (NMS_GRID_SIZE * NMS_GRID_SIZE / 4).max(1)))
        .collect();

    for (i, detection) in detections.iter().enumerate() {
        let range = bounds.cell_range_for_bbox(&detection.bbox, NMS_GRID_SIZE);
        for row in range.min_row..=range.max_row {
            let row_offset = row * NMS_GRID_SIZE;
            for col in range.min_col..=range.max_col {
                cells[row_offset + col].push(i);
            }
        }
    }

    SpatialGrid {
        grid_size: NMS_GRID_SIZE,
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

/// Drop detections whose center is within `min_relative_distance` of a higher-
/// scoring detection, expressed as a fraction of the smaller bbox's longest
/// edge. Also drops detections whose absolute center distance is under
/// `min_absolute_distance_px`, which keeps degenerate tiny boxes from slipping
/// through when the relative-distance metric collapses.
///
/// Belt-and-suspenders pass that runs after IoU-based NMS. GPU compute shader
/// reductions (used in the WGSL YuNet backend) produce ULP-level wobble in
/// score and bbox coordinates between runs; that wobble can flip an IoU
/// comparison across the NMS threshold for a few tightly-clustered pairs per
/// batch. YuNet also emits at three strides (8/16/32) so the same face can
/// produce boxes of different sizes with centers up to ~30 px apart that NMS
/// reads as "different shapes" because their IoU is low. Centroid distance
/// scaled by face size catches both kinds of survivors without merging
/// genuinely-separate faces.
///
/// Expects `detections` to already be sorted by score descending. O(N^2) in
/// the worst case but N is typically tiny (post-NMS detection counts are
/// usually 1–10 per image).
pub(crate) fn dedup_close_centers(
    detections: &mut Vec<Detection>,
    min_relative_distance: f32,
    min_absolute_distance_px: f32,
) {
    if detections.len() <= 1 {
        return;
    }
    let min_abs_sq = min_absolute_distance_px * min_absolute_distance_px;
    let mut keep_idx = 0;
    while keep_idx < detections.len() {
        let kept = detections[keep_idx].bbox;
        let kept_cx = kept.width.mul_add(0.5, kept.x);
        let kept_cy = kept.height.mul_add(0.5, kept.y);
        let kept_longest = kept.width.max(kept.height);
        let mut probe = keep_idx + 1;
        while probe < detections.len() {
            let other = detections[probe].bbox;
            let dx = other.width.mul_add(0.5, other.x) - kept_cx;
            let dy = other.height.mul_add(0.5, other.y) - kept_cy;
            let dist_sq = dx.mul_add(dx, dy * dy);

            // Use the larger of the two bboxes' longest edge as the scale.
            // YuNet's stride-8/16/32 outputs for the same face can produce
            // boxes whose centers drift up to ~half the bigger box's extent —
            // taking the min collapses the threshold and misses those.
            let other_longest = other.width.max(other.height);
            let scale = kept_longest.max(other_longest).max(1.0);
            let rel_threshold = scale * min_relative_distance;
            let rel_threshold_sq = rel_threshold * rel_threshold;

            if dist_sq < rel_threshold_sq || dist_sq < min_abs_sq {
                detections.remove(probe);
            } else {
                probe += 1;
            }
        }
        keep_idx += 1;
    }
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

    fn detection_with_score(score: f32, bbox: BoundingBox) -> Detection {
        Detection {
            bbox,
            landmarks: [Landmark::new(0.0, 0.0); 5],
            score,
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
    fn dedup_close_centers_drops_multi_scale_duplicates_around_one_face() {
        // Simulates YuNet's multi-stride output for one face: same centroid area,
        // different bbox sizes from different anchor strides. NMS may miss these
        // because IoU between mismatched-size boxes is low; centroid dedup catches them.
        let mut detections = vec![
            // Stride-16 box, the strongest detection
            detection_with_score(
                0.95,
                BoundingBox {
                    x: 100.0,
                    y: 100.0,
                    width: 80.0,
                    height: 80.0,
                },
            ), // center (140, 140)
            // Stride-8 box on the same face — smaller, slightly shifted center
            detection_with_score(
                0.92,
                BoundingBox {
                    x: 125.0,
                    y: 125.0,
                    width: 30.0,
                    height: 30.0,
                },
            ), // center (140, 140) — within 35% of 30 = 10.5 px
            // Stride-32 box on the same face — larger, center drifted ~15 px
            detection_with_score(
                0.90,
                BoundingBox {
                    x: 70.0,
                    y: 80.0,
                    width: 130.0,
                    height: 130.0,
                },
            ), // center (135, 145) — within 35% of 80 = 28 px of (140, 140)
            // Genuinely different face elsewhere in the image
            detection_with_score(
                0.88,
                BoundingBox {
                    x: 400.0,
                    y: 100.0,
                    width: 80.0,
                    height: 80.0,
                },
            ),
        ];

        dedup_close_centers(&mut detections, 0.35, 8.0);

        assert_eq!(detections.len(), 2);
        assert!((detections[0].score - 0.95).abs() < f32::EPSILON);
        assert!((detections[1].score - 0.88).abs() < f32::EPSILON);
    }

    #[test]
    fn dedup_close_centers_no_op_for_empty_or_singleton() {
        let mut empty: Vec<Detection> = vec![];
        dedup_close_centers(&mut empty, 0.35, 8.0);
        assert_eq!(empty.len(), 0);

        let mut singleton = vec![detection_with_score(
            0.9,
            BoundingBox {
                x: 0.0,
                y: 0.0,
                width: 10.0,
                height: 10.0,
            },
        )];
        dedup_close_centers(&mut singleton, 0.35, 8.0);
        assert_eq!(singleton.len(), 1);
    }

    #[test]
    fn dedup_close_centers_keeps_widely_separated_detections() {
        // Three faces in different image regions — all should survive.
        let mut detections = vec![
            detection_with_score(
                0.9,
                BoundingBox {
                    x: 0.0,
                    y: 0.0,
                    width: 40.0,
                    height: 40.0,
                },
            ),
            detection_with_score(
                0.8,
                BoundingBox {
                    x: 500.0,
                    y: 0.0,
                    width: 40.0,
                    height: 40.0,
                },
            ),
            detection_with_score(
                0.7,
                BoundingBox {
                    x: 0.0,
                    y: 500.0,
                    width: 40.0,
                    height: 40.0,
                },
            ),
        ];
        dedup_close_centers(&mut detections, 0.35, 8.0);
        assert_eq!(detections.len(), 3);
    }

    #[test]
    fn dedup_close_centers_absolute_floor_catches_tiny_boxes() {
        // Two pathologically small bboxes whose relative threshold would be
        // essentially zero — the absolute floor should still merge them.
        let mut detections = vec![
            detection_with_score(
                0.9,
                BoundingBox {
                    x: 100.0,
                    y: 100.0,
                    width: 2.0,
                    height: 2.0,
                },
            ), // center (101, 101)
            detection_with_score(
                0.8,
                BoundingBox {
                    x: 103.0,
                    y: 102.0,
                    width: 2.0,
                    height: 2.0,
                },
            ), // center (104, 103) — distance ~3.2 px, absolute floor catches it
        ];
        dedup_close_centers(&mut detections, 0.35, 8.0);
        assert_eq!(detections.len(), 1);
        assert!((detections[0].score - 0.9).abs() < f32::EPSILON);
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
