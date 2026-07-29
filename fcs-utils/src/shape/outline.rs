//! Outline generation for crop shapes.

use crate::point::Point;

use super::types::{CropShape, MIN_POLYGON_SIDES, PolygonCornerStyle};
use std::f32::consts::{FRAC_PI_2, PI, TAU};
use tiny_skia::PathBuilder;

/// Number of segments used for ellipse outlines.
const ELLIPSE_SEGMENTS: usize = 128;
/// Number of segments per corner for rounded rectangles.
const ROUNDED_RECT_CORNER_SEGMENTS: usize = 16;
/// Number of segments per corner for rounded polygons.
const ROUNDED_POLYGON_CORNER_SEGMENTS: usize = 8;
/// Number of segments for Bezier polygon interpolation.
const BEZIER_POLYGON_SEGMENTS: usize = 16;
const KOCH_SIN_60: f32 = 0.866_025_4; // sin(60 degrees)
const KOCH_COS_60: f32 = 0.5; // cos(60 degrees)

/// Generate outline points for a shape fitted to the supplied width/height.
fn outline_points(width: u32, height: u32, shape: &CropShape) -> Vec<Point> {
    let w = width.max(1) as f32;
    let h = height.max(1) as f32;
    let shape = shape.sanitized();

    let mut points = match &shape {
        CropShape::Rectangle => vec![
            Point { x: 0.0, y: 0.0 },
            Point { x: w, y: 0.0 },
            Point { x: w, y: h },
            Point { x: 0.0, y: h },
        ],
        CropShape::Ellipse => {
            let cx = w * 0.5;
            let cy = h * 0.5;
            (0..ELLIPSE_SEGMENTS)
                .map(|i| {
                    let theta = (i as f32 / ELLIPSE_SEGMENTS as f32) * TAU;
                    Point {
                        x: theta.cos().mul_add(cx, cx),
                        y: theta.sin().mul_add(cy, cy),
                    }
                })
                .collect()
        }
        // `sanitized()` above already caps both percentages at 0.5, so the
        // scaled value can never exceed half the short side and needs no clamp
        // of its own. (`mask.rs` does still clamp: `apply_shape_mask` takes an
        // unsanitized shape.)
        CropShape::RoundedRectangle { radius_pct } => {
            rounded_rect_points(w, h, w.min(h) * radius_pct, ROUNDED_RECT_CORNER_SEGMENTS)
        }
        CropShape::ChamferedRectangle { size_pct } => {
            chamfered_rect_points(w, h, w.min(h) * size_pct)
        }
        CropShape::Polygon {
            sides,
            rotation_deg,
            corner_style,
        } => polygon_points(w, h, *sides, *rotation_deg, corner_style.clone()),
        CropShape::Star {
            points,
            inner_radius_pct,
            rotation_deg,
        } => star_points(w, h, *points, *inner_radius_pct, *rotation_deg),
        CropShape::KochPolygon {
            sides,
            rotation_deg,
            iterations,
        } => {
            let base_poly = polygon_points(w, h, *sides, *rotation_deg, PolygonCornerStyle::Sharp);
            koch_fractal(&base_poly, *iterations)
        }
        CropShape::KochRectangle { iterations } => {
            let base_rect = vec![
                Point { x: 0.0, y: 0.0 },
                Point { x: w, y: 0.0 },
                Point { x: w, y: h },
                Point { x: 0.0, y: h },
            ];
            koch_fractal(&base_rect, *iterations)
        }
    };

    // Fit complex shapes to bounds to prevent clipping.
    match shape {
        CropShape::Polygon { .. }
        | CropShape::Star { .. }
        | CropShape::KochPolygon { .. }
        | CropShape::KochRectangle { .. } => {
            fit_points_to_bounds(&mut points, w, h);
        }
        _ => {}
    }

    points
}

fn fit_points_to_bounds(points: &mut [Point], width: f32, height: f32) {
    if points.is_empty() {
        return;
    }
    let mut min_points = points[0];
    let mut max_points = points[0];

    for p in points.iter().skip(1) {
        if p.x < min_points.x {
            min_points.x = p.x;
        }
        if p.x > max_points.x {
            max_points.x = p.x;
        }
        if p.y < min_points.y {
            min_points.y = p.y;
        }
        if p.y > max_points.y {
            max_points.y = p.y;
        }
    }

    let bbox = max_points - min_points;

    if bbox.x <= f32::EPSILON || bbox.y <= f32::EPSILON {
        return;
    }

    let scale_x = width / bbox.x;
    let scale_y = height / bbox.y;
    let scale = scale_x.min(scale_y);

    let new_width = bbox.x * scale;
    let new_height = bbox.y * scale;

    let offset_x = (width - new_width).mul_add(0.5, -min_points.x * scale);
    let offset_y = (height - new_height).mul_add(0.5, -min_points.y * scale);

    for p in points.iter_mut() {
        p.x = p.x.mul_add(scale, offset_x);
        p.y = p.y.mul_add(scale, offset_y);
    }
}

fn koch_fractal(vertices: &[Point], iterations: u8) -> Vec<Point> {
    if iterations == 0 {
        return vertices.to_vec();
    }

    let mut current_vertices = vertices.to_vec();

    for _ in 0..iterations {
        let mut next_vertices = Vec::with_capacity(current_vertices.len() * 4);
        let len = current_vertices.len();

        for i in 0..len {
            let p0 = current_vertices[i];
            let p1 = current_vertices[(i + 1) % len];
            let dxy = p1 - p0;
            let p_a = p0 + dxy / 3.0;
            let p_c = (dxy / 3.0).mul_add(2.0, p0);

            // Vector from p_a to p_c is rotated -60 degrees, outward for a CCW polygon.
            let v = p_c - p_a;

            let p_b_x = p_a.x + v.y.mul_add(KOCH_SIN_60, v.x * KOCH_COS_60);
            let p_b_y = p_a.y + v.y.mul_add(KOCH_COS_60, -v.x * KOCH_SIN_60);

            let p_b = Point { x: p_b_x, y: p_b_y };

            next_vertices.push(p0);
            next_vertices.push(p_a);
            next_vertices.push(p_b);
            next_vertices.push(p_c);
        }
        current_vertices = next_vertices;
    }

    current_vertices
}

fn rounded_rect_points(width: f32, height: f32, radius: f32, segments: usize) -> Vec<Point> {
    if radius <= 0.0 {
        return vec![
            Point { x: 0.0, y: 0.0 },
            Point { x: width, y: 0.0 },
            Point {
                x: width,
                y: height,
            },
            Point { x: 0.0, y: height },
        ];
    }

    let mut points = Vec::with_capacity(segments * 4);
    let mut add_corner = |cx: f32, cy: f32, start: f32, end: f32| {
        let steps = segments.max(3);
        let delta = (end - start) / steps as f32;
        for i in 0..=steps {
            let angle = delta.mul_add(i as f32, start);
            push_point(&mut points, angle, cx, cy, radius);
        }
    };

    add_corner(width - radius, radius, -FRAC_PI_2, 0.0);
    add_corner(width - radius, height - radius, 0.0, FRAC_PI_2);
    add_corner(radius, height - radius, FRAC_PI_2, PI);
    add_corner(radius, radius, PI, 1.5 * PI);

    points
}

fn chamfered_rect_points(width: f32, height: f32, inset: f32) -> Vec<Point> {
    if inset <= 0.0 {
        return vec![
            Point { x: 0.0, y: 0.0 },
            Point { x: width, y: 0.0 },
            Point {
                x: width,
                y: height,
            },
            Point { x: 0.0, y: height },
        ];
    }

    vec![
        Point { x: inset, y: 0.0 },
        Point {
            x: width - inset,
            y: 0.0,
        },
        Point { x: width, y: inset },
        Point {
            x: width,
            y: height - inset,
        },
        Point {
            x: width - inset,
            y: height,
        },
        Point {
            x: inset,
            y: height,
        },
        Point {
            x: 0.0,
            y: height - inset,
        },
        Point { x: 0.0, y: inset },
    ]
}

fn polygon_points(
    width: f32,
    height: f32,
    sides: u8,
    rotation_deg: f32,
    corner_style: PolygonCornerStyle,
) -> Vec<Point> {
    let n = sides.max(MIN_POLYGON_SIDES) as usize;
    let cx = width * 0.5;
    let cy = height * 0.5;
    let radius = 0.5 * width.min(height);
    let rotation = rotation_deg.to_radians();

    let mut base_vertices = Vec::with_capacity(n);
    for i in 0..n {
        let angle = rotation + TAU * i as f32 / n as f32;
        push_point(&mut base_vertices, angle, cx, cy, radius);
    }

    match corner_style {
        PolygonCornerStyle::Sharp => base_vertices,
        PolygonCornerStyle::Chamfered { size_pct } => {
            let inset = (width.min(height) * size_pct).clamp(0.0, radius);
            chamfer_polygon(&base_vertices, inset)
        }
        PolygonCornerStyle::Rounded { radius_pct } => {
            let r = (width.min(height) * radius_pct).clamp(0.0, radius);
            rounded_polygon(&base_vertices, r, ROUNDED_POLYGON_CORNER_SEGMENTS)
        }
        PolygonCornerStyle::Bezier { tension } => {
            bezier_polygon(&base_vertices, tension, BEZIER_POLYGON_SEGMENTS)
        }
    }
}

fn chamfer_polygon(vertices: &[Point], inset: f32) -> Vec<Point> {
    if inset <= 0.0 {
        return vertices.to_vec();
    }

    let len = vertices.len();
    let mut points = Vec::with_capacity(len * 2);

    for i in 0..len {
        let prev = vertices[(i + len - 1) % len];
        let current = vertices[i];
        let next = vertices[(i + 1) % len];

        let prev_vec = normalize(current - prev);
        let next_vec = normalize(next - current);

        let prev_edge_len = distance(prev, current);
        let next_edge_len = distance(current, next);
        let offset_prev = inset.min(prev_edge_len * 0.5);
        let offset_next = inset.min(next_edge_len * 0.5);

        points.push((-prev_vec).mul_add(offset_prev, current));
        points.push(next_vec.mul_add(offset_next, current));
    }

    points
}

fn rounded_polygon(vertices: &[Point], radius: f32, segments: usize) -> Vec<Point> {
    if radius <= 0.0 {
        return vertices.to_vec();
    }

    let len = vertices.len();
    let mut points = Vec::with_capacity(len * segments);

    for i in 0..len {
        let prev = vertices[(i + len - 1) % len];
        let current = vertices[i];
        let next = vertices[(i + 1) % len];

        let incoming = normalize(current - prev);
        let outgoing = normalize(next - current);

        let angle_cos = (-incoming) * outgoing;
        let angle_cos = angle_cos.clamp(-0.999_9, 0.999_9);
        let half_angle = 0.5 * angle_cos.acos();
        let mut offset = radius / half_angle.tan();
        let incoming_len = distance(prev, current);
        let outgoing_len = distance(current, next);
        // min(o, a*0.5, b*0.5) == min(o, min(a,b)*0.5); the chained form halved
        // both edges separately, so on the regular polygons this is called with
        // the second clamp was always a no-op.
        offset = offset.min(incoming_len.min(outgoing_len) * 0.5);

        let start = (-incoming).mul_add(offset, current);
        let end = outgoing.mul_add(offset, current);

        let bisector = normalize(outgoing - incoming);
        let center_distance = radius / half_angle.sin();

        let center = bisector.mul_add(center_distance, current);

        let start_angle = (start.y - center.y).atan2(start.x - center.x);
        let end_angle = (end.y - center.y).atan2(end.x - center.x);
        let mut delta = end_angle - start_angle;
        while delta <= 0.0 {
            delta += TAU;
        }
        let steps = segments.max(3);
        let step = delta / steps as f32;
        for j in 0..=steps {
            let angle = step.mul_add(j as f32, start_angle);
            push_point(&mut points, angle, center.x, center.y, radius);
        }
    }

    points
}

fn bezier_polygon(vertices: &[Point], tension: f32, segments: usize) -> Vec<Point> {
    if tension <= 0.0 {
        return vertices.to_vec();
    }

    let len = vertices.len();
    let mut points = Vec::with_capacity(len * segments);
    let mut control_points = Vec::with_capacity(len * 2);

    for i in 0..len {
        let prev = vertices[(i + len - 1) % len];
        let current = vertices[i];
        let next = vertices[(i + 1) % len];

        let tangent = next - prev;
        let cp_dist = tension * 0.5;

        let cp1 = (-tangent).mul_add(cp_dist, current);
        let cp2 = tangent.mul_add(cp_dist, current);

        control_points.push((cp1, cp2));
    }

    for i in 0..len {
        let p0 = vertices[i];
        let p1 = vertices[(i + 1) % len];

        let cp1 = control_points[i].1;
        let cp2 = control_points[(i + 1) % len].0;

        for j in 0..segments {
            let t = j as f32 / segments as f32;
            points.push(cubic_bezier(p0, cp1, cp2, p1, t));
        }
    }

    points
}

fn cubic_bezier(p0: Point, p1: Point, p2: Point, p3: Point, t: f32) -> Point {
    let t2 = t * t;
    let t3 = t2 * t;
    let mt = 1.0 - t;
    let mt2 = mt * mt;
    let mt3 = mt2 * mt;

    p0 * mt3 + p1 * 3.0 * mt2 * t + p2 * 3.0 * mt * t2 + p3 * t3
}

#[inline]
fn distance(a: Point, b: Point) -> f32 {
    (a - b).hypot()
}

fn normalize(v: Point) -> Point {
    let len = v.hypot();
    if len <= f32::EPSILON {
        Point { x: 0.0, y: 0.0 }
    } else {
        v / len
    }
}

pub(super) fn build_path(width: u32, height: u32, shape: &CropShape) -> Option<tiny_skia::Path> {
    let points = outline_points(width, height, shape);
    if points.is_empty() {
        return None;
    }

    let mut builder = PathBuilder::new();
    builder.move_to(points[0].x, points[0].y);
    for point in points.iter().skip(1) {
        builder.line_to(point.x, point.y);
    }
    builder.close();
    builder.finish()
}

fn star_points(
    width: f32,
    height: f32,
    points: u8,
    inner_radius_pct: f32,
    rotation_deg: f32,
) -> Vec<Point> {
    let n = points.max(MIN_POLYGON_SIDES) as usize;
    let cx = width * 0.5;
    let cy = height * 0.5;
    let outer_radius = 0.5 * width.min(height);
    let inner_radius = outer_radius * inner_radius_pct;
    let rotation = rotation_deg.to_radians();

    let mut vertices = Vec::with_capacity(n * 2);
    let step_angle = PI / n as f32;

    for i in 0..n {
        let angle_outer = rotation + TAU * i as f32 / n as f32;
        push_point(&mut vertices, angle_outer, cx, cy, outer_radius);

        let angle_inner = angle_outer + step_angle;
        push_point(&mut vertices, angle_inner, cx, cy, inner_radius);
    }

    vertices
}

/// Generate outline points scaled to an arbitrary rectangle.
pub fn outline_points_for_rect(
    rect_width: f32,
    rect_height: f32,
    shape: &CropShape,
) -> Vec<(f32, f32)> {
    let width_px = rect_width.max(1.0).round() as u32;
    let height_px = rect_height.max(1.0).round() as u32;
    outline_points(width_px, height_px, shape)
        .into_iter()
        .map(|p| (p.x, p.y))
        .collect()
}

#[inline]
fn push_point(points: &mut Vec<Point>, angle: f32, cx: f32, cy: f32, radius: f32) {
    points.push(Point {
        x: angle.cos().mul_add(radius, cx),
        y: angle.sin().mul_add(radius, cy),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every probe below is placed on an exact geometric landmark, but the
    /// trig involved means `cos(PI/2)` lands a few ULPs off zero. The
    /// tolerance is far below any error an operator swap could produce.
    #[track_caller]
    fn approx_point(actual: Point, x: f32, y: f32) {
        assert!(
            (actual.x - x).abs() < 1e-3 && (actual.y - y).abs() < 1e-3,
            "expected ({x}, {y}), got ({}, {})",
            actual.x,
            actual.y
        );
    }

    fn square() -> Vec<Point> {
        vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(0.0, 10.0),
        ]
    }

    #[test]
    fn distance_is_euclidean() {
        assert_eq!(distance(Point::new(0.0, 0.0), Point::new(3.0, 4.0)), 5.0);
        assert_eq!(distance(Point::new(3.0, 4.0), Point::new(0.0, 0.0)), 5.0);
        assert_eq!(distance(Point::new(1.0, 1.0), Point::new(1.0, 1.0)), 0.0);
    }

    #[test]
    fn normalize_returns_unit_vector_and_guards_zero_length() {
        approx_point(normalize(Point::new(0.0, -10.0)), 0.0, -1.0);
        approx_point(normalize(Point::new(3.0, 4.0)), 0.6, 0.8);
        // Degenerate input must not divide by zero.
        approx_point(normalize(Point::new(0.0, 0.0)), 0.0, 0.0);
    }

    #[test]
    fn cubic_bezier_hits_exact_interpolants() {
        let p0 = Point::new(0.0, 0.0);
        let p1 = Point::new(1.0, 2.0);
        let p2 = Point::new(3.0, 4.0);
        let p3 = Point::new(4.0, 0.0);

        // Endpoints are interpolated exactly.
        approx_point(cubic_bezier(p0, p1, p2, p3, 0.0), 0.0, 0.0);
        approx_point(cubic_bezier(p0, p1, p2, p3, 1.0), 4.0, 0.0);

        // t = 0.5 weights the control points 1/8, 3/8, 3/8, 1/8.
        approx_point(cubic_bezier(p0, p1, p2, p3, 0.5), 2.0, 2.25);

        // t = 0.25 gives four distinct weights (0.421875, 0.421875,
        // 0.140625, 0.015625), so a swapped coefficient cannot hide.
        approx_point(cubic_bezier(p0, p1, p2, p3, 0.25), 0.90625, 1.40625);
    }

    #[test]
    fn chamfered_rect_points_cuts_each_corner() {
        let pts = chamfered_rect_points(10.0, 8.0, 2.0);
        let expected = [
            (2.0, 0.0),
            (8.0, 0.0),
            (10.0, 2.0),
            (10.0, 6.0),
            (8.0, 8.0),
            (2.0, 8.0),
            (0.0, 6.0),
            (0.0, 2.0),
        ];
        assert_eq!(pts.len(), expected.len());
        for (got, (x, y)) in pts.iter().zip(expected) {
            approx_point(*got, x, y);
        }
    }

    #[test]
    fn chamfered_rect_points_degenerates_to_a_rectangle() {
        let pts = chamfered_rect_points(10.0, 8.0, 0.0);
        assert_eq!(pts.len(), 4);
        approx_point(pts[0], 0.0, 0.0);
        approx_point(pts[2], 10.0, 8.0);
    }

    #[test]
    fn rounded_rect_points_lands_on_the_tangent_points() {
        // 16 segments per corner -> 17 points each, 68 total.
        let pts = rounded_rect_points(10.0, 8.0, 2.0, 16);
        assert_eq!(pts.len(), 68);

        // Each corner arc starts and ends where the straight edges meet it.
        approx_point(pts[0], 8.0, 0.0);
        approx_point(pts[16], 10.0, 2.0);
        approx_point(pts[17], 10.0, 6.0);
        approx_point(pts[33], 8.0, 8.0);
        approx_point(pts[34], 2.0, 8.0);
        approx_point(pts[50], 0.0, 6.0);
        approx_point(pts[51], 0.0, 2.0);
        approx_point(pts[67], 2.0, 0.0);
    }

    #[test]
    fn rounded_rect_points_degenerates_to_a_rectangle() {
        let pts = rounded_rect_points(10.0, 8.0, 0.0, 16);
        assert_eq!(pts.len(), 4);
        approx_point(pts[1], 10.0, 0.0);
        approx_point(pts[3], 0.0, 8.0);
    }

    #[test]
    fn fit_points_to_bounds_scales_uniformly_and_centres() {
        // 2x1 bbox into a 10x10 box: the limiting axis is x, so scale = 5 and
        // the 5-tall result is centred vertically with a 2.5 offset.
        let mut pts = vec![
            Point::new(0.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(2.0, 1.0),
            Point::new(0.0, 1.0),
        ];
        fit_points_to_bounds(&mut pts, 10.0, 10.0);
        approx_point(pts[0], 0.0, 2.5);
        approx_point(pts[1], 10.0, 2.5);
        approx_point(pts[2], 10.0, 7.5);
        approx_point(pts[3], 0.0, 7.5);
    }

    #[test]
    fn fit_points_to_bounds_is_translation_invariant() {
        // Same shape shifted away from the origin must land identically —
        // this is what catches a dropped `-min * scale` offset term.
        let mut pts = vec![
            Point::new(1.0, 1.0),
            Point::new(3.0, 1.0),
            Point::new(3.0, 2.0),
            Point::new(1.0, 2.0),
        ];
        fit_points_to_bounds(&mut pts, 10.0, 10.0);
        approx_point(pts[0], 0.0, 2.5);
        approx_point(pts[2], 10.0, 7.5);
    }

    #[test]
    fn fit_points_to_bounds_leaves_degenerate_input_alone() {
        // Zero-area bbox would divide by zero.
        let mut collapsed = vec![Point::new(4.0, 4.0), Point::new(4.0, 4.0)];
        fit_points_to_bounds(&mut collapsed, 10.0, 10.0);
        approx_point(collapsed[0], 4.0, 4.0);

        let mut empty: Vec<Point> = Vec::new();
        fit_points_to_bounds(&mut empty, 10.0, 10.0);
        assert!(empty.is_empty());
    }

    #[test]
    fn koch_fractal_replaces_each_edge_with_four() {
        // One segment pair (there and back) subdivided once. Each edge becomes
        // p0, p0+d/3, the apex, p0+2d/3 — the apex sits sqrt(3)/6 of the edge
        // length off to one side.
        let base = vec![Point::new(0.0, 0.0), Point::new(3.0, 0.0)];
        let out = koch_fractal(&base, 1);
        assert_eq!(out.len(), 8);

        approx_point(out[0], 0.0, 0.0);
        approx_point(out[1], 1.0, 0.0);
        approx_point(out[2], 1.5, -KOCH_SIN_60);
        approx_point(out[3], 2.0, 0.0);
        // The return edge bulges the opposite way.
        approx_point(out[4], 3.0, 0.0);
        approx_point(out[5], 2.0, 0.0);
        approx_point(out[6], 1.5, KOCH_SIN_60);
        approx_point(out[7], 1.0, 0.0);
    }

    #[test]
    fn koch_fractal_iteration_count_drives_growth() {
        let base = square();
        // Zero iterations is a straight passthrough.
        assert_eq!(koch_fractal(&base, 0), base);
        // Each iteration quadruples the vertex count.
        assert_eq!(koch_fractal(&base, 1).len(), 16);
        assert_eq!(koch_fractal(&base, 2).len(), 64);
    }

    #[test]
    fn polygon_points_places_vertices_on_the_circumcircle() {
        // A 4-gon in a 20x20 box: radius 10 about (10, 10), first vertex at
        // angle 0 and the rest counter-clockwise in screen coordinates.
        let pts = polygon_points(20.0, 20.0, 4, 0.0, PolygonCornerStyle::Sharp);
        assert_eq!(pts.len(), 4);
        approx_point(pts[0], 20.0, 10.0);
        approx_point(pts[1], 10.0, 20.0);
        approx_point(pts[2], 0.0, 10.0);
        approx_point(pts[3], 10.0, 0.0);
    }

    #[test]
    fn polygon_points_applies_rotation_and_minimum_sides() {
        // 90 degrees of rotation moves the first vertex a quarter turn on.
        let rotated = polygon_points(20.0, 20.0, 4, 90.0, PolygonCornerStyle::Sharp);
        approx_point(rotated[0], 10.0, 20.0);

        // Fewer than three sides is clamped up to a triangle.
        assert_eq!(
            polygon_points(20.0, 20.0, 1, 0.0, PolygonCornerStyle::Sharp).len(),
            MIN_POLYGON_SIDES as usize
        );
    }

    #[test]
    fn star_points_alternates_outer_and_inner_radii() {
        // 4-point star in a 20x20 box: outer radius 10, inner 5, and the inner
        // vertices sit half a step (PI/4) past each outer one.
        let pts = star_points(20.0, 20.0, 4, 0.5, 0.0);
        assert_eq!(pts.len(), 8);

        let diag = 5.0 * std::f32::consts::FRAC_1_SQRT_2;
        approx_point(pts[0], 20.0, 10.0);
        approx_point(pts[1], 10.0 + diag, 10.0 + diag);
        approx_point(pts[2], 10.0, 20.0);
        approx_point(pts[3], 10.0 - diag, 10.0 + diag);
        approx_point(pts[4], 0.0, 10.0);
        approx_point(pts[6], 10.0, 0.0);
    }

    #[test]
    fn chamfer_polygon_trims_each_vertex_along_both_edges() {
        // A 10x10 square inset by 2 yields two points per corner, each 2 units
        // back from the vertex along the incoming and outgoing edges.
        let pts = chamfer_polygon(&square(), 2.0);
        let expected = [
            (0.0, 2.0),
            (2.0, 0.0),
            (8.0, 0.0),
            (10.0, 2.0),
            (10.0, 8.0),
            (8.0, 10.0),
            (2.0, 10.0),
            (0.0, 8.0),
        ];
        assert_eq!(pts.len(), expected.len());
        for (got, (x, y)) in pts.iter().zip(expected) {
            approx_point(*got, x, y);
        }
    }

    #[test]
    fn chamfer_polygon_clamps_inset_to_half_the_edge() {
        // An inset larger than half the edge would overshoot past the midpoint
        // and invert the corner; it must clamp to exactly the midpoint.
        let pts = chamfer_polygon(&square(), 50.0);
        approx_point(pts[0], 0.0, 5.0);
        approx_point(pts[1], 5.0, 0.0);

        // Non-positive inset is a passthrough.
        assert_eq!(chamfer_polygon(&square(), 0.0), square());
    }

    #[test]
    fn rounded_polygon_arcs_between_the_tangent_points() {
        // Right-angle corners: half-angle PI/4, so offset == radius and the
        // arc centre sits at (2, 2) for the corner at the origin.
        let pts = rounded_polygon(&square(), 2.0, 8);
        // 4 corners x (8 segments + 1) points.
        assert_eq!(pts.len(), 36);

        // The arc runs from the tangent point on the incoming edge to the one
        // on the outgoing edge, sweeping a quarter turn.
        approx_point(pts[0], 0.0, 2.0);
        approx_point(pts[8], 2.0, 0.0);
        // Midway round the arc, 45 degrees from centre (2, 2) at radius 2.
        let inset = 2.0 - 2.0 * std::f32::consts::FRAC_1_SQRT_2;
        approx_point(pts[4], inset, inset);

        // The next corner picks up at the far end of the top edge.
        approx_point(pts[9], 8.0, 0.0);
        approx_point(pts[17], 10.0, 2.0);
    }

    #[test]
    fn rounded_polygon_passthrough_on_zero_radius() {
        assert_eq!(rounded_polygon(&square(), 0.0, 8), square());
    }

    #[test]
    fn bezier_polygon_passes_through_every_vertex() {
        let pts = bezier_polygon(&square(), 0.5, 16);
        // One run of `segments` points per edge.
        assert_eq!(pts.len(), 64);

        // t = 0 on each edge reproduces that edge's starting vertex exactly.
        approx_point(pts[0], 0.0, 0.0);
        approx_point(pts[16], 10.0, 0.0);
        approx_point(pts[32], 10.0, 10.0);
        approx_point(pts[48], 0.0, 10.0);

        // Non-positive tension is a passthrough.
        assert_eq!(bezier_polygon(&square(), 0.0, 16), square());
    }

    #[test]
    fn outline_points_fits_complex_shapes_to_the_bounds() {
        // A polygon is refitted to fill the box; a plain rectangle is not.
        let shape = CropShape::Polygon {
            sides: 5,
            rotation_deg: 0.0,
            corner_style: PolygonCornerStyle::Sharp,
        };
        let pts = outline_points(100, 100, &shape);

        let min_x = pts.iter().map(|p| p.x).fold(f32::MAX, f32::min);
        let max_x = pts.iter().map(|p| p.x).fold(f32::MIN, f32::max);
        let min_y = pts.iter().map(|p| p.y).fold(f32::MAX, f32::min);
        let max_y = pts.iter().map(|p| p.y).fold(f32::MIN, f32::max);

        // The fitted shape touches the box on its limiting axis and stays
        // inside on the other.
        assert!(min_x >= -1e-3 && max_x <= 100.0 + 1e-3);
        assert!(min_y >= -1e-3 && max_y <= 100.0 + 1e-3);
        assert!(
            (max_x - min_x - 100.0).abs() < 1e-2 || (max_y - min_y - 100.0).abs() < 1e-2,
            "fitted shape should span the box on at least one axis"
        );
    }

    #[test]
    fn outline_points_rectangle_and_ellipse_span_the_box() {
        let rect = outline_points(10, 8, &CropShape::Rectangle);
        assert_eq!(rect.len(), 4);
        approx_point(rect[0], 0.0, 0.0);
        approx_point(rect[2], 10.0, 8.0);

        // Ellipse starts at angle 0 — the rightmost point, vertically centred.
        let ellipse = outline_points(10, 8, &CropShape::Ellipse);
        assert_eq!(ellipse.len(), ELLIPSE_SEGMENTS);
        approx_point(ellipse[0], 10.0, 4.0);
        approx_point(ellipse[ELLIPSE_SEGMENTS / 4], 5.0, 8.0);
        approx_point(ellipse[ELLIPSE_SEGMENTS / 2], 0.0, 4.0);
    }

    #[test]
    fn outline_points_scales_corner_parameters_to_the_short_side() {
        // 10x8 box: the corner parameter is a fraction of min(w, h) = 8, so
        // 0.25 gives a radius/inset of 2. Note `sanitized()` caps these at 0.5,
        // which means the `.clamp(.., min*0.5)` upper bound can never bind.
        let rounded = outline_points(10, 8, &CropShape::RoundedRectangle { radius_pct: 0.25 });
        assert_eq!(
            rounded,
            rounded_rect_points(10.0, 8.0, 2.0, ROUNDED_RECT_CORNER_SEGMENTS)
        );

        let chamfered = outline_points(10, 8, &CropShape::ChamferedRectangle { size_pct: 0.25 });
        assert_eq!(chamfered, chamfered_rect_points(10.0, 8.0, 2.0));

        // At the sanitizer's ceiling the parameter reaches exactly half the
        // short side.
        let max_round = outline_points(10, 8, &CropShape::RoundedRectangle { radius_pct: 0.5 });
        assert_eq!(
            max_round,
            rounded_rect_points(10.0, 8.0, 4.0, ROUNDED_RECT_CORNER_SEGMENTS)
        );
    }

    #[test]
    fn polygon_points_scales_corner_styles_by_the_short_side() {
        let sharp = polygon_points(20.0, 20.0, 4, 0.0, PolygonCornerStyle::Sharp);

        // inset = min(w, h) * 0.1 = 2, well inside the circumradius of 10.
        let chamfered = polygon_points(
            20.0,
            20.0,
            4,
            0.0,
            PolygonCornerStyle::Chamfered { size_pct: 0.1 },
        );
        assert_eq!(chamfered, chamfer_polygon(&sharp, 2.0));

        let rounded = polygon_points(
            20.0,
            20.0,
            4,
            0.0,
            PolygonCornerStyle::Rounded { radius_pct: 0.1 },
        );
        assert_eq!(
            rounded,
            rounded_polygon(&sharp, 2.0, ROUNDED_POLYGON_CORNER_SEGMENTS)
        );

        let bezier = polygon_points(
            20.0,
            20.0,
            4,
            0.0,
            PolygonCornerStyle::Bezier { tension: 0.5 },
        );
        assert_eq!(bezier, bezier_polygon(&sharp, 0.5, BEZIER_POLYGON_SEGMENTS));
    }

    #[test]
    fn rounded_polygon_handles_corners_that_are_not_right_angles() {
        // On a square every corner is 90 degrees, where half_angle is PI/4 and
        // tan(PI/4) == 1 — so `radius / tan(half_angle)` and a sign flip on the
        // incoming vector both vanish. A right isoceles triangle has a 45
        // degree corner at (10, 0), where neither cancels out.
        let tri = vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(0.0, 10.0),
        ];
        let pts = rounded_polygon(&tri, 1.0, 8);
        assert_eq!(pts.len(), 27);

        // Corner 1 is the vertex at (10, 0): half_angle = PI/8, so the tangent
        // points sit 1/tan(PI/8) = 1 + sqrt(2) back along each edge.
        let offset = 1.0 + std::f32::consts::SQRT_2;
        approx_point(pts[9], 10.0 - offset, 0.0);
        let diag = offset * std::f32::consts::FRAC_1_SQRT_2;
        approx_point(pts[17], 10.0 - diag, diag);
    }

    #[test]
    fn rounded_polygon_clamps_the_arc_to_half_each_edge() {
        // A radius far larger than the shape: without the `edge_len * 0.5`
        // clamp the arc swings outside the polygon entirely.
        let pts = rounded_polygon(&square(), 20.0, 8);
        for p in &pts {
            assert!(
                p.x >= -0.01 && p.x <= 10.01 && p.y >= -0.01 && p.y <= 10.01,
                "arc point ({}, {}) escaped the polygon bounds",
                p.x,
                p.y
            );
        }
    }

    #[test]
    fn bezier_polygon_curves_between_the_vertices() {
        // The vertex-only assertions elsewhere are all at t = 0, which returns
        // p0 regardless of the control points. These probe mid-segment, where
        // the tangent and control-point indexing actually matter.
        //
        // Segment 0 runs (0,0) -> (10,0) with control points (2.5,-2.5) and
        // (7.5,-2.5); at t = 0.5 the weights are 1/8, 3/8, 3/8, 1/8.
        let pts = bezier_polygon(&square(), 0.5, 16);
        approx_point(pts[8], 5.0, -1.875);
        // Segment 1 runs (10,0) -> (10,10) via (12.5,2.5) and (12.5,7.5).
        approx_point(pts[24], 11.875, 5.0);
    }

    #[test]
    fn fit_points_to_bounds_ignores_a_single_collapsed_axis() {
        // A vertical line has zero width. The guard has to trip when *either*
        // axis is degenerate — requiring both would divide by a zero bbox.
        let mut line = vec![Point::new(4.0, 0.0), Point::new(4.0, 10.0)];
        fit_points_to_bounds(&mut line, 10.0, 10.0);
        approx_point(line[0], 4.0, 0.0);
        approx_point(line[1], 4.0, 10.0);
    }

    #[test]
    fn outline_points_treats_zero_dimensions_as_one_pixel() {
        // width.max(1) guards against a zero-area box.
        let pts = outline_points(0, 0, &CropShape::Rectangle);
        assert_eq!(pts.len(), 4);
        approx_point(pts[2], 1.0, 1.0);
    }
}
