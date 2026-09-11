//! Shape metadata shared by crop export and previews.

use serde::{Deserialize, Serialize};

/// Maximum Koch fractal iterations (to prevent excessive computation).
pub(super) const MAX_KOCH_ITERATIONS: u8 = 5;
/// Minimum number of polygon sides.
pub(super) const MIN_POLYGON_SIDES: u8 = 3;

/// Polygon corner styles.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(tag = "style", rename_all = "snake_case")]
pub enum PolygonCornerStyle {
    /// Keep the polygon's original vertices.
    #[default]
    Sharp,
    /// Round each corner with an arc.
    Rounded {
        /// Radius as a fraction of the shorter image side, clamped to 0..=0.5.
        radius_pct: f32,
    },
    /// Replace corners with straight bevels.
    Chamfered {
        /// Inset as a fraction of the shorter image side, clamped to 0..=0.5.
        size_pct: f32,
    },
    /// Connect vertices with cubic Bezier curves.
    Bezier {
        /// Curve tangent scale, clamped to 0..=2; zero keeps sharp vertices.
        tension: f32,
    },
}

/// Shapes supported by the crop exporter.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CropShape {
    /// Full rectangular image bounds without masking.
    #[default]
    Rectangle,
    /// Rectangle with rounded corners.
    RoundedRectangle {
        /// Corner radius as a fraction of the shorter side, clamped to 0..=0.5.
        radius_pct: f32,
    },
    /// Rectangle with straight bevelled corners.
    ChamferedRectangle {
        /// Corner inset as a fraction of the shorter side, clamped to 0..=0.5.
        size_pct: f32,
    },
    /// Ellipse fitted to the image bounds.
    Ellipse,
    /// Regular polygon with configurable corners.
    Polygon {
        /// Number of sides, raised to at least 3 when sanitized.
        sides: u8,
        /// Clockwise rotation in degrees in image coordinates.
        rotation_deg: f32,
        /// Treatment of each polygon vertex; defaults to sharp.
        #[serde(default)]
        corner_style: PolygonCornerStyle,
    },
    /// Star alternating between outer and inner vertices.
    Star {
        /// Number of outer points, raised to at least 3 when sanitized.
        points: u8,
        /// Inner radius divided by outer radius, clamped to 0..=1.
        inner_radius_pct: f32,
        /// Clockwise rotation in degrees in image coordinates.
        rotation_deg: f32,
    },
    /// Koch-fractal outline grown from a regular polygon.
    KochPolygon {
        /// Number of base polygon sides, raised to at least 3 when sanitized.
        sides: u8,
        /// Clockwise rotation of the base polygon in degrees.
        rotation_deg: f32,
        /// Number of Koch subdivisions, capped at 5; zero preserves the base polygon.
        iterations: u8,
    },
    /// Koch-fractal outline grown from the image rectangle.
    KochRectangle {
        /// Number of Koch subdivisions, capped at 5; zero preserves the rectangle.
        iterations: u8,
    },
}

impl CropShape {
    /// Sanitize values to keep them in a sensible range.
    pub fn sanitized(&self) -> Self {
        match self {
            CropShape::Rectangle => CropShape::Rectangle,
            CropShape::RoundedRectangle { radius_pct } => CropShape::RoundedRectangle {
                radius_pct: radius_pct.clamp(0.0, 0.5),
            },
            CropShape::ChamferedRectangle { size_pct } => CropShape::ChamferedRectangle {
                size_pct: size_pct.clamp(0.0, 0.5),
            },
            CropShape::Ellipse => Self::Ellipse,
            CropShape::Polygon {
                sides,
                rotation_deg,
                corner_style,
            } => CropShape::Polygon {
                sides: (*sides).max(MIN_POLYGON_SIDES),
                rotation_deg: *rotation_deg,
                corner_style: match corner_style {
                    PolygonCornerStyle::Sharp => PolygonCornerStyle::Sharp,
                    PolygonCornerStyle::Rounded { radius_pct } => PolygonCornerStyle::Rounded {
                        radius_pct: radius_pct.clamp(0.0, 0.5),
                    },
                    PolygonCornerStyle::Chamfered { size_pct } => PolygonCornerStyle::Chamfered {
                        size_pct: size_pct.clamp(0.0, 0.5),
                    },
                    PolygonCornerStyle::Bezier { tension } => PolygonCornerStyle::Bezier {
                        tension: tension.clamp(0.0, 2.0),
                    },
                },
            },
            CropShape::Star {
                points,
                inner_radius_pct,
                rotation_deg,
            } => CropShape::Star {
                points: (*points).max(MIN_POLYGON_SIDES),
                inner_radius_pct: inner_radius_pct.clamp(0.0, 1.0),
                rotation_deg: *rotation_deg,
            },
            CropShape::KochPolygon {
                sides,
                rotation_deg,
                iterations,
            } => CropShape::KochPolygon {
                sides: (*sides).max(MIN_POLYGON_SIDES),
                rotation_deg: *rotation_deg,
                iterations: (*iterations).min(MAX_KOCH_ITERATIONS),
            },
            CropShape::KochRectangle { iterations } => CropShape::KochRectangle {
                iterations: (*iterations).min(MAX_KOCH_ITERATIONS),
            },
        }
    }
}
