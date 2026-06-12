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
    #[default]
    Sharp,
    Rounded {
        radius_pct: f32,
    },
    Chamfered {
        size_pct: f32,
    },
    Bezier {
        tension: f32,
    },
}

/// Shapes supported by the crop exporter.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CropShape {
    #[default]
    Rectangle,
    RoundedRectangle {
        radius_pct: f32,
    },
    ChamferedRectangle {
        size_pct: f32,
    },
    Ellipse,
    Polygon {
        sides: u8,
        rotation_deg: f32,
        #[serde(default)]
        corner_style: PolygonCornerStyle,
    },
    Star {
        points: u8,
        inner_radius_pct: f32,
        rotation_deg: f32,
    },
    KochPolygon {
        sides: u8,
        rotation_deg: f32,
        iterations: u8,
    },
    KochRectangle {
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
