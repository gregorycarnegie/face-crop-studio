//! Shape utilities for custom crop geometry and masking.
//!
//! Provides the `CropShape` enum shared across the workspace together with helpers
//! for generating polygon outlines and applying alpha masks to RGBA images.

mod mask;
mod outline;
#[cfg(test)]
mod tests;
mod types;

pub use mask::{apply_shape_mask, apply_shape_mask_dynamic};
pub use outline::outline_points_for_rect;
pub use types::{CropShape, PolygonCornerStyle};

/// The shapes a shape picker offers, in display order: a label and the shape it starts from.
/// Every front end lists these, so they agree on names and starting parameters.
pub fn shape_catalogue() -> Vec<(&'static str, CropShape)> {
    let polygon = |corner_style| CropShape::Polygon {
        sides: 6,
        rotation_deg: 0.0,
        corner_style,
    };
    vec![
        ("Rectangle", CropShape::Rectangle),
        (
            "Rounded rectangle",
            CropShape::RoundedRectangle { radius_pct: 0.12 },
        ),
        (
            "Chamfered rectangle",
            CropShape::ChamferedRectangle { size_pct: 0.12 },
        ),
        ("Ellipse", CropShape::Ellipse),
        ("Polygon", polygon(PolygonCornerStyle::Sharp)),
        (
            "Polygon (rounded)",
            polygon(PolygonCornerStyle::Rounded { radius_pct: 0.1 }),
        ),
        (
            "Polygon (chamfered)",
            polygon(PolygonCornerStyle::Chamfered { size_pct: 0.1 }),
        ),
        (
            "Polygon (bezier)",
            polygon(PolygonCornerStyle::Bezier { tension: 0.5 }),
        ),
        (
            "Star",
            CropShape::Star {
                points: 5,
                inner_radius_pct: 0.5,
                rotation_deg: 0.0,
            },
        ),
        (
            "Koch polygon",
            CropShape::KochPolygon {
                sides: 3,
                rotation_deg: 0.0,
                iterations: 3,
            },
        ),
        ("Koch rectangle", CropShape::KochRectangle { iterations: 3 }),
    ]
}
