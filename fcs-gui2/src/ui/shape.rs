//! Crop shape controls panel for the inspector.

use crate::types::App2;
use crate::ui::widgets::{field_label, slider_with_label};
use egui::Ui;
use fcs_utils::{CropShape, PolygonCornerStyle};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ShapeVariant {
    Rectangle,
    RoundedRect,
    ChamferRect,
    Ellipse,
    PolygonSharp,
    PolygonRounded,
    PolygonChamfered,
    PolygonBezier,
    Star,
    KochPolygon,
    KochRectangle,
}

fn shape_variant(shape: &CropShape) -> ShapeVariant {
    match shape {
        CropShape::Rectangle => ShapeVariant::Rectangle,
        CropShape::RoundedRectangle { .. } => ShapeVariant::RoundedRect,
        CropShape::ChamferedRectangle { .. } => ShapeVariant::ChamferRect,
        CropShape::Ellipse => ShapeVariant::Ellipse,
        CropShape::Polygon { corner_style, .. } => match corner_style {
            PolygonCornerStyle::Sharp => ShapeVariant::PolygonSharp,
            PolygonCornerStyle::Rounded { .. } => ShapeVariant::PolygonRounded,
            PolygonCornerStyle::Chamfered { .. } => ShapeVariant::PolygonChamfered,
            PolygonCornerStyle::Bezier { .. } => ShapeVariant::PolygonBezier,
        },
        CropShape::Star { .. } => ShapeVariant::Star,
        CropShape::KochPolygon { .. } => ShapeVariant::KochPolygon,
        CropShape::KochRectangle { .. } => ShapeVariant::KochRectangle,
    }
}

fn default_for_variant(variant: ShapeVariant) -> CropShape {
    match variant {
        ShapeVariant::Rectangle => CropShape::Rectangle,
        ShapeVariant::RoundedRect => CropShape::RoundedRectangle { radius_pct: 0.12 },
        ShapeVariant::ChamferRect => CropShape::ChamferedRectangle { size_pct: 0.12 },
        ShapeVariant::Ellipse => CropShape::Ellipse,
        ShapeVariant::PolygonSharp => CropShape::Polygon {
            sides: 6, rotation_deg: 0.0, corner_style: PolygonCornerStyle::Sharp,
        },
        ShapeVariant::PolygonRounded => CropShape::Polygon {
            sides: 6, rotation_deg: 0.0, corner_style: PolygonCornerStyle::Rounded { radius_pct: 0.1 },
        },
        ShapeVariant::PolygonChamfered => CropShape::Polygon {
            sides: 6, rotation_deg: 0.0, corner_style: PolygonCornerStyle::Chamfered { size_pct: 0.1 },
        },
        ShapeVariant::PolygonBezier => CropShape::Polygon {
            sides: 6, rotation_deg: 0.0, corner_style: PolygonCornerStyle::Bezier { tension: 0.5 },
        },
        ShapeVariant::Star => CropShape::Star { points: 5, inner_radius_pct: 0.5, rotation_deg: 0.0 },
        ShapeVariant::KochPolygon => CropShape::KochPolygon { sides: 3, rotation_deg: 0.0, iterations: 3 },
        ShapeVariant::KochRectangle => CropShape::KochRectangle { iterations: 3 },
    }
}

fn polygon_corner_radius_max(sides: u32) -> f32 {
    0.5 * (std::f32::consts::PI / sides as f32).cos()
}

fn polygon_chamfer_size_max(sides: u32) -> f32 {
    0.5 * (std::f32::consts::PI / sides as f32).sin()
}

/// Returns whether any shape or vignette setting changed.
pub fn shape_controls(ui: &mut Ui, app: &mut App2) -> bool {
    let mut changed = false;
    let mut shape = app.settings.crop.shape.clone();
    let mut variant = shape_variant(&shape);

    // Shape type selector
    field_label(ui, "Shape");
    let mut variant_changed = false;
    egui::ComboBox::from_id_salt("shape_combo")
        .selected_text(variant_label(variant))
        .width(ui.available_width() - 28.0)
        .show_ui(ui, |ui| {
            for (label, target) in ALL_VARIANTS {
                if ui.selectable_label(variant == *target, *label).clicked() && variant != *target {
                    variant = *target;
                    variant_changed = true;
                }
            }
        });

    if variant_changed {
        shape = default_for_variant(variant);
        changed = true;
    }

    // Shape-specific parameters
    match &mut shape {
        CropShape::RoundedRectangle { radius_pct } => {
            let mut pct = (*radius_pct * 100.0).clamp(0.0, 50.0);
            field_label(ui, &format!("Corner radius · {pct:.0}%"));
            if slider_with_label(ui, "", &mut pct, 0.0, 50.0, "pct") {
                *radius_pct = (pct / 100.0).clamp(0.0, 0.5);
                changed = true;
            }
        }
        CropShape::ChamferedRectangle { size_pct } => {
            let mut pct = (*size_pct * 100.0).clamp(0.0, 50.0);
            field_label(ui, &format!("Chamfer size · {pct:.0}%"));
            if slider_with_label(ui, "", &mut pct, 0.0, 50.0, "pct") {
                *size_pct = (pct / 100.0).clamp(0.0, 0.5);
                changed = true;
            }
        }
        CropShape::Polygon { sides, rotation_deg, corner_style } => {
            let mut s = *sides as u32;
            field_label(ui, "Sides");
            if ui.add(egui::DragValue::new(&mut s).range(3..=24).speed(0.1)).changed() {
                *sides = s.clamp(3, 24) as u8;
                changed = true;
            }

            field_label(ui, &format!("Rotation · {rotation_deg:.0}°"));
            if slider_with_label(ui, "", rotation_deg, -180.0, 180.0, "deg") {
                changed = true;
            }

            match corner_style {
                PolygonCornerStyle::Sharp => {}
                PolygonCornerStyle::Rounded { radius_pct } => {
                    let max = polygon_corner_radius_max(s) * 100.0;
                    let mut pct = (*radius_pct * 100.0).clamp(0.0, max);
                    field_label(ui, &format!("Corner radius · {pct:.1}%"));
                    if slider_with_label(ui, "", &mut pct, 0.0, max, "pct") {
                        *radius_pct = (pct / 100.0).clamp(0.0, polygon_corner_radius_max(s));
                        changed = true;
                    }
                }
                PolygonCornerStyle::Chamfered { size_pct } => {
                    let max = polygon_chamfer_size_max(s) * 100.0;
                    let mut pct = (*size_pct * 100.0).clamp(0.0, max);
                    field_label(ui, &format!("Chamfer size · {pct:.1}%"));
                    if slider_with_label(ui, "", &mut pct, 0.0, max, "pct") {
                        *size_pct = (pct / 100.0).clamp(0.0, polygon_chamfer_size_max(s));
                        changed = true;
                    }
                }
                PolygonCornerStyle::Bezier { tension } => {
                    field_label(ui, &format!("Tension · {tension:.2}"));
                    if slider_with_label(ui, "", tension, 0.0, 2.0, "") {
                        changed = true;
                    }
                }
            }
        }
        CropShape::Star { points, inner_radius_pct, rotation_deg } => {
            let mut p = *points as u32;
            field_label(ui, "Points");
            if ui.add(egui::DragValue::new(&mut p).range(3..=24).speed(0.1)).changed() {
                *points = p.clamp(3, 24) as u8;
                changed = true;
            }

            let mut inner = (*inner_radius_pct * 100.0).clamp(10.0, 90.0);
            field_label(ui, &format!("Inner radius · {inner:.0}%"));
            if slider_with_label(ui, "", &mut inner, 10.0, 90.0, "pct") {
                *inner_radius_pct = (inner / 100.0).clamp(0.1, 0.9);
                changed = true;
            }

            field_label(ui, &format!("Rotation · {rotation_deg:.0}°"));
            if slider_with_label(ui, "", rotation_deg, -180.0, 180.0, "deg") {
                changed = true;
            }
        }
        CropShape::KochPolygon { sides, rotation_deg, iterations } => {
            let mut s = *sides as u32;
            field_label(ui, "Sides");
            if ui.add(egui::DragValue::new(&mut s).range(3..=24).speed(0.1)).changed() {
                *sides = s.clamp(3, 24) as u8;
                changed = true;
            }

            field_label(ui, &format!("Rotation · {rotation_deg:.0}°"));
            if slider_with_label(ui, "", rotation_deg, -180.0, 180.0, "deg") {
                changed = true;
            }

            let mut iter = *iterations as u32;
            field_label(ui, "Iterations");
            if ui.add(egui::DragValue::new(&mut iter).range(0..=5).speed(0.1)).changed() {
                *iterations = iter.clamp(0, 5) as u8;
                changed = true;
            }
        }
        CropShape::KochRectangle { iterations } => {
            let mut iter = *iterations as u32;
            field_label(ui, "Iterations");
            if ui.add(egui::DragValue::new(&mut iter).range(0..=5).speed(0.1)).changed() {
                *iterations = iter.clamp(0, 5) as u8;
                changed = true;
            }
        }
        CropShape::Rectangle | CropShape::Ellipse => {}
    }

    if changed {
        let sanitized = shape.sanitized();
        app.settings.crop.shape = sanitized;
    }

    // Vignette controls
    ui.add_space(4.0);
    let mut softness = app.settings.crop.vignette_softness * 100.0;
    field_label(ui, &format!("Vignette softness · {softness:.0}%"));
    if slider_with_label(ui, "", &mut softness, 0.0, 100.0, "pct") {
        app.settings.crop.vignette_softness = (softness / 100.0).clamp(0.0, 1.0);
        changed = true;
    }

    let mut intensity = app.settings.crop.vignette_intensity * 100.0;
    field_label(ui, &format!("Vignette intensity · {intensity:.0}%"));
    if slider_with_label(ui, "", &mut intensity, 0.0, 100.0, "pct") {
        app.settings.crop.vignette_intensity = (intensity / 100.0).clamp(0.0, 1.0);
        changed = true;
    }

    field_label(ui, "Vignette color");
    let vc = &mut app.settings.crop.vignette_color;
    let mut color = [vc.red, vc.green, vc.blue, vc.alpha];
    if ui.color_edit_button_srgba_unmultiplied(&mut color).changed() {
        vc.red   = color[0];
        vc.green = color[1];
        vc.blue  = color[2];
        vc.alpha = color[3];
        changed = true;
    }

    changed
}

fn variant_label(v: ShapeVariant) -> &'static str {
    match v {
        ShapeVariant::Rectangle       => "Rectangle",
        ShapeVariant::RoundedRect     => "Rounded rectangle",
        ShapeVariant::ChamferRect     => "Chamfered rectangle",
        ShapeVariant::Ellipse         => "Ellipse",
        ShapeVariant::PolygonSharp    => "Polygon",
        ShapeVariant::PolygonRounded  => "Polygon (rounded)",
        ShapeVariant::PolygonChamfered => "Polygon (chamfered)",
        ShapeVariant::PolygonBezier   => "Polygon (bezier)",
        ShapeVariant::Star            => "Star",
        ShapeVariant::KochPolygon     => "Koch polygon",
        ShapeVariant::KochRectangle   => "Koch rectangle",
    }
}

const ALL_VARIANTS: &[(&str, ShapeVariant)] = &[
    ("Rectangle",           ShapeVariant::Rectangle),
    ("Rounded rectangle",   ShapeVariant::RoundedRect),
    ("Chamfered rectangle", ShapeVariant::ChamferRect),
    ("Ellipse",             ShapeVariant::Ellipse),
    ("Polygon",             ShapeVariant::PolygonSharp),
    ("Polygon (rounded)",   ShapeVariant::PolygonRounded),
    ("Polygon (chamfered)", ShapeVariant::PolygonChamfered),
    ("Polygon (bezier)",    ShapeVariant::PolygonBezier),
    ("Star",                ShapeVariant::Star),
    ("Koch polygon",        ShapeVariant::KochPolygon),
    ("Koch rectangle",      ShapeVariant::KochRectangle),
];
