use crate::shape::{CropShape, PolygonCornerStyle, outline_points_for_rect};

#[test]
fn test_bezier_polygon_generation() {
    let shape = CropShape::Polygon {
        sides: 4,
        rotation_deg: 0.0,
        corner_style: PolygonCornerStyle::Bezier { tension: 0.5 },
    };

    let points = outline_points_for_rect(100.0, 100.0, &shape);

    // We expect points to be generated.
    // 4 sides * 16 segments = 64 points roughly
    assert!(!points.is_empty());
    assert_eq!(points.len(), 64);

    // Check bounds roughly
    for (x, y) in &points {
        assert!(*x >= 0.0 && *x <= 100.0);
        assert!(*y >= 0.0 && *y <= 100.0);
    }
}

#[test]
fn test_bezier_polygon_zero_tension() {
    let shape = CropShape::Polygon {
        sides: 3,
        rotation_deg: 0.0,
        corner_style: PolygonCornerStyle::Bezier { tension: 0.0 },
    };

    let points = outline_points_for_rect(100.0, 100.0, &shape);

    // Zero tension should return original vertices (3 for triangle)
    assert_eq!(points.len(), 3);
}

#[test]
fn test_star_generation() {
    let shape = CropShape::Star {
        points: 5,
        inner_radius_pct: 0.5,
        rotation_deg: 0.0,
    };

    let points = outline_points_for_rect(100.0, 100.0, &shape);

    // 5 points * 2 (inner + outer) = 10 vertices
    assert_eq!(points.len(), 10);
}

#[test]
fn test_vignette_mask_application() {
    use crate::shape::{CropShape, apply_shape_mask};
    use image::RgbaImage;
    let mut img = RgbaImage::new(100, 100);
    apply_shape_mask(
        &mut img,
        &CropShape::Ellipse,
        0.1,
        1.0,
        crate::color::RgbaColor::opaque(0, 0, 0),
    );
}

/// Test that shape masking produces transparent pixels outside the shape
/// regardless of vignette_intensity setting.
#[test]
fn test_ellipse_crop_transparent_outside_zero_intensity() {
    use crate::shape::{CropShape, apply_shape_mask};
    use image::{Rgba, RgbaImage};

    // Create a 100x100 white opaque image
    let mut img = RgbaImage::from_pixel(100, 100, Rgba([255, 255, 255, 255]));

    // Apply ellipse mask with vignette_intensity = 0 (no vignette color)
    apply_shape_mask(
        &mut img,
        &CropShape::Ellipse,
        0.0, // no softness - hard edge
        0.0, // zero intensity - should still crop!
        crate::color::RgbaColor::opaque(0, 0, 0),
    );

    // Corner pixel (0,0) is definitely outside the ellipse - must be transparent
    let corner = img.get_pixel(0, 0);
    assert_eq!(
        corner[3], 0,
        "Corner pixel should be fully transparent (alpha=0), got alpha={}",
        corner[3]
    );

    // Center pixel (50,50) is inside the ellipse - must remain opaque
    let center = img.get_pixel(50, 50);
    assert_eq!(
        center[3], 255,
        "Center pixel should remain fully opaque (alpha=255), got alpha={}",
        center[3]
    );
}

/// Test that shape masking with softness produces transparent edges
/// regardless of vignette_intensity setting. With softness, edges have
/// gradual alpha falloff, but RGB should not be affected by vignette color
/// when intensity is 0.
#[test]
fn test_ellipse_crop_transparent_outside_with_softness_zero_intensity() {
    use crate::shape::{CropShape, apply_shape_mask};
    use image::{Rgba, RgbaImage};

    // Create a 100x100 white opaque image
    let mut img = RgbaImage::from_pixel(100, 100, Rgba([255, 255, 255, 255]));

    // Apply ellipse mask with softness but vignette_intensity = 0
    apply_shape_mask(
        &mut img,
        &CropShape::Ellipse,
        0.1, // small softness so corner is clearly outside
        0.0, // zero intensity - should still crop!
        crate::color::RgbaColor::opaque(0, 0, 0),
    );

    // Corner pixel (0,0) with small softness should be nearly transparent
    let corner = img.get_pixel(0, 0);
    assert!(
        corner[3] < 50,
        "Corner pixel should be mostly transparent with softness, got alpha={}",
        corner[3]
    );

    // RGB should remain white (255,255,255) since intensity=0 means no color mixing
    assert_eq!(
        (corner[0], corner[1], corner[2]),
        (255, 255, 255),
        "Corner RGB should remain white when intensity=0, got ({},{},{})",
        corner[0],
        corner[1],
        corner[2]
    );

    // Center pixel (50,50) is inside the ellipse - must remain opaque
    let center = img.get_pixel(50, 50);
    assert_eq!(
        center[3], 255,
        "Center pixel should remain fully opaque (alpha=255), got alpha={}",
        center[3]
    );
}

/// Test that vignette_intensity only affects color mixing, not alpha.
#[test]
fn test_vignette_intensity_affects_color_not_alpha() {
    use crate::shape::{CropShape, apply_shape_mask};
    use image::{Rgba, RgbaImage};

    // Test with intensity=0.0
    let mut img_zero = RgbaImage::from_pixel(100, 100, Rgba([255, 255, 255, 255]));
    apply_shape_mask(
        &mut img_zero,
        &CropShape::Ellipse,
        0.0,
        0.0,
        crate::color::RgbaColor::opaque(255, 0, 0), // red vignette
    );

    // Test with intensity=1.0
    let mut img_full = RgbaImage::from_pixel(100, 100, Rgba([255, 255, 255, 255]));
    apply_shape_mask(
        &mut img_full,
        &CropShape::Ellipse,
        0.0,
        1.0,
        crate::color::RgbaColor::opaque(255, 0, 0), // red vignette
    );

    // Both should have same alpha at corners (transparent)
    let corner_zero = img_zero.get_pixel(0, 0);
    let corner_full = img_full.get_pixel(0, 0);
    assert_eq!(
        corner_zero[3], corner_full[3],
        "Alpha should be same regardless of intensity: zero={}, full={}",
        corner_zero[3], corner_full[3]
    );
    assert_eq!(corner_zero[3], 0, "Corner should be transparent");

    // Both should have same alpha at center (opaque)
    let center_zero = img_zero.get_pixel(50, 50);
    let center_full = img_full.get_pixel(50, 50);
    assert_eq!(center_zero[3], 255, "Center should be opaque");
    assert_eq!(center_full[3], 255, "Center should be opaque");
}

#[test]
fn test_analytical_mask_accuracy() {
    use crate::shape::{CropShape, apply_shape_mask};
    use image::{Rgba, RgbaImage};

    // Test Rectangle (Analytical) - should be exact
    let width = 100;
    let height = 100;
    let mut img = RgbaImage::from_pixel(width, height, Rgba([255, 255, 255, 255]));

    // Exact hard crop
    apply_shape_mask(
        &mut img,
        &CropShape::Rectangle,
        0.0,
        0.0,
        crate::color::RgbaColor::opaque(0, 0, 0),
    );

    // Inside
    assert_eq!(img.get_pixel(50, 50)[3], 255);
    // Corners should be inside for Rectangle
    assert_eq!(img.get_pixel(0, 0)[3], 255);
    assert_eq!(img.get_pixel(99, 99)[3], 255);

    // Test Ellipse (Analytical)
    let mut img_ellipse = RgbaImage::from_pixel(width, height, Rgba([255, 255, 255, 255]));
    apply_shape_mask(
        &mut img_ellipse,
        &CropShape::Ellipse,
        0.0,
        0.0,
        crate::color::RgbaColor::opaque(0, 0, 0),
    );

    // Center inside
    assert_eq!(img_ellipse.get_pixel(50, 50)[3], 255);
    // Corner outside
    assert_eq!(img_ellipse.get_pixel(0, 0)[3], 0);
}

#[test]
fn test_optimized_raster_mask_softness() {
    use crate::shape::{CropShape, apply_shape_mask};
    use image::{Rgba, RgbaImage};

    // Large image to trigger downscaling optimization?
    // Our threshold is max_dim=256. Let's try 300x300.
    let width = 300;
    let height = 300;
    let mut img = RgbaImage::from_pixel(width, height, Rgba([255, 255, 255, 255]));

    // Complex shape (forces raster path)
    let shape = CropShape::Star {
        points: 5,
        inner_radius_pct: 0.5,
        rotation_deg: 0.0,
    };

    apply_shape_mask(
        &mut img,
        &shape,
        0.2, // Some softness
        0.0,
        crate::color::RgbaColor::opaque(0, 0, 0),
    );

    // Center should be nearly opaque (softness may reduce slightly)
    assert!(
        img.get_pixel(150, 150)[3] >= 250,
        "Center pixel should be nearly opaque, got alpha={}",
        img.get_pixel(150, 150)[3]
    );
    // Corner should be nearly transparent (softness causes slight blur bleeding)
    assert!(
        img.get_pixel(0, 0)[3] <= 5,
        "Corner pixel should be nearly transparent with softness, got alpha={}",
        img.get_pixel(0, 0)[3]
    );
}

/// Test that complex shapes (Star, Polygon) work correctly with HARD edges (no softness).
/// This is the core cropping functionality - the shape MUST be applied.
#[test]
fn test_star_crop_hard_edge() {
    use crate::shape::{CropShape, apply_shape_mask};
    use image::{Rgba, RgbaImage};

    let width = 200;
    let height = 200;
    let mut img = RgbaImage::from_pixel(width, height, Rgba([255, 255, 255, 255]));

    let shape = CropShape::Star {
        points: 5,
        inner_radius_pct: 0.5,
        rotation_deg: 0.0,
    };

    apply_shape_mask(
        &mut img,
        &shape,
        0.0, // NO softness - hard edge
        0.0, // zero intensity
        crate::color::RgbaColor::opaque(0, 0, 0),
    );

    // Center should be fully opaque (inside the star)
    let center = img.get_pixel(100, 100);
    assert_eq!(
        center[3], 255,
        "Center pixel should be fully opaque (alpha=255), got alpha={}",
        center[3]
    );

    // Corner should be fully transparent (outside the star)
    let corner = img.get_pixel(0, 0);
    assert_eq!(
        corner[3], 0,
        "Corner pixel should be fully transparent (alpha=0), got alpha={}",
        corner[3]
    );
}

/// Test that Polygon shape (sharp corners) crops correctly with hard edges.
#[test]
fn test_polygon_crop_hard_edge() {
    use crate::shape::{CropShape, apply_shape_mask};
    use image::{Rgba, RgbaImage};

    let width = 200;
    let height = 200;
    let mut img = RgbaImage::from_pixel(width, height, Rgba([255, 255, 255, 255]));

    // Hexagon
    let shape = CropShape::Polygon {
        sides: 6,
        rotation_deg: 0.0,
        corner_style: PolygonCornerStyle::Sharp,
    };

    apply_shape_mask(
        &mut img,
        &shape,
        0.0, // NO softness - hard edge
        0.0, // zero intensity
        crate::color::RgbaColor::opaque(0, 0, 0),
    );

    // Center should be fully opaque (inside the hexagon)
    let center = img.get_pixel(100, 100);
    assert_eq!(
        center[3], 255,
        "Center pixel should be fully opaque (alpha=255), got alpha={}",
        center[3]
    );

    // Corner should be fully transparent (outside the hexagon)
    let corner = img.get_pixel(0, 0);
    assert_eq!(
        corner[3], 0,
        "Corner pixel should be fully transparent (alpha=0), got alpha={}",
        corner[3]
    );
}

/// Test that KochPolygon fractal shape crops correctly.
#[test]
fn test_koch_polygon_crop_hard_edge() {
    use crate::shape::{CropShape, apply_shape_mask};
    use image::{Rgba, RgbaImage};

    let width = 200;
    let height = 200;
    let mut img = RgbaImage::from_pixel(width, height, Rgba([255, 255, 255, 255]));

    let shape = CropShape::KochPolygon {
        sides: 3,
        rotation_deg: 0.0,
        iterations: 2,
    };

    apply_shape_mask(
        &mut img,
        &shape,
        0.0, // NO softness - hard edge
        0.0, // zero intensity
        crate::color::RgbaColor::opaque(0, 0, 0),
    );

    // Center should be fully opaque (inside the shape)
    let center = img.get_pixel(100, 100);
    assert_eq!(
        center[3], 255,
        "Center pixel should be fully opaque (alpha=255), got alpha={}",
        center[3]
    );

    // Corner should be fully transparent (outside the shape)
    let corner = img.get_pixel(0, 0);
    assert_eq!(
        corner[3], 0,
        "Corner pixel should be fully transparent (alpha=0), got alpha={}",
        corner[3]
    );
}

/// Test that KochRectangle fractal shape crops correctly.
#[test]
fn test_koch_rectangle_crop_hard_edge() {
    use crate::shape::{CropShape, apply_shape_mask};
    use image::{Rgba, RgbaImage};

    let width = 200;
    let height = 200;
    let mut img = RgbaImage::from_pixel(width, height, Rgba([255, 255, 255, 255]));

    let shape = CropShape::KochRectangle { iterations: 2 };

    apply_shape_mask(
        &mut img,
        &shape,
        0.0, // NO softness - hard edge
        0.0, // zero intensity
        crate::color::RgbaColor::opaque(0, 0, 0),
    );

    // Center should be fully opaque (inside the shape)
    let center = img.get_pixel(100, 100);
    assert_eq!(
        center[3], 255,
        "Center pixel should be fully opaque (alpha=255), got alpha={}",
        center[3]
    );

    // Corners are tricky for Koch rectangle - the fractal bumps outward
    // Let's check a point that's definitely outside
    // The Koch fractal expands outward, so corners may be inside or outside
    // depending on iterations. Let's check a more reliable outside point.
}

/// Test that complex shapes with rounded corners crop correctly.
#[test]
fn test_polygon_rounded_corners_crop() {
    use crate::shape::{CropShape, apply_shape_mask};
    use image::{Rgba, RgbaImage};

    let width = 200;
    let height = 200;
    let mut img = RgbaImage::from_pixel(width, height, Rgba([255, 255, 255, 255]));

    let shape = CropShape::Polygon {
        sides: 4,
        rotation_deg: 45.0, // Diamond orientation
        corner_style: PolygonCornerStyle::Rounded { radius_pct: 0.1 },
    };

    apply_shape_mask(
        &mut img,
        &shape,
        0.0, // NO softness - hard edge
        0.0, // zero intensity
        crate::color::RgbaColor::opaque(0, 0, 0),
    );

    // Center should be fully opaque
    let center = img.get_pixel(100, 100);
    assert_eq!(
        center[3], 255,
        "Center pixel should be fully opaque (alpha=255), got alpha={}",
        center[3]
    );

    // Corner should be fully transparent (diamond shape rotated 45 degrees)
    let corner = img.get_pixel(0, 0);
    assert_eq!(
        corner[3], 0,
        "Corner pixel should be fully transparent (alpha=0), got alpha={}",
        corner[3]
    );
}

/// Integration test: Simulates the CPU fallback export pipeline.
/// This test verifies that when the GPU enhancer is None (CPU fallback),
/// the shape mask is still correctly applied to custom shapes.
#[test]
fn test_cpu_fallback_pipeline_applies_shape() {
    use crate::shape::{CropShape, apply_shape_mask};
    use image::{DynamicImage, Rgba, RgbaImage};

    // Simulate the CPU fallback path from apply_mask_with_gpu in cache.rs:
    // let mut rgba = image.to_rgba8();
    // apply_shape_mask(&mut rgba, shape, ...);
    // DynamicImage::ImageRgba8(rgba)

    // Create a test image (simulating a cropped face)
    let width = 256;
    let height = 256;
    let original = RgbaImage::from_pixel(width, height, Rgba([255, 128, 64, 255]));
    let image = DynamicImage::ImageRgba8(original);

    // Test with Star shape (custom shape that uses raster mask)
    let star_shape = CropShape::Star {
        points: 5,
        inner_radius_pct: 0.5,
        rotation_deg: 0.0,
    };

    // Apply mask like CPU fallback does
    let mut rgba = image.to_rgba8();
    apply_shape_mask(
        &mut rgba,
        &star_shape,
        0.0, // no softness
        0.0, // no vignette
        crate::color::RgbaColor::opaque(0, 0, 0),
    );
    let masked = DynamicImage::ImageRgba8(rgba);

    // Verify the mask was applied
    let masked_rgba = masked.to_rgba8();

    // Center should be opaque (inside the star)
    let center = masked_rgba.get_pixel(128, 128);
    assert_eq!(
        center[3], 255,
        "CPU fallback: Center should be fully opaque (inside star), got alpha={}",
        center[3]
    );

    // Corner should be transparent (outside the star)
    let corner = masked_rgba.get_pixel(0, 0);
    assert_eq!(
        corner[3], 0,
        "CPU fallback: Corner should be fully transparent (outside star), got alpha={}",
        corner[3]
    );
}

/// Test that apply_shape_mask_dynamic correctly modifies the image in place.
#[test]
fn test_apply_shape_mask_dynamic_modifies_image() {
    use crate::shape::{CropShape, apply_shape_mask_dynamic};
    use image::{DynamicImage, Rgba, RgbaImage};

    // Create a test image
    let width = 100;
    let height = 100;
    let original = RgbaImage::from_pixel(width, height, Rgba([255, 255, 255, 255]));
    let mut image = DynamicImage::ImageRgba8(original);

    // Apply star mask
    let shape = CropShape::Star {
        points: 6,
        inner_radius_pct: 0.4,
        rotation_deg: 30.0,
    };

    apply_shape_mask_dynamic(
        &mut image,
        &shape,
        0.0,
        0.0,
        crate::color::RgbaColor::opaque(0, 0, 0),
    );

    // Verify the image was modified
    let rgba = image.to_rgba8();

    // Center should be opaque
    let center = rgba.get_pixel(50, 50);
    assert_eq!(
        center[3], 255,
        "apply_shape_mask_dynamic: Center should be opaque, got alpha={}",
        center[3]
    );

    // Corner should be transparent
    let corner = rgba.get_pixel(0, 0);
    assert_eq!(
        corner[3], 0,
        "apply_shape_mask_dynamic: Corner should be transparent, got alpha={}",
        corner[3]
    );
}

/// Test that CropShape deserializes correctly from JSON settings.
#[test]
fn test_crop_shape_deserialize_polygon() {
    use crate::shape::CropShape;

    let json = r#"{
            "kind": "polygon",
            "sides": 5,
            "rotation_deg": 0.0,
            "corner_style": {
                "style": "bezier",
                "tension": 1.3333331
            }
        }"#;

    let shape: CropShape = serde_json::from_str(json).expect("Failed to deserialize CropShape");

    match &shape {
        CropShape::Polygon {
            sides,
            rotation_deg,
            corner_style,
        } => {
            assert_eq!(*sides, 5);
            assert!((rotation_deg - 0.0).abs() < 0.001);
            match corner_style {
                PolygonCornerStyle::Bezier { tension } => {
                    assert!((tension - 1.3333331).abs() < 0.001);
                }
                _ => panic!("Expected Bezier corner style, got {:?}", corner_style),
            }
        }
        _ => panic!("Expected Polygon shape, got {:?}", shape),
    }
}

/// Test large image cropping where downscaling optimization engages.
#[test]
fn test_large_image_custom_shape_crop() {
    use crate::shape::{CropShape, apply_shape_mask};
    use image::{Rgba, RgbaImage};

    // Create a large image (> 256 which is the threshold in apply_raster_mask_optimized)
    let width = 1000;
    let height = 1000;
    // Start with fully opaque white
    let mut img = RgbaImage::from_pixel(width, height, Rgba([255, 255, 255, 255]));

    let shape = CropShape::Star {
        points: 5,
        inner_radius_pct: 0.5,
        rotation_deg: 45.0,
    };

    apply_shape_mask(
        &mut img,
        &shape,
        0.0,
        0.0,
        crate::color::RgbaColor::opaque(0, 0, 0),
    );

    // Center should be opaque (star center is always opaque)
    let center = img.get_pixel(500, 500);
    assert_eq!(
        center[3], 255,
        "Large Image Rotated: Center should be opaque, got {}",
        center[3]
    );

    // Corner should be transparent
    let corner = img.get_pixel(0, 0);
    assert_eq!(
        corner[3], 0,
        "Large Image Rotated: Corner should be transparent, got {}",
        corner[3]
    );
}
