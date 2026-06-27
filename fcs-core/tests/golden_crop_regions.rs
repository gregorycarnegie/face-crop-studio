//! Golden-output regression tests for the deterministic crop-geometry pipeline.
//!
//! The property tests in `cropper.rs` check invariants (aspect ratio, in-bounds
//! clamping) and the OpenCV parity test depends on the ONNX model. Neither pins
//! the *exact* crop the geometry produces. These tests lock the integer
//! `CropRegion` for a small pack of representative scenarios. The crop math is
//! model-independent and bit-stable, so any change shows up here as a concrete
//! coordinate diff rather than a silent behavior shift.
//!
//! Golden values live in `fixtures/golden/crop_regions.json`. After an
//! *intentional* change to the crop math, regenerate and review the diff:
//!
//! ```text
//! UPDATE_GOLDEN=1 cargo test -p fcs-core --test golden_crop_regions
//! ```

use fcs_core::{BoundingBox, CropRegion, CropSettings, PositioningMode, calculate_crop_region};
use fcs_utils::fixtures_dir;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, env, fs};

const GOLDEN_RELATIVE: &str = "golden/crop_regions.json";

/// Serializable mirror of `CropRegion` (the core type intentionally stays free
/// of serde derives).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
struct GoldenRegion {
    x: i32,
    y: i32,
    width: u32,
    height: u32,
    pad_left: u32,
    pad_top: u32,
    pad_right: u32,
    pad_bottom: u32,
}

impl From<CropRegion> for GoldenRegion {
    fn from(r: CropRegion) -> Self {
        Self {
            x: r.x,
            y: r.y,
            width: r.width,
            height: r.height,
            pad_left: r.pad_left,
            pad_top: r.pad_top,
            pad_right: r.pad_right,
            pad_bottom: r.pad_bottom,
        }
    }
}

/// One golden case: the inputs (echoed for readability) plus the expected crop.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GoldenCase {
    name: String,
    image: [u32; 2],
    face_bbox: [f32; 4],
    output: [u32; 2],
    face_height_pct: f32,
    positioning: String,
    offset: [f32; 2],
    region: GoldenRegion,
}

struct Scenario {
    name: &'static str,
    img_w: u32,
    img_h: u32,
    bbox: BoundingBox,
    settings: CropSettings,
}

#[allow(clippy::too_many_arguments)]
fn scenario(
    name: &'static str,
    img_w: u32,
    img_h: u32,
    bbox: [f32; 4],
    output: [u32; 2],
    face_height_pct: f32,
    positioning_mode: PositioningMode,
    offset: [f32; 2],
) -> Scenario {
    Scenario {
        name,
        img_w,
        img_h,
        bbox: BoundingBox {
            x: bbox[0],
            y: bbox[1],
            width: bbox[2],
            height: bbox[3],
        },
        settings: CropSettings {
            output_width: output[0],
            output_height: output[1],
            face_height_pct,
            positioning_mode,
            horizontal_offset: offset[0],
            vertical_offset: offset[1],
            ..Default::default()
        },
    }
}

fn positioning_label(mode: PositioningMode) -> &'static str {
    match mode {
        PositioningMode::Center => "center",
        PositioningMode::RuleOfThirds => "rule_of_thirds",
        PositioningMode::Custom => "custom",
    }
}

/// Representative crop scenarios covering in-bounds crops, padding on every
/// edge, both portrait and landscape outputs, all positioning modes, and the
/// extremes of `face_height_pct`.
fn scenarios() -> Vec<Scenario> {
    use PositioningMode::{Center, Custom, RuleOfThirds};
    vec![
        // Face centered well inside a square source: no padding expected.
        scenario(
            "center_square_in_bounds",
            512,
            512,
            [160.0, 160.0, 140.0, 140.0],
            [256, 256],
            70.0,
            Center,
            [0.0, 0.0],
        ),
        // Large face in a small source at low coverage: the crop is bigger than
        // the source, so it pads on every edge (asymmetrically, since the face
        // is off-center).
        scenario(
            "center_requires_padding_all_sides",
            200,
            200,
            [40.0, 40.0, 140.0, 140.0],
            [256, 256],
            50.0,
            Center,
            [0.0, 0.0],
        ),
        // Face at the very top-left corner: padding on left and top only.
        scenario(
            "face_top_left_corner",
            500,
            500,
            [0.0, 0.0, 120.0, 120.0],
            [256, 256],
            80.0,
            Center,
            [0.0, 0.0],
        ),
        // Rule-of-thirds positioning on a landscape source.
        scenario(
            "rule_of_thirds_landscape",
            800,
            600,
            [300.0, 200.0, 160.0, 160.0],
            [400, 400],
            60.0,
            RuleOfThirds,
            [0.0, 0.0],
        ),
        // Custom offsets pushing the face right and down.
        scenario(
            "custom_offset_right_down",
            600,
            600,
            [250.0, 250.0, 120.0, 120.0],
            [300, 300],
            50.0,
            Custom,
            [0.5, 0.3],
        ),
        // Portrait 9:16 output.
        scenario(
            "portrait_output_9x16",
            1080,
            1920,
            [400.0, 500.0, 300.0, 300.0],
            [360, 640],
            70.0,
            Center,
            [0.0, 0.0],
        ),
        // Landscape 16:9 output.
        scenario(
            "landscape_output_16x9",
            1920,
            1080,
            [800.0, 400.0, 260.0, 260.0],
            [640, 360],
            65.0,
            Center,
            [0.0, 0.0],
        ),
        // Small face with a low coverage target: large crop, likely padded.
        scenario(
            "small_face_low_pct",
            1000,
            1000,
            [450.0, 450.0, 80.0, 80.0],
            [400, 400],
            20.0,
            Center,
            [0.0, 0.0],
        ),
    ]
}

fn compute(s: &Scenario) -> GoldenCase {
    let region = calculate_crop_region(s.img_w, s.img_h, s.bbox, &s.settings);
    GoldenCase {
        name: s.name.to_string(),
        image: [s.img_w, s.img_h],
        face_bbox: [s.bbox.x, s.bbox.y, s.bbox.width, s.bbox.height],
        output: [s.settings.output_width, s.settings.output_height],
        face_height_pct: s.settings.face_height_pct,
        positioning: positioning_label(s.settings.positioning_mode).to_string(),
        offset: [s.settings.horizontal_offset, s.settings.vertical_offset],
        region: region.into(),
    }
}

#[test]
fn crop_regions_match_golden() {
    let computed: Vec<GoldenCase> = scenarios().iter().map(compute).collect();

    let golden_path = fixtures_dir()
        .expect("locate fixtures dir")
        .join(GOLDEN_RELATIVE);

    let update = env::var_os("UPDATE_GOLDEN").is_some();

    // Regenerate when explicitly requested, or bootstrap on first run if the
    // file has not been committed yet.
    if update || !golden_path.exists() {
        if let Some(parent) = golden_path.parent() {
            fs::create_dir_all(parent).expect("create golden dir");
        }
        let json = serde_json::to_string_pretty(&computed).expect("serialize golden");
        fs::write(&golden_path, format!("{json}\n")).expect("write golden file");
        eprintln!("wrote golden file at {}", golden_path.display());
        return;
    }

    let contents = fs::read_to_string(&golden_path).expect("read golden file");
    let expected: Vec<GoldenCase> = serde_json::from_str(&contents).expect("parse golden file");

    let expected_by_name: BTreeMap<&str, &GoldenRegion> = expected
        .iter()
        .map(|c| (c.name.as_str(), &c.region))
        .collect();

    assert_eq!(
        computed.len(),
        expected.len(),
        "scenario count changed; run `UPDATE_GOLDEN=1 cargo test -p fcs-core --test golden_crop_regions` to refresh fixtures/{GOLDEN_RELATIVE}",
    );

    for case in &computed {
        let want = expected_by_name.get(case.name.as_str()).unwrap_or_else(|| {
            panic!(
                "no golden entry for scenario '{}'; run UPDATE_GOLDEN=1 to refresh fixtures/{GOLDEN_RELATIVE}",
                case.name,
            )
        });
        assert_eq!(
            &case.region, *want,
            "crop region for scenario '{}' changed; if this is intentional, run `UPDATE_GOLDEN=1 cargo test -p fcs-core --test golden_crop_regions` to refresh fixtures/{}",
            case.name, GOLDEN_RELATIVE,
        );
    }
}
