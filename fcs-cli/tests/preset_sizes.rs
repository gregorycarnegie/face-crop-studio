/// Integration tests for crop preset sizes
use image::GenericImageView;
use std::fs;

#[macro_use]
mod common;

use common::copy_fixture_to;

use rstest::rstest;

#[rstest]
#[case::linkedin("linkedin", 400, 400, "LinkedIn")]
#[case::passport("passport", 413, 531, "Passport")]
#[case::instagram("instagram", 1080, 1080, "Instagram")]
#[case::avatar("avatar", 512, 512, "Avatar")]
#[case::headshot("headshot", 600, 800, "Headshot")]
fn preset_has_expected_dimensions(
    #[case] preset: &str,
    #[case] width: u32,
    #[case] height: u32,
    #[case] display_name: &str,
) {
    let (model, _fixture, _temp_dir, input_path, output_dir) = cli_test_setup!(load_image);
    let output = run_cli!(
        input_path,
        model,
        output_dir,
        ["--crop", "--preset", preset]
    );
    assert_cli_success!(output, "CLI should succeed with preset");
    let output_files = verify_output_files!(output_dir, "png");
    assert!(!output_files.is_empty(), "Should have output files");
    for entry in output_files {
        let img = image::open(entry.path()).expect("open output");
        assert_eq!(
            img.dimensions(),
            (width, height),
            "{display_name} preset should be {width}x{height}"
        );
    }
}

#[rstest]
fn test_all_presets_succeed(
    #[values("linkedin", "passport", "instagram", "idcard", "avatar", "headshot")] preset: &str,
) {
    let (model, fixture, _temp_dir, _input_path, _) = cli_test_setup!();

    let temp_dir = tempfile::TempDir::new().expect("create temp dir");
    let input_path = temp_dir.path().join("input.jpg");
    let output_dir = temp_dir.path().join("output");
    fs::create_dir_all(&output_dir).expect("create output dir");

    copy_fixture_to(&fixture, &input_path).expect("copy input");

    let output = run_cli!(
        input_path,
        model,
        output_dir,
        ["--crop", "--preset", preset]
    );

    assert!(
        output.status.success(),
        "Preset '{}' should succeed",
        preset
    );

    let output_files = verify_output_files!(output_dir, "png");
    assert!(
        !output_files.is_empty(),
        "Preset '{}' should produce output",
        preset
    );
}
