/// Integration tests for batch processing with quality filtering
use std::fs;

#[macro_use]
mod common;

use common::copy_fixture_to;

#[test]
fn test_batch_processing_with_multiple_images() {
    let (model, fixture, _temp_dir, input_dir, output_dir) = cli_test_setup!(load_to_dir);

    // Copy real fixture images for testing
    copy_fixture_to(&fixture, &input_dir.join("image1.jpg")).expect("copy image1");
    copy_fixture_to(&fixture, &input_dir.join("image2.jpg")).expect("copy image2");
    copy_fixture_to(&fixture, &input_dir.join("image3.jpg")).expect("copy image3");

    let output = run_cli!(
        input_dir,
        model,
        output_dir,
        ["--crop", "--output-width", "200", "--output-height", "200"]
    );

    assert_cli_success!(output, "CLI should succeed processing batch");

    let output_files = verify_output_files!(output_dir, "png");
    assert!(!output_files.is_empty(), "Should produce output files");
}

#[test]
fn test_skip_low_quality_flag() {
    let (model, fixture, _temp_dir, input_dir, output_dir) = cli_test_setup!(load_to_dir);

    copy_fixture_to(&fixture, &input_dir.join("test.jpg")).expect("copy fixture");

    let output = run_cli!(
        input_dir,
        model,
        output_dir,
        [
            "--crop",
            "--output-width",
            "200",
            "--output-height",
            "200",
            "--skip-low-quality",
            "true"
        ]
    );

    assert_cli_success!(output, "CLI should succeed with quality filtering");

    let output_count = verify_output_files!(output_dir, "png").len();
    eprintln!("Quality filtering produced {} output files", output_count);
}

#[test]
fn test_min_quality_high() {
    let (model, fixture, _temp_dir, input_path, output_dir) = cli_test_setup!();
    copy_fixture_to(&fixture, &input_path).expect("copy fixture");

    let output = run_cli!(
        input_path,
        model,
        output_dir,
        [
            "--crop",
            "--output-width",
            "200",
            "--output-height",
            "200",
            "--min-quality",
            "high"
        ]
    );

    assert_cli_success!(output, "CLI should succeed with high quality threshold");
}

#[test]
fn test_min_quality_medium() {
    let (model, fixture, _temp_dir, input_path, output_dir) = cli_test_setup!();
    copy_fixture_to(&fixture, &input_path).expect("copy fixture");

    let output = run_cli!(
        input_path,
        model,
        output_dir,
        [
            "--crop",
            "--output-width",
            "200",
            "--output-height",
            "200",
            "--min-quality",
            "medium"
        ]
    );

    assert_cli_success!(output, "CLI should succeed with medium quality threshold");
}

#[test]
fn test_batch_with_face_index_selection() {
    let (model, fixture, _temp_dir, input_path, output_dir) = cli_test_setup!();
    copy_fixture_to(&fixture, &input_path).expect("copy fixture");

    let output = run_cli!(
        input_path,
        model,
        output_dir,
        [
            "--crop",
            "--output-width",
            "200",
            "--output-height",
            "200",
            "--face-index",
            "0"
        ]
    );

    assert_cli_success!(output, "CLI should succeed with face index selection");
}

#[test]
fn test_batch_with_naming_template() {
    let (model, fixture, _temp_dir, input_dir, output_dir) = cli_test_setup!(load_to_dir);
    copy_fixture_to(&fixture, &input_dir.join("test.jpg")).expect("copy fixture");

    let output = run_cli!(
        input_dir,
        model,
        output_dir,
        [
            "--crop",
            "--output-width",
            "200",
            "--output-height",
            "200",
            "--naming-template",
            "{original}_crop_{index}.{ext}"
        ]
    );

    assert_cli_success!(output, "CLI should succeed with naming template");

    // Verify output exists with expected naming pattern
    let output_files: Vec<_> = fs::read_dir(&output_dir)
        .expect("read output dir")
        .filter_map(Result::ok)
        .map(|e| e.file_name().into_string().unwrap())
        .collect();

    assert!(
        output_files.iter().any(|name| name.contains("_crop_")),
        "Output should use naming template, got: {:?}",
        output_files
    );
}

#[test]
fn test_batch_processing_summary_output() {
    let (model, fixture, _temp_dir, input_dir, output_dir) = cli_test_setup!(load_to_dir);

    copy_fixture_to(&fixture, &input_dir.join("img1.jpg")).expect("copy img1");
    copy_fixture_to(&fixture, &input_dir.join("img2.jpg")).expect("copy img2");

    let output = run_cli!(
        input_dir,
        model,
        output_dir,
        ["--crop", "--output-width", "200", "--output-height", "200"]
    );

    assert_cli_success!(output, "CLI should succeed processing batch");

    let stdout = String::from_utf8_lossy(&output.stdout);
    eprintln!("Batch processing output:\n{}", stdout);
}

#[test]
fn a_crop_that_cannot_be_saved_fails_the_run() {
    let (model, fixture, _temp_dir, input_dir, output_dir) = cli_test_setup!(load_to_dir);

    copy_fixture_to(&fixture, &input_dir.join("test.jpg")).expect("copy fixture");
    // A directory where the first crop's file should go makes its final rename fail.
    fs::create_dir(output_dir.join("test_face1.png")).expect("blocking directory");

    let output = run_cli!(input_dir, model, output_dir, ["--crop"]);

    assert!(!output.status.success(), "a lost crop must fail the run");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("could not be saved"), "stderr: {stderr}");
}
