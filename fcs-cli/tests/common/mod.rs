/// Common test utilities and macros for CLI integration tests
use std::{
    fs,
    io::Result,
    path::{Path, PathBuf},
};

/// Turn a missing asset into a hard failure when `FCS_STRICT_TESTS` is set (CI does).
///
/// Every test here skips itself when the model or fixtures are absent, which is right
/// on a fresh clone but means a broken path makes the whole suite pass vacuously. In
/// CI the assets are always present, so "not found" is a bug, not a reason to skip.
fn require(found: Option<PathBuf>, what: &str) -> Option<PathBuf> {
    assert!(
        !(found.is_none() && std::env::var_os("FCS_STRICT_TESTS").is_some()),
        "{what} not found while FCS_STRICT_TESTS is set (cwd {:?})",
        std::env::current_dir()
    );
    found
}

pub fn find_model_path() -> Option<PathBuf> {
    let candidates = vec![
        "models/face_detection_yunet_2023mar_640.onnx",
        "../models/face_detection_yunet_2023mar_640.onnx",
    ];
    require(
        candidates
            .into_iter()
            .map(PathBuf::from)
            .find(|p| p.exists()),
        "ONNX model",
    )
}

pub fn find_fixture_image() -> Option<PathBuf> {
    let candidates = vec!["fixtures/images/006.jpg", "../fixtures/images/006.jpg"];
    require(
        candidates
            .into_iter()
            .map(PathBuf::from)
            .find(|p| p.exists()),
        "fixture image",
    )
}

#[allow(dead_code)]
pub fn copy_fixture_to(src: &PathBuf, dest: &Path) -> Result<()> {
    fs::copy(src, dest).map(|_| ())
}

/// Macro to set up common test environment with model, fixture, and temp directories.
///
/// Creates:
/// - `model: PathBuf` - path to the ONNX model
/// - `fixture: PathBuf` - path to fixture image
/// - `temp_dir: TempDir` - temporary directory
/// - `input_path: PathBuf` - path for input image
/// - `output_dir: PathBuf` - path for output directory
///
/// Automatically skips test if model or fixture not found.
/// Optionally loads and saves the fixture image to input_path.
///
/// # Usage
///
/// ```ignore
/// cli_test_setup!(); // Just paths, no image loading
/// cli_test_setup!(load_image); // Also loads fixture to input_path
/// ```
#[macro_export]
macro_rules! cli_test_setup {
    () => {{
        let model = match $crate::common::find_model_path() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: model not found");
                return;
            }
        };

        let fixture = match $crate::common::find_fixture_image() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: fixture not found");
                return;
            }
        };

        let temp_dir = tempfile::TempDir::new().expect("create temp dir");
        let input_path = temp_dir.path().join("input.jpg");
        let output_dir = temp_dir.path().join("output");
        std::fs::create_dir_all(&output_dir).expect("create output dir");

        (model, fixture, temp_dir, input_path, output_dir)
    }};

    (load_image) => {{
        let (model, fixture, temp_dir, input_path, output_dir) = $crate::cli_test_setup!();

        $crate::common::copy_fixture_to(&fixture, &input_path).expect("copy input image");

        (model, fixture, temp_dir, input_path, output_dir)
    }};

    (load_to_dir) => {{
        let model = match $crate::common::find_model_path() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: model not found");
                return;
            }
        };

        let fixture = match $crate::common::find_fixture_image() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: fixture not found");
                return;
            }
        };

        let temp_dir = tempfile::TempDir::new().expect("create temp dir");
        let input_dir = temp_dir.path().join("input");
        let output_dir = temp_dir.path().join("output");
        fs::create_dir_all(&input_dir).expect("create input dir");
        fs::create_dir_all(&output_dir).expect("create output dir");

        (model, fixture, temp_dir, input_dir, output_dir)
    }};
}

/// Macro to verify output files exist in directory.
///
/// Returns a Vec of DirEntry for files matching the extension.
///
/// # Usage
///
/// ```ignore
/// let files = verify_output_files!(output_dir, "png");
/// assert!(!files.is_empty(), "Should have output files");
/// ```
#[macro_export]
macro_rules! verify_output_files {
    ($output_dir:expr, $ext:literal) => {{
        fs::read_dir(&$output_dir)
            .expect("read output dir")
            .filter_map(Result::ok)
            .filter(|e| {
                e.path()
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e == $ext)
                    .unwrap_or(false)
            })
            .collect::<Vec<_>>()
    }};
}

/// Macro to run CLI with common arguments and handle errors.
///
/// Returns the Command output.
///
/// # Usage
///
/// ```ignore
/// let output = run_cli!(
///     input_path,
///     model,
///     output_dir,
///     ["--crop", "--output-width", "200", "--output-height", "200"]
/// );
/// assert!(output.status.success());
/// ```
#[macro_export]
macro_rules! run_cli {
    ($input:expr, $model:expr, $output_dir:expr, [$($arg:expr),*]) => {{
        std::process::Command::new(env!("CARGO_BIN_EXE_fcs-cli"))
            .args([
                "--input",
                $input.to_str().unwrap(),
                "--model",
                $model.to_str().unwrap(),
                $($arg,)*
                "--output-dir",
                $output_dir.to_str().unwrap(),
            ])
            .output()
            .expect("execute CLI")
    }};
}

/// Macro to assert CLI success and optionally print stderr on failure.
///
/// # Usage
///
/// ```ignore
/// assert_cli_success!(output, "CLI should succeed with positioning mode");
/// ```
#[macro_export]
macro_rules! assert_cli_success {
    ($output:expr, $msg:literal) => {{
        if !$output.status.success() {
            eprintln!("CLI stderr: {}", String::from_utf8_lossy(&$output.stderr));
        }
        assert!($output.status.success(), $msg);
    }};
}
