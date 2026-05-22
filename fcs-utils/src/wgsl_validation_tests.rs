use std::{
    fs,
    path::{Path, PathBuf},
};

use naga::valid::{Capabilities, ShaderStages, SubgroupOperationSet, ValidationFlags, Validator};

#[test]
fn validate_all_wgsl_files() {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("fcs-utils should live under the workspace root");
    let mut shader_paths = Vec::new();
    collect_wgsl_files(workspace_root, &mut shader_paths);
    shader_paths.sort();

    assert!(
        !shader_paths.is_empty(),
        "expected at least one WGSL shader under {}",
        workspace_root.display()
    );

    let mut failures = Vec::new();
    for path in shader_paths {
        match validate_wgsl_file(&path) {
            Ok(()) => {}
            Err(error) => {
                let relative_path = path.strip_prefix(workspace_root).unwrap_or(&path);
                failures.push(format!("{}\n{}", relative_path.display(), error.trim_end()));
            }
        }
    }

    if !failures.is_empty() {
        panic!(
            "Naga WGSL validation failed for {} shader(s):\n\n{}",
            failures.len(),
            failures.join("\n\n")
        );
    }
}

fn collect_wgsl_files(dir: &Path, shader_paths: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(dir)
        .unwrap_or_else(|err| panic!("failed to read directory {}: {err}", dir.display()));

    for entry in entries {
        let entry =
            entry.unwrap_or_else(|err| panic!("failed to read entry in {}: {err}", dir.display()));
        let path = entry.path();
        let file_type = entry
            .file_type()
            .unwrap_or_else(|err| panic!("failed to inspect {}: {err}", path.display()));

        if file_type.is_dir() {
            if should_skip_dir(&path) {
                continue;
            }
            collect_wgsl_files(&path, shader_paths);
        } else if path
            .extension()
            .is_some_and(|extension| extension == "wgsl")
        {
            shader_paths.push(path);
        }
    }
}

fn should_skip_dir(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some(".git" | "target")
    )
}

fn validate_wgsl_file(path: &Path) -> Result<(), String> {
    let source =
        fs::read_to_string(path).map_err(|err| format!("failed to read WGSL file: {err}"))?;
    let path_display = path.display().to_string();
    let module = naga::front::wgsl::parse_str(&source)
        .map_err(|err| err.emit_to_string_with_path(&source, &path_display))?;

    Validator::new(ValidationFlags::all(), Capabilities::all())
        .subgroup_stages(ShaderStages::all())
        .subgroup_operations(SubgroupOperationSet::all())
        .validate(&module)
        .map(|_| ())
        .map_err(|err| err.emit_to_string_with_path(&source, &path_display))
}
