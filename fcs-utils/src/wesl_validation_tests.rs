use std::{
    fs,
    path::{Path, PathBuf},
};

use naga::valid::{Capabilities, ShaderStages, SubgroupOperationSet, ValidationFlags, Validator};

#[test]
fn validate_all_wesl_files() {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("fcs-utils should live under the workspace root");
    let mut shader_paths = Vec::new();
    collect_wesl_files(workspace_root, &mut shader_paths);
    shader_paths.sort();

    assert!(
        !shader_paths.is_empty(),
        "expected at least one WESL shader under {}",
        workspace_root.display()
    );

    let mut failures = Vec::new();
    for path in shader_paths {
        let result = wesl::Compiler::default()
            .compile(&path)
            .map_err(|err| err.to_string())
            .and_then(|compiled| {
                validate_wgsl_source(&compiled.to_string(), &path.display().to_string())
            });
        match result {
            Ok(()) => {}
            Err(error) => {
                let relative_path = path.strip_prefix(workspace_root).unwrap_or(&path);
                failures.push(format!("{}\n{}", relative_path.display(), error.trim_end()));
            }
        }
    }

    if !failures.is_empty() {
        panic!(
            "WESL compilation or Naga validation failed for {} shader(s):\n\n{}",
            failures.len(),
            failures.join("\n\n")
        );
    }
}

#[test]
fn validate_wesl_generated_shaders() {
    for (name, source) in [
        ("pixel_adjust.wesl", crate::gpu::PIXEL_ADJUST_WGSL),
        ("gaussian_blur.wesl", crate::gpu::GAUSSIAN_BLUR_WGSL),
        ("bilateral_filter.wesl", crate::gpu::BILATERAL_FILTER_WGSL),
        ("background_blur.wesl", crate::gpu::BACKGROUND_BLUR_WGSL),
        ("shape_mask.wesl", crate::gpu::SHAPE_MASK_WGSL),
        ("hist_equalize.wesl", crate::gpu::HIST_EQUALIZE_WGSL),
    ] {
        validate_wgsl_source(source, name)
            .unwrap_or_else(|err| panic!("WESL output validation failed for {name}: {err}"));
    }
}

fn collect_wesl_files(dir: &Path, shader_paths: &mut Vec<PathBuf>) {
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
            collect_wesl_files(&path, shader_paths);
        } else if path
            .extension()
            .is_some_and(|extension| extension == "wesl")
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

fn validate_wgsl_source(source: &str, path_display: &str) -> Result<(), String> {
    let module = naga::front::wgsl::parse_str(source)
        .map_err(|err| err.emit_to_string_with_path(source, path_display))?;

    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .subgroup_stages(ShaderStages::all())
        .subgroup_operations(SubgroupOperationSet::all())
        .validate(&module)
        .map_err(|err| err.emit_to_string_with_path(source, path_display))?;

    // The macOS and Linux CI legs have no GPU adapter, so no test there ever creates a
    // pipeline. Lowering to SPIR-V (Vulkan) and MSL (Metal) here is the part of pipeline
    // creation that fails for a shader those backends cannot take, and it needs no adapter.
    naga::back::spv::write_vec(&module, &info, &naga::back::spv::Options::default(), None)
        .map_err(|err| format!("SPIR-V translation failed: {err}"))?;
    let msl = naga::back::msl::Options {
        // wgpu's MSL version on macOS 11, Rust's default deployment target for Apple Silicon.
        lang_version: (2, 3),
        ..Default::default()
    };
    naga::back::msl::write_string(
        &module,
        &info,
        &msl,
        &naga::back::msl::PipelineOptions::default(),
    )
    .map_err(|err| format!("MSL translation failed: {err}"))?;
    Ok(())
}
