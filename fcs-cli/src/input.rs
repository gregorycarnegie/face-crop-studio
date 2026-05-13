//! Input collection and mapping file handling.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use fcs_utils::{
    mapping::{
        ColumnSelector, MappingFormat, MappingReadOptions, detect_format as detect_mapping_format,
        load_mapping_entries,
    },
    normalize_path,
};
use log::{debug, info, warn};
use walkdir::WalkDir;

use crate::args::DetectArgs;

#[derive(Clone, Debug)]
pub struct ProcessingItem {
    pub source: PathBuf,
    pub output_override: Option<PathBuf>,
    pub mapping_row: Option<usize>,
}

/// Collect all image paths from a file or directory.
pub fn collect_images(path: &Path) -> Result<Vec<PathBuf>> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }

    if !path.is_dir() {
        anyhow::bail!(
            "input path is neither file nor directory: {}",
            path.display()
        );
    }

    let exts = ["jpg", "jpeg", "png", "bmp", "webp"];
    let mut images = Vec::new();
    for entry in WalkDir::new(path)
        .follow_links(false)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
    {
        if let Some(ext) = entry.path().extension().and_then(|e| e.to_str()) {
            let ext_lower = ext.to_ascii_lowercase();
            if exts.contains(&ext_lower.as_str()) {
                images.push(entry.path().to_path_buf());
            } else {
                debug!("Skipping non-image file {}", entry.path().display());
            }
        }
    }
    images.sort();
    Ok(images)
}

pub fn collect_standard_targets(input_path: &Path) -> Result<Vec<ProcessingItem>> {
    let images = collect_images(input_path)?;
    if images.is_empty() {
        anyhow::bail!(
            "no images found at {} (supported extensions: jpg, jpeg, png, bmp)",
            input_path.display()
        );
    }
    Ok(images
        .into_iter()
        .map(|path| ProcessingItem {
            source: path,
            output_override: None,
            mapping_row: None,
        })
        .collect())
}

pub fn collect_mapping_targets(
    mapping_file: &Path,
    args: &DetectArgs,
) -> Result<Vec<ProcessingItem>> {
    let mapping_path = normalize_path(mapping_file)?;
    let source_selector = ColumnSelector::parse_token(
        args.mapping_source_col
            .as_deref()
            .ok_or_else(|| anyhow!("--mapping-source-col is required with --mapping-file"))?,
    )?;
    let output_selector = ColumnSelector::parse_token(
        args.mapping_output_col
            .as_deref()
            .ok_or_else(|| anyhow!("--mapping-output-col is required with --mapping-file"))?,
    )?;

    let user_format = match args.mapping_format.as_deref() {
        Some(token) => Some(parse_mapping_format_token(token)?),
        None => None,
    };
    let mut read_options = MappingReadOptions {
        format: user_format,
        has_headers: args.mapping_has_headers,
        delimiter: args.mapping_delimiter.map(|c| c as u8),
        sheet_name: args.mapping_sheet.clone(),
        sql_table: args.mapping_sql_table.clone(),
        sql_query: args.mapping_sql_query.clone(),
        ..Default::default()
    };
    let resolved_format = read_options
        .format
        .unwrap_or_else(|| detect_mapping_format(&mapping_path));
    read_options.format = Some(resolved_format);

    info!(
        "Loading mapping ({}) from {}",
        resolved_format.display_name(),
        mapping_path.display()
    );

    let entries = load_mapping_entries(
        &mapping_path,
        &read_options,
        &source_selector,
        &output_selector,
    )
    .with_context(|| format!("failed to load mapping from {}", mapping_path.display()))?;
    if entries.is_empty() {
        anyhow::bail!(
            "no usable rows found in mapping file {}",
            mapping_path.display()
        );
    }

    let mapping_dir = mapping_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));

    let mut items = Vec::new();
    for (idx, entry) in entries.into_iter().enumerate() {
        let row_no = idx + 1;
        let raw_source = PathBuf::from(entry.source_path);

        // Security: reject absolute paths from mapping files; sources must be
        // relative to the mapping file's directory.
        if raw_source.is_absolute() {
            warn!(
                "Skipping mapping row {}: absolute source paths are not allowed ({})",
                row_no,
                raw_source.display()
            );
            continue;
        }

        let resolved_source = mapping_dir.join(&raw_source);

        // Verify the resolved path does not escape the mapping directory via `..`.
        if let (Ok(canon_source), Ok(canon_dir)) =
            (resolved_source.canonicalize(), mapping_dir.canonicalize())
            && !canon_source.starts_with(&canon_dir)
        {
            warn!(
                "Skipping mapping row {}: source path escapes the mapping directory ({})",
                row_no,
                raw_source.display()
            );
            continue;
        }

        if !resolved_source.exists() {
            warn!(
                "Skipping mapping row {}: source {} was not found",
                row_no,
                resolved_source.display()
            );
            continue;
        }
        items.push(ProcessingItem {
            source: resolved_source,
            output_override: Some(PathBuf::from(entry.output_name)),
            mapping_row: Some(row_no),
        });
    }

    if items.is_empty() {
        anyhow::bail!(
            "mapping file {} did not produce any usable rows",
            mapping_path.display()
        );
    }

    info!(
        "Loaded {} mapping row(s) from {}",
        items.len(),
        mapping_path.display()
    );

    Ok(items)
}

fn parse_mapping_format_token(token: &str) -> Result<MappingFormat> {
    match token.to_ascii_lowercase().as_str() {
        "csv" | "delimited" | "text" => Ok(MappingFormat::Csv),
        "excel" | "xlsx" | "xls" => Ok(MappingFormat::Excel),
        "parquet" | "pq" => Ok(MappingFormat::Parquet),
        "sqlite" | "sql" | "db" => Ok(MappingFormat::Sqlite),
        other => anyhow::bail!(
            "unknown mapping format '{other}' (supported: csv, excel, parquet, sqlite)"
        ),
    }
}

pub fn resolve_override_output_path(
    output_dir: &Path,
    override_target: &Path,
    ext: &str,
    face_index: usize,
    multi_face: bool,
) -> PathBuf {
    let cleaned_ext = ext.trim_start_matches('.').to_string();

    // Security: reject absolute paths and path traversal components from mapping data.
    let safe_target = sanitize_override_path(override_target);

    let rel_parent = safe_target.parent().unwrap_or_else(|| Path::new(""));
    let parent = output_dir.join(rel_parent);

    let base_name = safe_target
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "output".to_string());
    let final_base = if multi_face {
        format!("{base_name}_face{}", face_index + 1)
    } else {
        base_name
    };
    let mut final_path = parent;
    final_path.push(format!("{final_base}.{cleaned_ext}"));
    final_path
}

/// Strip path traversal components and root prefixes so the result is always
/// a relative path that stays inside the output directory.
fn sanitize_override_path(p: &Path) -> PathBuf {
    use std::path::Component;
    p.components()
        .filter(|c| matches!(c, Component::Normal(_)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use std::fs;
    use tempfile::tempdir;

    fn write_file(path: &Path) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create parent directories");
        }
        fs::write(path, b"fixture").expect("write test file");
    }

    fn write_mapping(path: &Path, contents: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create mapping directory");
        }
        fs::write(path, contents).expect("write mapping file");
    }

    fn parse_detect_args<I, T>(args: I) -> DetectArgs
    where
        I: IntoIterator<Item = T>,
        T: Into<std::ffi::OsString> + Clone,
    {
        DetectArgs::parse_from(args)
    }

    fn assert_same_existing_path(actual: &Path, expected: &Path) {
        assert_eq!(
            actual.canonicalize().expect("canonicalize actual path"),
            expected.canonicalize().expect("canonicalize expected path")
        );
    }

    #[test]
    fn collect_images_returns_supported_files_in_sorted_order() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        let nested = root.join("nested");

        write_file(&root.join("b.JPG"));
        write_file(&root.join("a.png"));
        write_file(&nested.join("c.webp"));
        write_file(&nested.join("ignore.txt"));

        let images = collect_images(root).expect("collect images from dir");

        assert_eq!(
            images,
            vec![
                root.join("a.png"),
                root.join("b.JPG"),
                nested.join("c.webp"),
            ]
        );
    }

    #[test]
    fn collect_images_errors_for_missing_path() {
        let dir = tempdir().expect("tempdir");
        let err = collect_images(&dir.path().join("missing"))
            .unwrap_err()
            .to_string();

        assert!(err.contains("input path is neither file nor directory"));
    }

    #[test]
    fn collect_images_returns_single_file_when_input_is_a_file() {
        let dir = tempdir().expect("tempdir");
        let image = dir.path().join("single.jpeg");
        write_file(&image);

        let images = collect_images(&image).expect("collect single image file");
        assert_eq!(images, vec![image]);
    }

    #[test]
    fn collect_standard_targets_errors_when_directory_has_no_images() {
        let dir = tempdir().expect("tempdir");
        write_file(&dir.path().join("notes.txt"));

        let err = collect_standard_targets(dir.path())
            .unwrap_err()
            .to_string();

        assert!(err.contains("no images found"));
    }

    #[test]
    fn collect_mapping_targets_requires_source_and_output_columns() {
        let dir = tempdir().expect("tempdir");
        let mapping_path = dir.path().join("mapping.csv");
        write_mapping(&mapping_path, "source,output\nimg.jpg,out\n");

        let args = parse_detect_args([
            "fcs-cli",
            "--mapping-file",
            mapping_path.to_str().expect("mapping path is valid UTF-8"),
            "--mapping-output-col",
            "output",
        ]);

        let err = collect_mapping_targets(&mapping_path, &args)
            .unwrap_err()
            .to_string();

        assert!(err.contains("--mapping-source-col is required"));
    }

    #[test]
    fn collect_mapping_targets_requires_output_column() {
        let dir = tempdir().expect("tempdir");
        let mapping_path = dir.path().join("mapping.csv");
        write_mapping(&mapping_path, "source,output\nimg.jpg,out\n");

        let args = parse_detect_args([
            "fcs-cli",
            "--mapping-file",
            mapping_path.to_str().expect("mapping path is valid UTF-8"),
            "--mapping-source-col",
            "source",
        ]);

        let err = collect_mapping_targets(&mapping_path, &args)
            .unwrap_err()
            .to_string();

        assert!(err.contains("--mapping-output-col is required"));
    }

    #[test]
    fn collect_mapping_targets_loads_relative_sources_from_mapping_directory() {
        let dir = tempdir().expect("tempdir");
        let mapping_dir = dir.path().join("maps");
        let mapping_path = mapping_dir.join("mapping.csv");
        let image_a = mapping_dir.join("images").join("a.jpg");
        let image_b = mapping_dir.join("b.png");
        write_file(&image_a);
        write_file(&image_b);
        write_mapping(
            &mapping_path,
            "source,output\nimages/a.jpg,nested/out-a\nb.png,out-b\n",
        );

        let args = parse_detect_args([
            "fcs-cli",
            "--mapping-file",
            mapping_path.to_str().expect("mapping path is valid UTF-8"),
            "--mapping-source-col",
            "source",
            "--mapping-output-col",
            "output",
        ]);

        let items = collect_mapping_targets(&mapping_path, &args).expect("collect mapping targets");

        assert_eq!(items.len(), 2);
        assert_same_existing_path(&items[0].source, &image_a);
        assert_eq!(
            items[0].output_override,
            Some(PathBuf::from("nested/out-a"))
        );
        assert_eq!(items[0].mapping_row, Some(1));
        assert_same_existing_path(&items[1].source, &image_b);
        assert_eq!(items[1].output_override, Some(PathBuf::from("out-b")));
        assert_eq!(items[1].mapping_row, Some(2));
    }

    #[test]
    fn collect_mapping_targets_skips_absolute_and_missing_source_paths() {
        let dir = tempdir().expect("tempdir");
        let mapping_dir = dir.path().join("maps");
        let mapping_path = mapping_dir.join("mapping.csv");
        let valid_image = mapping_dir.join("valid.jpg");
        let missing_image = mapping_dir.join("missing.jpg");
        write_file(&valid_image);

        let absolute = dir.path().join("absolute.jpg");
        write_mapping(
            &mapping_path,
            &format!(
                "source,output\n{},abs-out\nmissing.jpg,missing-out\nvalid.jpg,valid-out\n",
                absolute.display()
            ),
        );

        let args = parse_detect_args([
            "fcs-cli",
            "--mapping-file",
            mapping_path.to_str().expect("mapping path is valid UTF-8"),
            "--mapping-source-col",
            "source",
            "--mapping-output-col",
            "output",
        ]);

        let items = collect_mapping_targets(&mapping_path, &args).expect("collect mapping targets");

        assert_eq!(items.len(), 1);
        assert_same_existing_path(&items[0].source, &valid_image);
        assert_eq!(items[0].output_override, Some(PathBuf::from("valid-out")));
        assert_eq!(items[0].mapping_row, Some(3));
        assert!(!missing_image.exists());
    }

    #[test]
    fn collect_mapping_targets_skips_parent_directory_escapes() {
        let dir = tempdir().expect("tempdir");
        let mapping_dir = dir.path().join("maps");
        let mapping_path = mapping_dir.join("mapping.csv");
        let outside = dir.path().join("outside.jpg");
        let inside = mapping_dir.join("inside.jpg");
        write_file(&outside);
        write_file(&inside);
        write_mapping(
            &mapping_path,
            "source,output\n../outside.jpg,escaped\ninside.jpg,inside-out\n",
        );

        let args = parse_detect_args([
            "fcs-cli",
            "--mapping-file",
            mapping_path.to_str().expect("mapping path is valid UTF-8"),
            "--mapping-source-col",
            "source",
            "--mapping-output-col",
            "output",
        ]);

        let items = collect_mapping_targets(&mapping_path, &args).expect("collect mapping targets");

        assert_eq!(items.len(), 1);
        assert_same_existing_path(&items[0].source, &inside);
        assert_eq!(items[0].output_override, Some(PathBuf::from("inside-out")));
        assert_eq!(items[0].mapping_row, Some(2));
    }

    #[test]
    fn collect_mapping_targets_supports_explicit_text_format_and_delimiter() {
        let dir = tempdir().expect("tempdir");
        let mapping_dir = dir.path().join("maps");
        let mapping_path = mapping_dir.join("mapping.txt");
        let image = mapping_dir.join("semi.jpg");
        write_file(&image);
        write_mapping(&mapping_path, "source;output\nsemi.jpg;out-name\n");

        let args = parse_detect_args([
            "fcs-cli",
            "--mapping-file",
            mapping_path.to_str().expect("mapping path is valid UTF-8"),
            "--mapping-source-col",
            "source",
            "--mapping-output-col",
            "output",
            "--mapping-format",
            "text",
            "--mapping-delimiter",
            ";",
        ]);

        let items = collect_mapping_targets(&mapping_path, &args)
            .expect("collect mapping targets with text format");
        assert_eq!(items.len(), 1);
        assert_same_existing_path(&items[0].source, &image);
        assert_eq!(items[0].output_override, Some(PathBuf::from("out-name")));
    }

    #[test]
    fn collect_mapping_targets_errors_when_mapping_has_no_usable_rows() {
        let dir = tempdir().expect("tempdir");
        let mapping_path = dir.path().join("mapping.csv");
        write_mapping(&mapping_path, "source,output\n");

        let args = parse_detect_args([
            "fcs-cli",
            "--mapping-file",
            mapping_path.to_str().expect("mapping path is valid UTF-8"),
            "--mapping-source-col",
            "source",
            "--mapping-output-col",
            "output",
        ]);

        let err = collect_mapping_targets(&mapping_path, &args)
            .unwrap_err()
            .to_string();

        assert!(err.contains("no usable rows found"));
    }

    #[test]
    fn collect_mapping_targets_errors_when_every_row_is_filtered_out() {
        let dir = tempdir().expect("tempdir");
        let mapping_dir = dir.path().join("maps");
        let mapping_path = mapping_dir.join("mapping.csv");
        write_mapping(&mapping_path, "source,output\nmissing.jpg,out\n");

        let args = parse_detect_args([
            "fcs-cli",
            "--mapping-file",
            mapping_path.to_str().expect("mapping path is valid UTF-8"),
            "--mapping-source-col",
            "source",
            "--mapping-output-col",
            "output",
        ]);

        let err = collect_mapping_targets(&mapping_path, &args)
            .unwrap_err()
            .to_string();

        assert!(err.contains("did not produce any usable rows"));
    }

    #[test]
    fn parse_mapping_format_token_accepts_aliases() {
        assert_eq!(
            parse_mapping_format_token("csv").expect("csv token"),
            MappingFormat::Csv
        );
        assert_eq!(
            parse_mapping_format_token("delimited").expect("delimited token"),
            MappingFormat::Csv
        );
        assert_eq!(
            parse_mapping_format_token("xlsx").expect("xlsx token"),
            MappingFormat::Excel
        );
        assert_eq!(
            parse_mapping_format_token("pq").expect("pq token"),
            MappingFormat::Parquet
        );
        assert_eq!(
            parse_mapping_format_token("db").expect("db token"),
            MappingFormat::Sqlite
        );
        assert!(parse_mapping_format_token("toml").is_err());
    }

    #[test]
    fn resolve_override_output_path_sanitizes_traversal_and_preserves_extension() {
        let dir = tempdir().expect("tempdir");
        let output = resolve_override_output_path(
            dir.path(),
            Path::new("../nested/folder/custom.name.png"),
            ".webp",
            0,
            false,
        );

        assert_eq!(
            output,
            dir.path()
                .join("nested")
                .join("folder")
                .join("custom.name.webp")
        );
    }

    #[test]
    fn resolve_override_output_path_appends_face_suffix_for_multi_face_exports() {
        let dir = tempdir().expect("tempdir");
        let output = resolve_override_output_path(
            dir.path(),
            Path::new("gallery/portrait.jpg"),
            "png",
            2,
            true,
        );

        assert_eq!(
            output,
            dir.path().join("gallery").join("portrait_face3.png")
        );
    }

    #[test]
    fn resolve_override_output_path_falls_back_to_default_name_when_override_is_sanitized_away() {
        let dir = tempdir().expect("tempdir");
        let output = resolve_override_output_path(dir.path(), Path::new("../../"), "jpg", 0, false);

        assert_eq!(output, dir.path().join("output.jpg"));
    }
}
