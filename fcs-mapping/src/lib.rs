//! Tabular mapping ingestion for batch crop source/output pairs.
//!
//! Reads CSV, Excel, Parquet, and SQLite tables into `MappingEntry` pairs of
//! source image path and desired output name, plus the previews and catalogs
//! the GUI needs to let a user pick which columns to use.

use anyhow::{Context, Result};
use calamine::{Reader as _, open_workbook_auto};
use std::path::Path;

mod common;
mod csv;
mod excel;
mod parquet;
mod sqlite;
#[cfg(test)]
mod tests;
mod types;

pub use types::{
    ColumnSelector, DEFAULT_PREVIEW_ROWS, MappingCatalog, MappingEntry, MappingFormat,
    MappingPreview, MappingReadOptions,
};

use common::{MappingTable, resolve_selector};
use csv::table_csv_internal;
use excel::table_excel_internal;
use parquet::table_parquet_internal;
use sqlite::{list_sqlite_tables_conn, table_sqlite_internal};
#[cfg(test)]
use {
    common::{encode_bytes, format_excel_cell, format_excel_header, format_parquet_field},
    sqlite::quote_identifier,
};

/// Detects a mapping format from the file extension.
pub fn detect_format(path: &Path) -> MappingFormat {
    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default()
        .as_str()
    {
        "xlsx" | "xls" | "xlsm" | "ods" => MappingFormat::Excel,
        "parquet" | "pq" => MappingFormat::Parquet,
        "db" | "sqlite" | "sqlite3" => MappingFormat::Sqlite,
        _ => MappingFormat::Csv,
    }
}

/// Builds a preview for the supplied file.
pub fn load_mapping_preview(path: &Path, options: &MappingReadOptions) -> Result<MappingPreview> {
    let format = options.format.unwrap_or_else(|| detect_format(path));
    let limit = options.preview_rows.max(1);
    let table = match format {
        MappingFormat::Csv => table_csv_internal(path, options, Some(limit))?,
        MappingFormat::Excel => table_excel_internal(path, options, Some(limit))?,
        MappingFormat::Parquet => table_parquet_internal(path, Some(limit))?,
        MappingFormat::Sqlite => table_sqlite_internal(path, options, Some(limit))?,
    };
    Ok(to_preview(format, table))
}

/// Loads every mapping entry, resolving the selected columns to source/output pairs.
pub fn load_mapping_entries(
    path: &Path,
    options: &MappingReadOptions,
    source: &ColumnSelector,
    output: &ColumnSelector,
) -> Result<Vec<MappingEntry>> {
    let format = options.format.unwrap_or_else(|| detect_format(path));
    let table = match format {
        MappingFormat::Csv => table_csv_internal(path, options, None)?,
        MappingFormat::Excel => table_excel_internal(path, options, None)?,
        MappingFormat::Parquet => table_parquet_internal(path, None)?,
        MappingFormat::Sqlite => table_sqlite_internal(path, options, None)?,
    };

    let source_idx = resolve_selector(&table.columns, source)?;
    let output_idx = resolve_selector(&table.columns, output)?;

    let entries = table
        .rows
        .into_iter()
        .filter_map(|row| {
            let source_value = row.get(source_idx)?.trim();
            let output_value = row.get(output_idx)?.trim();
            if source_value.is_empty() || output_value.is_empty() {
                return None;
            }
            Some(MappingEntry {
                source_path: source_value.to_string(),
                output_name: output_value.to_string(),
            })
        })
        .collect();
    Ok(entries)
}

/// Lists auxiliary metadata such as sheet/table names for UI drop-downs.
pub fn inspect_mapping_sources(
    path: &Path,
    options: &MappingReadOptions,
) -> Result<MappingCatalog> {
    let format = options.format.unwrap_or_else(|| detect_format(path));
    match format {
        MappingFormat::Excel => {
            let workbook = open_workbook_auto(path)
                .with_context(|| format!("failed to open workbook {}", path.display()))?;
            Ok(MappingCatalog {
                sheets: workbook.sheet_names().to_vec(),
                sql_tables: Vec::new(),
            })
        }
        MappingFormat::Sqlite => Ok(MappingCatalog {
            sheets: Vec::new(),
            sql_tables: list_sqlite_tables(path)?,
        }),
        _ => Ok(MappingCatalog::default()),
    }
}

/// Enumerates SQLite tables in the supplied database.
pub fn list_sqlite_tables(path: &Path) -> Result<Vec<String>> {
    let (conn, _denied) = sqlite::open_guarded(path)?;
    list_sqlite_tables_conn(&conn)
}

fn to_preview(format: MappingFormat, table: MappingTable) -> MappingPreview {
    let truncated = table.total_rows > table.rows.len();
    MappingPreview {
        format,
        columns: table.columns,
        total_rows: table.total_rows,
        truncated,
        rows: table.rows,
    }
}

/// The mapped output name for each of `paths`, or `None` where no row matches.
///
/// A row matches by file name, else by stem (so a row may leave the extension off). Each tries
/// the exact spelling before ignoring case: Windows and macOS treat `Photo.JPG` and `photo.jpg`
/// as one file, and spreadsheet exports recase names. The first row wins within a tier.
pub fn match_entries<'a, 'p>(
    entries: &'a [MappingEntry],
    paths: impl IntoIterator<Item = &'p Path>,
) -> Vec<Option<&'a str>> {
    let mut index = std::collections::HashMap::new();
    for entry in entries {
        let keys = match_keys(Path::new(&entry.source_path));
        for (tier, key) in keys.into_iter().enumerate() {
            if let Some(key) = key {
                index
                    .entry((tier, key))
                    .or_insert(entry.output_name.as_str());
            }
        }
    }
    paths
        .into_iter()
        .map(|path| {
            match_keys(path)
                .into_iter()
                .enumerate()
                .find_map(|(tier, key)| index.get(&(tier, key?)).copied())
        })
        .collect()
}

/// Lookup keys, most specific first: name, name ignoring case, stem, stem ignoring case.
fn match_keys(path: &Path) -> [Option<String>; 4] {
    let name = path.file_name().and_then(|s| s.to_str());
    let stem = path.file_stem().and_then(|s| s.to_str());
    [
        name.map(str::to_owned),
        name.map(str::to_lowercase),
        stem.map(str::to_owned),
        stem.map(str::to_lowercase),
    ]
}

#[cfg(test)]
mod match_tests {
    use super::*;

    #[test]
    fn mapping_matches_by_name_then_stem_preferring_the_exact_spelling() {
        let entry = |source: &str, output: &str| MappingEntry {
            source_path: source.into(),
            output_name: output.into(),
        };
        let entries = [
            // A spreadsheet export recased this one; the file on disk is lower case.
            entry("shoots/FaceCropOTBreak_1.jpg", "recased"),
            // Same stem, other extension: must not beat a name match listed later.
            entry("photo.png", "other-extension"),
            entry("PHOTO.JPG", "photo-folded"),
            entry("photo.jpg", "photo-exact"),
            entry("no-extension", "by-stem"),
        ];
        let queue = [
            "in/facecropotbreak_1.jpg",
            "in/photo.jpg",
            "in/Photo.jpg",
            "in/NO-EXTENSION.webp",
            "in/unlisted.jpg",
        ];
        assert_eq!(
            match_entries(&entries, queue.iter().map(Path::new)),
            [
                Some("recased"),
                Some("photo-exact"),
                Some("photo-folded"),
                Some("by-stem"),
                None
            ]
        );
    }
}
