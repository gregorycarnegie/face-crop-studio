//! Mapping file ingestion for batch crop source/output pairs.

use anyhow::{Context, Result};
use calamine::{Reader as _, open_workbook_auto};
use rusqlite::Connection;
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
    let conn = Connection::open(path)
        .with_context(|| format!("failed to open sqlite database {}", path.display()))?;
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
