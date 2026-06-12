//! CSV and delimited mapping reader.

use super::{
    common::{MappingTable, ensure_columns, format_header, normalize_row, sanitize_value},
    types::MappingReadOptions,
};
use anyhow::{Context, Result};
use csv::ReaderBuilder;
use std::path::Path;

pub(super) fn table_csv_internal(
    path: &Path,
    options: &MappingReadOptions,
    row_limit: Option<usize>,
) -> Result<MappingTable> {
    let has_headers = options.has_headers.unwrap_or(true);
    let delimiter = options.delimiter.unwrap_or(b',');
    let mut reader = ReaderBuilder::new()
        .has_headers(has_headers)
        .delimiter(delimiter)
        .from_path(path)
        .with_context(|| format!("failed to open {}", path.display()))?;

    let mut columns = if has_headers {
        reader
            .headers()
            .context("failed to read CSV headers")?
            .iter()
            .enumerate()
            .map(|(idx, raw)| format_header(raw, idx))
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };

    let mut rows = Vec::new();
    let mut total_rows = 0usize;
    for record in reader.records() {
        let record = record?;
        let mut row: Vec<String> = record.iter().map(sanitize_value).collect();
        if row.iter().all(|v| v.is_empty()) {
            continue;
        }
        ensure_columns(&mut columns, row.len());
        normalize_row(&mut row, columns.len());
        if row_limit.is_none_or(|limit| rows.len() < limit) {
            rows.push(row);
        }
        total_rows += 1;
    }

    Ok(MappingTable {
        columns,
        rows,
        total_rows,
    })
}
