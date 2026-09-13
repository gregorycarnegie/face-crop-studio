//! Parquet mapping reader.

use super::common::{MappingTable, ensure_columns, format_parquet_field};
use anyhow::{Context, Result};
use parquet::{
    file::reader::{FileReader, SerializedFileReader},
    record::Row,
};
use std::{fs::File, path::Path};

pub(super) fn table_parquet_internal(
    path: &Path,
    row_limit: Option<usize>,
) -> Result<MappingTable> {
    let file = File::open(path)
        .with_context(|| format!("failed to open parquet file {}", path.display()))?;
    let reader =
        SerializedFileReader::new(file).with_context(|| "failed to create parquet reader")?;

    // Headers from the same level the row iterator yields values at: top-level fields, not leaf
    // columns, or a nested group shifts every later value under the wrong header.
    let mut columns: Vec<String> = reader
        .metadata()
        .file_metadata()
        .schema()
        .get_fields()
        .iter()
        .map(|field| field.name().to_string())
        .collect();
    if columns.is_empty() {
        anyhow::bail!("parquet file {} has no columns", path.display());
    }

    let mut rows = Vec::new();
    let mut total_rows = 0usize;
    let iter = reader
        .get_row_iter(None)
        .with_context(|| "failed to build parquet row iterator")?;
    for row in iter {
        let row: Row = row?;
        let values: Vec<String> = row
            .get_column_iter()
            .map(|(_, field)| format_parquet_field(field))
            .collect();
        if values.iter().all(|v| v.is_empty()) {
            continue;
        }
        ensure_columns(&mut columns, values.len());
        if row_limit.is_none_or(|limit| rows.len() < limit) {
            rows.push(values);
        }
        total_rows += 1;
    }

    Ok(MappingTable {
        columns,
        rows,
        total_rows,
    })
}
