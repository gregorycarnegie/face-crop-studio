//! Excel mapping reader.

use super::{
    common::{MappingTable, ensure_columns, format_excel_cell, format_excel_header, normalize_row},
    types::MappingReadOptions,
};
use anyhow::{Context, Result, anyhow};
use calamine::{Reader as _, open_workbook_auto};
use std::path::Path;

pub(super) fn table_excel_internal(
    path: &Path,
    options: &MappingReadOptions,
    row_limit: Option<usize>,
) -> Result<MappingTable> {
    let mut workbook = open_workbook_auto(path)
        .with_context(|| format!("failed to open workbook {}", path.display()))?;
    let sheet_name = match options
        .sheet_name
        .as_ref()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
    {
        Some(explicit) => explicit.to_string(),
        None => workbook
            .sheet_names()
            .first()
            .cloned()
            .ok_or_else(|| anyhow!("workbook {} has no sheets", path.display()))?,
    };

    let has_headers = options.has_headers.unwrap_or(true);
    let range = workbook
        .worksheet_range(&sheet_name)
        .map_err(|e| anyhow!("failed to read sheet {sheet_name}: {e}"))?;

    let mut rows_iter = range.rows();
    let mut columns = if has_headers {
        rows_iter
            .next()
            .map(|header_row| {
                header_row
                    .iter()
                    .enumerate()
                    .map(|(idx, cell)| format_excel_header(cell, idx))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    let mut rows = Vec::new();
    let mut total_rows = 0usize;
    for row in rows_iter {
        let mut values: Vec<String> = row.iter().map(format_excel_cell).collect();
        if values.iter().all(|v| v.is_empty()) {
            continue;
        }
        ensure_columns(&mut columns, values.len());
        normalize_row(&mut values, columns.len());
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
