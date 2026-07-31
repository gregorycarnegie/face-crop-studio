//! SQLite mapping reader and query validation.

use super::{
    common::{MappingTable, format_sql_value},
    types::MappingReadOptions,
};
use anyhow::{Context, Result, anyhow};
use rusqlite::Connection;
use std::path::Path;

pub(super) fn table_sqlite_internal(
    path: &Path,
    options: &MappingReadOptions,
    row_limit: Option<usize>,
) -> Result<MappingTable> {
    let conn = Connection::open(path)
        .with_context(|| format!("failed to open sqlite database {}", path.display()))?;
    let sql = resolve_sql_query(&conn, options)?;
    let mut stmt = conn.prepare(&sql)?;
    let columns = stmt
        .column_names()
        .iter()
        .enumerate()
        .map(|(idx, name)| {
            let trimmed = name.trim();
            if trimmed.is_empty() {
                format!("Column {}", idx + 1)
            } else {
                trimmed.to_string()
            }
        })
        .collect::<Vec<_>>();
    if columns.is_empty() {
        anyhow::bail!("SQL query returned no columns");
    }

    let mut rows_iter = stmt.query([])?;

    let mut rows = Vec::new();
    let mut total_rows = 0usize;
    while let Some(row) = rows_iter.next()? {
        let mut values = Vec::with_capacity(columns.len());
        for idx in 0..columns.len() {
            let value = row.get_ref(idx)?;
            values.push(format_sql_value(value));
        }
        if values.iter().all(|v| v.is_empty()) {
            continue;
        }
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

fn resolve_sql_query(conn: &Connection, options: &MappingReadOptions) -> Result<String> {
    if let Some(query) = options
        .sql_query
        .as_ref()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
    {
        validate_sql_query(query)?;
        return Ok(query.to_string());
    }

    let table_name = if let Some(explicit) = options
        .sql_table
        .as_ref()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
    {
        explicit.to_string()
    } else {
        list_sqlite_tables_conn(conn)?
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("database does not contain any tables"))?
    };
    Ok(format!("SELECT * FROM {}", quote_identifier(&table_name)))
}

pub(super) fn list_sqlite_tables_conn(conn: &Connection) -> Result<Vec<String>> {
    let mut stmt =
        conn.prepare("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY LOWER(name)")?;
    let rows = stmt
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(rows)
}

pub(super) fn quote_identifier(name: &str) -> String {
    let escaped = name.replace('"', "\"\"");
    format!("\"{escaped}\"")
}

/// Validate a user-supplied SQL query to prevent injection.
/// Only read-only SELECT statements are permitted; DDL/DML and multiple
/// statements are rejected.
fn validate_sql_query(query: &str) -> Result<()> {
    let normalized = query.trim().to_ascii_uppercase();

    anyhow::ensure!(
        normalized.starts_with("SELECT"),
        "custom SQL queries must begin with SELECT"
    );

    // Reject statement separators (prevents multi-statement injection).
    anyhow::ensure!(
        !query.contains(';'),
        "custom SQL queries must not contain semicolons"
    );

    // Reject DDL/DML keywords that could modify the database.
    let forbidden = [
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "REPLACE", "ATTACH", "DETACH",
        "PRAGMA", "REINDEX", "VACUUM",
    ];
    for keyword in &forbidden {
        // Match as whole word: check that the character before and after is not alphanumeric.
        let upper = normalized.as_str();
        let mut start = 0;
        while let Some(pos) = upper[start..].find(keyword) {
            let abs_pos = start + pos;
            let before_ok = abs_pos == 0 || !upper.as_bytes()[abs_pos - 1].is_ascii_alphanumeric();
            let after_pos = abs_pos + keyword.len();
            let after_ok =
                after_pos >= upper.len() || !upper.as_bytes()[after_pos].is_ascii_alphanumeric();
            if before_ok && after_ok {
                anyhow::bail!("custom SQL queries must not contain the {keyword} keyword");
            }
            start = abs_pos + keyword.len();
        }
    }

    Ok(())
}
