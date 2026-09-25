//! SQLite mapping reader and query validation.

use super::{
    common::{MappingTable, format_sql_value},
    types::MappingReadOptions,
};
use anyhow::{Context, Result, anyhow};
use rusqlite::{
    Connection, OpenFlags,
    hooks::{AuthAction, AuthContext, Authorization},
};
use std::{
    path::Path,
    sync::{Arc, Mutex},
};

/// Engine-level read-only. `SQLITE_OPEN_URI` is deliberately absent — rusqlite's default
/// flags include it, and with it a path spelled `file:x.db?mode=rwc` reopens writable.
pub(super) const READ_ONLY: OpenFlags =
    OpenFlags::SQLITE_OPEN_READ_ONLY.union(OpenFlags::SQLITE_OPEN_NO_MUTEX);

/// What the authorizer last refused, so a denied `prepare` can name the action instead of
/// reporting SQLite's bare "not authorized".
pub(super) type Denied = Arc<Mutex<Option<String>>>;

/// Open a mapping database with both guards in place.
///
/// The authorizer is the real one: SQLite calls it for every action while it compiles a
/// statement, so it is an allowlist checked by the parser rather than a denylist of ours
/// checked against the query text. The keyword scan this replaced could only ever be as
/// complete as its twelve words, and had to hand-roll word boundaries to avoid rejecting a
/// column called `updated_at`.
pub(super) fn open_guarded(path: &Path) -> Result<(Connection, Denied)> {
    let conn = Connection::open_with_flags(path, READ_ONLY)
        .with_context(|| format!("failed to open sqlite database {}", path.display()))?;
    let denied: Denied = Denied::default();
    let sink = Arc::clone(&denied);
    conn.authorizer(Some(move |ctx: AuthContext<'_>| {
        if is_read_only(&ctx.action) {
            return Authorization::Allow;
        }
        if let Ok(mut slot) = sink.lock() {
            *slot = Some(format!("{:?}", ctx.action));
        }
        Authorization::Deny
    }))
    .context("failed to install the SQLite authorizer")?;
    Ok((conn, denied))
}

/// Whether an action SQLite is about to take only reads.
///
/// A named function rather than a `match` inside the closure above, because cargo-mutants
/// generates no mutants at all for a closure body: the allowlist would be the one piece of
/// this module measured by nothing. As a `-> bool` it gets `replace with true` and `replace
/// with false`, both of which `the_authorizer_refuses_every_action_but_reading` kills.
fn is_read_only(action: &AuthAction<'_>) -> bool {
    matches!(
        action,
        // Everything a read-only query needs: the SELECT itself, the columns it touches,
        // the functions it calls, and the recursion a CTE may use. Every write, schema
        // change, ATTACH, DETACH and PRAGMA is absent, as is any action a future SQLite
        // adds that this code has never heard of.
        AuthAction::Select
            | AuthAction::Read { .. }
            | AuthAction::Function { .. }
            | AuthAction::Recursive
    )
}

/// Compile `sql`, reporting a refusal by the authorizer as such.
///
/// Without this a denied statement surfaces as SQLite's "not authorized", which tells the
/// person who typed the query nothing about which part of it was refused.
pub(super) fn prepare_guarded<'c>(
    conn: &'c Connection,
    denied: &Denied,
    sql: &str,
) -> Result<rusqlite::Statement<'c>> {
    // The slot holds whatever was last refused on this connection, so a later `prepare`
    // failing for an ordinary syntax error would inherit the earlier refusal and report it.
    // Only one user query is compiled per connection today, which is precisely when a
    // mix-up like this is cheap to remove rather than expensive to diagnose.
    if let Ok(mut slot) = denied.lock() {
        *slot = None;
    }
    conn.prepare(sql).map_err(
        |err| match denied.lock().ok().and_then(|slot| slot.clone()) {
            Some(action) => anyhow!("custom SQL queries may only read; SQLite refused {action}"),
            None => anyhow::Error::new(err).context("failed to prepare the SQL query"),
        },
    )
}

pub(super) fn table_sqlite_internal(
    path: &Path,
    options: &MappingReadOptions,
    row_limit: Option<usize>,
) -> Result<MappingTable> {
    let (conn, denied) = open_guarded(path)?;
    let sql = resolve_sql_query(&conn, options)?;
    let mut stmt = prepare_guarded(&conn, &denied, &sql)?;
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

/// Reject a user-supplied query that is not a single SELECT.
///
/// This is not what makes the query read-only — [`open_guarded`]'s authorizer is. It
/// rejects the two shapes that would otherwise be silently misread: `prepare` compiles the
/// first statement and ignores any tail, so a trailing statement would vanish rather than
/// run, and a non-SELECT would come back with no columns rather than an error.
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

    Ok(())
}
