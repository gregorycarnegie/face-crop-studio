//! Public mapping data types.

use anyhow::Result;

pub const DEFAULT_PREVIEW_ROWS: usize = 32;

/// Supported mapping formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MappingFormat {
    Csv,
    Excel,
    Parquet,
    Sqlite,
}

impl MappingFormat {
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Csv => "CSV / Delimited",
            Self::Excel => "Excel",
            Self::Parquet => "Parquet",
            Self::Sqlite => "SQLite",
        }
    }
}

/// Column selector used to resolve user selections to a zero-based index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ColumnSelector {
    Index(usize),
    Name(String),
}

impl ColumnSelector {
    pub fn by_index(index: usize) -> Self {
        Self::Index(index)
    }

    pub fn by_name(name: impl Into<String>) -> Self {
        Self::Name(name.into())
    }

    pub fn describe(&self) -> String {
        match self {
            Self::Index(idx) => format!("column #{idx}"),
            Self::Name(name) => format!("column \"{name}\""),
        }
    }

    /// Parses a CLI-style token (`#3` or `3` for indices, any other value for names).
    pub fn parse_token(token: &str) -> Result<Self> {
        let trimmed = token.trim();
        if trimmed.is_empty() {
            anyhow::bail!("column selector cannot be empty");
        }
        let digits = trimmed.strip_prefix('#').unwrap_or(trimmed);
        if digits.chars().all(|c| c.is_ascii_digit()) {
            let idx: usize = digits.parse()?;
            return Ok(Self::Index(idx));
        }
        Ok(Self::Name(trimmed.to_string()))
    }
}

/// Options that influence how a mapping file is read.
#[derive(Clone, Debug)]
pub struct MappingReadOptions {
    pub format: Option<MappingFormat>,
    pub has_headers: Option<bool>,
    pub delimiter: Option<u8>,
    pub sheet_name: Option<String>,
    pub sql_table: Option<String>,
    pub sql_query: Option<String>,
    pub preview_rows: usize,
}

impl Default for MappingReadOptions {
    fn default() -> Self {
        Self {
            format: None,
            has_headers: None,
            delimiter: None,
            sheet_name: None,
            sql_table: None,
            sql_query: None,
            preview_rows: DEFAULT_PREVIEW_ROWS,
        }
    }
}

/// Preview payload shared with the GUI.
#[derive(Clone, Debug)]
pub struct MappingPreview {
    pub format: MappingFormat,
    pub columns: Vec<String>,
    pub rows: Vec<Vec<String>>,
    pub total_rows: usize,
    pub truncated: bool,
}

/// Fully materialised mapping entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MappingEntry {
    pub source_path: String,
    pub output_name: String,
}

/// Additional metadata exposed to the UI (e.g. sheet or table names).
#[derive(Clone, Debug, Default)]
pub struct MappingCatalog {
    pub sheets: Vec<String>,
    pub sql_tables: Vec<String>,
}
