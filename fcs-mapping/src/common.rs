//! Shared table and value formatting helpers.

use super::types::ColumnSelector;
use anyhow::{Result, anyhow};
use base64::{Engine as _, engine::general_purpose};
use calamine::Data as ExcelData;
use parquet::record::Field;
use rusqlite::types::ValueRef;

pub(super) struct MappingTable {
    pub(super) columns: Vec<String>,
    pub(super) rows: Vec<Vec<String>>,
    pub(super) total_rows: usize,
}

pub(super) fn resolve_selector(columns: &[String], selector: &ColumnSelector) -> Result<usize> {
    match selector {
        ColumnSelector::Index(idx) => {
            if *idx >= columns.len() {
                anyhow::bail!(
                    "{} is out of range ({} column(s) detected)",
                    selector.describe(),
                    columns.len()
                );
            }
            Ok(*idx)
        }
        ColumnSelector::Name(name) => columns
            .iter()
            .position(|col| col.eq_ignore_ascii_case(name))
            .ok_or_else(|| {
                anyhow!(
                    "column named \"{name}\" not found (available: {})",
                    columns.join(", ")
                )
            }),
    }
}

#[inline]
pub(super) fn sanitize_value(value: &str) -> String {
    value.trim().to_string()
}

pub(super) fn format_header(raw: &str, idx: usize) -> String {
    let trimmed = sanitize_value(raw);
    if trimmed.is_empty() {
        format!("Column {}", idx + 1)
    } else {
        trimmed
    }
}

pub(super) fn ensure_columns(columns: &mut Vec<String>, desired: usize) {
    if columns.len() >= desired {
        return;
    }
    let current = columns.len();
    for idx in current..desired {
        columns.push(format!("Column {}", idx + 1));
    }
}

pub(super) fn normalize_row(row: &mut Vec<String>, len: usize) {
    row.resize(len, String::new());
}

pub(super) fn format_excel_header(cell: &ExcelData, idx: usize) -> String {
    let text = format_excel_cell(cell);
    if text.is_empty() {
        format!("Column {}", idx + 1)
    } else {
        text
    }
}

pub(super) fn format_excel_cell(cell: &ExcelData) -> String {
    match cell {
        ExcelData::Empty => String::new(),
        ExcelData::String(s) => s.trim().to_string(),
        ExcelData::Float(f) => {
            if f.fract() == 0.0 {
                format!("{:.0}", f)
            } else {
                f.to_string()
            }
        }
        ExcelData::Int(i) => i.to_string(),
        ExcelData::Bool(b) => b.to_string(),
        ExcelData::Error(_) => String::new(),
        ExcelData::DateTime(dt) => dt.to_string(),
        ExcelData::DateTimeIso(s) => s.to_string(),
        ExcelData::DurationIso(s) => s.to_string(),
    }
}

pub(super) fn format_parquet_field(field: &Field) -> String {
    match field {
        Field::Null => String::new(),
        Field::Bool(b) => b.to_string(),
        Field::Byte(v) => v.to_string(),
        Field::Short(v) => v.to_string(),
        Field::Int(v) => v.to_string(),
        Field::Long(v) => v.to_string(),
        Field::UByte(v) => v.to_string(),
        Field::UShort(v) => v.to_string(),
        Field::UInt(v) => v.to_string(),
        Field::ULong(v) => v.to_string(),
        Field::Float16(v) => v.to_string(),
        Field::Float(v) => v.to_string(),
        Field::Double(v) => v.to_string(),
        Field::Str(s) => s.trim().to_string(),
        Field::Bytes(bytes) => encode_bytes(bytes.data()),
        Field::Decimal(value) => format!("{value:?}"),
        Field::Date(v) => v.to_string(),
        Field::TimeMillis(v) => v.to_string(),
        Field::TimeMicros(v) => v.to_string(),
        Field::TimestampMillis(v) => v.to_string(),
        Field::TimestampMicros(v) => v.to_string(),
        Field::Group(group) => format!(
            "{{{}}}",
            group
                .get_column_iter()
                .map(|(name, value)| format!("{name}: {}", format_parquet_field(value)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Field::ListInternal(list) => format!(
            "[{}]",
            list.elements()
                .iter()
                .map(format_parquet_field)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Field::MapInternal(map) => format!("{map:?}"),
    }
}

#[inline]
pub(super) fn format_sql_value(value: ValueRef<'_>) -> String {
    match value {
        ValueRef::Null => String::new(),
        ValueRef::Integer(i) => i.to_string(),
        ValueRef::Real(r) => r.to_string(),
        ValueRef::Text(text) => String::from_utf8_lossy(text).trim().to_string(),
        ValueRef::Blob(blob) => encode_bytes(blob),
    }
}

pub(super) fn encode_bytes<B: AsRef<[u8]>>(bytes: B) -> String {
    let slice = bytes.as_ref();
    if slice.is_empty() {
        String::new()
    } else {
        general_purpose::STANDARD.encode(slice)
    }
}
