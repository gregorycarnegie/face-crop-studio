use super::{
    encode_bytes, format_excel_cell, format_excel_header, format_parquet_field, quote_identifier, *,
};
use std::{
    fs,
    path::{Path, PathBuf},
};
use tempfile::tempdir;

fn write_text(path: &Path, contents: &str) {
    fs::write(path, contents).expect("write fixture file");
}

fn write_csv(dir: &Path, name: &str, contents: &str) -> PathBuf {
    let path = dir.join(name);
    write_text(&path, contents);
    path
}

fn write_sqlite(dir: &Path, name: &str, setup_sql: &str) -> PathBuf {
    let path = dir.join(name);
    let conn = Connection::open(&path).expect("open sqlite");
    conn.execute_batch(setup_sql).expect("seed sqlite");
    path
}

#[test]
fn column_selector_parse_token_accepts_indices_and_names() {
    assert_eq!(
        ColumnSelector::parse_token("#3").unwrap(),
        ColumnSelector::Index(3)
    );
    assert_eq!(
        ColumnSelector::parse_token("7").unwrap(),
        ColumnSelector::Index(7)
    );
    assert_eq!(
        ColumnSelector::parse_token(" output_name ").unwrap(),
        ColumnSelector::Name("output_name".to_string())
    );
    assert!(ColumnSelector::parse_token("   ").is_err());
}

#[test]
fn detect_format_uses_common_file_extensions() {
    assert_eq!(detect_format(Path::new("mapping.csv")), MappingFormat::Csv);
    assert_eq!(
        detect_format(Path::new("mapping.xlsx")),
        MappingFormat::Excel
    );
    assert_eq!(
        detect_format(Path::new("mapping.PQ")),
        MappingFormat::Parquet
    );
    assert_eq!(
        detect_format(Path::new("mapping.sqlite3")),
        MappingFormat::Sqlite
    );
}

#[test]
fn csv_preview_truncates_rows_and_keeps_headers() {
    let dir = tempdir().unwrap();
    let path = write_csv(
        dir.path(),
        "mapping.csv",
        "source,output,label\n a.jpg , out-a , one \n\n b.jpg,out-b,two\n c.jpg,out-c,three\n",
    );

    let options = MappingReadOptions {
        preview_rows: 2,
        ..Default::default()
    };

    let preview = load_mapping_preview(&path, &options).unwrap();

    assert_eq!(preview.format, MappingFormat::Csv);
    assert_eq!(preview.columns, vec!["source", "output", "label"]);
    assert_eq!(preview.total_rows, 3);
    assert!(preview.truncated);
    assert_eq!(preview.rows.len(), 2);
    assert_eq!(preview.rows[0], vec!["a.jpg", "out-a", "one"]);
    assert_eq!(preview.rows[1], vec!["b.jpg", "out-b", "two"]);
}

#[test]
fn csv_preview_without_headers_generates_default_column_names() {
    let dir = tempdir().unwrap();
    let path = write_csv(
        dir.path(),
        "mapping.csv",
        " a.jpg , out-a \n b.jpg , out-b \n\n",
    );

    let options = MappingReadOptions {
        has_headers: Some(false),
        ..Default::default()
    };

    let preview = load_mapping_preview(&path, &options).unwrap();

    assert_eq!(preview.columns, vec!["Column 1", "Column 2"]);
    assert_eq!(preview.total_rows, 2);
    assert!(!preview.truncated);
    assert_eq!(preview.rows[0], vec!["a.jpg", "out-a"]);
    assert_eq!(preview.rows[1], vec!["b.jpg", "out-b"]);
}

#[test]
fn load_mapping_entries_by_name_skips_blank_source_or_output_values() {
    let dir = tempdir().unwrap();
    let path = write_csv(
        dir.path(),
        "mapping.csv",
        "source,output\n img1.jpg , out1 \n , out2 \n img3.jpg ,   \n img4.jpg , out4 \n",
    );

    let entries = load_mapping_entries(
        &path,
        &MappingReadOptions::default(),
        &ColumnSelector::by_name("source"),
        &ColumnSelector::by_name("output"),
    )
    .unwrap();

    assert_eq!(
        entries,
        vec![
            MappingEntry {
                source_path: "img1.jpg".to_string(),
                output_name: "out1".to_string(),
            },
            MappingEntry {
                source_path: "img4.jpg".to_string(),
                output_name: "out4".to_string(),
            },
        ]
    );
}

#[test]
fn load_mapping_entries_by_index_uses_zero_based_selector() {
    let dir = tempdir().unwrap();
    let path = write_csv(
        dir.path(),
        "mapping.csv",
        "ignore,source,output\n 1 , a.jpg , out-a \n 2 , b.jpg , out-b \n",
    );

    let entries = load_mapping_entries(
        &path,
        &MappingReadOptions::default(),
        &ColumnSelector::by_index(1),
        &ColumnSelector::by_index(2),
    )
    .unwrap();

    assert_eq!(
        entries,
        vec![
            MappingEntry {
                source_path: "a.jpg".to_string(),
                output_name: "out-a".to_string(),
            },
            MappingEntry {
                source_path: "b.jpg".to_string(),
                output_name: "out-b".to_string(),
            },
        ]
    );
}

#[test]
fn inspect_mapping_sources_lists_sqlite_tables() {
    let dir = tempdir().unwrap();
    let path = write_sqlite(
        dir.path(),
        "mapping.db",
        r#"
            CREATE TABLE photos (source TEXT, output TEXT);
            CREATE TABLE queue (source TEXT, output TEXT);
            "#,
    );

    let catalog = inspect_mapping_sources(
        &path,
        &MappingReadOptions {
            format: Some(MappingFormat::Sqlite),
            ..Default::default()
        },
    )
    .unwrap();

    assert!(catalog.sheets.is_empty());
    assert_eq!(catalog.sql_tables, vec!["photos", "queue"]);
}

#[test]
fn sqlite_mapping_uses_first_table_by_default() {
    let dir = tempdir().unwrap();
    let path = write_sqlite(
        dir.path(),
        "mapping.db",
        r#"
            CREATE TABLE alpha (source TEXT, output TEXT);
            INSERT INTO alpha VALUES ('a.jpg', 'out-a');
            CREATE TABLE beta (source TEXT, output TEXT);
            INSERT INTO beta VALUES ('b.jpg', 'out-b');
            "#,
    );

    let preview = load_mapping_preview(
        &path,
        &MappingReadOptions {
            format: Some(MappingFormat::Sqlite),
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(preview.columns, vec!["source", "output"]);
    assert_eq!(
        preview.rows,
        vec![vec!["a.jpg".to_string(), "out-a".to_string()]]
    );
}

#[test]
fn sqlite_mapping_respects_explicit_table() {
    let dir = tempdir().unwrap();
    let path = write_sqlite(
        dir.path(),
        "mapping.db",
        r#"
            CREATE TABLE alpha (source TEXT, output TEXT);
            INSERT INTO alpha VALUES ('a.jpg', 'out-a');
            CREATE TABLE beta (source TEXT, output TEXT);
            INSERT INTO beta VALUES ('b.jpg', 'out-b');
            "#,
    );

    let entries = load_mapping_entries(
        &path,
        &MappingReadOptions {
            format: Some(MappingFormat::Sqlite),
            sql_table: Some("beta".to_string()),
            ..Default::default()
        },
        &ColumnSelector::by_name("source"),
        &ColumnSelector::by_name("output"),
    )
    .unwrap();

    assert_eq!(
        entries,
        vec![MappingEntry {
            source_path: "b.jpg".to_string(),
            output_name: "out-b".to_string(),
        }]
    );
}

#[test]
fn mapping_format_display_names_are_human_readable() {
    assert_eq!(MappingFormat::Csv.display_name(), "CSV / Delimited");
    assert_eq!(MappingFormat::Excel.display_name(), "Excel");
    assert_eq!(MappingFormat::Parquet.display_name(), "Parquet");
    assert_eq!(MappingFormat::Sqlite.display_name(), "SQLite");
}

#[test]
fn column_selector_describe_formats_index_and_name() {
    assert_eq!(ColumnSelector::by_index(0).describe(), "column #0");
    assert_eq!(ColumnSelector::by_index(7).describe(), "column #7");
    assert_eq!(
        ColumnSelector::by_name("output_file").describe(),
        "column \"output_file\""
    );
}

#[test]
fn detect_format_covers_all_extensions() {
    for ext in &["xlsx", "xls", "xlsm", "ods"] {
        let path = PathBuf::from(format!("data.{ext}"));
        assert_eq!(detect_format(&path), MappingFormat::Excel, "ext={ext}");
    }
    for ext in &["parquet", "pq"] {
        let path = PathBuf::from(format!("data.{ext}"));
        assert_eq!(detect_format(&path), MappingFormat::Parquet, "ext={ext}");
    }
    for ext in &["db", "sqlite", "sqlite3"] {
        let path = PathBuf::from(format!("data.{ext}"));
        assert_eq!(detect_format(&path), MappingFormat::Sqlite, "ext={ext}");
    }
    // Unknown extension falls back to CSV
    assert_eq!(detect_format(Path::new("data.txt")), MappingFormat::Csv);
    assert_eq!(detect_format(Path::new("no_extension")), MappingFormat::Csv);
}

#[test]
fn csv_custom_delimiter_parses_pipe_separated_values() {
    let dir = tempdir().unwrap();
    let path = write_csv(
        dir.path(),
        "mapping.csv",
        "source|output\nimg1.jpg|out1\nimg2.jpg|out2\n",
    );

    let options = MappingReadOptions {
        delimiter: Some(b'|'),
        ..Default::default()
    };

    let preview = load_mapping_preview(&path, &options).unwrap();
    assert_eq!(preview.columns, vec!["source", "output"]);
    assert_eq!(preview.total_rows, 2);
    assert_eq!(preview.rows[0], vec!["img1.jpg", "out1"]);
}

#[test]
fn csv_not_truncated_when_all_rows_fit_in_preview() {
    let dir = tempdir().unwrap();
    let path = write_csv(
        dir.path(),
        "mapping.csv",
        "source,output\nimg1.jpg,out1\nimg2.jpg,out2\n",
    );

    let options = MappingReadOptions {
        preview_rows: 5,
        ..Default::default()
    };

    let preview = load_mapping_preview(&path, &options).unwrap();
    assert_eq!(preview.total_rows, 2);
    assert!(!preview.truncated);
    assert_eq!(preview.rows.len(), 2);
}

#[test]
fn csv_generates_column_names_for_empty_headers() {
    let dir = tempdir().unwrap();
    // Header row has an empty middle column
    let path = write_csv(dir.path(), "mapping.csv", "source,,output\na.jpg,,out-a\n");

    let preview = load_mapping_preview(&path, &MappingReadOptions::default()).unwrap();
    assert_eq!(preview.columns, vec!["source", "Column 2", "output"]);
}

#[test]
fn load_mapping_entries_errors_on_out_of_range_index() {
    let dir = tempdir().unwrap();
    let path = write_csv(dir.path(), "mapping.csv", "source,output\na.jpg,out-a\n");

    let err = load_mapping_entries(
        &path,
        &MappingReadOptions::default(),
        &ColumnSelector::by_index(5),
        &ColumnSelector::by_index(1),
    )
    .unwrap_err()
    .to_string();

    assert!(err.contains("out of range"), "unexpected error: {err}");
}

#[test]
fn load_mapping_entries_errors_on_unknown_column_name() {
    let dir = tempdir().unwrap();
    let path = write_csv(dir.path(), "mapping.csv", "source,output\na.jpg,out-a\n");

    let err = load_mapping_entries(
        &path,
        &MappingReadOptions::default(),
        &ColumnSelector::by_name("nonexistent"),
        &ColumnSelector::by_index(1),
    )
    .unwrap_err()
    .to_string();

    assert!(err.contains("nonexistent"), "unexpected error: {err}");
    assert!(err.contains("not found"), "unexpected error: {err}");
}

#[test]
fn inspect_mapping_sources_returns_empty_catalog_for_csv_and_parquet() {
    let dir = tempdir().unwrap();
    let csv_path = write_csv(dir.path(), "mapping.csv", "source,output\n");

    for format in [MappingFormat::Csv, MappingFormat::Parquet] {
        let catalog = inspect_mapping_sources(
            &csv_path,
            &MappingReadOptions {
                format: Some(format),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(
            catalog.sheets.is_empty(),
            "sheets should be empty for {format:?}"
        );
        assert!(
            catalog.sql_tables.is_empty(),
            "sql_tables should be empty for {format:?}"
        );
    }
}

#[test]
fn sqlite_valid_custom_sql_query_is_accepted() {
    let dir = tempdir().unwrap();
    let path = write_sqlite(
        dir.path(),
        "mapping.db",
        r#"
            CREATE TABLE photos (source TEXT, output TEXT);
            INSERT INTO photos VALUES ('a.jpg', 'out-a');
            INSERT INTO photos VALUES ('b.jpg', 'out-b');
            "#,
    );

    let entries = load_mapping_entries(
        &path,
        &MappingReadOptions {
            format: Some(MappingFormat::Sqlite),
            sql_query: Some("SELECT source, output FROM photos".to_string()),
            ..Default::default()
        },
        &ColumnSelector::by_name("source"),
        &ColumnSelector::by_name("output"),
    )
    .unwrap();

    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].source_path, "a.jpg");
    assert_eq!(entries[1].source_path, "b.jpg");
}

#[test]
fn validate_sql_query_allows_keyword_as_substring() {
    // "SELECTALL", "INSERTED" etc. should not trigger rejections
    let dir = tempdir().unwrap();
    let path = write_sqlite(
        dir.path(),
        "mapping.db",
        r#"
            CREATE TABLE my_updates (source TEXT, output TEXT);
            INSERT INTO my_updates VALUES ('a.jpg', 'out-a');
            "#,
    );

    // "my_updates" contains "update" as a substring — should be allowed
    let result = load_mapping_preview(
        &path,
        &MappingReadOptions {
            format: Some(MappingFormat::Sqlite),
            sql_query: Some("SELECT source, output FROM my_updates".to_string()),
            ..Default::default()
        },
    );
    assert!(
        result.is_ok(),
        "keyword-as-substring should be allowed: {result:?}"
    );
}

/// Build a minimal `.xlsx` holding `sheets` of `(name, rows)`.
///
/// Only the five parts calamine needs, with inline strings so there is no shared-string
/// table to maintain. An empty cell string is written as a genuinely absent cell, which is
/// what the reader's blank-row skipping keys off.
fn write_xlsx(path: &Path, sheets: &[(&str, &[&[&str]])]) {
    use std::io::Write as _;
    use zip::{ZipWriter, write::SimpleFileOptions};

    fn escape(text: &str) -> String {
        text.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
    }

    fn part(zw: &mut ZipWriter<fs::File>, name: &str, body: &str) {
        zw.start_file(name, SimpleFileOptions::default()).unwrap();
        zw.write_all(body.as_bytes()).unwrap();
    }

    let mut zw = ZipWriter::new(fs::File::create(path).expect("create workbook"));

    let overrides: String = (1..=sheets.len())
        .map(|n| {
            format!(
                r#"<Override PartName="/xl/worksheets/sheet{n}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>"#
            )
        })
        .collect();
    part(
        &mut zw,
        "[Content_Types].xml",
        &format!(
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
{overrides}</Types>"#
        ),
    );

    part(
        &mut zw,
        "_rels/.rels",
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"#,
    );

    let sheet_tags: String = sheets
        .iter()
        .enumerate()
        .map(|(i, (name, _))| {
            format!(
                r#"<sheet name="{}" sheetId="{n}" r:id="rId{n}"/>"#,
                escape(name),
                n = i + 1
            )
        })
        .collect();
    part(
        &mut zw,
        "xl/workbook.xml",
        &format!(
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets>{sheet_tags}</sheets></workbook>"#
        ),
    );

    let sheet_rels: String = (1..=sheets.len())
        .map(|n| {
            format!(
                r#"<Relationship Id="rId{n}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{n}.xml"/>"#
            )
        })
        .collect();
    part(
        &mut zw,
        "xl/_rels/workbook.xml.rels",
        &format!(
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{sheet_rels}</Relationships>"#
        ),
    );

    for (idx, (_, rows)) in sheets.iter().enumerate() {
        let body: String = rows
            .iter()
            .enumerate()
            .map(|(r, row)| {
                let cells: String = row
                    .iter()
                    .enumerate()
                    .filter(|(_, value)| !value.is_empty())
                    .map(|(c, value)| {
                        let col = (b'A' + u8::try_from(c).expect("column index")) as char;
                        format!(
                            r#"<c r="{col}{}" t="inlineStr"><is><t>{}</t></is></c>"#,
                            r + 1,
                            escape(value)
                        )
                    })
                    .collect();
                format!(r#"<row r="{}">{cells}</row>"#, r + 1)
            })
            .collect();
        part(
            &mut zw,
            &format!("xl/worksheets/sheet{}.xml", idx + 1),
            &format!(
                r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>{body}</sheetData></worksheet>"#
            ),
        );
    }

    zw.finish().expect("finish workbook");
}

#[test]
fn excel_preview_reads_headers_and_truncates_rows() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("mapping.xlsx");
    write_xlsx(
        &path,
        &[(
            "Sheet1",
            &[
                &["source", "output"],
                &["a.jpg", "out-a"],
                &["b.jpg", "out-b"],
                &["c.jpg", "out-c"],
            ],
        )],
    );

    let options = MappingReadOptions {
        preview_rows: 2,
        ..Default::default()
    };
    let preview = load_mapping_preview(&path, &options).unwrap();

    assert_eq!(preview.format, MappingFormat::Excel);
    assert_eq!(preview.columns, vec!["source", "output"]);
    assert_eq!(preview.total_rows, 3);
    assert!(preview.truncated);
    assert_eq!(
        preview.rows,
        vec![vec!["a.jpg", "out-a"], vec!["b.jpg", "out-b"]]
    );
}

#[test]
fn excel_without_headers_generates_default_column_names() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("mapping.xlsx");
    write_xlsx(
        &path,
        &[("Sheet1", &[&["a.jpg", "out-a"], &["b.jpg", "out-b"]])],
    );

    let options = MappingReadOptions {
        has_headers: Some(false),
        ..Default::default()
    };
    let preview = load_mapping_preview(&path, &options).unwrap();

    assert_eq!(preview.columns.len(), 2);
    assert_eq!(preview.total_rows, 2);
    assert_eq!(preview.rows[0], vec!["a.jpg", "out-a"]);
}

#[test]
fn excel_skips_fully_blank_rows_but_keeps_partial_ones() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("mapping.xlsx");
    write_xlsx(
        &path,
        &[(
            "Sheet1",
            &[
                &["source", "output"],
                &["a.jpg", "out-a"],
                &["", ""],
                &["b.jpg", ""],
            ],
        )],
    );

    let preview = load_mapping_preview(&path, &MappingReadOptions::default()).unwrap();

    // The all-empty row is dropped; the half-empty one is padded back to the column count.
    assert_eq!(preview.total_rows, 2);
    assert_eq!(
        preview.rows,
        vec![vec!["a.jpg", "out-a"], vec!["b.jpg", ""]]
    );
}

#[test]
fn excel_uses_explicit_sheet_name_over_the_first_sheet() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("mapping.xlsx");
    write_xlsx(
        &path,
        &[
            ("First", &[&["source", "output"], &["wrong.jpg", "no"]]),
            ("Second", &[&["source", "output"], &["right.jpg", "yes"]]),
        ],
    );

    let options = MappingReadOptions {
        sheet_name: Some("Second".to_string()),
        ..Default::default()
    };
    let preview = load_mapping_preview(&path, &options).unwrap();

    assert_eq!(preview.rows, vec![vec!["right.jpg", "yes"]]);

    // A blank or whitespace-only name falls back to the first sheet rather than erroring.
    let options = MappingReadOptions {
        sheet_name: Some("   ".to_string()),
        ..Default::default()
    };
    let preview = load_mapping_preview(&path, &options).unwrap();
    assert_eq!(preview.rows, vec![vec!["wrong.jpg", "no"]]);
}

#[test]
fn excel_entries_resolved_by_name() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("mapping.xlsx");
    write_xlsx(
        &path,
        &[(
            "Sheet1",
            &[
                &["source", "output"],
                &["a.jpg", "out-a"],
                &["b.jpg", "out-b"],
            ],
        )],
    );

    let entries = load_mapping_entries(
        &path,
        &MappingReadOptions::default(),
        &ColumnSelector::Name("source".to_string()),
        &ColumnSelector::Name("output".to_string()),
    )
    .unwrap();

    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].source_path, "a.jpg");
    assert_eq!(entries[1].output_name, "out-b");
}

#[test]
fn excel_reports_context_when_the_workbook_cannot_be_opened() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("missing.xlsx");

    let err = load_mapping_preview(&path, &MappingReadOptions::default())
        .expect_err("missing workbook should fail");

    assert!(
        format!("{err:#}").contains("failed to open workbook"),
        "unexpected error: {err:#}"
    );
}

fn write_parquet(path: &Path, rows: &[(&str, &str)]) {
    use ::parquet::{
        basic::{ConvertedType, Repetition, Type as PhysicalType},
        column::writer::ColumnWriter,
        data_type::ByteArray,
        file::{properties::WriterProperties, writer::SerializedFileWriter},
        schema::types::Type,
    };
    use std::sync::Arc;

    let source_col = Type::primitive_type_builder("source", PhysicalType::BYTE_ARRAY)
        .with_repetition(Repetition::REQUIRED)
        .with_converted_type(ConvertedType::UTF8)
        .build()
        .unwrap();
    let output_col = Type::primitive_type_builder("output", PhysicalType::BYTE_ARRAY)
        .with_repetition(Repetition::REQUIRED)
        .with_converted_type(ConvertedType::UTF8)
        .build()
        .unwrap();
    let schema = Arc::new(
        Type::group_type_builder("schema")
            .with_fields(vec![Arc::new(source_col), Arc::new(output_col)])
            .build()
            .unwrap(),
    );

    let props = Arc::new(WriterProperties::builder().build());
    let file = std::fs::File::create(path).unwrap();
    let mut writer = SerializedFileWriter::new(file, schema, props).unwrap();
    let mut rg = writer.next_row_group().unwrap();

    let sources: Vec<ByteArray> = rows.iter().map(|(s, _)| ByteArray::from(*s)).collect();
    let outputs: Vec<ByteArray> = rows.iter().map(|(_, o)| ByteArray::from(*o)).collect();

    {
        let mut col = rg.next_column().unwrap().unwrap();
        match col.untyped() {
            ColumnWriter::ByteArrayColumnWriter(typed) => {
                typed.write_batch(&sources, None, None).unwrap();
            }
            _ => panic!("expected byte array column for source"),
        }
        col.close().unwrap();
    }
    {
        let mut col = rg.next_column().unwrap().unwrap();
        match col.untyped() {
            ColumnWriter::ByteArrayColumnWriter(typed) => {
                typed.write_batch(&outputs, None, None).unwrap();
            }
            _ => panic!("expected byte array column for output"),
        }
        col.close().unwrap();
    }
    rg.close().unwrap();
    writer.close().unwrap();
}

#[test]
fn parquet_preview_reads_columns_and_rows() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("mapping.parquet");
    write_parquet(
        &path,
        &[
            ("a.jpg", "out-a"),
            ("b.jpg", "out-b"),
            ("c.jpg", "out-c"),
            ("d.jpg", "out-d"),
        ],
    );

    let options = MappingReadOptions {
        preview_rows: 2,
        ..Default::default()
    };
    let preview = load_mapping_preview(&path, &options).unwrap();

    assert_eq!(preview.format, MappingFormat::Parquet);
    assert_eq!(preview.columns, vec!["source", "output"]);
    assert_eq!(preview.total_rows, 4);
    assert!(preview.truncated);
    assert_eq!(preview.rows.len(), 2);
    assert_eq!(preview.rows[0], vec!["a.jpg", "out-a"]);
    assert_eq!(preview.rows[1], vec!["b.jpg", "out-b"]);
}

#[test]
fn parquet_entries_resolved_by_name() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("mapping.parquet");
    write_parquet(
        &path,
        &[("img1.jpg", "out1"), (" ", "out2"), ("img3.jpg", "out3")],
    );

    let entries = load_mapping_entries(
        &path,
        &MappingReadOptions::default(),
        &ColumnSelector::by_name("source"),
        &ColumnSelector::by_name("output"),
    )
    .unwrap();

    // blank source row is filtered out
    assert_eq!(
        entries,
        vec![
            MappingEntry {
                source_path: "img1.jpg".to_string(),
                output_name: "out1".to_string(),
            },
            MappingEntry {
                source_path: "img3.jpg".to_string(),
                output_name: "out3".to_string(),
            },
        ]
    );
}

#[test]
fn parquet_entries_resolved_by_index() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("mapping.parquet");
    write_parquet(&path, &[("img1.jpg", "out1"), ("img2.jpg", "out2")]);

    let entries = load_mapping_entries(
        &path,
        &MappingReadOptions::default(),
        &ColumnSelector::by_index(0),
        &ColumnSelector::by_index(1),
    )
    .unwrap();

    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].source_path, "img1.jpg");
    assert_eq!(entries[1].output_name, "out2");
}

#[test]
fn parquet_preview_not_truncated_when_rows_fit() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("mapping.parquet");
    write_parquet(&path, &[("a.jpg", "out-a"), ("b.jpg", "out-b")]);

    let preview = load_mapping_preview(&path, &MappingReadOptions::default()).unwrap();
    assert_eq!(preview.total_rows, 2);
    assert!(!preview.truncated);
}

#[test]
fn quote_identifier_escapes_embedded_double_quotes() {
    assert_eq!(quote_identifier("photos"), "\"photos\"");
    assert_eq!(quote_identifier(r#"my"table"#), r#""my""table""#);
    assert_eq!(quote_identifier(""), "\"\"");
}

#[test]
fn encode_bytes_empty_returns_empty_string() {
    assert_eq!(encode_bytes(&[] as &[u8]), "");
    assert_eq!(encode_bytes(b""), "");
}

#[test]
fn encode_bytes_nonempty_roundtrips_as_base64() {
    use base64::{Engine as _, engine::general_purpose};
    let payload = b"hello world";
    let encoded = encode_bytes(payload);
    assert!(!encoded.is_empty());
    let decoded = general_purpose::STANDARD.decode(&encoded).unwrap();
    assert_eq!(decoded, payload);
}

#[test]
fn format_excel_cell_handles_scalar_variants() {
    use calamine::Data as ExcelData;
    assert_eq!(format_excel_cell(&ExcelData::Empty), "");
    assert_eq!(
        format_excel_cell(&ExcelData::String("  hello  ".to_string())),
        "hello"
    );
    assert_eq!(format_excel_cell(&ExcelData::Float(2.5_f64)), "2.5");
    assert_eq!(format_excel_cell(&ExcelData::Float(5.0_f64)), "5");
    assert_eq!(format_excel_cell(&ExcelData::Int(42)), "42");
    assert_eq!(format_excel_cell(&ExcelData::Bool(true)), "true");
    assert_eq!(format_excel_cell(&ExcelData::Bool(false)), "false");
    assert_eq!(
        format_excel_cell(&ExcelData::DateTimeIso("2024-01-15".to_string())),
        "2024-01-15"
    );
    assert_eq!(
        format_excel_cell(&ExcelData::DurationIso("PT1H".to_string())),
        "PT1H"
    );
    // Error variants always produce an empty string
    assert_eq!(
        format_excel_cell(&ExcelData::Error(calamine::CellErrorType::Div0)),
        ""
    );
    assert_eq!(
        format_excel_cell(&ExcelData::Error(calamine::CellErrorType::NA)),
        ""
    );
}

#[test]
fn format_excel_header_uses_column_n_for_blank() {
    use calamine::Data as ExcelData;
    assert_eq!(format_excel_header(&ExcelData::Empty, 0), "Column 1");
    assert_eq!(
        format_excel_header(&ExcelData::String("  ".to_string()), 2),
        "Column 3"
    );
    assert_eq!(
        format_excel_header(&ExcelData::String("source".to_string()), 0),
        "source"
    );
}

#[test]
fn format_parquet_field_scalar_variants() {
    use ::parquet::record::Field;
    assert_eq!(format_parquet_field(&Field::Null), "");
    assert_eq!(format_parquet_field(&Field::Bool(true)), "true");
    assert_eq!(format_parquet_field(&Field::Bool(false)), "false");
    assert_eq!(format_parquet_field(&Field::Byte(42i8)), "42");
    assert_eq!(format_parquet_field(&Field::Short(1000i16)), "1000");
    assert_eq!(format_parquet_field(&Field::Int(99999i32)), "99999");
    assert_eq!(
        format_parquet_field(&Field::Long(123456789i64)),
        "123456789"
    );
    assert_eq!(format_parquet_field(&Field::UByte(255u8)), "255");
    assert_eq!(format_parquet_field(&Field::UShort(65535u16)), "65535");
    assert_eq!(
        format_parquet_field(&Field::UInt(4294967295u32)),
        "4294967295"
    );
    assert_eq!(
        format_parquet_field(&Field::ULong(u64::MAX)),
        u64::MAX.to_string()
    );
    assert_eq!(format_parquet_field(&Field::Float(1.5_f32)), "1.5");
    assert_eq!(format_parquet_field(&Field::Double(2.5_f64)), "2.5");
    assert_eq!(
        format_parquet_field(&Field::Str("  hello  ".to_string())),
        "hello"
    );
    // Bytes encodes as base64
    let bytes = ::parquet::data_type::ByteArray::from(vec![1u8, 2, 3]);
    let encoded = format_parquet_field(&Field::Bytes(bytes));
    assert!(!encoded.is_empty());
}

#[test]
fn validate_sql_remaining_forbidden_keywords_are_rejected() {
    let dir = tempdir().unwrap();
    let path = write_sqlite(dir.path(), "mapping.db", "CREATE TABLE t (a TEXT);");

    let cases = [
        ("SELECT a FROM t ALTER", "ALTER"),
        ("SELECT a FROM t CREATE x", "CREATE"),
        ("SELECT a FROM t REPLACE x", "REPLACE"),
        ("SELECT a FROM t ATTACH x", "ATTACH"),
        ("SELECT a FROM t DETACH x", "DETACH"),
        ("SELECT a FROM t PRAGMA x", "PRAGMA"),
        ("SELECT a FROM t REINDEX", "REINDEX"),
        ("SELECT a FROM t VACUUM", "VACUUM"),
        ("SELECT a FROM t INSERT x", "INSERT"),
    ];

    for (query, keyword) in cases {
        let err = load_mapping_preview(
            &path,
            &MappingReadOptions {
                format: Some(MappingFormat::Sqlite),
                sql_query: Some(query.to_string()),
                ..Default::default()
            },
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains(keyword),
            "expected rejection for keyword {keyword} in query '{query}', got: {err}"
        );
    }
}

#[test]
fn sqlite_blob_column_is_encoded_as_base64() {
    let dir = tempdir().unwrap();
    let path = write_sqlite(
        dir.path(),
        "mapping.db",
        r#"
            CREATE TABLE files (source TEXT, data BLOB);
            INSERT INTO files VALUES ('img.jpg', x'deadbeef');
            "#,
    );

    let preview = load_mapping_preview(
        &path,
        &MappingReadOptions {
            format: Some(MappingFormat::Sqlite),
            sql_query: Some("SELECT source, data FROM files".to_string()),
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(preview.rows[0][0], "img.jpg");
    // blob must be non-empty base64
    let blob_value = &preview.rows[0][1];
    assert!(!blob_value.is_empty(), "blob should be encoded as base64");
    use base64::{Engine as _, engine::general_purpose};
    let decoded = general_purpose::STANDARD.decode(blob_value).unwrap();
    assert_eq!(decoded, vec![0xde, 0xad, 0xbe, 0xef]);
}

#[test]
fn sqlite_mapping_rejects_non_select_queries_and_semicolons() {
    let dir = tempdir().unwrap();
    let path = write_sqlite(
        dir.path(),
        "mapping.db",
        r#"
            CREATE TABLE photos (source TEXT, output TEXT);
            INSERT INTO photos VALUES ('a.jpg', 'out-a');
            "#,
    );

    let bad_cases = [
        ("DELETE FROM photos", "must begin with SELECT"),
        ("UPDATE photos SET output = 'x'", "must begin with SELECT"),
        (
            "SELECT * FROM photos; DROP TABLE photos",
            "must not contain semicolons",
        ),
        (
            "SELECT * FROM photos DROP",
            "must not contain the DROP keyword",
        ),
    ];

    for (query, expected) in bad_cases {
        let err = load_mapping_preview(
            &path,
            &MappingReadOptions {
                format: Some(MappingFormat::Sqlite),
                sql_query: Some(query.to_string()),
                ..Default::default()
            },
        )
        .unwrap_err()
        .to_string();

        assert!(
            err.contains(expected),
            "unexpected error for {query}: {err}"
        );
    }
}

/// Run a query through the sqlite reader and report whether it was accepted.
fn query_is_accepted(path: &Path, query: &str) -> Result<(), String> {
    load_mapping_preview(
        path,
        &MappingReadOptions {
            format: Some(MappingFormat::Sqlite),
            sql_query: Some(query.to_string()),
            ..Default::default()
        },
    )
    .map(|_| ())
    .map_err(|e| e.to_string())
}

#[test]
fn sqlite_preview_truncates_rows_but_counts_them_all() {
    // The CSV reader has this covered; the SQLite reader has its own copy of
    // the row-limit and total-count logic and had neither pinned.
    let dir = tempdir().unwrap();
    let path = write_sqlite(
        dir.path(),
        "rows.db",
        r#"
            CREATE TABLE photos (source TEXT, output TEXT);
            INSERT INTO photos VALUES ('a.jpg', 'out-a');
            INSERT INTO photos VALUES ('b.jpg', 'out-b');
            INSERT INTO photos VALUES ('c.jpg', 'out-c');
            "#,
    );

    let preview = load_mapping_preview(
        &path,
        &MappingReadOptions {
            format: Some(MappingFormat::Sqlite),
            preview_rows: 2,
            ..Default::default()
        },
    )
    .unwrap();

    // The limit is exclusive: exactly two rows are kept, not three.
    assert_eq!(preview.rows.len(), 2);
    assert_eq!(preview.rows[0], vec!["a.jpg", "out-a"]);
    assert_eq!(preview.rows[1], vec!["b.jpg", "out-b"]);
    // But every row is still counted, including the ones dropped.
    assert_eq!(preview.total_rows, 3);
    assert!(preview.truncated);
}

#[test]
fn validate_sql_query_matches_forbidden_keywords_on_word_boundaries() {
    // The guard scans for DDL/DML keywords but must only reject them as whole
    // words, otherwise ordinary table names get caught. The existing coverage
    // checks one substring case; these pin both sides of the boundary test
    // and the scan-advance that lets a later match still be found.
    let dir = tempdir().unwrap();
    let path = write_sqlite(
        dir.path(),
        "boundaries.db",
        r#"
            CREATE TABLE undrop (source TEXT, output TEXT);
            CREATE TABLE droplet (source TEXT, output TEXT);
            INSERT INTO undrop VALUES ('a.jpg', 'out-a');
            INSERT INTO droplet VALUES ('b.jpg', 'out-b');
            "#,
    );

    // Preceded by an alphanumeric: "unDROP" is not the DROP keyword.
    assert!(
        query_is_accepted(&path, "SELECT source, output FROM undrop").is_ok(),
        "a keyword at the end of a longer word must be allowed"
    );
    // Followed by an alphanumeric: "DROPlet" is not the DROP keyword either.
    assert!(
        query_is_accepted(&path, "SELECT source, output FROM droplet").is_ok(),
        "a keyword at the start of a longer word must be allowed"
    );

    // A non-matching occurrence must not stop the scan: the first DROP here
    // is inside "undrop", the second is a real keyword at the very end.
    let err = query_is_accepted(
        &path,
        "SELECT source FROM undrop WHERE source = source OR DROP",
    )
    .expect_err("the trailing DROP is a whole word");
    assert!(err.contains("DROP keyword"), "unexpected error: {err}");
}

#[test]
fn validate_sql_query_is_case_insensitive_and_trims() {
    let dir = tempdir().unwrap();
    let path = write_sqlite(
        dir.path(),
        "case.db",
        r#"
            CREATE TABLE photos (source TEXT, output TEXT);
            INSERT INTO photos VALUES ('a.jpg', 'out-a');
            "#,
    );

    // Leading whitespace and lower case still count as beginning with SELECT.
    assert!(query_is_accepted(&path, "   select source, output from photos").is_ok());

    // And a lower-case DDL keyword is still caught.
    let err = query_is_accepted(
        &path,
        "select source from photos where source = source or drop",
    )
    .expect_err("lower-case keywords are normalised before matching");
    assert!(err.contains("DROP keyword"), "unexpected error: {err}");
}

#[test]
fn validate_sql_query_rejects_each_forbidden_keyword() {
    let dir = tempdir().unwrap();
    let path = write_sqlite(
        dir.path(),
        "forbidden.db",
        "CREATE TABLE photos (source TEXT, output TEXT);",
    );

    for keyword in [
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "REPLACE", "ATTACH", "DETACH",
        "PRAGMA", "REINDEX", "VACUUM",
    ] {
        let query = format!("SELECT source FROM photos WHERE source = source OR {keyword}");
        let err = match query_is_accepted(&path, &query) {
            Ok(()) => panic!("{keyword} should have been rejected"),
            Err(e) => e,
        };
        assert!(
            err.contains(&format!("{keyword} keyword")),
            "wrong error for {keyword}: {err}"
        );
    }
}
