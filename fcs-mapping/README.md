# fcs-mapping

Tabular mapping ingestion for batch crop jobs.

Reads CSV, Excel, Parquet, and SQLite tables into `MappingEntry` pairs (source image path, desired output name). Also provides previews (`MappingPreview`) and catalogs (`MappingCatalog`) so the GUI can let users choose columns via `ColumnSelector` and `MappingReadOptions`.

## Testing

```sh
cargo test -p fcs-mapping
```