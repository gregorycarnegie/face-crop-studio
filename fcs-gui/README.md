# fcs-gui

The Face Crop Studio desktop application, built with egui. The binary is named `face-crop-studio`.

Built on `fcs-core`, `fcs-utils` and `fcs-mapping`; includes batch processing with per-file panic safety, interactive crop editing, and installer packaging metadata (e.g. `cargo-deb`).

## Run

```sh
cargo run --release -p fcs-gui
```

## Testing

```sh
cargo test -p fcs-gui
```