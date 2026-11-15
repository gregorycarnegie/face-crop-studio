# fcs-cli

Command-line interface for Face Crop Studio: batch face detection and cropping.

Supports directory/file inputs, mapping files, watch mode, webcam input, annotation, enhancement, and quality filtering. Built on `fcs-core`, `fcs-utils` and `fcs-mapping`.

## Usage

```sh
cargo run --release -p fcs-cli -- --help
```

## Testing

```sh
cargo test -p fcs-cli
```