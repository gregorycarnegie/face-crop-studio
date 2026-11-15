# fcs-utils

Shared utilities for the Face Crop Studio workspace.

Includes image loading and helpers, color handling, enhancement, output path/format handling, atomic file writes, geometry primitives (points, shapes), config, webcam support, quality metrics, telemetry, and process environment helpers.

This is an internal crate used by `fcs-core`, `fcs-cli` and `fcs-gui`.

## Testing

```sh
cargo test -p fcs-utils
```