# CLI Recipes

The `fcs-cli` crate exposes a flexible command-line tool for running YuNet detection, cropping, quality filtering, and enhancement workflows. This document captures a handful of common invocations you can adapt to your projects.

## Basic Detection

```bash
cargo run -p fcs-cli -- --input fixtures/images/006.jpg --model models/face_detection_yunet_2023mar_640.onnx
```

Prints a summary of detections to stdout. Add `--json detections.json` to capture structured output for downstream tooling.

## Crop a Single Portrait

```bash
cargo run -p fcs-cli -- \
  --input portraits/alex.png \
  --model models/face_detection_yunet_2023mar_640.onnx \
  --crop \
  --preset headshot \
  --output-dir crops/
```

The `--crop` flag enables the crop pipeline, respecting the selected preset. Results are saved to the `crops/` directory.

## Enforce Minimum Quality

```bash
cargo run -p fcs-cli -- \
  --input portraits/*.jpg \
  --crop \
  --output-dir crops/ \
  --min-quality high \
  --quality-suffix true
```

Only exports faces classified as `High` quality and appends `_highq` to filenames. Faces below the threshold are reported and skipped.

## Apply Enhancements

```bash
cargo run -p fcs-cli -- \
  --input portraits/group_photo.jpg \
  --crop \
  --output-dir crops/ \
  --enhance true \
  --enhancement-preset vivid \
  --enhance-saturation 1.2 \
  --enhance-brightness 12
```

Starts with the `vivid` preset and overrides the saturation and brightness sliders for the current invocation.

## Pad Crops With Custom Colour

```bash
cargo run -p fcs-cli -- \
  --input portraits/outdoor.png \
  --crop \
  --output-dir crops/ \
  --crop-fill-color "hsv(210, 65%, 35%)"
```

`--crop-fill-color` accepts `#RRGGBB`/`#RRGGBBAA`, `rgb(r,g,b)`, `rgba(r,g,b,a)`, or `hsv(h,s,v)` tokens. Any portion of the crop that extends beyond the source image is padded with the chosen colour (defaults to solid black).

## Batch Pipeline with Metadata

```bash
cargo run -p fcs-cli -- \
  --input portraits/ \
  --model models/face_detection_yunet_2023mar_640.onnx \
  --crop \
  --output-dir exports/ \
  --metadata-mode custom \
  --metadata-include-crop true \
  --metadata-tag photographer=Alice \
  --metadata-tag campaign="Holiday 2025"
```

Processes every supported image in the directory tree, exporting crops with rich metadata embedded.

## Export Selected Face Index

```bash
cargo run -p fcs-cli -- \
  --input composites/family.png \
  --crop \
  --output-dir solo/ \
  --face-index 2
```

Saves only the second detected face (1-based indexing) from the source image—useful when you want a consistent subject from group photos.

Refer to `cargo run -p fcs-cli -- --help` for a full list of flags, and combine them with the recipes above to build repeatable pipelines.
