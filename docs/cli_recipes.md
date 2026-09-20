# CLI Recipes

The `fcs-cli` crate exposes a flexible command-line tool for running face detection, cropping, quality filtering, and enhancement workflows. This document captures a handful of common invocations you can adapt to your projects.

## Basic Detection

```bash
cargo run -p fcs-cli -- --input fixtures/images/006.jpg --model models/scrfd80k_500m_640.onnx
```

Prints a summary of detections to stdout. Add `--json detections.json` to capture structured output for downstream tooling.

## Crop a Single Portrait

```bash
cargo run -p fcs-cli -- \
  --input portraits/alex.png \
  --model models/scrfd80k_500m_640.onnx \
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

## Level the Eyes

```bash
cargo run -p fcs-cli -- \
  --input portraits/ \
  --crop \
  --output-dir crops/ \
  --eye-line-align
```

`--eye-line-align` rotates each crop so the subject's eyes sit horizontal, which is what makes a set of portraits look consistent rather than subtly tilted. There is no negative form: the setting is off unless asked for.

The rotation is only as good as the eye points behind it, so the crop uses `models/eye_refiner.onnx` in preference to the detector's own landmarks — 1.18° median eye-line error against 4.05°, measured on 1,382 hand-clicked faces (`tools/dataset/CURVE_RESULTS.md`). The refiner needs ONNX Runtime, which every packaged release bundles. A build without it logs one line and falls back to the detector's landmarks, so crops are still levelled, just less precisely.

## Batch Pipeline with Metadata

```bash
cargo run -p fcs-cli -- \
  --input portraits/ \
  --model models/scrfd80k_500m_640.onnx \
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
