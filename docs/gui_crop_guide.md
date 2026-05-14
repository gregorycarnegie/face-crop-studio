# GUI Crop Features Guide

The desktop app bundles a full crop workflow on top of YuNet detection so you can preview, fine‑tune, and export professional headshots without juggling multiple tools. This guide walks through the key surfaces available in the left-hand control panel and the associated keyboard shortcuts.

## Quick Start

1. Launch the GUI with `cargo run -p fcs-gui` (or run the packaged binary once released).
2. Click **Open image…** to select a portrait photo, or simply drag an image from your desktop (or paste one from the clipboard) to load it instantly. YuNet will run automatically as soon as the model has been configured.
3. Detected faces appear in the **Detected Faces** list. Click a thumbnail or the accompanying **Select** button to include a face in the export set. Hold *Shift* and use the number keys (1‑6) to jump between the built-in crop presets.
4. Hit **Export selected faces** (or press *Enter*) to run the crop, optional enhancements, and save pipeline.

## Crop Controls

- **Presets** – Pick from LinkedIn, Passport, Instagram, ID Card, Avatar, or Headshot. Selecting “Custom” enables the width/height inputs for arbitrary sizes.
- **Face height %** – Target proportion of the output height that should be filled by the detected face. Use the slider or press `+`/`–` for one per cent adjustments (hold *Shift* for ±5).
- **Positioning mode** – Choose between Center, Rule of Thirds, or Custom. When Custom is active the horizontal/vertical sliders appear; arrow keys nudge by 0.05 (hold *Shift* for 0.1). Values are clamped to ±1.0 and updates are undoable.
- **Undo / Redo** – `Ctrl+Z` to step back through crop changes, `Ctrl+Y` or `Ctrl+Shift+Z` to redo. The application keeps a bounded history of the last 100 adjustments.
- **Overlay toggle** – Enable the overlay checkbox in the preview to visualise the calculated crop region. The overlay auto-updates as you tweak settings.
- **Fill colour padding** – If the crop extends past the source image, the empty pixels are filled with the configured colour. Use the colour picker, hex input, or RGB/HSV spinners to override the default black padding.

## Face Selection & Quality

- Every detection shows the YuNet confidence, Laplacian variance score, and quality bucket (Low/Medium/High). These scores drive automation such as auto-selecting the best face or skipping blurry exports.
- Use **Select All / Deselect All** to batch toggle the current detections.
- Quality automation rules live under *Crop ▸ Quality Rules*. You can require a minimum quality level, append `_highq/_medq/_lowq` suffixes, or skip exports if no face reaches High quality.

## Enhancement Pipeline

When **Enable enhancements** is ticked, the crop output is routed through the pure-Rust enhancement pipeline from `fcs-utils`:

- Presets (Natural/Vivid/Professional) reconfigure the sliders to sensible defaults. Manual adjustments are always allowed—preset selection simply provides a starting point.
- Controls cover exposure, brightness, contrast, saturation, sharpness, skin smoothing, red-eye removal, and portrait-style background blur.
- Press **Reset to defaults** to return to the preset/slider defaults without losing the current crop configuration.
- Keyboard shortcut: `Space` toggles the enhancement preview, letting you compare raw vs enhanced output.

## Metadata & Export

- Configure metadata mode (Preserve, Strip, or Custom) and author custom tags in `key=value` format. Tags are written for PNG/JPEG/WebP exports when the metadata mode allows it.
- Exports respect the auto-detected format, or the explicit format picker if auto-detect is disabled.
- Filename suffixes derived from quality (`_highq`, etc.) are appended when the option is enabled in Quality Rules.

## Batch Mode

- Use **Load multiple…** to enqueue a directory of images. The batch table tracks progress (Pending/Processing/Completed/Failed) and displays per-image stats.
- Drag-and-drop a folder from your file explorer (or paste a folder path from the clipboard) to append its supported images to the batch queue without opening the dialogue.
- Drop a CSV/XLSX/Parquet/SQLite mapping file (or paste its path) to autopopulate the Mapping Import panel so you can combine spreadsheet data with batch exports immediately.
- Start the batch export from the footer once rules are configured. Progress updates stream live in the status bar.

By combining these surfaces you can rapidly iterate on crop framing, dial in enhancement presets, and export the most flattering results without leaving the app. For deeper technical details see `ARCHITECTURE.md` and the inline documentation in `fcs-core/src/cropper.rs`.
