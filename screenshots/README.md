# Screenshot & GIF Asset Checklist

Use this folder for release screenshots/GIFs referenced by the README and GitHub
release notes. The desktop shots are of the Qt app (`fcs-qt`); retake them when
its layout changes.

## Static screenshots (`v3.0.0`)

- `gui-main.png` - The whole window: a queue of several photos, the current
  photo with its face box and crop outline, the Crop tab, the preview and the
  export button. Backend on GPU (WGSL), not CUDA.
- `gui-crop-config.png` - The inspector column on the Crop tab: size presets,
  aspect ratio, width and height, and the preview.
- `gui-enhancement.png` - The inspector column on the Enhance tab, with one
  setting visibly in use (e.g. background blur) so the preview shows it.
- `gui-batch.png` - The sidebar's Photos tab with several photos and their
  face counts.
- `cli-example.png` - One `fcs-cli --input … --crop --output-dir …` run on a
  single portrait: the log lines, the detection JSON and the summary.

The three panel shots are tall crops of the window and sit side by side in the
README at 240 px wide.

## Workflow GIFs (to record)

Short looping clips of the main workflows. The README has a ready-to-uncomment
"Workflow recordings" block; drop these files in and uncomment it.

| File | Shows | Suggested length |
|------|-------|------------------|
| `workflow-detect-crop-export.gif` | Add a photo → faces found → adjust a box or the framing → export. The headline flow. | 8–12 s |
| `workflow-batch.gif` | Drop a folder → queue fills as faces are found → export every selected face → the done toast. | 8–12 s |
| `workflow-enhance.gif` | Move enhancement sliders (exposure/contrast/smoothing) with the preview updating live. | 6–10 s |
| `workflow-mapping.gif` | Drop a CSV/XLSX → Mapping tab → pick the photo and name columns → matched count and first rows. | 8–12 s |

## Capture guidance

- Prefer 16:9 at readable scale (e.g. 1600x900+); for GIFs, 1280x720 keeps file
  size reasonable. The app opens at 1440x900 logical pixels.
- Use real, non-sensitive sample images: `fixtures/images/` holds licence-clean
  Open Images photos (CC BY 2.0).
- Avoid including personal file paths/usernames in frame.
- Keep the UI in a realistic state (models loaded, meaningful detections).
- Keep each GIF under ~5 MB so it loads quickly on GitHub. Trim to a single
  clean pass of the workflow and let it loop.
- The app saves every setting change to `config/gui_settings.json`. Back it up
  before switching the backend or toggling enhancements for a shot, and restore
  it afterwards.

### Scripted capture (Windows)

The current shots were taken this way:

- Pass photos on the command line to queue them at start:
  `fcs-qt.exe face.png fixtures\images\*.jpg`. Run it from the repo root so it
  finds `models/`, with Qt's `bin` on `PATH`.
- Drive the app through UI Automation. Find controls by name ("Enhance",
  "Background blur", "GPU (WGSL)") and use their Invoke pattern. Don't click at
  their bounding rectangle: a control scrolled out of the inspector still reports
  one, and the click lands on whatever covers that spot.
- Capture with `PrintWindow(hwnd, dc, 2)`, then crop to
  `DWMWA_EXTENDED_FRAME_BOUNDS` to drop the invisible resize border.

### CLI screenshot

The CLI logs canonical absolute paths, so run it from a neutral folder (the
current shot used `C:\fcs-demo`, holding `fcs-cli.exe`, `models/` and the
photo) to keep usernames out of frame. With no `config/gui_settings.json`
there, it runs on built-in defaults. Set `RUST_LOG=warn,fcs_cli=info` to drop
the wgpu adapter lines. The current shot is the real output, rendered in a
monospace font on a terminal background.

### Recording tools

- **Windows**: [ScreenToGif](https://www.screentogif.com/) (record a window
  region, edit frames, export GIF directly).
- **Cross-platform**: record MP4 (OBS, or the OS recorder) then convert:

  ```bash
  # MP4 → optimized looping GIF via ffmpeg + a shared palette
  ffmpeg -i workflow.mp4 -vf "fps=15,scale=1280:-1:flags=lanczos,palettegen" palette.png
  ffmpeg -i workflow.mp4 -i palette.png -vf "fps=15,scale=1280:-1:flags=lanczos,paletteuse" workflow-detect-crop-export.gif
  ```
