# Desktop App Guide

Face Crop Studio's desktop app finds the faces in your photos, lets you check and adjust them, and exports one cropped portrait per face. Detection runs on your computer; nothing is uploaded.

The window has three parts:

- **Sidebar** (left): the photo queue, the spreadsheet mapping and the activity log.
- **Canvas** (centre): the current photo with a box around each face.
- **Inspector** (right): previews of the selected faces, the settings in four tabs (**Crop**, **Shape**, **Enhance** and **Output**) and the export button.

## Quick Start

1. Launch the app from its installer or package. From a checkout, run `./fcs-qt/dev.ps1 run` on Windows or `./fcs-qt/dev.sh run` on macOS and Linux.
2. Add photos. Click **Add photos** (`Ctrl+O`), drop files or a whole folder on the window, or paste an image (`Ctrl+V`). The app reads JPEG, PNG, WebP, TIFF, HEIC and camera RAW files.
3. The app finds the faces in each photo as it arrives, and every face starts out selected. Click a face's number chip to leave that face out of the export.
4. Pick a size on the **Crop** tab, then click **Export N faces** (`Ctrl+E`) and choose a folder.

## Photo Queue

- The **Photos** tab lists every photo with its face count and how many faces are selected (for example `2/3`). While photos are still being processed, the line above the list reads *Finding faces · N to go*.
- Click a photo to open it. With the list focused, `↑` and `↓` move between photos and `Delete` removes one from the queue. Your files on disk are never touched.
- **Clear** empties the queue after you confirm. The **⋯** menu next to it saves the queue as a text file, or saves the report of the last export as CSV or JSON.
- The **Activity** tab logs what the app has done, including errors.

## Faces on the Canvas

- Each face box shows its number. Hover over a box to see its sharpness (low, medium or high) and the detector's confidence.
- Click a face's number chip to include it in the export or leave it out. With a box focused, `Space` does the same. Below the canvas, **All** and **None** select or clear every face in the photo, and `Ctrl+A` selects every face in every photo.
- Drag a box to move it, and drag its handles to resize it. The box's toolbar can put it back where the detector found it, or remove it (`Delete` or `Backspace`).
- **Draw a face** (`D`) adds a face the detector missed: drag over the face, or press `Esc` to cancel. `Esc` also deselects the active box.
- Undo (`Ctrl+Z`) and redo (`Ctrl+Y` or `Ctrl+Shift+Z`) step through the current photo's box edits and face selection. Setting changes are not part of this history.
- **Zoom** with the mouse wheel, `Ctrl++` and `Ctrl+-`; `Ctrl+0` fits the photo to the window.
- **Rotate left**, **Rotate right** and **Straighten** turn the photo, then find its faces again. Straighten waits until you let go of its slider. A re-run replaces any hand edits on that photo.

## Crop Tab

- **Output size**: pick a preset (LinkedIn, Passport, Instagram, ID Card, Avatar or Headshot), an aspect ratio (1:1, 4:5 or 3:4), or type an exact width and height. A size that matches no preset shows as *Custom*.
- **Framing**:
  - **Face size** sets how much of the crop's height the face fills.
  - **Position** places the face in the centre, on the rule of thirds, or at your own **Offset X** and **Offset Y**.
  - **Eye-line align** rotates each crop so the eyes are level.
- **Padding**: the **Fill colour** paints the area where a crop runs past the edge of the photo.
- **View and import**:
  - **Show crop overlay** outlines each face's crop on the photo.
  - **Auto-orient via EXIF** turns photos upright using the camera's orientation. It applies to photos you add after turning it on.

## Shape Tab

- Pick a shape for the crops, such as a rectangle, circle or polygon. **Shape settings** shows only the controls that apply to the chosen shape: corner radius, chamfer size, number of sides or points, inner radius, tension, rotation and iterations.
- **Vignette** darkens the crop's edges toward the **Vignette colour**; set how far it reaches with **Softness** and how strong it is with **Intensity**.

## Enhance Tab

- **Automatic**:
  - **Auto colour** balances white and levels before the adjustments below.
  - **Red-eye removal** removes red eyes caused by camera flash.
  - **Background blur** softens everything around the face.
- **Adjust**: exposure, brightness, contrast, saturation, sharpness and skin smoothing. A slider at its neutral value leaves the crop unchanged.
- **Restore defaults** resets this tab only.

Every crop is enhanced first, then scored for quality, then shaped and filled. This is the same pipeline the CLI uses: it runs on the GPU when one is available, and the identical CPU code otherwise.

## Preview

The **Preview** card at the top of the inspector shows the selected faces of the current photo exactly as they will be saved: cropped, enhanced and shaped, over a checkerboard where they are transparent. The card updates as you change settings or edit a box.

## Output Tab

- **Format**: PNG, JPEG, WebP or AVIF. When JPEG is chosen, a **JPEG quality** slider appears.
- **Metadata**:
  - **Preserve** keeps the photo's EXIF.
  - **Custom** writes only the details you choose.
  - **Strip** writes no metadata at all.
  - With Preserve or Custom, you can also write the crop settings and the quality metrics into each crop.
- **Quality rules**:
  - **Only the best face per photo** exports only the sharpest selected face of each photo.
  - **Skip photos without a sharp face** leaves out every face of a photo whose best crop is not high quality.
  - **Add quality to file names** adds `_highq`, `_medq` or `_lowq` to each crop's name.
  - **Minimum quality** sets the lowest quality that gets exported.
- **Detection**: faces scored below the **Confidence floor** are ignored.
- **Processing**: choose the backend that runs detection and enhancement:
  - **GPU (WGSL)** is the usual choice.
  - **CUDA** is experimental, needs a supported NVIDIA GPU, and asks you to confirm before turning on.
  - **CPU** runs with no GPU.
  - **Respect environment overrides** lets variables such as `WGPU_POWER_PREF` choose the graphics adapter.
- **Telemetry**: write stage timings to the log, and set the log level.
- **Report**: save a CSV or JSON report next to the crops, named `face-crop-report`, with one line per photo.

Changing the confidence floor or the backend finds the faces in every photo again, and that discards hand edits.

## Export

- **Export N faces** (`Ctrl+E`) saves every selected face from every photo in the queue to the folder you choose. Its label counts the faces and the line above it counts the photos. While the export runs, the button shows its progress.
- Crops are named after their photo and face number, for example `photo_face_01.jpg`.
- If some crops would replace files already in the folder, the app asks first. **Keep both** saves the new crops with a number added (`photo_face_01(2).jpg`). **Replace** overwrites the old files. **Cancel** stops the export.

## Naming Crops from a Spreadsheet

The **Mapping** tab names crops from a table. Choose a CSV, Excel, Parquet or SQLite file, or drop it on the window. Then pick the sheet or table, the column of photo file names, and the column of output names. The tab shows how many queued photos match. A matched photo's crops are saved under its name from the table, and the queue shows that name next to the photo (*· as name*).

## Camera

**Camera** in the header opens your webcam. **Find faces live** detects faces in every frame. **Take snapshot** (`Space`) adds the current frame to the queue as a photo, and **Done** (`Esc`) closes the camera.

## Settings and About

The **⋯** menu in the header has **Restore default settings…**, which resets every setting to how the app ships (your photos stay in the queue), and **About Face Crop Studio**, which shows the version and the Qt licence notice. The app saves its settings when you change them and restores them at the next launch.

## Keyboard Shortcuts

| Keys | Action |
| --- | --- |
| `Ctrl+O` | Add photos |
| `Ctrl+V` | Paste an image |
| `Ctrl+E` | Export the selected faces |
| `Ctrl+A` | Select every face in every photo |
| `Ctrl+Z` / `Ctrl+Y`, `Ctrl+Shift+Z` | Undo / redo box edits in the current photo |
| `D` | Draw a face the detector missed |
| `Space` | Include or skip the focused face; take a snapshot in camera view |
| `Delete`, `Backspace` | Remove the active face; remove the focused photo from the queue |
| `Esc` | Cancel drawing, deselect the face, or close the camera |
| `Ctrl++`, `Ctrl+-`, `Ctrl+0` | Zoom in, zoom out, fit to window |
| `↑` / `↓` | Previous / next photo (photo list focused) |

On macOS, use `Cmd` in place of `Ctrl`.

For the technical side, see [ARCHITECTURE.md](../ARCHITECTURE.md) and [fcs-qt/README.md](../fcs-qt/README.md).
