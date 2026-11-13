# fcs-qt: a Qt Quick front end (experiment)

Face Crop Studio's core workflow on Qt Quick/QML through
[Qt Bridge for Rust](https://doc.qt.io/qtbridge-rust/qtbridge/index.html) (`qtbridge` 0.4.1,
a Technology Preview). It sits beside the egui app (`fcs-gui`) and uses the same models,
pipeline and settings file.

![fcs-qt with the sample photos loaded](screenshot.png)

## What it does

- Add photos with the file dialog, by dropping files or folders on the window, or from the
  command line (`fcs-qt photo.jpg folder/`).
- Detection runs in the background, one photo at a time, on the same GPU/CPU detector and eye
  refiner as fcs-gui. The queue updates row by row.
- The photo shows its faces as boxes. The number chip (or Space) includes or skips a face; All
  / None act on the current photo.
- Face editing: click a box to make it active, then drag it or its handles to move and
  resize it, reset it to the detector's box, or delete it (Delete). The draw tool (D) adds a
  face the detector missed. Undo and redo (Ctrl+Z, Ctrl+Y) cover every edit of the photo.
- The canvas zooms with the mouse wheel or Ctrl +/-/0 and pans by dragging. Photos rotate in
  quarter turns or straighten by up to 45°; either finds the faces again on the turned photo,
  and previews and export use it too.
- The inspector has four tabs, as fcs-gui's settings do:
  - **Crop**: presets or a custom width and height, face size, position (center, thirds, or
    custom X/Y offsets), eye-line align, padding colour, crop overlay, EXIF orientation.
  - **Shape**: all eleven crop shapes, drawn from their real outlines, with their parameters,
    plus vignette.
  - **Enhance**: auto colour, red-eye removal, background blur, exposure, brightness,
    contrast, saturation, sharpness, skin smoothing.
  - **Output**: PNG, JPEG (with quality), WebP or AVIF.
- Each face's crop is outlined on the photo in the chosen shape. Previews are rendered
  through the real export path, so they show exactly what export writes.
- Export writes the selected faces of every photo to a folder, as `{stem}_face_NN.{ext}`. If
  any of those names already exist there, it asks first: keep both (the new crop gets a
  `(2)` suffix) or replace.
- Output options, as in fcs-gui and the CLI:
  - Quality rules: best face only, skip photos without a sharp face, a minimum quality, and
    quality suffixes in names. They judge the finished crop, as the CLI does.
  - The detection confidence floor; changing it finds the faces in every photo again.
  - An optional CSV or JSON report beside the crops, in fcs-gui's columns.
- Mapping files: a CSV, Excel, Parquet or SQLite table (opened from the Mapping tab, or
  dropped on the window) names each photo's crops. Its photo and name columns are guessed
  from the headers and can be changed. Rows match photos by file name, then by stem, exact
  case first, as in fcs-gui, and the tab shows how many matched.
- Photos can be removed from the queue one at a time, or all at once with a confirmation.
  An Activity tab logs what happened: detections, failures, exports, reports.
- Settings go to the same file as fcs-gui (`AppSettings::default_settings_path`).

Not ported yet: webcam, watch folders.

Settings reach QML as one JSON object (`Studio.settings`, the serialized `AppSettings`), and
every control writes through one slot, `Studio.update(patch)`, which merges the patch, checks
it still deserializes, saves and re-renders. A new setting needs no Rust changes.

## Build and run

You need Qt 6.10 or newer (MSVC 64-bit kit) and the MSVC build tools. VS 2019 Build Tools link
against the prebuilt Qt 6.10.3 MSVC 2022 binaries without trouble.

```powershell
./fcs-qt/dev.ps1 run                # or: test, clippy --all-targets, build --release
./fcs-qt/dev.ps1 run '--' samples   # photos queued; quote the --, PowerShell eats a bare one
```

HEIC and AVIF decoding need libheif and dav1d built for the dynamic CRT, once:

```powershell
C:\vcpkg\vcpkg.exe install "libheif[core]:x64-windows-static-md" "dav1d:x64-windows-static-md"
```

`[core]` leaves out the default `hevc` feature. That feature is the x265 *encoder* (GPL-2.0),
which decoding doesn't use; decoding goes through libde265.

`dev.ps1` finds Qt (`$env:QT_DIR`, else the newest `%USERPROFILE%\Qt\6.*\msvc2022_64`), puts
its `bin` on PATH, sets `RUSTFLAGS`, and points `PKG_CONFIG_PATH` at the static-md triplet. The
crate is not a workspace member, because CI builds `--workspace` on runners without Qt.

Qt came from `aqtinstall` (`aqt install-qt windows desktop 6.10.3 win64_msvc2022_64`). Qt 6.11.2
and later moved to a new repository layout that aqtinstall 3.3.0 cannot read yet. Use the
official online installer for those.

## Findings

**Works well**

- The bridge is genuinely small to use. One `#[qobject(Singleton)]` impl with `qproperty!`,
  `#[qslot]` and `#[qsignal]` is the whole Rust/QML boundary. `src/studio.rs` holds all the
  app state; QML only renders it and calls slots.
- Background work is easy: `get_thread_handle()` plus `invoke_mut` brings a worker thread's
  result back onto the Qt thread with `&mut self`. Detection, previews and export all use it.
- `Vec<serde_json::Value>` properties give QML plain arrays of objects, enough for every list
  here.
- QML makes a modern, animated, accessible UI cheap: about 1,400 lines of QML and 1,250 of
  Rust for this screen.
- Detection and export reuse fcs-core/fcs-utils unchanged. The only shared-code changes are
  `fcs_core::apply_preset` (moved out of the egui inspector so both front ends agree) and
  `ImageFormatHint::extension`.

**Rough edges**

- **The CRT.** The workspace's `.cargo/config.toml` links the static CRT (`+crt-static`), but
  Qt uses the dynamic one. Mixing them risks one heap freeing another's memory. A nested config
  cannot undo it, because rustc lets `+crt-static` win over a later `-crt-static`. Only
  `RUSTFLAGS` can override it, so `dev.ps1` sets it and `build.rs` refuses to build otherwise.
  The C libraries have to match too: the workspace's libheif and dav1d come from vcpkg's
  `x64-windows-static` triplet (static CRT), so fcs-qt uses `x64-windows-static-md` (static
  libraries on the dynamic CRT); see the vcpkg line below.
- **libheif loads DLLs from `C:\`.** vcpkg builds libheif with an empty plugin directory,
  so on Windows `heif_init` calls `LoadLibrary` on every `*.dll` in the root of the current
  drive. All three binaries now set `LIBHEIF_PLUGIN_PATH` to a folder next to the exe before
  anything starts (`fcs_utils::process_env::confine_libheif_plugins`). It has to go through the
  CRT's `_wputenv_s`: libheif and Qt read variables with `getenv`, which never sees
  `std::env::set_var`.
- **No image provider.** QML's `Image` loads only URLs, and the bridge has no way to register
  a `QQuickImageProvider`. fcs-qt adds one itself: [cpp/image_provider.cpp](cpp/image_provider.cpp)
  (about 60 lines, compiled by `build.rs` through `qt-build-utils`) serves `image://fcs/<key>`
  from an in-memory store in [src/images.rs](src/images.rs). It gets the engine through the
  bridge's doc-hidden `QApp::qml_engine()`, which a later bridge release could change. The
  store holds a thumbnail per photo, canvas copies of the 8 most recent photos, and the
  previews on screen. Nothing goes through the disk.
- **Models are awkward.** `QListModel` exists but is a separate QObject with little
  documentation. Here QML mirrors the queue array into a `ListModel` with `set()`/`append()`,
  which keeps the list from resetting on each update.
- **Packaging is untested.** A release would need `windeployqt` (or the macOS/Linux
  equivalents) and Qt's LGPL obligations, on top of the current installers.

**If this goes further:** box editing (drag handles) is the biggest missing workflow, and maps
onto QML `DragHandler` directly. Ask upstream for an image provider API in the bridge, so the
C++ file can go.
