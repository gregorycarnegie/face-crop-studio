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
- The photo shows its faces as boxes. Click a box (or focus it and press Space) to include or
  skip that face. All / None act on the current photo.
- Presets, face size, shape (square, rounded, oval) and format (PNG, JPEG, WebP). Previews
  are rendered through the real export path, so they show exactly what export writes.
- Export writes the selected faces of every photo to a folder, as `{stem}_face_NN.{ext}`. An
  existing file is never replaced; the new one gets a `(2)` suffix.
- Settings go to the same file as fcs-gui (`AppSettings::default_settings_path`).

Not ported: manual box editing, enhancement controls, mapping files, webcam, history/undo,
quality automation, watch folders.

## Build and run

You need Qt 6.10 or newer (MSVC 64-bit kit) and the MSVC build tools. VS 2019 Build Tools link
against the prebuilt Qt 6.10.3 MSVC 2022 binaries without trouble.

```powershell
./fcs-qt/dev.ps1 run                # or: test, clippy --all-targets, build --release
./fcs-qt/dev.ps1 run -- "$PWD\samples"   # with photos queued (dev.ps1 runs from fcs-qt/)
```

`dev.ps1` finds Qt (`$env:QT_DIR`, else the newest `%USERPROFILE%\Qt\6.*\msvc2022_64`), puts
its `bin` on PATH and sets `RUSTFLAGS`. The crate is not a workspace member, because CI builds
`--workspace` on runners without Qt.

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
- QML makes a modern, animated, accessible UI cheap: about 1,400 lines of QML and 1,000 of
  Rust for this screen.
- Detection and export reuse fcs-core/fcs-utils unchanged. The only shared-code changes are
  `fcs_core::apply_preset` (moved out of the egui inspector so both front ends agree) and
  `ImageFormatHint::extension`.

**Rough edges**

- **The CRT.** The workspace's `.cargo/config.toml` links the static CRT (`+crt-static`), but
  Qt uses the dynamic one. Mixing them risks one heap freeing another's memory. A nested config
  cannot undo it, because rustc lets `+crt-static` win over a later `-crt-static`. Only
  `RUSTFLAGS` can override it, so `dev.ps1` sets it and `build.rs` refuses to build otherwise.
  For the same reason there is no HEIC here: libheif comes from vcpkg's static triplet. dav1d
  (AVIF) still links statically and triggers one LNK4098 warning; it is a self-contained C
  decoder, so this is not the Qt heap issue.
- **No image provider.** The bridge has no `QQuickImageProvider` or `QImage`, so the canvas
  copy and previews go through PNG/JPEG files in a temp folder, loaded as `file:` URLs. It
  works, but costs an encode and decode per preview.
- **Models are awkward.** `QListModel` exists but is a separate QObject with little
  documentation. Here QML mirrors the queue array into a `ListModel` with `set()`/`append()`,
  which keeps the list from resetting on each update.
- **Packaging is untested.** A release would need `windeployqt` (or the macOS/Linux
  equivalents) and Qt's LGPL obligations, on top of the current installers.

**If this goes further:** a Rust image provider (C++ via the bridge's cxx-qt layer) would drop
the temp files. Box editing (drag handles) is the biggest missing workflow, and maps onto QML
`DragHandler` directly.
