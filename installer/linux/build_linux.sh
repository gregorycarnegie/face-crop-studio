#!/usr/bin/env bash
# Build a Face Crop Studio AppImage and .deb from a release binary.
#
# Run from the repo root. Assumes:
#   - target/release/fcs-gui exists (built by caller, x86_64-unknown-linux-gnu)
#   - models/face_detection_yunet_2023mar_640.onnx exists (downloaded by caller)
#   - rsvg-convert, appimagetool, cargo-deb available on PATH

set -euo pipefail

VERSION="${1:?usage: build_linux.sh <version>}"
# ARCH is the slug in the artifact filenames, kept uniform with the Windows
# release assets (x86_64 / arm64). APPIMAGE_ARCH is what appimagetool insists on
# in its ARCH env var, where aarch64 is the only spelling it recognises.
case "$(uname -m)" in
    x86_64)  ARCH="x86_64"; APPIMAGE_ARCH="x86_64" ;;
    aarch64) ARCH="arm64";  APPIMAGE_ARCH="aarch64" ;;
    *) echo "error: unsupported architecture $(uname -m)" >&2; exit 1 ;;
esac
APP_NAME="face-crop-studio"
BINARY_NAME="fcs-gui"
SVG_SOURCE="fcs-gui/assets/app_logo.svg"
BIN_SRC="target/release/${BINARY_NAME}"
MODEL_FILE="models/face_detection_yunet_2023mar_640.onnx"
DESKTOP_FILE="installer/linux/face-crop-studio.desktop"
ICON_PNG="installer/linux/face-crop-studio.png"

DIST_DIR="dist/linux"
APPDIR="$DIST_DIR/${APP_NAME}.AppDir"
APPIMAGE_PATH="$DIST_DIR/face-crop-studio-${VERSION}-${ARCH}.AppImage"

for f in "$BIN_SRC" "$MODEL_FILE" "$SVG_SOURCE" "$DESKTOP_FILE"; do
    if [ ! -f "$f" ]; then
        echo "error: required file missing at $f" >&2
        exit 1
    fi
done

rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"

# --- Icon: SVG -> 256x256 PNG (used by both AppImage and .deb) ---------------
mkdir -p "$(dirname "$ICON_PNG")"
rsvg-convert -w 256 -h 256 "$SVG_SOURCE" -o "$ICON_PNG"

# --- AppImage: assemble AppDir ----------------------------------------------
mkdir -p "$APPDIR/usr/bin"
mkdir -p "$APPDIR/usr/share/applications"
mkdir -p "$APPDIR/usr/share/icons/hicolor/256x256/apps"
mkdir -p "$APPDIR/usr/share/face-crop-studio/models"

cp "$BIN_SRC" "$APPDIR/usr/bin/$BINARY_NAME"
chmod +x "$APPDIR/usr/bin/$BINARY_NAME"
cp "$MODEL_FILE" "$APPDIR/usr/share/face-crop-studio/models/"
cp "$DESKTOP_FILE" "$APPDIR/usr/share/applications/"
cp "$ICON_PNG" "$APPDIR/usr/share/icons/hicolor/256x256/apps/$APP_NAME.png"

if [ -d samples ]; then
    mkdir -p "$APPDIR/usr/share/face-crop-studio/samples"
    cp -R samples/. "$APPDIR/usr/share/face-crop-studio/samples/"
fi

# AppImage runtime requires .desktop + icon at the AppDir root.
cp "$DESKTOP_FILE" "$APPDIR/$APP_NAME.desktop"
cp "$ICON_PNG" "$APPDIR/$APP_NAME.png"

# AppRun is the entry point; symlink to the binary so std::env::current_exe()
# resolves through the symlink to the real binary path. resolve_data_path then
# finds the model via <exe_dir>/../share/face-crop-studio/.
ln -sf "usr/bin/$BINARY_NAME" "$APPDIR/AppRun"

# --- AppImage: package -------------------------------------------------------
echo "Building AppImage at $APPIMAGE_PATH"
ARCH="$APPIMAGE_ARCH" appimagetool --no-appstream "$APPDIR" "$APPIMAGE_PATH"

# --- .deb: cargo-deb reads metadata from fcs-gui/Cargo.toml -----------------
echo "Building .deb"
cargo deb -p fcs-gui --no-build --no-strip --output "$DIST_DIR/face-crop-studio-${VERSION}-${ARCH}.deb"

echo "Linux build complete:"
ls -lh "$DIST_DIR"
