#!/usr/bin/env bash
# Build a Face Crop Studio AppImage and .deb from a release binary.
#
# Run from the repo root. Assumes:
#   - target/release/fcs-gui exists (built by caller, x86_64-unknown-linux-gnu)
#   - models/face_detection_yunet_2023mar_640.onnx exists (downloaded by caller)
#   - models/eye_refiner.onnx exists (fetched by caller from a release asset)
#   - models/scrfd80k_500m_640.onnx exists (likewise)
#   - rsvg-convert, appimagetool, cargo-deb available on PATH
#
# Optional: set FCS_ORT_LIB to a libonnxruntime.so to bundle it, which makes
# detection roughly 3x faster. Without it the packages still work and fall back
# to the built-in CPU graph, so local builds need no download.

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
# Better eye points for levelling crops. Required rather than optional: the packages are
# what users get, and a release that quietly shipped without it would level by YuNet's
# landmarks while the release notes claimed otherwise.
REFINER_FILE="models/eye_refiner.onnx"
# The detector itself; without it the package falls back to YuNet, which is a
# silent downgrade rather than a failure, so it is required here too.
DETECTOR_FILE="models/scrfd80k_500m_640.onnx"
DESKTOP_FILE="installer/linux/face-crop-studio.desktop"
ICON_PNG="installer/linux/face-crop-studio.png"

ORT_LIB="${FCS_ORT_LIB:-}"
if [ -n "$ORT_LIB" ] && [ ! -f "$ORT_LIB" ]; then
    echo "error: FCS_ORT_LIB is set but $ORT_LIB does not exist" >&2
    exit 1
fi

DIST_DIR="dist/linux"
APPDIR="$DIST_DIR/${APP_NAME}.AppDir"
APPIMAGE_PATH="$DIST_DIR/face-crop-studio-${VERSION}-${ARCH}.AppImage"

for f in "$BIN_SRC" "$MODEL_FILE" "$REFINER_FILE" "$DETECTOR_FILE" "$SVG_SOURCE" "$DESKTOP_FILE"; do
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
cp "$REFINER_FILE" "$APPDIR/usr/share/face-crop-studio/models/"
cp "$DETECTOR_FILE" "$APPDIR/usr/share/face-crop-studio/models/"
# Beside the executable, which is the first place fcs-ort looks. Putting it in a
# lib directory instead would rely on the loader's search path and could collide
# with a distro-provided onnxruntime.
if [ -n "$ORT_LIB" ]; then
    cp "$ORT_LIB" "$APPDIR/usr/bin/libonnxruntime.so"
    chmod 644 "$APPDIR/usr/bin/libonnxruntime.so"
fi
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
# The bundled-ort variant exists because cargo-deb asset lists are static and it
# fails on a missing file: without the variant, every local build would need the
# library downloaded first. The variant is selected only when it is actually
# present, and sets the same package name so the artifact is identical either way.
DEB_VARIANT=()
if [ -n "$ORT_LIB" ]; then
    cp "$ORT_LIB" "target/release/libonnxruntime.so"
    DEB_VARIANT=(--variant bundled-ort)
    echo "Bundling ONNX Runtime from $ORT_LIB"
else
    echo "No FCS_ORT_LIB set; packaging without ONNX Runtime (built-in graph only)"
fi

echo "Building .deb"
cargo deb -p fcs-gui --no-build --no-strip "${DEB_VARIANT[@]}"     --output "$DIST_DIR/face-crop-studio-${VERSION}-${ARCH}.deb"

echo "Linux build complete:"
ls -lh "$DIST_DIR"
