#!/usr/bin/env bash
# Assemble a Face Crop Studio .app bundle and .dmg from a release binary.
#
# Run from the repo root. Assumes:
#   - target/aarch64-apple-darwin/release/fcs-gui and fcs-cli exist (built by caller)
#   - models/eye_refiner.onnx exists (fetched by caller from a release asset)
#   - models/scrfd80k_500m_640.onnx exists (likewise)
#   - librsvg (rsvg-convert), create-dmg installed via Homebrew
#
# Optional environment for signing/notarization (all required together):
#   APPLE_DEVELOPER_ID_NAME   "Developer ID Application: Your Name (TEAMID)"
#   APPLE_NOTARIZE_USER       Apple ID email
#   APPLE_NOTARIZE_PASS       App-specific password
#   APPLE_TEAM_ID             10-character team ID
# If APPLE_DEVELOPER_ID_NAME is unset the bundle ships unsigned and notarization
# is skipped — users need `xattr -dr com.apple.quarantine` to bypass Gatekeeper.

set -euo pipefail

VERSION="${1:?usage: build_macos.sh <version>}"
TARGET_TRIPLE="aarch64-apple-darwin"
APP_NAME="Face Crop Studio"
BINARY_NAME="fcs-gui"
BUNDLE_ID="dev.facecropstudio.app"
# Better eye points for levelling crops. Required rather than optional, for the same
# reason as the detector: a bundle without it levels by the detector's own landmarks,
# silently.
REFINER_FILE="models/eye_refiner.onnx"
# The detector. Required: there is no second detector to fall back to, so a bundle
# without it cannot detect anything at all.
DETECTOR_FILE="models/scrfd80k_500m_640.onnx"
SVG_SOURCE="fcs-gui/assets/app_logo.svg"

DIST_DIR="dist/macos"
APP_DIR="$DIST_DIR/$APP_NAME.app"
DMG_PATH="$DIST_DIR/face-crop-studio-${VERSION}-${TARGET_TRIPLE}.dmg"

BIN_SRC="target/${TARGET_TRIPLE}/release/${BINARY_NAME}"
CLI_SRC="target/${TARGET_TRIPLE}/release/fcs-cli"
for binary in "$BIN_SRC" "$CLI_SRC"; do
    if [ ! -f "$binary" ]; then
        echo "error: release binary missing at $binary" >&2
        exit 1
    fi
done
if [ ! -f "$REFINER_FILE" ]; then
    echo "error: eye refiner missing at $REFINER_FILE" >&2
    exit 1
fi
if [ ! -f "$DETECTOR_FILE" ]; then
    echo "error: detector missing at $DETECTOR_FILE" >&2
    exit 1
fi
if [ ! -f "$SVG_SOURCE" ]; then
    echo "error: app logo missing at $SVG_SOURCE" >&2
    exit 1
fi

rm -rf "$DIST_DIR"
mkdir -p "$APP_DIR/Contents/MacOS/models"
mkdir -p "$APP_DIR/Contents/Resources"

# --- Icon: SVG -> iconset -> .icns -------------------------------------------
ICONSET_TMP="$(mktemp -d)/AppIcon.iconset"
mkdir -p "$ICONSET_TMP"
# Apple's required iconset sizes (size@scale pairs).
declare -a ICON_PAIRS=(
    "16:1"   "16:2"
    "32:1"   "32:2"
    "128:1"  "128:2"
    "256:1"  "256:2"
    "512:1"  "512:2"
)
for pair in "${ICON_PAIRS[@]}"; do
    base="${pair%%:*}"
    scale="${pair##*:}"
    px=$((base * scale))
    if [ "$scale" = "1" ]; then
        name="icon_${base}x${base}.png"
    else
        name="icon_${base}x${base}@2x.png"
    fi
    rsvg-convert -w "$px" -h "$px" "$SVG_SOURCE" -o "$ICONSET_TMP/$name"
done
iconutil -c icns -o "$APP_DIR/Contents/Resources/AppIcon.icns" "$ICONSET_TMP"

# --- Info.plist (template substitution) --------------------------------------
sed -e "s|{{VERSION}}|$VERSION|g" \
    -e "s|{{BUNDLE_ID}}|$BUNDLE_ID|g" \
    installer/macos/Info.plist.in \
    > "$APP_DIR/Contents/Info.plist"

# --- Binary + bundled assets -------------------------------------------------
cp "$BIN_SRC" "$APP_DIR/Contents/MacOS/$BINARY_NAME"
cp "$CLI_SRC" "$APP_DIR/Contents/MacOS/fcs-cli"
chmod +x "$APP_DIR/Contents/MacOS/$BINARY_NAME" "$APP_DIR/Contents/MacOS/fcs-cli"
cp "$REFINER_FILE" "$APP_DIR/Contents/MacOS/models/"
cp "$DETECTOR_FILE" "$APP_DIR/Contents/MacOS/models/"

if [ -d samples ]; then
    cp -R samples "$APP_DIR/Contents/Resources/samples"
fi

for license in LICENSE LICENSE-MIT LICENSE-APACHE README.md; do
    [ -f "$license" ] && cp "$license" "$APP_DIR/Contents/Resources/"
done

# --- Code signing ------------------------------------------------------------
SIGNING_IDENTITY="${APPLE_DEVELOPER_ID_NAME:-}"
if [ -n "$SIGNING_IDENTITY" ]; then
    echo "Signing bundle with: $SIGNING_IDENTITY"
    # Sign both executables before signing the enclosing bundle.
    for binary in "$BINARY_NAME" fcs-cli; do
        codesign --force --sign "$SIGNING_IDENTITY" \
                 --options runtime \
                 --entitlements installer/macos/entitlements.plist \
                 --timestamp \
                 "$APP_DIR/Contents/MacOS/$binary"
    done
    codesign --force --sign "$SIGNING_IDENTITY" \
             --options runtime \
             --entitlements installer/macos/entitlements.plist \
             --timestamp \
             "$APP_DIR"
    codesign --verify --deep --strict --verbose=2 "$APP_DIR"
else
    echo "APPLE_DEVELOPER_ID_NAME not set — shipping unsigned bundle."
fi

# --- DMG ----------------------------------------------------------------------
echo "Building DMG at $DMG_PATH"
rm -f "$DMG_PATH"
create-dmg \
    --volname "$APP_NAME $VERSION" \
    --window-size 600 400 \
    --icon-size 100 \
    --icon "$APP_NAME.app" 175 190 \
    --app-drop-link 425 190 \
    --hide-extension "$APP_NAME.app" \
    "$DMG_PATH" \
    "$APP_DIR"

# --- Notarization ------------------------------------------------------------
if [ -n "$SIGNING_IDENTITY" ] \
    && [ -n "${APPLE_NOTARIZE_USER:-}" ] \
    && [ -n "${APPLE_NOTARIZE_PASS:-}" ] \
    && [ -n "${APPLE_TEAM_ID:-}" ]; then
    echo "Submitting $DMG_PATH for notarization (wait may take several minutes)"
    xcrun notarytool submit "$DMG_PATH" \
        --apple-id "$APPLE_NOTARIZE_USER" \
        --password "$APPLE_NOTARIZE_PASS" \
        --team-id "$APPLE_TEAM_ID" \
        --wait
    echo "Stapling notarization ticket"
    xcrun stapler staple "$DMG_PATH"
else
    echo "Skipping notarization (signing identity and/or notarization creds absent)."
fi

echo "macOS build complete: $DMG_PATH"
