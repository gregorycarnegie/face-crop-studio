#!/usr/bin/env bash
# Package fcs-qt with Qt deployed beside it: an AppImage on Linux, a .app and .dmg on macOS.
#   ./fcs-qt/packaging/package.sh        -> fcs-qt/dist/
# Needs the models in models/ (as for the other release builds) and what dev.sh needs; on Linux
# also rsvg-convert, and network access the first time (linuxdeploy is downloaded).
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
qt="$root/fcs-qt"
dist="$qt/dist"

"$qt/dev.sh" build --release
qmake="${QMAKE:-$(command -v qmake6 || command -v qmake)}"
qt_version="$("$qmake" -query QT_VERSION)"
rm -rf "$dist"
mkdir -p "$dist"

# Face Crop Studio's licences, Qt's (LGPL-3.0 + the GPL-3.0 it builds on) and the notice.
notices() {
    mkdir -p "$1/licenses"
    cp "$root/LICENSE-MIT" "$root/LICENSE-APACHE" "$1/"
    cp "$qt/packaging/licenses/"* "$1/licenses/"
    sed -e "s/{{QT_VERSION}}/$qt_version/g" -e "s/{{QT_SERIES}}/${qt_version%.*}/g" \
        "$qt/packaging/NOTICE-Qt.txt" > "$1/NOTICE-Qt.txt"
}
models() {
    mkdir -p "$1"
    cp "$root/models/scrfd80k_500m_640.onnx" "$root/models/eye_refiner.onnx" "$1/"
}

case "$(uname -s)" in
Linux)
    arch="$(uname -m)"
    tools="$qt/target/packaging-tools"
    mkdir -p "$tools"
    # shortcut: linuxdeploy only publishes moving "continuous" builds; pin a mirror if one breaks.
    for tool in linuxdeploy linuxdeploy-plugin-qt; do
        [ -x "$tools/$tool-$arch.AppImage" ] || {
            curl -fsSL -o "$tools/$tool-$arch.AppImage" \
                "https://github.com/linuxdeploy/$tool/releases/download/continuous/$tool-$arch.AppImage"
            chmod +x "$tools/$tool-$arch.AppImage"
        }
    done

    appdir="$dist/fcs-qt.AppDir"
    # resolve_data_path looks in <exe>/../share/face-crop-studio, as for fcs-gui's AppImage.
    models "$appdir/usr/share/face-crop-studio/models"
    notices "$appdir/usr/share/doc/fcs-qt"
    cat > "$tools/fcs-qt.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=Face Crop Studio (Qt preview)
Comment=Face detection and batch portrait cropping
Exec=fcs-qt
Icon=fcs-qt
Terminal=false
Categories=Graphics;Photography;
EOF
    rsvg-convert -w 256 -h 256 "$root/fcs-gui/assets/app_logo.svg" -o "$tools/fcs-qt.png"

    # QML_SOURCES_PATHS: the QML is compiled into the binary, so the plugin scans its sources.
    # APPIMAGE_EXTRACT_AND_RUN: no FUSE on CI runners or in containers.
    (cd "$dist" && PATH="$tools:$PATH" QMAKE="$qmake" \
        LD_LIBRARY_PATH="$("$qmake" -query QT_INSTALL_LIBS):${LD_LIBRARY_PATH:-}" \
        QML_SOURCES_PATHS="$qt/src/fcs_qt" APPIMAGE_EXTRACT_AND_RUN=1 \
        LDAI_OUTPUT="fcs-qt-linux-$arch.AppImage" \
        "$tools/linuxdeploy-$arch.AppImage" --appdir "$appdir" \
            --executable "$qt/target/release/fcs-qt" \
            --desktop-file "$tools/fcs-qt.desktop" --icon-file "$tools/fcs-qt.png" \
            --plugin qt --output appimage)
    ;;
Darwin)
    app="$dist/Face Crop Studio Qt.app"
    mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
    cp "$qt/target/release/fcs-qt" "$app/Contents/MacOS/"
    # resolve_data_path looks next to the executable, as for fcs-gui's bundle.
    models "$app/Contents/MacOS/models"
    notices "$app/Contents/Resources"
    cat > "$app/Contents/Info.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key><string>fcs-qt</string>
    <key>CFBundleIdentifier</key><string>dev.facecropstudio.qtpreview</string>
    <key>CFBundleName</key><string>Face Crop Studio Qt</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>0.1.0</string>
    <key>LSMinimumSystemVersion</key><string>13.0</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSCameraUsageDescription</key><string>Face Crop Studio uses the camera to capture faces for cropping.</string>
</dict>
</plist>
EOF
    # macdeployqt copies Qt and every other non-system dylib (Homebrew's libheif, dav1d) in.
    macdeployqt "$app" -qmldir="$qt/src/fcs_qt" -always-overwrite
    # Apple silicon refuses unsigned code, and macdeployqt's edits void the linker's ad-hoc
    # signatures; re-sign ad hoc (no Developer ID yet, so Gatekeeper still warns).
    codesign --force --deep --sign - "$app"
    hdiutil create -volname "Face Crop Studio Qt" -srcfolder "$app" -ov -format UDZO \
        "$dist/fcs-qt-macos-$(uname -m).dmg"
    ;;
*)
    echo "unsupported system: $(uname -s)" >&2
    exit 1
    ;;
esac
echo "Packaged:"
ls -lh "$dist"
