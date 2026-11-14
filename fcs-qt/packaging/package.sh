#!/usr/bin/env bash
# Package fcs-qt with Qt deployed beside it: an AppImage and .deb on Linux, a .app and .dmg on
# macOS.
#   ./fcs-qt/packaging/package.sh        -> fcs-qt/dist/
# Needs the models in models/ (as for the other release builds), what dev.sh needs and
# rsvg-convert (icons); on Linux also network access the first time (linuxdeploy is downloaded).
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
qt="$root/fcs-qt"
dist="$qt/dist"

"$qt/dev.sh" build --release
qmake="${QMAKE:-$(command -v qmake6 || command -v qmake)}"
qt_version="$("$qmake" -query QT_VERSION)"
# Face Crop Studio's version (the workspace's), which the About box shows too.
version="$(sed -n 's/^version = "\(.*\)"/\1/p' "$root/Cargo.toml" | head -n 1)"
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
        LDAI_OUTPUT="face-crop-studio-qt-$version-$arch.AppImage" \
        "$tools/linuxdeploy-$arch.AppImage" --appdir "$appdir" \
            --executable "$qt/target/release/fcs-qt" \
            --desktop-file "$tools/fcs-qt.desktop" --icon-file "$tools/fcs-qt.png" \
            --plugin qt --output appimage)

    # A .deb of the same AppDir, under /opt: it carries its own Qt, since a build only runs on
    # the exact Qt it was compiled against. Depends lists what the system still has to provide.
    deb_arch="$(dpkg --print-architecture)"
    pkg="$dist/deb"
    opt="$pkg/opt/face-crop-studio-qt"
    mkdir -p "$opt" "$pkg/DEBIAN" "$pkg/usr/bin" "$pkg/usr/share/applications" \
        "$pkg/usr/share/icons/hicolor/256x256/apps"
    cp -a "$appdir/usr/." "$opt/"
    # The binary finds its libraries, Qt and models relative to its resolved path.
    ln -s /opt/face-crop-studio-qt/bin/fcs-qt "$pkg/usr/bin/fcs-qt"
    cp "$tools/fcs-qt.desktop" "$pkg/usr/share/applications/"
    cp "$tools/fcs-qt.png" "$pkg/usr/share/icons/hicolor/256x256/apps/"
    mkdir -p "$tools/shlibs/debian"
    # dpkg-shlibdeps reads a source package's control file; this is the least it accepts.
    printf 'Source: fcs-qt\n\nPackage: face-crop-studio-qt\nArchitecture: any\n' > "$tools/shlibs/debian/control"
    mapfile -t elves < <(find "$opt" -type f -exec sh -c 'head -c 4 "$1" | grep -q ELF' _ {} \; -print)
    depends="$(cd "$tools/shlibs" && dpkg-shlibdeps -O --ignore-missing-info -l"$opt/lib" \
        "${elves[@]}" | sed -n 's/^shlibs:Depends=//p')"
    [ -n "$depends" ] || { echo "dpkg-shlibdeps found no dependencies" >&2; exit 1; }
    cat > "$pkg/DEBIAN/control" <<EOF
Package: face-crop-studio-qt
Version: $version
Architecture: $deb_arch
Maintainer: Gregory Carnegie <gregory.carnegie+claude_ai@gmail.com>
Depends: $depends
Section: graphics
Priority: optional
Homepage: https://facecropstudio.com/
Description: Face Crop Studio (Qt preview)
 Face detection and batch portrait cropping, with the Qt Quick interface. Installs
 beside the face-crop-studio package; Qt is bundled under /opt/face-crop-studio-qt.
EOF
    dpkg-deb --build --root-owner-group "$pkg" "$dist/face-crop-studio-qt-$version-$arch.deb"
    rm -rf "$pkg"
    ;;
Darwin)
    app="$dist/Face Crop Studio Qt.app"
    mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
    cp "$qt/target/release/fcs-qt" "$app/Contents/MacOS/"
    # resolve_data_path looks next to the executable, as for fcs-gui's bundle.
    models "$app/Contents/MacOS/models"
    notices "$app/Contents/Resources"
    # The Finder and Dock icon: fcs-gui's logo at every size Apple's iconset asks for.
    iconset="$qt/target/packaging-tools/AppIcon.iconset"
    rm -rf "$iconset" && mkdir -p "$iconset"
    for size in 16 32 128 256 512; do
        rsvg-convert -w "$size" -h "$size" "$root/fcs-gui/assets/app_logo.svg" -o "$iconset/icon_${size}x${size}.png"
        rsvg-convert -w $((size * 2)) -h $((size * 2)) "$root/fcs-gui/assets/app_logo.svg" -o "$iconset/icon_${size}x${size}@2x.png"
    done
    iconutil -c icns -o "$app/Contents/Resources/AppIcon.icns" "$iconset"
    cat > "$app/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key><string>fcs-qt</string>
    <key>CFBundleIconFile</key><string>AppIcon</string>
    <key>CFBundleIdentifier</key><string>dev.facecropstudio.qtpreview</string>
    <key>CFBundleName</key><string>Face Crop Studio Qt</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>$version</string>
    <key>LSMinimumSystemVersion</key><string>13.0</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSCameraUsageDescription</key><string>Face Crop Studio uses the camera to capture faces for cropping.</string>
</dict>
</plist>
EOF
    # macdeployqt copies Qt and every other non-system dylib (Homebrew's libheif, dav1d) in.
    macdeployqt "$app" -qmldir="$qt/src/fcs_qt" -always-overwrite
    # It also copies Qt's SQL drivers, which fcs-qt never uses and whose ODBC/PostgreSQL/Mimer
    # client libraries it cannot bundle, leaving references into Homebrew and /Applications.
    rm -rf "$app/Contents/PlugIns/sqldrivers"
    # Apple silicon refuses unsigned code, and macdeployqt's edits void the linker's ad-hoc
    # signatures; re-sign ad hoc (no Developer ID yet, so Gatekeeper still warns).
    codesign --force --deep --sign - "$app"
    hdiutil create -volname "Face Crop Studio Qt" -srcfolder "$app" -ov -format UDZO \
        "$dist/face-crop-studio-qt-$version-$(uname -m | sed 's/arm64/aarch64/')-apple-darwin.dmg"
    ;;
*)
    echo "unsupported system: $(uname -s)" >&2
    exit 1
    ;;
esac
echo "Packaged:"
ls -lh "$dist"
