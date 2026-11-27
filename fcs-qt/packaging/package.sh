#!/usr/bin/env bash
# Package Face Crop Studio (fcs-qt with Qt deployed beside it, and fcs-cli): an AppImage and .deb
# on Linux, a .app and .dmg on macOS. These are the release assets for both platforms.
#   ./fcs-qt/packaging/package.sh [version]   -> fcs-qt/dist/
# The version defaults to the workspace's; release.yml passes the tag's, so an rc is named as one.
# Needs the models in models/ (as for the other release builds), what dev.sh needs and
# rsvg-convert (icons); on Linux also network access the first time (linuxdeploy is downloaded),
# on macOS create-dmg.
# macOS signing and notarization, optional (all four together for notarization):
#   APPLE_DEVELOPER_ID_NAME   "Developer ID Application: Your Name (TEAMID)"; unset signs ad hoc
#   APPLE_NOTARIZE_USER, APPLE_NOTARIZE_PASS, APPLE_TEAM_ID
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
qt="$root/fcs-qt"
dist="$qt/dist"

"$qt/dev.sh" build --release
# fcs-cli is a workspace member, built from the root manifest into the root target/.
cargo build --release -p fcs-cli --manifest-path "$root/Cargo.toml"
cli="$root/target/release/fcs-cli"
qmake="${QMAKE:-$(command -v qmake6 || command -v qmake)}"
qt_version="$("$qmake" -query QT_VERSION)"
# Face Crop Studio's version (the workspace's), which the About box shows too.
version="${1:-$(sed -n 's/^version = "\(.*\)"/\1/p' "$root/Cargo.toml" | head -n 1)}"
version="${version#v}"
rm -rf "$dist"
mkdir -p "$dist"

# Face Crop Studio's licences, Qt's (LGPL-3.0 + the GPL-3.0 it builds on), the notice and README.
notices() {
    mkdir -p "$1/licenses"
    cp "$root/LICENSE-MIT" "$root/LICENSE-APACHE" "$root/README.md" "$1/"
    cp "$qt/packaging/licenses/"* "$1/licenses/"
    sed -e "s/{{QT_VERSION}}/$qt_version/g" -e "s/{{QT_SERIES}}/${qt_version%.*}/g" \
        "$qt/packaging/NOTICE-Qt.txt" > "$1/NOTICE-Qt.txt"
}
models() {
    mkdir -p "$1"
    cp "$root/models/scrfd80k_500m_640.onnx" "$root/models/eye_refiner.onnx" "$1/"
}
samples() {
    mkdir -p "$1"
    cp -R "$root/samples/." "$1/"
}

case "$(uname -s)" in
Linux)
    arch="$(uname -m)"
    # Release file names say arm64, not aarch64 (linuxdeploy's downloads keep uname's).
    file_arch="${arch/aarch64/arm64}"
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

    appdir="$dist/face-crop-studio.AppDir"
    # resolve_data_path looks in <exe>/../share/face-crop-studio, for both executables.
    models "$appdir/usr/share/face-crop-studio/models"
    samples "$appdir/usr/share/face-crop-studio/samples"
    notices "$appdir/usr/share/doc/face-crop-studio"
    # linuxdeploy names the icon after its file, which has to match the desktop file's Icon=.
    rsvg-convert -w 256 -h 256 "$root/assets/app_logo.svg" -o "$tools/face-crop-studio.png"
    desktop="$root/installer/linux/face-crop-studio.desktop"

    # QML_SOURCES_PATHS: the QML is compiled into the binary, so the plugin scans its sources.
    # APPIMAGE_EXTRACT_AND_RUN: no FUSE on CI runners or in containers.
    # AppRun starts the desktop file's Exec, fcs-qt; fcs-cli comes along with its libraries.
    (cd "$dist" && PATH="$tools:$PATH" QMAKE="$qmake" \
        LD_LIBRARY_PATH="$("$qmake" -query QT_INSTALL_LIBS):${LD_LIBRARY_PATH:-}" \
        QML_SOURCES_PATHS="$qt/src/fcs_qt" APPIMAGE_EXTRACT_AND_RUN=1 \
        LDAI_OUTPUT="face-crop-studio-$version-$file_arch.AppImage" \
        "$tools/linuxdeploy-$arch.AppImage" --appdir "$appdir" \
            --executable "$qt/target/release/fcs-qt" --executable "$cli" \
            --desktop-file "$desktop" --icon-file "$tools/face-crop-studio.png" \
            --plugin qt --output appimage)

    # A .deb of the same AppDir, under /opt: it carries its own Qt, since a build only runs on
    # the exact Qt it was compiled against. Depends lists what the system still has to provide.
    # 2.x's package name, so it upgrades in place; it replaces 2.4's Qt preview package, which
    # owned /usr/bin/fcs-qt.
    deb_arch="$(dpkg --print-architecture)"
    pkg="$dist/deb"
    opt="$pkg/opt/face-crop-studio"
    mkdir -p "$opt" "$pkg/DEBIAN" "$pkg/usr/bin" "$pkg/usr/share/applications" \
        "$pkg/usr/share/icons/hicolor/256x256/apps"
    cp -a "$appdir/usr/." "$opt/"
    # The binaries find their libraries, Qt and models relative to their resolved paths.
    ln -s /opt/face-crop-studio/bin/fcs-qt "$pkg/usr/bin/fcs-qt"
    ln -s /opt/face-crop-studio/bin/fcs-cli "$pkg/usr/bin/fcs-cli"
    cp "$desktop" "$pkg/usr/share/applications/"
    cp "$tools/face-crop-studio.png" "$pkg/usr/share/icons/hicolor/256x256/apps/"
    mkdir -p "$tools/shlibs/debian"
    # dpkg-shlibdeps reads a source package's control file; this is the least it accepts.
    printf 'Source: face-crop-studio\n\nPackage: face-crop-studio\nArchitecture: any\n' > "$tools/shlibs/debian/control"
    mapfile -t elves < <(find "$opt" -type f -exec sh -c 'head -c 4 "$1" | grep -q ELF' _ {} \; -print)
    depends="$(cd "$tools/shlibs" && dpkg-shlibdeps -O --ignore-missing-info -l"$opt/lib" \
        "${elves[@]}" | sed -n 's/^shlibs:Depends=//p')"
    [ -n "$depends" ] || { echo "dpkg-shlibdeps found no dependencies" >&2; exit 1; }
    cat > "$pkg/DEBIAN/control" <<EOF
Package: face-crop-studio
Version: $version
Architecture: $deb_arch
Maintainer: Gregory Carnegie <gregory.carnegie+claude_ai@gmail.com>
Depends: $depends
Conflicts: face-crop-studio-qt
Replaces: face-crop-studio-qt
Section: graphics
Priority: optional
Homepage: https://facecropstudio.com/
Description: Face detection and batch portrait cropping
 Face Crop Studio finds faces with its bundled detector and crops them to passport, portrait
 and custom sizes, one photo at a time or in batches. Ships the desktop app (fcs-qt, Qt Quick)
 and the command line (fcs-cli); Qt is bundled under /opt/face-crop-studio.
EOF
    dpkg-deb --build --root-owner-group "$pkg" "$dist/face-crop-studio-$version-$file_arch.deb"
    rm -rf "$pkg"
    ;;
Darwin)
    # 2.x's name and bundle id, so it replaces that app in Applications.
    app="$dist/Face Crop Studio.app"
    mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
    cp "$qt/target/release/fcs-qt" "$cli" "$app/Contents/MacOS/"
    # resolve_data_path looks next to the executable.
    models "$app/Contents/MacOS/models"
    samples "$app/Contents/Resources/samples"
    notices "$app/Contents/Resources"
    # The Finder and Dock icon: the logo at every size Apple's iconset asks for.
    iconset="$qt/target/packaging-tools/AppIcon.iconset"
    rm -rf "$iconset" && mkdir -p "$iconset"
    for size in 16 32 128 256 512; do
        rsvg-convert -w "$size" -h "$size" "$root/assets/app_logo.svg" -o "$iconset/icon_${size}x${size}.png"
        rsvg-convert -w $((size * 2)) -h $((size * 2)) "$root/assets/app_logo.svg" -o "$iconset/icon_${size}x${size}@2x.png"
    done
    iconutil -c icns -o "$app/Contents/Resources/AppIcon.icns" "$iconset"
    sed -e "s|{{VERSION}}|$version|g" -e "s|{{BUNDLE_ID}}|dev.facecropstudio.app|g" \
        "$root/installer/macos/Info.plist.in" > "$app/Contents/Info.plist"
    # macdeployqt copies Qt and every other non-system dylib in.
    macdeployqt "$app" -qmldir="$qt/src/fcs_qt" -always-overwrite
    # It also copies Qt's SQL drivers, which fcs-qt never uses and whose ODBC/PostgreSQL/Mimer
    # client libraries it cannot bundle, leaving references into Homebrew and /Applications.
    rm -rf "$app/Contents/PlugIns/sqldrivers"
    # Apple silicon refuses unsigned code, and macdeployqt's edits void the linker's ad-hoc
    # signatures. Without a Developer ID, sign ad hoc (Gatekeeper still warns).
    identity="${APPLE_DEVELOPER_ID_NAME:-}"
    if [ -n "$identity" ]; then
        codesign --force --deep --options runtime --timestamp \
            --entitlements "$root/installer/macos/entitlements.plist" --sign "$identity" "$app"
        codesign --verify --deep --strict --verbose=2 "$app"
    else
        codesign --force --deep --sign - "$app"
    fi
    dmg="$dist/face-crop-studio-$version-$(uname -m | sed 's/arm64/aarch64/')-apple-darwin.dmg"
    # A preview until someone checks the Qt app on a real Mac (decided 2026-11-17).
    create-dmg \
        --volname "Face Crop Studio $version (macOS preview)" \
        --window-size 600 400 \
        --icon-size 100 \
        --icon "Face Crop Studio.app" 175 190 \
        --app-drop-link 425 190 \
        --hide-extension "Face Crop Studio.app" \
        "$dmg" "$app"
    if [ -n "$identity" ] && [ -n "${APPLE_NOTARIZE_USER:-}" ] \
        && [ -n "${APPLE_NOTARIZE_PASS:-}" ] && [ -n "${APPLE_TEAM_ID:-}" ]; then
        xcrun notarytool submit "$dmg" --apple-id "$APPLE_NOTARIZE_USER" \
            --password "$APPLE_NOTARIZE_PASS" --team-id "$APPLE_TEAM_ID" --wait
        xcrun stapler staple "$dmg"
    fi
    ;;
*)
    echo "unsupported system: $(uname -s)" >&2
    exit 1
    ;;
esac
echo "Packaged:"
ls -lh "$dist"
