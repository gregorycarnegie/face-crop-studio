#!/usr/bin/env bash
# Run cargo for fcs-qt on Linux or macOS (dev.ps1 is the Windows twin).
#   ./fcs-qt/dev.sh run      ./fcs-qt/dev.sh test      ./fcs-qt/dev.sh run -- samples
# Needs Qt 6.10+ with its private headers (qtbridge includes them) and qmake6 or $QMAKE.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
export QMAKE="${QMAKE:-$(command -v qmake6 || command -v qmake)}"

# libheif built from source, as .github/actions/linux-build-deps does: link it statically with
# libde265. libheif-sys needs libheif 1.23+, newer than any distro's (Ubuntu 26.04 ships 1.21),
# and with a distro libheif-dev installed it would otherwise win and may lack HEVC decode.
# Looked for in $FCS_NATIVE_PREFIX, then /usr/local (CI), then ~/.local/fcs-native (no sudo).
for prefix in ${FCS_NATIVE_PREFIX:-} /usr/local "$HOME/.local/fcs-native"; do
    if [ -f "$prefix/lib/libheif.a" ]; then
        export PKG_CONFIG_PATH="$prefix/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
        export SYSTEM_DEPS_LIBHEIF_LINK=static SYSTEM_DEPS_LIBHEIF_SEARCH_NATIVE="$prefix/lib"
        export SYSTEM_DEPS_LIBHEIF_LIB="heif de265" SYSTEM_DEPS_LIBHEIF_INCLUDE="$prefix/include"
        break
    fi
done

# A terminal inside a snap (VS Code's snap, for one) passes the snap's GTK and GSettings
# environment on. Qt's GTK3 platform theme then starts the system GTK against the snap's
# schemas, and on GNOME 50 GTK aborts at startup on a key they lack (xsettings
# 'antialiasing'). Drop the snap's variables and its XDG_DATA_DIRS entries.
if [ -n "${SNAP:-}" ]; then
    unset GSETTINGS_SCHEMA_DIR GTK_EXE_PREFIX GTK_PATH GTK_IM_MODULE_FILE GIO_MODULE_DIR \
        GDK_PIXBUF_MODULE_FILE GDK_PIXBUF_MODULEDIR LOCPATH
    data_dirs=""
    IFS=: read -ra dirs <<< "${XDG_DATA_DIRS:-}"
    for dir in "${dirs[@]}"; do
        case "$dir" in
            "$SNAP"/* | "$SNAP" | "${SNAP_USER_DATA:-/nonexistent}"/* | "${SNAP_USER_DATA:-/nonexistent}") ;;
            *) data_dirs="${data_dirs:+$data_dirs:}$dir" ;;
        esac
    done
    export XDG_DATA_DIRS="$data_dirs"
fi

[ $# -eq 0 ] && set -- run
command="$1"; shift
# From the repo root so the app finds models/, as dev.ps1 does.
cd "$here/.."
exec cargo --config "build.target-dir='$here/target'" "$command" --manifest-path "$here/Cargo.toml" "$@"
