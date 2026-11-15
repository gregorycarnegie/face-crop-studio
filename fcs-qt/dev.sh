#!/usr/bin/env bash
# Run cargo for fcs-qt on Linux or macOS (dev.ps1 is the Windows twin).
#   ./fcs-qt/dev.sh run      ./fcs-qt/dev.sh test      ./fcs-qt/dev.sh run -- samples
# Needs Qt 6.10+ with its private headers (qtbridge includes them) and qmake6 or $QMAKE.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
export QMAKE="${QMAKE:-$(command -v qmake6 || command -v qmake)}"

# libheif built from source into /usr/local, as .github/actions/linux-build-deps does: link it
# statically with libde265. Without these, a distro libheif-dev wins and may lack HEVC decode.
if [ -f /usr/local/lib/libheif.a ]; then
    export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
    export SYSTEM_DEPS_LIBHEIF_LINK=static SYSTEM_DEPS_LIBHEIF_SEARCH_NATIVE=/usr/local/lib
    export SYSTEM_DEPS_LIBHEIF_LIB="heif de265" SYSTEM_DEPS_LIBHEIF_INCLUDE=/usr/local/include
fi

[ $# -eq 0 ] && set -- run
command="$1"; shift
# From the repo root so the app finds models/, as dev.ps1 does.
cd "$here/.."
exec cargo --config "build.target-dir='$here/target'" "$command" --manifest-path "$here/Cargo.toml" "$@"
