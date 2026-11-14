#!/usr/bin/env bash
# Run a packaged fcs-qt on the sample photos for 20 s, from an empty folder so the models have to
# come from the package, and check it stayed up, loaded the detector and its QML.
#   smoke.sh <command...>      e.g. smoke.sh xvfb-run -a dist/fcs-qt-linux-x86_64.AppImage
set -uo pipefail
samples="$(cd "$(dirname "$0")/../.." && pwd)/samples"
log="$(mktemp)"
cd "$(mktemp -d)"
"$@" "$samples" > "$log" 2>&1 &
pid=$!
sleep 20
alive=0
kill -0 "$pid" 2>/dev/null && alive=1
kill "$pid" 2>/dev/null
wait "$pid" 2>/dev/null
cat "$log"
[ "$alive" = 1 ] || { echo "smoke: the app exited early" >&2; exit 1; }
grep -q "SCRFD on" "$log" || { echo "smoke: the detector did not load from the package" >&2; exit 1; }
if grep -E "qrc:/|is not installed|Cannot load library|Could not load the Qt platform plugin" "$log"; then
    echo "smoke: QML or plugin errors (above)" >&2
    exit 1
fi
echo "smoke: ok"
