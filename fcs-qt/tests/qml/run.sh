#!/usr/bin/env bash
# Run fcs-qt's QML tests (tst_*.qml) against MockStudio, with Qt's qmltestrunner on PATH:
#   ./fcs-qt/tests/qml/run.sh [qmltestrunner options]
# The module is copied to a temporary folder with MockStudio declared as its Studio, so the
# production QML and qmldir stay as the app loads them.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
src="$here/../../src/fcs_qt"
modules="$(mktemp -d)"
trap 'rm -rf "$modules"' EXIT
mkdir "$modules/fcs_qt"
cp "$src"/*.qml "$here/MockStudio.qml" "$here/fixture.js" "$modules/fcs_qt/"
# `prefer` points at the app's compiled-in resources, which a test run does not have.
grep -v '^prefer ' "$src/qmldir" > "$modules/fcs_qt/qmldir"
echo 'singleton Studio 1.0 MockStudio.qml' >> "$modules/fcs_qt/qmldir"
# Basic, as main.rs pins it; a desktop style would draw (and warn) differently.
QT_QUICK_CONTROLS_STYLE=Basic QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}" qmltestrunner -import "$modules" -input "$here" "$@"
