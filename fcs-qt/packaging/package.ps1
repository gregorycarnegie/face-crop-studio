# Build a portable fcs-qt folder and zip for Windows, with Qt deployed beside the exe.
#   ./fcs-qt/packaging/package.ps1          -> fcs-qt/dist/fcs-qt-windows-x86_64(.zip)
# Needs the models in models/ (as for the other release builds) and what dev.ps1 needs.
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path "$PSScriptRoot\..\..").Path
$name = 'fcs-qt-windows-x86_64'
$dist = "$root\fcs-qt\dist\$name"

& "$root\fcs-qt\dev.ps1" build --release   # also puts Qt's bin on PATH for windeployqt
if ($LASTEXITCODE) { exit $LASTEXITCODE }

if (Test-Path $dist) { Remove-Item -Recurse -Force $dist }
New-Item -ItemType Directory -Force "$dist\models", "$dist\licenses" | Out-Null
Copy-Item "$root\fcs-qt\target\release\fcs-qt.exe" $dist
foreach ($model in 'scrfd80k_500m_640.onnx', 'eye_refiner.onnx') {
    Copy-Item "$root\models\$model" "$dist\models\"   # fails loudly if a model is missing
}
Copy-Item "$root\LICENSE-MIT", "$root\LICENSE-APACHE" $dist
Copy-Item "$root\fcs-qt\packaging\licenses\*" "$dist\licenses\"
$qtVersion = (qmake -query QT_VERSION).Trim()
(Get-Content "$root\fcs-qt\packaging\NOTICE-Qt.txt" -Raw).
    Replace('{{QT_VERSION}}', $qtVersion).
    Replace('{{QT_SERIES}}', ($qtVersion -replace '\.\d+$', '')) |
    Set-Content "$dist\NOTICE-Qt.txt" -NoNewline

# --qmldir: the QML is compiled into the exe, so point the import scanner at its sources.
# No software OpenGL: Qt Quick renders through Direct3D 11 on Windows. No D3D compiler either:
# windeployqt's copy is Windows 8.1's, too old for wgpu's DX12 shaders (cs_5_1), and an
# app-local copy shadows the current one every Windows 10+ system has.
windeployqt --release --qmldir "$root\fcs-qt\src\fcs_qt" --no-opengl-sw --no-system-dxc-compiler --no-system-d3d-compiler "$dist\fcs-qt.exe"
if ($LASTEXITCODE) { exit $LASTEXITCODE }

# Qt is built against the dynamic MSVC runtime (fcs-gui links it statically), so ship it
# app-locally. Only VS 2022's (14.3x) is new enough for Qt's MSVC 2022 binaries.
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$crt = & $vswhere -latest -products * -find 'VC\Redist\MSVC\14.*\x64\Microsoft.VC14*.CRT' |
    Where-Object { $_ -match '\\14\.(3|4)\d' } | Sort-Object | Select-Object -Last 1
if ($crt) {
    Copy-Item "$crt\*.dll" $dist
} else {
    Write-Warning 'No VS 2022 MSVC runtime found: the package relies on the installed VC++ redistributable.'
}

$zip = "$root\fcs-qt\dist\$name.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path "$dist\*" -DestinationPath $zip
"Packaged $dist and $zip"
