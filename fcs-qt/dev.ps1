# Run cargo for fcs-qt with Qt on PATH and the dynamic CRT (see build.rs).
#   ./fcs-qt/dev.ps1 run          ./fcs-qt/dev.ps1 test          ./fcs-qt/dev.ps1 build --release
#   ./fcs-qt/dev.ps1 run '--' samples      (quote the --: PowerShell swallows a bare one)
# Qt location: $env:QT_DIR, else $env:QT_ROOT_DIR, else the newest %USERPROFILE%\Qt\6.*\msvc2022_64.
# $args rather than a param() block: a param block rejects `pwsh -File dev.ps1 run -- x`.
$CargoArgs = $args
$ErrorActionPreference = 'Stop'

$qt = $env:QT_DIR
if (-not $qt) { $qt = $env:QT_ROOT_DIR }  # set by install-qt-action in CI
if (-not $qt) {
    $qt = Get-ChildItem "$env:USERPROFILE\Qt\6.*\msvc2022_64" -Directory -ErrorAction SilentlyContinue |
        Sort-Object { [version]$_.Parent.Name } | Select-Object -Last 1 -ExpandProperty FullName
}
if (-not $qt -or -not (Test-Path "$qt\bin\qmake.exe")) {
    throw "Qt 6.10+ (MSVC 64-bit) not found. Set QT_DIR to e.g. C:\Qt\6.10.3\msvc2022_64."
}
$env:PATH = "$qt\bin;$env:PATH"
# Replaces the workspace's rustflags, dropping +crt-static but keeping its CPU baseline.
$env:RUSTFLAGS = '-C target-cpu=x86-64-v3'
# C libraries to match: static libs built on the dynamic CRT. libheif-sys's vcpkg lookup picks
# this triplet by itself once +crt-static is gone; dav1d (AVIF) goes through pkg-config.
$vcpkg = if ($env:VCPKG_ROOT) { $env:VCPKG_ROOT } else { 'C:\vcpkg' }
$env:VCPKG_ROOT = $vcpkg
$env:PKG_CONFIG_PATH = "$vcpkg\installed\x64-windows-static-md\lib\pkgconfig"

if (-not $CargoArgs) { $CargoArgs = @('run') }
# Run from the repo root: the app finds models/ relative to its working directory, as fcs-gui
# does under `cargo run`. Cargo would then read only the workspace's .cargo/config.toml, so
# fcs-qt's own target dir and test environment are passed here instead.
$env:LIBHEIF_PLUGIN_PATH = "$PSScriptRoot\target\libheif-plugins"  # for tests; see process_env
$command = $CargoArgs[0]
# Always an array: with one extra argument a destructured $rest is a string, and @rest would
# then splat it character by character.
$rest = @($CargoArgs | Select-Object -Skip 1)
Push-Location "$PSScriptRoot\.."
try {
    # --manifest-path right after the subcommand, so it never lands after a `--`.
    cargo --config "build.target-dir='$PSScriptRoot\target'" $command `
        --manifest-path "$PSScriptRoot\Cargo.toml" @rest
} finally { Pop-Location }
exit $LASTEXITCODE
