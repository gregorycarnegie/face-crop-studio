# Run cargo for fcs-qt with Qt on PATH and the dynamic CRT (see build.rs).
#   ./fcs-qt/dev.ps1 run          ./fcs-qt/dev.ps1 test          ./fcs-qt/dev.ps1 build --release
# Qt location: $env:QT_DIR, else the newest %USERPROFILE%\Qt\6.*\msvc2022_64.
param([Parameter(ValueFromRemainingArguments = $true)] [string[]] $CargoArgs)
$ErrorActionPreference = 'Stop'

$qt = $env:QT_DIR
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

if (-not $CargoArgs) { $CargoArgs = @('run') }
# From here, not --manifest-path: cargo reads .cargo/config.toml (fcs-qt's own target dir)
# relative to the working directory.
Push-Location $PSScriptRoot
try { cargo @CargoArgs } finally { Pop-Location }
exit $LASTEXITCODE
