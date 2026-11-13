$ErrorActionPreference = 'Stop'
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$cache = Join-Path $root 'cache/verus'
$version = '0.2026.11.02.8bf68ee'
$hash = '221f471d5925c4b55362cbd20a1bb15333338962902c2283b3f2b4d4c8930456'
New-Item -ItemType Directory -Force $cache | Out-Null
$archive = Join-Path $cache 'verus.zip'
if (!(Test-Path $archive)) {
    Invoke-WebRequest "https://github.com/verus-lang/verus/releases/download/release/$version/verus-$version-x86-win.zip" -OutFile $archive
}
if ((Get-FileHash $archive -Algorithm SHA256).Hash -ne $hash) {
    throw "Verus archive checksum mismatch: $archive"
}
Expand-Archive -LiteralPath $archive -DestinationPath $cache -Force

$previousRustupHome = $env:RUSTUP_HOME
try {
    $env:RUSTUP_HOME = Join-Path $cache 'rustup'
    & rustup toolchain install 1.99.1-x86_64-pc-windows-msvc --profile minimal --no-self-update
    if ($LASTEXITCODE -ne 0) { throw 'Verus Rust toolchain installation failed' }
} finally {
    $env:RUSTUP_HOME = $previousRustupHome
}
