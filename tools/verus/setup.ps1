$ErrorActionPreference = 'Stop'
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$cache = Join-Path $root 'cache/verus'
$version = '0.2026.11.22.1ce6a59'
$hash = '3e7d238871baf6fe90d4fa978ee2fdf99f2f4589b198e3eeb1f2ddc7038f08a5'
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
    & rustup toolchain install 1.100.0-x86_64-pc-windows-msvc --profile minimal --no-self-update
    if ($LASTEXITCODE -ne 0) { throw 'Verus Rust toolchain installation failed' }
} finally {
    $env:RUSTUP_HOME = $previousRustupHome
}
