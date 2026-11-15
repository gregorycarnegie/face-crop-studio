# Writes the NSIS uninstall commands for one payload: a Delete per file `File /r` installed,
# then a non-recursive RMDir per directory, deepest first, so files a user added survive.
# face-crop-studio.nsi runs this at compile time through !system.
param(
    [Parameter(Mandatory)][string]$DistDir,
    [Parameter(Mandatory)][string]$OutFile
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath $DistDir).Path.TrimEnd('\')
# `$$` is NSIS's literal dollar sign.
$relative = { param($path) $path.Substring($root.Length + 1).Replace('$', '$$') }

$lines = @(Get-ChildItem -LiteralPath $root -Recurse -File -Force |
    ForEach-Object { "Delete `"`$INSTDIR\$(& $relative $_.FullName)`"" })
$lines += @(Get-ChildItem -LiteralPath $root -Recurse -Directory -Force |
    Sort-Object { $_.FullName.Length } -Descending |
    ForEach-Object { "RMDir `"`$INSTDIR\$(& $relative $_.FullName)`"" })
if (-not $lines) { throw "no files under $DistDir" }
# NSIS 3 reads a BOM'd include as UTF-8, so non-ASCII file names survive.
[IO.File]::WriteAllLines($OutFile, [string[]]$lines, [Text.UTF8Encoding]::new($true))
