<#
.SYNOPSIS
    Builds the Microsoft Store MSIX bundle from already-built Windows payloads.

.DESCRIPTION
    Mirrors installer/linux/build_linux.sh and installer/macos/build_macos.sh:
    the release workflow builds the binaries, this script does the packaging.

    Takes the per-architecture dist directories produced by the windows-release
    matrix, wraps each one in a .msix, bundles them into a single .msixbundle,
    and zips that into the .msixupload that `msstore publish` wants.

    Deliberately does NOT sign anything. The Store re-signs every package with
    the publisher's Store certificate, and signing here with a different
    certificate whose subject does not match <Identity Publisher> just makes
    signtool fail. The zip/MSI/NSIS artifacts remain the signed, sideloadable
    ones.

.PARAMETER Version
    Three-part product version, e.g. "1.5.1". A ".0" revision is appended
    because MSIX requires four parts and the Store requires the revision to be 0.

.PARAMETER PayloadRoot
    Directory containing one subdirectory per architecture, named with the same
    arch slugs the release workflow uses: x86_64 and/or arm64.

.PARAMETER OutDir
    Where the .msix, .msixbundle and .msixupload are written.

.EXAMPLE
    ./build_msix.ps1 -Version 1.5.1 -PayloadRoot dist/windows -OutDir dist/msix
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Version,
    [Parameter(Mandatory = $true)][string]$PayloadRoot,
    [string]$OutDir = "dist/msix",

    # Partner Center identity. Defaults are the committed production values;
    # override them to build a package for a different Store listing.
    [string]$IdentityName = $env:MSIX_IDENTITY_NAME,
    [string]$Publisher = $env:MSIX_PUBLISHER,
    [string]$PublisherDisplayName = $env:MSIX_PUBLISHER_DISPLAY_NAME
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# PLACEHOLDER — replace with the values from Partner Center > Product identity.
# Identity/Publisher must match byte-for-byte or the Store rejects the upload.
if (-not $IdentityName)         { $IdentityName = "REPLACE.WithPartnerCenterIdentityName" }
if (-not $Publisher)            { $Publisher = "CN=REPLACE-WITH-PARTNER-CENTER-PUBLISHER-GUID" }
if (-not $PublisherDisplayName) { $PublisherDisplayName = "REPLACE With Publisher Display Name" }

if ($Version -notmatch '^\d+\.\d+\.\d+$') {
    throw "Version must be three-part (x.y.z), got '$Version'"
}
$MsixVersion = "$Version.0"

$ScriptDir = $PSScriptRoot
$ManifestTemplate = Join-Path $ScriptDir "AppxManifest.xml.in"
# Committed rather than generated in CI. Rasterised from the largest rendition
# in fcs-gui/assets/app_icon.ico (app_logo.svg is 210x263, and MSIX tiles must
# be square). To regenerate after an icon change:
#   uvx --with pillow python -c "from PIL import Image; ico =
#   Image.open('fcs-gui/assets/app_icon.ico'); ico.size = max(ico.ico.sizes());
#   src = ico.convert('RGBA'); [src.resize((n, n), Image.LANCZOS).save(p) for p,
#   n in [('Square44x44Logo.png',44),
#   ('Square44x44Logo.targetsize-24_altform-unplated.png',24),
#   ('Square150x150Logo.png',150),('StoreLogo.png',50)]]"
$AssetsDir = Join-Path $ScriptDir "Assets"

foreach ($required in @($ManifestTemplate, $AssetsDir)) {
    if (-not (Test-Path $required)) { throw "Missing packaging input: $required" }
}

# ── Locate the Windows SDK tools ──────────────────────────────────────────────
# Same discovery shape as the signtool lookup in release.yml: newest SDK wins.
function Find-SdkTool([string]$Name) {
    $tool = Get-ChildItem "C:\Program Files (x86)\Windows Kits\10\bin" -Recurse -Filter $Name -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch '\\arm(64)?\\' } |
        Sort-Object FullName -Descending |
        Select-Object -First 1
    if (-not $tool) { throw "$Name not found in the Windows 10/11 SDK" }
    return $tool.FullName
}

$MakeAppx = Find-SdkTool "makeappx.exe"
$MakePri = Find-SdkTool "makepri.exe"
Write-Host "makeappx: $MakeAppx"
Write-Host "makepri:  $MakePri"

# MSIX ProcessorArchitecture values differ from the artifact naming slugs.
$ArchMap = @{ "x86_64" = "x64"; "arm64" = "arm64" }

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDirFull = (Resolve-Path $OutDir).Path
$StagingRoot = Join-Path ([IO.Path]::GetTempPath()) "fcs-msix-$([Guid]::NewGuid().ToString('N'))"
$BundleDir = Join-Path $StagingRoot "bundle"
New-Item -ItemType Directory -Force -Path $BundleDir | Out-Null

$built = @()

foreach ($slug in $ArchMap.Keys | Sort-Object) {
    $payload = Join-Path $PayloadRoot $slug
    if (-not (Test-Path $payload)) {
        Write-Host "Skipping $slug (no payload at $payload)"
        continue
    }

    $msixArch = $ArchMap[$slug]
    Write-Host "── Packaging $slug ($msixArch) ─────────────────────────────"

    # Stage a copy so the manifest/assets/PRI additions never touch the payload
    # that the zip and MSI are built from.
    $stage = Join-Path $StagingRoot $slug
    New-Item -ItemType Directory -Force -Path $stage | Out-Null
    Copy-Item -Recurse -Force "$payload\*" $stage

    # The .ico is for the MSI/NSIS shortcuts; MSIX uses the PNGs in Assets.
    Remove-Item (Join-Path $stage "app_icon.ico") -Force -ErrorAction SilentlyContinue
    Copy-Item -Recurse -Force $AssetsDir (Join-Path $stage "Assets")

    foreach ($exe in @("fcs-gui.exe", "fcs-cli.exe")) {
        if (-not (Test-Path (Join-Path $stage $exe))) {
            throw "Payload for $slug is missing $exe"
        }
    }

    (Get-Content -Raw $ManifestTemplate).
        Replace("@IDENTITY_NAME@", $IdentityName).
        Replace("@PUBLISHER@", $Publisher).
        Replace("@PUBLISHER_DISPLAY_NAME@", $PublisherDisplayName).
        Replace("@VERSION@", $MsixVersion).
        Replace("@ARCH@", $msixArch) |
        Set-Content -Encoding UTF8 (Join-Path $stage "AppxManifest.xml")

    # resources.pri. The app has no MRT resources, but <Resources> in the
    # manifest makes Store certification expect a PRI file, and generating one
    # is cheaper than arguing with the certification report.
    $priConfig = Join-Path $StagingRoot "priconfig-$slug.xml"
    & $MakePri createconfig /ConfigXml $priConfig /Default en-US /Overwrite
    if ($LASTEXITCODE -ne 0) { throw "makepri createconfig failed ($LASTEXITCODE)" }
    & $MakePri new /ProjectRoot $stage /ConfigXml $priConfig /OutputFile (Join-Path $stage "resources.pri") /Overwrite
    if ($LASTEXITCODE -ne 0) { throw "makepri new failed ($LASTEXITCODE)" }

    $msix = Join-Path $BundleDir "FaceCropStudio-$MsixVersion-$msixArch.msix"
    & $MakeAppx pack /d $stage /p $msix /overwrite
    if ($LASTEXITCODE -ne 0) { throw "makeappx pack failed for $slug ($LASTEXITCODE)" }

    $built += $msixArch
    Copy-Item $msix $OutDirFull -Force
}

if ($built.Count -eq 0) {
    throw "No architecture payloads found under $PayloadRoot (expected x86_64 and/or arm64)"
}
Write-Host "Packaged architectures: $($built -join ', ')"

# ── Bundle + .msixupload ──────────────────────────────────────────────────────
$Bundle = Join-Path $OutDirFull "FaceCropStudio-$MsixVersion.msixbundle"
& $MakeAppx bundle /d $BundleDir /p $Bundle /bv $MsixVersion /overwrite
if ($LASTEXITCODE -ne 0) { throw "makeappx bundle failed ($LASTEXITCODE)" }

# .msixupload is just a zip around the bundle; it is the format `msstore publish`
# accepts for multi-architecture submissions.
$Upload = Join-Path $OutDirFull "FaceCropStudio-$MsixVersion.msixupload"
Compress-Archive -Path $Bundle -DestinationPath $Upload -Force

Remove-Item -Recurse -Force $StagingRoot -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "Wrote:"
Get-ChildItem $OutDirFull | ForEach-Object { Write-Host ("  {0,-52} {1,10:N0} bytes" -f $_.Name, $_.Length) }
