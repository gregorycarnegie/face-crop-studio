@echo off
setlocal

pushd "%~dp0" >nul || exit /b 1

for /f "usebackq delims=" %%V in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; $meta = cargo metadata --format-version 1 --no-deps | ConvertFrom-Json; $pkg = $meta.packages | Where-Object { $meta.workspace_members -contains $_.id } | Select-Object -First 1; if (-not $pkg) { throw 'No workspace package found in cargo metadata.' }; $pkg.version"`) do set "VERSION=%%V"

if not defined VERSION (
    echo error: could not read workspace package version from Cargo metadata.
    popd >nul
    exit /b 1
)

set "TAG=v%VERSION%"

for /f "delims=" %%S in ('git status --porcelain') do set "DIRTY=1"
if defined DIRTY (
    echo error: working tree is not clean.
    git status --short
    popd >nul
    exit /b 1
)

git rev-parse -q --verify "refs/tags/%TAG%" >nul
if not errorlevel 1 (
    echo error: tag %TAG% already exists.
    popd >nul
    exit /b 1
)

echo Release version: %VERSION%
echo Release tag:     %TAG%
echo.
set /p CONFIRM="Create and push this release tag? [y/N] "
if /i not "%CONFIRM%"=="y" (
    echo cancelled.
    popd >nul
    exit /b 1
)

git tag -a "%TAG%" -m "Release %TAG%" || (
    popd >nul
    exit /b 1
)

git push origin "%TAG%" || (
    popd >nul
    exit /b 1
)

echo.
echo Pushed %TAG%.
popd >nul
