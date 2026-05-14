# Release Runbook (Windows-first)

This runbook covers the release flow used by `.github/workflows/release.yml`.

## 1) Prepare default branch

1. Ensure CI is green on `master` (or your repo default branch).
2. Confirm release notes are updated in `docs/releases/v1.0.0.md`.
3. Confirm `README.md` installation section is current.

## 2) Cut a release candidate

```bash
git checkout master
git pull --ff-only
git tag v1.0.0-rc2
git push origin v1.0.0-rc2
```

This triggers the tag-based release workflow and publishes Windows assets for validation.

## 3) Validate release artefacts

From GitHub Release assets:

- `face-crop-studio-windows-x86_64.msi`
- `face-crop-studio-windows-x86_64-setup.exe`
- `face-crop-studio-windows-x86_64.zip`
- `SHA256SUMS.txt`

Validate checksum(s) in PowerShell:

```powershell
Get-FileHash .\face-crop-studio-windows-x86_64.msi -Algorithm SHA256
Get-FileHash .\face-crop-studio-windows-x86_64-setup.exe -Algorithm SHA256
Get-FileHash .\face-crop-studio-windows-x86_64.zip -Algorithm SHA256
```

Confirm executables launch:

- `fcs-cli.exe --help`
- `fcs-gui.exe`

## 4) Publish final v1.0.0

If RC artefacts are good:

```bash
git tag v1.0.0
git push origin v1.0.0
```

Then publish/edit release notes using `docs/releases/v1.0.0.md`.

## 5) Post-release

1. Mark release checklist items complete in `TODO.md`.
2. If needed, create hotfix tag `v1.0.1`.
