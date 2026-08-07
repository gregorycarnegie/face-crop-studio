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
- `SHA256SUMS-windows-x86_64.txt`
- `face-crop-studio-<version>-arm64.AppImage` / `.deb` (Linux arm64)

There is one checksum file per platform and architecture. They are deliberately
not a single `SHA256SUMS.txt`: every release job uploads to the same release, so
a shared filename means the last job to finish silently overwrites the rest.

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

## Microsoft Store

The `msix-package` job bundles every architecture the `windows-release` matrix
produced into a `.msixbundle` and wraps it as a `.msixupload`, the format
`msstore publish` accepts. Today that is x86_64 only — Windows arm64 is blocked
upstream in `tract-linalg`, see the matrix comment in `release.yml`. The job
picks up architectures by artifact pattern rather than by name, so restoring the
arm64 leg needs no change here. It is
uploaded as a *workflow artifact only*, never a GitHub release asset: the bundle
is intentionally unsigned, because the Store re-signs every package with the
publisher certificate, and signing it here with a certificate whose subject does
not match `<Identity Publisher>` just makes signtool fail.

### One-time setup

The `store-submit` job is skipped until this is done, so a tag pushed before
setup still produces every other artifact normally.

1. **The first submission must be made by hand.** Microsoft's GitHub Action path
   only performs *updates*; the app has to already be published and live in the
   Store. Download the `msix-store-package` artifact from a release run and
   upload it through Partner Center for that first submission.
2. Only **free** products are supported by the automated update path. A paid
   listing has to keep going through Partner Center.
3. Copy the identity values from Partner Center → *Product identity* into
   `installer/windows/msix/build_msix.ps1` (or set them as the repository
   variables `MSIX_IDENTITY_NAME`, `MSIX_PUBLISHER`,
   `MSIX_PUBLISHER_DISPLAY_NAME`, which take precedence). `Publisher` is the
   full certificate subject, e.g. `CN=A1B2C3D4-....` These must match Partner
   Center byte-for-byte or the upload is rejected.
4. Add these repository **secrets** (Partner Center → Account settings, and the
   Microsoft Entra app registration with the *Manager* role):
   - `AZURE_AD_TENANT_ID`
   - `AZURE_AD_APPLICATION_CLIENT_ID`
   - `AZURE_AD_APPLICATION_SECRET`
   - `SELLER_ID`
5. Add the repository **variable** `MSSTORE_PRODUCT_ID` (the Store product ID).
   Setting it is what enables the `store-submit` job.

### Per-release

Nothing. Pushing a final `v*` tag builds the bundle and submits it.
Certification still runs on Microsoft's side afterwards — check Partner Center
for the result.

Release-candidate tags (`v1.5.1-rc1`) build the package but deliberately do not
submit it, so an RC is the right way to get a `.msixupload` for a manual
Partner Center upload. See the versioning note below for why RCs must not
submit.

### Versioning

MSIX versions are four-part with a forced `.0` revision (`1.5.1` → `1.5.1.0`);
the Store requires the revision to be 0. The Store also rejects a re-upload of a
version it has already seen, so a failed certification needs a new patch tag,
not a re-run of the same one.

The prerelease suffix is stripped, so `v1.5.1-rc1` and `v1.5.1` both produce
`1.5.1.0`. That is why `store-submit` skips prerelease tags: submitting an RC
would consume the version number the final release needs, and it could not be
reused.
