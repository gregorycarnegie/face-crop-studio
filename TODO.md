# Face Crop Studio — TODO

Completed work lives in `CHANGELOG.md` and the git history. Only open items belong here.

## GPU cross-platform validation

CI now builds, tests, and lints on Windows, macOS, and Linux, but only the Windows
runner has a usable GPU adapter — the macOS and Linux legs cover the CPU paths and
let the GPU tests self-skip. Validating the wgpu backends needs real hardware, or a
software rasteriser wired into CI (`mesa-vulkan-drivers` on Linux), which risks
tripping the nextest slow-timeout on the GPU parity suite.

- [ ] Test on macOS (Metal via wgpu).
- [ ] Test on Linux (Vulkan via wgpu).

## Code signing (deferred)

- [ ] **Windows** — requires `CODE_SIGN_PFX` (base64-encoded PFX) and `CODE_SIGN_PASSWORD` as repository secrets. Signing is skipped silently if absent.
- [ ] **macOS** — requires five repo secrets: `APPLE_DEVELOPER_ID_CERT` (base64 of the .p12), `APPLE_DEVELOPER_ID_PASS` (.p12 password), `APPLE_DEVELOPER_ID_NAME` (keychain identity string), `APPLE_NOTARIZE_USER`/`APPLE_NOTARIZE_PASS` (Apple ID + app-specific password), and `APPLE_TEAM_ID`. No code changes needed.

## Competitive feature parity (from Face Crop Jet review)

- [ ] **Watch-folder mode** — add `--watch <dir>` to `fcs-cli` that monitors a directory and runs the existing batch crop/export path on new/changed image files. Use the `notify` crate as a thin event loop around the current batch workflow (no new processing logic). FCJ calls this "Robot/Directory Monitor" mode.
