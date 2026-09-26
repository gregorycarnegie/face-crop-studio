# Face Crop Studio — TODO

Completed work lives in `CHANGELOG.md` and the git history. Only open items belong here.

## GPU cross-platform validation

Every CI leg now has an adapter — WARP (D3D12) on Windows, Metal on macOS, and Mesa's
lavapipe (Vulkan) on Linux — and `FCS_REQUIRE_GPU=1` makes a missing one fail rather than
skip. That is correctness coverage of the shaders on three backends, run on software or
virtual devices. It says nothing about speed or driver quirks on real hardware.

- [ ] Test on real macOS hardware (Metal via wgpu).
- [ ] Test on real Linux hardware (Vulkan via wgpu).

## Code signing (deferred)

- [ ] **Windows** — requires `CODE_SIGN_PFX` (base64-encoded PFX) and `CODE_SIGN_PASSWORD` as repository secrets. Signing is skipped silently if absent.
- [ ] **macOS** — requires five repo secrets: `APPLE_DEVELOPER_ID_CERT` (base64 of the .p12), `APPLE_DEVELOPER_ID_PASS` (.p12 password), `APPLE_DEVELOPER_ID_NAME` (keychain identity string), `APPLE_NOTARIZE_USER`/`APPLE_NOTARIZE_PASS` (Apple ID + app-specific password), and `APPLE_TEAM_ID`. No code changes needed.

## Competitive feature parity (from Face Crop Jet review)

- [x] **Watch-folder mode** — `fcs-cli --watch <dir>` monitors a directory and runs the existing batch crop/export path on images as they arrive. Shipped; see CHANGELOG.
