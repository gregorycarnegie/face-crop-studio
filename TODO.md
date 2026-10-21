# Face Crop Studio — TODO

Completed work lives in `CHANGELOG.md` and the git history. Only open items belong here.

## GPU cross-platform validation

Every CI leg now has an adapter — WARP (D3D12) on Windows, Metal on macOS, and Mesa's
lavapipe (Vulkan) on Linux — and `FCS_REQUIRE_GPU=1` makes a missing one fail rather than
skip. That is correctness coverage of the shaders on three backends, run on software or
virtual devices. It says nothing about speed or driver quirks on real hardware.

- [ ] Test on real macOS hardware (Metal via wgpu).
- [ ] Test on real Linux hardware (Vulkan via wgpu).

## CUDA kernel optimisation

After the 2.3.0 work (memory pool, merged split-K heads, graph replay, GPU normalisation and
the shortlist decode), a 640-wide CUDA detection takes ~1.0 ms on an RTX 4090, and ~0.74 ms of
it is the kernels themselves. Every kernel is still one thread per output with no tiling, so
that is where the remaining time is.

- [ ] Profile the kernels per step (CUDA events around each launch; there is no Nsight on the
  dev machine). Last breakdown: pointwise convolutions ~0.41 ms, merged heads ~0.15 ms,
  depthwise ~0.13 ms, other 3x3 convolutions ~0.13 ms.
- [ ] Tile the pointwise (1x1) convolutions, which are matrix multiplies, through shared memory
  or registers. Both compilers have to accept the same `tools/cuda/kernels.rs` -- Rust's NVPTX
  target on Windows and cuda-oxide on Linux -- so check how each declares shared memory and
  block barriers before designing around them.
- [ ] Re-measure against WESL with `cargo run --release -p fcs-core --example cuda_comparison`
  (it uses mimalloc, like the application; the system allocator adds ~0.6 ms of page faults).

## Code signing (deferred)

- [ ] **Windows** — requires `CODE_SIGN_PFX` (base64-encoded PFX) and `CODE_SIGN_PASSWORD` as repository secrets. Signing is skipped silently if absent.
- [ ] **macOS** — requires five repo secrets: `APPLE_DEVELOPER_ID_CERT` (base64 of the .p12), `APPLE_DEVELOPER_ID_PASS` (.p12 password), `APPLE_DEVELOPER_ID_NAME` (keychain identity string), `APPLE_NOTARIZE_USER`/`APPLE_NOTARIZE_PASS` (Apple ID + app-specific password), and `APPLE_TEAM_ID`. No code changes needed.

## Competitive feature parity (from Face Crop Jet review)

- [x] **Watch-folder mode** — `fcs-cli --watch <dir>` monitors a directory and runs the existing batch crop/export path on images as they arrive. Shipped; see CHANGELOG.
