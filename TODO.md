# Face Crop Studio — TODO

Completed work lives in `CHANGELOG.md` and the git history. Only open items belong here.

## GPU cross-platform validation

Every CI leg now has an adapter — WARP (D3D12) on Windows, Metal on macOS, and Mesa's
lavapipe (Vulkan) on Linux — and `FCS_REQUIRE_GPU=1` makes a missing one fail rather than
skip. That is correctness coverage of the shaders on three backends, run on software or
virtual devices. It says nothing about speed or driver quirks on real hardware.

- [ ] Test on real macOS hardware (Metal via wgpu). No real Mac is available; fcs-qt's macOS
  package ships labelled as a preview until someone checks it.

## Detector false positives off its home domain

On COCO (`docs/MODEL_COMPARISON.md`), SCRFD-80k fires 5–6x as often as RetinaFace or the
public SCRFD on photographs with no people, mostly on animal faces and round objects (clocks,
tomatoes, plates, wheels). At equal false-positive rates it finds the fewest faces of the
three. Every one of its 80,000 training photographs contains faces, so it has seen almost no
face-free scenes.

**Retraining stopped (2026-11-24); the shipped detector stays the default.** People point a
cropping app at faces, so fewer false positives buy little. Hard negatives alone lost large
faces; the shipped model's large-face recall turned out to be a lucky draw, and zoom crops
(`train_scrfd --zoom`) were the lever that matched it. Zoom 0.35 plus the balanced negatives
(ZLNZ35) missed the gate by 0.01 points on large faces, but in a crop test on 600 fresh photos,
at a threshold of 0.343 matched to shipped's false-positive rate, it halved missed group faces
(182 -> 92) and portrait failures (10 -> 5, one of them new). Full record:
`tools/dataset/HARD_NEGATIVES.md`.

- [x] Full 100-epoch zoom run (`ZL_full_seed0`): matches shipped on every shipped-relative gate.
- [x] Balanced negatives on top (`ZLN_full_seed0`), then zoom 0.35 (`ZLNZ35_full_seed0`): both
  fail 3 of 9 gates; the user's rule was one more run, then keep shipped.
- [x] Crop test, shipped vs ZLNZ35 through `fcs-cli --crop`.
- [ ] Next release candidate: ZLNZ35 at 0.343. Before changing the default, test it with shipped
  as a fallback when it returns no crop, on new fresh photos with the current build, and review
  every extra group crop for usability. Shipping it also means a default confidence of about
  0.34 and handling saved settings at 0.5. Only when the user resumes it.

## CUDA kernel optimisation

After the 2.3.0 work (memory pool, merged split-K heads, graph replay, GPU normalisation and
the shortlist decode), a 640-wide CUDA detection takes ~1.0 ms on an RTX 4090, and ~0.74 ms of
it is the kernels themselves. Every kernel is still one thread per output with no tiling, so
that is where the remaining time is.

**Low priority.** Even halving the pointwise time saves ~0.2 ms per photo, which disappears
next to decoding and encoding, on a backend that is opt-in.

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
