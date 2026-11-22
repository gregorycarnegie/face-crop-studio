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

**Resumed (2026-11-20), low priority.** People point a cropping app at faces, so fewer false
positives buy little, and the shipped detector stays until a retrain passes its gate. Two
hard-negative retrains cut COCO's false positives (0.030 -> 0.005 per no-person photo) but lost
large faces. The native trainer matches MMDetection and the shipped model's large-face recall is
a lucky draw, so any retrain has to target large faces. Zoom crops plus a higher learning rate
(`train_scrfd --zoom 0.25 --lr 0.02`) win the 10-epoch screen, but at 30 epochs the outcome
still depends on the seed (one of two seeds took). A good seed, decayed at 35 epochs, reaches
82.6 large / 87.7 COCO / 77.3 OI against shipped's 86.5 / 88.9 / 78.2. Full record:
`tools/dataset/HARD_NEGATIVES.md`.

- [x] Full 100-epoch zoom run (`ZL_full_seed0`): matches shipped on every shipped-relative gate,
  +1.3 OI recall. Kept as the base; the shipped model stays.
- [ ] Add the balanced negatives to it: seeds 0-2 screening at 10 epochs, best to 100 epochs and
  the gate (`zig_runs\proxy\chain_neg.sh`).

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
