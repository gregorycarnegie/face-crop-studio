# What a detection costs, and on which engine

Measured 2026-09-21 on the 1.9.0 tree, RTX 4090 / DX12 / Ryzen 7950X, Windows 11, release
build. Reproduce with `cargo run --release -p fcs-core --example engine_speed -- <image>`
(add `--features fcs-utils/raw` for a camera RAW, or it silently decodes the embedded
thumbnail — which is how the first run of this measurement reported a 120×160 source).

Median of 15 runs after 3 warmups. Median rather than mean: one scheduling hiccup skews a
mean over 15 runs, and the question is what a typical detection costs.

## The GPU number was never measured honestly before

wgpu queues dispatches and returns. A timer stopped before the results are read measures
the driver accepting commands, not the work.

| Network only, 640×640 input | Median |
|---|---|
| ONNX Runtime | **3.67 ms** |
| WGSL, submit only — *not a detection* | 2.85 ms |
| WGSL, including readback | **3.78 ms** |
| Built-in CPU graph | 23.19 ms |

The submit-only figure reads as a **22% GPU win**. The honest one, which waits for the
readback that `ScrfdDetector::detect` must do before it can decode, is **3% slower than ONNX
Runtime**. The two engines are a tie, and every earlier GPU-inference claim in this repo was
measuring the queue.

Why there is no win: the model is 2.5 MB at a fixed 640×640. That is far too little
arithmetic to saturate a 4090, so the run is dominated by per-dispatch overhead and the
~0.9 ms readback. The WGSL engine exists so the app works without ONNX Runtime — which is
what let YuNet go — not because it is faster.

The built-in CPU graph is 6.3× slower than ONNX Runtime. That is the floor, and it is a
usable floor: 23 ms is still interactive for one image.

## Preprocessing is the largest single item, not the network

On a 4928×7380 (36.4 MP) Nikon NEF:

| Stage | Median |
|---|---|
| Preprocess (CPU resize to 640 + normalise) | **5.18 ms** |
| Network (ONNX Runtime) | 3.67 ms |
| End to end, `detect` on ONNX Runtime | 12.32 ms |
| End to end, `detect` on WGSL | 10.23 ms |

Preprocessing is **42% of a detection** on an image this size, and larger than the network.
The stages do not sum to the end-to-end figure — 5.18 + 3.67 = 8.85 against 12.32 — so
roughly 3.5 ms is decode and NMS over the 25,600 anchors. That gap is smaller on the WGSL
path (1.3 ms) for reasons this measurement does not explain; the two paths decode different
layouts (`decode` vs `decode_maps`), which is the obvious suspect and is not evidence.

Note the GUI reported 42 ms for this same image. That is a single cold measurement taken
right after load, against a warm median here. Plausibly arena allocation and page faults on
the first run, but that is a guess, not something measured.

## GPU preprocessing cannot help, and was already known not to

The resize is a CPU cost, so moving it to the GPU is the obvious idea. It does not work, and
the repo had already measured why (experiment 10, `WgpuPreprocessor::upload_pays_for_source`):
preprocessing uploads the source at full resolution, so the transfer grows with the image
while the win — skipping a 4.9 MB round trip of the 640×640 tensor — does not. The crossover
sat between **1.1 and 2.5 MP**. Above it the GPU route loses, on a discrete card and an iGPU
alike.

So `WgpuPreprocessor` carried a pixel cutoff and fell back to the CPU above it. On this 36.4 MP
NEF the fallback is what runs, which is why `--benchmark-preprocess` printed `cpu_resize`
traces under its `gpu:` label and a time within noise of the CPU path (5.14 vs 6.62 ms avg).

The 5.18 ms is already the SIMD path (`fast_image_resize`), about 7 GP/s. There is no cheap
win left there, which is why the GPU preprocessing machinery was deleted in 1.9.0 rather than
wired up to SCRFD: it could only ever have helped under ~2 MP, where the network dominates
anyway.

## What this settled

- **`gpu.inference` deleted** rather than made to work. It had been inert since SCRFD landed;
  the measurement says there is no win to expose.
- **`gpu.preprocessing`, `WgpuPreprocessor` and `fcs-core::preprocess` deleted.** Nothing had
  called them since the detector started letterboxing internally, and they cannot beat the CPU
  above 2 MP.
- **The `input` settings section deleted** (`width`, `height`, `resize_quality`). The export
  fixes the input at 640×640, so the dimensions cannot be chosen. `resize_quality` is a real
  trade in principle — preprocessing is 42% of a detection — but its quality cost was measured
  against YuNet's preprocessing path (experiment 54: 1.76× faster end to end, losing 2 of 51
  faces), and **is unmeasured against SCRFD**. Re-adding it needs that measurement first, on
  the corpus, not a plausible argument.

## Not measured

- Batch throughput. Every figure here is one image at a time; the CLI runs rayon over a folder
  and shares one ONNX Runtime session, so per-image cost under contention is a different
  number.
- Whether the decode/NMS gap above is really the layout difference.
- The WGSL engine on anything but a 4090. It is the fallback path, so the machines that matter
  most for it are the ones without ONNX Runtime, and none were measured.
