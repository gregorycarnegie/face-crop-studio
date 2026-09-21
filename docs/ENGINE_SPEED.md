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

## Batch throughput, which is what the CLI actually does

Everything above is one image at a time. The real workload is a folder through rayon on a
shared ONNX Runtime session, and that is a different number. Measured 2026-09-21 on the
1,239-image reference folder (9.8 MP average), detect + crop + write, same machine:

| Workers | Wall clock | Per image | Speedup |
|---|---|---|---|
| 1 (`RAYON_NUM_THREADS=1`) | 54.82 s | 44.25 ms | 1.0x |
| 4 | 17.02 s | 13.74 ms | 3.2x |
| 8 | 11.48 s | 9.26 ms | 4.8x |
| **16** | **10.14 / 10.55 s** | **8.2 ms** | **5.4x** |
| 32 (rayon's default here) | 10.87 / 10.94 / 10.99 / 11.05 s | 8.8 ms | 5.0x |

114 images per second at the default, and **8.8 ms per image is throughput, not latency**: a
single detection of one of these images costs 6.67 ms end to end (2.53 ms preprocess, 3.75 ms
network), so the batch is doing roughly five images in the time one would take alone.

Crop, shape mask and write together add about **0.9 ms per image**: detection alone runs the
folder in 9.71 s against 10.87 s with cropping.

### Two things worth acting on

**16 workers now beat 32, which inverts the earlier measurement.** Experiment 60 found 32
workers at 7.85 s against 16 at 8.9 s and made rayon's default the deliberate choice; the note
recording it warned that "this number moved as soon as the work around it changed". It has.
16 wins both alternated pairs by 5-8%, and the regression is **not** the atomic write added in
2.0 -- with no crops written at all, 16 workers take 8.79 s against 32 at 9.73 s, so it is in
the detection path. The work that changed there is the detector itself: YuNet's preprocessing
and inference are gone, replaced by SCRFD's CPU resize and an ONNX Runtime session with four
intra-op threads per run.

The default is left alone rather than capped at 16 on the strength of a 5-8% difference on one
machine: `RAYON_NUM_THREADS` already overrides it, "physical cores" counts P and E cores alike,
and the comment in `fcs-cli/src/main.rs` explains why automatic capping was rejected before.
But the ranking in that comment and in the README is now wrong, and both say so.

**5.4x on 16 physical cores is 34% parallel efficiency.** Something serialises and this
measurement does not say what. Ruled out: file writes (above), and ONNX Runtime's intra-op
threads (experiment 69 measured the folder at 10.32 s against 10.18 s for 1 against 4 threads --
no difference, and my 10.87 s matches). Not ruled out, in rough order of suspicion: JPEG decode
and resize are memory-bandwidth bound and 1,239 x 9.8 MP is a lot of pixel traffic; the single
shared `fcs_ort::Session` may lock internally despite documenting concurrent `Run` as safe; and
the page cache may not be holding all ~1.2 GB of source images between runs.

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

- **Why** batch parallelism saturates at 5.4x; the section above lists what is ruled out.
- Whether the decode/NMS gap above is really the layout difference.
- The WGSL engine on anything but a 4090. It is the fallback path, so the machines that matter
  most for it are the ones without ONNX Runtime, and none were measured.
