# Performance Analysis & Optimization Guide

> **Historical (pre-1.9).** Everything below measures or designs around **YuNet**, which was the
> detector until 1.8.0 and no longer ships: its weights come from WIDER FACE, licensed for
> non-commercial academic research only. The engines it describes still exist and still run
> SCRFD; the numbers, node names and topology do not describe anything current, and the examples
> named here were deleted with it. The full YuNet implementation and these experiments are in the
> `face-crop-studio-yunet-archive` fork. Kept as a design and measurement record.

## Current Performance Profile

### Where the GPU graph stands

The YuNet graph runs in **0.366 ms of GPU compute** on RTX 4090 / D3D12 / FXC and **10.51 ms**
on the Ryzen 9 7950X's integrated Radeon, with output bit-identical to the original graph
(decoded-output fingerprint `0xa116e42f7c2dabdb`). The first shader round (experiments 0-4,
baseline `c332075`) took it from 0.910 to 0.536 ms with pointwise specialisation, depthwise
input reuse and a four-channel pointwise tile. Later rounds fused the head branches (37),
tiled the stem (34), fused the neck's upsample-and-add (36) and rewrote loop-indexed local
arrays as registers (41). FP16 and subgroups were measured twice and rejected both times
(4, 43-46).

That first round's whole-detection A/B/B/A estimates -- **3.561 / 3.504 / 3.495 / 3.463 ms**,
Criterion `inference_pipeline/detect_image/gpu`, CPU speed resize plus GPU inference, decode
excluded -- established **no reliable wall-time gain**. Later results are measured per section
below; the [GPU experiment results](#gpu-experiment-results) keep the first round's detail.

Every experiment has a stable ID, and code comments cite those IDs: the
[experiment index](#experiment-index) records what each one found. Finishing that list does
**not** mean the performance search is exhausted; see [Open questions](#open-questions).

### Where warm detection actually spends its time (2026-09-05, later round)

Phase timings from `cargo run --release -p fcs-core --example phase_timings`,
RTX 4090 / D3D12, warm, single request. Indentation is containment; children
must not be summed with their parent, and `readback_wait` **contains** the
forward pass rather than idling beside it.

Two different paths, chosen by source size. At the time (experiment 5)
`upload_pays_for_source` declined above 2.5 MP, so a large photo preprocessed
entirely on the CPU and a small one did not. Since 50 a large photo is resized on
the CPU and converted on the GPU, and since 54 the cutoff is 1.75 MP.

| Phase | 0.17 MP | 10 MP |
| --- | ---: | ---: |
| detect_image | 1.57 | 4.68 |
| - CPU preprocess (large only) | - | **2.65** |
| - on-device preprocess (small only) | ~0.21 | - |
| - inference | 1.35 | 1.97 |
| - - record (host, before submit) | 0.21 -> **0.10** | 0.28 |
| - - finish + submit | 0.13 | 0.16 |
| - - readback (incl. GPU execution) | 0.79 -> **0.49** | 1.04 |
| - - CHW to HWC + sigmoid | 0.11 | 0.11 |
| - - decode | 0.19 | 0.19 |

**Host recording is now 0.10 ms, not 0.21.** The four max-pool, two resize and
two add dispatches created a fresh uniform buffer each, and
`create_buffer_init` costs **8.1 us** on this device -- ten times a
`create_bind_group`, for sixteen bytes. Caching them the way convolution already
did cut `gpu_record` 43% and small-image `detect_image` 10%, bit-exact
(experiment 20). `examples/encode_cost.rs` prices both objects, and it is also
the reason bind-group caching (22) was not written: all 61 of them are 0.049 ms.

**Large-image detection is a preprocessing problem, not a shader problem.**
The ~0.54 ms of profiled GPU compute is a minority of even the small-image
case, and on a 10 MP source CPU preprocessing alone is 57% of the whole
detection.

**Host work between the inference submit and the readback wait is free.** The
GPU is busy through that window, so shortening it does not shorten detection:
removing the staging allocation shed 0.039 ms and the wait grew by 0.046 ms.
Moving work *into* that window helps, and moving work out of it hurts -
encoding the head copies with inference cost 0.09 ms. The recoverable time is
before the submit, in the GPU work itself, and after the wait.

**Measurement caveats that invalidated earlier attempts.** Cross-process
comparison cannot resolve anything at this scale: repeated identical GPU runs
drift +/-0.08 ms as clocks ramp, and CPU throughput on this machine moved by
**1.6x between two builds of identical code**. Both are handled by alternating
variants inside one warm process (`phase_timings --ab VAR`), whose A/A control
sits within +/-0.003 ms per phase.

Repeated across independent processes (experiment 7), the smallest readable
effect is **0.01 ms** of wall time on a small image and **~0.1 ms** on a 10 MP one,
where the CPU resize alone moves +/-0.03-0.05 ms while the GPU-side phases in the
same run stay within 0.001. GPU timestamp totals agree to one 1.024 us tick per
family across processes. A p95 from 30 samples moved 0.89-1.20 ms between five
identical processes, so tails need hundreds of samples before they mean anything.

### On an integrated GPU the shader work is the detection

Everything above was measured on an RTX 4090. The 7950X's integrated Radeon (experiment 10)
runs the same graph, with identical output, in **11.35 ms of GPU compute** -- 92% of
`run_on_device`, against 35% on the 4090. The findings that the dispatches are latency-bound
and that shader arithmetic is the wrong lever (8, 26, 27) are about the 4090; on this adapter
the arithmetic is the detection.

It also exposed a routing rule that had only been true on the first adapter. Integrated GPUs
were exempt from the 1.75 MP preprocessing cutoff, on the reasoning that with no bus the
upload is free. The upload was not the cost: the whole-source route has the GPU read every
source pixel through the resize shader.

| Source, iGPU | GPU route | CPU resize route |
| --- | ---: | ---: |
| 0.8 MP | 12.98 ms | 12.67 ms |
| 4 MP | 20.45 | 13.74 |
| 10 MP | **42.46** | **14.30** |
| 1239-image folder | 55.4, 53.2 s | **26.7, 26.5 s** |

The exemption is gone; one cutoff applies everywhere. ONNX Runtime on the same CPU runs
inference in 2.86 ms against the iGPU's 11.93, but GPU inference stays the default: one
16-core desktop and the weakest current iGPU are not enough to reorder a laptop.

### The DXC regression was a local array

Experiment 4 found DXC running the pointwise tile at 67.6 us where FXC took 27.6, and a
whole-context compiler switch was ruled out on the strength of it. Dumping naga's HLSL and
compiling it with both SDK compilers (experiment 41) located the cause: an accumulator array
updated inside a short loop -- `array<vec4<f32>, 4>` in the pointwise and stem tiles, and the
depthwise kernel's `array<f32, 6>` row -- becomes stack memory under DXC, with a load and a store
per multiply-add. Written as registers, bit-exact:

| GPU compute | FXC | DXC |
| --- | ---: | ---: |
| 4090, before | 0.367 ms | 0.696 ms |
| 4090, after | **0.366** | **0.375** |
| Radeon iGPU, before | ~11.3 | 18.26 |
| Radeon iGPU, after | **10.51** | **10.27** |

FXC ships, so production speed on the 4090 does not move; the iGPU gains under both. What
changed is that the two compilers are now at parity, so DXC-only features (subgroups, f16) are
no longer ruled out by the compiler itself.

### A threshold edit no longer rebuilds the detector

Score threshold, NMS and top-k are applied after inference, but NMS and top-k edits rebuilt the
whole detector and re-decoded the file, and the inspector's confidence slider changed a setting
the detector never saw (experiment 66). `YuNetDetector::with_postprocess` shares the loaded model;
an edit now re-detects the in-memory image. Per edit on a 10 MP image: **6.2 ms on the UI thread
plus 21.5 ms of work before, 3.0 ms of work after**, and the slider takes effect.

### Live webcam detection, and the cost of sharing a device with the renderer

Detection now runs on every webcam frame in the GUI, tracking **95% of frames at
15 fps** with detection at 6.9 ms against a 69 ms interval.

Two numbers a CLI probe could not produce (experiment 97). The per-frame texture
upload is **1.8 us**, because egui queues it rather than performing it -- that was
the risk and it is free. And detection costs **6.89 ms in the GUI against 3.5 ms
standalone**: sharing eframe's device with the renderer roughly doubles it. Still
10% of the frame interval, so it moves the margin rather than the answer.

One detection is in flight at a time; a frame arriving during one is dropped
rather than queued, so the overlay tracks the picture instead of trailing it.

### The webcam loop is capture-bound, and the detector is shown a squashed face

A C920 frame costs 35-44 ms of `webcam_grab` -- blocking until the camera
produces the next frame -- against 0.8-4.3 ms of MJPEG decode and 2.5-3.6 ms of
detection. Cutting the work from 8.6 ms to 3.3 moved the frame rate by 0.6 fps,
because the pipeline is idle about 90% of each frame. That retires experiments 53
and 65 for this hardware: neither can raise a frame rate that the camera sets.

Two defects fell out (experiments 95 and 96):

- **`--webcam-width`/`--webcam-height` did nothing.** The camera was opened with
  `AbsoluteHighestResolution` and then asked to `set_resolution`, which does not
  take, so a C920 asked for 640x480 delivered 1920x1080. Now fixed.
- **The preprocessor stretched to 640x640 rather than letterboxing**, so every
  non-square source reached the model distorted and scored lower. Over all 1239
  corpus images at production's 0.8 threshold, letterboxing takes 1030 faces to
  1130: **77 images gain a detection, 19 lose one**, concentrated on 16:9. Every
  box on a non-square source also moves -- median IoU 0.76-0.87, landmarks 34-46
  source pixels -- and which set is more correct is not decidable from counts, so
  a folder was cropped both ways and the crops compared. **Adopted**; see below.
  An earlier version of this note quoted much larger gains taken at the library
  default threshold of 0.9 rather than production's 0.8; those measured a
  configuration nobody runs.

### Peak memory scales with worker count, and nothing else does

GPU retention is flat: 44.1 MB across 400 sources up to 23.4 MP, largest first
(`examples/memory_growth.rs`). The buffer pool has an idle ceiling, the conv
caches are keyed by a graph production runs only at 640x640 and are cleared past
512 entries, and the preprocessor's texture is
bounded by the 1.75 MP upload gate, on every adapter since experiment 10 removed the
integrated-GPU exemption.

Host memory is the one that moves. Peak working set over the 1239-image folder:

| Workers | Wall s | Peak RSS |
| ---: | ---: | ---: |
| 8 | 10.61 | 1.06 GB |
| 16 | 7.84 | 1.79 GB |
| 32 (default here) | **7.36** | **3.18 GB** |
| 64 | 8.02 | 5.43 GB |

About **85 MB per worker**, while wall time flattens after 16: 16 to 32 buys 6%
for 78% more memory. The default is one worker per logical processor, so the bill
is set by core count. This adds the axis experiment 60 did not measure rather
than overturning it -- 60's speed ranking still holds. No cap has been applied;
see experiment 84 for why.

### One entry point per kernel halves shader compilation

`conv2d.wgsl` carried four kernels behind one `main` that branches on the
uniforms, so every launch compiled all four: ~197 ms. Compilation is superlinear
in what a single entry point can reach -- the three kernels YuNet dispatches cost
about 120 ms apart and 205 ms together -- and naga and FXC emit only what each
entry point reaches, so **one module with four entry points** gets the win without
splitting the file.

`compile_conv2d` falls to **~120 ms**, runtime is unchanged (GPU compute 0.372 ms,
detection wall unchanged) and output is bit-exact. `main` remains as the grouped
fallback and is built on first use, so production never compiles it
(experiment 82).

`Features::PIPELINE_CACHE` -- the obvious way to persist compiled pipelines --
is exposed on Vulkan but not D3D12 (`examples/adapter_cost.rs`), so it is not
available on the backend the app ships on for Windows.

### Cold start is 900 ms, and model loading is 0.15% of it

Launch to first face is 850-1213 ms on RTX 4090 / D3D12, and two stages own 94%
of it: `request_adapter` at 546 ms and compiling `conv2d.wgsl` at 197 ms. Parsing
the ONNX and uploading every weight is **1.4 ms** -- `benchmark_model_load.rs`
measures a number that has never been the startup cost. The first detection is
1.6 ms against a steady 0.74, so nothing meaningful is deferred into it
(experiment 80, `examples/cold_start.rs`).

The other four shaders compile in 20 ms together, so compiling in parallel would
win almost nothing; it is one shader.

Vulkan brings an adapter up in 278-312 ms against D3D12's 578-668
(`examples/adapter_cost.rs`), but `platform_safe_backends` excludes it on Windows
because Intel's ICD crashes during bring-up. That is a stability decision with a
measured price, not an oversight.

The price has both sides now (experiment 42). Measured within `cold_start`, Vulkan's
`request_adapter` is 6-10 ms against 504-807, and after its first launch the NVIDIA
driver's own pipeline cache answers `conv2d` compilation in 1.8 ms against FXC's
~120 ms every time -- a warm Vulkan launch reaches its first face in 293-354 ms against
732-1215. But D3D12 runs the graph in **0.370 ms of GPU compute against 0.501** on the
4090, and 11.3 against 19.9 ms on the Radeon iGPU, with identical output on all four.
The backend is per instance, so the two cannot be combined.

**The GUI pays a different bill**, and it is smaller than 80 recorded on both
counts. It shares eframe's device, so `App::new` never issues that 546 ms call --
though eframe issues one of its own to build the window, so the user still waits
for it. Its own cost is the shader compilation, which 82 took from ~235 ms to
150-160.

Launch to first painted frame is **894 ms**, measured in the running application
over 12 launches, and `build_detector` was 17% of it because it ran before the
first frame. Building it on a thread instead takes that to **737 ms** while the
detector still becomes usable at the same wall time -- 896 ms against 894, since
the build now overlaps the renderer's first frames rather than preceding them
(experiment 81; the `startup:` lines in the GUI log are the measurement).

### The decode was three-quarters exponentials

`decode_yunet_outputs_with` decoded all 8400 cells at 0.077 ms, and
`apply_postprocess` then discarded nearly every one. `examples/decode_cost.rs`
shows why an early-out pays: the gathers and writes are 0.021 ms of it and the
rest is four exponentials and a square root per cell.

The test is exact. `score^2` equals `s(cls) * s(obj)`, which is at most
`min(s(cls), s(obj))`, and both sigmoid and sqrt are monotonic, so a cell whose
smaller logit is below `logit(threshold^2)` cannot reach the threshold.
**`gpu_decode` 0.077 to 0.014 ms**, whole detection 0.875 to 0.753
(experiment 93).

Only `run_on_device_filtered` gates, and only the detector calls it, because it is
what knows the threshold. Everything else still decodes in full, which keeps
`readback_parity` a complete check of the decode arithmetic rather than of the
threshold. The invariant is guarded by a test instead: no row above the threshold
may change, and the gate may not manufacture one.

### Host encoding costs 3.1 us per dispatch, and most of it was avoidable

`gpu_submit` turned out to be two different things: splitting it shows
**`gpu_finish` is 0.059 ms** -- wgpu turning the recorded pass into backend
commands -- against 0.036 for the submit itself. So host encoding is `record`
plus `finish`, 0.134 ms over 43 dispatches, **3.1 us each**. That is the number
that makes dispatch-count reductions worth more than their GPU time alone, and it
is why experiment 37's wall gain exceeded its GPU saving.

`create_bind_group` is 0.8 us of that, and the convolutions build 34 of them per
inference. Caching them keyed on the five buffers they bind takes recording from
0.073 to 0.029 ms and a detection from 0.856 to 0.805 -- five alternated pairs,
no overlap, bit-exact (experiment 22). It works only because the buffer pool
settles into handing the same intermediates to the same layers, which nothing in
the pool promises, so a test asserts the hit rate holds.

### The stem was reading its source sixteen times

`conv2d/general` -- the 640x640 3->16 stride-2 stem, one dispatch -- was 42 us,
10.6% of GPU compute after the head fusion above. The cause was not its loops: the
general path computes **one output channel per thread**, so all 16 channels
gathered the same 27 input values independently. An ungrouped general convolution
has the same property pointwise does, that every output channel gathers the same
inputs, so it takes the same four-channel tile: **42.0 -> 18.4 us**, 5% off the
graph, bit-exact. Grouped convolutions keep one channel per thread, where `oc + j`
can cross a group boundary. See experiment 34.

The host predicate and `main` in `conv2d.wgsl` now spell out all three path
conditions side by side, because the host sizes dispatch z for whichever path the
shader will pick and a disagreement silently drops three quarters of the output
channels.

### Four head branches are one convolution

Each detection level ran cls, obj, bbox and kps as four separate branches, each a
1x1 convolution from the shared feature map followed by a per-channel 3x3. They
differ only in output channels -- 1, 1, 4, 10 -- and both halves concatenate
along that axis: a pointwise output channel depends only on its own row of
weights, a depthwise channel only on its own 3x3 kernel. Concatenating the four
sets of weights once at upload turns eight dispatches per level into two.

| | Dispatches | GPU compute | `readback_wait` |
| --- | ---: | ---: | ---: |
| before | 61 | 0.537 ms | 0.435-0.469 ms |
| after | **43** | **0.396 ms** | **0.361-0.375 ms** |

**26% off the graph**, bit-exact (`0xa116e42f7c2dabdb` unchanged), and about
0.1 ms off small-image `detect_image`. Two side effects: twelve per-model weight
buffers stopped being uploaded, and head readback went from 12 staging buffers to
3. A folder job cannot see it -- 0.09 ms per image is 0.11 s over 1239 against a
1 s spread -- so this lands on interactive and webcam-sized work. See experiment
37.

### The shaders are not compute-bound, so shader arithmetic is the wrong lever

`examples/pass_overhead.rs` records 26 identical dispatches as 26 timestamped
passes and then as one, and settles two things at once.

A pass boundary costs **0.83 us**, so `gpu_pass_breakdown`'s 61 passes carry
about 51 us of profiling and its 0.538 ms is really 0.487 ms of work. The
breakdown is trustworthy (experiment 8).

The per-dispatch floor is **1.85 us**, not the ~12 us most pointwise layers cost,
so those layers are not paying overhead -- they have no parallelism to hide their
latency behind:

| Pointwise layer | Arithmetic | Per dispatch | Workgroups |
| --- | ---: | ---: | ---: |
| 160x160 64->64 | 1x | 26.3 us | 1600 |
| 80x80 64->64 | 1/4 | 12.8 us | 480 |
| 20x20 64->64 | 1/64 | 10.7 us | 48 |

Four pixels and four output channels per thread leaves a 20x20x64 output with 48
workgroups on a 128-SM adapter: 0.3 TFLOPS on a part that does eighty. **Tuning
the arithmetic cannot help a kernel that is 0.4% utilised.**

The obvious reading -- give it more workgroups -- was then tested and is wrong.
One output channel per thread quadruples the 20x20 layer's workgroups and runs
8% slower, because four channels share each loaded input vector and dropping to
one quadruples the loads per multiply-add (experiment 27). Eight channels wins
15-19% on the three largest layers and loses the whole graph by 23%, because
YuNet's pointwise work is mostly small layers. Eight pixels per thread loses
everywhere. **Both tile axes are at a local optimum**, and the remaining shader
lever is graph structure -- fewer, larger dispatches -- not geometry.

Consistent with that, a sweep of nine workgroup shapes against the production
8x8 found nothing worth adopting (experiment 26): every variant is level or
worse on the expensive 160x160 layer, narrow-x shapes lose 26-69%, and the one
reproducible win is a single 1.024 us tick on one 320x320 dispatch. The harness
A/A control reads exactly 0.0%, which is what makes a one-tick difference
readable at all.

### The biggest saving found is switching cropping off the GPU

`GpuBatchCropper::crop` converts the full-resolution source to RGBA, packs every
pixel to `u32` and uploads all of it -- 40 MB for a 10 MP photo -- to produce one
512x512 crop. It accounts for 11% of batch CPU through `memcpy` alone. Forcing
the CPU path takes a 1239-image folder from a median **17.25 s to 9.8 s, 43%**,
order-alternated and winning every pair.

Adopted, and `GpuBatchCropper` deleted with it -- about 740 lines. The saving
holds for the shaped crops too (38% rectangle, 47% koch snowflake, 38% star),
since the mask costs the same on both sides and lands on the finished 512x512
image. Crops changed: `crop.wgsl` sampled a fixed 2x2 neighbourhood where the
CPU uses Lanczos3, so 18% of crops shifted quality label and one image in 171
selects a different face. See experiment 71.

### The batch path is not serialised

Profiled over 1239 real images, all threads: **194.7 s of CPU across 51 threads
in about 16 s of wall time**, roughly 12.5 cores busy on a 16-core machine, with
the top ten workers within 15% of each other. Nothing -- not the GPU queue, not
the buffer-pool mutex, not export -- serialises the batch, so experiments 60, 61
and 64 have no contention to fix. Warm throughput is about 65 images/s.

Two hazards for anyone measuring this path. `fcs-cli` loads
`config/gui_settings.json` **from the working directory**, so the same command
detects 1020 faces from the repository root and 423 from `target/release`;
always pass `--config`. And batch wall time swings about 10% run to run, so
comparisons need their order alternated -- see experiment 63, where an
unalternated pair produced a confident conclusion that reversed under
alternation.

### Decode, not detection, sets what a folder costs

Warm, from memory, 10 MP fixture: **JPEG decode 24.0 ms against 3.1 ms of
`detect_image`**. Every remaining item in the detection path is small against
that, so experiments 67 and 68 outrank them for folder work. Interactive
single-image latency is the case where the detection numbers below still
dominate.

libjpeg-turbo, already linked in through `nokhwa`, decodes the same corpus
**1.23x faster** (517 ms vs 420 ms over 15 images of 8-22 MP) and is now the
default for `.jpg`/`.jpeg`. Decoded pixels differ from the previous decoder by
up to 5/255, which moves detection boxes 0.28-1.30 px and an exported crop by
about a pixel; the crops were reviewed before adopting it, and the CLI JSON
snapshot was updated by at most 0.26 px. See experiment 67.

A build issue found along the way and fixed: `mozjpeg-sys` silently compiles
`jsimd_none.c` when NASM is absent, and the Windows and macOS release legs did
not install it, so released binaries decoded webcam MJPEG frames ~2.2x slower
than they should. Both legs now install NASM and the Windows leg fails if it is
not on `PATH`.

### The source resize is now a third of a folder job

After the changes above, a 1239-image folder spends 112 s of CPU, and
`resize_image` on the way into the detector is 32.3% of it. `detect_image` is the
same 32.3%, because a GPU inference wait costs no CPU: detection *is* the resize.

Feeding the detector a reduced-scale JPEG decode instead was measured and
rejected (experiment 90). The arithmetic works -- a 1/2 or 1/4 DCT decode costs
less than the resize it removes, about 13% off the folder -- but it moves
landmarks by up to 158 px, and backing off to where nothing moves past 35 px
leaves about 4%. `examples/scaled_decode.rs` re-runs the cost half.

### Letterboxing is free, and on a non-square source it is faster

Fitting the source into the model input instead of stretching it to fill it costs
nothing per detection and usually saves. The resize target is now the drawn region
rather than the full square -- 44% fewer output pixels for a 16:9 source -- and the
byte upload shrinks with it. Medians of three `phase_timings --mp N` runs on each
build:

| Source | stretched | letterboxed | delta |
| --- | ---: | ---: | ---: |
| 0.5 MP | 0.935 ms | 0.914 ms | -0.02 (inside the spread) |
| 2.0 MP | 1.860 ms | **1.480 ms** | **-0.38** |
| 6.0 MP | 1.900 ms | 1.820 ms | -0.08 |

`cpu_resize` goes 0.947 -> 0.669 ms at 2 MP and 0.929 -> 0.863 at 6 MP. The gain
shrinks as the source grows, because the resize is bounded below by reading the
source once and only the writing side got smaller.

The one cost it did add is invisible. `preprocess.wgsl` had to sample twice as
densely: its old tap rule guaranteed *coverage* of the source box rather than the
weighting the CPU resize uses, which stayed hidden while the short axis of a
stretched source was an upscale, and showed up as a CPU/GPU detection mismatch
once letterboxing made that axis a downscale. Four times the samples moved
`gpu_preprocess` from 0.269 to 0.270 ms, because that phase is host-side RGBA
conversion and texture upload rather than shader time.

### The preprocessing route crosses over at 1.75 MP, not 1.5

Sources above the gate are resized on the CPU and converted on the GPU; below it
the whole source goes up and the GPU does both. The 1.5 MP gate was measured
against the route experiment 50 has since replaced, so the losing side had got
cheaper and nobody re-ran it. Alternating the two routes in one process at a
fixed source size (`phase_timings --mp N --ab FCS_MAX_GPU_PREPROCESS_PIXELS=...`,
noise floor +/-0.07 ms from three A/A controls): the GPU route wins by 1.37 ms at
0.5 MP, 0.29 ms at 1.55 MP, ties from 1.65 to 1.85 MP, and loses by 0.13 ms at
1.9 MP and 0.87 ms at 3 MP. The gate is now 1.75 MP, worth 0.25-0.29 ms on
sources in the band it opened.

Under CPU contention the crossover moves up -- with the machine otherwise busy
the GPU route won by 0.47-0.61 ms at 1.6-1.9 MP -- so a batch job with 32 rayon
workers is the case that most wants the higher gate.

The two routes are not identical, so moving the gate moves detections: over 120
fixtures rescaled to 1.6 MP, 0 faces lost or gained, landmarks 0.65 px at p50 and
11.09 px at worst, IoU no lower than 0.98 (A/A control: 0.00 px, IoU 1.0000).
Forcing the two routes against each other at 1.4 MP -- a size the old gate already
sent to the GPU -- disagrees by the same p50 and p95 and gains a face, so the
seam is a standing property of shipping two routes rather than something the new
gate introduced. Neither route is a reference for the other.

### The `Speed` resize setting costs faces, not just quality

`ResizeQuality::Speed` is a `Nearest` filter, and nothing had measured what it
does to detections. Over 120 fixtures against production's `Quality`
(`resize_quality nearest`): **2 of 51 faces lost**, landmarks moved 10.7 px at
p95 and 33.6 px at worst, box IoU down to 0.93, in exchange for 1.76x. That is
the same failure the `Interpolation` candidate was rejected for below. The
default is `Quality`; the numbers now sit on the enum variant.

### Where a folder job's CPU actually goes

`samply` over `fcs-cli --crop` on 1239 images, all threads, 109.5 s of CPU over a
6.6 s wall run, summed by name across the top 400 self-time rows:

| Bucket | share |
| --- | ---: |
| `fast_image_resize` convolution (AVX2) | **34%** |
| `zlib_rs` deflate (PNG encode) | **22%** |
| libjpeg-turbo decode | **22%** |
| `DynamicImage::get_pixel` | 4.8% |
| memset/memcpy | 4.4% |
| our own code | **3.6%** |

**83% is third-party hand-written SIMD**, which is what closes experiment 70: there
is no autovectorisation opportunity in 3.6% of a profile, and `target-cpu=x86-64-v3`
is already set. The `get_pixel` row had no caller in our source -- it was
`crop_imm(..).to_image()` inside `crop_face_from_image` filling a second copy of the
crop region through the enum-matching accessor, then copying that into the canvas a
pixel at a time. A row-wise blit removed about 4% of the job's CPU with
byte-identical crops (experiment 98).

It removed no wall time. Six alternated folder runs each side sit inside one
band, which is the third result in a row saying the same thing: at 32 workers on 16
cores this job is not bound by CPU throughput.

### One process caps at ~1000 detections/s, and it is the resize

`concurrent_latency` runs N threads over pre-decoded images with nothing logged
until the end -- the first attempt at this used CLI telemetry and measured the
stderr lock instead. Detection throughput saturates around **1000/s** however the
concurrency is created, but the shape depends on how, and that matters:

| Threads | plain threads | rayon (what the app does) |
| ---: | ---: | ---: |
| 2 | 940 | 713 |
| 8 | 938 | 1002 |
| 16 | 863 | **1061** |
| 32 | 754 | 813 |

`samply` over both shapes (experiment 24) clears every suspect the backlog named:
no buffer-pool, convolution-cache, workspace or wgpu device lock appears. CPU is
`fast_image_resize`'s AVX2 vertical convolution, spread evenly across the 16
workers in the rayon profile.

The plain-thread column is capped by something the application never reaches.
`threading_pays` returns false under 4 MP and installs the resize into
`single_thread_pool()` -- one rayon pool with one thread, process-wide -- so eight
plain-thread callers serialise on one core, and the profile shows a single thread
holding 31.5% of all CPU. Every concurrent detection in production is on a rayon
worker, where that branch is skipped. Four processes reach 1259 det/s against one
process's 994, so the remaining cross-process headroom is ~27%.

In proportion: detection is about 3 ms of the 65 ms of CPU a folder image costs,
so this bounds a detection-heavy workload rather than the one the application
spends its time on.

### The head readback is not paying for its bytes

525 KB comes back per detection and the decode throws almost all of it away, so
compacting survivors on the GPU looks obvious. Timing the real allocate / copy /
map / wait / collect sequence with no inference in flight
(`examples/readback_bytes.rs`) prices it: 0.208 ms for production's three
buffers, 0.116 ms compacted to 2048 cells -- and then **nothing at all** for the
128x reduction from there down to 1 KB. A download costs ~0.11 ms before its
first byte.

Of the 0.09 ms ceiling, the allocation and copy halves land in the post-submit
window experiments 12-14 showed is free. What is actually recoverable is the DMA
inside the wait (~0.028 ms) and the host copy out of the mapped range
(~0.030 ms). Compaction was rejected: an atomic append also makes NMS
tie-breaking nondeterministic, and keeping the order needs a prefix scan and a
scatter.

**About a third of that host half was real** (experiment 17). The heads were
copied out of the mapping to own them and then cut into their four branches,
which copied most of them again; the branches now come off the mapping directly,
worth 0.011 ms at 0.8 MP and 0.021 at 10. Removing the surviving copy as well --
decoding straight from the mapped view -- was measured and **loses**:
`readback_bytes --reads` puts a decode-shaped read at 0.020 ms out of a `Vec`
against 0.031 ms out of the mapping, and that 0.011 ms penalty is larger than the
0.009 ms copy it would remove. The copy is load-bearing.

### The resize is at its floor

After the changes above, the CPU source resize is the largest single cost in
detecting a large image -- 1.39 ms at 10 MP, 2.78 ms at 22 MP -- and experiment
51 establishes that it cannot be made faster without changing what is detected.
`SuperSampling` measured **slower** (its nearest-neighbour first pass still
reads every source pixel, then adds a convolution over the intermediate).
`Interpolation` is 0.7-2.0 ms faster precisely because its fixed two-tap kernel
reads about four source pixels per output instead of the ~14 the ratio calls
for, and it moved landmarks by up to **35 px** over the fixture corpus.

At a fixed source resolution the resize is bounded below by reading the source
once, which production's adaptive-kernel convolution already does. The remaining
lever is fewer source pixels -- a decode-stage change (experiment 86), not a
resize-stage one.

`examples/resize_quality.rs` is what makes that kind of question decidable: it
runs the detector twice over the corpus and reports faces lost or gained,
landmark displacement in source pixels, box IoU and score deltas, against an
A/A control that reads exactly 0.00 px and IoU 1.0000.

### Changes retained in this round

| Change | Effect | Where |
| --- | --- | --- |
| Upload resized bytes and convert on the GPU | **-0.6 to -1.0 ms** on large images | `preprocess.rs`, `rgb_to_chw.wgsl` |
| Threaded source resize above 4 MP | **-0.6 ms** on large images | `fcs-utils/src/image_utils.rs` |
| No one-thread-pool hop inside a rayon worker | **-18%** of batch wall time | `fcs-utils/src/image_utils.rs` |
| RGBA fast resize for crops | -2% of batch wall time | `image_utils.rs`, `face_cropper.rs` |
| RGBA fast resize for the quality metric | **-9%** of batch wall time | `quality.rs` |
| Borrow rather than clone to encode an export | below the noise floor; one less copy | `output/encoders.rs` |
| One `fir::Resizer` per thread | -0.1 to -0.15 ms on large images | `fcs-utils/src/image_utils.rs` |
| Stop zeroing the BGR/CHW buffer | -0.1 to -0.2 ms on large images | `fcs-utils/src/image_utils.rs` |
| Decode straight from the GPU's channel-major heads | **-0.15 to -0.2 ms** | `model.rs`, `gpu/runtime.rs` |
| `Tensor::from_vec` instead of copying | -0.02 to -0.03 ms | `gpu/runtime.rs`, `model.rs` |
| One readback poll instead of two | no speed change; less code | `gpu/runtime.rs` |
| Cache the eight small-op uniform buffers | **-0.08 ms**, -10% of small-image detection | `gpu/utils.rs`, `max_pool.rs`, `add.rs`, `upsample2x.rs` |
| Fuse the four head branches per level | **-26% of GPU compute**, 61 dispatches to 43 | `gpu/graph.rs`, `gpu/runtime.rs` |
| At most four in-flight inferences per model | GPU pool -58 to -75% at 16-32 callers; iGPU folder -10% | `gpu/runtime.rs` |
| Backbone keeps only the outputs the neck reads | GPU pool -7 to -16% at every concurrency | `gpu/graph.rs` |
| Register accumulators and rows instead of local arrays | DXC 0.696 -> 0.375 ms (4090), 18.3 -> 10.3 ms (iGPU); FXC iGPU -4 to -26% per layer | `gpu/conv2d.wgsl` |
| Threshold edits swap postprocessing instead of rebuilding | 6.2 ms UI stall + 21.5 ms -> 3.0 ms per edit; confidence slider fixed | `detector.rs`, `fcs-gui` |
| One preprocessing cutoff for integrated adapters too | **2.07x** on a folder on the Radeon iGPU; -28 ms per 10 MP image | `preprocess.rs` |
| Upsample and add in one dispatch (neck) | -0.012 to -0.017 ms on a 4090, -0.13 to -0.19 ms on an iGPU; 43 dispatches to 41 | `gpu/resize2x_add.wgsl`, `gpu/upsample2x.rs`, `gpu/graph.rs` |
| Four-channel tile for the ungrouped general conv | **-56% of the stem**, 42.0 to 18.4 us | `gpu/conv2d.wgsl`, `gpu/conv2d.rs` |
| Cache convolution bind groups | **-60% of recording**, -6% of a detection | `gpu/conv2d.rs` |
| Skip decoding cells below the score threshold | **-82% of decode**, 0.077 to 0.014 ms | `model.rs`, `gpu/runtime.rs` |
| Branch the heads off the mapped view, not a copy of it | -0.011 to -0.021 ms | `gpu/runtime.rs` |
| Blit the crop region instead of `crop_imm().to_image()` | **-4% of batch CPU**, byte-identical | `face_cropper.rs` |
| Build the detector off the GUI's first frame | **-157 ms to first paint**, detector no later | `fcs-gui/src/app.rs` |
| Letterbox instead of stretching to the model input | **+100 faces over 1239 images**, -0.38 ms at 2 MP | `image_utils.rs`, both preprocess shaders, `detector.rs` |
| Delete the GPU batch cropper, crop on the CPU | **-43%** of folder wall time; ~740 lines removed | `face_cropper.rs` |
| libjpeg-turbo for JPEG decode | **1.23x** decode | `fcs-utils/src/image_utils.rs` |
| One shader entry point per convolution kernel | `compile_conv2d` ~197 -> ~120 ms at start-up | `gpu/conv2d.wgsl`, `gpu/conv2d.rs` |
| Preprocessing cutoff 1.75 MP instead of 1.5 | -0.25 to -0.29 ms on 1.5-1.75 MP sources | `preprocess.rs` |
| NMS grid sized to the boxes; bitmap dedup | clustered, n=5000: NMS 23.1 -> 0.25 ms, dedup 6.1 -> 0.009 ms | `nms.rs` |
| ONNX Runtime intra-op threads = logical CPUs / 4, clamped 1..=4 | CPU inference 7.27 -> 4.17 ms; folder unchanged | `fcs-ort/src/session.rs` |
| Preview texture clamp through `fast_image_resize` | **3.2 s -> 0.1 s** on a 133 MP preview | `fcs-gui/src/core/detection.rs` |
| Delete the three unread GUI caches, and `lru` | up to ~1.3 GB no longer retained; 179 lines | `fcs-gui` |
| 30 s deadline on every GPU wait; one test device per binary | no runtime change; `fcs-utils` suite 2.6x faster, harness hang gone | `fcs-utils/src/gpu/mod.rs` |
| Stem sized from its input tensor | none at 640; makes a 320 input runnable (75) | `gpu/graph.rs` |
| One submission per SCRFD forward pass | **-1.6 to -1.8 ms** a detection: -44% on a 4090, -7% on an iGPU. Bit-identical on the 1,239-image folder. YuNet's runtime had this; SCRFD's port had lost it | `scrfd/gpu.rs` |

Output-changing rows, each adopted only after the output was reviewed: the GPU
cropper deletion (Lanczos3 instead of fixed 2x2 taps; 18% of quality labels shift,
71), libjpeg-turbo (pixels within 5/255, boxes 0.28-1.30 px, 67), letterboxing
(96), the 1.75 MP cutoff (moves the seam between the two preprocessing routes, 54)
and the RGBA crop resize, which swaps one Lanczos3 implementation for another and
so differs by rounding -- at most 23 per channel over a 1239-image folder, one crop
in 901 shifting quality label (88). The rest are bit-exact: the 126000-float
decoded-output fingerprint is unchanged (`readback_parity`), resize output is
byte-identical between one thread and many
(`threaded_and_single_threaded_resize_agree`), and the full workspace suite passes
under `FCS_STRICT_TESTS=1` with ONNX Runtime 1.24.4.

### Historical stage breakdown (before the latest GPU optimizations)

Release build, 640x640, RTX 4090 / Ryzen 9 7950X. The following measurements
belong to an earlier revision and are retained for CPU/decode context. The GPU
8.2 ms figure is not the current baseline; do not combine this table with the
latest shader timings to calculate stage shares or gains.

Measured per stage on `fixtures/images/006.jpg` (2384x4240 -> 640x640) with
`cargo run --release -p fcs-core --example stage_breakdown`, mimalloc enabled as
in the shipped binaries.

| Stage              | tract    | ONNX Runtime | GPU (WGSL) |
|--------------------|----------|--------------|------------|
| JPEG decode        | 20.7 ms  | 20.7 ms      | 20.7 ms    |
| Preprocessing      | 2.9 ms   | 2.9 ms       | on device  |
| Inference          | 65.9 ms  | 6.7 ms       | —          |
| Postprocessing     | <1 ms    | <1 ms        | <1 ms      |
| **`detect_image`** | 69.6 ms  | **10.7 ms**  | **8.2 ms** |
| End-to-end / image | 86.9 ms  | 31.3 ms      | ~29 ms     |

Batch, 20 images through rayon over one shared detector: tract 236 ms,
ONNX Runtime 118 ms. The batch gain (2.0x) is far smaller than the single-image
gain (6.5x) because batch was already parallel across images and becomes memory
bound once inference stops being the constraint — worth remembering before
quoting the single-image ratio at anyone.

### Bottleneck Summary

- **Current GPU inference:** convolution still dominates GPU compute, but shader
  time is only one part of detection. Measure CPU recording, submission, buffer
  allocation, synchronization, readback and output conversion separately before
  selecting the next latency change.
- **Remaining shader budget:** 0.366 ms of GPU compute on the 4090, where
  detection is latency-bound rather than compute-bound (8), against 10.5 ms on
  the Radeon iGPU, where the arithmetic is 92% of detection (10). These are
  fixed-workload ceilings, not promised wall-time savings. Profiling uses
  separate passes (0.83 us each); normal inference uses one merged pass, so
  validate gains in the normal path.
- **Full-resolution image processing:** JPEG decode was about 21 ms in the
  historical fixture measurement and remains a separate workload from
  `detect_image`. Reduced-resolution decode does not preserve crop quality.
- **Batch and interactive work:** CPU parallelism, GPU queue occupancy and
  latency overlap differ. Remeasure the actual export or webcam/preview path;
  a single-image shader result cannot select the fastest batch backend.
- **Preprocessing and postprocessing:** now measured, not assumed. CPU
  preprocessing is 2.65 ms of a 4.68 ms detection at 10 MP and is the single
  largest cost in the application; the resize inside it is now threaded above
  4 MP, and unconditionally when a rayon worker is already running it -- holding
  it to one core there means a cross-registry `install()` hop that cost 18% of
  batch wall time (experiment 88). Postprocessing is 0.005-0.010 ms and is not worth attention. Output
  conversion and decode together are about 0.3 ms and sit on the critical path
  after the readback wait -- now about 0.08 ms, since the CHW-to-HWC transpose
  the decoder used to require has been removed rather than optimised.

---

## Optimisations Implemented

### Phase 1 — Quick Wins

| Optimization                  | Impact                   |
|-------------------------------|--------------------------|
| `image` crate `rayon` feature | 2–4ms on first load      |
| App-level image cache (GUI)   | 34ms saved per cache hit |
| tract already uses rayon pool | Baseline (no change)     |

### Phase 2 — Medium Effort

| Optimization                                | Impact                     |
|---------------------------------------------|----------------------------|
| Thread-local preprocessing buffer pool      | 2–5ms per detection        |
| GPU preprocessing (`WgpuPreprocessor`)      | 20–25ms when GPU available |
| Conditional resize bypass (`Cow<RgbImage>`) | 15–20ms for 640×640 inputs |

### Phase 11 — Enhancement Pipeline

| Optimization                     | Impact                   |
|----------------------------------|--------------------------|
| Skin smoothing rayon parallelism | 4.25s → 116ms (36×)      |
| Background blur rayon + no-sqrt  | 182ms → 136ms (25%)      |
| Exposure/brightness/contrast LUT | Criterion: 895ms → 798ms |
| Plain ops + saturating cast in hot loops (replaced `mul_add`/`round`/`wide::f32x4`; see below) | Saturation 7.6→6.2ms, skin 8.0→2.3ms, unsharp 4.9→1.9ms |
| `target-cpu=x86-64-v3` on x86_64  | Lets LLVM autovectorise the above |

### Phase 12 — GPU Acceleration

| Optimization                      | Status     | Impact                                      |
|-----------------------------------|------------|---------------------------------------------|
| GPU preprocessing (WGSL shader)   | ✅ Shipped | CPU ~162ms → GPU ~51ms (Criterion)          |
| GPU enhancement pipeline          | ✅ Shipped | All filters have WGSL kernels               |
| Custom WGPU YuNet inference graph | ✅ Shipped | Conv2D/BN/Activation on GPU, parity-tested  |
| GPU batch crop extraction         | ❌ Removed | CPU cropping is 43% faster on a folder (experiment 71) |
| GPU buffer/texture pool           | ✅ Shipped | Avoids repeated allocation overhead         |

### GPU experiment results

Baseline `c332075` already contains convolution uniform caching and the merged
compute pass. The latter previously saved about 0.40 ms in paired
encode/finish/submit/wait measurements while retaining all 61 per-op profiling
records. That CPU/driver saving is separate from the shader results below.
See [CHANGELOG.md](../CHANGELOG.md) for those earlier measurements.

| Experiment | Paired full-graph GPU time, A/B/B/A | Decision |
| --- | --- | --- |
| 0. Timestamp families and genuine shader A/B probe | Baseline about 0.910 ms | Keep measurement tools; retire duplicate standard/vec4 benchmark labels |
| 1. Specialized 1x1 convolution | 0.909 / 0.673 / 0.672 / 0.910 ms | Keep; about 0.237 ms saved |
| 2. Depthwise 3x3 overlapping-input reuse | 0.673 / 0.648 / 0.646 / 0.674 ms | Keep; about 0.027 ms saved beyond step 1 |
| 3. Four-channel pointwise register tile | 0.647 / 0.535 / 0.538 / 0.651 ms | Keep; about 0.113 ms saved beyond step 2 |
| 4a. FP16 storage, f32 accumulation | Microbenchmarks only; eight cases 3-15% slower, two unchanged | Reject this candidate; no full-graph trial warranted |
| 4b. Subgroup channel reduction | Microbenchmarks only; large cases 2.9-9.9x as slow, some small cases faster | Reject as a general replacement; selective use remains untested |

Each A/B/B/A row compares the previous retained version with that experiment's
candidate. The savings are rounded from separate comparisons; they are not
independent percentages to add. The final comparison with the original baseline
is about 41% less GPU time, not 41% less detection time.

Pointwise specialization requires a 1x1 kernel, unit stride, zero padding and
one group. The four-channel tile shares each loaded four-pixel input vector
across four accumulators; dispatch-z rounds up and guards channel tails.
Depthwise specialization requires a 3x3 kernel, unit stride, pad 1 and channel
multiplier 1. It loads six values per row for four adjacent outputs, expressing
18 distinct input loads rather than 36. Other configurations retain the general
shader. Fused activations and the shared graph/profiling path are preserved.

A representative final profiled graph breaks down as follows (20 runs; values
rounded, so sums can differ slightly):

| Family | Dispatches | Before shaders | After shaders | Current GPU share |
| --- | ---: | ---: | ---: | ---: |
| Pointwise | 26 | 673.8 us | 320.5 us | 60.0% |
| Depthwise | 26 | 145.4 us | 123.9 us | 23.2% |
| General/stem | 1 | 42.0 us | 41.0 us | 7.7% |
| Pool/resize/add | 8 | 49.1 us | 49.2 us | 9.2% |
| **Total** | **61** | **about 910 us** | **about 535 us** | **100%** |

The reusable shader probe checks finite raw outputs, warms up for 20 pairs,
then measures 50 pairs with alternating order in one process. Compilation,
upload, validation readback and timestamp resolution are outside its GPU timer.
Identical-file controls established roughly 1.024 us timestamp granularity;
one-tick changes on tiny layers are weak evidence. Full-graph runs are separate
binary A/B/B/A comparisons. The final Criterion latency check used 3 seconds of
warm-up and 30 samples over at least 6 seconds per run; historical cached
Criterion change percentages were not used for the conclusion.

Final validation: **823 strict workspace tests plus two doctests passed**, with
eight workspace tests and five doctests skipped/ignored. Coverage included
ONNX raw-output and detection parity, profiled/merged equality, concurrent
inference, spatial/channel tails, exposed activations and stride/pad/group
fallbacks. Workspace and core all-target Clippy, formatting and diff checks
passed. This is validation on the tested hardware, not cross-adapter performance
validation.

---

## Benchmark Infrastructure

```bash
# CPU vs GPU preprocessing (Criterion)
cargo bench -p fcs-core --bench preprocessing

# Lightweight CLI benchmark over an image set
cargo run -p fcs-cli -- --input fixtures/images --benchmark-preprocess

# Full pipeline example
cargo run --release --example profile_pipeline -p fcs-core

# Full-graph GPU timestamp families (profiling path)
cargo run --release -p fcs-core --example gpu_pass_breakdown

# Actual shader A/B, current four-channel pointwise grid on both sides
cargo run --release -p fcs-core --example conv2d_experiment -- fcs-core/src/gpu/conv2d.wgsl fcs-core/src/gpu/conv2d.wgsl 32 8 4 32 8 4

# Paired separate/merged pass encoding costs
cargo run --release -p fcs-core --example gpu_encode_comparison

# Wall-clock phase breakdown of one detection, and in-process A/B of a flag
cargo run --release -p fcs-core --example phase_timings [image]
cargo run --release -p fcs-core --example phase_timings [image] --ab SOME_ENV_FLAG

# Bit-exact fingerprint of the decoded output, for readback and decode changes
cargo run --release -p fcs-core --example readback_parity

# Where the decode spends its time: transcendentals against gathers and writes
cargo run --release -p fcs-core --example decode_cost

# What a head download costs per byte, with no inference hiding the copies
cargo run --release -p fcs-core --example readback_bytes

# Sustained operation: the same corpus N times in one process (drift, growth, drift in
# what it finds)
cargo run --release -p fcs-core --example memory_growth -- <dir> 400 --passes 50

# Detection latency and throughput with N in flight, no logging in the way
cargo run --release -p fcs-core --example concurrent_latency -- <dir> --threads 8
cargo run --release -p fcs-core --example concurrent_latency -- <dir> --ab SOME_ENV_FLAG

# Whether the decode should read the mapped range or a copy of it
cargo run --release -p fcs-core --example readback_bytes -- --reads

# Both preprocessing routes at one source size, in one process
cargo run --release -p fcs-core --example phase_timings -- --mp 1.8 --ab FCS_MAX_GPU_PREPROCESS_PIXELS=99000000

# What a cheaper resize costs in detections (super2 | super3 | interp | nearest)
cargo run --release -p fcs-core --example resize_quality -- nearest [image-count]

# Cold start, stage by stage: adapter, shader compile, model parse, first detection
cargo run --release -p fcs-core --example cold_start

# GPU pool and host RSS as a session goes on, largest source first
cargo run --release -p fcs-core --example memory_growth -- <dir> [limit]

# Where a webcam frame's time goes (opens the camera)
cargo run --release -p fcs-cli --example webcam_cost -- [frames] [w] [h] [fps]

# One captured frame detected at several sizes and letterboxed, to separate
# aspect ratio from resolution
cargo run --release -p fcs-cli --example webcam_resolution -- x [frames] [outdir]

# Whether squashing to square costs detections on files too, bucketed by aspect
cargo run --release -p fcs-core --example aspect_recall -- <dir> [limit]

# Adapter selection cost per backend set (one process per configuration)
cargo run --release -p fcs-core --example adapter_cost -- dx12

# What each WGSL file or entry point costs to compile (order matters: the first
# pipeline in a process carries ~180 ms of one-off warm-up)
cargo run --release -p fcs-core --example shader_compile_cost -- fcs-core/src/gpu/conv2d.wgsl

# Cost of the per-dispatch host objects: uniform buffers and bind groups
cargo run --release -p fcs-core --example encode_cost

# Pass-boundary cost and the per-dispatch floor, merged vs per-pass timestamps
cargo run --release -p fcs-core --example pass_overhead

# Whether threading the source resize pays, at several megapixel counts
cargo run --release -p fcs-core --example resize_threading

# Whole detection; keep gpu, gpu_on_device and gpu_quality results separate
cargo bench -p fcs-core --bench inference_pipeline

# Quality against production for a changed detector path: faces lost and gained,
# landmark shift, IoU. INT8 model (77); 320 first pass, cascade, screen, crop refinement (75, 79)
cargo run --release -p fcs-core --example int8_quality -- <int8.onnx> <dir> [limit]
cargo run --release -p fcs-core --example coarse_to_fine -- <dir> [limit] [roi scale]

# CPU inference backends, cpu-graph against ONNX Runtime (69)
cargo run --release -p fcs-core --example cpu_backends -- <image> [reps]

# JPEG decoders over the largest fixtures (67); compare two folders of exported crops
cargo run --release -p fcs-core --example decode_bench [image-count]
cargo run --release -p fcs-core --example cropdiff -- <dir-a> <dir-b>

# Container formats decoded from identical pixels (6); scaled DCT decode against the resize (90)
cargo run --release -p fcs-core --example decode_formats -- <image> [reps]
cargo run --release -p fcs-core --example scaled_decode -- <dir-of-jpegs> [limit]

# Per-image decode and detect, bucketed by resolution, orientation, format and faces (6)
cargo run --release -p fcs-core --example workload_matrix -- <dir> [more dirs...]

# Warm file reads at batch concurrency (68); PNG encoder settings over exported crops (72)
cargo run --release -p fcs-core --example io_cost -- <dir> [passes]
cargo run --release -p fcs-core --example png_bench -- <dir-of-pngs> [limit]

# A threshold edit: detector rebuild against a postprocessing swap (66)
cargo run --release -p fcs-core --example threshold_edit_cost -- [image] [reps]

# Preview texture build and the oversize clamp's resize (91)
cargo run --release -p fcs-gui --example preview_texture_cost -- <image> [reps]

# GPU/CPU parity validation (set FCS_STRICT_TESTS=1 to reject missing prerequisites)
cargo test -p fcs-core gpu_inference_matches_cpu_baseline -- --nocapture
```

Criterion results are written to `target/criterion/`. Do not commit benchmark output text files.

---

## What Did Not Work

### FP16 storage and a general subgroup replacement

The default D3D12 context selected FXC and exposed neither optional shader
feature. Making Windows SDK DXC 1.9.2602.17 (SDK 10.0.28000.0) available on the
probe's process-local PATH enabled both SHADER_F16 and SUBGROUP on the same
RTX 4090. Lack of compiler support was not lack of physical GPU support.

The FP16 candidate packed input, weights and bias to f16, accumulated in f32,
and returned f32. Its maximum synthetic raw error was 0.000824, within the
probe's absolute 1e-3 screening budget. It was slower even with packing excluded
from timing, so it did not advance to full-graph or detection parity testing.
This rejects that storage/conversion strategy, not every FP16 arithmetic, packed
layout or mixed-precision implementation.

The subgroup candidate reduced input channels across one 32-lane subgroup for
four output pixels. Raw f32 comparisons passed, but 320x320 16 -> 16 cost
201.728 us versus 20.480 us for the f32 tile under DXC. Small layers did improve:
20x20 64 -> 64 fell from 29.696 to 8.192 us. It was not suitable across the graph.

DXC also slowed the retained f32 tile: 160x160 64 -> 64 measured 67.584 us
versus 27.648 us under the existing FXC setup. A whole-context compiler switch
would sacrifice large-layer performance to enable the small-layer subgroup
candidate. Selective kernels need a compiler/deployment solution and a full-graph
measurement before adoption. Other subgroup algorithms have not been exhausted.

The standalone [FP16](../fcs-core/examples/shaders/pointwise_f16.wgsl) and
[subgroup](../fcs-core/examples/shaders/pointwise_subgroup.wgsl) probes reproduce
these; neither adds a production feature requirement. Run them in a fresh
PowerShell process so the PATH change stays local (adjust the SDK path):

```powershell
cargo build --release -p fcs-core --example conv2d_experiment
$env:PATH='C:/Program Files (x86)/Windows Kits/10/bin/10.0.28000.0/x64;' + $env:PATH
& target/release/examples/conv2d_experiment.exe fcs-core/src/gpu/conv2d.wgsl fcs-core/examples/shaders/pointwise_f16.wgsl 32 8 4 32 8 4 --f16-storage
& target/release/examples/conv2d_experiment.exe fcs-core/src/gpu/conv2d.wgsl fcs-core/examples/shaders/pointwise_subgroup.wgsl 32 8 4 4 1 1
```

The subgroup kernel assumes a 32-lane subgroup and is not portable; with this naga
the native SUBGROUP feature enables the builtins directly and `enable subgroups;` is
rejected. The DXC regression above was later traced to loop-indexed local arrays and
removed (41), and both candidates were re-measured without it and rejected again
(43-46).

### GPU memory scaled with concurrency, not with images

GPU retention is flat as a session goes on (experiment 84), but that was measured one inference at a
time. Every in-flight inference parks its intermediates in its own execution scope, so the pool grew
about 36 MB per concurrent caller: **1.2 GB at 32 rayon workers**, on the 4090 and the Radeon iGPU
alike. Two changes: the backbone stopped handing the neck two 6.5 MB outputs it never reads
(experiment 40, 7-16% off), and a model now admits at most four inferences at once (experiment 21).
At 32 callers the pool is **346 MB on the 4090 and 281 MB on the iGPU**, where past four callers
only contend for a busy adapter -- the iGPU's folder job got 10% faster. 0 crops differ.

### INT8 is slower on the CPU path, and moves landmarks

A static per-channel QDQ model on ONNX Runtime 1.24.4 took 11.5 s of detection over 1239 images
against 8.4 s for f32 -- 0.73x on a Zen 4 with VNNI -- and lost 13 faces, gained 7, and moved 111
landmarks past 35 px (experiment 77). YuNet is mostly depthwise convolution, which the quantised path
does not accelerate. The bundled export also needs an opset 11 -> 13 conversion before per-channel
quantisation produces a model ONNX Runtime will load at all.

### Shader arithmetic that did not pay: constants, packed weights, interior paths, shared tiles

Measured on both the 4090 and the Radeon iGPU, because the iGPU is where arithmetic matters:

- **A compile-time channel count** (experiment 28) is one timestamp tick either way on the 4090,
  nothing under DXC, and under FXC *slower* on three of four iGPU layers (+46% at 80x80). No gain
  to pay a pipeline per shape for.
- **Weights prepacked as one vec4 per tile** (30) costs one to three ticks on most of the 4090's
  pointwise layers and helps only the iGPU's two smallest.
- **An interior fast path for depthwise** (33) is slower on the 4090 and a net loss across the
  iGPU's layers under FXC. Its large DXC win turned out to be 41's mechanism, not the bounds checks.
- **Cooperative workgroup tiles** (29) are not built: naga lowers `workgroupBarrier()` to HLSL's
  non-synchronising `GroupMemoryBarrier()`, so staged workgroup memory is a race on D3D12.

- **Subgroups and FP16, retried once DXC stopped regressing** (43-46). Subgroup channel reduction
  still wins only the 4090's smallest layers -- about 25 us of graph -- and is 5-31x slower on the
  iGPU. FP16 arithmetic misses the 1e-3 raw-error screen on every real shape and is 19-58% slower
  on the iGPU; f16 storage alone is 13-50% slower there. Neither is worth shipping DXC for.

- **NHWC activations** (31) make both kernels of a segment several times slower on both adapters,
  and **fusing depthwise into the following pointwise** (35) costs 2.2-2.6x the separate pair,
  because every pointwise tile recomputes every channel's depthwise value.

- **A 320 first pass** (75, 79), the same weights re-exported at 320, moves landmarks past 35 px
  more than ten times as often as rejected INT8 wherever its detections are kept -- alone,
  behind a confidence gate, or refined on crops. As a pure no-face screen it keeps production's
  detections, but on the reference corpus it is slower on both adapters (0.80x iGPU, 0.61x
  4090): 86% of images still need the 640 pass.

Closed without a new measurement because an existing one removes the cost they target: buffer
arenas (23), a host dispatch plan (25), preprocessing-stem fusion (39), two-stage decoding (73)
and webcam tracking (76). Native runtimes (47) were checked with no change. Smaller models and
trained early exits (78, 79) are blocked on a training pipeline and labelled data this
repository does not have.

### Nested loop parallelisation in `decode_yunet_outputs`

Parallelizing the inner `row`/`col` loops within each stride added overhead rather than saving
time. Thread coordination cost exceeded the per-row work. The existing stride-level parallelism
(3 parallel tasks for strides 8/16/32) is the right granularity for this workload.

### Decoding JPEGs at reduced scale

The arithmetic is tempting: detection runs at 640x640, sources are routinely
2384x4240, so decode appears to produce 93% pixels that are thrown away. JPEG
supports 1/2, 1/4 and 1/8 scale decode natively in the DCT at a fraction of the
cost.

It does not work here, because those pixels are not thrown away — they are the
product. `process_single_image` decodes once and passes the same image to both
detection and `crop_face_from_image`, and the output crop is cut from the
full-resolution pixels. Detection only needs 640x640; cropping needs everything.
Decoding at reduced scale would degrade every crop the app produces. Decoding
twice (scaled for detection, full for cropping) is strictly more work for any
image that actually contains a face, which is most of them.

Two further measurements explain why reduced-scale or within-image parallel
decode was not pursued for these fixtures:

- **The decoder was already fast.** zune-jpeg, via `image`, ran at 390-570
  Mpx/s on this hardware; the ~21 ms is what 10.1 megapixels costs. libjpeg-turbo
  has since replaced it for JPEG at 1.23x (67), which is the whole of the decoder
  gain found.
- **It cannot be parallelised for these files.** Decode is single-threaded —
  identical timings under `RAYON_NUM_THREADS=1` and 32 — and splitting one image
  across threads requires restart markers to give independent entry points into
  the entropy-coded stream. None of the fixtures have any (no DRI segment, zero
  RST markers): baseline sequential JPEG is one continuous Huffman run, so no
  MCU can be decoded without decoding every MCU before it.

Whole images already decode in parallel across rayon workers. Any new decoder
or scheduling strategy needs a separate batch measurement; the figures above
are single-image latency measurements.

### Hand-written SIMD via the `wide` crate

`wide::f32x4` was used for the saturation pass in Phase 11 and removed again in `616ffa0`: the
plain scalar loop benched **18% faster**, so the dependency was dropped. Two reasons it lost, both
still true:

- The build now sets `target-cpu=x86-64-v3` on x86_64, so LLVM autovectorises the scalar loops at
  AVX2 width. Hand-written `wide` has to beat the autovectoriser, not scalar code.
- Every remaining hot loop is either an interleaved-pixel deinterleave (RGBA saturation, RGB→BGR
  CHW) or a table lookup (tone LUTs, histogram equalisation, the bilateral colour LUT). Those need
  `pshufb`/`vgather`, which `wide` deliberately does not expose — it is an elementwise-lanes crate.
  The loops `wide` *could* express are already autovectorised, or live inside `tract` and
  `fast_image_resize`, which are hand-vectorised already.

Related: `f32::mul_add` and `f32::round` were removed from the same loops. On the pre-v3 SSE2
baseline both compiled to software libm calls. The hot loops use plain `a * b + c` and a
saturating `(x + 0.5) as u8` cast instead, marked with `ponytail:` comments at each site.

Re-measured on the v3 build (2026-09-25), where `mul_add` is a single `vfmadd`. It still does
not pay. Reassociation is not the reason: Rust never contracts or reassociates plain `a * b + c`
either. These loops are bound by loads, stores and u8↔f32 conversion, not by arithmetic.
Single-threaded, 1024²: saturation 1.69 → 1.91 ms and unsharp 2.04 → 2.16 ms (slower with FMA);
the 9x9 skin bilateral 44-50 → 44-46 ms, within noise and dominated by the colour-LUT gather.
The CPU convolution (`cpu/conv2d.rs`) behaves the same: `mul_add` in all four accumulation
loops left `engine_speed`'s built-in CPU graph at 23-27 ms either way. The WGSL accumulation
loops already use `fma()`, and the shaders that do not are index math or one `mix` per pixel.

---

## Open questions

What the experiments left open, each with the ID whose record says why. None is a
promised saving.

- **Other hardware** (10). Everything here is one RTX 4090 and one Ryzen 7950X
  iGPU. Not measured: Intel, Apple/Metal, discrete AMD, a strong unified-memory
  part. On the 7950X, ONNX Runtime on the CPU (2.86 ms) beats the iGPU (11.93 ms of
  inference), and GPU inference stays the default anyway; re-run `cpu_backends`
  against the iGPU on a mainstream laptop before changing that. DirectML was never
  measured on an integrated adapter (47).
- **Cross-adapter output.** The 4090 and the iGPU disagree on 132 of 959 crops
  although inference fingerprints identically; the likely source, the whole-source
  route's texture sampling, was not traced (10).
- **The GUI under contention.** Waiting on a readback's own submission index cut
  `readback_wait` 24-30% under 32 workers and changed nothing end to end, but the
  GUI shares its device with the renderer, where a device-wide wait also waits for
  rendering (16). Device loss and model switching are untested (81).
- **Long sessions.** Export drift is measured (85); webcam sessions, VRAM pressure,
  background GPU work, thermals and energy per image are not.
- **Memory.** Host RSS is ~85 MB per worker and the default worker count follows
  logical processors, so a many-thread CPU with little RAM is the case to check
  before any cap (84). Under the in-flight limit the GPU pool still grows with
  callers, probably from input tensors and uploads acquired before the gate (21).
  Worker count on hybrid-core CPUs is unmeasured (63).
- **Cold file I/O.** A cold folder run once took 39.8 s against ~17 s warm, before
  most of this work, and was never attributed; warm reads are 2.4% of a job (68).
- **Small remainders.** `dedup_close_centers` on 5000 separated survivors is still
  14.3 ms, reachable only with NMS at 0 (58); `to_luma8` in `laplacian_variance` is
  worth ~1% of a folder's CPU (98); the nine non-convolution bind groups are ~7 us
  (22); max-pool into pointwise fusion was not tried (36); restoring the GUI
  detection cache with a correct key would save ~60 ms per re-selection (52).
- **Model research** (74, 75, 78, 79). A smaller or distilled model and trained
  early exits need a training pipeline and labelled data this repository does not
  have. Every quality comparison here is against production's own detections, so
  whether 960-input detections are real faces (74) or a 320-derived landmark is
  worse than production's (75) needs ground truth. The reference corpus has no
  image with four or more faces. A no-face screen could pay on a mostly faceless
  workload on an iGPU-class adapter, measured together with two-stage decoding
  (73, 79).
- **Faster cameras.** Webcam GPU residency, fresh-frame scheduling and tracking
  (53, 65, 76) reopen only for a camera, or several, fast enough to saturate the
  pipeline (95, 97).

DirectML, CoreML and hand-written `wide` SIMD results are scoped negatives, not
proofs that every runtime, compiler or architecture behaves the same way.

## CPU inference

### The ONNX Runtime CPU backend (shipped in fcs-core)

`tract` lowers YuNet's 3x3 depthwise convolutions to a scalar fallback — it only
unrolls zones with at most 4 taps — which 1.5.3 measured at 59-61% of a CPU
detection and concluded was "not reachable from this side". That was true within
tract. Swapping the runtime reaches it: ONNX Runtime vectorises those
convolutions and runs the same graph in 6.7 ms against 65.9 ms.

Scope deliberately excludes DirectML. Measured end to end it lands at ~9.8 ms
against the then-current 8.2 ms WGSL graph without an additional runtime DLL,
so that comparison did not justify adopting DirectML. It did not measure CoreML
or establish a permanent ceiling for other execution providers. Only the CPU EP
is used, which also means one library (20.1 MB) rather than the ~38 MB a DirectML build needs,
and no per-platform execution-provider matrix.

Loaded dynamically rather than linked: the prebuilt static library is built
against the dynamic CRT and collides with the workspace's `+crt-static`, and
dynamic loading keeps DirectML out of the build entirely.

The binding lives in `fcs-ort` and is ours — the `ort` crate is no longer a
dependency. `fcs-core/src/ort_backend.rs` only adapts between tract tensors and
that binding. Two things to know before touching either:

1. **`sys::OrtApi` is a hand-maintained prefix of a 424-entry function table.**
   A wrong offset is undefined behaviour, not a compile error. Read the rules in
   `fcs-ort/src/sys.rs` before adding a field, and note that generated bindings
   declare some fields twice under `#[cfg]` — counting those duplicates shifts
   every later offset. Two guards exist: a `const` assertion that the struct is
   exactly one pointer per declared field, and `fcs-ort/tests/end_to_end.rs`,
   which calls through offsets 3 to 100 against a real runtime.
2. **A bad runtime must never reach the FFI.** `fcs_ort::locate` reproduces the
   version rule and adds an ABI check, because a machine with an unrelated
   `onnxruntime.dll` on PATH will otherwise be resolved and used. This mattered
   more when `ort` was in the picture — its version rejection aborted the
   process — but a mismatched library is still worth refusing outright.

Sessions are shared, not pooled: ONNX Runtime permits concurrent `Run` on one
session, which `concurrent_runs_match_sequential` verifies against eight
threads. `SessionOptions::intra_threads` defaults to 1 because parallelism comes
from rayon running whole images at once.

### INT8 quantisation is not the `ort` question

These are often conflated. They are separate decisions:

- **Runtime selection** changes the inference implementation. This workspace now
  uses its own `fcs-ort` binding and a roughly 20 MB dynamically loaded ONNX
  Runtime library; the external `ort` crate is no longer the integration.
- **INT8** is a change to the *model file*, and needs no runtime change at all. `tract-onnx`
  already registers `QuantizeLinear`, `DequantizeLinear`, `QLinearConv` and `QLinearMatMul`, and
  `tract-linalg` ships x86_64 i8 GEMM kernels (`avx2_mmm_i32_8x8`, `avxvnni_mmm_i32_8x8`,
  `avx512vnni_mmm_i32_*`) selected by runtime CPUID. Quantising is a one-off offline step whose
  output is a committed `.onnx`, exactly like `face_detection_yunet_2023mar_640.onnx` — nothing
  extra ships and the pure-Rust build story is unchanged.

Experiment 77 measured it, cheapest killer first. **Loading:** per-channel QDQ
needs opset 13 and the bundled export is opset 11, so it needs
`onnx.version_converter` first -- and without that the loader's error blames the
runtime rather than the model. **Speed and recall** together: static per-channel
QDQ calibrated on 100 fixtures and evaluated on the separate 1239-image corpus ran
at **0.73x** of f32 on ONNX Runtime and moved 111 landmarks past 35 px; see "INT8 is
slower on the CPU path" above. YuNet is mostly depthwise convolution, which the
quantised path does not accelerate.

Two further costs are easy to overlook. INT8 gains depend strongly on the CPU: with AVX512-VNNI
(Zen 4, Cascade Lake+, Alder Lake+) `VPDPBUSD` gives a 4-way i8 dot per lane, but on the shipped
`x86-64-v3` baseline the AVX2 fallback computes i8 products in i16 lanes — the same lane count as
f32 FMA, so the win is cache footprint rather than arithmetic. Benchmarking only on a VNNI
developer machine will overstate what most users get. And the WGSL GPU inference path gains
nothing, so an INT8 CPU path diverges numerically from the f32 GPU path that
`gpu_inference_matches_cpu_baseline` and docs/parity_report.md compare against.

---

## Measurement rules and lessons

### Rules

- **Alternate inside one warm process, with an A/A control first** (`phase_timings --ab`,
  `concurrent_latency --ab`). Cross-process comparisons of single detections cannot resolve
  these effects. Readable (7): 0.01 ms of wall time on a small image, ~0.1 ms at 10 MP, one
  1.024 us timestamp tick per GPU family, ~1 s on a folder job.
- **Compare outputs before timing.** `readback_parity` fingerprints the *decoded* output --
  8400 cells x 15 columns = 126000 floats, `0xa116e42f7c2dabdb` -- not the raw heads, so a
  bit-exact claim covers readback and decode, and a change to what decode writes changes the
  fingerprint by design.
- **A faster microbenchmark earns a full-graph trial, not adoption.** Eight channels per
  pointwise thread won the three largest layers by 15-19% and lost the graph by 23% (27).
- **Quality-changing candidates are judged against production**, since there is no ground
  truth: faces lost and gained, landmark shift in source pixels and box IoU, through
  `resize_quality`, `int8_quality` or `coarse_to_fine`. Since 51 a candidate that moves
  landmarks past 35 px is rejected. Never weaken a parity test to admit a candidate; where a
  test had to change what it asserts (36, 37, 96), the entry says why.
- **Batch wall time swings ~10% run to run and drifts between batches**, so only compare
  order-alternated pairs from the same batch (60, 63, 72).
- **A change to a shared cache, a thread-local or the resize path needs a folder run** as well
  as the test suite: nested rayon is where such changes break, and nothing else nests it (67).
- **p95 from 30 samples is not a statistic**: it moved 0.89-1.20 ms across five identical
  processes (7). Quote tails from hundreds of samples.
- Do not run GPU benchmarks concurrently. State the configuration, including thresholds and
  resize quality. Record negative results, remove losing production code, and keep the smallest
  probe that makes the decision reviewable.

### Traps that invalidated earlier measurements

- CPU throughput on this machine moved **1.6x between two builds of identical code**; never
  compare CPU work across builds (48).
- The first pipeline built in a process carries ~180 ms of one-off FXC and D3D12 warm-up, so
  compare shader compile costs within one process (82).
- `fcs-cli` reads `config/gui_settings.json` from the working directory: the same command found
  1020 faces from the repository root and 423 elsewhere. Pass `--config` (63).
- The library's default score threshold is 0.9 and production's is 0.8; 96's first numbers
  measured the wrong one.
- `YuNetDetector::new_gpu` pairs GPU inference with the CPU preprocessor, while the CLI and GUI
  build `with_gpu_preprocessor`; the wrong one inverted 6's first conclusion.
- Repeating one image understates per-image cost by a quarter to a half (5.48 against 8.40 ms
  over the corpus), so `phase_timings`' absolute numbers are a floor (6).
- A cold first folder run looks like a 33% win for whatever ran second, and an unalternated
  series of six runs decreased monotonically (63). A 20-image subset produced a curve the full
  corpus inverted (74).
- Telemetry through `env_logger` locks stderr, and 32 workers then measure the lock (16).
- `cargo build` does not compile tests -- `cargo clippy --all-targets` and a test run do (71).
  `cargo build -p fcs-cli --example X` builds the example *instead of* the CLI binary, and the
  CLI needs `--gpu-env auto` for `WGPU_POWER_PREF` or `WGPU_BACKEND` to reach it (10). A
  backgrounded `cargo test` reports the shell's exit code (88). `cargo clean -p mozjpeg-sys`
  does not invalidate its cached build-script output (67).
- Without NASM on `PATH`, `mozjpeg-sys` builds its scalar fallback with only a warning, ~2.2x
  slower; CI now installs it and the Windows leg fails without it (67).
- `preprocess_cost.rs` reports 0.00 ms of GPU preprocessing wherever `upload_pays_for_source`
  declines, and then names the GPU the winner (48).
- A rate means something only over the interval the thing ran: 97's first reading counted
  frames from before live detection was switched on.
- Back-of-envelope arithmetic pointed the wrong way three times in one round (27, 92, 93), so
  probe before deciding. And a rejection whose record names its blocker is worth re-running once
  the blocker goes: 89 re-ran 87 and got 9%.

---

## Experiment index

One entry per ID: what was tried, what it found, and when to reopen where that is known.
**Kept**, **rejected**, **premise removed** (another result took away the cost it targeted) or
**blocked**. Sections above carry the full tables for most kept changes. RTX 4090 / D3D12 / FXC
unless the Radeon iGPU is named; "the corpus" is the 1239-JPEG reference folder (~10.7 GP),
and "folder" means `fcs-cli --crop` over it.

### First shader round (0-4)

- **0. Measurement tools - kept.** Baseline GPU compute 0.910 ms: pointwise 673.8 us (26
  dispatches, 74%), depthwise 145.4 (16%), stem 42.0 (4.6%), pool/resize/add 49.1 (5.4%).
  `gpu_pass_breakdown` reports those families; `conv2d_experiment` A/Bs two WGSL files, checking
  outputs within `1e-4 + 1e-4 * |ref|` and timing 50 alternated pairs after 20 warm-ups.
- **1. Pointwise specialisation - kept.** 1x1, unit stride, no padding, one group: A/B/B/A
  0.909 / 0.673 / 0.672 / 0.910 ms, **-0.237 ms** of GPU.
- **2. Depthwise input reuse - kept.** Six loads per row feed four outputs: 0.673 / 0.648 /
  0.646 / 0.674 ms, -0.027 ms.
- **3. Four-channel pointwise tile - kept.** 0.647 / 0.535 / 0.538 / 0.651 ms, -0.113 ms; 1-3
  together 0.910 -> 0.536 ms. `conv2d_experiment` needs output coverage `32 8 4` for the tiled
  shader.
- **4. FP16 storage and subgroups - rejected.** See "What Did Not Work"; both retried in 43-46
  once 41 showed the DXC baseline had been handicapped.

### Measurement and controls (5-10)

- **5. Phase timings - kept (measurement).** Eleven timing guards and `phase_timings`. 0.17 MP:
  detection 1.57 ms; 10 MP: 4.68 ms, 2.65 of it CPU preprocessing (57%). Re-run after 20, 34 and
  37 with guards on the on-device preprocessor: a small-image detection was 0.841 ms, its two
  submits 0.055 + 0.093 ms -- 18%, taken up in 92 -- and input allocation and pool acquisition
  free.
- **6. Workload matrix - kept (measurement).** `workload_matrix`: detection is 64% of per-image
  decode plus detection under 1 MP, 44% at 1-4 MP, 27-28% at 4-16 MP and 19% above; orientation
  and face count matter only through size. On identical 12.2 MP pixels (`decode_formats`) TIFF
  decodes at 0.77 ms/MP, BMP 1.92, JPEG through `image` 3.92, PNG 5.89 and WebP 9.75, about 3x
  libjpeg-turbo; not acted on, since the corpus is all JPEG and libwebp would be a new native
  dependency. Peak memory: 84.
- **7. Noise floor - kept (measurement).** See Rules. Conditions recorded: High performance power
  plan, NVIDIA 610.47, not isolated (27 desktop processes holding GPU contexts). The GPU graph
  was 0.370-0.371 ms over three processes, every family within one tick.
- **8. Profiled against normal execution - kept (measurement).** `pass_overhead`: a timestamped
  pass boundary is 0.83 us, so the 61-pass breakdown overstated work by ~51 us (9%). The
  per-dispatch floor is 1.85 us; small pointwise layers at 11-13 us are latency with no
  parallelism to hide it.
- **9. Native CPU timeline.** Done together with 19.
- **10. A second adapter - kept (routing fix).** The Radeon iGPU (driver 32.0.21043.5001) beside
  the 4090 (32.0.16.1047), on D3D12 and Vulkan, chosen with `WGPU_POWER_PREF`; identical
  fingerprint on all four. The iGPU spends 11.35 ms in GPU compute, 92% of `run_on_device`
  against the 4090's 35%. Deleting the integrated-adapter exemption from the preprocessing cutoff
  took a 10 MP detection 42.46 -> 14.30 ms and the folder ~54 -> 26.6 s (**2.07x**). See "On an
  integrated GPU the shader work is the detection" and Open questions.

### Readback and synchronisation (11-18)

- **11. Request maps before the blocking poll - kept (simplicity).** One wait instead of two; the
  second poll had been 0.003-0.005 ms.
- **12. Pooled staging buffers - rejected.** Allocation -0.039 ms, `readback_wait` +0.046,
  detection unchanged.
- **13. One packed staging buffer - rejected.** Allocation -0.034, wait +0.038. Established that
  host work between the inference submit and the readback wait is free.
- **14. Readback copies encoded with inference - rejected.** +0.07 to +0.09 ms: ~0.084 ms of
  staging work moved in front of the submit and delayed the GPU.
- **15. Map on submit - rejected on its ceiling.** `readback_map` is 0.001 ms.
- **16. Wait on the copy's own submission - rejected.** Under 32 workers `readback_wait` p50 went
  0.229 -> 0.175 ms and p99 18.4 -> 12.9, and throughput from 2 to 32 threads did not move. The
  map callback is now collected with `recv_timeout`. The GUI case is open.
- **17. Reuse CPU output storage - half kept.** Branches come straight off the mapped range:
  -0.011 ms at 0.8 MP, -0.021 at 10 MP. Decoding from the mapping rather than a copy loses: a
  decode-shaped read is 0.031 ms in the mapping against 0.009 to copy plus 0.020 to read.
- **18. Staging ring - premise removed.** One process saturates at ~950-1000 detections/s with
  cross-thread overlap already in place (24). Reopen if that ceiling moves.

### CPU recording and resources (19-25)

- **19 (with 9). Warm CPU profile - kept (one fix).** Per-layer bookkeeping is ~0.9% of CPU and
  not worth pre-resolving. A `Resizer` built per call re-zeroed an 8.1 MB scratch buffer; one per
  thread saves 0.1-0.15 ms on large images. That thread-local later panicked with `RefCell
  already borrowed` under nested rayon, found by 67's folder run and fixed by taking the
  `Resizer` out for the duration. The `quality` and `speed` Criterion cases are CPU inference.
- **20. Cache the small-op uniforms - kept.** `create_buffer_init` is 8.1 us, ten times
  `create_bind_group`: `gpu_record` 0.186 -> 0.103 ms, small-image detection -10%, crops
  identical.
- **21. Bound in-flight inferences - kept.** Four per model (`FCS_MAX_IN_FLIGHT`): GPU pool -58
  to -75% at 16-32 callers, iGPU detections/s +9-14%, iGPU folder -10%, 0 crops differ. See "GPU
  memory scaled with concurrency".
- **22. Cache convolution bind groups - kept.** Once dismissed on the assumption that pooled
  buffers change identity; measured, the pool settles after ~3 inferences (730 hits, 110 misses
  over 25). Recording -60%, detection -6%, and a test asserts the settled hit rate.
- **23. Buffer arenas or dynamic offsets - premise removed by 22.** Every settled inference hits
  all 35 bind groups; recording is 0.028 ms on the 4090 and 0.068 on the iGPU. Reopen if bind-group
  creation reappears in a profile.
- **24. Pool and cache contention - answered.** `samply` at saturation shows no pool, cache,
  workspace or device lock. Plain-thread callers serialise on `single_thread_pool()`, a
  process-wide one-thread pool that only non-rayon callers under 4 MP reach -- not production.
  Rayon dispatch peaks at 1061 detections/s at 16 threads, and four processes reach 1259 against
  one's 994. The limit is the resize.
- **25. Host dispatch plan - premise removed.** Recording is 0.027-0.028 ms for 41 dispatches,
  including what a plan would keep. Reopen if `gpu_record` passes ~0.1 ms.

### Shader geometry (26-34)

- **26. Pointwise workgroup shapes - nothing to take.** Nine shapes against 8x8: level or worse on
  the expensive 160x160 layer, narrow-x shapes 26-69% slower, one reproducible one-tick win on one
  dispatch.
- **27. Pixels and channels per thread - local optimum.** One channel per thread is up to 248%
  slower; eight wins 15-19% on the three largest layers and loses the graph by 23% (0.611 against
  0.538 ms); eight pixels per thread is 15-108% slower.
- **28. Constant channel count - rejected.** One tick either way on the 4090, nothing under DXC,
  and slower under FXC on three of four iGPU layers (+46% at 80x80).
- **29. Cooperative workgroup tiles - not implementable safely on D3D12.** naga lowers
  `workgroupBarrier()` to non-synchronising `GroupMemoryBarrier()`. Reopen when naga emits a
  synchronising barrier for workgroup memory, or for Metal.
- **30. Prepacked pointwise weights - rejected.** One to three ticks slower on most 4090 layers;
  8-15% faster only on the iGPU's 40x40 and 20x20 layers.
- **31. NHWC across a segment - rejected.** Both kernels several times slower on both adapters
  before any conversion (160x160 pointwise +1063% on the 4090).
- **32. Depthwise tiles - rejected.** Eight pixels per thread +50-100%; a 4x2 register tile gains a
  tick or two under FXC and loses 12-34% under DXC on the iGPU.
- **33. Interior depthwise fast path - rejected.** Slower under FXC; its DXC gain was 41's
  mechanism, not the bounds checks.
- **34. Stem tiling - kept.** The stem computed one output channel per thread, gathering the same
  27 inputs sixteen times; the four-channel tile takes it 42.0 -> 18.4 us and the graph 0.396 ->
  0.376 ms, bit-exact.

### Graph fusion (35-40)

- **35. Depthwise, ReLU and pointwise in one dispatch - rejected.** 2.2-2.6x the separate pair on
  both adapters, because each pointwise tile recomputes all 64 depthwise values. Reopen only if a
  fused kernel can share them without a local array or staged workgroup memory.
- **36. The neck's upsample-and-add in one dispatch - kept.** -0.012 to -0.017 ms on the 4090,
  -0.13 to -0.19 on the iGPU; 43 -> 41 dispatches; 0 crops differ. Max-pool into pointwise was
  not tried.
- **37. Head branches computed together - kept.** See "Four head branches are one convolution".
- **38. Head outputs in decode order - premise removed by 55.** No CPU reorder is left.
- **39. Preprocessing fused into the stem - premise removed.** The whole-source route would
  resample for every stem tap; the bytes route's phase is 0.18-0.22 ms including an upload fusion
  keeps, and 92 measured moving that dispatch as a loss. Reopen for a third preprocessing route.
- **40. Intermediate lifetimes - kept.** The backbone stopped returning two 6.5 MB outputs the neck
  never reads: GPU pool 7-16% smaller at every concurrency, bit-exact.

### Compilers, backends and precision (41-47)

- **41. The FXC/DXC regression - fixed.** See "The DXC regression was a local array". Found by
  compiling naga 30.0.1's HLSL with both Windows SDK 10.0.28000.0 compilers.
- **42. D3D12 against Vulkan - no change.** Identical output. Vulkan costs 35% more GPU compute on
  the 4090 and 75% more on the iGPU; its adapter comes up in 6-10 ms against 504-807, and its
  driver cache lets a warm launch reach a first face in 293-354 ms against 732-1215. Vulkan stays
  excluded on Windows for the Intel ICD crash (80). Reopen for launch-per-image workloads, with
  evidence that driver is fixed.
- **43, 44. Subgroups after 41 - rejected.** Channel reduction still wins only the 4090's small
  layers (-42 to -74%, about 25 us of graph) and is 442-3122% slower on the iGPU; it would also mean
  shipping DXC. 44's broadcast sharing goes with it. Reopen for hardware where subgroup
  collectives are cheap and detection binds.
- **45, 46. FP16 after 41 - rejected.** f16 arithmetic misses the 1e-3 raw-error screen
  (0.002-0.0068) and is 19-58% slower on the iGPU; f16 storage is 13-50% slower there. 46 was
  gated on a useful 45.
- **47. Native runtimes - checked, no change.** CUDA and TensorRT are excluded by product decision
  (nothing for the user to install); DirectML measured ~9.8 ms against the then 8.2 ms graph and
  adds ~38 MB; the bundled ONNX Runtime CPU path already beats this iGPU; no CoreML hardware.

### Preprocessing and upload (48-54)

- **48, 49. Threaded source resize - kept.** `fast_image_resize`'s rayon feature above 4 MP (not for
  Nearest): -0.61 ms at 10 and 22 MP, byte-identical output. There is no redundant source
  conversion to remove (49). A second pass stopped zeroing the 4.9 MB BGR/CHW buffer, -0.1-0.2 ms.
  On rayon workers the gate now always threads (88).
- **50. Upload resized bytes, convert on the GPU - kept.** The 640x640 result goes up as 1.2 MB of
  bytes and `rgb_to_chw.wgsl` writes the tensor: -0.6 to -1.0 ms on large images, bit-exact. Its
  test caught a `COPY_BUFFER_ALIGNMENT` panic that 640x640 never hits.
- **51. Cheaper resize algorithms - rejected.** `SuperSampling` is slower (+0.17 to +1.43 ms);
  `Interpolation` is 0.7-2.0 ms faster and moves landmarks up to 35.39 px. See "The resize is at
  its floor".
- **52. GUI caches - all three dead, deleted.** The detection cache was written and never read and
  kept up to 50 decoded images alive (~1.3 GB); the other two were never used.
- **53. Webcam frames on the GPU - premise removed by 95.**
- **54. Adaptive routing - kept (1.75 MP cutoff).** See "The preprocessing route crosses over at
  1.75 MP" and "The `Speed` resize setting costs faces".

### Detection output (55-59)

- **55. Output conversion and decode - kept.** Decoding the GPU's channel-major logits directly
  removed a transpose of all twelve heads: **-0.15 to -0.2 ms**, 11-15% of a small-image detection.
  `Tensor::from_vec` instead of copies, -0.02 to -0.03 ms. A pixel-major reorder loop measured
  slower and was reverted.
- **56. GPU decode - premise removed.** Decode was 0.072-0.082 ms after 55, and 93 cut it to
  0.014. Reopen for a much larger input or anchor count.
- **57. Compact survivors before readback - rejected.** See "The head readback is not paying for
  its bytes".
- **58. NMS at worst-case counts - kept.** A fixed 32x32 grid over the scene bounds made a tight
  cluster insert each box into ~640 cells: sizing cells to the mean box took clustered NMS at
  n=5000 from 23.1 to 0.25 ms, and a bitmap instead of `Vec::remove` took `dedup_close_centers`
  from 6.1 to 0.009 ms. Output unchanged.
- **59. Approximate pruning - premise removed with 56.** No time in the stage to trade recall for.

### Batch and webcam scheduling (60-66)

- **60. Worker count - default confirmed.** One worker per logical processor: 7.82 s against 8.90 at
  16 workers, flat above 32. Reversed 63's answer after per-image CPU fell 2.4x; 84 has the memory
  price.
- **61. Producer/consumer pipeline - premise removed.** Rayon already overlaps every stage across
  images, and the iGPU's contended stage is bounded by 21.
- **62. True batches - premise removed.** On the iGPU 91.5% of an inference is arithmetic a batch
  cannot remove; on the 4090 detection (~1000/s) outruns the ~190 images/s a folder supplies.
- **63. Thread budgets - no change.** At the time 16 workers beat 32 by ~8% (since reversed, 60).
  Skipping the inner resize threading on rayon workers is 27% slower (15.75 against 21.35 s).
- **64. GPU submission worker - not built.** The contention it waited for appeared only on the iGPU,
  and a counting gate on the callers (21) recovered it.
- **65. Fresh webcam frames - premise removed by 95.** Drain-to-latest already existed.
- **66. Duplicate preview work - kept.** See "A threshold edit no longer rebuilds the detector".

### Decode, CPU execution and export (67-73)

- **67. JPEG decoders - kept (libjpeg-turbo).** See "Decode, not detection, sets what a folder
  costs". Its folder validation found 19's `RefCell` regression.
- **68. File I/O - skipped by its own gate.** Warm reads are 0.17 s of a 7.1 s folder at 32 threads;
  cold reads could not be measured on this machine.
- **69. CPU backends - kept (thread default).** ONNX Runtime 7.03 ms p50 against `cpu-graph` 29.70
  (4.2x; 204.7 against 90.8 images/s), so `Auto` is right. Four intra-op threads: single CPU
  inference 7.27 -> 4.17 ms, `--no-gpu` folder unchanged.
- **70. Vectorisation and PGO - nothing to vectorise.** See "Where a folder job's CPU actually goes".
  PGO was not measured, because removing 4% of CPU (98) moved no wall time.
- **71. GPU batch cropping - deleted.** See "The biggest saving found is switching cropping off the
  GPU".
- **72. Export encoding - no setting change.** PNG `fast` is 16x quicker in isolation but moved the
  folder 6.90 -> 7.00 s for 14% more bytes; `best` costs 26% of wall time for 2% fewer bytes.
  Encoding borrows RGBA8 crops instead of cloning them (byte-identical; speed not measurable).
- **73. Two-stage decoding - premise removed by 90.** The screening decode moves landmarks past
  35 px wherever it saves time. Reopen with a screening decode whose downscale matches the
  convolution resize.

### Model and algorithm changes (74-79)

- **74. Input resolution - premise removed at the time.** Only `cpu-graph` could vary the input.
  Its corpus sweep: 320 -> 907 faces in 7.6 s, 480 -> 1006 in 12.1, 640 -> 1032 in 18.4, 800 ->
  1041 in 31.4, 960 -> 1070 in 59.1, with no ground truth on whether the extra faces are real.
  `probe_input_size` now rejects an unrunnable size at construction.
- **75, 79. A 320 first pass - rejected.** The fixed input was a hardcoded stem and a fixed-shape
  export, not a retraining problem: the stem now sizes from its input, and
  `face_detection_yunet_2023mar_320.onnx` is the same weights through `onnxsim` (CI only).
  `coarse_to_fine` against production's 1130 faces:

| Strategy | Faces lost | Landmarks > 35 px | iGPU | 4090 |
| --- | ---: | ---: | ---: | ---: |
| 320 alone | 158 | 1512 of 4860 | 2.54x | 1.23-1.29x |
| confidence cascade, t 0.3-0.7 | 9-90 | 1269-1477 | 1.48-2.16x | 0.95-1.26x |
| refine each candidate on a crop, t 0.3-0.7 | 39-102 | 1320-1451 | 1.02-1.20x | 0.48-0.61x |
| no-face screen, t 0.3-0.7 | 7-71 | 0 | 0.80-0.88x | 0.61-0.67x |

  Keeping any 320-derived landmark fails the 35 px bar at more than ten times INT8's rate. The
  screen keeps production's detections but still escalates 74-86% of images, and pays only where
  about 40% (iGPU) or 80% (4090) of images yield no candidate. 105 of 320's 158 losses are faces
  of 64+ model pixels that it scores under 0.8. Trained screeners stay blocked, as 78.

- **76. Webcam tracking - premise removed by 95 and 97.** Detection fits every frame: 6.89 ms of a
  69 ms interval in the GUI, 12.2 ms on the iGPU.
- **77. INT8 - rejected.** See "INT8 is slower on the CPU path, and moves landmarks".
- **78. Smaller models, pruning, distillation - blocked.** Needs a training pipeline and labelled
  data.

### Startup, caching and lifetime (80-85)

- **80. Cold start split - answered.** See "Cold start is 900 ms, and model loading is 0.15% of
  it".
- **81. Detector built off the GUI's first frame - kept.** First frame 894 -> 737 ms over 24
  alternated launches, detector usable at the same time; a file dropped during the build no
  longer reports a missing model. Device loss and model switching are untested.
- **82. Pipeline caches - kept (one entry point per kernel).** `PIPELINE_CACHE` is Vulkan-only; see
  "One entry point per kernel halves shader compilation".
- **83. Cache growth over many input sizes - premise removed.** The pools stay at 44.1 MB (84) and
  production runs one input size. Since 75 a 320 input is runnable; the bind-group cache clears
  past 512 entries.
- **84. Pool retention - no change.** See "Peak memory scales with worker count, and nothing else
  does".
- **85. Sustained operation - no drift.** 50 passes in one process (20,000 detections): 14.43 ->
  14.39 s per pass, no host RSS slope, GPU pool 44.1 MB and identical detections throughout. 15
  consecutive folder jobs: 1129 faces and 959 crops every run, wall slope -39 ms per run (cache
  warming, not throttling).

### Later findings (86-98)

- **86. Detect from a reduced-scale decode - measured in 90.** 82% of the folder needs the full
  decode for its crop regardless.
- **87. Batch profile - no serial bottleneck.** 194.7 s of CPU across 51 threads in ~16 s, the top
  ten workers within 15%. The quality metric's downscale swap was neutral because it converted
  RGBA to RGB first (fixed by 88, retried in 89).
- **88. RGBA crop resize, and the gate's pool hop - kept.** From a rayon worker, "don't thread"
  meant a cross-registry `install` hop costing ~18% of folder wall time; the gate now threads on
  workers. With an RGBA `fast_image_resize` path for crops: 9.58 -> 8.17 s.
- **89. Quality metric through the RGBA resize - kept.** Folder 7.77 -> 7.10 s (9%), crops
  byte-identical; the JSON report's per-detection `quality_score` moves up to 6% (p95 2%) with no
  label flips on the corpus.
- **90. Detect from a scaled JPEG decode - rejected.** See "The source resize is now a third of a
  folder job".
- **91. Single-image latency - kept (preview clamp).** A 12 MP detection is 3.39 ms (48% of it the
  resize) against a 27 ms decode, which closed the GPU-overhead chain for that workload. The
  clamp for previews over 8192 px used `resize_exact`: 3215 ms against 103 through
  `fast_image_resize` on a 133 MP image.
- **92. Preprocessing in the inference pass - rejected.** It saved 60 us of host time, but the
  preprocess dispatch had been running while the host recorded inference; merged, the wait grew
  ~43 us and lost four pairs of five. `encode_cost`: an encoder plus pass ~30 us, a submit ~26.
- **93. Skip decoding cells below the threshold - kept.** See "The decode was three-quarters
  exponentials".
- **94. A deadline on GPU waits - kept; its premise was wrong.** Every wait goes through
  `wait_for_gpu` with a 30 s deadline. The hang behind it was the test harness opening a D3D12
  device per test on 32 threads, stalling inside the NVIDIA driver; one device per test binary:
  `fcs-utils` suite 3.8-4.2 -> 1.5 s, 0 hangs in 70 runs.
- **95. The webcam path - capture-bound, two defects fixed.** See "The webcam loop is
  capture-bound".
- **96. Letterboxing - kept.** See the webcam section and "Letterboxing is free".
- **97. Live webcam detection in the GUI - kept.** See "Live webcam detection".
- **98. Blit the crop region - kept.** See "Where a folder job's CPU actually goes".

---

## References

Primary sources behind the shader and readback experiments -- technique support, not evidence of
a speedup in this application:

- ONNX Runtime WebGPU [convolution selection](https://github.com/microsoft/onnxruntime/blob/main/js/web/lib/wasm/jsep/webgpu/ops/conv.ts),
  [depthwise implementation](https://github.com/microsoft/onnxruntime/blob/main/js/web/lib/wasm/jsep/webgpu/ops/conv-grouped.ts)
  and [packed matmul](https://github.com/microsoft/onnxruntime/blob/main/js/web/lib/wasm/jsep/webgpu/ops/3rd-party/matmul_packed_webgpu.ts).
- [Chrome's WebAssembly and WebGPU measurements](https://developer.chrome.com/blog/io24-webassembly-webgpu-2):
  FP16, subgroup and memory-access gains vary by GPU.
- wgpu [features](https://wgpu.rs/doc/wgpu/struct.Features.html),
  [mapping on submit](https://docs.rs/wgpu/30.0.1/wgpu/struct.CommandEncoder.html#method.map_buffer_on_submit)
  and [buffer mapping](https://docs.rs/wgpu/30.0.1/wgpu/struct.Buffer.html#mapping-buffers).
- [ONNX Runtime quantization guide](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).
