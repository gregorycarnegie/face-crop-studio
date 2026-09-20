# Can a licence-clean detector beat YuNet? 80,000 images, measured

The question left open by `CURVE_RESULTS.md`: that document trained four SCRFD-500M models on
2,482 images and none of them beat YuNet 2023, concluding that images rather than eye labels
were the constraint. This is the run that tests it -- 80,000 Open Images photographs, about the
scale of WIDER FACE, which is what YuNet and SCRFD were both trained on.

**Status: complete.** 100 epochs on an RTX 4090, 19-20 September 2026, ~20.4 hours of compute.

## The answer

Every figure below is scored on the Open Images validation+test split with *complete* boxes
(`labelv2_test_all.txt`, 4,597 real faces over 1,859 images), and compared at matched
false-positive rates rather than at each model's favourite threshold.

| | All-face recall | FP/image | Clicked-face recall | Eye-line angle |
|---|---|---|---|---|
| YuNet 2023 (0.8, its production threshold) | 71.6% | 0.14 | 88.5% | **4.05 deg** |
| Curve SCRFD, 2,482 images (0.2) | 73.7% | 0.41 | 89.9% | 5.31 deg |
| **80k run, epoch 100 (0.4)** | **86.0%** | **0.11** | **95.3%** | 5.95 deg |
| 80k run, epoch 100 (0.5) | 81.8% | 0.07 | 93.2% | 5.88 deg |

**It finds 14.4 points more faces than YuNet at fewer false positives**, and at 0.5 it still
leads recall by 10 points with half the false positives. The full sweep:

| Threshold | Clicked recall | All-face recall | FP/image | Angle median | Eye distance |
|-----------|---------------|-----------------|----------|--------------|--------------|
| 0.20 | 97.2% | 91.1% | 0.51 | 5.98 deg | 0.119w |
| 0.30 | 96.5% | 88.6% | 0.21 | 5.96 deg | 0.119w |
| **0.40** | **95.3%** | **86.0%** | **0.11** | 5.95 deg | 0.118w |
| 0.50 | 93.2% | 81.8% | 0.07 | 5.88 deg | 0.117w |
| 0.60 | 89.6% | 76.6% | 0.04 | 5.72 deg | 0.114w |
| 0.70 | 78.2% | 64.2% | 0.02 | 5.42 deg | 0.109w |

Epoch 80 already read 85.2% at 0.11, so the last 20 epochs and the final learning-rate drop
added 0.8 of a point: the run had converged, and a shorter schedule would have done.

**The eye points are worse, and that is by design.** 5.95 deg against YuNet's 4.05, because
2,500 of 244,683 training faces carry clicked eyes and the landmark head is barely supervised.
It does not matter: `fcs-core::EyeRefiner` replaces those two points at 1.18 deg median
(`CURVE_RESULTS.md`), so the detector's job is finding and boxing faces. This run is better at
that job; the refiner is better at the eye line than either detector.

## What made the difference

Two changes, and they cannot be fully separated:

1. **32x the images.** 80,000 against 2,482.
2. **Every face boxed.** The curve's label files boxed only the clicked faces -- 2,500 boxes in
   images where Open Images drew 12,070 -- so each curve run was taught that ~9,300 real faces
   were background. `oi_to_labelv2.py` writes every box; see the correction at the top of
   `CURVE_RESULTS.md`.

Whoever wants the split can have it for ~7 hours of GPU: train the 2,482 images again with
complete boxes and compare. Nothing here needs the answer, so it was not spent.

## The data

| | |
|---|---|
| Images | 80,000, all CC BY 2.0, all with an author recorded |
| Faces boxed | 244,683 real faces (2,500 with clicked eyes) |
| Ignore regions | 11,607 -- 5,695 group-of boxes and depictions, 5,912 YuNet detections Open Images never boxed |
| Selection | train split only, >=1 face at >=5% of image width, `Rotation` recorded as 0.0 |
| Attribution | `train_manifest.csv`, one row per image: licence, author, author URL, landing page |
| On disk | 25 GB, copied to WSL ext4 -- 80,000 files through the 9p bridge starves the loaders |

`Rotation` is left out rather than handled: ~1% of train images are marked 90/180/270 and 14%
record nothing, and nothing in this pipeline reads the column, so rather than guess which frame
the boxes were drawn in those images are skipped. The eligible pool was 236,495, three times
what was needed.

**Ignore regions do nothing unless the assigner is told.** `RetinaFaceDataset` loads a 5-value
line into `gt_bboxes_ignore`, the pipeline passes it, `scrfd_head` hands it to the assigner --
and `ATSSAssigner` defaults `ignore_iof_thr` to -1, which skips it. The label file, the loader
and the pipeline can all be right and every ignore region still trains as background.
`scrfd_fcs80k_500m.py` sets it to 0.5 on the assigner inside `bbox_head.train_cfg`, which is
the one the head actually calls.

## The schedule, and what set the speed

Upstream trains WIDER's 12,880 images for 640 epochs, ~8.2 M image passes. Keeping the epoch
count would have cost six times that, so the *budget* was kept instead: 100 epochs over 80,000
images, learning-rate drops at the same fractions of the run (epochs 69 and 85).

Batch 64 at 12 loaders, chosen from measurement rather than taste:

| | Throughput | Iteration | VRAM |
|---|---|---|---|
| Batch 16, 8 loaders | 96 img/s | 0.167 s | 2.0 GB |
| **Batch 64, 12 loaders** | **103 img/s** | 0.62 s | 8.0 GB |
| Batch 64, 24 loaders | no iteration logged in 4 minutes | -- | -- |

The card was never the constraint: 8 GB of 24 used, and the same ~100 images/s at either batch,
with 0.13-0.14 s of each iteration spent waiting on data. So the batch was picked for
optimisation instead -- 64 gives 125,000 steps, near upstream's WIDER recipe of ~64,000 at
batch 128 with this same 0.01 learning rate, where batch 16 would have taken 500,000. Going
faster means a faster data pipeline, most obviously pre-resizing the JPEGs so each epoch stops
decoding them at full resolution.

24 loaders first died on `RuntimeError: received 0 items of ancdata` -- torch passing tensors
between workers through file descriptors against the open-file limit -- and then, with the limit
raised, simply started nothing in four minutes.

## Off home turf: the VinaSkyy corpus, judged by eye

Open Images is home turf for this model and away turf for YuNet, so the margin above is partly
that. The check is a corpus neither has seen: 1,239 photographs in `Downloads\VinaSkyy`, no
labels, so `compare_detectors.py` matches the two models at IoU 0.5 and a human judges every
disagreement as a crop (`scrfd_detect.py` writes the checkpoint's detections in `fcs-cli --json`
shape first).

At the 0.4 threshold matched on Open Images FP/image, SCRFD found 1,323 faces to YuNet's 1,129:
1,120 found by both, 9 only by YuNet, 203 only by SCRFD, and 166 of those 203 over 200 px wide,
so not small-face noise. **Counting is not judging, though, and the judgement was less
flattering.** Of the 9 YuNet-only detections, 5 were faces and 4 false positives. Of the 60
largest SCRFD-only detections, 25 were faces, 32 were false positives and 3 were animals -- over
half of its extra finds wrong, at sizes large enough to produce a confidently wrong crop.

**Score separates them, which is what makes this fixable.** The judged faces score 0.634 at the
median against 0.456 for the false positives:

| Threshold | Judged faces kept | False positives kept | Animals kept |
|-----------|-------------------|----------------------|--------------|
| 0.40 | 25 | 32 | 3 |
| **0.50** | **23** | **7** | 1 |
| 0.60 | 16 | 2 | 1 |
| 0.70 | 3 | 0 | 0 |

Across the whole corpus that reads:

| SCRFD threshold | Faces found | Both | Only YuNet 0.8 | Only SCRFD |
|-----------------|-------------|------|----------------|------------|
| 0.40 | 1,323 | 1,120 | 9 | 203 |
| **0.50** | **1,251** | 1,116 | 13 | 135 |
| 0.60 | 1,180 | 1,093 | 36 | 87 |

**0.50 is the operating point**, chosen on this corpus rather than on Open Images: it keeps 23 of
the 25 judged faces while dropping 25 of the 32 false positives, and on Open Images it still
reads 81.8% of all faces at 0.07 FP/image against YuNet's 71.6% at 0.14. So the detector is
better off its own turf too -- 135 unique finds against 13, at roughly three-quarters precision
on the large ones -- but the useful margin is a good deal smaller than the raw 203-to-9 suggests.

Two things that judgement does not cover. Only the **60 largest** of the 203 were shown and
judged, so the precision figure belongs to large detections and says nothing about the remaining
143, which are smaller. And **animals**: three of the extras were a pet's face. The training
labels are `Human face` boxes only, so a dog scoring 0.47 is a false positive by this project's
definition, and this is the first measurement of how often that happens.

## What shipping it would take

Scoped 20 September 2026, by reading the engines rather than estimating.

**The export is not blocked after all, and is done.** `tools/scrfd2onnx.py` fails under torch
2.x because it builds its dummy input through mmdet's pipeline, which hands the tracer numpy
arrays. The model traces fine when handed a plain tensor, so `export_scrfd.py` wraps
`feature_test` -- extract features, run the head, return the raw per-stride tensors -- and
exports that. No NMS in the graph, which is the part a tracer chokes on anyway.
`scrfd80k_500m_640.onnx`, 2.5 MB, matches torch to **3.8e-06**.

Two traps in verifying it, both of which first looked like a broken export. Under
`torch.onnx.is_in_onnx_export()` the head switches to the deployment layout, so each
`(1, A*C, H, W)` map becomes `(1, H*W*A, C)`; and the class maps get a sigmoid the eager path
does not apply. Compare raw against exported and the "error" is 6.7. Reshape and sigmoid the
reference first and it is 3.8e-06. That layout is what upstream's wrapper and
`eval_eye_error.py` already decode.

**Cost: 6.4 ms/image against YuNet's 5.8 on ONNX Runtime CPU at 640x640** -- 10% slower for a
much better detector, and 2.5 MB against 232 KB on disk.

**The op inventory says no new kernels are needed.** The graph is 60 convolutions (40 dense,
20 depthwise), 41 ReLU, 4 Add, 2 nearest Resize, 3 Sigmoid, and no BatchNorm at all -- the
export folded it into the convolution weights. Everything else is Reshape/Transpose/Concat
plumbing, which matters only to a generic interpreter; both engines run a compiled topology and
read weights by name, so that plumbing becomes Rust rather than kernels. Against what exists:

| Needed | GPU (WGSL) | Built-in CPU graph |
|---|---|---|
| 1x1 dense, stride 1 (23) | `Kernel::Pointwise` | pointwise path |
| 3x3 depthwise, stride 1 (16) | `Kernel::Depthwise` | depthwise path |
| 3x3 dense, stride 1 (14) | `Kernel::General` | general path |
| 3x3 depthwise, **stride 2** (4) | `Kernel::Grouped` -- exists, reads stride from its uniforms | general path (`groups`) |
| 3x3 dense, stride 2 (3) | `Kernel::General` (YuNet's own stem) | general path |
| ReLU (41), Add (4), nearest 2x (2) | present | present |
| Sigmoid (3) | `ActivationKind::Sigmoid` | 3 final outputs; plain Rust |

So a port is topology and decode, not infrastructure. For scale: YuNet's own topology is 208
lines of constants plus 67 of macros, executed by 366 (CPU) and 373 (GPU) lines. SCRFD-500M is
bigger -- MobileNetV1 (2,3,2,6 blocks), PAFPN, and a head that does not share weights across
strides -- but the same shape of work, and its decode is *simpler* than YuNet's because
`loss_dfl=False` means direct distance regression with no distribution to integrate.

**The staging that matters.** Replacing YuNet outright would make the app ONNX-Runtime-only,
and a machine without it would have no detector rather than a slower one. Keeping YuNet as the
fallback avoids that entirely: SCRFD when a runtime is present (which every release bundles),
YuNet's built-in graph when it is not. That is the same degradation rule `EyeRefiner` already
uses, it needs no port, and it is what makes the port optional rather than a prerequisite.

* **Phase 1, small:** decode the 9 raw tensors in Rust (anchors at strides 8/16/32, two scales,
  `distance2bbox`, keypoints, then the existing NMS), a detector that selects SCRFD under ORT
  and falls back to YuNet, ship the model through the fetch-and-verify path the refiner already
  uses, parity tests against the numpy decoder.
* **Phase 2, optional:** declare the topology and run it on the CPU and WGSL engines, so the
  better detector no longer needs ORT. Mechanical, and `backend_parity` plus tract (dev-only)
  are the oracles -- though whether tract accepts this graph is untested, and it needed
  `onnxsim` for YuNet.

## What this does not establish

* **Precision on small faces is unmeasured.** See above: the judged sample was the 60 largest
  disagreements of 203.
* **One corpus, one photographer.** VinaSkyy is a single subject shot in similar conditions.
  It answers "does the Open Images margin survive off-distribution" and not "how does this behave
  on arbitrary photographs".
* **Nothing here is shippable yet.** `fcs-core` compiles YuNet's topology into both the built-in
  CPU graph (`crate::yunet`) and the WGSL kernels, so SCRFD would run under ONNX Runtime alone:
  a machine without it would have no detector rather than a slower one. Export is also blocked --
  SCRFD's `tools/scrfd2onnx.py` fails under torch 2.x, its tracer refusing the numpy arrays
  mmdet's input builder passes (see `eval_eye_error.py`, which reads the checkpoint directly for
  this reason).
* **FP/image remains an upper bound.** Open Images boxes are not exhaustive even now: an audit
  put ~97% of YuNet's unmatched detections on real, unlabelled faces (DATA_CARD.md section 1).
  Both models are judged the same way, so the comparison holds, but the absolute figures flatter
  neither.
* **Small faces are in the labels.** 244,683 boxed faces against the 164,000 "croppable" ones
  DATA_CARD.md counts, because everything below 32 px at 640 is boxed too. WIDER does the same.
* **The run lost 3 hours.** It died at epoch 80 in mmdet's evaluation hook, which the launch
  script had failed to disable, and sat idle until someone looked. Training was unaffected and
  it resumed from the epoch-80 checkpoint (TRAINING_SETUP.md fix 12).
