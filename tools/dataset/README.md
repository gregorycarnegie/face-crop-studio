# Dataset tools and the native migration

The target is a Python-free toolkit, **including training**, not Zig launchers around Python.
The application and release packages already run without Python; this migration concerns the
offline tools used to prepare data, train, export and compare models.

## Completed ONNX tools

These use the same Zig **0.17.0** toolchain pinned in CI and have no third-party dependencies:

```sh
zig run tools/dataset/scrfd_topology.zig -- models/scrfd80k_500m_640.onnx topology.rs
zig run tools/dataset/stamp_model_metadata.zig -- scrfd model.onnx
zig run tools/dataset/stamp_model_metadata.zig -- eye_refiner model.onnx

zig test tools/dataset/onnx.zig
zig test tools/dataset/scrfd_topology.zig
zig test tools/dataset/stamp_model_metadata.zig
```

The topology generator preserves the executor's one-based slots, named weights, fused ReLUs
and nine raw head outputs. Format its output with `rustfmt` before comparing with the
committed Rust table. It targets the existing folded SCRFD graph, not arbitrary ONNX networks.

The metadata stamper copies the graph and unknown fields byte for byte, keeps custom metadata,
replaces existing provenance keys, and writes through an atomic file replacement. It validates
protobuf framing, not ONNX operator semantics. The exporters still verify outputs against the
checkpoint after stamping; the standalone stamper expects an already valid ONNX model.
The `produced_by` metadata continues to name the exporters that actually produced the model.

Both exporters are Zig now and stamp the model in-process before verifying it, so the training
environment needs no Zig of its own: export from the host, pointing at the checkpoint.
Re-stamping retains the existing `produced_by` field; a new exporter sets its own provenance.

## Native checkpoint reader

`checkpoint.zig` now reads the existing refiner `.pt` and SCRFD `.pth` checkpoints directly,
including the optimizer state, without LibTorch or Python:

```sh
zig run -O ReleaseFast tools/dataset/checkpoint.zig -- checkpoint.pth > checkpoint.json
zig test tools/dataset/checkpoint.zig
```

The importable `load` function returns dictionaries, lists, tuples, scalar metadata, tensor
views and shared storages. `Tensor.floats` materialises a view in logical row-major order.
The CLI emits tagged JSON, including tensor hashes, source devices and dictionary attributes;
integer optimizer keys and tuples are retained rather than flattened into JSON strings/lists.
It reads the source file without modifying it or executing pickled objects.

Verified against PyTorch on the existing `eye_refiner_val.pt` (64 tensors) and SCRFD's
`epoch_100.pth` (446 tensors). Every logical tensor-byte hash, shape, stride, offset, storage
alias, source device, model attribute, optimizer field and scalar metadata matched. CI uses
the small synthetic archive in `testdata/`, including non-contiguous views and long memo IDs.

The supported format is the one those checkpoints use: protocol 2, stored PyTorch ZIP
members, float32/int64 storages and little-endian tensor data (also the legacy default when
the byte-order member is absent). Other protocols, storage types, compression methods and
Python globals fail explicitly. ZIP CRCs and tensor extents are checked; JSON reports have
depth and size limits to bound expansion through pickle aliases. This is a checkpoint data
reader, not a general Python pickle interpreter or a native training engine.

## Native eye-refiner export

`export_eye_refiner.zig` replaces the Python exporter. It loads the legacy checkpoint, exports
the graph itself, and checks ONNX Runtime against independent, unfused checkpoint inference
on a batch of three crops before atomically replacing the output. Missing runtimes, invalid
parameters, non-finite outputs and tolerance failures leave an existing output untouched.
It prints checkpoint selection statistics and retains the CC BY 4.0 attribution.

```sh
# Unix: link libc for the platform's dynamic loader. On Windows, omit -lc.
zig run -O ReleaseFast -lc tools/dataset/export_eye_refiner.zig -- eye_refiner_val.pt out.onnx
zig test tools/dataset/refiner.zig
zig test tools/dataset/ort.zig
zig test tools/dataset/export_eye_refiner.zig
```

Set `ORT_DYLIB_PATH` to an ONNX Runtime library, as for the Rust parity tests, or pass
`--ort <library>`. The native oracle requires API 24 or newer. Neither Python packages nor
LibTorch are needed. `--tolerance` retains the default `1e-5` maximum absolute error.
The supported model is the existing 112-pixel, width-32 architecture; incompatible checkpoints
fail explicitly. The exporter fixes the initializer names consumed by the Rust refiner,
retains the dynamic batch axis, and records `export_eye_refiner.zig` as the actual exporter.

The validation-selected real checkpoint passed the official ONNX checker and independent
PyTorch comparisons at batches 1, 3 and 5 (maximum error `1.79e-7`). CI exports the synthetic
`testdata/refiner.pt` on every native platform using the downloaded ONNX Runtime, without
Python. This does not regenerate or replace the shipped model or its published checksums.

## Images and the detector benchmark

`image.zig` decodes and encodes with stb (`vendor/`, public domain, pinned in `vendor/stb.c`)
and reimplements OpenCV's fixed-point `resize` (INTER_LINEAR, INTER_AREA) and `warpAffine`.
Tools that use it link the C file:

```sh
zig run -O ReleaseFast -lc tools/dataset/detector_bench.zig tools/dataset/vendor/stb.c -- coco <data> <models> <cache>
zig test -lc tools/dataset/image.zig tools/dataset/vendor/stb.c
```

Against `cv2` 4.10 on Open Images photographs, given the same pixels: letterbox resizing,
halving, INTER_AREA and `warpAffine` are bit-identical; upscaling differs by one level on 0.02%
of values. Decoding is where the two differ: stb and libjpeg disagree on about 11% of values,
by up to 28 levels at sharp colour edges. EXIF orientation is applied as `cv2.imread` does.

`detector_bench.zig` replaces `detector_bench.py`. Re-scoring the existing COCO and Open Images
caches reproduces the Python tables character for character. Fresh detection on 150 COCO
images left every count above the frozen thresholds unchanged: no detection crossed a
threshold, scores moved by about 3e-4 and coordinates by about 0.1% of the box width.

`eval_refiner_vs_yunet.zig` and `refiner_contact_sheet.zig` replace their Python versions. They
run the checkpoint through the graph `export_eye_refiner.zig` builds, on ONNX Runtime, so torch
and a GPU are no longer needed. Against the Python versions on `eye_refiner_val.pt` and the
Open Images corpus: the paired comparison matches on all 1,382 faces in every share and count,
with the refiner's median angle error 1.17 against 1.18 degrees; the contact sheet picks the
same 10,880 faces and the same 40 worst pairs (one adjacent near-tie swapped), and its tiles
look identical.

`make_accuracy_fixtures.zig` replaces its Python version. Rebuilt into a scratch folder, it
chose the committed 60 images and wrote `ground_truth.json` and `ATTRIBUTION.md` identical to
the committed files (apart from the generator's own name), with the same image sizes and a
pixel-identical `test_pattern.png`; the JPEGs differ by 0.67 levels on average, the gap between
two encoders.

## Native SCRFD export, detection and evaluation

`export_scrfd.zig` replaces the Python exporter and needs neither torch nor mmdet. `scrfd.zig`
reads the architecture out of the checkpoint (channels, groups and block counts from the tensor
shapes; the wiring is mmdet's MobileNetV1 + PAFPN + SCRFDHead) and runs an independent forward
pass with BatchNorm unfolded. The exporter writes that program as an opset-11 graph, folds each
BatchNorm in float32 as numpy did, keeps every weight's checkpoint name, and refuses to write
unless ONNX Runtime matches the unfolded forward pass (default tolerance `1e-4`).

```sh
zig run -O ReleaseFast -lc tools/dataset/export_scrfd.zig -- epoch_100.pth scrfd.onnx
zig run -O ReleaseFast -lc tools/dataset/scrfd_detect.zig tools/dataset/vendor/stb.c -- epoch_100.pth <images> out.json
zig run -O ReleaseFast -lc tools/dataset/eval_eye_error.zig tools/dataset/vendor/stb.c -- epoch_100.pth <openimages> --thresh 0.02
```

Exported from `work_dirs/oi80k/epoch_100.pth`, all 120 weights are bit-identical to the shipped
`scrfd80k_500m_640.onnx` and ONNX Runtime gives identical outputs; the official checker
passes. `scrfd_detect.zig` and `eval_eye_error.zig` run that in-memory export through
`Detector`; against the Python originals (torch on CUDA) they found the same 189 faces on 150
COCO images and matched the full eye-error sweep to within 0.1 points, three to eight times
faster without a GPU. The difference is stb decoding, as above.

CI exports the synthetic `testdata/scrfd.pt` and checks that `scrfd_topology.zig` regenerates
the committed `fcs-core/src/scrfd/topology.rs` from it, which pins the graph's wiring and weight
names to the plan the Rust engines run.

## Native eye-refiner training on the GPU

`train_eye_refiner.zig` replaces the PyTorch trainer. The network runs on an NVIDIA GPU through
the driver alone: `cuda.zig` loads `nvcuda.dll` / `libcuda.so.1` at run time, and the kernels
are Zig (`gpu/refiner_kernels.zig`), compiled by Zig's own NVPTX backend to the committed
`gpu/refiner_kernels.ptx`, which the driver JIT-compiles for the GPU present (compute
capability 8.0+). No CUDA toolkit, cuDNN, Python or torch. CI regenerates the PTX from the Zig
source and fails if the committed file differs.

```sh
zig run -O ReleaseFast -lc tools/dataset/train_eye_refiner.zig tools/dataset/vendor/stb.c -- --check
zig run -O ReleaseFast -lc tools/dataset/train_eye_refiner.zig tools/dataset/vendor/stb.c -- <openimages> --out eye_refiner.pt
```

The recipe is the PyTorch original's: the same image-level validation split (Python's
`random.Random(1234).shuffle` reproduced, giving the same 1,684 / 304 / 1,562 faces), the same
augmentation, smooth-L1 points plus 0.5 x (1 - cos) angle loss, training-mode BatchNorm with
PyTorch's momentum and unbiased running variance, PyTorch's default initialisation, AdamW under
OneCycleLR with its momentum cycling, selection on validation and one look at the test set.
Crops are cut on the CPU with OpenCV's warpAffine arithmetic (`image.zig`), in parallel.
It writes a PyTorch ZIP checkpoint (`checkpoint.save`) that `torch.load` reads with every key
matching, and that `export_eye_refiner.zig` exports and verifies; the checkpoint records
`trained_by`, which the export carries into its provenance.

`--check` verifies the GPU network without data: eval-mode output against `refiner.zig`'s CPU
reference (agreement 7e-8), training-mode gradients of every layer type against central finite
differences (36 of 36), and the loss gradient. A 300-epoch run takes 9.3 minutes on an RTX 4090.

Three native seeds against two PyTorch runs, on the test set with ground-truth boxes (the
checkpoint each run selected on validation):

| Run | Validation pick | Test angle median | Within 5 deg |
|---|---|---|---|
| PyTorch, shipped model | 1.20 deg, epoch 280 | 1.39 deg | -- |
| PyTorch, seed 1 | 1.22 deg, epoch 280 | 1.41 deg | 87.0% |
| Zig, seed 0 | 1.31 deg, epoch 290 | 1.51 deg | 87.6% |
| Zig, seed 1 | 1.34 deg, epoch 300 | 1.49 deg | 86.3% |
| Zig, seed 2 | 1.29 deg, epoch 270 | 1.40 deg | 88.5% |

On the paired YuNet-box comparison that the project reports, the Zig seed-0 checkpoint scores
1.24 deg median (92.1% within 5 deg, closer than YuNet on 78.8% of faces) against the shipped
model's 1.17 deg. So the native trainer lands about 0.07 deg behind PyTorch on the test set
(0.1 deg on validation), consistently across seeds. The cause is not identified: the recipe
matches line by line, and the remaining differences are stb rather than libjpeg decoding and
cuDNN's default TF32 convolutions. Training does not checkpoint optimiser or schedule state, so a run cannot be
resumed; it restarts. The shipped model is unchanged.

## Remaining scope

Two Python files remain, both SCRFD training configurations consumed by upstream MMDetection:

| Current files | Native replacement needed |
|---|---|
| `scrfd_fcs_500m.py`, `scrfd_fcs80k_500m.py` | A native SCRFD trainer replacing upstream MMDetection, not a change of config syntax |

`run_curve.sh` must move to the native SCRFD trainer too: it currently invokes the external
checkout's Python `tools/train.py`. Renaming the two configuration files alone would break it.

The GPU path the refiner trainer proved (`cuda.zig`, Zig kernels compiled to PTX, finite-
difference checks) is the foundation for it. What the SCRFD trainer must reproduce:

* **Model.** MobileNetV1, PAFPN and the three SCRFD head branches with training-mode
  BatchNorm (not the folded inference graph `scrfd.zig` builds), depthwise convolutions,
  nearest upsampling, and its checkpoint names, so `export_scrfd.zig` reads the result.
* **Targets and losses.** Two anchors per location at strides 8/16/32, ATSS assignment with
  top-k 9 (ignore IoF 0.5 in the 80k recipe), Quality Focal Loss, DIoU box loss, Smooth L1
  keypoint loss with unlabelled landmarks ignored.
* **Data.** Square cropping with the recipe's crop choices, resize to 640, flips that swap eye
  identity, photometric distortion, every real face box and the ignore regions.
* **Schedules.** The label-count recipe (SGD 0.01, warm-up, steps at 55/68 of 80 x 8 epochs)
  and the 80k run's 100 epochs with drops at 69/85.
* **Acceptance.** Assignments, masks, losses and one optimiser step compared with MMDetection
  on fixed batches; then a full run scored with `eval_eye_error.zig` and `detector_bench.zig`
  on both corpora against the shipped model's figures.
