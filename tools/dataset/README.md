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

Both Python exporters now invoke the Zig stamper, so **Zig 0.17.0 must also be on PATH in the
training environment**, including WSL. A Windows-only installation does not put `zig` on WSL's
PATH. A stamping failure stops the export before it can be declared verified.

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

## Remaining scope

Twelve Python files remain. None of the training or inference replacements below is complete.
Keep the existing working tools until their native replacements pass the listed checks.

| Current files | Native replacement needed |
|---|---|
| `compare_detectors.py` | JSON matching, image crops and self-contained disagreement HTML |
| `make_accuracy_fixtures.py` | Deterministic SHA-256 selection, image resizing, scaled boxes and attribution |
| `scrfd_detect.py`, `eval_eye_error.py` | Checkpoint loading, SCRFD forward/decode/NMS, threshold sweep and ground-truth scoring |
| `eval_refiner_vs_yunet.py`, `refiner_contact_sheet.py` | Checkpoint loading, paired refiner inference, affine crops and rotation checks |
| `detector_bench.py` | Model checksum verification, ONNX inference for three detectors, licensed COCO fetching, caches and both scoring protocols |
| `train_eye_refiner.py` | Native forward/backward, augmentation, optimiser/schedule, validation selection and checkpoint I/O |
| `export_eye_refiner.py`, `export_scrfd.py` | Native checkpoint loading, ONNX graph/weight export and independent inference parity |
| `scrfd_fcs_500m.py`, `scrfd_fcs80k_500m.py` | Native SCRFD recipes and training entry point, replacing upstream MMDetection rather than just changing config syntax |

`run_curve.sh` must move to the native SCRFD trainer too: it currently invokes the external
checkout's Python `tools/train.py`. Renaming the two configuration files alone would break it.

## Training implementation and acceptance checks

Use Zig for the data pipeline, model definitions, losses, scheduling, checkpoint selection and
CLI. The smallest route to retaining CUDA training is LibTorch behind a narrow C ABI bridge:
LibTorch is a C++ API, so Zig's C interop does not directly translate its classes. The C++
bridge should provide tensor/autograd/device operations; it must not launch a Python process.
This is a proposed backend, not an installed or implemented dependency of these tools.

A native C++ probe against the existing WSL LibTorch and `eye_refiner_val.pt` found that
`torch::pickle_load` fails inside its unpickler (`Expected GenericDict but got None`). The
native reader above bypasses that API and resolves reading the legacy checkpoint data.
Restoring those values into a native training backend, saving new checkpoints and proving
save/resume equivalence remain to be implemented.

ONNX Runtime's inference C API is suitable for the benchmark and export checks. It does not
by itself replace the existing PyTorch/MMDetection training recipe. Its training API would
need training graphs and an equivalent gradient/optimiser pipeline as well.

Implement and verify these contracts before removing the remaining Python:

1. **Native backend and legacy checkpoints.** Pin the native libraries and compiler. Prove CPU
   and CUDA forward/backward with a finite-difference gradient check and a parameter update.
   The existing `.pt`/`.pth` tensor dictionaries and provenance can now be loaded and verified
   without Python via `checkpoint.zig`; next, restore them into the native tensor backend.
   Save atomically and prove save/resume equivalence, including optimiser and schedule state.
   Do not silently accept only a new format and strand the recorded checkpoints.
2. **Images and refiner training.** Preserve the 112-pixel RGB normalisation, five pairs of
   Conv/BatchNorm/ReLU blocks, two linear layers, affine geometry, eye order on mirroring,
   rotation/brightness augmentation, Smooth L1 beta 0.02, cosine angle loss, AdamW and OneCycleLR.
   Reuse `util.zig`'s Python-compatible RNG where the existing image split depends on it.
   Split by image with seed 1234; choose checkpoints only on held-out training images; score
   the test split once, using the selected checkpoint. Compare crops, forward outputs, loss
   terms and gradients with saved reference inputs before running a real training smoke test.
3. **Native SCRFD training.** Reproduce MobileNetV1, PAFPN and the three SCRFD head branches,
   including training BatchNorm rather than training the already folded inference graph.
   Preserve two anchors per location at strides 8/16/32, Quality Focal Loss, DIoU box loss,
   Smooth L1 keypoint loss, ignored unlabelled landmarks, ATSS top-k 9 and ignore IoF 0.5 for
   the 80k recipe. Retain square cropping, photometric augmentation, all real face boxes and
   ignore regions. Preserve both schedules: the label-count recipe and the 80k run's
   100 epochs with LR drops at 69/85. Compare assignments, masks, losses and one optimiser
   step on fixed batches; then check loss decreases on actual data on the target GPU.
4. **Exports and evaluators.** Keep named weights while materialising aliases and folding
   BatchNorm; keep SCRFD's nine deployment outputs and the refiner's dynamic batch axis.
   Preserve both model licences and the checkpoint's selection provenance. Run independent
   ONNX inference checks at the existing tolerances (`1e-4` SCRFD, `1e-5` refiner, batch > 1).
   Match the old evaluation counts, IoU tie rules, inclusive-pixel NMS, frozen thresholds,
   FP budgets, eye-angle conventions and paired sample selection on fixed detection caches.
   Keep codec/orientation handling, fixture licence checks and the contact sheet's synthetic
   rotation-sign check. Update commands in `TRAINING_SETUP.md`, `run_curve.sh` and the result
   documents only when the replacement command actually works.

The migration is complete when every row has a working native command, all callers use it,
and a clean training/export/evaluation run needs no Python interpreter or Python packages.
The original model reports remain historical evidence; a new training run is not assumed to
reproduce their headline accuracy just because its architecture matches.
