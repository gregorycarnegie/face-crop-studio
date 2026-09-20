# Models

This directory contains the ONNX models used by the application. Both are this project's own
exports, so neither can be regenerated from a public URL: CI and releases fetch them as release
assets and verify the digests below.

YuNet used to be here too, and was the default detector until 1.8.0. Its weights come from
WIDER FACE, released for "non-commercial academic research only", which is the licence question
this project set out to remove; it left once SCRFD ran on all three engines and no longer needed
a fallback. The history, including the WGSL and CPU implementations of YuNet's topology and the
experiments that measured them, is in the `face-crop-studio-yunet-archive` fork.

## Eye refiner

- **File**: `eye_refiner.onnx`
  - **SHA256**: `b01dd218b46a7466eb13125b011e81b57e2e2e252794c403029db1d7c7628e76`
  - **Input**: `crop`, `[batch, 3, 112, 112]`, RGB, `(pixel - 127.5) / 128`. **Output**: `eyes`, `[batch, 4]` — two eye points as fractions of the crop side, in the detector's landmark order (index 0 on the viewer's left).
  - **Notes**: Not a detector. It reads a face box somebody else found and replaces landmarks 0 and 1, which is all `fcs-core::face_cropper` uses for eye-line alignment. Trained from scratch on 1,988 hand-clicked eye pairs from the licence-clean Open Images corpus; 1.5 M parameters, opset 12, `Conv`/`Relu`/`GlobalAveragePool`/`Flatten`/`Gemm` only. Against 1,382 held-out test faces it gives **1.18 deg** median eye-line error where YuNet gives 4.05, and 93.3% of faces within 5 degrees against 57.6%. See `tools/dataset/CURVE_RESULTS.md` for the measurement, the generalisation checks, and what is still unproven.
  - **Runtime**: ONNX Runtime only. The WGSL kernels and the built-in CPU graph cover what the detector needs -- convolution, add, nearest upsample -- and not `GlobalAveragePool` or `Gemm`, so `EyeRefiner::load` returns `None` without a runtime and callers keep the detector's own landmarks. Releases bundle ONNX Runtime on all three platforms; this fallback is for source builds.
  - **Provenance**: produced by `tools/dataset/train_eye_refiner.py` and exported with a dynamic batch dimension.

## SCRFD-80k, the detector

- **File**: `scrfd80k_500m_640.onnx`
  - **SHA256**: `f31f1f01ca33d24184059bba0627a54b8351e6f2b167474a4d71f2361ea53c39`
  - **Input**: `input.1`, `[1, 3, 640, 640]`, RGB, `(pixel - 127.5) / 128`, the source letterboxed into the **top-left** (not centred, unlike YuNet's preprocessing). **Output**: nine tensors, three per stride 8/16/32 — class scores already sigmoided `[1, N, 1]`, box distances `[1, N, 4]` and keypoint distances `[1, N, 10]`, the last two in units of the stride. No NMS in the graph.
  - **Why it exists**: it is licence-clean, and better. It finds **86.0%** of the faces in the Open Images test split at 0.11 false positives per image, where YuNet finds 71.6% at 0.14, and on a corpus neither model has seen it finds 135 faces YuNet misses while missing 13 that YuNet finds. 2.5 MB, 6.4 ms/image against YuNet's 5.8 on ONNX Runtime CPU. See `tools/dataset/SCRFD_80K.md`.
  - **Runtime**: runs on all three engines — ONNX Runtime, the WGSL kernels, or the built-in CPU graph — which agree to about 1e-05 and produce identical detections. The weights are read by name, which is why the export folds BatchNorm itself rather than letting torch's constant folding rename them. Run at score 0.5.
  - **Provenance**: SCRFD-500M trained by this project on 80,000 CC BY 2.0 Open Images photographs (244,683 boxed faces), exported by `tools/dataset/export_scrfd.py`. The attribution manifest for the training images is `train_manifest.csv` beside the corpus.
