# Synthetic checkpoint

`checkpoint.pt` contains six numbers (1 through 6), tensor views sharing their storage, an
empty tensor, an int64 scalar, and invented training metadata and optimizer state. It has no
trained weights, photographs or third-party data and uses the repository's code licence.

Generated with PyTorch's `torch.save(..., pickle_protocol=2)` to test the independent Zig
reader against a real PyTorch archive. The native tests check shared storage, non-contiguous
strides and offsets, zero-sized/scalar tensors, signed integers, tuple/list distinctions,
integer dictionary keys, Unicode, model attributes and CRC failures. CI does not need Python.
Three hundred distinct metadata strings also exercise pickle's long memo IDs and references.

`refiner.pt` is a complete synthetic width-32 refiner checkpoint. Zero-stride tensor views
share scalar zero/one storages so the file stays small. Every output is the final layer's
bias `[0.25, 0.5, -0.75, 1.25]`, independently checked in PyTorch. It includes invented
validation/test statistics and an epoch, and contains no trained weights. Native export CI
uses it to exercise checkpoint import, BatchNorm folding, dynamic batches, the final layer's
negative output, model provenance and the actual ONNX Runtime C API on every native platform.

`scrfd.pt` is a synthetic SCRFD-500M checkpoint with the real model's 285 parameter names and
shapes and no trained weights. Every float tensor is an overlapping-stride view (all strides 1)
into one of two shared storages of 1,024 values -- signed values at 0.05 scale, so activations
stay in a realistic range, and positive values in [0.5, 1) for running variances and BatchNorm
scales -- which keeps the file at 39 KB while values still vary along every axis. CI exports it
through ONNX Runtime and regenerates `fcs-core/src/scrfd/topology.rs` from the export.
