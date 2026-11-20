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


`codecs/` contains 17 tiny synthetic image fixtures, with no photographs or third-party data.
The 13x9 color patterns use RGB `((19x+7y)%256, (5x+29y)%256, (31x+11y)%256)`.
Variants cover RGB/RGBA, 16-bit samples, Adam7 passes, 1/2/4/8-bit grayscale, grayscale alpha,
a four-color palette with transparency, and bottom-up/top-down BMP rows. Grayscale JPEGs
are 8x8 constant blocks; the progressive variant includes DC/AC first and refinement scans.
The four-component JPEG covers Adobe CMYK conversion. All expected decoded pixels
were checked against the original stb C implementation before its removal. The fixtures use
the repository's code licence; the tests need only Zig.
