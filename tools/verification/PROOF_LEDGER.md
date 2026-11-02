# Remaining proof inventory

This inventory covers the application sources audited after the convolution-read
proofs. A completed entry means the stated property of the production helper is
checked by Lean. It does not imply verification of its external dependencies or
the whole application. Pending entries stay visible until proved or a concrete
limitation is recorded. Existing geometry, tensor, RGB-row, partition, and
convolution proofs are described in README.md.

| ID | Production area | Property to establish | Status |
| --- | --- | --- | --- |
| C1 | CPU channel extraction | Channel ranges reject overflow and out-of-range requests | Proved: TransformProofs |
| C2 | CPU blocked layout | Channel/block/lane decomposition is a bijection; extraction addresses are bounded | Proved: TransformProofs |
| C3 | CPU upsampling | Doubled dimensions do not wrap; each output maps to the correct in-bounds source pixel | Proved: TransformProofs |
| C4 | CPU weights/global average | Partial blocks include exactly the real channels; padded lanes cannot become real channels | Proved: TransformProofs |
| C5 | CPU allocation | Element-to-byte sizing respects Rust allocation/slice limits | Proved: TransformProofs |
| G1 | GPU host geometry | Convolution/pooling dimensions reject overflow and invalid geometry | Proved: GpuProofs |
| G2 | GPU dispatch | Ceiling division covers each logical item, with a bounded final partial workgroup | Proved: GpuProofs |
| G3 | GPU upsampling/grouping | Doubled dimensions and channel groups retain exact indices without narrowing wrap | Proved: GpuProofs and TransformProofs |
| M1 | ONNX and ORT shape boundaries | Signed dimensions are representable; shape products and raw float payload lengths agree | Proved: ShapeProofs and TensorProofs |
| P1 | PNG/JPEG metadata | Segment/chunk ranges stay in bounds, advance, and reject overflow/truncation | Proved: ByteProofs |
| P2 | Metadata injection | Insertion offsets and encoded length fields are representable | Proved: ByteProofs |
| L1 | SCRFD graph liveness | Slots are released only after their final read; inputs/outputs are retained | Proved: GraphProofs, for valid numeric read-index tables |
| L2 | Merged heads | Channel partitions cover the concatenated heads without overlap | Proved: HeadProofs |
| B1 | GPU buffer pool | Snapshot budget admission cannot wrap; the selection transition/fold yields a smallest eligible buffer | Proved: BufferProofs; see LIMITS.md for iterator/usage integration |
| B2 | GPU buffer pool lifecycle | Eviction/accounting and concurrent scope ownership remain consistent | Unproved: locking/device/ownership models are absent; see LIMITS.md |
| D1 | Crop pixel copying | Clipped rectangles, row addresses, and RGB byte preservation | Proved: CopyProofs, composed with the existing intersection bounds |
| D2 | Hex color decoding | Accepted byte pairs decode exactly and invalid bytes are rejected, preserving the existing leading-plus behavior | Proved: ByteProofs |
| X1 | NMS, IoU, detector decoding, colors/enhancement | Floating-point numerical and selection properties | Unproved: actual floating-point probe fails translation; see LIMITS.md |
| X2 | GPU shaders/CUDA/FFI | Device execution, pointer ownership, and external-library contracts | Unproved: raw-pointer probe fails; external calls become axioms; see LIMITS.md |
| X3 | Atomic export/mapping import | Filesystem replacement, SQL read-only policy, format decoder behavior | Unproved: file I/O and external engines lack semantic models; see LIMITS.md |

The completion pass also reviews callers, tests, workflow path filters, and this
ledger. No new runtime dependency is needed for integer/byte helper proofs.
