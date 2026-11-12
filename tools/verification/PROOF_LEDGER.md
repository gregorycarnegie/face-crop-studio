# Remaining proof inventory

This inventory covers the application sources audited after the convolution-read
proofs. A completed entry means the stated property of the production helper is
checked by the listed verifier. It does not imply verification of its external dependencies or
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
| C6 | CPU upsample output initialization | Every slot in a correctly sized output row is initialized with input `i / 2`; the writer terminates with bounded indices | Proved: Verus checks the production generic `MaybeUninit<T>` writer in `tools/verus`; Rayon, arena ownership, and final vector conversion remain outside the proof |
| G1 | GPU host geometry | Convolution/pooling dimensions reject overflow and invalid geometry | Proved: GpuProofs |
| G2 | GPU dispatch | Ceiling division covers each logical item, with a bounded final partial workgroup | Proved: GpuProofs |
| G3 | GPU upsampling/grouping | Doubled dimensions and channel groups retain exact indices without narrowing wrap | Proved: GpuProofs and TransformProofs |
| M1 | ONNX and ORT shape boundaries | Signed dimensions are representable; shape products and raw float payload lengths agree | Proved: ShapeProofs and TensorProofs |
| P1 | PNG/JPEG metadata | Segment/chunk ranges stay in bounds, advance, and reject overflow/truncation | Proved: ByteProofs |
| P2 | Metadata injection | Insertion offsets and encoded length fields are representable | Proved: ByteProofs |
| L1 | SCRFD graph liveness | Slots are released only after their final read; inputs/outputs are retained | Proved: GraphProofs, for valid numeric read-index tables |
| L2 | Merged heads | Channel partitions cover the concatenated heads without overlap | Proved: HeadProofs |
| B1 | GPU buffer pool | Snapshot budget admission cannot wrap; the selection transition/fold yields a smallest eligible buffer | Proved: BufferProofs; see LIMITS.md for iterator/usage integration |
| B2 | GPU buffer pool budget | Concurrent reservations and releases never take allocated bytes past the limit, under any interleaving | Proved: BufferProofs (`counter_within_budget`), over atomic `try_update` steps; see LIMITS.md |
| B3 | GPU buffer pool lifecycle | Idle bytes equal the buffers held; scope parking; buffers dropped without `recycle` | Idle accounting: by construction in `IdleBuffers`, property-tested over random operation sequences. Scope parking: Rust tests. Drops outside `recycle` overcount, which only refuses earlier; closing that needs a self-recycling buffer type (an API change). See LIMITS.md |
| D1 | Crop pixel copying | Clipped rectangles, row addresses, and RGB byte preservation | Proved: CopyProofs, composed with the existing intersection bounds |
| D2 | Hex color decoding | Accepted byte pairs decode exactly and invalid bytes are rejected, preserving the existing leading-plus behavior | Proved: ByteProofs |
| N1 | NMS selection | For any overlap oracle: no two kept detections overlap, and each suppressed one overlaps an earlier kept one | Proved: GreedyProofs for the naive path (under 200 detections), which calls the proved function. The grid path keeps its own loop, checked against the naive one by Rust tests |
| N2 | IoU and NMS spatial grid | IoU is NaN-free, non-negative and positive only on overlap; grid offsets and cell lookup are monotone and in range | Checked by Kani (tools/kani), bit-precise f32; IoU within ±1e18, division monotonicity trusted from IEEE 754 |
| N3 | SCRFD row decoding and score activation | Decoded boxes and eye landmarks are finite, boxes non-negative for non-negative distances, untrained landmarks absent; sigmoid is a probability, above 0.5 only for positive logits | Checked by Kani (tools/kani) at detector ranges; sigmoid against a stated `exp` contract, its overflow arm by unit test |
| K1 | CPU colour conversions and tone | HSV/HSL/CMYK round trips within one step; tone LUTs monotone; equalization monotone and spanning 0..255 | Round trips: exhaustive over all 2^24 colours. Tone: exhaustive over the 256 levels on a parameter grid. Equalization: property test |
| X1 | GPU enhancement passes | Floating-point behavior of the WGSL/CUDA enhancement kernels | Blocked: device code; see X2 |
| X2 | GPU shaders/CUDA/FFI | Device execution, pointer ownership, and external-library contracts | Unproved: raw-pointer probe fails; external calls become axioms; see LIMITS.md |
| X3 | Atomic export/mapping import | Filesystem replacement, SQL read-only policy, format decoder behavior | Out of scope: OS rename and SQLite/decoder behavior are library contracts; read-only SQL is enforced by a read-only open plus an authorizer, both tested; see LIMITS.md |

The completion pass also reviews callers, tests, workflow path filters, and this
ledger. No new runtime dependency is needed for integer/byte helper proofs.
