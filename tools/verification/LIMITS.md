# Checked limits of the current proof pipeline

These are observed limitations of the pinned Charon/Aeneas toolchain, not claims
that Lean cannot express the corresponding mathematics. They prevent us from
claiming whole-program verification with the current source translation and
standard-library models.

Disposable dependency-free probes were extracted with Charon's `aeneas` preset,
then translated with `-backend lean -abort-on-error -warnings-as-errors` using the
versions pinned in setup.sh. Their results were:

| Minimal Rust operation | Observed result |
| --- | --- |
| `pub fn divide(a: f32, b: f32) -> f32 { a / b }` | Translation failed: `Invalid inputs for binop` |
| `pub unsafe fn read(p: *const u8) -> u8 { unsafe { *p } }` | Translation failed: `Aeneas does not yet support dereferencing raw pointers.` |
| `std::fs::read(path)` | Generated an unmodeled `axiom std.fs.read` and opaque I/O/path types |
| `*value.lock().unwrap() += 1` on a `Mutex<usize>` | Generated unmodeled axioms for `Mutex.lock` and `MutexGuard.deref_mut` |
| Calling an `unsafe extern "C"` function | Generated an unmodeled axiom for the external function |

No production proof imports these probes or assumes their external axioms.

Consequently, IEEE floating-point NMS/IoU/detection/color arithmetic, raw-pointer
ownership, GPU/CUDA execution, concurrent pool ownership, atomic filesystem
replacement, and SQL/format-library enforcement remain **unproved**. Establishing
them needs explicit semantic models, external-library contracts, or additional
verification tools. Finite tests, fuzzing, Miri, and parity checks complement the
integer/byte proofs; they do not turn those unproved operations into theorems.

The buffer budget theorem is a predicate over one accounting snapshot. It does
not establish that concurrent allocations reserve the budget atomically, or that
dropping a raw GPU buffer updates the pool's accounting. The pool returns raw
`wgpu::Buffer` objects and cannot observe their destruction outside `recycle`.
The best-fit theorem is compositional: the production selection transition and a
fold applying it are proved; usage filtering and Rust iterator wiring also have
Rust tests. Locking and device allocation are outside that theorem.

The graph proof covers the numeric read-index table used by the production
schedule builder. The small projection from generated `Step` values into that
table and the interpreter's tensor/arena operations also have Rust tests. Model
names and tensor storage are excluded from the graph proof's representation.

Metadata proofs cover byte ranges, progress, and encoded length fields. EXIF tag
editing, CRC computation, JSON/Base64 encoding, and third-party format decoders
remain library contracts. Crop-copy proofs assume valid packed image layouts and
cover clipping, addresses, and RGB byte preservation; resizing/rotation and image
library allocation remain outside them. Hex proofs cover the byte decoder; string
trimming, prefix removal, and underscore normalization remain library operations.
