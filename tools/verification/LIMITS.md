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

Consequently, IEEE floating-point detection/color arithmetic, raw-pointer
ownership, GPU/CUDA execution, buffer drop tracking, atomic filesystem
replacement, and SQL/format-library enforcement remain **unproved**. Establishing
them needs explicit semantic models, external-library contracts, or additional
verification tools. Finite tests, fuzzing, Miri, and parity checks complement the
integer/byte proofs; they do not turn those unproved operations into theorems.

The buffer budget theorem covers concurrent use. The pool reserves and releases
budget with single atomic read-modify-write steps whose transitions are the
proved `reserve` and `release`. Updates to one atomic location are totally
ordered, so every thread interleaving is a sequence of those transitions, and
`counter_within_budget` holds for every such sequence. Two things are trusted
rather than proved: that `AtomicU64::try_update` applies its closure atomically
(a standard-library contract), and that no other code writes the counter (it is
private to the pool). Before this change the check and the add were separate
steps; 16 racing threads all succeeded against a 4-buffer budget, which the Rust
test `concurrent_acquires_stay_within_the_budget` now rejects.

The pool still returns raw `wgpu::Buffer` objects and cannot observe their
destruction outside `recycle`. Such a drop leaves its bytes counted, so the
budget errs toward refusing, never toward overshooting. A panic inside
`create_buffer` gives its reservation back through a drop guard. Idle bytes now live in
`IdleBuffers` beside the lists, under the same lock, and change only through its
methods. A property test drives random push/take/evict/clear sequences, without a
GPU, against a list model. It checks that the total always equals the bytes held,
that takes are best-fit, and that eviction frees smallest first. That is a tested
invariant, not a Lean proof: the lists are a `HashMap` of buffers, which Aeneas does
not translate. Scope parking relies on Rust tests. The best-fit theorem is compositional: the production selection
transition and a fold applying it are proved; usage filtering and Rust iterator
wiring also have Rust tests. Device allocation is outside these theorems.

NMS selection is proved over an opaque overlap oracle for the naive path, which
handles fewer than 200 detections. The grid path for larger sets keeps its own
loop; Rust tests check it against the naive one. The production closure
compares floating-point IoU against the threshold; Lean sees only its Boolean
answers, so the theorem holds whatever that arithmetic returns. The floating-point
side (IoU and the spatial grid's cell lookup) is checked bit-precisely by Kani in
`tools/kani`, not by Lean.

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
