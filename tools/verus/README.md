# Verus: production buffer writes

This checks the real generic row writers in
[`fcs-core/src/cpu/output.rs`](../../fcs-core/src/cpu/output.rs). Their contracts
and loop invariants live directly in the production Rust source inside Verus's
`verus!` macro. They handle the actual `MaybeUninit<T>` writes, not a separately
implemented integer or Boolean model.

| Writer | Caller | Proved |
| --- | --- | --- |
| `write_upsampled_row` | `upsample2x` | Every slot initialized with input `i / 2` (details below) |
| `write_each` | `add` | Every slot `i` initialized with the closure's result for `i`, for any input lengths |
| `write_tiled_row` | depthwise convolution | Every slot initialized by the border/interior/tail schedule; the unchecked 4-pixel tile is only called with its whole tile inside `lo..hi` |
| `write_tiled_rows` | dense convolution (`Row::run`) | Every slot of all `F` row segments initialized; the unchecked `P`-pixel tiles and single pixels are only called inside `lo..hi` |

The `write_tiled_row` and `write_tiled_rows` call-domain guarantees discharge half
of the depthwise and dense kernels' `get_unchecked` arguments; that `lo..hi` is the
in-bounds interior is the Lean `interior` proof. Tap arithmetic stays outside both
proofs.

The dense kernel's jobs each write `F` output blocks' row segments, which lie in
different planes, so they are cut from one raw pointer with
`slice::from_raw_parts_mut` rather than handed out by Rayon. That they are inside
the allocation and disjoint between jobs is Lean's (`decode_job_spec`,
`chunks_disjoint`, `output_indices_unique`); that the job writes all of them is
Verus's. Miri runs the kernel over those segments under tree borrows.

[`fcs-utils/src/image_utils/planes.rs`](../../fcs-utils/src/image_utils/planes.rs)
holds `write_planes`, which every row of the CPU preprocessors (`rgb_to_bgr_chw` and
its letterboxed form) goes through: Verus proves it initializes every slot of the
three plane rows with the closure's value for that pixel, which is what their
uninitialized allocation's `set_len` relies on.

[`fcs-utils/src/image_utils/unpack.rs`](../../fcs-utils/src/image_utils/unpack.rs)
holds `unpack_rows`, which copies tightly packed pixel rows into a buffer with
padded rows: fcs-qt's image provider uses it to fill each `QImage`. Verus proves
it copies exactly `height` rows, row `y` to offset `y * stride`, leaves the
padding between rows alone, and returns `false` without writing anything unless
the stride holds a row and both buffers are long enough. Two defects in the copy
it replaced are therefore excluded: an `image` buffer longer than its pixels no
longer writes extra rows past the `QImage`, and a zero-width image no longer
panics (and aborts) across the C boundary. The `QImage` bytes reach it through
`slice::from_raw_parts_mut`, argued in its SAFETY comment from the allocator's
contract; the C++ side zero-fills the image first, so the slice covers
initialized bytes.

For any `T: Copy`, any input values, and an initially arbitrary output slice
whose length is exactly twice the input length, Verus proves:

- The loop terminates without indexing outside either slice or overflowing its counter.
- Every output slot is initialized, including when the output started uninitialized.
- Output slot `i` contains exactly input slot `i / 2`.
- The output length is preserved. Empty rows are supported.

The pinned `verus.exe` verifies the production source through a temporary Rust
module wrapper per file with `--no-cheating`. The check then applies fifteen
mutants to temporary copies (skipped slots, columns, rows or planes in each
writer, swapped channels, the tiled rows running unchecked tiles or pixels outside
`lo..hi`, and the row copy dropping its stride check or ignoring the stride) and
requires Verus to reject each one. The
wrappers and mutation diagnostics stay in the ignored `cache/verus` directory.
Normal Cargo builds use the pinned `vstd` crate dependency to expand `verus!`;
the verifier uses the standard-library specifications bundled with its pinned
Verus release.

These negative controls confirm that writers which miss output slots, or
leave the interior, are rejected. Both runs use `--no-cheating`; there are no project axioms, assumed
postconditions, or external function bodies in this pilot.

## Run on Windows x64

Requires PowerShell, `rustup`, and network access for the first setup:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/verus/setup.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File tools/verus/check.ps1
cargo test --locked --manifest-path fcs-core/Cargo.toml
cargo test --locked --manifest-path tools/safety/Cargo.toml --release miri_cpu_buffers
```

Setup downloads official Verus `0.2026.11.22.1ce6a59`, checks its pinned SHA-256,
and installs its required Rust `1.100.0` in `cache/verus/rustup`. It restores the
calling process's Rust environment and does not change the application's
toolchain or dependencies. CI runs these commands in
[`.github/workflows/verus.yml`](../../.github/workflows/verus.yml).
The existing Miri job also exercises the production writer with uninitialized
storage and the full production upsample kernel.

## Proof boundary

This is a component proof, not verification of the whole unsafe output path.
The caller must supply a correctly sized row. Existing Lean proofs cover doubled
dimensions and source-row indices; Rust tests and Miri exercise their integration.
Distinct mutable row slices are supplied by Rayon, whose implementation and
parallel completion are trusted library contracts. The proof relies on Rust's
exclusive mutable-borrow rules, rather than modeling the Rayon scheduler.

Arena allocation/reuse, `Vec::from_raw_parts`, SIMD layout and arithmetic, and
the final conversion to an initialized vector are unverified here. Dense
convolution's segments are cut from a raw `OutPtr` (see above): Verus checks what
is written through them, Lean the addresses, and the `from_raw_parts_mut` call that
joins the two is argued in its SAFETY comment. Verus permission tokens could check
that call too, but splitting them across Rayon tasks would need an external,
trusted function, which `--no-cheating` here rules out.

We trust Verus, its bundled standard-library specifications (including
`MaybeUninit::write`), Z3, and Rust/LLVM compilation. The contract is maintained
alongside the implementation; the script only points the verifier at the module
and constructs the negative-control copy.
The [official guide](https://verus-lang.github.io/verus/guide/) explains the
verification model; this pilot supplements the existing Lean, Kani, and Miri checks.
