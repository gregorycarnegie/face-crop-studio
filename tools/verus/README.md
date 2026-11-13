# Verus pilot: production output-row initialization

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

The `write_tiled_row` call-domain guarantee discharges half of the depthwise
kernel's `get_unchecked` argument; that `lo..hi` is the in-bounds interior is the
Lean `interior` proof. Tap arithmetic stays outside both proofs.

For any `T: Copy`, any input values, and an initially arbitrary output slice
whose length is exactly twice the input length, Verus proves:

- The loop terminates without indexing outside either slice or overflowing its counter.
- Every output slot is initialized, including when the output started uninitialized.
- Output slot `i` contains exactly input slot `i / 2`.
- The output length is preserved. Empty rows are supported.

The pinned `verus.exe` verifies the production source through a temporary Rust
module wrapper with `--no-cheating`. The check then applies four mutants to a
temporary copy (a skipped slot in each writer, and the tiled row calling its
interior tile past `hi`) and requires Verus to reject each one. The
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

Setup downloads official Verus `0.2026.11.02.8bf68ee`, checks its pinned SHA-256,
and installs its required Rust `1.99.1` in `cache/verus/rustup`. It restores the
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

Arena allocation/reuse, `Vec::from_raw_parts`, dense convolution's `OutPtr`, SIMD
layout and arithmetic, and the final conversion to an initialized vector are
unverified here. Dense convolution is the remaining `Blocked::written` caller
whose initialization is argued in a SAFETY comment rather than proved: it writes
through a raw `OutPtr`, which Verus would need permission tokens to check.

We trust Verus, its bundled standard-library specifications (including
`MaybeUninit::write`), Z3, and Rust/LLVM compilation. The contract is maintained
alongside the implementation; the script only points the verifier at the module
and constructs the negative-control copy.
The [official guide](https://verus-lang.github.io/verus/guide/) explains the
verification model; this pilot supplements the existing Lean, Kani, and Miri checks.
