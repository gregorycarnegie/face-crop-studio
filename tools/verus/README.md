# Verus pilot: production output-row initialization

This checks the real generic `write_upsampled_row` function in
[`fcs-core/src/cpu/output.rs`](../../fcs-core/src/cpu/output.rs), called by
the CPU `upsample2x` kernel. It handles the actual `MaybeUninit<T>` writes,
not a separately implemented integer or Boolean model.

For any `T: Copy`, any input values, and an initially arbitrary output slice
whose length is exactly twice the input length, Verus proves:

- The loop terminates without indexing outside either slice or overflowing its counter.
- Every output slot is initialized, including when the output started uninitialized.
- Output slot `i` contains exactly input slot `i / 2`.
- The output length is preserved. Empty rows are supported.

`check.ps1` reads the production file afresh, excludes its unit tests, changes
module visibility and doc comments for the standalone crate, and inserts the
contract and loop invariants. It preserves the executable body. Each insertion
anchor must occur exactly once; source changes that invalidate an anchor fail
the check. Generated files stay in the ignored `cache/verus` directory.

The check then changes `x += 1` to `x += 2` in a generated copy and requires a
failed invariant or postcondition. This negative control confirms that a writer
which misses output slots is rejected. Its diagnostics are in
`cache/verus/mutation.log`. Both runs use `--no-cheating`; there are no project
axioms, assumed postconditions, or external function bodies in this pilot.

## Run on Windows x64

Requires PowerShell, `rustup`, and network access for the first setup:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/verus/setup.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File tools/verus/check.ps1
cargo test --locked --manifest-path tools/safety/Cargo.toml --release miri_cpu_buffers
```

Setup downloads official Verus `0.2026.11.02.8bf68ee`, checks its pinned SHA-256,
and installs its required Rust `1.99.1` in `cache/verus/rustup`. It restores the
calling process's Rust environment and does not change the application's
toolchain or dependencies. CI runs these commands in
[`.github/workflows/verus.yml`](../../.github/workflows/verus.yml).
The existing Miri job also exercises the extracted writer with uninitialized
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
unverified here. In particular, verifying one row writer does not discharge the
initialization obligations of the other `Blocked::written` callers.

We trust Verus, its bundled standard-library specifications (including
`MaybeUninit::write`), Z3, and Rust/LLVM compilation. Proof annotations are added
by this script, so its source extraction is also part of the trust boundary.
The [official guide](https://verus-lang.github.io/verus/guide/) explains the
verification model; this pilot supplements the existing Lean, Kani, and Miri checks.
