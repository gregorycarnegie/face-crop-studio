# Lean crop geometry pilot

This verifies the production `CropRegion` implementation, rather than a separately
written Lean version. `lib.rs` imports `fcs-core/src/cropper/region.rs` directly;
the app re-exports the same type through its existing public paths.

Every check translates that Rust file through Charon and Aeneas, then checks
`Proofs.lean` against the freshly generated model. Generated files and toolchains
stay outside the tracked source tree.

The check also changes the width clamp from `min` to `max` in a disposable Rust
copy, translates that copy, and requires Lean to reject its proof. Production
sources are never changed by this negative control; its log is in
`~/.cache/fcs-lean-pilot/project/mutation/proof.log` (or the configured cache).

## Properties

- `in_bounds_rect_safe`: for **every** representable region, padding, and image
  size, the query succeeds. Any returned rectangle has positive dimensions and
  lies entirely inside the image. Padding need not be consistent with the region.
- `new_spec`: construction succeeds for all `i32` coordinates and `u32` dimensions,
  and its fields match the Rust arithmetic, including narrowing-cast wraparound.
- `new_intersection_correct`: construction followed by intersection with the
  **same image dimensions** returns exactly the geometric overlap. It returns
  `None` exactly when either overlap length is zero, including zero-sized images
  and crops completely outside the image.

The independent Rust boundary test checks 117,649 combinations of coordinates,
crop sizes, and image sizes against an `i64` geometric oracle. Unlike that finite
test, the Lean theorems quantify over every value of their input types.

## Run

The prebuilt toolchain setup supports Linux x86_64, including Ubuntu under WSL.
Prerequisites: `rustup`, `curl`, `git`, `tar`, and `zstd` on the Linux side.
On Ubuntu, install missing system tools with `sudo apt-get install curl git zstd`.
Run from the repository root:

```powershell
# One-time setup (downloads several GB; allow roughly 13 GB of disk space).
wsl -d Ubuntu -- bash tools/verification/setup.sh
# Regenerates the model, runs the Rust boundary test, and checks the Lean proofs.
wsl -d Ubuntu -- bash tools/verification/check.sh
```

On Linux, omit `wsl -d Ubuntu --`. Tools default to
`~/.cache/fcs-lean-pilot`; set `FCS_LEAN_CACHE` to choose another directory for both
scripts. Keep this cache on the Linux filesystem under WSL for faster access.
The generated model is `$FCS_LEAN_CACHE/project/CropRegion.lean` (using that
default if the variable is unset).

Versions are pinned to Aeneas `nightly-2026.11.01-ab34bd2` (including its matching
Charon), Lean `4.35.0`, and Rust `nightly-2026-09-17`. The verification Rust compiler
is isolated from the app's normal Rust toolchain. The Aeneas release supplies
precompiled Aeneas libraries; `lake update` fetches their pinned mathlib dependencies.

The Lean workflow checks changes to the geometry, verification files, and workflow.
Normal application builds do not require Lean or any new runtime dependencies.

## Scope and trust

The proofs cover these integer geometry operations. They do not cover floating
point crop placement, pixel copying, detection, GPU kernels, FFI, or the whole app.
They do not claim that padding magnitudes always equal unbounded mathematical
distances: existing narrowing casts can wrap for extreme off-image coordinates.
The intersection proofs account for that behavior.

Lean checks the translated model. We still trust Charon/Aeneas's translation and
Rust standard-library models, Lean's kernel, and the compiler/toolchain used to
run the app. Each main theorem prints its axiom dependencies; the checked proofs
use only `propext`, `Classical.choice`, and `Quot.sound`, with no `sorryAx` or added
axioms. Warnings, including admitted proofs, fail the check.

References: [Aeneas](https://github.com/AeneasVerif/aeneas),
[translation and proof workflow](https://github.com/AeneasVerif/aeneas/blob/main/documentation/aeneas-overview.md).
