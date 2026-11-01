#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
cache="${FCS_LEAN_CACHE:-$HOME/.cache/fcs-lean-pilot}"
backend="$cache/aeneas/backends/lean"
build="$cache/project"
export PATH="$cache/lean-4.35.0-linux/bin:$HOME/.cargo/bin:$PATH"
export RUSTUP_HOME="$cache/rustup"
export RUSTUP_TOOLCHAIN=nightly-2026-09-17
export RUST_LOG=warn CHARON_LOG=warn
export LEAN_PATH="$build"
mkdir -p "$build"

cd "$root/tools/verification"
cargo test --locked --target-dir "$cache/rust-target"
"$cache/aeneas/charon" cargo --preset=aeneas --sysroot default \
    --dest-file "$build/crop_region.llbc" -- --lib --target-dir "$cache/rust-target"
"$cache/aeneas/aeneas" -backend lean -dest "$build" \
    -abort-on-error -warnings-as-errors -no-progress-bar "$build/crop_region.llbc"
lake -d "$backend" env lean -DwarningAsError=true \
    --root="$build" -o "$build/CropRegion.olean" "$build/CropRegion.lean"
lake -d "$backend" env lean -DwarningAsError=true "$root/tools/verification/Proofs.lean"
lake -d "$backend" env lean -DwarningAsError=true "$root/tools/verification/TensorProofs.lean"

# Prove the checker rejects a real Rust regression without touching production sources.
mutant="$build/mutation"
mkdir -p "$mutant"
cp Cargo.toml Cargo.lock "$mutant/"
printf 'pub mod region;\n' > "$mutant/lib.rs"
sed 's/self.in_bounds_width().min(max_w)/self.in_bounds_width().max(max_w)/' \
    "$root/fcs-core/src/cropper/region.rs" > "$mutant/region.rs"
if cmp -s "$root/fcs-core/src/cropper/region.rs" "$mutant/region.rs"; then
    echo 'Mutation did not change the source' >&2
    exit 1
fi
cd "$mutant"
"$cache/aeneas/charon" cargo --preset=aeneas --sysroot default \
    --dest-file "$mutant/crop_region.llbc" -- --lib --target-dir "$cache/mutation-target"
"$cache/aeneas/aeneas" -backend lean -dest "$mutant" \
    -abort-on-error -warnings-as-errors -no-progress-bar "$mutant/crop_region.llbc"
export LEAN_PATH="$mutant"
lake -d "$backend" env lean -DwarningAsError=true \
    --root="$mutant" -o "$mutant/CropRegion.olean" "$mutant/CropRegion.lean"
if lake -d "$backend" env lean -DwarningAsError=true \
    --root="$root/tools/verification" "$root/tools/verification/Proofs.lean" > "$mutant/proof.log" 2>&1; then
    echo 'Lean accepted the deliberately broken intersection' >&2
    exit 1
fi
if ! grep -Fq 'scalar_tac failed to prove' "$mutant/proof.log"; then
    cat "$mutant/proof.log" >&2
    exit 1
fi
echo 'Lean rejected the deliberately broken intersection'
