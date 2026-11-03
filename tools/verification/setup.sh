#!/usr/bin/env bash
set -euo pipefail

# shortcut: official prebuilt Linux x86_64 tools only; add other hosts if the pilot expands.
test "$(uname -sm)" = "Linux x86_64"
cache="${FCS_LEAN_CACHE:-$HOME/.cache/fcs-lean-pilot}"
mkdir -p "$cache"
export PATH="$cache/lean-4.35.0-linux/bin:$HOME/.cargo/bin:$PATH"
export RUSTUP_HOME="$cache/rustup"

if [ ! -x "$cache/aeneas/aeneas" ]; then
    curl -fsSL --retry 3 -o "$cache/aeneas.tar.gz" \
        https://github.com/AeneasVerif/aeneas/releases/download/nightly-2026.11.01-ab34bd2/aeneas-linux-x86_64.tar.gz
    mkdir -p "$cache/aeneas"
    tar -xzf "$cache/aeneas.tar.gz" -C "$cache/aeneas"
fi
if [ ! -x "$cache/lean-4.35.0-linux/bin/lean" ]; then
    curl -fsSL --retry 3 -o "$cache/lean.tar.zst" \
        https://github.com/leanprover/lean4/releases/download/v4.35.0/lean-4.35.0-linux.tar.zst
    tar --zstd -xf "$cache/lean.tar.zst" -C "$cache"
fi
if ! rustup run nightly-2026-09-17 rustc --version >/dev/null 2>&1; then
    rustup toolchain install nightly-2026-09-17 --profile minimal --component rustc-dev,rust-src
fi

"$cache/aeneas/aeneas" -version
lean --version
if [ ! -f "$cache/.dependencies-ready" ]; then
    lake -d "$cache/aeneas/backends/lean" update
    touch "$cache/.dependencies-ready"
fi
