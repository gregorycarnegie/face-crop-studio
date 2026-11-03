#!/usr/bin/env bash
# Run every Kani harness over the production NMS, IoU and SCRFD decode sources. Needs
# `cargo install --locked kani-verifier && cargo kani setup` on Linux (WSL works); set
# FCS_KANI_TARGET to move the build.
set -euo pipefail

cd "$(dirname "$0")"
cargo kani -Z stubbing --target-dir "${FCS_KANI_TARGET:-$HOME/.cache/fcs-kani-target}"
