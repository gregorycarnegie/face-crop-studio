# Contributing to Face Crop Studio

Thanks for helping improve Face Crop Studio. This repository is a Rust workspace with five application crates and one development-only inference oracle:

- `fcs-core`: face detection, the built-in CPU and WGSL inference engines, crop math, and GPU pieces.
- `fcs-ort`: development-only ONNX Runtime bindings for parity tests and benchmarks.
- `fcs-utils`: shared configuration, image enhancement, export helpers, and webcam support.
- `fcs-mapping`: CSV/Excel/Parquet/SQLite ingestion for the batch mapping workflow.
- `fcs-cli`: command-line workflows for batch processing and automation.
- `fcs-gui`: the egui/eframe desktop application.

## Quick Start (clone → running GUI)

On a machine that already has the [build tools](#development-setup) below:

```powershell
# 1. Clone
git clone https://github.com/gregorycarnegie/face-crop-studio
cd face-crop-studio

# 2. Build and launch the desktop app (auto-detects GPU, falls back to CPU)
cargo run -p fcs-gui

# ...or the CLI
cargo run -p fcs-cli -- --help
```

The detector model is **not** in the repository: `models/scrfd80k_500m_640.onnx` is fetched
from a release asset (see `models/README.md` for the URL and digest), which CI does in its
`Prepare Models` job. Detection is disabled without it and the tests that need it skip, so grab
it before running them. The first build is slow (it compiles the full dependency graph including
wgpu); subsequent runs are incremental.

If `cargo run` fails to compile, you're almost certainly missing one of the
build tools below (most commonly NASM, or the `dav1d`/`pkg-config` setup for
AVIF support).

## Development Setup

Shaders are authored in WESL under `fcs-core/src/gpu`, `fcs-utils/src/gpu`, and
`fcs-core/examples/shaders`. Each crate's `build.rs` compiles its embedded shaders
to WGSL and tracks imported files for rebuilds. Edit the WESL sources; generated
WGSL stays in Cargo's output directory. The profiling examples accept WESL paths
and link them before timing GPU compilation or execution. Run
`cargo test -p fcs-utils --lib wesl_validation_tests` to compile and validate every
WESL file, including SPIR-V and Metal translation of the resulting WGSL.
The convolution experiments share uniforms and output/activation code in
`examples/shaders/common.wesl`; `bindings_f32.wesl`, `bindings_f16.wesl`, and
`bindings_packed.wesl` define their input buffer layouts. Kernel experiments keep
their own arithmetic and sampling helpers.

Install the usual Rust and Windows build tools first:

- Rust `1.99.0`, matching the workspace minimum and CI/release toolchain.
- Visual Studio Build Tools with the C++ desktop workload and Windows SDK.
- Git.
- NASM on `PATH`; for example `C:\Program Files\NASM`. Installing it is not
  enough: `mozjpeg-sys` only *warns* when NASM is missing and then compiles
  libjpeg-turbo's scalar fallback, so the build succeeds and JPEG work is about
  2.2x slower. Check with `nasm -v`, and if a build has already cached the
  fallback, delete `target/*/build/mozjpeg-sys-*` and
  `target/*/.fingerprint/mozjpeg-sys-*` -- `cargo clean -p mozjpeg-sys` does not
  re-run the build script.
- `pkg-config` and `dav1d` for AVIF support.
- `libheif` (>= 1.17) for HEIC/HEIF support, located via `pkg-config`.

On Windows, the project CI uses `pkgconfiglite` and vcpkg:

```powershell
choco install pkgconfiglite --no-progress -y

if (-not (Test-Path "C:\vcpkg\vcpkg.exe")) {
  git clone https://github.com/microsoft/vcpkg C:\vcpkg
  & C:\vcpkg\bootstrap-vcpkg.bat
}

& C:\vcpkg\vcpkg.exe install dav1d:x64-windows-static "libheif[core]:x64-windows-static"
```

Set the environment variables for your current terminal:

```powershell
$env:PKG_CONFIG="C:\ProgramData\chocolatey\bin\pkg-config.exe"
$env:PKG_CONFIG_PATH="C:\vcpkg\installed\x64-windows-static\lib\pkgconfig"
$env:PKG_CONFIG_ALL_STATIC="1"
```

To make the vcpkg settings permanent for new terminals:

```powershell
[Environment]::SetEnvironmentVariable("PKG_CONFIG_PATH", "C:\vcpkg\installed\x64-windows-static\lib\pkgconfig", "User")
[Environment]::SetEnvironmentVariable("PKG_CONFIG_ALL_STATIC", "1", "User")
```

The full test suite and both front-ends use the model at:

```text
models/scrfd80k_500m_640.onnx
```

See `models/README.md` for the download URL and checksum. Tests that need it skip when it is
absent; set `FCS_STRICT_TESTS=1` to turn those skips into failures, which is how CI runs so a leg
cannot pass having checked nothing.

## Common Commands

Run a fast workspace check:

```powershell
cargo check --workspace
```

Run the full test suite:

```powershell
cargo test --workspace --all-features
```

`FCS_STRICT_TESTS=1` turns "prerequisite missing, skipping" into a failure, which
is what CI runs. The ONNX Runtime comparison in `scrfd_parity` needs the
library on hand for that, so point `ORT_DYLIB_PATH` at it:

```powershell
$env:ORT_DYLIB_PATH = "$HOME/.onnxruntime/onnxruntime-win-x64-1.24.4/lib/onnxruntime.dll"
$env:FCS_STRICT_TESTS = "1"
cargo test --workspace --release --no-fail-fast
```

`--no-fail-fast` is worth the habit: `cargo test` stops at the first test binary
that fails, so a failure in one crate hides every target after it.

Run formatting and lint checks:

```powershell
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
```

`--all-targets` is not optional. Without it clippy skips test code, and so does
`cargo build`: a change that breaks only a test helper passes both.

Launch the CLI and GUI:

```powershell
cargo run -p fcs-cli -- --help
cargo run -p fcs-gui
```

## Contribution Workflow

1. Create a branch for your change.
2. Keep changes focused and avoid unrelated formatting churn.
3. Add or update tests when changing behavior.
4. Update docs when changing user-facing workflows, setup, configuration, or CLI flags.
5. Run the relevant checks before opening a pull request.

For high-risk changes, prefer smaller pull requests that isolate behavior changes from refactors.

### Choosing a test style

- **A table of known inputs and outputs** → `rstest` with `#[case]` (or `#[values]` for a
  matrix). Each row becomes its own named test, so a failing row is reported by name and the
  rows after it still run. A hand-written `for case in [...]` loop stops at the first failure.
- **A rule that should hold for every input** (round trips, "stays in range", "never
  panics", "matches the naive reference") → `proptest`. Parsers of user input such as CLI
  flags and settings files should have a never-panics property over `any::<String>()`.
  Run with `PROPTEST_CASES=100000` for a deeper search before you change one of them.
- **A specific golden value** → a plain `#[test]`. Property tests check rules, so they do
  not replace asserting the actual number for a chosen input.

## Safety, fuzzing, and JSON snapshots

`CPU Safety and Parser Fuzzing` runs focused Miri scenarios and four bounded
AddressSanitizer fuzz targets on pull requests. These isolated Cargo workspaces
compile the production sources directly, without models, GPU setup, or native
image codecs. Their lockfiles keep the harness dependencies reproducible.

Run the CPU buffer checks (PowerShell):

```powershell
rustup toolchain install nightly-2026-08-24 --component miri
$env:MIRIFLAGS = "-Zmiri-tree-borrows -Zmiri-ignore-leaks"
cargo +nightly-2026-08-24 miri test --locked --manifest-path tools/safety/Cargo.toml --release miri_cpu_buffers -- --test-threads=1
```

Use `--release`: debug builds initialize output buffers with NaN, which would hide
missing writes from Miri. Tree Borrows handles Crossbeam's aliasing pattern;
leak reporting is disabled for its deferred global cleanup. Memory validity and
race checks stay enabled. Crossbeam uses exposed pointer provenance, so this
check does not establish strict-provenance correctness.

Run a bounded fuzz target on Linux (CI uses the same commands):

```bash
rustup toolchain install nightly-2026-08-24
cargo install cargo-fuzz --version 0.13.2 --locked
mkdir -p fuzz/corpus/metadata
cargo +nightly-2026-08-24 fuzz run metadata fuzz/corpus/metadata fuzz/seeds/metadata -- -max_total_time=30 -max_len=16384 -timeout=5 -rss_limit_mb=2048 -malloc_limit_mb=512
```

The 2 GiB process limit includes sanitizer and corpus bookkeeping; any individual
allocation is still capped at 512 MiB.

Replace `metadata` with `metadata_roundtrip`, `onnx_initializers` or `color` to
exercise the other parsers. `metadata` only checks the EXIF walkers never panic;
`metadata_roundtrip` builds valid JPEG and PNG files around a fuzzed EXIF payload
and checks the walkers return it exactly. Synthetic seeds cover float payloads,
malformed shapes, EXIF orientation, and color syntax. CI restores each target's
corpus from earlier passing runs. New inputs go into the ignored first corpus directory; tracked
seeds stay unchanged. Crashes go into `fuzz/artifacts/` and CI uploads them on
failure. Turn reproducible crashes into ordinary regression tests before fixing
the parser. Windows additionally needs a compatible AddressSanitizer runtime.

Insta snapshots pin default settings, custom export metadata, and CLI detection
JSON (including omitted optional fields and null landmarks). They run with the
normal test suite. Review intentional changes with targeted updates:

```powershell
$env:INSTA_UPDATE = "always"
cargo test -p fcs-utils --lib default_settings_json_matches_contract
cargo test -p fcs-utils --lib build_custom_metadata_payload_includes_crop_quality_and_custom_tags
cargo test -p fcs-cli --bin fcs-cli detection_json_matches_contract
Remove-Item Env:INSTA_UPDATE
git diff -- '*.snap'
```

Review the snapshot diff before committing it. Export metadata normalizes only
the generator version, which has a separate exact assertion. Image/crop goldens
and live inference results retain their existing tolerance-based comparisons.

## Dependency Guidance

- Prefer workspace dependencies in the root `Cargo.toml`.
- Update `Cargo.lock` with the manifest change.
- Avoid local `[patch.crates-io]` overrides unless there is no practical alternative.
- If an upstream compatible release is broken, prefer an exact version pin with a short explanation in the pull request.
- Keep dependency upgrades separate from feature work when practical.

## Code Style

- Follow the existing crate boundaries.
- Keep application-level error context helpful.
- Keep shared helpers small and testable.
- Use `cargo fmt` for formatting.
- Do not commit build outputs from `target/`.

## Reporting Issues

When reporting a bug, include:

- Operating system and GPU model if relevant.
- The command or GUI workflow used.
- The input format involved, such as image type, CSV, Excel, Parquet, or SQLite.
- The full error output or logs.
- Whether GPU acceleration was enabled or disabled.
