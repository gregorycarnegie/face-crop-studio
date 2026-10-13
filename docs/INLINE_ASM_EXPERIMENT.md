# Inline assembly experiment — 12 October 2026

Handwritten AVX2 speeds up the extracted SCRFD RGB-to-CHW conversion versus the
original pixel loop, but a safe Rust row loop matches or slightly beats it. There
is no repeatable improvement in preprocessing plus the full CPU graph. Assembly
remains an isolated probe. After the initial experiment, the safe Rust row loop
was adopted in production at the user's request; no runtime dependencies were added.

The existing CPU convolution already uses `wide::f32x8` and fused multiply-adds.
The conversion in `fcs-core/src/scrfd.rs::preprocess` was a more useful first
candidate: it splits interleaved RGB bytes into three normalised float planes.

## Reproduce

From the workspace root, with the existing SCRFD model in `models/`:

```powershell
cargo build --release --offline -p fcs-core --example inline_asm_experiment
cargo clippy --release --offline -p fcs-core --example inline_asm_experiment -- -D warnings
& ./target/release/examples/inline_asm_experiment.exe (Get-ChildItem fixtures/oracle/images/*.jpg | ForEach-Object FullName)
```

Finish compilation and linting before timing. The executable always includes a
synthetic 640x640 image; additional image paths are positional arguments.

## Method

- AMD Ryzen 9 7950X, Windows x86_64, Rust 1.99.0 / LLVM 23.1.1; 32 Rayon threads.
- Workspace release settings: LTO, one codegen unit, `target-cpu=x86-64-v3`.
- Two runs of the same binary, each covering the eight licensed oracle JPEGs
  (portrait and landscape) plus the synthetic square. Image decoding is untimed.
- Four variants: original pixel loop, safe row slices, AVX2 intrinsics, and `asm!`.
  Both SIMD versions shuffle eight RGB pixels at a time and use the same exact
  subtract/multiply normalisation. Partial vectors use the Rust loop.
- Three untimed warmups, then 31 samples with rotating/reversing variant order.
  Eight calls/sample for conversion, three for preprocessing, one for the graph.
- Conversion includes output allocation and letterbox padding. Preprocessing
  includes the existing resize. Graph timing includes preprocessing and all CPU
  layers and heads, but excludes image decoding, detection decoding and export.
- The A/A conversion control uses the same compiled function as A/B. Its paired
  median ratios were 0.996–1.002 across the two runs. `normalise` and the probe's
  preprocessing wrapper are not inlined, preventing closure-specific baseline
  optimisation. In these initial runs, the full-preprocessing baseline called
  production `scrfd::preprocess`, which still used the pixel loop.
- Reported speedups are medians of paired `current / candidate` sample times,
  not ratios of separately computed time medians. An IQR is printed for each.

## Initial experiment results

The table gives the median of the eight per-photo speedups, separately for each
run. This is descriptive, not a measured folder-throughput result. Greater than
1.000 means faster; the synthetic image is excluded from these aggregates.

| Timed work | Safe Rust rows, runs 1 / 2 | Intrinsics, runs 1 / 2 | Assembly, runs 1 / 2 |
| --- | --- | --- | --- |
| Conversion, allocation and padding | 1.127 / 1.124 | 1.096 / 1.095 | 1.114 / 1.111 |
| Full preprocessing | 1.018 / 1.019 | 1.018 / 1.025 | 1.019 / 1.020 |
| Preprocessing + full CPU graph | 0.994 / 1.002 | 0.985 / 1.017 | 0.989 / 1.014 |

Across all 16 photo/run cases, assembly conversion speedups were 1.065–1.134
(about 6–12% less time); safe Rust rows were 1.059–1.142. Thus the gain does not
require assembly. Full preprocessing's assembly speedups were 1.009–1.035.
Full-graph ratios varied from 0.962–1.031, and every assembly graph IQR crossed
1.000. This probe cannot resolve a gain at that level.

For the synthetic square in run 2, conversion medians were 212.96 us current,
187.18 us safe Rust, 194.45 us intrinsics and 188.80 us assembly. Full preprocessing
was 274.67, 253.70, 260.77 and 258.03 us respectively. Graph timing was about 5.5 ms
for all four variants, so tens of microseconds saved in conversion have little
effect on total inference latency.

## Validation and limits

The example's assert-based checks passed for every byte value, zero-length and
short rows, all tails up to width 64, unaligned source/destination slices, output
sentinels, portrait/landscape padding and full preprocessing. On all eight photos,
all preprocessed floats and all nine CPU graph heads matched exactly.
The release Clippy check with `-D warnings` passed.

The assembly only runs after an AVX2 check, reads exactly 24 bytes per vector,
declares its register clobbers and memory writes, and leaves tails to safe Rust.
The experiment requires x86_64 AVX2; other architectures were not tested. No GPU,
folder throughput, GUI, decoding/export or full-workspace test run was measured.
These results do not establish a folder-throughput or GPU performance gain.

## Safe Rust adoption

`scrfd::preprocess` now splits the output into disjoint R/G/B slices and walks
each resized RGB row with safe iterators. Geometry, resize, normalisation,
padding and allocation are unchanged. No unsafe code or threading was added.

The probe retains the original pixel loop as `Legacy`. Its `Rows` mode calls
production `scrfd::preprocess` for full-preprocessing and graph comparisons;
conversion-only comparisons still use the extracted row loop. Thus future runs
compare the new production path with the historical implementation.

Validation: all 25 SCRFD unit tests passed, including seven named `rstest`
cases for exact pixel/plane
and padding check for short, odd, square, portrait and landscape images. Both
Python parity tests passed on all eight oracle photographs. Release Clippy
passed for the library and the experiment with `-D warnings`.

A follow-up release run compared the integrated production path with `Legacy`
on the same eight photos. The median per-photo preprocessing speedup was 1.026
(range 1.002–1.033). Every production graph IQR crossed 1.000, so the graph result
remains unresolved. Exact preprocessed tensors and all nine CPU heads matched
the historical implementation on every photo. Raw output is in
`target/row-slices-adoption.txt`.

Source: [isolated example](../fcs-core/examples/inline_asm_experiment.rs).
Local raw timing output: `target/inline-asm-run1.txt` and `target/inline-asm-run2.txt`
(ignored build artifacts).
Technique references: [Rust Reference](https://doc.rust-lang.org/reference/inline-assembly.html)
and [Rust By Example](https://doc.rust-lang.org/rust-by-example/unsafe/asm.html).
