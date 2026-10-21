# Experimental CUDA Rust backend

CUDA Rust is an explicit runtime opt-in for SCRFD detection, image enhancement and crop
masking. WESL/CPU remain the default on every platform. There is no promise that the
experimental CUDA implementation is faster; use the comparison below on your hardware.

## Enable

- GUI: Settings → GPU → CUDA Rust (experimental). The option stays disabled until a
  background probe finds a compatible NVIDIA GPU, loads the kernels, and resolves their
  entry points. Confirm the warning before the setting changes.
- CLI: `fcs-cli --input photo.jpg --experimental-cuda`. Activation prints a warning to
  stderr, including when enabled through settings JSON.
- JSON: `gpu.experimental_cuda` defaults to `false`. Existing settings files keep CUDA off.
  GPU acceleration must also be enabled. `--no-gpu` overrides a saved CUDA preference.

The warning says the backend may crash, run slower, or produce different
results. The status bar identifies CUDA Rust while it is actually running. Initialization
or inference failures are reported; an explicit CUDA request never silently switches detection
engines. Enhancement/masking failures log a warning and use the existing CPU fallback to
preserve the crop.
Uncheck the option to return to automatic WESL/CPU selection.

## Platforms and requirements

| Platform | Kernel compiler | Runtime requirements |
| --- | --- | --- |
| Windows | Rust's built-in NVPTX target, pinned nightly 2026-08-24 | NVIDIA compute capability 8.0+, CUDA driver API 12.0+ and successful PTX loading |
| Linux (x86_64 / aarch64) | NVIDIA CUDA Rust `cuda-oxide`, pinned revision in `tools/cuda/linux/Cargo.toml` | NVIDIA compute capability 8.0+, CUDA driver API 12.0+ and successful PTX loading |
| macOS and other platforms | Standard WESL/CPU engines | CUDA option unavailable |

Both compilers use the same Rust bodies for general/grouped convolution, dedicated
pointwise/depthwise convolution, fused bias/ReLU, add and nearest-neighbour upsampling.
Each level's class, box and keypoint heads run as one merged convolution, split over input
channels into partial sums that a second kernel adds in a fixed order (no atomics), so the
narrow heads fill the GPU and results stay reproducible.
The existing SCRFD topology, ONNX weight reader, input preprocessing, decoding and NMS are reused. Weights stay on the device;
concurrent inference uses separate streams and temporary buffers. The host uses cudarc's
driver wrapper with dynamic loading. End users need a compatible NVIDIA driver, without
installing a CUDA Toolkit or Rust toolchain.

## Enhancement and masking coverage

CUDA implements all production image WESL passes: exposure/brightness/contrast/saturation,
separable Gaussian blur, bilateral smoothing, background blur blending, per-channel
histogram equalization and polygon masks with antialiasing/vignettes. Histogram, LUT and
application are separate stream-ordered kernels; ordinary GPU stages keep pixels on device.
Both front-ends use the same runtime and NVIDIA context as detection.
CUDA and WESL use the same half-open edge rule with a coordinate-scaled floating-point
margin to stabilize supersample coverage on diagonal boundaries.

Red-eye statistics, sharpening recombination, eye refinement, geometric cropping, quality
scoring and transparent fill retain the existing CPU implementations. These are CPU work
in the WESL pipeline too. Rectangles and outlines above 512 points retain the CPU mask;
complete outlines are preserved. The files in `fcs-core/examples/shaders` are profiling
variants, not separate product features; they stay as benchmark/reference shaders.

## Build and validate

Normal builds use the checked-in PTX and stay on the workspace's stable Rust toolchain.
CI checks source fingerprints so a Rust kernel change cannot leave stale embedded PTX.

Windows kernel regeneration (Python and rustup required):

```powershell
rustup toolchain install nightly-2026-08-24 --profile minimal --component rust-src
python tools/cuda/build.py --platform windows
```

Linux regeneration uses NVIDIA's compiler and its pinned nightly:

```bash
rustup toolchain install nightly-2026-10-11 --profile minimal --component rust-src,rustc-dev,llvm-tools
cargo +nightly-2026-10-11 install --git https://github.com/NVIDIA/cuda-rust.git --rev e027ac324f663e8408c13c116d7dddd58d5e8daf --locked cargo-oxide
python3 tools/cuda/build.py --platform linux
```

The current kernels need no toolkit math libraries during compilation. See
[NVIDIA's compiler setup](https://nvidia.github.io/cuda-rust/cuda-oxide/getting-started/installation.html)
for the complete upstream prerequisites when adding such operations. Windows uses
[Rust's NVPTX target](https://doc.rust-lang.org/rustc/platform-support/nvptx64-nvidia-cuda.html).

Run the hardware test explicitly on a compatible NVIDIA machine:

```bash
cargo test -p fcs-core --lib cuda_matches_cpu -- --ignored --nocapture
cargo test -p fcs-core --test cuda_detection -- --ignored --nocapture
cargo test -p fcs-utils --lib cuda_image_passes -- --ignored --nocapture
cargo run --release -p fcs-core --example cuda_comparison -- photo.jpg
python tools/cuda/build.py --check
```

The hardware test compares all nine network output maps against the built-in CPU graph
and checks simultaneous inferences using both the Windows and NVIDIA Linux compiler outputs.
It fails on missing models/drivers/hardware. Ordinary CI deliberately ignores this hardware
test; it still tests legacy configuration, opt-in policy, CLI conflicts and GUI confirmation.
The second hardware test compares CUDA and CPU face counts, scores and boxes on all the
licensed image fixtures.

The enhancement hardware test compares each stage and the combined pipeline against WESL
on tiny, odd-sized and translucent images, checks shape fallback boundaries and simultaneous
exports, and loads both compiler outputs. Per-stage rounding tolerance is one 8-bit level;
the combined pipeline allows four. Tests deliberately fail when the required hardware is
absent. Timing comparisons include GPU readback and CPU processing rather than just dispatch.
