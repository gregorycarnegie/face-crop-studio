# ONNX Runtime Options

> **Historical (pre-1.9).** Everything below measures or designs around **YuNet**, which was the
> detector until 1.8.0 and no longer ships: its weights come from WIDER FACE, licensed for
> non-commercial academic research only. The engines it describes still exist and still run
> SCRFD; the numbers, node names and topology do not describe anything current, and the examples
> named here were deleted with it. The full YuNet implementation and these experiments are in the
> `face-crop-studio-yunet-archive` fork. Kept as a design and measurement record.

> **Phase 12 outcome:** We shipped Phase 12 Option C — a custom WGPU/WGSL YuNet inference graph
> that keeps tensors resident on GPU without an external runtime dependency. The `ort` path below
> remains the best route to DirectML/CoreML acceleration if we ever want full ONNX op coverage or
> hardware-vendor-optimised kernels without maintaining our own shaders.

> **Scope:** this note is only about replacing the *runtime*. INT8 quantisation is a change to the
> *model file* and needs none of what follows — `tract` parses quantised graphs already. See the
> "INT8 quantisation is not the `ort` question" section of PERFORMANCE.md.

Phase 12 explored accelerators beyond the current `tract-onnx` CPU backend. This note captures the practical options we can integrate from Rust today, their maturity, and what it would take to adopt them inside Face Crop Studio.

## Comparison Summary

| Runtime                                          | GPU Support                                       | Build Story                                                      | Pros                                                                 | Cons                                                                           | Recommended Use                                         |
|--------------------------------------------------|---------------------------------------------------|------------------------------------------------------------------|----------------------------------------------------------------------|--------------------------------------------------------------------------------|---------------------------------------------------------|
| `tract-onnx` (current)                           | CPU (SIMD), experimental CLA (no DirectML/Metal)  | Pure Rust, no external binaries                                  | Fast cold starts, easy cross‑platform builds, good tracing hooks     | CPU‑only, incomplete ONNX operator coverage (fails on malformed shapes)        | Default pipeline, low-friction builds                   |
| `ort` crate (ONNX Runtime bindings)              | DirectML (Windows), CUDA, CoreML, NNAPI, TensorRT | Requires shipping ONNX Runtime shared libs per platform          | Full ONNX op coverage, mature GPU story, dynamic execution providers | Large binary/runtime dependency, extra installation steps, unsafe FFI boundary | Windows/macOS builds where GPU acceleration is required |
| `onnxruntime` crate (official, similar to `ort`) | Same as ONNX Runtime                              | Same as above (vend prebuilt binaries or download at build time) | Official Microsoft bindings, example-rich                            | Less ergonomic than `ort` (manual memory mgmt)                                 | Alternative if `ort` API changes                        |
| `candle` / `burn` backends                       | WGPU/CUDA                                         | Depends on `metal`/`cuda` toolchains                             | Unified model runtime with WGPU story                                | Requires model conversion or reimplementation; no YuNet importer yet           | Future research if we rewrite inference in Rust shaders |
| `ncnn`/`MNN` via FFI                             | Vulkan/OpenCL                                     | Need to package prebuilt native libs                             | Proven mobile perf                                                   | C++ interop, custom model format                                               | Only if we abandon ONNX                                 |

## `ort` crate integration considerations

1. **Dependencies**: Add `ort = { version = "2", features = ["load-dynamic"] }`. To enable DirectML on Windows, add `features = ["load-dynamic", "directml"]` and ship the `onnxruntime_directml.dll` next to the binary (or instruct the user to install the Microsoft-provided package).
2. **Model loading**: Replace the `tract` loader with `ort::Environment` + `SessionBuilder`. The YuNet tensor preprocessing code can be reused; we only need to convert the `ndarray` buffer into an `ort::Tensor`. Session inputs are borrowed slices, so we can avoid the extra `Tensor` allocation we currently need for `tract`.
3. **Execution providers**:
   - **Windows**: DirectML works on any DX12 GPU (NVIDIA, AMD, Intel). Best option for the GUI.
   - **macOS (Apple Silicon)**: Use `features = ["coreml"]` to route through CoreML.
   - **Linux**: CUDA, TensorRT, or just CPU EP. Requires separate ONNX Runtime builds per provider.
4. **Packaging**: ONNX Runtime DLL/dylib/so files must be distributed with the app. The CLI/GUI launchers should validate that the shared library is present and fall back to `tract` if it is missing.
5. **Error reporting**: ONNX Runtime surfaces rich status codes; wrap them in `anyhow::Error` just like we do for `tract`. Add telemetry spans so we can compare latency.

## Migration Strategy

1. Keep `tract` as the baseline and introduce an optional `ort` backend guarded by a feature flag (`gpu-runtime`).
2. Define a `DetectorBackend` trait abstracting `run(&Tensor) -> DetectionOutput`. Implement it for both `tract` and `ort` so CLI/GUI can select backends via config flags (`--runtime tract|ort` or GUI combo box).
3. When `ort` is selected but the shared library is missing, log a warning and fall back to `tract`.
4. Benchmark against the existing Criterion suites (preprocessing + inference). Record DirectML/CoreML numbers alongside CPU baseline.

## Open Questions

- **Binary size**: Shipping ONNX Runtime increases download size by ~160 MB (full package). We can slim it down by only bundling the Execution Providers we need.
- **Licensing**: ONNX Runtime is MIT, so redistribution is OK. Document attribution in `NOTICE.md`.
- **Model compatibility**: ONNX Runtime handles the malformed `face_detection_yunet_2023mar.onnx` because it relaxes shape constraints. If we adopt it, we could allow users to run the original model when they explicitly opt into ORT.
- **Telemetry**: Extend the existing timers to emit `runtime=tract|ort` so we can compare metrics in production logs.

## Next Steps

1. Prototype an `ort`-powered `YuNetOrtModel` behind a `gpu-runtime` Cargo feature.
2. Add CLI/GUI flags for runtime selection and configuration of execution providers (DirectML by default on Windows).
3. Expand the Criterion benchmark harness to run both runtimes for apples-to-apples comparisons.
4. Update `TODO.md` once the prototype lands; this document will serve as the reference for the implementation plan.
