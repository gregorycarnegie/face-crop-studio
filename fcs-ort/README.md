# fcs-ort

**Development-only** ONNX Runtime oracle for parity tests and benchmarks.

Application crates do not depend on or load this library. It dynamically loads an ONNX Runtime library (via `libloading`) from an explicit path or one discoverable by the platform loader, and requires API level 24. It exposes `Environment`, `Session`, `SessionOptions` and `OutputTensor` for comparing `fcs-core` output against the reference runtime.
