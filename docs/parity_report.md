# OpenCV Parity Snapshot

Generated with:

```bash
cargo run --release -p fcs-core --example parity_report
```

Results (IoU threshold = 0.5):

- Expected detections: **164**
- YuNet detections: **168**
- Matched detections: **163**
- Average recall: **0.992**
- Average precision: **0.991**
- Mean IoU (matched): **0.983**
- Mean |score delta|: **0.0010**
- Worst IoU across matches: **0.899**
- Worst |score delta|: **0.0058**

## Notable gaps

- `074.jpg` — OpenCV reports no faces but the current pipeline produces a low-score false positive near the background. Tightening the score threshold to `0.92` suppresses it but would also hide valid low-confidence faces elsewhere, so it is documented instead of adjusted.
- `238_g.jpg` — two false positives on dense crowd background (OpenCV sees none). These occur at scores `0.91–0.92`; future work will experiment with crowd-aware NMS or adaptive thresholds.
- `240_g.jpg` — one of the six expected faces is missed (heavy occlusion). The IoU of the remaining matches is ≥0.98, so the gap is isolated to the occluded subject.
- `232_g.webp` — extra detection yielding local precision `0.667`; also linked to background clutter.

All other fixtures reach 100% recall and precision with score deltas within ±0.003. The telemetry example makes it easy to re-run this report after changes to preprocessing or postprocessing.

## GPU/CPU parity coverage

`fcs-core/tests/gpu_cpu_parity.rs` compares the WGSL path against the CPU path
over six fixtures (`001`, `006`, `068`, `168_o`, `190_g`, `014_n`) with absolute
tolerances: score `1e-3`, bbox `5.0` px, landmark `5.0` px. It passes.

`200.jpg` is **not** in that set and would fail it. Running both paths through
the CLI:

| quantity   | CPU      | GPU      | delta  | tolerance |
|------------|----------|----------|--------|-----------|
| score      | 0.948538 | 0.947798 | 0.0007 | 1e-3      |
| bbox w     | 864.481  | 852.635  | 11.85  | 5.0       |
| bbox h     | 1345.066 | 1338.849 | 6.22   | 5.0       |
| landmark 4 | —        | —        | 5.76   | 5.0       |

The score barely moves, so both paths find the same face with the same
confidence; the box is simply drawn slightly differently. This is the GPU
preprocessing resampling difference that 1.6.0 reduced but did not eliminate,
not an inference difference — with preprocessing held constant, the CPU graph
and ONNX Runtime agree to 0.000000 score and 0.0002 px.

The size dependence is the thing worth deciding. The tolerances are absolute, so
a 5 px allowance is 0.12% of a 4032-tall image and 0.8% of a 609-tall one: large
fixtures are held to a proportionally stricter standard, and 200.jpg is
3024x4032. Note this is not simply a downscale-factor effect — `168_o` is the
same 3024x4032 and passes, and `190_g` downscales harder still (9.0x). It is
content-dependent.

Two reasonable responses, neither taken here because both are policy decisions
about detection quality rather than bugs to fix: add `200.jpg` and relax the
limits, or make the tolerances relative to the image or box size. Adding the
fixture under the current absolute limits would simply turn the suite red.
