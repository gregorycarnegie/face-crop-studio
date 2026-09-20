"""Export a trained SCRFD checkpoint to ONNX, bypassing the broken upstream exporter.

  export_scrfd.py <config.py> <checkpoint.pth> <out.onnx> [--size 640]

`tools/scrfd2onnx.py` fails under torch 2.x: it builds its dummy input through mmdet's
`build_model_from_cfg`/input pipeline, which hands the tracer numpy arrays, and the tracer only
takes tensors, lists and tuples. The model itself traces fine -- the export just has to be
handed a plain tensor.

So this wraps the detector in a module whose `forward` is `feature_test`: extract features, run
the head, return the raw per-stride tensors (scores, box distances, keypoints for strides
8/16/32). Those are exactly what `eval_eye_error.py` already decodes in numpy, so nothing about
the decoding has to be re-derived -- and the ONNX graph has no NMS in it, which is the part that
breaks under a tracer anyway.

Verified before it is allowed to write: onnxruntime must reproduce torch's own outputs. An
export that does not match is worse than none, because it looks like a model.
"""

import argparse
from pathlib import Path

import numpy as np
import torch


class RawHead(torch.nn.Module):
    """The detector, exposed as image-in, raw per-stride tensors-out."""

    def __init__(self, detector):
        super().__init__()
        self.detector = detector

    def forward(self, image):
        outputs = self.detector.feature_test(image)
        flat = []
        for group in outputs:
            flat.extend(group if isinstance(group, (list, tuple)) else [group])
        return tuple(flat)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("out")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--opset", type=int, default=11)
    ap.add_argument("--tolerance", type=float, default=1e-4)
    args = ap.parse_args()

    from mmdet.apis import init_detector

    detector = init_detector(args.config, args.checkpoint, device="cpu")
    detector.forward = detector.feature_test  # keep hooks off the traced path
    model = RawHead(detector).eval()

    example = torch.randn(1, 3, args.size, args.size)
    with torch.no_grad():
        expected = model(example)
    names = [f"out_{i}" for i in range(len(expected))]
    print(f"{len(expected)} output tensors: {[tuple(t.shape) for t in expected]}")

    torch.onnx.export(
        model,
        example,
        args.out,
        input_names=["input.1"],
        output_names=names,
        opset_version=args.opset,
        do_constant_folding=True,
    )

    try:
        import onnxruntime as rt
    except ImportError:
        raise SystemExit("onnxruntime not installed: wrote the file WITHOUT verifying it")

    session = rt.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    probe = torch.randn(1, 3, args.size, args.size)
    with torch.no_grad():
        want = [t.numpy() for t in model(probe)]
    got = session.run(None, {"input.1": probe.numpy()})
    print(f"onnx output shapes: {[tuple(t.shape) for t in got]}")

    # Under `torch.onnx.is_in_onnx_export()` SCRFDHead emits the deployment layout: each
    # (1, A*C, H, W) map becomes (1, H*W*A, C), spatial-major with the anchors interleaved, and
    # the class maps additionally get a sigmoid. That is what upstream's ONNX wrapper and
    # `eval_eye_error.py` both decode, so the eager tensors are put through the same two steps
    # before comparing -- otherwise this compares logits against probabilities and "fails" on a
    # model that is perfectly correct.
    def as_deployed(eager: np.ndarray, channels: int) -> np.ndarray:
        deployed = eager.transpose(0, 2, 3, 1).reshape(1, -1, channels)
        if channels == 1:  # cls_out_channels == 1: the class map, which the export sigmoids
            deployed = 1.0 / (1.0 + np.exp(-deployed))
        return deployed

    worst = 0.0
    for eager, onnx_out in zip(want, got):
        reference = eager if eager.shape == onnx_out.shape else as_deployed(eager,
                                                                           onnx_out.shape[-1])
        if reference.shape != onnx_out.shape:
            raise SystemExit(f"cannot line up {eager.shape} with {onnx_out.shape}")
        worst = max(worst, float(np.abs(reference - onnx_out).max()))
    print(f"wrote {args.out}")
    print(f"worst torch-vs-onnx difference {worst:.3e} (tolerance {args.tolerance:.0e})")
    if worst > args.tolerance:
        raise SystemExit(f"export does not match the checkpoint: {worst:.3e}")
    print("EXPORT VERIFIED")


if __name__ == "__main__":
    main()
