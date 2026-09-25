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
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stamp_model_metadata import stamp  # same directory by design


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


def fold_batchnorm(path: str) -> int:
    """Fold each BatchNorm into the convolution feeding it, in place.

    For `y = gamma * (conv(x) - mean) / sqrt(var + eps) + beta` the convolution absorbs both:

        w' = w * gamma / sqrt(var + eps)
        b' = (b - mean) * gamma / sqrt(var + eps) + beta

    The convolution keeps its own weight name, which is the whole point -- see the note at the
    export call. Returns how many were folded, so a graph that silently stops matching this
    shape reports zero rather than passing quietly.
    """
    import onnx
    from onnx import numpy_helper

    model = onnx.load(path)
    graph = model.graph
    tensors = {i.name: numpy_helper.to_array(i) for i in graph.initializer}

    # Give aliased weights a tensor of their own first.
    #
    # torch deduplicates parameters that happen to hold identical values: here
    # `neck.downsample_convs.0.conv.bias` and `neck.fpn_convs.1.conv.bias` are bit-identical
    # (their weights are not), so the export keeps one initializer and an Identity node
    # renaming it. ONNX Runtime follows that; a reader that looks weights up by name finds a
    # name with nothing behind it. Two of the sixty convolutions were affected.
    alias_nodes = [n for n in graph.node if n.op_type == "Identity" and n.input[0] in tensors]
    for node in alias_nodes:
        tensors[node.output[0]] = tensors[node.input[0]]
        graph.node.remove(node)
    if alias_nodes:
        print(f"materialised {len(alias_nodes)} weights that were aliases of another")
    producer = {output: node for node in graph.node for output in node.output}
    consumers: dict[str, list] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    folded, dead_nodes, dead_tensors = 0, [], set()
    for node in graph.node:
        if node.op_type != "BatchNormalization":
            continue
        conv = producer.get(node.input[0])
        # Only when the convolution feeds this BatchNorm and nothing else, or folding would
        # change what those other consumers see.
        if conv is None or conv.op_type != "Conv" or len(consumers.get(conv.output[0], [])) != 1:
            continue

        gamma, beta, mean, var = (tensors[n] for n in node.input[1:5])
        eps = next((a.f for a in node.attribute if a.name == "epsilon"), 1e-5)
        scale = gamma / np.sqrt(var + eps)

        weight_name = conv.input[1]
        weight = tensors[weight_name]
        tensors[weight_name] = weight * scale.reshape(-1, *([1] * (weight.ndim - 1)))

        has_bias = len(conv.input) > 2
        bias = tensors[conv.input[2]] if has_bias else np.zeros_like(mean)
        bias_name = conv.input[2] if has_bias else weight_name.rsplit(".", 1)[0] + ".bias"
        tensors[bias_name] = (bias - mean) * scale + beta
        if has_bias:
            conv.input[2] = bias_name
        else:
            conv.input.append(bias_name)

        conv.output[0] = node.output[0]  # the convolution now produces what the BatchNorm did
        dead_nodes.append(node)
        dead_tensors.update(node.input[1:5])
        folded += 1

    for node in dead_nodes:
        graph.node.remove(node)
    del graph.initializer[:]
    for name, array in tensors.items():
        if name not in dead_tensors:
            graph.initializer.append(numpy_helper.from_array(array, name))

    onnx.checker.check_model(model)
    onnx.save(model, path)
    return folded


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

    # `do_constant_folding=False` on purpose. Folding is what makes the graph fast, but torch's
    # folding discards the module paths: the folded run produced 82 tensors named
    # `onnx::Conv_533` out of 118. `fcs-core`'s built-in CPU and WGSL engines look weights up by
    # name (`crate::onnx`), and those numbers are assigned per export, so anything
    # pinned to them breaks the next time the model is regenerated. So the BatchNorms are
    # folded below instead, which keeps every name and produces the same numbers.
    torch.onnx.export(
        model,
        example,
        args.out,
        input_names=["input.1"],
        output_names=names,
        opset_version=args.opset,
        do_constant_folding=False,
    )
    folded = fold_batchnorm(args.out)
    print(f"folded {folded} BatchNorm nodes into their convolutions, keeping their names")
    stamp(args.out, "scrfd")  # credit travels with the file; verified below like the rest

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
