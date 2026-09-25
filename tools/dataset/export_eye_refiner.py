"""Export a trained eye refiner to ONNX, and prove the export matches the checkpoint.

  export_eye_refiner.py <checkpoint.pt> <out.onnx>

This exists because the first export did not have it. The model was exported by hand in a
shell, and the command chose `eye_refiner.pt` -- the checkpoint selected by the best of thirty
evaluations against the *test* set -- rather than `eye_refiner_val.pt`, the retrained one
selected on held-out validation that every reported figure comes from. The two differ by
about five pixels on a 500-pixel face, which is exactly the selection bias that retraining
was done to remove, and nothing in the pipeline would have caught it: both files load, both
run, both produce plausible eyes.

So the export is a script, it names the checkpoint it read, and it refuses to write a file it
has not checked against torch.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stamp_model_metadata import stamp  # same directory by design
from train_eye_refiner import Refiner  # same directory by design

#: Names the Rust side depends on; see `fcs-core/src/eye_refiner.rs`.
INPUT_NAME = "crop"
OUTPUT_NAME = "eyes"

#: opset 12 keeps the graph to Conv/Relu/GlobalAveragePool/Flatten/Gemm, which is what the
#: bundled ONNX Runtime and every other consumer handle without surprises.
OPSET = 12


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("out")
    ap.add_argument("--tolerance", type=float, default=1e-5,
                    help="max absolute difference allowed between torch and the export")
    args = ap.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu")
    size = state["size"]
    model = Refiner()
    model.load_state_dict(state["model"])
    model.eval()

    # Whatever the checkpoint recorded about its own selection, printed so the provenance of a
    # shipped model is visible in the log that produced it rather than inferred later.
    for key in ("val_stats", "test_stats", "stats", "epoch"):
        if key in state:
            print(f"{key}: {state[key]}")

    example = torch.randn(1, 3, size, size)
    torch.onnx.export(
        model,
        example,
        args.out,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        opset_version=OPSET,
        dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
    )
    stamp(args.out, "eye_refiner")  # credit travels with the file; verified below like the rest

    # An export that does not reproduce the checkpoint is worse than no export, because it
    # looks like a model. Check before anyone can use the file.
    try:
        import onnxruntime as rt
    except ImportError:
        print("onnxruntime not installed: wrote the file WITHOUT verifying it", file=sys.stderr)
        raise SystemExit(2)

    session = rt.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    probe = torch.randn(3, 3, size, size)  # batch > 1 also exercises the dynamic axis
    with torch.no_grad():
        expected = model(probe).numpy()
    got = session.run(None, {INPUT_NAME: probe.numpy()})[0]

    worst = float(np.abs(expected - got).max())
    print(f"checkpoint: {Path(args.checkpoint).name}")
    print(f"wrote {args.out}")
    print(f"worst torch-vs-onnx difference {worst:.3e} (tolerance {args.tolerance:.0e})")
    if worst > args.tolerance:
        raise SystemExit(f"export does not match the checkpoint: {worst:.3e}")
    print("EXPORT VERIFIED")


if __name__ == "__main__":
    main()
