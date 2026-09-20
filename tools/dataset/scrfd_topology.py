"""Turn the exported SCRFD graph into the step table `fcs-core` executes.

  scrfd_topology.py <model.onnx> <out.rs>

`fcs-core`'s CPU and WGSL engines run a compiled topology rather than interpreting ONNX, the
way `crate::yunet` does for YuNet. YuNet's was written by hand, which is reasonable for its
size; SCRFD-500M is 60 convolutions across a backbone, a PAFPN neck and three head branches,
and hand-copying kernel shapes, strides and group counts is how a silent mismatch gets in.

So the table is generated from the graph that ships. Each step writes one slot and refers to
earlier slots by index, slot 0 being the input tensor. Only four kinds of step exist, because
that is all the graph needs once BatchNorm is folded (`export_scrfd.py`) and the final
reshape/transpose/sigmoid is left to the Rust decoder:

* `Conv` -- with stride, padding, groups and whether a ReLU follows
* `Add` -- the neck's lateral joins
* `Upsample2x` -- nearest, which is what the neck's `Resize` nodes are at these scales

The nine outputs are the per-stride class, box and keypoint convolutions, in the layout the
head produces them: `(1, A*C, H, W)`, before the deployment reshape.
"""

import argparse
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()

    import onnx
    from onnx import numpy_helper

    graph = onnx.load(args.model).graph
    shapes = {i.name: numpy_helper.to_array(i).shape for i in graph.initializer}
    producer = {output: node for node in graph.node for output in node.output}

    # Nodes that only compute the Resize target size from the input's own shape. A compiled
    # topology knows the factor is 2, so they carry no information worth reproducing.
    PLUMBING = {"Shape", "Constant", "Gather", "Unsqueeze", "Concat", "Slice", "Cast"}

    relu_after: dict[str, bool] = {}
    for node in graph.node:
        if node.op_type == "Relu":
            relu_after[node.input[0]] = True

    slots: dict[str, int] = {"input.1": 0}
    steps: list[str] = []

    def slot_of(name: str) -> int:
        """Index of the tensor, following ReLU and the size-computation nodes back to a step."""
        while name not in slots:
            node = producer.get(name)
            if node is None:
                sys.exit(f"no producer for {name}; the graph is not the shape this expects")
            # Followed through rather than reproduced: ReLU is folded into the convolution
            # step, and the sigmoid and the deployment reshape at the very end are the Rust
            # decoder's job (`scrfd::decode` reads the raw `(1, A*C, H, W)` maps).
            if node.op_type in PLUMBING | {"Relu", "Sigmoid", "Transpose", "Reshape", "Identity"}:
                name = node.input[0]
                continue
            sys.exit(f"{name} comes from {node.op_type}, which has no step kind")
        return slots[name]

    for node in graph.node:
        if node.op_type in PLUMBING | {"Relu", "Sigmoid", "Transpose", "Reshape", "Identity"}:
            continue

        if node.op_type == "Conv":
            attrs = {a.name: (list(a.ints) if a.ints else a.i) for a in node.attribute}
            weight = node.input[1]
            bias = node.input[2] if len(node.input) > 2 else ""
            out_channels, in_per_group = shapes[weight][0], shapes[weight][1]
            groups = attrs.get("group", 1)
            kernel = attrs.get("kernel_shape", [1, 1])[0]
            stride = attrs.get("strides", [1, 1])[0]
            pad = attrs.get("pads", [0, 0, 0, 0])[0]
            steps.append(
                f'    Step::Conv {{ input: {slot_of(node.input[0])}, weight: "{weight}", '
                f'bias: "{bias}", out_channels: {out_channels}, in_per_group: {in_per_group}, '
                f"kernel: {kernel}, stride: {stride}, pad: {pad}, groups: {groups}, "
                f"relu: {str(bool(relu_after.get(node.output[0], False))).lower()} }},"
            )
        elif node.op_type == "Add":
            steps.append(
                f"    Step::Add {{ a: {slot_of(node.input[0])}, b: {slot_of(node.input[1])} }},"
            )
        elif node.op_type == "Resize":
            steps.append(f"    Step::Upsample2x {{ input: {slot_of(node.input[0])} }},")
        else:
            sys.exit(f"unhandled op {node.op_type}; the generator needs a step kind for it")

        # Slot numbering is the executor's: the network input is slot 0, and step `i` writes
        # slot `i + 1`. Numbering steps from 0 instead makes step 1 read the input image rather
        # than step 0's output, which fails immediately on channel count -- but only because
        # the first two layers differ in shape. Later in the network it would have run.
        slots[node.output[0]] = len(steps)

    outputs = [slot_of(o.name) for o in graph.output]

    body = "\n".join(steps)
    args.out.write_text(
        "//! SCRFD-500M's topology, generated by `tools/dataset/scrfd_topology.py`.\n"
        "//!\n"
        "//! Do not edit: re-run the generator against the exported model instead. Every step\n"
        "//! writes one slot and refers to earlier slots by index, slot 0 being the input.\n"
        "//! Weights are looked up by name, which is why the exporter folds BatchNorm itself\n"
        "//! rather than letting torch's constant folding rename everything.\n\n"
        "use super::plan::Step;\n\n"
        f"/// The nine head outputs, as `(1, anchors*channels, H, W)` maps: class, box and\n"
        f"/// keypoint convolutions for strides 8, 16 and 32.\n"
        f"pub const OUTPUTS: [usize; {len(outputs)}] = {outputs};\n\n"
        f"/// {len(steps)} steps in execution order.\n"
        f"pub const STEPS: [Step; {len(steps)}] = [\n{body}\n];\n",
        encoding="utf-8",
    )
    kinds = {"Conv": 0, "Add": 0, "Upsample2x": 0}
    for step in steps:
        for kind in kinds:
            if f"Step::{kind} " in step:
                kinds[kind] += 1
    print(f"wrote {args.out}: {len(steps)} steps {kinds}, outputs at slots {outputs}")


if __name__ == "__main__":
    main()
