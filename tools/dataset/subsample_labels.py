"""Make nested label variants, to measure how many eye points the landmark head needs.

  subsample_labels.py <labelv2.txt> <out_dir> [sizes...]     (default: 400 800 1600 all)

Every variant keeps every image and every box. Only the *landmark* supervision shrinks: in the
400-face variant, 400 faces keep their eye points and the rest are rewritten as `-1 -1 -1`,
which SCRFD's `RetinaFaceDataset` drops from the landmark loss while still training the box.
That isolates "how many eye labels" from "how much data", which a curve built by throwing away
whole images could not do.

The selection is seeded and nested, so 400 is a subset of 800 is a subset of 1600. Without
that, a difference between two points on the curve could just be which faces got picked.
"""

import random
import sys
from pathlib import Path

KEYPOINT_FIELDS = 15  # 5 landmarks x (x, y, weight)
IGNORED = " ".join(["-1"] * KEYPOINT_FIELDS)


def has_points(line: str) -> bool:
    values = line.split()
    return len(values) >= 4 + KEYPOINT_FIELDS and any(v != "-1" for v in values[4:])


def main() -> None:
    source = Path(sys.argv[1]).resolve()
    out_dir = Path(sys.argv[2]).resolve()
    sizes = sys.argv[3:] or ["400", "800", "1600", "all"]
    lines = source.read_text(encoding="utf-8").splitlines()

    # Index the face lines that carry eye points; those are the only ones a size limits.
    labelled = [i for i, line in enumerate(lines)
                if not line.startswith("#") and has_points(line)]
    order = labelled[:]
    random.Random(0).shuffle(order)
    print(f"{source.name}: {len(lines)} lines, {len(labelled)} faces with eye points")

    for size in sizes:
        count = len(order) if size == "all" else min(int(size), len(order))
        keep = set(order[:count])
        out = list(lines)
        for index in labelled:
            if index not in keep:
                values = out[index].split()
                out[index] = " ".join(values[:4]) + " " + IGNORED
        name = f"{source.stem}_{count}.txt"
        (out_dir / name).write_text("\n".join(out) + "\n", encoding="utf-8")
        kept = sum(1 for i in labelled if i in keep)
        print(f"  {name}: {kept} faces keep their eye points, "
              f"{len(labelled) - kept} rewritten as ignored")
        if size != "all" and count < int(size):
            print(f"    (asked for {size}, only {len(order)} labelled faces exist)")


if __name__ == "__main__":
    main()
