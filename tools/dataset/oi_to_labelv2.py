"""Write SCRFD labelv2 files that box *every* face Open Images drew, not just the clicked ones.

  oi_to_labelv2.py <openimages_dir> train [--detections det.json]
  oi_to_labelv2.py <openimages_dir> test

Why this exists: `to_labelv2.py` builds from the eye-labelling JSON, which holds one entry per
*clicked* face. So `labelv2_train.txt` carried 2,500 boxes for 2,482 images whose Open Images
boxes number 12,070, and `labelv2_test.txt` 2,000 against 4,597. Every curve run was trained to
call ~9,300 real faces background, and every FP/image figure in CURVE_RESULTS.md counted real,
merely unclicked faces as false positives. The eye labels themselves were right; the boxes
around them were missing.

Each image's block is written from the Open Images CSV, so it holds every face box:

* A real face (not group-of, not a depiction) is a full 19-value line. If that face's eyes were
  clicked -- its box matches a line in the old labelv2 file at IoU >= 0.9 -- the old line is
  carried over verbatim, eye points and all. Otherwise its keypoints are `-1`, so it trains
  the box and not the landmark head.
* Group-of boxes and depictions become 5-value ignore regions (`x1 y1 x2 y2 1`): neither a face
  to find nor background to suppress. Depictions are the judgement call here -- statues, drawings,
  faces on screens -- and ignoring them leaves the detector free either way rather than teaching
  it one answer.
* With `--detections` (train only): a YuNet detection at 0.8 overlapping no Open Images box
  (IoU < 0.3) also becomes an ignore region. Open Images leaves ~0.4 faces per image unboxed,
  and DATA_CARD.md measured YuNet's unmatched detections at ~97-99% real faces; unlabelled, those
  would train as background.

The test file gets no YuNet-derived regions, deliberately: it is the ruler, and a ruler built
from the incumbent's own output would flatter the incumbent.
"""

import argparse
import csv
import json
import struct
from collections import defaultdict
from pathlib import Path

SPLITS = {
    "train": ["faces-oidv6-train-annotations-bbox.csv"],
    "test": ["faces-validation-annotations-bbox.csv", "faces-test-annotations-bbox.csv"],
}
OLD_LABELS = {"train": "labelv2_train.txt", "test": "labelv2_test.txt"}
NO_POINTS = " ".join(["-1"] * 15)


def jpeg_size(path: Path) -> tuple[int, int] | None:
    """Width and height from the SOF marker, reading only the header."""
    with open(path, "rb") as f:
        if f.read(2) != b"\xff\xd8":
            return None
        while True:
            byte = f.read(1)
            while byte and byte != b"\xff":
                byte = f.read(1)
            while byte == b"\xff":
                byte = f.read(1)
            if not byte:
                return None
            marker = byte[0]
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                continue  # markers with no length field
            (length,) = struct.unpack(">H", f.read(2))
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                _precision, height, width = struct.unpack(">BHH", f.read(5))
                return width, height
            f.seek(length - 2, 1)


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def old_lines(path: Path) -> dict[str, list[tuple[list[float], str]]]:
    """Clicked faces from the previous labelv2 file, keyed by image id."""
    out = defaultdict(list)
    image = None
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            image = line.split()[1].rsplit(".", 1)[0]
        elif line.strip():
            values = line.split()
            out[image].append(([float(v) for v in values[:4]], line.strip()))
    return out


def stem_of(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data", type=Path)
    ap.add_argument("split", choices=["train", "test"])
    ap.add_argument("--detections", type=Path, help="fcs-cli --json output (train only)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.detections and args.split == "test":
        ap.error("the test file must not carry YuNet-derived regions")

    boxes = defaultdict(list)
    for name in SPLITS[args.split]:
        with open(args.data / name, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                boxes[row["ImageID"]].append(row)

    if args.split == "train":
        with open(args.data / "train_manifest.csv", newline="", encoding="utf-8") as f:
            images = [row["ImageID"] for row in csv.DictReader(f)]
    else:
        images = sorted(old_lines(args.data / OLD_LABELS["test"]))  # the existing test images

    clicked = old_lines(args.data / OLD_LABELS[args.split])
    detections = defaultdict(list)
    if args.detections:
        for record in json.loads(args.detections.read_text(encoding="utf-8")):
            for det in record["detections"]:
                x, y, w, h = det["bbox"]
                detections[stem_of(record["image"])].append([x, y, x + w, y + h])

    counts = defaultdict(int)
    out_lines = []
    for image in images:
        path = args.data / "images" / f"{image}.jpg"
        if not path.exists():
            counts["missing_image"] += 1
            continue
        size = jpeg_size(path)
        if size is None:
            counts["unreadable_header"] += 1
            continue
        width, height = size
        out_lines.append(f"# {image}.jpg {width} {height}")
        counts["images"] += 1

        drawn = []
        carried = set()
        for row in boxes[image]:
            box = [float(row["XMin"]) * width, float(row["YMin"]) * height,
                   float(row["XMax"]) * width, float(row["YMax"]) * height]
            drawn.append(box)
            corners = " ".join(f"{v:.2f}" for v in box)
            if row["IsGroupOf"] == "1" or row["IsDepiction"] == "1":
                out_lines.append(f"{corners} 1")
                counts["ignore_group_or_depiction"] += 1
                continue
            match = next((i for i, (old_box, _) in enumerate(clicked.get(image, []))
                          if i not in carried and iou(box, old_box) >= 0.9), None)
            if match is not None:
                carried.add(match)
                out_lines.append(clicked[image][match][1])
                counts["faces_with_eyes"] += 1
            else:
                out_lines.append(f"{corners} {NO_POINTS}")
                counts["faces_box_only"] += 1
        counts["clicked_unmatched"] += len(clicked.get(image, [])) - len(carried)

        for det in detections.get(image, []):
            if all(iou(det, box) < 0.3 for box in drawn):
                out_lines.append(" ".join(f"{v:.2f}" for v in det) + " 1")
                counts["ignore_unboxed_yunet"] += 1

    out = args.out or args.data / f"labelv2_{args.split}_all.txt"
    out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    for key in sorted(counts):
        print(f"{key:>28}: {counts[key]:,}")
    print(f"wrote {out}")
    # Every clicked face must land on an Open Images box; if one did not, its eyes would be lost.
    if counts["clicked_unmatched"]:
        raise SystemExit(f"{counts['clicked_unmatched']} clicked faces matched no Open Images box")


if __name__ == "__main__":
    main()
