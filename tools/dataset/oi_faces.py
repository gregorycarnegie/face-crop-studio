"""How much licence-clean, croppable face data does Open Images hold?

  oi_faces.py [data_dir]     (default: the current directory)

Inputs, in `data_dir`: faces-*-annotations-bbox.csv (box rows filtered to /m/0dzct Human face)
and *-images*-with-rotation.csv (per-image licence, author, URLs).

Box coordinates are normalised, and the image tables carry no pixel dimensions, so face size is
reported as box width as a fraction of image width. At the detector's 640 px input that fraction
times 640 is roughly the face width in pixels (exactly so for landscape images).
"""
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
SPLITS = {
    "train": ("faces-oidv6-train-annotations-bbox.csv", "train-images-boxable-with-rotation.csv"),
    "validation": ("faces-validation-annotations-bbox.csv", "validation-images-with-rotation.csv"),
    "test": ("faces-test-annotations-bbox.csv", "test-images-with-rotation.csv"),
}
WIDTH_BUCKETS = [0.025, 0.05, 0.10, 0.20, 1.01]  # upper bounds: <16, <32, <64, <128, >=128 px at 640


def width_bucket(frac):
    lo = 0.0
    for hi in WIDTH_BUCKETS:
        if frac < hi:
            return f"{lo * 640:>4.0f}-{min(hi, 1.0) * 640:<4.0f}px"
        lo = hi
    return "  full"


grand = defaultdict(Counter)
for split, (box_file, image_file) in SPLITS.items():
    boxes = defaultdict(list)
    with open(DATA /box_file, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            boxes[row["ImageID"]].append(row)

    licence_of = {}
    with open(DATA /image_file, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["ImageID"] in boxes:
                licence_of[row["ImageID"]] = (row["License"], bool(row["Author"].strip()))

    print(f"\n=== {split}: {len(boxes)} images with face boxes, "
          f"{sum(map(len, boxes.values()))} boxes, {len(boxes) - len(licence_of)} without a licence row")
    lic = Counter(l for l, _ in licence_of.values())
    for name, n in lic.most_common():
        print(f"  {n:>8}  {name}")
    print(f"  images with no Author recorded: {sum(not a for _, a in licence_of.values())}")

    for tier, keep in (("CC BY 2.0", lambda l: l.rstrip("/").endswith("licenses/by/2.0")),
                       ("any licence", lambda _: True)):
        c = grand[f"{tier}"]
        for image_id, rows in boxes.items():
            entry = licence_of.get(image_id)
            if entry is None or not keep(entry[0]):
                continue
            c["images"] += 1
            croppable_here = 0
            for r in rows:
                c["boxes"] += 1
                c[f"source {r['Source']}"] += 1
                group, depiction = r["IsGroupOf"] == "1", r["IsDepiction"] == "1"
                if group:
                    c["group-of boxes (ignore regions)"] += 1
                    continue
                if depiction:
                    c["depictions (drawings, statues, screens)"] += 1
                if r["IsOccluded"] == "1":
                    c["occluded"] += 1
                if r["IsTruncated"] == "1":
                    c["truncated"] += 1
                w = float(r["XMax"]) - float(r["XMin"])
                c[f"width {width_bucket(w)}"] += 1
                if w >= 0.05 and not depiction:
                    c["croppable faces (>=32 px, real, single)"] += 1
                    croppable_here += 1
            if croppable_here:
                c["images with a croppable face"] += 1
            if any(r["IsGroupOf"] == "1" for r in rows):
                c["images containing a group-of box"] += 1

for tier, c in grand.items():
    print(f"\n=== all splits, {tier}")
    for key in sorted(c):
        print(f"  {key:<44} {c[key]:>9}")
