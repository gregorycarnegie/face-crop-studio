"""Are YuNet's unmatched detections false positives, or faces Open Images never boxed?

  sample_unmatched.py <data_dir> <detections.json> <out.html> [count]

Builds a contact sheet from the detections that matched no ground-truth box. Tiles are cut
out of the local JPEGs with CSS, so it needs no image library -- open the HTML locally.
Matched detections are shown underneath as a control: if those look like faces and the
unmatched ones do too, the "false positives" are mostly missing labels.
"""
import json
import random
import struct
import sys
from collections import defaultdict
from pathlib import Path

DATA = Path(sys.argv[1]).resolve()  # absolute: tile backgrounds are file:// URIs
DETECTIONS = Path(sys.argv[2])
OUT = Path(sys.argv[3])
COUNT = int(sys.argv[4]) if len(sys.argv) > 4 else 60
IMAGES = DATA / "images"
TILE = 170  # px
PAD = 0.35  # of box size, so the tile shows context around the detection


def jpeg_size(path):
    with open(path, "rb") as f:
        if f.read(2) != b"\xff\xd8":
            return None
        while True:
            b = f.read(1)
            while b and b != b"\xff":
                b = f.read(1)
            m = f.read(1)
            while m == b"\xff":
                m = f.read(1)
            if not m:
                return None
            if 0xC0 <= m[0] <= 0xCF and m[0] not in (0xC4, 0xC8, 0xCC):
                f.read(3)
                h, w = struct.unpack(">HH", f.read(4))
                return w, h
            (ln,) = struct.unpack(">H", f.read(2))
            f.seek(ln - 2, 1)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


detected = defaultdict(list)
for rec in json.loads(DETECTIONS.read_text(encoding="utf-8")):
    stem = Path(rec["image"]).stem
    for d in rec["detections"]:
        x, y, w, h = d["bbox"]
        detected[stem].append((d["score"], (x, y, x + w, y + h)))

truth, ignore, sizes = defaultdict(list), defaultdict(list), {}
import csv  # noqa: E402  (kept next to its only use)

for split in ("validation", "test"):
    with open(DATA / f"faces-{split}-annotations-bbox.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stem = row["ImageID"]
            if stem not in sizes:
                path = IMAGES / f"{stem}.jpg"
                sizes[stem] = jpeg_size(path) if path.exists() else None
            if sizes[stem] is None:
                continue
            iw, ih = sizes[stem]
            box = (float(row["XMin"]) * iw, float(row["YMin"]) * ih,
                   float(row["XMax"]) * iw, float(row["YMax"]) * ih)
            target = ignore if row["IsGroupOf"] == "1" or row["IsDepiction"] == "1" else truth
            target[stem].append(box)

unmatched, matched = [], []
for stem, size in sizes.items():
    if size is None:
        continue
    gt = list(truth.get(stem, []))
    for score, box in sorted(detected.get(stem, []), key=lambda d: -d[0]):
        hit = max(((iou(box, g), i) for i, g in enumerate(gt)), default=(0.0, -1))
        if hit[0] >= 0.5:
            gt.pop(hit[1])
            matched.append((stem, box, score))
        elif not any(iou(box, ig) > 0.3 for ig in ignore.get(stem, [])):
            unmatched.append((stem, box, score))

random.seed(0)
picked = random.sample(unmatched, min(COUNT, len(unmatched)))
controls = random.sample(matched, min(COUNT // 3, len(matched)))
print(f"{len(unmatched)} unmatched of {sum(len(v) for v in detected.values())} detections; "
      f"showing {len(picked)} plus {len(controls)} matched controls")


def tiles(items):
    out = []
    for stem, (x0, y0, x1, y1), score in items:
        iw, ih = sizes[stem]
        pad = PAD * max(x1 - x0, y1 - y0)
        rx, ry = x0 - pad, y0 - pad
        side = max(x1 - x0, y1 - y0) + 2 * pad
        scale = TILE / side
        url = (IMAGES / f"{stem}.jpg").as_uri()
        out.append(
            f'<figure><div class="t" style="background-image:url({url});'
            f"background-size:{iw * scale:.1f}px {ih * scale:.1f}px;"
            f'background-position:{-rx * scale:.1f}px {-ry * scale:.1f}px"></div>'
            f"<figcaption>{score:.2f}</figcaption></figure>"
        )
    return "\n".join(out)


OUT.write_text(
    "<!doctype html><meta charset=utf-8><title>Unmatched detections</title>"
    "<style>body{font:14px system-ui;margin:24px;background:#111;color:#eee}"
    "section{display:flex;flex-wrap:wrap;gap:10px;margin:16px 0 32px}"
    "figure{margin:0}"
    f".t{{width:{TILE}px;height:{TILE}px;background-repeat:no-repeat;border-radius:4px}}"
    "figcaption{text-align:center;color:#999;font-size:12px}"
    "h2{font-weight:600} p{color:#aaa;max-width:60em}</style>"
    f"<h1>Do these look like faces?</h1><p>{len(unmatched)} detections matched no Open Images "
    "box. If most of the first group are real faces, they are missing labels rather than "
    "false positives, and the measured recall is pessimistic.</p>"
    f"<h2>Unmatched ({len(picked)} sampled)</h2><section>{tiles(picked)}</section>"
    f"<h2>Matched, as a control ({len(controls)})</h2><section>{tiles(controls)}</section>",
    encoding="utf-8",
)
print(f"wrote {OUT}")
