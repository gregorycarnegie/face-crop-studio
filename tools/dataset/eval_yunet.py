"""Score YuNet against Open Images' human-drawn face boxes (validation + test).

  eval_yunet.py <detections.json> [images_dir] [data_dir]

Ground truth: faces-{validation,test}-annotations-bbox.csv, minus group-of boxes (one box over
several faces, used as ignore regions) and depictions (drawings, statues, screens).

Face size is reported as box width scaled to the detector's 640 px input, since that, not the
original pixel size, decides whether a face is findable. Image dimensions come from the JPEG
header, so no image library is needed.
"""
import json
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

DATA = Path(sys.argv[3]) if len(sys.argv) > 3 else Path.cwd()
IOU_MATCH = 0.5
BUCKETS = [(16, 32), (32, 64), (64, 128), (128, 10**9)]


def jpeg_size(path):
    """(width, height) from the first SOF marker, or None if it is not parseable."""
    with open(path, "rb") as f:
        if f.read(2) != b"\xff\xd8":
            return None
        while True:
            b = f.read(1)
            while b and b != b"\xff":
                b = f.read(1)
            marker = f.read(1)
            while marker == b"\xff":
                marker = f.read(1)
            if not marker:
                return None
            m = marker[0]
            if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
                f.read(3)
                h, w = struct.unpack(">HH", f.read(4))
                return w, h
            (length,) = struct.unpack(">H", f.read(2))
            f.seek(length - 2, 1)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def bucket(width_at_640):
    for lo, hi in BUCKETS:
        if lo <= width_at_640 < hi:
            return f"{lo}-{hi if hi < 10**9 else 'inf'}px"
    return "<16px"


det_path = Path(sys.argv[1])
images_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else DATA / "images"

# Detections, keyed by image id (the file stem).
detected = defaultdict(list)
raw = json.loads(det_path.read_text(encoding="utf-8"))
records = raw if isinstance(raw, list) else raw.get("images", raw.get("results", []))
for rec in records:
    stem = Path(rec["image"]).stem
    for d in rec["detections"]:
        x, y, w, h = d["bbox"]
        detected[stem].append((d["score"], (x, y, x + w, y + h)))

# Ground truth, in pixels, plus the ignore regions.
truth, ignore = defaultdict(list), defaultdict(list)
sizes, missing_size = {}, 0
for split in ("validation", "test"):
    with open(DATA / f"faces-{split}-annotations-bbox.csv", newline="", encoding="utf-8") as f:
        import csv

        for row in csv.DictReader(f):
            stem = row["ImageID"]
            if stem not in sizes:
                path = images_dir / f"{stem}.jpg"
                sizes[stem] = jpeg_size(path) if path.exists() else None
            size = sizes[stem]
            if size is None:
                missing_size += 1
                continue
            iw, ih = size
            box = (float(row["XMin"]) * iw, float(row["YMin"]) * ih,
                   float(row["XMax"]) * iw, float(row["YMax"]) * ih)
            width_at_640 = (float(row["XMax"]) - float(row["XMin"])) * 640
            if row["IsGroupOf"] == "1" or row["IsDepiction"] == "1":
                ignore[stem].append(box)
            else:
                truth[stem].append((box, width_at_640))

scored = [s for s in sizes if sizes[s] is not None]
print(f"{len(scored)} images scored, {len(sizes) - len(scored)} missing or unreadable "
      f"({missing_size} boxes skipped)")
print(f"{sum(len(v) for v in truth.values())} ground-truth faces, "
      f"{sum(len(v) for v in ignore.values())} ignore regions, "
      f"{sum(len(detected[s]) for s in scored)} detections\n")

found, total, extra = Counter(), Counter(), 0
h_ratio, w_ratio, y_shift = [], [], []
for stem in scored:
    gt = truth.get(stem, [])
    dets = sorted(detected.get(stem, []), key=lambda d: -d[0])
    taken = set()
    for _, dbox in dets:
        best, best_iou = None, IOU_MATCH
        for i, (gbox, _) in enumerate(gt):
            if i in taken:
                continue
            score = iou(dbox, gbox)
            if score >= best_iou:
                best, best_iou = i, score
        if best is None:
            if not any(iou(dbox, ig) > 0.3 for ig in ignore.get(stem, [])):
                extra += 1
            continue
        taken.add(best)
        gbox, width_at_640 = gt[best]
        found[bucket(width_at_640)] += 1
        gh, gw = gbox[3] - gbox[1], gbox[2] - gbox[0]
        h_ratio.append((dbox[3] - dbox[1]) / gh)
        w_ratio.append((dbox[2] - dbox[0]) / gw)
        y_shift.append((((dbox[1] + dbox[3]) / 2) - ((gbox[1] + gbox[3]) / 2)) / gh)
    for _, width_at_640 in gt:
        total[bucket(width_at_640)] += 1

print(f"{'face width at 640px':<22} {'faces':>8} {'found':>8} {'recall':>8}")
for key in ["<16px", *[f"{lo}-{hi if hi < 10**9 else 'inf'}px" for lo, hi in BUCKETS]]:
    if total[key]:
        print(f"{key:<22} {total[key]:>8} {found[key]:>8} {found[key] / total[key]:>7.1%}")
t, f_ = sum(total.values()), sum(found.values())
croppable_keys = [f"{lo}-{hi if hi < 10**9 else 'inf'}px" for lo, hi in BUCKETS if lo >= 32]
croppable = sum(total[k] for k in croppable_keys)
croppable_found = sum(found[k] for k in croppable_keys)
print(f"{'all':<22} {t:>8} {f_:>8} {f_ / t:>7.1%}")
print(f"{'croppable (>=32px)':<22} {croppable:>8} {croppable_found:>8} {croppable_found / croppable:>7.1%}")
print(f"\nunmatched detections (excluding ignore regions): {extra}")
if h_ratio:
    print("\nbox extent, YuNet / Open Images, on matched pairs:")
    print(f"  height ratio   median {median(h_ratio):.2f}")
    print(f"  width ratio    median {median(w_ratio):.2f}")
    print(f"  centre y shift median {median(y_shift):+.2f} of GT height (+ is lower)")
