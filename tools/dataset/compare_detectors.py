"""Two detectors on a corpus with no labels: count the disagreements, then show them.

  compare_detectors.py <a.json> <b.json> <out.html> [--label-a YuNet] [--label-b SCRFD]

Both files are `fcs-cli --json` shaped (YuNet writes it; `scrfd_detect.py` writes it for a
checkpoint). Boxes are matched at IoU >= 0.5, and the counts that come back are: found by both,
found only by A, found only by B.

Without ground truth, "more detections" is not "better" -- the extra boxes could be faces or
furniture. So the numbers are only half of it, and the page is the other half: a crop of every
disagreement, biggest first, captioned with which model found it. The question a human answers
is the only one that matters here, one crop at a time: is that a face?

Comparison is only fair at matched false-positive rates, which this cannot measure and does not
try to. Pick the thresholds elsewhere -- on Open Images, YuNet at 0.8 and the 80k SCRFD at 0.4
both sit near 0.11-0.14 FP/image (SCRFD_80K.md) -- and pass detections already filtered.
"""

import argparse
import base64
import json
import re
import sys
from pathlib import Path


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def load(path: Path) -> dict[str, list]:
    out = {}
    for record in json.loads(path.read_text(encoding="utf-8")):
        name = record["image"].replace("\\", "/").rsplit("/", 1)[-1]
        out[name] = [[d["bbox"][0], d["bbox"][1], d["bbox"][0] + d["bbox"][2],
                      d["bbox"][1] + d["bbox"][3]] for d in record["detections"]]
    return out


def to_local(path: str) -> str:
    """A Windows path from the JSON, opened wherever this runs (WSL included)."""
    path = path.replace("\\", "/")
    path = path.removeprefix("//?/")
    if sys.platform != "win32" and re.match(r"^[A-Za-z]:/", path):
        path = f"/mnt/{path[0].lower()}/{path[3:]}"
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--count", type=int, default=60, help="crops per column on the page")
    ap.add_argument("--min-side", type=float, default=24.0,
                    help="ignore boxes narrower than this; tiny disagreements are unjudgeable")
    args = ap.parse_args()

    import cv2

    first = json.loads(args.a.read_text(encoding="utf-8"))
    paths = {r["image"].replace("\\", "/").rsplit("/", 1)[-1]: r["image"] for r in first}
    for record in json.loads(args.b.read_text(encoding="utf-8")):
        paths.setdefault(record["image"].replace("\\", "/").rsplit("/", 1)[-1], record["image"])

    a_boxes, b_boxes = load(args.a), load(args.b)
    both = only_a = only_b = 0
    images_with_any = 0
    unique = {"a": [], "b": []}

    for name in sorted(paths):
        aa = [x for x in a_boxes.get(name, []) if x[2] - x[0] >= args.min_side]
        bb = [x for x in b_boxes.get(name, []) if x[2] - x[0] >= args.min_side]
        if aa or bb:
            images_with_any += 1
        taken = set()
        for box in aa:
            match = next((i for i, other in enumerate(bb)
                          if i not in taken and iou(box, other) >= 0.5), None)
            if match is None:
                only_a += 1
                unique["a"].append((name, box))
            else:
                taken.add(match)
                both += 1
        for i, box in enumerate(bb):
            if i not in taken:
                only_b += 1
                unique["b"].append((name, box))

    print(f"images with a detection: {images_with_any}")
    print(f"{'both models':>22}: {both}")
    print(f"{'only ' + args.label_a:>22}: {only_a}")
    print(f"{'only ' + args.label_b:>22}: {only_b}")

    def crops(which: str) -> str:
        rows = sorted(unique[which], key=lambda item: -(item[1][2] - item[1][0]))[: args.count]
        cells = []
        for name, box in rows:
            frame = cv2.imread(to_local(paths[name]))
            if frame is None:
                continue
            height, width = frame.shape[:2]
            side = max(box[2] - box[0], box[3] - box[1]) * 1.6
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            x1, y1 = max(0, int(cx - side / 2)), max(0, int(cy - side / 2))
            x2, y2 = min(width, int(cx + side / 2)), min(height, int(cy + side / 2))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_AREA)
            ok, buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if not ok:
                continue
            uri = "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")
            cells.append(f'<figure><img src="{uri}">'
                         f'<figcaption>{name[:22]}<br>{int(box[2] - box[0])}px</figcaption></figure>')
        return "\n".join(cells)

    args.out.write_text(
        "<!doctype html><meta charset=utf-8><title>Detector disagreements</title>"
        "<style>body{font:13px system-ui;margin:24px;background:#111;color:#eee}"
        "section{display:flex;flex-wrap:wrap;gap:14px;margin-bottom:34px}figure{margin:0}"
        "img{width:160px;height:160px;border-radius:4px}"
        "figcaption{text-align:center;color:#9ad;font-size:11px;margin-top:4px}"
        "h1{font-size:18px}h2{font-size:15px;color:#ffd79a}p{color:#aaa;max-width:64em}</style>"
        "<h1>What one detector found and the other missed</h1>"
        f"<p>Matched at IoU&nbsp;0.5 over {images_with_any} images with any detection: "
        f"<b>{both}</b> faces found by both, <b>{only_a}</b> only by {args.label_a}, "
        f"<b>{only_b}</b> only by {args.label_b}. Boxes under {args.min_side:.0f}px wide are left "
        f"out. Largest first. The only question is whether each crop is a face.</p>"
        f"<h2>Found only by {args.label_a} ({only_a})</h2><section>{crops('a')}</section>"
        f"<h2>Found only by {args.label_b} ({only_b})</h2><section>{crops('b')}</section>",
        encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
