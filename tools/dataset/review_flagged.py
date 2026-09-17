"""Contact sheet of eye labels whose left/right order is doubtful, for checking by eye.

  review_flagged.py <data_dir> [out.html]

Two patterns need a human: clicks that arrived right-to-left on screen, and pairs whose eyes
sit nearly in vertical line. Neither is wrong by itself -- a face rolled past 90 degrees
legitimately reverses, and a steeply tilted one legitimately stacks -- but neither can be told
from a misclick by arithmetic, and getting it wrong swaps a face's eyes in training.

Faces already carrying `upside_down` or `steep_roll` are left out, so this shows only what has
not been judged yet. Tiles are cut from the local JPEGs with CSS, so no image library is
needed; open the file locally. Green 1 is the first click, orange 2 the second.

ponytail: a fourth copy of the JPEG header reader. If another turns up, move it to a shared
module here rather than copying again.
"""

import json
import struct
import sys
from pathlib import Path

TILE = 200
STEEP = 0.08  # horizontal eye separation, as a fraction of box width


def jpeg_size(path: Path):
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
            (length,) = struct.unpack(">H", f.read(2))
            f.seek(length - 2, 1)


def main() -> None:
    data = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else data / "review_flagged.html"
    images = data / "images"

    last = {}
    for line in (data / "eye_labels.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            last[record["id"]] = record

    flagged = []
    for record in last.values():
        if record.get("skipped") or "viewer_left" not in record:
            continue
        if record.get("upside_down") or record.get("steep_roll"):
            continue  # already judged
        left, right = record["viewer_left"], record["viewer_right"]
        width = (record["box"][2] - record["box"][0]) or 1e-9
        if left[0] >= right[0]:
            flagged.append((record, "clicked right-to-left"))
        elif abs(right[0] - left[0]) / width < STEEP:
            flagged.append((record, "eyes nearly in vertical line"))

    tiles = []
    for record, why in flagged:
        stem = record["image"]
        size = jpeg_size(images / f"{stem}.jpg")
        if size is None:
            continue
        iw, ih = size
        x0, y0, x1, y1 = record["box"]
        bw, bh = (x1 - x0) * iw, (y1 - y0) * ih
        side = max(bw, bh) * 1.9
        sx, sy = x0 * iw + bw / 2 - side / 2, y0 * ih + bh / 2 - side / 2
        k = TILE / side
        dots = ""
        for index, (px, py) in enumerate((record["viewer_left"], record["viewer_right"])):
            dots += (f'<div class="d" style="left:{(px * iw - sx) * k - 5:.1f}px;'
                     f'top:{(py * ih - sy) * k - 5:.1f}px;'
                     f'background:{"#4f8" if index == 0 else "#fa4"}">{index + 1}</div>')
        tiles.append(
            f'<figure><div class="t" style="background-image:url({(images / f"{stem}.jpg").as_uri()});'
            f"background-size:{iw * k:.1f}px {ih * k:.1f}px;"
            f'background-position:{-sx * k:.1f}px {-sy * k:.1f}px">{dots}</div>'
            f"<figcaption>{stem[:8]}<br>{why}<br>{record.get('split', 'first pass')}</figcaption></figure>"
        )

    out.write_text(
        "<!doctype html><meta charset=utf-8><title>Eye labels to check</title><style>"
        "body{font:13px system-ui;margin:24px;background:#111;color:#eee}"
        "section{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0;width:200px}"
        f".t{{position:relative;width:{TILE}px;height:{TILE}px;background-repeat:no-repeat;"
        "border-radius:4px}"
        ".d{position:absolute;width:10px;height:10px;border-radius:50%;font-size:8px;"
        "color:#000;text-align:center;line-height:10px;font-weight:700}"
        "figcaption{color:#9ad;text-align:center;font-size:11px}h1{font-size:18px}"
        "p{color:#aaa;max-width:60em}</style>"
        "<h1>Click order to check</h1><p>Green <b>1</b> was the first click, orange <b>2</b> the "
        "second, and the order should run left to right across the face <i>as you see it</i>. "
        "A face rolled past vertical reverses that legitimately, which is the judgement needed "
        "here: upside down, or a slip?</p>"
        f"<section>{''.join(tiles)}</section>",
        encoding="utf-8",
    )
    print(f"{len(flagged)} faces to review -> {out}")


if __name__ == "__main__":
    main()
