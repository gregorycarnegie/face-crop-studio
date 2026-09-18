"""Aligned crops, YuNet's eye line beside the refiner's, for judging by eye.

  refiner_contact_sheet.py <detections.json> <eye_refiner.pt> <out.html> [--count 40]

Every number measured so far comes from Open Images photos with clicked eye points. A real
corpus has neither, so this compares the two on images that do not belong to the test set at
all -- and compares what a user actually sees.

It shows *aligned crops*, not dots on faces. A three-degree eye-line error is invisible as two
markers and obvious as a tilted portrait, and tilt is the thing `fcs-core::face_cropper`
produces. Each row is one face: the crop levelled by YuNet's own landmarks, then the same crop
levelled by the refiner's, then the disagreement in degrees. Rows are ordered by that
disagreement, largest first, so the faces where the two models differ most come first -- if the
refiner is worse anywhere, it shows up at the top rather than buried.
"""

import argparse
import base64
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_eye_refiner import Refiner  # noqa: E402 - same directory by design


def stem_of(path: str) -> str:
    """Basename without extension, for a Windows path possibly read on Linux."""
    return path.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]


def to_local(path: str) -> str:
    """A path recorded on Windows, opened wherever this happens to be running.

    `fcs-cli --json` writes `\\\\?\\C:\\...`. Stripping that prefix leaves `C:/...`, which Linux
    cannot open at all -- and this has to run under WSL, because torch is not installed against
    the Windows interpreter. So the drive letter is mapped to /mnt as well.
    """
    path = path.replace("\\", "/")
    if path.startswith("//?/"):
        path = path[4:]
    if sys.platform != "win32" and re.match(r"^[A-Za-z]:/", path):
        path = f"/mnt/{path[0].lower()}/{path[3:]}"
    return path


def level(image, centre, side, tilt_degrees, size=160):
    """The crop face_cropper would produce: centred, scaled, rotated to level the eyes.

    `tilt_degrees` is the measured eye-line tilt, and is passed through to OpenCV
    unnegated. That is not the sign `face_cropper` uses, and the difference is not a
    bug in either: `imageproc::rotate_about_center` is documented as rotating
    *clockwise* for positive theta, so Rust negates; `cv2.getRotationMatrix2D` is
    counter-clockwise-positive, so Python must not. Reasoning by analogy from the Rust
    call site produced exactly that error once, and it is invisible in the output --
    the wrong sign turns a tilt of theta into 2*theta, which still looks like a
    plausibly-rotated face rather than anything obviously broken. Hence `_self_check`.
    """
    matrix = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])),
                                     tilt_degrees, size / side)
    matrix[0, 2] += size / 2 - centre[0]
    matrix[1, 2] += size / 2 - centre[1]
    return cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR)


def as_data_uri(image) -> str:
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def _self_check() -> None:
    """A synthetic tilted eye line must come out level, not twice as tilted."""
    size, half, tilt = 160, 40.0, 25.0
    centre = np.array([size / 2, size / 2], dtype=np.float32)
    radians = math.radians(tilt)
    offset = np.array([half * math.cos(radians), half * math.sin(radians)], dtype=np.float32)
    eyes = np.stack([centre - offset, centre + offset])

    measured = math.degrees(math.atan2(eyes[1][1] - eyes[0][1], eyes[1][0] - eyes[0][0]))
    matrix = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])), measured, 1.0)
    turned = (eyes @ matrix[:, :2].T) + matrix[:, 2]
    residual = math.degrees(math.atan2(turned[1][1] - turned[0][1],
                                       turned[1][0] - turned[0][0]))
    assert abs(residual) < 1e-4, f"rotation sign is wrong: {tilt} deg tilt left {residual} deg"


def main() -> None:
    _self_check()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("detections")
    ap.add_argument("checkpoint")
    ap.add_argument("out")
    ap.add_argument("--count", type=int, default=40)
    ap.add_argument("--min-box", type=float, default=80.0,
                    help="ignore faces narrower than this many pixels")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu")
    size = state["size"]
    device = torch.device(args.device)
    model = Refiner().to(device)
    model.load_state_dict(state["model"])
    model.eval()

    rows = []
    for record in json.loads(Path(args.detections).read_text(encoding="utf-8")):
        path = to_local(record["image"])
        frame = None
        for detection in record["detections"]:
            x, y, w, h = detection["bbox"]
            if w < args.min_box:
                continue
            if frame is None:
                frame = cv2.imread(path)
                if frame is None:
                    break
            centre = np.array([x + w / 2, y + h / 2], dtype=np.float32)
            side = max(w, h) * 1.25

            matrix = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])), 0,
                                             size / side)
            matrix[0, 2] += size / 2 - centre[0]
            matrix[1, 2] += size / 2 - centre[1]
            crop = cv2.warpAffine(frame, matrix, (size, size), flags=cv2.INTER_LINEAR)
            tensor = torch.from_numpy(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
            tensor = ((tensor.float() - 127.5) / 128.0).unsqueeze(0).to(device)
            with torch.no_grad():
                predicted = model(tensor)[0].cpu().numpy().reshape(2, 2) * size
            predicted = (predicted - size / 2) * (side / size) + centre

            own = detection["landmarks"]
            yunet_angle = math.degrees(math.atan2(own[1][1] - own[0][1], own[1][0] - own[0][0]))
            refiner_angle = math.degrees(math.atan2(predicted[1][1] - predicted[0][1],
                                                    predicted[1][0] - predicted[0][0]))
            gap = abs(yunet_angle - refiner_angle) % 360
            gap = min(gap, 360 - gap)

            rows.append({
                "name": stem_of(record["image"])[:24],
                "gap": gap,
                "yunet": as_data_uri(level(frame, centre, side, yunet_angle)),
                "refiner": as_data_uri(level(frame, centre, side, refiner_angle)),
                "yunet_angle": yunet_angle,
                "refiner_angle": refiner_angle,
            })

    rows.sort(key=lambda r: -r["gap"])
    shown = rows[: args.count]
    print(f"{len(rows)} faces at least {args.min_box:.0f}px wide; "
          f"showing the {len(shown)} where the two disagree most")
    if rows:
        gaps = sorted(r["gap"] for r in rows)
        print(f"disagreement median {gaps[len(gaps) // 2]:.2f} deg, "
              f"90th percentile {gaps[int(len(gaps) * 0.9)]:.2f} deg, "
              f"max {gaps[-1]:.2f} deg")

    cells = "\n".join(
        f'<figure><div class="pair"><img src="{r["yunet"]}" title="YuNet {r["yunet_angle"]:.1f} deg">'
        f'<img src="{r["refiner"]}" title="refiner {r["refiner_angle"]:.1f} deg"></div>'
        f'<figcaption>{r["name"]}<br>{r["gap"]:.1f} deg apart</figcaption></figure>'
        for r in shown)

    Path(args.out).write_text(
        "<!doctype html><meta charset=utf-8><title>Eye-line alignment: YuNet vs refiner</title>"
        "<style>body{font:13px system-ui;margin:24px;background:#111;color:#eee}"
        "section{display:flex;flex-wrap:wrap;gap:18px}figure{margin:0}"
        ".pair{display:flex;gap:4px}.pair img{width:160px;height:160px;border-radius:4px}"
        "figcaption{text-align:center;color:#9ad;font-size:11px;margin-top:4px}"
        "h1{font-size:18px}p{color:#aaa;max-width:62em}</style>"
        "<h1>Eye-line alignment: YuNet (left) vs refiner (right)</h1>"
        "<p>Each pair is the same face levelled by each model's eye points, which is what the "
        "cropper produces. Sorted by how much the two disagree, largest first, so if the "
        "refiner is worse anywhere it appears at the top. The question is simply which column "
        "looks more upright.</p>"
        f"<section>{cells}</section>", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
