"""Does the refiner give a better eye line than YuNet's own landmarks, on YuNet's own boxes?

  eval_refiner_vs_yunet.py <data_dir> <eye_refiner.pt> [detections.json]

`train_eye_refiner.py` scores itself against ground-truth boxes, which measures the refiner in
isolation. This measures what would actually ship: YuNet detects a face, the refiner re-reads
the eyes from YuNet's box, and the result is compared against YuNet's own landmarks on exactly
the same faces. A paired comparison, so neither side gets an easier set of faces than the
other.

Faces YuNet does not find are excluded from both columns. They are a recall problem, and the
refiner cannot fix a face that was never detected.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import median

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_eye_refiner import Refiner, usable  # same directory by design


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def stem_of(path: str) -> str:
    """Basename without extension, for a path recorded on Windows and possibly read on Linux.

    `fcs-cli --json` writes `image` as a Windows path, backslashes and all. `PurePath` on Linux
    does not treat those as separators, so `Path(...).stem` hands back the whole string and
    every lookup misses -- silently, as an empty result rather than an error.
    """
    return path.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]


def line_angle(first, second) -> float:
    return math.degrees(math.atan2(second[1] - first[1], second[0] - first[0]))


def difference(a: float, b: float) -> float:
    gap = abs(a - b) % 360
    return min(gap, 360 - gap)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_dir")
    ap.add_argument("checkpoint")
    ap.add_argument("detections", nargs="?", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    data = Path(args.data_dir).resolve()
    dump = Path(args.detections) if args.detections else data / "det_0.8.json"

    state = torch.load(args.checkpoint, map_location="cpu")
    size = state["size"]
    device = torch.device(args.device)
    model = Refiner().to(device)
    model.load_state_dict(state["model"])
    model.eval()
    # Checkpoints from before the validation split carried a single "stats"; later ones carry
    # "val_stats" for the epoch selection and "test_stats" for the one look at the test set.
    selected = state.get("val_stats") or state.get("stats") or {}
    held_out = state.get("test_stats") or {}
    described = [f"refiner from {Path(args.checkpoint).name}"]
    if selected:
        described.append(f"selected at {selected['angle_median']:.2f} deg")
    if held_out:
        described.append(f"{held_out['angle_median']:.2f} deg on the test set, ground-truth boxes")
    print(", ".join(described))

    document = json.loads((data / "face_keypoints_test.json").read_text(encoding="utf-8"))
    images = {image["id"]: image for image in document["images"]}
    by_image: dict[int, list] = {}
    for annotation in document["annotations"]:
        by_image.setdefault(annotation["image_id"], []).append(annotation)
    detected = {stem_of(record["image"]): record["detections"]
                for record in json.loads(dump.read_text(encoding="utf-8"))}

    refiner_angles, yunet_angles = [], []
    refiner_distance, yunet_distance = [], []
    better = 0

    for image in images.values():
        faces = [a for a in by_image.get(image["id"], []) if usable(a)]
        if not faces:
            continue
        dets = detected.get(Path(image["file_name"]).stem, [])
        if not dets:
            continue
        frame = cv2.imread(str(data / "images" / image["file_name"]))
        if frame is None:
            continue
        boxes = [(x, y, x + w, y + h) for x, y, w, h in (d["bbox"] for d in dets)]

        for annotation in faces:
            x, y, w, h = annotation["bbox"]
            truth_box = (x, y, x + w, y + h)
            best, best_iou = -1, 0.5
            for index, candidate in enumerate(boxes):
                overlap = iou(truth_box, candidate)
                if overlap >= best_iou:
                    best, best_iou = index, overlap
            if best < 0:
                continue

            keypoints = annotation["keypoints"]
            truth_eyes = ((keypoints[0], keypoints[1]), (keypoints[3], keypoints[4]))
            truth_angle = line_angle(*truth_eyes)

            # The refiner re-reads the eyes from YuNet's box, exactly as it would in the app.
            dx, dy, dw, dh = dets[best]["bbox"]
            centre = np.array([dx + dw / 2, dy + dh / 2], dtype=np.float32)
            side = max(dw, dh) * 1.25
            matrix = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])), 0,
                                             size / side)
            matrix[0, 2] += size / 2 - centre[0]
            matrix[1, 2] += size / 2 - centre[1]
            crop = cv2.warpAffine(frame, matrix, (size, size), flags=cv2.INTER_LINEAR)
            tensor = torch.from_numpy(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
            tensor = ((tensor.float() - 127.5) / 128.0).unsqueeze(0).to(device)
            with torch.no_grad():
                predicted = model(tensor)[0].cpu().numpy().reshape(2, 2) * size
            # Rotation is zero here, so the affine inverts to a scale about the crop centre.
            predicted = (predicted - size / 2) * (side / size) + centre

            refiner_angles.append(difference(truth_angle, line_angle(*predicted)))
            own = dets[best]["landmarks"]
            yunet_angles.append(difference(truth_angle, line_angle(own[0], own[1])))
            if refiner_angles[-1] < yunet_angles[-1]:
                better += 1
            for expected, got_refiner, got_yunet in zip(truth_eyes, predicted, own):
                refiner_distance.append(math.dist(expected, got_refiner) / w if w else 0.0)
                yunet_distance.append(math.dist(expected, (got_yunet[0], got_yunet[1])) / w
                                      if w else 0.0)

    total = len(refiner_angles)
    if not total:
        print("nothing matched, so there is nothing to compare")
        return

    print(f"\npaired on {total} faces that YuNet detected at IoU>=0.5\n")
    print(f"{'':<22} {'refiner':>10} {'YuNet':>10}")
    print(f"{'angle error, median':<22} {median(refiner_angles):>9.2f}d "
          f"{median(yunet_angles):>9.2f}d")
    for limit in (2, 5, 10):
        share_r = sum(1 for a in refiner_angles if a <= limit) / total
        share_y = sum(1 for a in yunet_angles if a <= limit) / total
        print(f"{'  within ' + str(limit) + ' deg':<22} {share_r:>9.1%} {share_y:>9.1%}")
    print(f"{'eye distance, median':<22} {median(refiner_distance):>9.3f}w "
          f"{median(yunet_distance):>9.3f}w")
    print(f"\nrefiner closer on {better} of {total} faces ({better / total:.1%})")


if __name__ == "__main__":
    main()
