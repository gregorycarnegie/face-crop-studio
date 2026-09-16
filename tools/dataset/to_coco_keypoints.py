"""Turn clicked eye points into COCO-format keypoint annotations.

  to_coco_keypoints.py <data_dir> [out.json]

Reads <data_dir>/eye_labels.jsonl (from label_eyes.py) plus the Open Images box CSVs, and
writes a COCO keypoint file: images, one annotation per labelled face, and a `face` category
with five keypoints.

Why COCO and not RetinaFace's label.txt, which is what SCRFD's own training scripts read:
these labels are partial. Eyes are clicked, nose and mouth corners are not, and label.txt has
no way to say "these two points are known and those three are not" -- a face either carries
all five or a -1 placeholder. Writing it would discard every eye point. COCO keypoints carry a
per-point visibility flag, mmdetection and mmpose read the format directly, and it matches the
shape of COCO's own eye points (DATA_CARD section 3), so the two sources can merge later.

Keypoint order is the RetinaFace/YuNet one, in the subject's own frame:
right_eye, left_eye, nose, right_mouth, left_mouth. Resolving "which eye is which" from a
click needs the flags, because screen position only implies a side while the head's roll stays
within +/-90 degrees:

  unflagged    viewer-left click is the subject's RIGHT eye (measured: YuNet agrees 99.8%)
  upside_down  the face is inverted, so the two swap
  steep_roll   eyes nearly in vertical line; side is genuinely unknown, so neither is
               written as a keypoint. Both points are kept under `fcs.eyes_unordered` so a
               trainer that can use unordered pairs still has them.

Faces skipped during labelling keep their box with no keypoints: they are real human-drawn
faces where a second eye was not visible, which is exactly what a detector head should train
on and a landmark head should not.

ponytail: a third copy of the JPEG header reader lives here, after eval_yunet and
sample_unmatched. If a fourth appears, move it to a shared module in this directory.
"""

import csv
import json
import struct
import sys
from pathlib import Path

KEYPOINT_NAMES = ["right_eye", "left_eye", "nose", "right_mouth", "left_mouth"]
LICENSE = {"id": 1, "name": "CC BY 2.0", "url": "https://creativecommons.org/licenses/by/2.0/"}


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


def load_labels(path: Path) -> list[dict]:
    last = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            last[record["id"]] = record  # last line for a face wins
    return list(last.values())


def eye_keypoints(record: dict) -> tuple[list[float], int, list | None]:
    """(keypoints, num_keypoints, unordered_pair) for one labelled face."""
    flat = [0.0] * (len(KEYPOINT_NAMES) * 3)
    if record.get("skipped") or "viewer_left" not in record:
        return flat, 0, None
    left, right = record["viewer_left"], record["viewer_right"]
    if record.get("steep_roll"):
        return flat, 0, [left, right]  # side unknown: keep the points, supervise neither
    subject_right, subject_left = (right, left) if record.get("upside_down") else (left, right)
    for index, point in ((0, subject_right), (1, subject_left)):
        flat[index * 3] = point[0]
        flat[index * 3 + 1] = point[1]
        flat[index * 3 + 2] = 2.0  # COCO: labelled and visible
    return flat, 2, None


def main() -> None:
    data = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else data / "face_keypoints_coco.json"
    labels = load_labels(data / "eye_labels.jsonl")

    # Boxes are stored normalised, so every image needs its pixel size, and the CSVs carry the
    # occlusion flags worth passing through.
    flags: dict[tuple[str, str], dict] = {}
    for split in ("validation", "test"):
        path = data / f"faces-{split}-annotations-bbox.csv"
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row["ImageID"], f"{float(row['XMin']):.6f}")
                flags[key] = {
                    "occluded": row["IsOccluded"] == "1",
                    "truncated": row["IsTruncated"] == "1",
                    "split": split,
                }

    images: dict[str, dict] = {}
    annotations = []
    counts = {"with_eyes": 0, "side_unknown": 0, "skipped": 0, "no_size": 0}
    for record in sorted(labels, key=lambda r: r["id"]):
        stem = record["image"]
        if stem not in images:
            size = jpeg_size(data / "images" / f"{stem}.jpg")
            if size is None:
                counts["no_size"] += 1
                continue
            images[stem] = {
                "id": len(images) + 1,
                "file_name": f"{stem}.jpg",
                "width": size[0],
                "height": size[1],
                "license": LICENSE["id"],
                "open_images_id": stem,
            }
        image = images[stem]
        iw, ih = image["width"], image["height"]
        x0, y0, x1, y1 = record["box"]
        keypoints, num, unordered = eye_keypoints(record)
        for index in range(0, len(keypoints), 3):
            if keypoints[index + 2]:
                keypoints[index] = round(keypoints[index] * iw, 1)
                keypoints[index + 1] = round(keypoints[index + 1] * ih, 1)
        box = [round(x0 * iw, 1), round(y0 * ih, 1), round((x1 - x0) * iw, 1),
               round((y1 - y0) * ih, 1)]
        extra = flags.get((stem, f"{x0:.6f}"), {})
        annotations.append({
            "id": len(annotations) + 1,
            "image_id": image["id"],
            "category_id": 1,
            "bbox": box,
            "area": round(box[2] * box[3], 1),
            "iscrowd": 0,
            "keypoints": keypoints,
            "num_keypoints": num,
            "fcs": {
                "upside_down": bool(record.get("upside_down")),
                "steep_roll": bool(record.get("steep_roll")),
                "eyes_skipped": bool(record.get("skipped")),
                "eyes_unordered": [[round(p[0] * iw, 1), round(p[1] * ih, 1)]
                                   for p in unordered] if unordered else None,
                **extra,
            },
        })
        if record.get("skipped"):
            counts["skipped"] += 1
        elif unordered:
            counts["side_unknown"] += 1
        else:
            counts["with_eyes"] += 1

    document = {
        "info": {
            "description": "Open Images face boxes with eye keypoints clicked for Face Crop Studio",
            "source": "Open Images V7 validation+test, /m/0dzct Human face",
            "keypoint_order": "RetinaFace/YuNet, in the subject's own frame",
        },
        "licenses": [LICENSE],
        "images": list(images.values()),
        "annotations": annotations,
        "categories": [{
            "id": 1,
            "name": "face",
            "supercategory": "person",
            "keypoints": KEYPOINT_NAMES,
            "skeleton": [[1, 2]],
        }],
    }
    out.write_text(json.dumps(document), encoding="utf-8")
    print(f"{len(images)} images, {len(annotations)} annotations -> {out}")
    print(f"  both eyes, side known: {counts['with_eyes']}")
    print(f"  points kept but side unknown (steep_roll): {counts['side_unknown']}")
    print(f"  box only, eyes not visible: {counts['skipped']}")
    if counts["no_size"]:
        print(f"  dropped, image missing or unreadable: {counts['no_size']}")


if __name__ == "__main__":
    main()
