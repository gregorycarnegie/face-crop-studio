"""Turn the COCO keypoint files into SCRFD's labelv2.txt annotation format.

  to_labelv2.py <data_dir> [out_dir]

Reads face_keypoints_{train,test}.json and writes labelv2_{train,test}.txt beside them, the
format `RetinaFaceDataset` in SCRFD's vendored mmdet parses. Point `img_prefix` at the image
directory and `ann_file` at one of these.

Layout, one image at a time:

    # <file name> <width> <height>
    x1 y1 x2 y2  kp1x kp1y w1  kp2x kp2y w2  ...  kp5x kp5y w5

Boxes are corners here, not COCO's x/y/width/height. A keypoint written as `-1 -1 -1` is
ignored by the landmark loss while the box still trains, which is what makes partial labels
expressible: this corpus has eyes and never nose or mouth, and faces flagged `steep_roll` have
eyes whose left/right identity is unknown, so those are written as ignored too rather than
guessed at. `_parse_ann_line` rewrites any surviving weight to 1.0, so the value written for a
present point only has to be non-negative.
"""

import json
import sys
from pathlib import Path

NK = 5  # SCRFD's landmark count: right eye, left eye, nose, right mouth, left mouth
IGNORED = "-1 -1 -1"


def convert(coco_path: Path, out_path: Path) -> dict:
    doc = json.loads(coco_path.read_text(encoding="utf-8"))
    images = {image["id"]: image for image in doc["images"]}
    by_image: dict[int, list] = {}
    for annotation in doc["annotations"]:
        by_image.setdefault(annotation["image_id"], []).append(annotation)

    counts = {"images": 0, "faces": 0, "with_points": 0, "points_ignored": 0}
    lines = []
    for image_id, image in sorted(images.items(), key=lambda kv: kv[1]["file_name"]):
        faces = by_image.get(image_id, [])
        if not faces:
            continue
        counts["images"] += 1
        lines.append(f"# {image['file_name']} {image['width']} {image['height']}")
        for annotation in faces:
            counts["faces"] += 1
            x, y, w, h = annotation["bbox"]
            parts = [f"{x:.2f}", f"{y:.2f}", f"{x + w:.2f}", f"{y + h:.2f}"]
            keypoints = annotation["keypoints"]
            labelled = False
            for index in range(NK):
                px, py, visibility = keypoints[index * 3: index * 3 + 3]
                if visibility > 0:
                    parts.append(f"{px:.2f} {py:.2f} 1")
                    labelled = True
                else:
                    parts.append(IGNORED)
                    counts["points_ignored"] += 1
            counts["with_points"] += int(labelled)
            lines.append(" ".join(parts))

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return counts


def main() -> None:
    data = Path(sys.argv[1]).resolve()
    out_dir = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else data
    for split in ("train", "test"):
        coco_path = data / f"face_keypoints_{split}.json"
        if not coco_path.exists():
            print(f"no {coco_path.name}, skipping")
            continue
        out_path = out_dir / f"labelv2_{split}.txt"
        counts = convert(coco_path, out_path)
        print(f"{split}: {counts['images']} images, {counts['faces']} faces "
              f"({counts['with_points']} with eye points, "
              f"{counts['points_ignored']} individual keypoints ignored) -> {out_path}")


if __name__ == "__main__":
    main()
