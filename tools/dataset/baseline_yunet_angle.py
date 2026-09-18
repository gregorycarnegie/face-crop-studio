"""Score YuNet's own eye points on the test set, as the baseline a new model has to beat.

  baseline_yunet_angle.py <data_dir> [detections.json]

Reads face_keypoints_test.json and a detection dump from `fcs-cli --json` (default
det_0.8.json), matches them the same way `eval_eye_error.py` does, and reports the same
metrics. Without this, the numbers from a trained model are uninterpretable: nobody knows
whether 6 degrees of eye-line error is good or bad until the shipping detector is measured on
identical faces with identical matching.

Landmark order matches ours: index 0 is the subject's right eye, which sits on the viewer's
left for an upright face (measured over 10,880 detections -- see the note in
`fcs-core/src/face_cropper.rs`). Only faces whose eyes were clicked and whose side is known are
scored, so `steep_roll` faces do not turn our labelling ambiguity into model error.
"""

import json
import math
import sys
from pathlib import Path
from statistics import median


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
    every lookup misses -- silently, as an empty result rather than an error. This script ran
    correctly only because it happened to be run under Windows Python.
    """
    return path.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]


def main() -> None:
    data = Path(sys.argv[1]).resolve()
    dump = Path(sys.argv[2]) if len(sys.argv) > 2 else data / "det_0.8.json"

    truth = json.loads((data / "face_keypoints_test.json").read_text(encoding="utf-8"))
    images = {image["id"]: image for image in truth["images"]}
    by_image: dict[int, list] = {}
    for annotation in truth["annotations"]:
        by_image.setdefault(annotation["image_id"], []).append(annotation)

    detected = {stem_of(record["image"]): record["detections"]
                for record in json.loads(dump.read_text(encoding="utf-8"))}

    scored = matched = 0
    angles, relative = [], []
    for image in images.values():
        faces = [a for a in by_image.get(image["id"], [])
                 if a["num_keypoints"] == 2 and not a["fcs"]["steep_roll"]]
        if not faces:
            continue
        dets = detected.get(Path(image["file_name"]).stem, [])
        boxes = [(x, y, x + w, y + h) for x, y, w, h in (d["bbox"] for d in dets)]
        for annotation in faces:
            scored += 1
            x, y, w, h = annotation["bbox"]
            box = (x, y, x + w, y + h)
            best, best_iou = -1, 0.5
            for index, candidate in enumerate(boxes):
                overlap = iou(box, candidate)
                if overlap >= best_iou:
                    best, best_iou = index, overlap
            if best < 0:
                continue
            matched += 1
            keypoints = annotation["keypoints"]
            truth_eyes = ((keypoints[0], keypoints[1]), (keypoints[3], keypoints[4]))
            predicted = dets[best]["landmarks"][0], dets[best]["landmarks"][1]
            truth_angle = math.atan2(truth_eyes[1][1] - truth_eyes[0][1],
                                     truth_eyes[1][0] - truth_eyes[0][0])
            pred_angle = math.atan2(predicted[1][1] - predicted[0][1],
                                    predicted[1][0] - predicted[0][0])
            difference = abs(math.degrees(truth_angle - pred_angle)) % 360
            angles.append(min(difference, 360 - difference))
            for expected, got in zip(truth_eyes, predicted):
                relative.append(math.dist(expected, (got[0], got[1])) / w if w else 0.0)

    print(f"baseline: {dump.name}")
    print(f"  scorable faces: {scored}")
    if scored:
        print(f"  detected at IoU>=0.5: {matched} ({matched / scored:.1%})")
    if angles:
        print(f"  eye-line angle error median: {median(angles):.2f} deg")
        for limit in (2, 5, 10):
            share = sum(1 for a in angles if a <= limit) / len(angles)
            print(f"    within {limit} deg: {share:.1%}")
        print(f"  eye distance median: {median(relative):.3f} of box width")
    else:
        print("  nothing matched, so there is no baseline to report")


if __name__ == "__main__":
    main()
