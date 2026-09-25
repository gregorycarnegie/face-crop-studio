"""Run a trained SCRFD checkpoint over a folder, writing fcs-cli's detection JSON format.

  scrfd_detect.py <checkpoint.pth> <image_dir> <out.json> [--thresh 0.4] [--config ...]

So a checkpoint can be compared against YuNet on a corpus neither model was trained on, using
the same tooling: `fcs-cli --json` writes this shape, and `compare_detectors.py` reads it.

The decoding is `eval_eye_error.py`'s `Detector`, imported rather than repeated -- it reads the
PyTorch checkpoint directly because SCRFD's own `tools/scrfd2onnx.py` fails under torch 2.x.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_eye_error import Detector  # same directory by design

SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("image_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--thresh", type=float, default=0.4)
    ap.add_argument("--config",
                    default="/home/grego/work/insightface/detection/scrfd/configs/scrfd/"
                            "scrfd_fcs80k_500m.py")
    ap.add_argument("--scrfd-tools",
                    default="/home/grego/work/insightface/detection/scrfd/tools")
    args = ap.parse_args()

    import cv2

    detector = Detector(args.config, args.checkpoint, args.scrfd_tools)
    paths = sorted(p for p in args.image_dir.iterdir() if p.suffix.lower() in SUFFIXES)
    records, faces, unreadable = [], 0, 0
    for index, path in enumerate(paths, 1):
        frame = cv2.imread(str(path))
        if frame is None:
            unreadable += 1
            continue
        dets, kpss = detector.detect(frame, thresh=args.thresh)
        detections = []
        for row, points in zip(dets, kpss):
            x1, y1, x2, y2, score = (float(v) for v in row[:5])
            detections.append({
                "score": score,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "landmarks": [[float(p[0]), float(p[1])] for p in points],
            })
        faces += len(detections)
        records.append({"image": str(path), "detections": detections})
        if index % 200 == 0 or index == len(paths):
            print(f"{index}/{len(paths)} images, {faces} faces", flush=True)

    args.out.write_text(json.dumps(records), encoding="utf-8")
    print(f"wrote {args.out}: {len(records)} images, {faces} faces, {unreadable} unreadable")


if __name__ == "__main__":
    main()
