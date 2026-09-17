"""Measure a trained SCRFD model's eye-point error on the held-out test set.

  eval_eye_error.py <checkpoint.pth> <data_dir> [--thresh 0.5] [--limit N]

Reports detection recall and eye-point error against face_keypoints_test.json, so models
trained on different numbers of eye labels can be compared on the same footing.

Straight from the PyTorch checkpoint, deliberately. SCRFD's own inference wrapper reads an
exported ONNX model, and `tools/scrfd2onnx.py` fails under torch 2.x: its tracer refuses the
numpy arrays that mmdet's input builder passes through (`Only tuples, lists and Variables are
supported as JIT inputs`). The decoding in that wrapper is plain numpy over raw per-stride
outputs, though, so it is reused here against the head's own tensors instead -- no export, no
tracer.

Two details the head hands over and this has to finish itself: outside an ONNX export it
returns raw (N, C, H, W) maps with no sigmoid, and with two anchors per location the class map
is (1, 2, H, W), so anchor centres are duplicated per anchor exactly as the wrapper does.
Preprocessing and rescaling follow `SCRFD.detect`: letterbox into the top-left of a 640x640
canvas with the aspect kept, then divide predictions by that scale to return to image space.

Error is normalised by ground-truth box width rather than interocular distance: the usual
5-point NME divides by eye separation, which collapses towards zero on the steeply rolled
faces this corpus contains and inflates their error arbitrarily. Box width is stable across
roll and is the scale everything else here is reported in. Raw pixels are printed too, since
a normalised figure hides whether an error matters for cropping.

Only faces whose eyes were clicked *and* whose side is known are scored: `steep_roll` faces
carry points but no reliable left/right identity, so scoring them would measure our labelling
ambiguity rather than the model.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import median

import numpy as np


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(dets, thresh=0.4):
    """The wrapper's own NMS, kept identical so scores are comparable with upstream."""
    x1, y1, x2, y2, scores = (dets[:, i] for i in range(5))
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        overlap = (w * h) / (areas[i] + areas[order[1:]] - w * h)
        order = order[1:][overlap <= thresh]
    return keep


class Detector:
    """Boxes and keypoints from a checkpoint, decoded the way SCRFD's wrapper does."""

    def __init__(self, config_path, checkpoint_path, scrfd_tools, size=640):
        import torch
        from mmcv import Config
        from mmcv.runner import load_checkpoint
        from mmdet.models import build_detector

        sys.path.insert(0, scrfd_tools)
        from scrfd import distance2bbox, distance2kps  # SCRFD's own decoding helpers

        self.torch = torch
        self.distance2bbox = distance2bbox
        self.distance2kps = distance2kps
        self.size = size

        cfg = Config.fromfile(config_path)
        cfg.model.train_cfg = None
        self.model = build_detector(cfg.model)
        load_checkpoint(self.model, checkpoint_path, map_location="cpu")
        self.model.cuda().eval()

        head = self.model.bbox_head
        self.strides = [s[0] for s in head.anchor_generator.strides]
        self.num_anchors = head.anchor_generator.num_base_anchors[0]
        self.nk = head.NK
        self.centres: dict = {}

    def _anchor_centres(self, height, width, stride):
        key = (height, width, stride)
        if key not in self.centres:
            centres = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            centres = (centres * stride).reshape((-1, 2))
            if self.num_anchors > 1:
                centres = np.stack([centres] * self.num_anchors, axis=1).reshape((-1, 2))
            self.centres[key] = centres
        return self.centres[key]

    def detect(self, image, thresh=0.5):
        import cv2

        # Letterbox into the top-left of a square canvas, as SCRFD.detect does.
        im_ratio = image.shape[0] / image.shape[1]
        if im_ratio > 1:
            new_height, new_width = self.size, int(self.size / im_ratio)
        else:
            new_width, new_height = self.size, int(self.size * im_ratio)
        scale = new_height / image.shape[0]
        canvas = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        canvas[:new_height, :new_width, :] = cv2.resize(image, (new_width, new_height))

        blob = cv2.dnn.blobFromImage(
            canvas, 1.0 / 128, (self.size, self.size), (127.5, 127.5, 127.5), swapRB=True)
        with self.torch.no_grad():
            cls_scores, bbox_preds, kps_preds = self.model.forward_dummy(
                self.torch.from_numpy(blob).cuda())

        scores_all, boxes_all, kps_all = [], [], []
        for level, stride in enumerate(self.strides):
            # The head skips the sigmoid and the flatten outside an ONNX export, so do both.
            score = cls_scores[level].sigmoid().permute(0, 2, 3, 1).reshape(-1, 1)
            bbox = bbox_preds[level].permute(0, 2, 3, 1).reshape(-1, 4)
            kps = kps_preds[level].permute(0, 2, 3, 1).reshape(-1, self.nk * 2)
            score = score.cpu().numpy()
            bbox = bbox.cpu().numpy() * stride
            kps = kps.cpu().numpy() * stride

            height, width = cls_scores[level].shape[2:4]
            centres = self._anchor_centres(height, width, stride)
            keep = np.where(score.ravel() >= thresh)[0]
            if keep.size == 0:
                continue
            scores_all.append(score[keep])
            boxes_all.append(self.distance2bbox(centres, bbox)[keep])
            kps_all.append(self.distance2kps(centres, kps).reshape(-1, self.nk, 2)[keep])

        if not scores_all:
            return np.zeros((0, 5), dtype=np.float32), np.zeros((0, self.nk, 2), dtype=np.float32)

        scores = np.vstack(scores_all)
        boxes = np.vstack(boxes_all) / scale
        kpss = np.vstack(kps_all) / scale
        order = scores.ravel().argsort()[::-1]
        dets = np.hstack((boxes, scores)).astype(np.float32)[order]
        kpss = kpss[order]
        keep = nms(dets)
        return dets[keep], kpss[keep]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("data_dir")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N images")
    ap.add_argument("--config",
                    default="/home/grego/work/insightface/detection/scrfd/configs/scrfd/scrfd_fcs_500m.py")
    ap.add_argument("--scrfd-tools",
                    default="/home/grego/work/insightface/detection/scrfd/tools")
    args = ap.parse_args()

    import cv2

    data = Path(args.data_dir).resolve()
    doc = json.loads((data / "face_keypoints_test.json").read_text(encoding="utf-8"))
    images = {image["id"]: image for image in doc["images"]}
    by_image: dict[int, list] = {}
    for annotation in doc["annotations"]:
        by_image.setdefault(annotation["image_id"], []).append(annotation)

    detector = Detector(args.config, args.checkpoint, args.scrfd_tools)

    scored = matched = 0
    errors_px, errors_rel = [], []
    order = sorted(images.values(), key=lambda i: i["file_name"])
    if args.limit:
        order = order[: args.limit]

    for image in order:
        faces = [a for a in by_image.get(image["id"], [])
                 if a["num_keypoints"] == 2 and not a["fcs"]["steep_roll"]]
        if not faces:
            continue
        frame = cv2.imread(str(data / "images" / image["file_name"]))
        if frame is None:
            continue
        dets, kpss = detector.detect(frame, thresh=args.thresh)
        for annotation in faces:
            scored += 1
            x, y, w, h = annotation["bbox"]
            truth = (x, y, x + w, y + h)
            best, best_iou = -1, 0.5
            for index in range(len(dets)):
                overlap = iou(truth, dets[index][:4])
                if overlap >= best_iou:
                    best, best_iou = index, overlap
            if best < 0:
                continue
            matched += 1
            keypoints = annotation["keypoints"]
            for slot in (0, 1):  # right_eye, left_eye, in the subject's own frame
                gx, gy = keypoints[slot * 3], keypoints[slot * 3 + 1]
                px, py = kpss[best][slot]
                distance = math.dist((gx, gy), (float(px), float(py)))
                errors_px.append(distance)
                errors_rel.append(distance / w if w else 0.0)

    print(f"checkpoint: {Path(args.checkpoint).name}")
    print(f"faces scored: {scored} (eyes clicked, side known)")
    if scored:
        print(f"  detected at IoU>=0.5: {matched} ({matched / scored:.1%})")
    if errors_px:
        print(f"  eye error median: {median(errors_px):.2f} px "
              f"= {median(errors_rel):.3f} of box width")
        print(f"  eye error mean:   {sum(errors_px) / len(errors_px):.2f} px "
              f"= {sum(errors_rel) / len(errors_rel):.3f} of box width")
        within = sum(1 for e in errors_rel if e <= 0.05) / len(errors_rel)
        print(f"  within 5% of box width: {within:.1%} of eye points")
    else:
        print("  no eye points scored: nothing matched, so there is no error to report")


if __name__ == "__main__":
    main()
