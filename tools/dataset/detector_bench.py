"""Score face detectors on COCO 2017, a labelled corpus none of them was trained on.

  detector_bench.py fetch <annotations_trainval2017.zip> <out_dir> [--negatives 2000]
  detector_bench.py oi <labelv2_test_all.txt> <images_dir> <models_dir> <cache_dir>
  detector_bench.py coco <out_dir from fetch> <models_dir> <cache_dir>

docs/MODEL_COMPARISON.md compared SCRFD-80k, RetinaFace R34 and the public SCRFD-500M on held-out
Open Images -- the domain SCRFD-80k was trained on, which can favour it. COCO is the second
domain: SCRFD-80k trained only on Open Images, the other two on WIDER FACE, so here none of the
three is at home. `oi` re-runs the October comparison with this harness's decoders; matching its
recall at the same budgets is what makes the `coco` numbers trustworthy.

COCO has no face boxes, but its person keypoints carry both eyes, placed by hand. A face counts
as found when a detection contains both of its eye points and is no wider than 8x their
separation (so one giant box cannot claim a crowd); matching is one-to-one in score order. False
positives come from two places: images with no person in them at all, where any detection is
wrong, and detections in person images landing outside every person box. A detection on a
person but on no labelled face -- a profile, a back of a head, a hand -- is neither, and is left
out. Only commercially usable image licences are fetched (NonCommercial excluded, as in
DATA_CARD.md); evaluation makes no derivative, so ShareAlike and NoDerivs images are usable.

Thresholds are the October ones, frozen before COCO was scored. The mismatch-budget table that
follows them selects thresholds on COCO itself and is descriptive only.

Models (models_dir): scrfd80k_500m_640.onnx, eye_refiner.onnx, retinaface_r34.onnx and
scrfd_500m.onnx, the last two from huggingface.co/yakhyo/uniface-weights; their SHA-256 are in
MODEL_COMPARISON.md and are checked here.
"""

import argparse
import hashlib
import json
import math
import random
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_eye_error import iou, nms  # same directory by design

SIZE = 640
FLOOR = 0.02  # proposal floor, as in October
FROZEN = {"SCRFD-80k": 0.4851, "RetinaFace R34": 0.9807, "Public SCRFD-500M": 0.6415}
SHA256 = {
    "scrfd80k_500m_640.onnx": "2437def9510f2af6431a5b2090ae9f357bc189cb0223be1b6c4d4ff085798e64",
    "retinaface_r34.onnx": "bd0263dc2a465d32859555cb1741f2d98991eb0053696e8ee33fec583d30e630",
    "scrfd_500m.onnx": "5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a",
    "eye_refiner.onnx": "16b4ed6e42889c0eb3d6ce5068e399c19f12ad686bd121bcf959db34f8cef976",
}
BUDGETS = (0.05, 0.1, 0.2, 0.5)
# COCO's no-person images hold few detections at the frozen thresholds (0.005-0.03 each), so
# its budgets start lower than October's.
COCO_BUDGETS = (0.01, 0.02, 0.05, 0.1)
IOD_BUCKETS = ((0, 8), (8, 16), (16, 32), (32, math.inf))


# ── Models ────────────────────────────────────────────────────────────────────

def session(path: Path):
    import onnxruntime as ort

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256[path.name]:
        sys.exit(f"{path.name}: sha256 {digest} is not the model MODEL_COMPARISON.md measured")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def letterbox(image):
    """Top-left into a 640 square, aspect kept -- SCRFD.detect's preprocessing, used for all."""
    import cv2

    scale = SIZE / max(image.shape[:2])
    width, height = int(image.shape[1] * scale), int(image.shape[0] * scale)
    canvas = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    canvas[:height, :width] = cv2.resize(image, (width, height))
    return canvas, scale


class Scrfd:
    """Either SCRFD export: nine tensors, scores already sigmoided, distances in stride units."""

    def __init__(self, path: Path):
        self.session = session(path)
        self.input = self.session.get_inputs()[0].name
        self.centres = {}
        for stride in (8, 16, 32):
            side = SIZE // stride
            grid = np.stack(np.mgrid[:side, :side][::-1], axis=-1).reshape(-1, 2) * stride
            self.centres[stride] = np.repeat(grid.astype(np.float32), 2, axis=0)  # 2 anchors

    def __call__(self, canvas):
        import cv2

        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128, (SIZE, SIZE), (127.5,) * 3, swapRB=True)
        outputs = self.session.run(None, {self.input: blob})
        rows = []
        for level, stride in enumerate((8, 16, 32)):
            scores = outputs[level].reshape(-1)
            keep = scores >= FLOOR
            centres = self.centres[stride][keep]
            box = outputs[level + 3].reshape(-1, 4)[keep] * stride
            kps = outputs[level + 6].reshape(-1, 10)[keep] * stride
            boxes = np.hstack([centres - box[:, :2], centres + box[:, 2:]])
            points = np.tile(centres, 5) + kps
            rows.append(np.hstack([boxes, scores[keep, None], points]))
        return np.vstack(rows)


class RetinaFace:
    """UniFace's RetinaFace: BGR minus the Caffe means, prior boxes, variances 0.1 and 0.2."""

    def __init__(self, path: Path):
        self.session = session(path)
        priors = []
        for step, sizes in ((8, (16, 32)), (16, (64, 128)), (32, (256, 512))):
            side = SIZE // step
            grid = np.stack(np.mgrid[:side, :side][::-1], axis=-1).reshape(-1, 2)
            centres = (grid + 0.5) * step / SIZE
            for centre in centres:
                for size in sizes:
                    priors.append((*centre, size / SIZE, size / SIZE))
        self.priors = np.array(priors, dtype=np.float32)

    def __call__(self, canvas):
        means = np.float32([104, 117, 123])
        blob = (canvas.astype(np.float32) - means).transpose(2, 0, 1)[None]
        loc, conf, landmarks = (o[0] for o in self.session.run(None, {"input": blob}))
        keep = conf[:, 1] >= FLOOR
        priors, loc, landmarks = self.priors[keep], loc[keep], landmarks[keep]
        centres = priors[:, :2] + loc[:, :2] * 0.1 * priors[:, 2:]
        sizes = priors[:, 2:] * np.exp(loc[:, 2:] * 0.2)
        boxes = np.hstack([centres - sizes / 2, centres + sizes / 2]) * SIZE
        points = (np.tile(priors[:, :2], 5) + landmarks * 0.1 * np.tile(priors[:, 2:], 5)) * SIZE
        return np.hstack([boxes, conf[keep, 1:2], points])


class Refiner:
    """fcs-core::eye_refiner in numpy: a 112 crop of the box grown 1.25x, eyes as fractions."""

    def __init__(self, path: Path):
        self.session = session(path)

    def __call__(self, image, box):
        import cv2

        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        scale = 112 / (max(box[2] - box[0], box[3] - box[1]) * 1.25)
        matrix = np.float32([[scale, 0, 56 - cx * scale], [0, scale, 56 - cy * scale]])
        crop = cv2.warpAffine(image, matrix, (112, 112), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        blob = ((crop[:, :, ::-1].astype(np.float32) - 127.5) / 128).transpose(2, 0, 1)[None]
        eyes = self.session.run(None, {"crop": blob})[0][0] * 112
        return [float((eyes[i] - 56) / scale + (cx if i % 2 == 0 else cy)) for i in range(4)]


def detect_all(names, images_dir: Path, models_dir: Path, cache: Path):
    """Every model over every image at the proposal floor, cached; refined eyes for the two
    detectors the crop pipeline was measured with, above their frozen thresholds."""
    import cv2

    cache.mkdir(parents=True, exist_ok=True)
    models = {
        "SCRFD-80k": lambda: Scrfd(models_dir / "scrfd80k_500m_640.onnx"),
        "RetinaFace R34": lambda: RetinaFace(models_dir / "retinaface_r34.onnx"),
        "Public SCRFD-500M": lambda: Scrfd(models_dir / "scrfd_500m.onnx"),
    }
    refined = {"SCRFD-80k", "RetinaFace R34"}
    results = {}
    for label, build in models.items():
        path = cache / f"{label.replace(' ', '_')}.json"
        if path.exists():
            results[label] = json.loads(path.read_text())
            continue
        model, refiner = build(), Refiner(models_dir / "eye_refiner.onnx")
        out = {}
        for index, name in enumerate(names, 1):
            image = cv2.imread(str(images_dir / name))
            if image is None:
                continue
            canvas, scale = letterbox(image)
            rows = model(canvas)
            rows[:, [0, 1, 2, 3, *range(5, 15)]] /= scale
            rows = rows[rows[:, 4].argsort()[::-1]]
            rows = rows[nms(rows[:, :5])] if len(rows) else rows
            dets = []
            for row in rows:
                det = {"box": row[:4].tolist(), "score": float(row[4]),
                       "eyes": row[5:9].tolist()}  # viewer's left first, as fcs-core
                if label in refined and row[4] >= FROZEN[label]:
                    det["refined"] = refiner(image, row[:4])
                dets.append(det)
            out[name] = dets
            if index % 500 == 0 or index == len(names):
                print(f"  {label}: {index}/{len(names)}", flush=True)
        path.write_text(json.dumps(out))
        results[label] = out
    return results


# ── Scoring ───────────────────────────────────────────────────────────────────

def budget_threshold(unmatched_scores, images, budget):
    """The lowest threshold leaving at most `budget` unmatched detections per image."""
    allowed = int(budget * images + 1e-9)
    scores = sorted(unmatched_scores, reverse=True)
    if len(scores) <= allowed:
        return FLOOR
    return float(np.nextafter(np.float32(scores[allowed]), np.float32(1)))


def recall_at(matches, total, threshold):
    return sum(1 for s in matches if s >= threshold) / total if total else 0.0


def eye_angle(right, left):
    """Eye-line tilt, subject's right eye to left: the angle face_cropper aligns."""
    return math.atan2(left[1] - right[1], left[0] - right[0])


def angle_error(truth, pred):
    difference = abs(math.degrees(eye_angle(*truth) - eye_angle(*pred))) % 360
    return min(difference, 360 - difference)


def score_oi(args):
    images, faces, ignore = [], {}, {}
    name = None
    for line in args.labels.read_text().splitlines():
        if line.startswith("#"):
            name = line.split()[1]
            images.append(name)
            faces[name], ignore[name] = [], []
        elif line.strip():
            values = line.split()
            box = [float(v) for v in values[:4]]
            (faces if len(values) > 5 else ignore)[name].append(box)

    results = detect_all(images, args.images, args.models, args.cache)
    total = sum(len(v) for v in faces.values())
    print(f"\nOpen Images held-out: {len(images)} images, {total} faces, "
          f"{sum(len(v) for v in ignore.values())} ignore regions")
    print(f"{'model':<20} {'thresh@0.1':>10} " + " ".join(f"{b:>8}" for b in BUDGETS))
    for label, by_image in results.items():
        matched, unmatched = [], []
        for name in images:
            taken = set()
            for det in by_image.get(name, []):
                best, best_iou = None, 0.5
                for i, truth in enumerate(faces[name]):
                    overlap = iou(det["box"], truth)
                    if i not in taken and overlap >= best_iou:
                        best, best_iou = i, overlap
                if best is not None:
                    taken.add(best)
                    matched.append(det["score"])
                elif not any(iou(det["box"], region) > 0.3 for region in ignore[name]):
                    unmatched.append(det["score"])
        cells = [recall_at(matched, total, budget_threshold(unmatched, len(images), b))
                 for b in BUDGETS]
        print(f"{label:<20} {budget_threshold(unmatched, len(images), 0.1):>10.4f} "
              + " ".join(f"{c:>8.1%}" for c in cells))
    print("  recall at each unmatched-detections-per-image budget; October: SCRFD-80k 0.4851, "
          "76.9% at 0.1")


def fetch(args):
    archive = zipfile.ZipFile(args.annotations)
    out = args.out
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(2017)
    manifest, downloads = [], []
    for split in ("train2017", "val2017"):
        data = json.load(archive.open(f"annotations/person_keypoints_{split}.json"))
        usable = {l["id"] for l in data["licenses"] if "NonCommercial" not in l["name"]}
        licences = {l["id"]: l["name"] for l in data["licenses"]}
        by_image = {}
        for ann in data["annotations"]:
            by_image.setdefault(ann["image_id"], []).append(ann)
        negatives = []
        for image in data["images"]:
            if image["license"] not in usable:
                continue
            anns = by_image.get(image["id"], [])
            if not anns:
                negatives.append(image)
                continue
            record = {"file": image["file_name"], "faces": [], "people": []}
            for ann in anns:
                x, y, w, h = ann["bbox"]
                person = [x, y, x + w, y + h]
                kp, v = ann["keypoints"], ann["keypoints"][2::3]
                if not ann["iscrowd"] and v[1] and v[2]:
                    # COCO's sides are the subject's own; index 2 is their right eye.
                    record["faces"].append({"right": kp[6:8], "left": kp[3:5],
                                            "visible": v[1] == 2 and v[2] == 2,
                                            "iod": math.dist(kp[3:5], kp[6:8])})
                record["people"].append(person)
            if record["faces"]:
                manifest.append(record | {"licence": licences[image["license"]],
                                          "flickr": image.get("flickr_url")})
                downloads.append(image)
        for image in rng.sample(negatives, min(len(negatives), args.negatives // 2)):
            manifest.append({"file": image["file_name"], "faces": [], "people": [],
                             "licence": licences[image["license"]],
                             "flickr": image.get("flickr_url")})
            downloads.append(image)

    def get(image):
        target = out / "images" / image["file_name"]
        if not target.exists():
            urllib.request.urlretrieve(image["coco_url"], target)

    with ThreadPoolExecutor(16) as pool:
        for index, _ in enumerate(pool.map(get, downloads), 1):
            if index % 1000 == 0 or index == len(downloads):
                print(f"{index}/{len(downloads)}", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest))
    faces = sum(len(r["faces"]) for r in manifest)
    print(f"wrote {out / 'manifest.json'}: {len(manifest)} images, {faces} faces, "
          f"{sum(1 for r in manifest if not r['people'])} negatives")


def score_coco(args):
    manifest = json.loads((args.data / "manifest.json").read_text())
    names = [r["file"] for r in manifest]
    results = detect_all(names, args.data / "images", args.models, args.cache)
    negatives = [r for r in manifest if not r["people"]]
    people = [r for r in manifest if r["people"]]
    faces = [(r["file"], i, f) for r in people for i, f in enumerate(r["faces"])]

    def bucket(face):
        lo, hi = next(b for b in IOD_BUCKETS if b[0] <= face["iod"] < b[1])
        return f"{lo}-{hi}" if hi != math.inf else f"{lo}+"

    # (file, face index) -> det per model; plus every scored detection's fate.
    found, false_neg, false_out = {}, {}, {}
    for label, by_image in results.items():
        found[label], false_neg[label], false_out[label] = {}, [], []
        for record in manifest:
            taken = set()
            for det in by_image.get(record["file"], []):
                x1, y1, x2, y2 = det["box"]
                inside = lambda p: x1 <= p[0] <= x2 and y1 <= p[1] <= y2
                candidates = [
                    (math.dist(((x1 + x2) / 2, (y1 + y2) / 2),
                               ((f["right"][0] + f["left"][0]) / 2,
                                (f["right"][1] + f["left"][1]) / 2)), i)
                    for i, f in enumerate(record["faces"])
                    if i not in taken and inside(f["right"]) and inside(f["left"])
                    and x2 - x1 <= 8 * max(f["iod"], 1.0)]
                if candidates:
                    i = min(candidates)[1]
                    taken.add(i)
                    found[label][(record["file"], i)] = det
                elif not record["people"]:
                    false_neg[label].append(det["score"])
                elif not any(p[0] <= (x1 + x2) / 2 <= p[2] and p[1] <= (y1 + y2) / 2 <= p[3]
                             for p in record["people"]):
                    false_out[label].append(det["score"])

    buckets = [bucket({"iod": lo}) for lo, _ in IOD_BUCKETS]
    counts = {b: sum(1 for *_, f in faces if bucket(f) == b) for b in buckets}
    print(f"\nCOCO 2017, licence-usable: {len(people)} images with {len(faces)} eye-labelled "
          f"faces, {len(negatives)} images with no person")
    print("faces by eye separation (px): " + ", ".join(f"{b}: {counts[b]}" for b in buckets))

    def recall(label, threshold, wanted=None):
        pool = [(n, i) for n, i, f in faces if wanted is None or bucket(f) in wanted]
        hits = sum(1 for key in pool if key in found[label]
                   and found[label][key]["score"] >= threshold)
        return hits / len(pool) if pool else 0.0

    usable = {b for b in buckets if not b.startswith("0-")}
    print("\nAt the October thresholds, frozen before scoring:")
    print(f"{'model':<20} {'thresh':>7} {'recall':>7} {'iod>=8':>7} "
          + " ".join(f"{b:>7}" for b in buckets) + f" {'FP/neg':>7} {'FP/out':>7}")
    for label, threshold in FROZEN.items():
        neg = sum(1 for s in false_neg[label] if s >= threshold) / len(negatives)
        out = sum(1 for s in false_out[label] if s >= threshold) / len(people)
        print(f"{label:<20} {threshold:>7.4f} {recall(label, threshold):>7.1%} "
              f"{recall(label, threshold, usable):>7.1%} "
              + " ".join(f"{recall(label, threshold, {b}):>7.1%}" for b in buckets)
              + f" {neg:>7.3f} {out:>7.3f}")
    print("  FP/neg: detections per no-person image. FP/out: per person image, outside every "
          "person box.")

    print("\nRecall (eyes >= 8 px apart) at a no-person-image FP budget, selected on COCO:")
    print(f"{'model':<20} " + " ".join(f"{b:>8}" for b in COCO_BUDGETS))
    for label in FROZEN:
        cells = [recall(label, budget_threshold(false_neg[label], len(negatives), b), usable)
                 for b in COCO_BUDGETS]
        print(f"{label:<20} " + " ".join(f"{c:>8.1%}" for c in cells))

    # Eye line on faces every detector found at its frozen threshold, eyes clearly visible.
    paired = [(n, i, f) for n, i, f in faces if f["visible"] and f["iod"] >= 16
              and all((n, i) in found[l] and found[l][(n, i)]["score"] >= FROZEN[l]
                      for l in FROZEN)]
    print(f"\nEye-line error on {len(paired)} faces found by all three, both eyes visible, "
          f">= 16 px apart:")
    print(f"{'pipeline':<36} {'median':>7} {'p90':>7} {'<=5 deg':>8} {'dist/iod':>9}")
    pipelines = [(f"{l} raw landmarks", l, "eyes") for l in FROZEN]
    pipelines += [("SCRFD-80k + eye refiner", "SCRFD-80k", "refined"),
                  ("RetinaFace + eye refiner", "RetinaFace R34", "refined")]
    for name, label, key in pipelines:
        errors, distances = [], []
        for n, i, f in paired:
            e = found[label][(n, i)][key]
            pred = ((e[0], e[1]), (e[2], e[3]))  # viewer's left = subject's right
            errors.append(angle_error((f["right"], f["left"]), pred))
            distances.append((math.dist(pred[0], f["right"]) + math.dist(pred[1], f["left"]))
                             / 2 / f["iod"])
        errors.sort()
        print(f"{name:<36} {median(errors):>6.2f}d {errors[int(0.9 * len(errors))]:>6.2f}d "
              f"{sum(1 for e in errors if e <= 5) / len(errors):>8.1%} "
              f"{median(distances):>9.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("annotations", type=Path)
    f.add_argument("out", type=Path)
    f.add_argument("--negatives", type=int, default=2000)
    o = sub.add_parser("oi")
    o.add_argument("labels", type=Path)
    o.add_argument("images", type=Path)
    o.add_argument("models", type=Path)
    o.add_argument("cache", type=Path)
    c = sub.add_parser("coco")
    c.add_argument("data", type=Path)
    c.add_argument("models", type=Path)
    c.add_argument("cache", type=Path)
    args = ap.parse_args()
    {"fetch": fetch, "oi": score_oi, "coco": score_coco}[args.command](args)


if __name__ == "__main__":
    main()
