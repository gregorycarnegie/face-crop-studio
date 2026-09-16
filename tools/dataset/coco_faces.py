"""How much licence-clean face data does COCO 2017 hold?

Reads annotations/person_keypoints_{train,val}2017.json straight from the zip.
Keypoint order: 0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear.
Visibility: 0 not labelled, 1 labelled but occluded, 2 labelled and visible.
"""
import json
import math
import sys
import zipfile
from collections import Counter

ZIP = sys.argv[1] if len(sys.argv) > 1 else "annotations_trainval2017.zip"
IOD_BUCKETS = [(0, 8), (8, 16), (16, 32), (32, math.inf)]  # inter-ocular distance, px


def bucket(iod: float) -> str:
    for lo, hi in IOD_BUCKETS[:-1]:
        if lo <= iod < hi:
            return f"{lo}-{hi}"
    return f"{IOD_BUCKETS[-1][0]}-inf"


def tiers(licences):
    strict, lenient = set(), set()
    for lid, name in licences.items():
        if "NonCommercial" in name:
            continue
        lenient.add(lid)
        if "ShareAlike" not in name and "NoDerivs" not in name:
            strict.add(lid)
    return strict, lenient


zf = zipfile.ZipFile(ZIP)
grand = Counter()
for split in ("train2017", "val2017"):
    data = json.load(zf.open(f"annotations/person_keypoints_{split}.json"))
    licences = {l["id"]: l["name"] for l in data["licenses"]}
    strict, lenient = tiers(licences)
    images = {im["id"]: im["license"] for im in data["images"]}

    per_image = {}
    for ann in data["annotations"]:
        s = per_image.setdefault(ann["image_id"], Counter())
        if ann["iscrowd"]:
            s["crowd"] += 1
            continue
        kp = ann["keypoints"]
        v = kp[2::3]
        if ann["num_keypoints"] == 0:
            s["unlabelled_person"] += 1
            continue
        if v[1] and v[2]:
            iod = math.dist(kp[3:5], kp[6:8])
            s["face_both_eyes"] += 1
            s["iod_" + bucket(iod)] += 1
            if v[1] == 2 and v[2] == 2:
                s["face_both_eyes_visible"] += 1
        elif v[0] or v[1] or v[2]:
            s["face_partial"] += 1  # profile or partly occluded: needs a hand-drawn box
        else:
            s["no_face_points"] += 1  # person facing away or head out of frame

    print(f"\n=== {split}: {len(images)} images")
    lic_counts = Counter(images.values())
    for lid in sorted(licences):
        tag = "strict" if lid in strict else "lenient" if lid in lenient else "EXCLUDED"
        print(f"  licence {lid} {licences[lid]:<45} {lic_counts[lid]:>7}  {tag}")

    for tier_name, ids in (("strict", strict), ("lenient", lenient), ("all licences", set(licences))):
        c = Counter()
        for img_id, lid in images.items():
            if lid not in ids:
                continue
            c["images"] += 1
            s = per_image.get(img_id)
            if not s:
                c["negatives (no person annotated)"] += 1
                continue
            c += Counter({k: v for k, v in s.items()})
            if s["face_both_eyes"]:
                c["images with a both-eyes face"] += 1
                if not s["crowd"] and not s["unlabelled_person"]:
                    c["  ...and no crowd / unlabelled person"] += 1
        grand += Counter({f"{tier_name}|{k}": v for k, v in c.items()})
        print(f"  [{tier_name}]")
        for k in ("images", "negatives (no person annotated)", "images with a both-eyes face",
                  "  ...and no crowd / unlabelled person", "face_both_eyes", "face_both_eyes_visible",
                  *("iod_" + bucket(lo) for lo, _ in IOD_BUCKETS),
                  "face_partial", "no_face_points", "unlabelled_person", "crowd"):
            print(f"    {k:<42} {c[k]:>8}")

print("\n=== train + val")
for tier_name in ("strict", "lenient", "all licences"):
    print(f"  [{tier_name}]")
    for key in sorted(k for k in grand if k.startswith(tier_name + "|")):
        print(f"    {key.split('|', 1)[1]:<42} {grand[key]:>8}")
