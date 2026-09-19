"""Pick N Open Images train images with croppable faces, record their credits, download them.

  oi_train_fetch.py <openimages_dir> [--count 80000] [--workers 32]

Why 80,000: that is about 164,000 croppable faces, the scale of WIDER FACE (159k), which is
what SCRFD and YuNet were trained on. The label-count curve (CURVE_RESULTS.md) found images,
not eye labels, to be the constraint; this is the run that tests it at industry scale.

Selection rules, each for a reason:

* **Train split only.** Validation and test are the held-out test set and carry clicked eyes.
* **At least one croppable face** -- real (not a depiction), not a group-of box, and at least
  5% of image width (32 px at the detector's 640 input), the same bar as DATA_CARD.md.
* **Rotation recorded as 0.0.** Nothing in this pipeline reads Open Images' `Rotation` column,
  and ~1% of train images are marked 90/180/270 while 14% record nothing. Rather than guess
  which frame the boxes were drawn in, those are left out; the pool is ~3x what is needed.
* **The 2,482 images behind `labelv2_train.txt` always go in**, because they carry the clicked
  eye pairs that supervise the landmark head.

Writes `train_manifest.csv` beside the data: one row per image with its licence, author,
author URL and landing page. CC BY 2.0 requires attribution, and DATA_CARD.md section 5 asks
for the manifest to travel with the weights. Downloads go into `images/` next to the existing
corpus, skip anything already there, and so can be interrupted and rerun.
"""

import argparse
import csv
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MIRROR = "https://open-images-dataset.s3.amazonaws.com/train/{image}.jpg"
SEED = 20260919
MIN_WIDTH = 0.05


def croppable_images(boxes_csv: Path) -> set[str]:
    keep = set()
    with open(boxes_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["IsGroupOf"] != "0" or row["IsDepiction"] != "0":
                continue
            if float(row["XMax"]) - float(row["XMin"]) >= MIN_WIDTH:
                keep.add(row["ImageID"])
    return keep


def labelled_images(labelv2: Path) -> set[str]:
    names = set()
    if labelv2.exists():
        for line in labelv2.read_text(encoding="utf-8").splitlines():
            if line.startswith("#"):
                names.add(line.split()[1].rsplit(".", 1)[0])
    return names


def fetch(image: str, dest: Path) -> tuple[str, str | None]:
    path = dest / f"{image}.jpg"
    if path.exists() and path.stat().st_size > 0:
        return image, None
    try:
        with urllib.request.urlopen(MIRROR.format(image=image), timeout=60) as response:
            data = response.read()
        tmp = path.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(path)  # never leave a truncated .jpg that a rerun would skip
        return image, None
    except Exception as err:  # noqa: BLE001 - reported per image, the batch carries on
        return image, str(err)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data", type=Path)
    ap.add_argument("--count", type=int, default=80_000)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--select-only", action="store_true")
    args = ap.parse_args()

    candidates = croppable_images(args.data / "faces-oidv6-train-annotations-bbox.csv")
    required = labelled_images(args.data / "labelv2_train.txt")
    print(f"train images with a croppable face: {len(candidates):,}; required (eye-labelled): "
          f"{len(required):,}", flush=True)

    rows = {}
    with open(args.data / "train-images-boxable-with-rotation.csv", newline="",
              encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image = row["ImageID"]
            if image in required or (image in candidates and row["Rotation"] == "0.0"):
                rows[image] = row

    missing = required - rows.keys()
    if missing:
        sys.exit(f"{len(missing)} eye-labelled images are absent from the image table")
    pool = sorted(rows.keys() - required)
    random.Random(SEED).shuffle(pool)
    chosen = sorted(required) + pool[: max(0, args.count - len(required))]
    print(f"eligible pool {len(pool) + len(required):,}; chosen {len(chosen):,}", flush=True)

    manifest = args.data / "train_manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        fields = ["ImageID", "License", "Author", "AuthorProfileURL", "OriginalLandingURL"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for image in chosen:
            writer.writerow({k: rows[image][k] for k in fields})
    print(f"wrote {manifest}", flush=True)
    if args.select_only:
        return

    dest = args.data / "images"
    dest.mkdir(exist_ok=True)
    failed = []
    with ThreadPoolExecutor(args.workers) as pool_:
        futures = [pool_.submit(fetch, image, dest) for image in chosen]
        for done, future in enumerate(as_completed(futures), 1):
            image, err = future.result()
            if err:
                failed.append((image, err))
            if done % 2000 == 0 or done == len(chosen):
                print(f"{done:,}/{len(chosen):,} done, {len(failed)} failed", flush=True)

    # Photos removed from the mirror since 2018 are expected (DATA_CARD.md caveat 2); they are
    # listed rather than fatal, and the manifest is rewritten without them.
    if failed:
        (args.data / "train_fetch_failed.txt").write_text(
            "".join(f"{i}\t{e}\n" for i, e in failed), encoding="utf-8")
        gone = {i for i, _ in failed}
        kept = [r for r in csv.DictReader(open(manifest, newline="", encoding="utf-8"))
                if r["ImageID"] not in gone]
        with open(manifest, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(kept[0].keys()))
            writer.writeheader()
            writer.writerows(kept)
    print(f"finished: {len(chosen) - len(failed):,} images on disk, {len(failed)} unavailable")


if __name__ == "__main__":
    main()
