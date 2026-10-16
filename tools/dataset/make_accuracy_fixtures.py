"""Build `fixtures/images/`: a committed, licence-clean sample of the held-out benchmark set.

The previous `fixtures/images/` was a local folder of photographs nobody here had a licence for,
so it could not be committed and CI never saw it. This replaces it with Open Images validation and
test photographs, every one CC BY 2.0, drawn from the same held-out set `docs/MODEL_COMPARISON.md`
scores (`labelv2_test_all.txt`: 1,859 images that SCRFD-80k never trained on).

Selection is a rule, not a choice: images ordered by the SHA-256 of their Open Images ID, the eight
already in `fixtures/oracle/` skipped, the first `--count` taken. That is a uniform sample, so the
recall measured on it should sit near the full set's, and anyone can rebuild the same set.

Each image is downscaled so its longer side is `--side` pixels and re-encoded, which keeps the set
small enough to commit. The detector letterboxes to 640 regardless, so it sees nearly the same
input. Boxes are scaled to match. Re-encoding also drops the EXIF block; the pixels are kept in
their stored orientation, which is what the boxes describe, and an image whose stored size does
not match its label is skipped rather than guessed at.

Run in the SCRFD venv (it has OpenCV), from the repository root:

    /home/grego/work/scrfd-venv/bin/python tools/dataset/make_accuracy_fixtures.py \
        /mnt/c/Users/grego/Downloads/face-data/openimages fixtures/images
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def labels(path: Path) -> dict[str, dict]:
    """labelv2 file -> {name: {size, faces, ignore}}. Five-value lines are ignore regions."""
    out: dict[str, dict] = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            _, name, width, height = line.split()
            current = out[name] = {"size": (int(width), int(height)), "faces": [], "ignore": []}
        elif line.strip():
            values = [float(v) for v in line.split()]
            (current["ignore"] if len(values) == 5 else current["faces"]).append(values[:4])
    return out


def attribution(data: Path) -> dict[str, dict]:
    rows = {}
    for split in ("validation", "test"):
        with open(data / f"{split}-images-with-rotation.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows[row["ImageID"]] = row
    return rows


def test_pattern(path: Path) -> None:
    """A 4x4 synthetic gradient that contains no face, made here so it needs no licence."""
    y, x = np.mgrid[0:4, 0:4]
    image = np.stack([x * 60, y * 60, 255 - x * 30], axis=-1).astype(np.uint8)
    cv2.imwrite(str(path), image)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", type=Path, help="openimages directory: images/, labelv2_test_all.txt, *-with-rotation.csv")
    ap.add_argument("out", type=Path)
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--side", type=int, default=640)
    ap.add_argument("--quality", type=int, default=85)
    args = ap.parse_args()

    truth = labels(args.data / "labelv2_test_all.txt")
    credits = attribution(args.data)
    oracle = {p.name for p in Path("fixtures/oracle/images").glob("*.jpg")}
    order = sorted((n for n in truth if n not in oracle), key=lambda n: hashlib.sha256(n.encode()).hexdigest())

    args.out.mkdir(parents=True, exist_ok=True)
    chosen, skipped = {}, []
    for name in order:
        if len(chosen) == args.count:
            break
        entry = truth[name]
        image = cv2.imread(str(args.data / "images" / name), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None or (image.shape[1], image.shape[0]) != entry["size"]:
            skipped.append(name)
            continue
        height, width = image.shape[:2]
        scale = min(1.0, args.side / max(width, height))
        if scale < 1.0:
            image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(args.out / name), image, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        scaled = lambda boxes: [[round(v * scale, 2) for v in box] for box in boxes]
        chosen[name] = {"faces": scaled(entry["faces"]), "ignore": scaled(entry["ignore"])}

    test_pattern(args.out / "test_pattern.png")

    faces = sum(len(e["faces"]) for e in chosen.values())
    (args.out / "ground_truth.json").write_text(json.dumps({
        "source": "labelv2_test_all.txt (Open Images validation+test, held out from SCRFD-80k training)",
        "rule": f"first {args.count} by SHA-256 of the file name, excluding fixtures/oracle; "
                f"longer side scaled to {args.side} px, boxes scaled to match",
        "images": chosen,
    }, indent=1) + "\n", encoding="utf-8")

    lines = [
        "# Fixture images: attribution",
        "",
        f"{len(chosen)} Open Images photographs ({faces} annotated faces), all licensed",
        "[CC BY 2.0](https://creativecommons.org/licenses/by/2.0/). Each was downscaled so its longer",
        f"side is {args.side} px and re-encoded as JPEG (quality {args.quality}), which also removed its",
        "metadata. `ground_truth.json` holds the Open Images face boxes, scaled to match.",
        "`test_pattern.png` is a synthetic gradient made by `tools/dataset/make_accuracy_fixtures.py`.",
        "",
        "| File | Title | Author | Source |",
        "|---|---|---|---|",
    ]
    for name in chosen:
        row = credits[name.removesuffix(".jpg")]
        assert row["License"] == "https://creativecommons.org/licenses/by/2.0/", name
        title = (row["Title"] or "untitled").replace("|", "\\|")
        author = (row["Author"] or "unknown").replace("|", "\\|")
        lines.append(f"| `{name}` | {title} | [{author}]({row['AuthorProfileURL']}) | [Flickr]({row['OriginalLandingURL']}) |")
    (args.out / "ATTRIBUTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    size = sum(p.stat().st_size for p in args.out.glob("*.jpg"))
    print(f"{len(chosen)} images, {faces} faces, {size / 1e6:.1f} MB; skipped {len(skipped)}: {skipped}")


if __name__ == "__main__":
    main()
