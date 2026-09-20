"""Train a small model that takes a face crop and returns its two eye points.

  train_eye_refiner.py <data_dir> [--epochs 300] [--out eye_refiner.pt]

The label-count curve (CURVE_RESULTS.md) found that training a whole detector on 2,482 images
could not beat YuNet's eye points, and that more eye labels did not help: point distance
improved 39% from 400 to 1988 labels while the eye-*line* angle stayed flat at ~5.3 degrees.
Detection was never the problem. This attacks the narrow task instead -- given a face box,
where are the eyes -- which is small enough that 1,988 labelled pairs may actually suffice.

Two choices follow directly from that finding:

* **Rotation augmentation, up to 30 degrees.** Angle error stayed flat because 2,482 images
  contain little roll variety, so more labels on them taught eye appearance rather than eye
  geometry. Rotating crops manufactures the variety the corpus lacks.
* **An auxiliary angle loss.** The metric that matters is the tilt of the line between the
  points, and it was the metric that would not move, so it is optimised directly rather than
  left to follow from point regression.

Trained from scratch rather than fine-tuned from an ImageNet backbone. A pretrained backbone
would likely need fewer labels, but this project has been careful about what it can ship, and
pretrained weights carry their own terms; raising that as a choice beats quietly depending on
it. If from-scratch stalls, that is the first thing to revisit.

Only faces whose eyes were clicked and whose side is known are used: `steep_roll` faces carry
points but no reliable left/right identity, and training on them would teach the ambiguity.
"""

import argparse
import itertools
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


def usable(annotation: dict) -> bool:
    return annotation["num_keypoints"] == 2 and not annotation["fcs"]["steep_roll"]


class Faces(Dataset):
    """Face crops with their two eye points, in crop-relative coordinates.

    `hold_out` carves a slice off the training split for checkpoint selection: pass a fraction
    and `part="fit"` or `part="val"`. The division is by *image*, not by face, because two
    faces from one photo share a camera, a scene and often a subject -- splitting by face would
    leak all of that across the boundary and flatter the very figure the split exists to keep
    honest.
    """

    def __init__(self, data: Path, split: str, size: int, train: bool,
                 hold_out: float = 0.0, part: str = "fit"):
        document = json.loads((data / f"face_keypoints_{split}.json").read_text(encoding="utf-8"))
        images = {image["id"]: image for image in document["images"]}

        held = set()
        if hold_out > 0:
            names = sorted({image["file_name"] for image in images.values()})
            random.Random(1234).shuffle(names)
            held = set(names[: int(len(names) * hold_out)])

        self.items = []
        for annotation in document["annotations"]:
            if not usable(annotation):
                continue
            name = images[annotation["image_id"]]["file_name"]
            if hold_out > 0 and ((part == "val") != (name in held)):
                continue
            self.items.append((name, annotation["bbox"], annotation["keypoints"]))
        self.dir = data / "images"
        self.size = size
        self.train = train

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        name, (x, y, w, h), keypoints = self.items[index]
        image = cv2.imread(str(self.dir / name))
        if image is None:  # a missing file must not silently become a black face
            raise FileNotFoundError(self.dir / name)
        eyes = np.array([[keypoints[0], keypoints[1]], [keypoints[3], keypoints[4]]],
                        dtype=np.float32)

        centre = np.array([x + w / 2, y + h / 2], dtype=np.float32)
        side = max(w, h)
        if self.train:
            side *= random.uniform(1.10, 1.55)
            centre += np.array([random.uniform(-0.06, 0.06) * w,
                                random.uniform(-0.06, 0.06) * h], dtype=np.float32)
            angle = random.uniform(-30, 30)
        else:
            side *= 1.25
            angle = 0.0

        # Rotate about the crop centre and scale to the network's input in one affine, so the
        # eye points can be carried through with the same matrix rather than re-derived.
        matrix = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])), angle,
                                         self.size / side)
        matrix[0, 2] += self.size / 2 - centre[0]
        matrix[1, 2] += self.size / 2 - centre[1]
        crop = cv2.warpAffine(image, matrix, (self.size, self.size), flags=cv2.INTER_LINEAR)
        points = (eyes @ matrix[:, :2].T) + matrix[:, 2]

        if self.train:
            if random.random() < 0.5:
                crop = cv2.flip(crop, 1)
                points[:, 0] = self.size - points[:, 0]
                points = points[::-1].copy()  # mirroring swaps which eye is which
            if random.random() < 0.5:
                crop = np.clip(crop.astype(np.float32) * random.uniform(0.7, 1.3)
                               + random.uniform(-20, 20), 0, 255).astype(np.uint8)

        tensor = torch.from_numpy(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
        tensor = (tensor.float() - 127.5) / 128.0
        return tensor, torch.from_numpy((points / self.size).astype(np.float32)).reshape(4)


class Refiner(nn.Module):
    """Small from-scratch convnet: 112x112 crop in, four numbers out."""

    def __init__(self, width: int = 32):
        super().__init__()
        channels = [3, width, width * 2, width * 4, width * 4, width * 8]
        blocks = []
        for inp, out in itertools.pairwise(channels):
            blocks += [nn.Conv2d(inp, out, 3, stride=2, padding=1, bias=False),
                       nn.BatchNorm2d(out), nn.ReLU(inplace=True),
                       nn.Conv2d(out, out, 3, padding=1, bias=False),
                       nn.BatchNorm2d(out), nn.ReLU(inplace=True)]
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(channels[-1], 128), nn.ReLU(inplace=True),
                                  nn.Linear(128, 4))

    def forward(self, x):
        return self.head(self.features(x))


def angle_of(points):
    """Tilt of the eye line, in radians, for a batch of (N, 4) predictions."""
    return torch.atan2(points[:, 3] - points[:, 1], points[:, 2] - points[:, 0])


def evaluate(model, loader, device) -> dict:
    model.eval()
    angles, distances = [], []
    with torch.no_grad():
        for crops, targets in loader:
            crops, targets = crops.to(device), targets.to(device)
            predicted = model(crops)
            difference = torch.rad2deg(angle_of(predicted) - angle_of(targets)).abs() % 360
            angles += torch.minimum(difference, 360 - difference).cpu().tolist()
            # Crop-relative distance; the crop is 1.25x the longer box side, so scale back.
            for slot in (0, 1):
                error = (predicted[:, slot * 2:slot * 2 + 2]
                         - targets[:, slot * 2:slot * 2 + 2]).norm(dim=1)
                distances += (error * 1.25).cpu().tolist()
    angles.sort()
    distances.sort()
    return {
        "angle_median": angles[len(angles) // 2],
        "within_5deg": sum(1 for a in angles if a <= 5) / len(angles),
        "distance_median": distances[len(distances) // 2],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_dir")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--angle-weight", type=float, default=0.5)
    ap.add_argument("--val-fraction", type=float, default=0.15,
                    help="share of training IMAGES held out for checkpoint selection")
    ap.add_argument("--out", default="eye_refiner.pt")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    data = Path(args.data_dir).resolve()
    # Selection happens on a slice of the training split. The test set is read once, at the
    # end: picking the best of thirty evaluations against it would report the luckiest epoch
    # rather than the model, and that figure is the one quoted against YuNet.
    train = Faces(data, "train", args.size, train=True, hold_out=args.val_fraction, part="fit")
    val = Faces(data, "train", args.size, train=False, hold_out=args.val_fraction, part="val")
    test = Faces(data, "test", args.size, train=False)
    print(f"{len(train)} fitting faces, {len(val)} validation faces (held out of training), "
          f"{len(test)} test faces", flush=True)

    train_loader = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=6,
                              drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val, batch_size=args.batch, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test, batch_size=args.batch, num_workers=6,
                             persistent_workers=True)

    device = torch.device(args.device)
    model = Refiner().to(device)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters", flush=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=3e-3, total_steps=args.epochs * len(train_loader))

    best = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for crops, targets in train_loader:
            crops, targets = crops.to(device), targets.to(device)
            predicted = model(crops)
            points_loss = F.smooth_l1_loss(predicted, targets, beta=0.02)
            # The angle term exists because angle was the metric that would not improve.
            angle_loss = (1 - torch.cos(angle_of(predicted) - angle_of(targets))).mean()
            loss = points_loss + args.angle_weight * angle_loss
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            schedule.step()
            total += loss.item()

        if epoch % 10 == 0 or epoch == args.epochs:
            stats = evaluate(model, val_loader, device)
            flag = ""
            if best is None or stats["angle_median"] < best["angle_median"]:
                best = dict(stats, epoch=epoch)
                torch.save({"model": model.state_dict(), "size": args.size,
                            "val_stats": stats, "epoch": epoch}, args.out)
                flag = "  <- saved"
            print(f"epoch {epoch:>4}  loss {total / len(train_loader):.4f}  "
                  f"val angle {stats['angle_median']:.2f} deg  "
                  f"within5 {stats['within_5deg']:.1%}  "
                  f"dist {stats['distance_median']:.4f} of box{flag}", flush=True)

    if best is None:  # only reachable with --epochs 0, but it should say so rather than crash
        print("no evaluation ran, so there is nothing to report", flush=True)
        return
    print(f"\nselected on validation: {best['angle_median']:.2f} deg median at epoch "
          f"{best['epoch']}, {best['within_5deg']:.1%} within 5 deg", flush=True)

    # The one and only look at the test set, using the checkpoint validation chose.
    model.load_state_dict(torch.load(args.out, map_location=device)["model"])
    held = evaluate(model, test_loader, device)
    state = torch.load(args.out, map_location="cpu")
    state["test_stats"] = held
    torch.save(state, args.out)
    print(f"test set, selected checkpoint: {held['angle_median']:.2f} deg median, "
          f"{held['within_5deg']:.1%} within 5 deg, "
          f"{held['distance_median']:.4f} of box width", flush=True)
    print("YuNet on the same test faces: 4.05 deg, 57.6% within 5 deg, 0.043 of box width",
          flush=True)


if __name__ == "__main__":
    main()
