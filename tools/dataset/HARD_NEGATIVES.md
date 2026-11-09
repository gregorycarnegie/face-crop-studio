# Face-free hard negatives for SCRFD-80k: two attempts, neither shipped

November 2026. The question: SCRFD-80k fires 5-6x as often as RetinaFace on COCO photographs
with nobody in them -- animal faces, clocks, plates, wheels (`docs/MODEL_COMPARISON.md`). Its
80,000 training photographs all contain faces. Does training on face-free photographs fix that
without losing faces?

**Short answer: it fixes the false positives and costs faces, and the cost does not stay put.**
The shipped model is unchanged.

## The negatives

`oi_negatives.zig` picks them. Lacking a person label is not enough: in a random 1,000 Open
Images train photographs with no person box or label, about half of SCRFD-80k's hits at its
shipped threshold were real people nobody had labelled. Only photographs where a human verified
Person, Human face or Human head **absent** are used (51,955 eligible). SCRFD-80k mined them,
RetinaFace at 0.9 vetoed 2,658 likely real faces, and a person reviewed the 400 highest-scoring
hits (8 real-face close-ups rejected, `negatives_rejected.txt`). Dolls and figures stay in as
background.

## Attempt 1: full retrain, 20,000 negatives

100 epochs, the 80k recipe plus `train_scrfd --negatives`, 15,224 hard (SCRFD-80k >= 0.3) and 4,776
easy. Scored by `detector_bench.zig`, everything at the shipped model's thresholds:

| | Shipped | Retrained |
|---|---:|---:|
| COCO false positives per no-person photo | 0.030 | **0.005** |
| COCO recall at 0.01 false positives per photo | 78.8% | **87.4%** |
| Open Images recall at 0.1 unmatched per image | 76.8% | 75.8% |
| **Open Images recall, faces >= 45% of the image width** | **78.9%** | **57.3%** |

Every overall figure stayed within a point, which is why `detector_bench oi` now prints recall
by face size: a quarter of the large faces were gone, and five of the six bundled sample
portraits dropped below 0.5. 6,672 of the hard negatives (44%) had their widest hit spanning
30% of the image or more -- frame-filling dog, cat and plush-toy close-ups -- against 3,876
training photographs holding a real face that large.

## Attempt 2: balanced negatives and more large faces, short controlled experiment

The design, fixed before any result:

* **Negatives (20,000):** 15% of the frame-filling hard negatives (1,000, drawn at random), all
  8,552 hard negatives with smaller hits, 10,448 easy (`oi_negatives pick --frame-keep 0.15`).
* **Positives:** the 80k plus 7,514 unused Open Images *train* photographs with a real face at
  least 45% of the width (`oi_train_fetch --large 0.45`, labelled by `oi_to_labelv2 --manifest`
  with the same rules as the 80k). No overlap with the 80k, the negatives or the test images.
* **Gates against the shipped model, same thresholds:** no Open Images face-size band more than
  2 points down; Open Images recall at the 0.1 budget within 1 point; COCO false positives per
  no-person photo at most 0.015; COCO recall (eyes >= 8 px apart) within 1 point and its largest
  band within 2; false positives outside people no higher than shipped + 0.005; identical graph.
  The six bundled samples were not used for any choice.

Each arm fine-tunes the shipped checkpoint (`train_scrfd --init`) on one data mix, seed 0:

| | Shipped | Full attempt 1 | A0: no negatives | A1: attempt-1 negatives | A3: balanced design | A1-hi | A3-hi |
|---|---:|---:|---:|---:|---:|---:|---:|
| Schedule | | 100 ep from scratch | 3 ep, lr 1e-3 | 3 ep, lr 1e-3 | 3 ep, lr 1e-3 | 6 ep, lr 1e-2 | 6 ep, lr 1e-2 |
| OI faces < 10% width | 65.7 | 66.4 | 66.2 | 65.1 | 64.6 | 65.0 | 64.6 |
| OI 10-20% | 87.4 | 87.4 | 87.7 | 86.5 | 86.3 | 85.2 | 85.5 |
| OI 20-30% | 88.3 | 88.1 | 88.5 | 87.5 | 87.9 | 87.9 | 87.7 |
| OI 30-45% | 89.1 | 88.4 | 89.4 | 88.7 | 89.1 | 88.1 | 89.4 |
| **OI >= 45%** | **78.9** | **57.3** | 79.8 | 76.6 | 77.1 | 75.7 | 76.6 |
| OI recall at the 0.1 budget | 76.8 | 75.8 | 76.8 | 76.1 | 76.9 | 76.4 | 76.4 |
| OI >= 45%, mean score | 0.649 | 0.516 | 0.647 | 0.640 | 0.650 | 0.623 | 0.640 |
| COCO recall, eyes >= 8 px | 84.8 | 84.9 | 85.0 | 83.5 | 83.0 | 82.8 | 83.0 |
| COCO eyes >= 32 px | 97.0 | 96.2 | 97.0 | 96.6 | 96.6 | 96.4 | 96.0 |
| **COCO false positives / no-person photo** | **0.030** | 0.005 | 0.031 | 0.005 | 0.007 | 0.004 | 0.004 |
| COCO false positives outside people | 0.017 | 0.014 | 0.018 | 0.015 | 0.014 | 0.013 | 0.014 |
| COCO recall at 0.01 FP / photo | 78.8 | 87.4 | 80.0 | 85.8 | 85.2 | 85.7 | 85.0 |

(An A2 arm, balanced negatives without the extra positives, read the same as A3 in the 3-epoch
format: large faces 77.1, false positives 0.007.)

What it shows:

1. **Short fine-tunes cannot validate large-face recall.** A1 and A1-hi use attempt 1's
   negatives, and reproduce its false-positive fix but only 2.3 and 3.2 points of its 21.6-point
   large-face loss. A mix can pass in this format and still fail at full length. Fine-tuning a
   converged model barely moves what it learned about large faces; training from scratch does.
2. **The balanced design keeps the false-positive fix** (0.004-0.007 against 0.030).
3. **It still fails two gates, even here.** A3-hi loses 2.3 points on faces >= 45% of the width
   and 1.8 points of COCO recall at the shipped threshold, mostly small faces. Every arm with
   negatives loses 1.3-1.8 points there while gaining 5-7 at equal false-positive rates: the
   negatives lower confidence broadly rather than separating faces from face-like objects.
4. **The extra large faces did not move large-face recall** in this format (A2 and A3 alike), so
   the data balance alone is not shown to be the lever.

## What it suggests

The detector is not learning *what makes a face human*; it is learning which inputs tend to be
background, and large, frame-filling, face-shaped inputs are among them. Rebalancing the
negatives moved the error between face sizes and between the operating point and recall rather
than removing it. Candidate causes worth isolating before any more data work:

* **Scale handling.** Faces >= 45% of a 640 input are 290 px and up, all on the stride-32 head
  with the coarsest features of a 0.5 GFLOP backbone. A cheap test: score attempt 1's model with
  large faces presented smaller (the image padded into a larger canvas). If recall returns, the
  failure is the large-scale head, not face semantics.
* **Augmentation.** `RandomSquareCrop` (0.3-2.0 of the short side) only enlarges a face when the
  crop is small and contains it; face-free images get an unconstrained crop, so frame-filling
  textures appear far more often in negatives than large faces do in positives.
* **Capacity and receptive field** at the largest scale, versus a bigger SCRFD variant or a
  wider stride-32 head.

Everything here is reproducible from the commands in each tool's header; the run folders are
`Downloads\face-data\zig_runs\oi80k_negatives` (attempt 1) and `zig_runs\ft_experiment` (attempt
2, with `run_arm.sh`, `run_arm_hi.sh` and `summarise.py`).
