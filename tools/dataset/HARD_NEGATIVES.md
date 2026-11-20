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

Any further retrain is judged by `detector_gate.zig` (November 2026): thresholds tuned on one
half of each held-out set, gates read on the other, one PASS/FAIL. Its header holds the command,
the gates, their tolerances and the baselines; attempt 1's model fails its large-face gate.

Everything here is reproducible from the commands in each tool's header; the run folders are
`Downloads\face-data\zig_runs\oi80k_negatives` (attempt 1) and `zig_runs\ft_experiment` (attempt
2, with `run_arm.sh`, `run_arm_hi.sh` and `summarise.py`).

## Diagnostics, 2026-11-18

Why does attempt 1 lose large faces? Inference and data statistics only, no training. Tools:
`large_face_diag.zig` (modes `scale`, `aug`, `audit`, `sheet`; commands in its header),
`oi_to_labelv2 test --large 0.45` and `detector_gate`'s new optional large-face input. Outputs
are in `Downloads\face-data\zig_runs\diagnostics` (`scale_table.sh` there joins the scale runs).
"Attempt 1" is `oi80k_negatives\scrfd80k_neg_640.onnx`; thresholds are each model's gate
threshold (shipped 0.5026, attempt 1 0.4876, RetinaFace 0.9763).

### A bigger large-face test set

1,058 Open Images validation+test photographs outside the gate's held-out set, each with a real
face at least 45% of the width, `Rotation` 0.0, boxed by the same rules (1,118 faces, 1,074 of
them >= 45%; 5 ignore regions). None is in the 80k, the extra large-face positives or the
negatives pool (the tool checks). The gate pools its test half with the held-out set's:
**622 large test faces instead of 94**, shipped's bootstrap SE 1.45 pt, so the large-face
tolerance becomes **2.4 pt** (1.645 x SE) when the file is given.

| Large-face recall, test half | Shipped | Attempt 1 | Public SCRFD-500M | RetinaFace R34 |
|---|---:|---:|---:|---:|
| New set alone (528 faces) | 86.7 | 62.9 | 56.1 | 80.1 |
| Pooled, the gate (622) | **86.5** | 62.1 (-24.4, FAIL) | 55.0 | 80.1 (-6.4, **now FAILS**) |

**About a fifth of these labels are not a detectable real face.** A contact sheet of 90 random
photos (`label_sheet_*.jpg`) shows 19 clear errors: 7 extreme eye close-ups boxed as a whole
face, 7 unflagged depictions (paintings, a statue, a doll, anime), 4 crowds or groups under one
box not marked group-of, one forehead; about 6 more are partial (mouth only, eyes only). No large
face was missing a box. The gate's own held-out set follows the same Open Images rules. Shipped,
trained on Open Images labels, has learned some of these (it finds 72% of faces >= 80% wide,
RetinaFace 46%), so part of RetinaFace's gap is RetinaFace being right. The tables below
therefore also show the **clean subset**: the 860 large faces RetinaFace itself finds.

### 1. Scale test: the faces come back when they are presented smaller

Each large face of the new set, natively and then with the photo shrunk into a black canvas so
the face spans 45/30/20/10% of the canvas width. Recall %, shipped / attempt 1:

| Presented at | All 1,074 | Clean 860 | Clean, native 45-60% (517) | 60-80% (241) | >= 80% (102) |
|---|---:|---:|---:|---:|---:|
| native | 87.1 / **61.2** | 98.6 / **74.8** | 98.5 / 93.4 | 99.2 / **62.2** | 98.0 / **9.8** |
| 45% of width | 86.5 / 82.0 | 98.7 / 98.1 | 98.6 / 98.3 | 99.2 / 98.3 | 98.0 / 97.1 |
| 30% | 83.1 / 81.9 | 98.3 / 97.8 | 98.6 / 98.5 | 99.2 / 97.5 | 94.1 / 95.1 |
| 20% | 82.0 / 80.8 | 97.4 / 96.6 | 98.3 / 97.9 | 97.9 / 96.7 | 92.2 / 90.2 |
| 10% | 73.4 / 69.3 | 88.7 / 85.0 | 91.9 / 90.3 | 88.4 / 83.0 | 73.5 / 62.7 |

A clean face filling 80% of the frame is found 9.8% of the time natively and 97.1% of the time
at 45%. The loss grows with the presented size (-5, -37, -88 points by band) and vanishes once
the face is no wider than ~45% of the input. Attempt 1 still knows a face; it rejects one that
fills the frame. (RetinaFace, all 1,074: 80.1 / 79.7 / 78.1 / 76.5 / 62.1.)

### 4. Head attribution: it is the stride-32 head

The SCRFD exports keep each stride's raw outputs, so per face: the best raw score among each
head's anchors that decode onto it (IoU >= 0.5), and which stride ATSS would train on it.
Every large face but 9 is owned by stride 32 (4-6 positives, essentially none at stride 16).
All 282 faces shipped finds and attempt 1 misses are stride-32 faces:

| Mean best raw score, stride 8 / 16 / 32 | Shipped | Attempt 1 |
|---|---:|---:|
| The 282 faces attempt 1 misses | 0.000 / 0.002 / **0.688** | 0.000 / 0.002 / **0.142** |
| All 1,065 stride-32 large faces | 0.000 / 0.023 / 0.704 | 0.000 / 0.027 / 0.535 |

No other head picks them up. Shrunk to 45% of the width, 97% of them are still stride-32 faces
(at 30%, 44%; by 20%, stride 16 owns all) and attempt 1 finds them, so the stride-32 head fails
at the *top* of its range, not throughout. Of 30 photos where
shipped finds the face and attempt 1 does not (`missed_sheet_1.jpg`), about 20 are plainly real,
frame-filling faces, 6 eye close-ups or depictions, 4 partial: attempt 1 is mostly losing real
faces.

### 2. Augmentation: face-like content gains on real faces fastest at the largest sizes

The trainer's own `RandomSquareCrop` draws (`scrfd_augment.drawPatch`, 3 per photo, seed 0),
no pixels. Width as a fraction of the crop (the 640 input), share of crops in %:

| | < 0.30 | 0.30-0.45 | 0.45-0.60 | 0.60-0.80 | >= 0.80 |
|---|---:|---:|---:|---:|---:|
| Positives: widest face / image width, before cropping | 85.3 | 9.9 | 3.2 | 1.2 | 0.4 |
| Positives: widest kept face / crop | 72.2 | 13.7 | 6.6 | 4.3 | 3.3 |
| Negatives: largest Open Images object / crop | 18.6 | 13.2 | 14.8 | 17.2 | 36.2 |
| Hard negatives: widest SCRFD-80k hit (>= 0.3) / crop | 65.1 | 8.5 | 6.8 | 7.3 | 12.4 |

Per epoch (80,000 positive photos, 20,000 negatives):

| Crops with ... at least this wide | >= 0.45 | >= 0.60 | >= 0.80 |
|---|---:|---:|---:|
| a real face (positives) | 11,294 | 6,023 | 2,613 |
| a face-like SCRFD hit (negatives) | 4,023 | 2,994 | 1,880 |
| face-like : real | 0.36 | 0.50 | **0.72** |
| any Open Images object (negatives; cat, dog, tree, toy lead) | 13,642 | 10,679 | 7,236 |

The crop already enlarges faces (4.9% of photos hold a face >= 45% wide, 14.1% of crops do),
but negatives get an unconstrained crop of mostly close-up subjects: frame-filling content is in
68% of negative crops against 14% of positive ones, and the face-like share climbs with size to
0.72 face-like crops per real one at >= 80% -- the band attempt 1 loses almost entirely. At the
anchor level, the anchors ATSS would give each face-like hit (all trained as background) per
epoch are 7.3k / 24.0k / 29.6k at stride 8 / 16 / 32, against 305k / 492k / 179k positives from
real faces: **2.4% / 4.9% / 16.5%** of the positive signal, seven times the pressure at stride
32 that stride 8 sees. Also measured, and small: 3,189 real faces per epoch at least 30% visible
whose centre falls outside the crop and so train as background (250 of them >= 0.3 of the crop).

### 3. Label audit: unlabelled training faces are too few and too small

RetinaFace >= 0.9 on 2,000 random 80k training photos (6,099 labelled faces): 5,294 confident
detections, **114 on no face box or ignore region** (IoU < 0.3; ignore IoU or IoF > 0.3), in 82
photos. 93% are under 10% of the image width and none is 45% or more. By eye (`audit_1.jpg`,
100 tiles) about 80% are real but tiny, blurred or dark background faces; the rest skulls,
prints, masks. That is ~0.045 unlabelled real faces per photo, ~3,500 over the 80k against
244,683 labelled (1.4%), all small. It cannot teach the model to reject large faces; at most it
nudges small-face confidence.

### Conclusion, ranked

1. **Supported: the stride-32 head learns that frame-filling, face-like content is background.**
   The loss is confined to the stride-32 head (raw score 0.69 -> 0.14 on the missed faces), grows
   with presented size, and disappears when the same faces are presented at <= 45% of the width
   (clean subset 74.8% -> 98.1%). It tracks the training mix: face-like negatives per real face
   rise from 0.36 to 0.72 across the >= 45/60/80% bands, and their background pressure is 16.5%
   of the positive signal at stride 32 against 2.4% at stride 8.
2. **Ruled out: lost face semantics.** The same faces, smaller, are found at the shipped rate.
3. **Ruled out as the cause: unlabelled faces in training backgrounds** (1.4%, all small).
4. **Not needed to explain it: capacity.** The same 0.5 GFLOP architecture (shipped) finds
   98-99% of clean large faces in every size band; nothing here shows the stride-32 head *cannot*
   hold both, only that the data asked it not to. Per journal:SCRFD-2:0006 that does not justify a
   bigger model yet; it stays the fallback if a size-balanced mix still fails.
5. **Measurement caveat:** a fifth of Open Images' >= 45% boxes are not detectable faces, which
   shipped partly learned; read large-face gaps on the clean subset too, and note RetinaFace now
   fails the tightened large-face gate (-6.4).

**For the training design (journal:SCRFD-2:0004):** balance by apparent size, per band, at the
crop the trainer actually produces, not by photo counts. Either (a) constrain negative crops so a
face-like hit rarely exceeds ~45% of the crop, or mark hits that do as ignore regions, and/or
(b) add face-centred zoom crops so real faces >= 60% and >= 80% of the crop clearly outnumber
face-like negatives in every band (today 2,613 against 1,880 at >= 80%). First check whether the
COCO false positives attempt 1 removed are themselves frame-filling; if they are, (a) hands some
of the fix back. Score per size band, on the clean subset as well as the gate. An inference-side
fallback, untested for false positives: a second pass with the photo shrunk to half size recovers
attempt 1's clean large faces (98.1% at 45%) for about twice the CPU time.

## Proxy validation, 2026-11-18

Is there a short from-scratch run whose outcome predicts the ~22 h one, so training designs can
be compared cheaply? Two arms, seed 0, batch 64, identical schedules, `train_scrfd --recipe 80k
--epochs 25 --steps 17,21` (the 100/69/85 recipe at a quarter): **A** attempt 1's data (the 80k
plus its 20,000 negatives), **B** the 80k alone. Fixed before running: the proxy is valid only if
A minus B reproduces the full runs' direction and rough size on the gate (with the large-face
set) -- large-face recall at least 10 points down, COCO recall at 0.01 FP per photo at least 3
up, and the stride-32 raw score on large faces at least 0.08 down. One variation allowed: 50
epochs (`--steps 34,42`), B first, A only if B reached 72 on large faces. Runs, scripts
(`score_arm.sh`, `compare.sh`) and gate reports are in `Downloads\face-data\zig_runs\proxy`.

| Test half | Shipped (MMDetection, 100 ep) | Attempt 1 (native, 100 ep, negatives) | A: 25 ep, negatives | B: 25 ep, no negatives | B: 50 ep, no negatives |
|---|---:|---:|---:|---:|---:|
| OI recall at 0.1 unmatched/image | 78.2 | 78.3 | 74.4 | 72.2 | 76.2 |
| **Large faces, pooled 622 (the gate)** | **86.5** | **62.1** | 61.3 | 58.4 | **61.1** |
| Large-face set, native, 1,074 | 87.1 | 61.2 | 60.7 | 58.5 | 61.0 |
| Stride-32 raw score, stride-32 large faces | 0.704 | 0.535 | 0.515 | 0.516 | 0.536 |
| Faces shipped finds and this misses: n, stride-32 raw | | 282, 0.142 | 284, 0.171 | 311, 0.201 | 283, 0.164 |
| COCO recall at 0.01 / 0.02 / 0.05 / 0.1 FP per photo | 81.9 / 84.2 / 86.9 / 88.9 | 88.6 / 90.1 / 91.9 / 92.5 | 80.3 / 82.3 / 84.6 / 86.9 | 72.4 / 77.0 / 80.1 / 83.5 | 79.3 / 81.3 / 85.9 / 88.0 |
| GPU hours (RTX 4090) | | ~22 | 5.5 | 4.3 | 7.1 (epochs 11-50, resumed from B's epoch 10) |

**The proxy is not valid**, and it found something more important. A minus B at 25 epochs:
large faces +2.9 (needed -10 or less), stride-32 raw score 0.000 (needed -0.08), COCO +7.9 at
0.01 (passes; full runs +6.7). The negatives' false-positive gain shows up at a quarter of the
schedule, the large-face loss does not -- because **the control without negatives loses the
large faces too.** At 50 epochs B has 61.1% on large faces and a stride-32 score of 0.536, which
is attempt 1 to within a point (62.1, 0.535), including the 283 faces shipped finds and B misses,
whose stride-32 score drops 0.685 -> 0.164 just as attempt 1's did. A 50-epoch A arm was not run:
B stayed under 72, so no A could have shown a 10-point gap.

So the full runs' large-face "delta" compares two different things: shipped was trained by
MMDetection, attempt 1 natively, and no native 80k run *without* negatives has been trained to
100 epochs. Two checks narrow it:

* **The native trainer learns large faces on the curve recipe.** curve_1988 (1,988 labels, batch
  8, 640 epochs, no ignore regions) native vs MMDetection: 84.6 vs 84.1 pooled, stride-32 score
  0.491 vs 0.511.
* **Training noise is small next to the gap.** A's epoch 10 against attempt 1's epoch 10 (same
  data, seed and schedule): large faces within 2 points, stride-32 score within 0.003.

Either the native 80k recipe only learns frame-filling faces between epochs 50 and 100 (the
step at 69), or it never does, and something in the 80k path (batch 64, ignore regions at IoF
0.5, which the curve recipe does not use) differs from MMDetection's. Both mean the diagnostics
above attribute to negatives a loss that a negative-free native run shows as well. Until a
native 80k control is trained to 100 epochs, no short run can be validated, and no training
design can be credited or blamed for large faces. That run costs ~11.7 h, resuming B's 50-epoch
run from its epoch 30 (still at lr 1e-2, the full recipe's schedule).

## Where it stopped, 2026-11-20

The 100-epoch control was not needed. Five 80k runs at epoch 30 -- three native (including one
from an MMDetection-generated init) and two by MMDetection's own trainer -- scored 58.4 to 61.4
on large faces; the shipped model's run scored 83.0. Same config, label file and environment.
**The native trainer matches MMDetection, and the shipped model is a lucky draw** (about 1 in 5)
that traded about 5 points of COCO recall for +22 on large faces. TF32 was the last suspect; it
was not settled, and no longer matters for that conclusion. A retrain of the current recipe
will likely land below shipped's 86.5 on large faces, so a new design has to target them.

A design screen at 10 epochs (native, 80k, seed 0; baseline 55.3 large, 77.1 COCO recall at
0.1 FP per photo):

| Arm | Large faces | COCO @ 0.1 | Verdict |
|---|---|---|---|
| Z: `--zoom 0.25`, a quarter of crops zoom on the widest face | 74.8 (+19.5) | 72.5 (-4.6) | large passes, COCO fails |
| L: `--lr 0.02` | 57.6 | 80.7 (+3.6) | large fails |
| S: `--batch 16` | 52.9 | 81.5 | large fails |

Zoom crops are the only lever found for large faces; `train_scrfd --zoom` is in the tree. The
untried next step was pairing it with L to pay back the COCO cost (`--zoom 0.25 --lr 0.02`, and
`--zoom 0.125`), winners to 30 epochs with two seeds.

**Paused.** Face Crop Studio is a cropping app: people point it at faces, so fewer false
positives buy little, and the shipped detector stays.
