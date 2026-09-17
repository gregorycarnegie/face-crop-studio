# How many eye labels does the landmark head need?

The question behind two labelling sessions: 3,605 faces were clicked, and nobody knew how many
were actually required. This records the measurement rather than the estimate. Four SCRFD-500M
models are trained on 400, 800, 1600 and 1988 labelled eye pairs, every run seeing every image
and every box so that only the landmark supervision varies (see `subsample_labels.py`), then
scored on the held-out test set of 1,562 faces whose eyes were clicked and whose side is known.

**Status: one of four models trained.** The 400 point and the baseline are below; 800, 1600 and
1988 follow at roughly seven hours each.

## The baseline to beat

YuNet 2023, the detector Face Crop Studio ships today, on the identical 1,562 faces with
identical matching, at its production 0.8 threshold:

| | YuNet 2023 |
|---|---|
| Detected at IoU>=0.5 | 88.5% |
| Eye-line angle error, median | **4.05 deg** |
| Within 2 / 5 / 10 deg | 28.7% / 57.6% / 80.7% |
| Eye distance, median | **0.043 of box width** |

Angle error is the number that matters: `fcs-core::face_cropper` uses the two eye points only
to level the crop, so degrees of eye-line error translate directly into a tilted output. Eye
distance is reported as a fraction of box width rather than of interocular distance, because
the usual 5-point NME divides by eye separation and collapses on the steeply rolled faces this
corpus contains.

For scale: YuNet was trained on WIDER FACE with roughly 85,000 landmark-annotated faces, some
40 times the supervision of our largest run, over far more images. A gap here is expected and
is the point of the exercise.

## 400 labels, fully trained (640 epochs)

Swept across score thresholds, from one forward pass per image:

| Threshold | Recall | FP/image | Angle median | Within 5 deg | Eye distance |
|-----------|--------|----------|--------------|--------------|--------------|
| 0.02 | 96.7% | 39.40 | 5.92 deg | 44.6% | 0.116w |
| 0.10 | 92.4% | 1.83 | 5.79 deg | 45.3% | 0.113w |
| **0.20** | **86.1%** | **0.98** | **5.72 deg** | **45.5%** | **0.111w** |
| 0.30 | 77.9% | 0.66 | 5.61 deg | 45.4% | 0.110w |
| 0.40 | 64.5% | 0.37 | 5.35 deg | 47.4% | 0.108w |
| 0.50 | 49.5% | 0.17 | 4.93 deg | 50.1% | 0.107w |
| 0.60 | 29.8% | 0.03 | 4.73 deg | 52.4% | 0.104w |
| 0.70 | 5.4% | 0.00 | 4.10 deg | 56.5% | 0.112w |

**Scored at 0.20 for the curve.** That is roughly one spurious box per image, and a recall
(86.1%) close enough to YuNet's 88.5% that the eye-point comparison is not confounded by
detecting a different population of faces. Every model in the curve is scored there; comparing
each at its own best threshold would measure calibration rather than labelling.

**A trap worth recording.** The first run of this evaluation used 0.02, because that is what
SCRFD's own evaluation uses, and reported 96.7% recall. At that threshold the model emits 39.4
boxes per image matching no annotation at all, so the figure measured nothing. Recall without a
false-positive count beside it is not a result.

## Where 400 labels leaves us

| At threshold 0.20 | SCRFD, 400 labels | YuNet 2023 |
|---|---|---|
| Recall | 86.1% | 88.5% |
| Angle error, median | 5.72 deg | **4.05 deg** |
| Within 5 deg | 45.5% | **57.6%** |
| Eye distance | 0.111w | **0.043w** |

Worse than the shipping detector on every axis, and by a wide margin on eye-point accuracy --
about 2.6 times the distance error. That is the expected shape of a first curve point, not a
verdict on the approach: it establishes that 400 eye labels is not enough and gives the three
remaining runs something concrete to close.

The interesting possibility is that they will not close it. Every run trains on the same 2,482
images, while upstream's schedule was tuned for WIDER FACE's 12,880. If 1988 labels land near
400, the binding constraint is images rather than eye points on those images -- which would be
a more useful answer than any number, and cheaper to act on than another labelling session.
