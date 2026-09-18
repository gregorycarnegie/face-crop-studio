# How many eye labels does the landmark head need?

The question behind two labelling sessions: 3,605 faces were clicked, and nobody knew how many
were actually required. This records the measurement rather than the estimate. Four SCRFD-500M
models are trained on 400, 800, 1600 and 1988 labelled eye pairs, every run seeing every image
and every box so that only the landmark supervision varies (see `subsample_labels.py`), then
scored on the held-out test set of 1,562 faces whose eyes were clicked and whose side is known.

**Status: complete.** All four trained 640 epochs on an RTX 4090, 17-18 September 2026, about
six and a half hours each. The answer is in "What the curve says" at the end.

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

## The curve

All four models at the fixed 0.20 threshold, against the shipping detector:

| Eye labels | Recall | FP/image | Angle median | Within 5 deg | Eye distance |
|-----------:|--------|----------|--------------|--------------|--------------|
| 400 | 86.1% | 0.98 | 5.72 deg | 45.5% | 0.111w |
| 800 | 88.3% | 1.03 | 5.50 deg | 46.8% | 0.079w |
| 1600 | 89.2% | 1.22 | 5.28 deg | 48.5% | 0.083w |
| 1988 | 89.9% | 1.31 | 5.31 deg | 47.9% | 0.068w |
| **YuNet 2023** | 88.5% | -- | **4.05 deg** | **57.6%** | **0.043w** |

## What the curve says

**More eye labels are not the constraint.** Five times the labels, 400 to 1988, cut eye-point
*distance* error by 39% (0.111w to 0.068w) but left *angle* error flat: 5.72 to 5.31 degrees,
with 1988 fractionally worse than 1600, which is inside the noise. Faces within 5 degrees
stayed at roughly 48% throughout.

That split is the finding. The points land closer to the truth, but the vector between them
does not improve -- and the vector is what `fcs-core::face_cropper` consumes to level a crop.
More labels on the same 2,482 images teach the model what an eye looks like; they cannot teach
it the range of head poses and rolls needed to get the eye *line* right.

**The constraint is images.** Every run saw the same 2,482, while upstream's schedule was tuned
for WIDER FACE's 12,880. Open Images holds 571,247 more labelled faces across 278,655 images
whose boxes are already drawn by hand (DATA_CARD.md section 2), which costs downloading rather
than clicking.

**So the honest recommendation is to stop labelling eyes and add images.** A run over tens of
thousands of images, with the 3,550 eye pairs already clicked supervising the landmark head on
the fraction that has them, is the experiment worth doing next. If angle error still sits near
5 degrees after that, the architecture or the schedule is the problem and no amount of data
will fix it.

**YuNet stays in front for now**, at 4.05 degrees against 5.31, and 57.6% of faces within 5
degrees against 47.9%. It was trained on roughly 85,000 landmark-annotated faces over far more
images, so this is a fair reflection of a 40-to-1 difference in supervision rather than a
verdict on SCRFD.

One number not to over-read: recall rises across the curve (86.1% to 89.9%) while false
positives rise with it (0.98 to 1.31 per image). Both moves are small, and the box head saw
identical supervision in every run -- every image and every box -- so the difference is
training noise, not an effect of eye labels.
