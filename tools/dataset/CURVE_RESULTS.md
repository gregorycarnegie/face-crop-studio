# How many eye labels does the landmark head need?

The question behind two labelling sessions: 3,605 faces were clicked, and nobody knew how many
were actually required. This records the measurement rather than the estimate. Four SCRFD-500M
models are trained on 400, 800, 1600 and 1988 labelled eye pairs, every run seeing every image
and every box so that only the landmark supervision varies (see `subsample_labels.py`), then
scored on the held-out test set of 1,562 faces whose eyes were clicked and whose side is known.

**Status: complete.** All four trained 640 epochs on an RTX 4090, 17-18 September 2026, about
six and a half hours each. The answer is in "What the curve says" at the end.

> **Correction, 19 September 2026: the boxes were not all there.** "Every run seeing every
> image and every box" was true of the label *file* and false of the photographs. `to_labelv2.py`
> builds from the eye-labelling JSON, which holds one entry per clicked face, so
> `labelv2_train.txt` boxed 2,500 faces in images where Open Images drew 12,070, and
> `labelv2_test.txt` boxed 2,000 where it drew 4,597.
>
> What that does and does not disturb:
>
> * **The curve's relative finding stands.** All four runs shared the same missing boxes, so
>   "more eye labels did not improve the eye line" is still a fair comparison between them.
> * **The absolute gap to YuNet is suspect.** Every run was taught that ~9,300 real faces were
>   background. That alone could explain much of the recall and angle deficit, so "the
>   constraint is images" was never actually isolated from "a fifth of the faces were labelled".
> * **Every FP/image figure below is inflated.** They count detections that match no box in
>   `labelv2_test.txt`, and 2,597 real faces had no box there.
> * **Recall and angle figures are unaffected** -- both are scored only on the clicked faces,
>   which were boxed correctly.
>
> `oi_to_labelv2.py` now writes every Open Images box (`labelv2_{train,test}_all.txt`), with
> group-of boxes, depictions and unboxed YuNet detections as ignore regions, and the 80,000-image
> run (`scrfd_fcs80k_500m.py`) trains and is scored on those.
>
> That run is done, and it settles the question this document could not: 86.0% of all faces at
> 0.11 FP/image against YuNet's 71.6% at 0.14. See `SCRFD_80K.md`.

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

## Postscript: the labels were enough, for a narrower problem

The curve showed that 3,550 eye labels over 2,482 images cannot train a detector to beat YuNet.
It does not follow that they cannot solve the task those labels were collected for. Detection
was never the weakness -- the eye *line* was -- so `train_eye_refiner.py` attacks that alone:
given a face box, where are the two eyes. No detection, no anchors, 1.5 M parameters, a few
minutes to train on the same 1,988 pairs.

Paired on the 1,382 test faces YuNet detects, with the refiner reading YuNet's own boxes,
which is what would actually ship:

| | Refiner | YuNet 2023 |
|---|---|---|
| Angle error, median | **1.18 deg** | 4.05 deg |
| Within 2 deg | **70.8%** | 28.7% |
| Within 5 deg | **93.3%** | 57.6% |
| Eye distance, median | **0.018w** | 0.043w |

Closer on 80.2% of faces individually, not merely in aggregate.

Those figures come from a checkpoint chosen on a validation slice held out of the *training*
split, with the test set read exactly once at the end. On ground-truth boxes the same
checkpoint gives 1.39 deg; YuNet's own boxes score better than the hand-drawn ones because
Open Images boxes vary in how much forehead and chin they include, and the same 0.2 deg gap
appeared in both training runs.

Two design choices came straight from what the curve exposed, and both look load-bearing.
**Rotation augmentation of up to 30 degrees** manufactures the roll variety 2,482 images do not
contain, which is what the angle metric was starving for. **An explicit angle term in the loss**
optimises the tilt of the line directly, rather than hoping it follows from regressing two
points -- the curve's central finding was that point accuracy improved while angle did not.

### Does it hold outside Open Images?

Every figure above comes from Open Images photographs. The application's real workload is
different, so the refiner was run over the 1,239-image reference corpus in `Downloads\VinaSkyy`
(1,129 faces, no eye labels) and compared against YuNet there.

They disagree far more than they did on the labelled set: **13.09 deg median**, with only 24.9%
agreeing within 5 deg and a smooth spread out to 90 deg. Two explanations were ruled out
before reaching for a third. Box geometry is near-identical between the corpora (aspect 0.80
against 0.75, box width as a share of image 0.163 against 0.127), so the refiner is not being
handed oddly-framed crops. And disagreement does not shrink on large faces -- 13.2, 14.8 and
12.3 deg across the 80-150px, 150-300px and 300px+ bands -- so it is not a hard-cases effect
either. On a well-resolved portrait two competent models should converge; these do not, so one
of them is substantially wrong on this distribution.

Which one can be settled without labels. Rotate a face by a known angle: a model that truly
locates eyes must report an eye line rotated by exactly that much, and one that is guessing
will not track it.

| Rotation tracking error, same faces | Median | 75th | 90th |
|---|---|---|---|
| **Refiner** | **1.20 deg** | 2.23 deg | 4.61 deg |
| YuNet 2023 | 4.71 deg | 7.31 deg | 10.93 deg |

The refiner tracks identically on Open Images (1.19 deg median), so this is not consistency on
one distribution only -- it locates eyes just as well on photographs it was never trained on.
YuNet is roughly 3.9 times worse, a ratio that mirrors the labelled comparison (4.05 against
1.18 deg) closely enough that two independent methods, one with ground truth and one without,
agree on direction and magnitude.

A second thing fell out of the same test: YuNet detected only 565 of 600 rotated crops at its
default 0.8 threshold, losing 21 of 120 faces at some rotation, on 320px faces it had already
found in the original photographs.

### What this does not establish

* Selection bias was measured rather than argued away. The first run chose its checkpoint by
  the best of thirty evaluations against the test set -- 1.44 deg on ground-truth boxes.
  Retraining with 15% of the training *images* held out for selection, and the test set read
  once at the end, gives 1.39 deg. So the bias was worth roughly 0.05 deg, and the result
  survives losing that training data as well. The split is by image rather than by face,
  because two faces from one photo share a camera and a scene.
* Recall is untouched. The refiner reads boxes; it cannot find a face YuNet missed, and YuNet
  still misses 11.5% of these faces.
* `steep_roll` faces and those whose eyes were never clicked are excluded throughout, so the
  refiner is unmeasured on the hardest poses in the corpus.
* Trained from scratch, no pretrained backbone, so nothing here depends on weights with
  awkward terms.
* Rotation self-consistency is necessary but not sufficient. A model could track rotation
  perfectly while sitting on a systematic offset -- predicting an eye line that turns correctly
  but starts from the wrong place. The test shows the refiner responds correctly to rotation on
  a corpus it never saw; it cannot show that its absolute placement is right there. Only labels
  on those images, or a human looking at aligned crops, can settle that.
  `refiner_contact_sheet.py` builds the latter: aligned crops from each model side by side,
  ordered by disagreement so the worst cases come first rather than being buried.

  **Settled by inspection, 18 September 2026.** On the corrected sheet -- the 40 faces of 1,096
  where the two models disagree most, so the hardest cases by construction -- the refiner's
  column is upright and YuNet's is visibly rolled, on every pair. That is the check rotation
  tracking could not perform, and it closes the last open question: absolute placement holds on
  photographs the model never trained on.

### A trap in the contact sheet itself

The first sheet rendered every crop at *twice* its measured tilt, so both columns looked
wrong and neither model could be judged. The cause was reasoning by analogy across two
libraries with opposite conventions: `imageproc::rotate_about_center` is documented as
rotating **clockwise** for positive theta, so `fcs-core::face_cropper` correctly passes
`-angle`; `cv2.getRotationMatrix2D` is **counter-clockwise**-positive, so the Python must
pass `+angle`. Copying the Rust negation turned a tilt of theta into 2*theta.

It is worth recording because of how it failed. A doubled tilt still produces a
plausibly-rotated face -- not a blank image, not an exception, just a portrait at the wrong
angle -- and every *number* in this document was unaffected, because the evaluation scripts
compare eye-line angles arithmetically and never render a crop. Only the one artefact meant
for human judgement was wrong, which is precisely the artefact with no automated check
behind it. `refiner_contact_sheet.py` now runs `_self_check` before it renders anything: a
synthetic eye line at a known tilt must come out level.

### And a second one, in the export

Wiring the refiner into the application turned up a worse version of the same class of bug.
The ONNX file in `models/` had been exported by hand in a shell, and the command named
`eye_refiner.pt` -- the checkpoint chosen by the best of thirty evaluations against the *test*
set -- rather than `eye_refiner_val.pt`, the retrained one selected on held-out validation that
every figure above comes from. The selection bias this document describes measuring and
removing was still sitting in the file about to ship.

Nothing would have caught it. Both checkpoints load, both run, both produce plausible eyes
about five pixels apart on a 500-pixel face. It surfaced only because the Rust preprocessing
was checked against the *checkpoint* rather than against the exported file, which made the two
disagree for a reason that had nothing to do with the code under test.

The fix is `export_eye_refiner.py`: the export is a committed script, it prints the provenance
the checkpoint carries (`val_stats`, `test_stats`, epoch), and it refuses to leave behind a
file it has not verified against torch. The shipped model reports
`val_stats 1.197 deg, test_stats 1.386 deg, epoch 280` and matches torch to 8.9e-08.

Both traps share a shape worth naming: the artefact that ends a measurement chain -- a
contact sheet for a human, an ONNX file for an application -- is the one with no automated
check behind it, and is therefore where the error lands.
