# Face data: what is usable, and how good YuNet is on it

Measured 2026-09-16. This exists to answer two questions before any work on replacing
YuNet 2023: **how much face training data can ship under this project's MIT OR Apache-2.0
licence**, and **what recall a replacement has to beat**.

The obvious corpus, WIDER FACE, is [released for non-commercial academic research
only](http://shuoyang1213.me/WIDERFACE/). Training on it would leave new weights in the
same grey area YuNet's already occupy, so it is measured nowhere below.

## 1. YuNet baseline — the bar to beat

Open Images validation + test, 12,416 images, 19,377 human-drawn face boxes (group-of and
depiction boxes excluded and treated as ignore regions). YuNet 2023 at 640x640, GPU
inference, `nms_threshold` 0.2 from `config/gui_settings.json`, on an RTX 4090 (DX12).
A detection counts as found at IoU >= 0.5.

| Face width at 640 px | Faces | Recall @ 0.8 (GUI default) | Recall @ 0.5 |
|----------------------|-------|----------------------------|--------------|
| <16 px               | 1,158 | 10.8%                      | 30.5%        |
| 16-32 px             | 2,744 | 53.4%                      | 69.9%        |
| 32-64 px             | 4,621 | 72.7%                      | 81.2%        |
| 64-128 px            | 5,140 | 79.4%                      | 85.5%        |
| 128+ px              | 5,714 | 77.8%                      | 85.0%        |
| **Croppable (>=32 px)** | **15,475** | **76.8%**           | **84.0%**    |
| All                  | 19,377 | 69.5%                     | 78.8%        |

Face size is box width scaled to the detector's 640 px input, because that, not the
original pixel size, decides whether a face is findable.

**Box extents already agree.** On matched pairs YuNet's boxes are 0.98 of the human box
height and 0.96 of the width, with centres aligned (median y shift -0.00). Training on
Open Images boxes would therefore not move the crop framing that `face_height_pct` and
the crop presets are tuned against.

**Most unmatched detections are unlabelled faces, not false positives.** 60 of the 5,376
detections that matched no box at 0.8 were sampled at random and reviewed by eye: 58 were
real faces Open Images never boxed (96.7%, 95% Wilson interval 88.6-99.1%). The two
exceptions were a spurious second box on a statue's neck, below a face the labels flag as
a depiction, and a puppy. Correcting the totals for that rate:

|           | Against the labels | Corrected for unlabelled faces |
|-----------|--------------------|--------------------------------|
| Recall, all faces | 69.5%      | ~76.0% (95% CI 75.6-76.1%)     |
| Precision | 71.5% apparent     | **~99.0%** (95% CI 96.8-99.7%) |

Recall barely moves, because a face YuNet found that nobody labelled adds to both sides of
the fraction. Precision moves a long way: the detector is right nearly every time it fires,
and it is the labels that are incomplete. A further 1,202 detections overlapped an ignore
region and are excluded from both columns as unjudgeable. Rebuild the contact sheet this
came from with `sample_unmatched.py`.

## 2. Open Images V7 — the training pool

Every image carrying a `Human face` box (`/m/0dzct`), across train, validation and test.

| | Count |
|---|---|
| Images with face boxes | 344,043 (**all** CC BY 2.0, **all** with an author recorded) |
| Face boxes | 1,060,312 (1,049,667 drawn by hand, 10,645 model-proposed then human-checked) |
| **Croppable faces** (>=32 px at 640, real, not group-of) | **586,722**, across 288,899 images |
| Width at 640 px: <16 / 16-32 / 32-64 / 64-128 / 128+ | 160,638 / 251,807 / 302,631 / 216,963 / 123,047 |
| Depictions (drawings, statues, faces on screens) | 94,600 |
| Group-of boxes (one box over several faces) | 5,226 |
| Marked occluded / truncated | 523,232 / 52,978 |

**No landmarks.** Open Images has boxes only. The eye points that
`fcs-core::face_cropper` uses for eye-line alignment have to come from elsewhere.

## 3. COCO 2017 — hand-labelled eye points

COCO's person keypoints include both eyes, the nose and both ears, each with a visibility
flag, under a CC BY 4.0 annotation licence. There are no mouth corners. Its images carry
individual Flickr licences, so they need filtering.

| | Strict (CC BY, no known restrictions) | Lenient (adds ShareAlike, NoDerivs) |
|---|---|---|
| Usable images | 20,797 | 38,353 |
| Faces with both eyes labelled | 13,194 | 23,883 |
| ...eyes >=16 px apart | 3,704 | 6,898 |
| Profile or partly hidden faces (need a hand-drawn box) | 5,598 | 10,351 |
| Images with no people (negatives) | 9,609 | 17,769 |
| Face images with no crowd or unlabelled person | 4,336 | 7,880 |

NonCommercial images excluded: 84,934 of 123,287 (69%).

**Unlabelled people are cheap to handle.** 19,935 people in the strict tier have no
keypoints, so their faces carry no label, but COCO still gives each a person box. Marking
those boxes as ignore regions stops the model being punished for finding a face there,
with no labelling work.

## 4. What this means for a replacement detector

- Open Images supplies roughly four times the ~150k faces needed to match an
  SCRFD-scale model, with boxes already drawn by people.
- The remaining labelling work is eye points, not boxes: about two clicks per face.
  COCO's 13,194 strict-tier faces with hand-labelled eyes are a free head start.
- Open Images' own validation and test splits (12,416 images) are a ready-made held-out
  test set, which is what section 1 measures.

### First eye-point pass, 2026-09-16

2,000 boxes from those splits went through `label_eyes.py`:

| | Count |
|---|---|
| Faces with two eye points | 1,599 |
| Skipped: no two visible eyes | 401 |
| ...of which screen order equals anatomical order | 1,549 |
| `upside_down`: roll past 90 degrees, so the subject's right eye is on the viewer's right | 34 |
| `steep_roll`: eyes within 8% of box width horizontally, so x-order means little | 37 (21 also `upside_down`) |

Quality checks on the 1,599: every point landed inside its box, horizontal eye separation came
out at a median 0.40 of box width, and the eye line at 0.39 of box height from the top. Those
are what real faces look like, not what careless clicking looks like.

The two flags exist because screen position only implies which eye is which while the head's
roll stays within +/-90 degrees, and 34 of these faces are past that. Candidates were found
automatically -- clicks arriving right-to-left, or eyes nearly in vertical line -- and all 50
were then checked by eye: the right-to-left ones were genuinely inverted faces, not misclicks.
A converter should read screen order from the coordinates and anatomy from the flags.

The 401 skips are a signal worth keeping rather than a loss: they mark boxes where no landmark
head should be supervised, since a second eye is not visible to supervise it with.

### Training labels

`to_coco_keypoints.py` turns those clicks into `face_keypoints_coco.json`: 1,859 images and
2,000 annotations, with a `face` category carrying five keypoints in RetinaFace/YuNet order
(right_eye, left_eye, nose, right_mouth, left_mouth) in the subject's own frame. Nose and
mouth are never set; nobody has clicked them.

| Annotation kind | Count |
|---|---|
| Both eyes, side known, face upright | 1,549 |
| Both eyes, side known, face inverted | 13 |
| Points kept but side unknown (`steep_roll`) | 37 |
| Box only, eyes not visible | 401 |

COCO rather than the RetinaFace `label.txt` that SCRFD's own scripts read, because these
labels are partial: `label.txt` has no way to say "two of five known" -- a face carries all
five points or a placeholder -- so writing it would discard every eye point. COCO keypoints
carry per-point visibility, mmdetection and mmpose read the format directly, and it is the
same shape as COCO's own eye points in section 3, so the two sources can merge.

Checked at generation, on all 2,000: no keypoint falls outside its box, every upright face
puts the subject's right eye to the left of its left eye on screen, every inverted face puts
it to the right, and every `steep_roll` face stores its pair under `fcs.eyes_unordered` with
no keypoints set.

## 5. Caveats before anyone trains on this

1. **CC BY means attribution.** Every image needs crediting. Open Images records the
   author; COCO records only a Flickr URL, so credits there need looking up, and some
   photos will have been deleted.
2. **The licences were recorded in 2018.** Photos may have been relicensed or removed
   since. Re-check availability at download time and keep the manifest with the weights.
3. **Open Images boxes are not exhaustive, and this is now measured rather than feared.**
   About 5,200 faces across these 12,416 images carry no box at all (section 1). Training
   on them as they stand teaches a detector that an unlabelled face is background. Either
   treat everything the labels do not cover as an ignore region, or complete the labels
   first -- YuNet's own output at 0.8 is a reasonable draft for that, being right ~99% of
   the time, and the same audit script checks any such pass.
4. **The lenient COCO tier is a legal judgement call**, not a technical one: whether a
   model trained on ShareAlike images is an adaptation of them is unsettled. The strict
   tier avoids the question.
5. **The 49% occlusion rate in Open Images looks high** for what it claims to mark. Read
   the annotation guide before relying on that flag.
6. **These are real people's faces.** Detection is not identification, but state the
   purpose in any published model card and check face-data law (for example Illinois'
   BIPA) before shipping weights.

## 6. Reproducing this

```sh
# COCO: licence tiers and hand-labelled eye points (~241 MB)
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip
python tools/dataset/coco_faces.py annotations_trainval2017.zip

# Open Images: face boxes, streamed so the 2.3 GB box file never lands on disk
for f in v6/oidv6-train-annotations-bbox v5/validation-annotations-bbox v5/test-annotations-bbox; do
  curl -sSL "https://storage.googleapis.com/openimages/$f.csv" \
    | grep -E "^ImageID|,/m/0dzct," > "faces-$(basename $f).csv"
done
for s in train/train-images-boxable-with-rotation validation/validation-images-with-rotation \
         test/test-images-with-rotation; do
  curl -sSL -O "https://storage.googleapis.com/openimages/2018_04/$s.csv"
done
python tools/dataset/oi_faces.py .        # argument is the directory holding those CSVs

# YuNet baseline: images from the public mirror, then detect and score
#   https://open-images-dataset.s3.amazonaws.com/{validation,test}/<ImageID>.jpg
fcs-cli --input images/ --json det.json --gpu --gpu-inference --score-threshold 0.8
python tools/dataset/eval_yunet.py det.json images/

# Audit: contact sheet of the detections that matched no box, to judge by eye
python tools/dataset/sample_unmatched.py . det.json unmatched.html 60

# Label eye points, which Open Images has none of: click-through page, resumable,
# appending to eye_labels.jsonl
python tools/dataset/label_eyes.py . --count 2000

# Convert those clicks into COCO keypoint annotations
python tools/dataset/to_coco_keypoints.py .
```

A by-product of the first full run: 7 of these images are CMYK JPEGs, which used to kill
the whole batch. Fixed in `38554fb`, with a regression test in `fcs-utils::image_utils`.
