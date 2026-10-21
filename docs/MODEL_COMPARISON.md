# Face detector comparison — 12 October 2026

Weights were evaluated locally; FaceCropJet was neither installed nor executed. Its exact extracted RetinaFace ResNet-34 weights were tested using the upstream reference preprocessing and decoder. This measures model performance, not the two finished applications.

![Detection and eye-alignment comparison](benchmarks/model-comparison.png)

## What the evidence supports

SCRFD-80k is competitive with RetinaFace R34 on this held-out domain, while its weights are about 35 times smaller and it runs about 9.5 times faster at the compared CPU operating points. SCRFD-80k plus the eye refiner gives lower paired eye-line error than RetinaFace's native landmarks.

**On a second domain the ranking does not hold.** On COCO, which none of the three models trained on, SCRFD-80k still finds the most faces at the October thresholds, but it also fires 5–6 times as often on photographs with no people in them, mostly on animals and round objects such as clocks, tomatoes and plates. Compared at equal false-positive rates, it finds the fewest faces of the three. See [Second domain: COCO 2017](#second-domain-coco-2017--21-october-2026).

**The Open Images ranking is provisional.** A visual spot-check of 20 randomly sampled unmatched detections per model found many genuine faces missing from the ground truth, and some real faces whose boxes narrowly fail the overlap cutoff. The table compares equal **annotation-mismatch budgets**, not human-verified false-positive rates. This was a spot-check, not a fully adjudicated new dataset. The underlying photographs, review sheets and per-image detections are not bundled with this document.

## Conditions

- 1,859 existing held-out Open Images validation/test photographs; 4,597 face annotations and 82 group/depiction ignore regions.
- 640 × 640 top-left letterboxing, each model’s native colour normalization, NMS 0.4, proposal floor 0.02.
- One-to-one score-ordered matching at box IoU ≥ 0.5. Unmatched detections overlapping an ignore region at IoU > 0.3 are excluded.
- Confidence scores are not calibrated across models. Each operating point below allows at most 0.1 unmatched detections per image; the thresholds were selected on this same set for descriptive comparison, not independent tuning.
- AMD Ryzen 9 7950X, WSL, ONNX Runtime 1.30.0 CPU provider, four inference threads. Timing includes resize, inference and decoding; excludes file reading. Same 48 images, two passes, five warmups. GPU/app throughput is not measured.

## At 0.1 unmatched detections per image

| Model | Threshold | Face recall | Unmatched/image | Median CPU time | Weight size |
|---|---:|---:|---:|---:|---:|
| Face Crop Studio SCRFD-80k | 0.4851 | 76.9% | 0.100 | 12.2 ms | 2.55 MB |
| RetinaFace R34 (FaceCropJet weights) | 0.9807 | 73.9% | 0.100 | 115.6 ms | 88.50 MB |
| Public SCRFD-500M | 0.6415 | 68.8% | 0.100 | 11.2 ms | 2.52 MB |

## Sensitivity to the annotation-mismatch budget

| Model | 0.05/image | 0.1/image | 0.2/image | 0.5/image |
|---|---:|---:|---:|---:|
| Face Crop Studio SCRFD-80k | 66.0% | 76.9% | 82.4% | 86.2% |
| RetinaFace R34 (FaceCropJet weights) | 61.1% | 73.9% | 81.4% | 86.1% |
| Public SCRFD-500M | 51.2% | 68.8% | 77.5% | 83.1% |

## At the same numeric threshold, 0.5

Scores are not calibrated across models, so these are different operating points. This shows why the mismatch-budget comparison should not be read as RetinaFace detecting fewer faces at its normal settings.

| Model | Annotated face recall | Unmatched/image |
|---|---:|---:|
| Face Crop Studio SCRFD-80k | 76.0% | 0.097 |
| RetinaFace R34 (FaceCropJet weights) | 85.3% | 0.371 |
| Public SCRFD-500M | 77.6% | 0.201 |

## Face size at the 640-pixel input

| Model | <16 px | 16–32 px | 32–64 px | 64–128 px | ≥128 px |
|---|---:|---:|---:|---:|---:|
| Face Crop Studio SCRFD-80k | 9.0% | 52.1% | 81.0% | 88.1% | 86.2% |
| RetinaFace R34 (FaceCropJet weights) | 5.9% | 44.3% | 77.3% | 86.8% | 85.2% |
| Public SCRFD-500M | 13.1% | 52.0% | 73.6% | 78.7% | 70.6% |

## Eye alignment

Paired on the same 1,318 side-known, manually clicked faces that all three detectors found at their selected thresholds. The refiner receives actual detected boxes. Eye distance is normalized by ground-truth box width.

| Pipeline | Median angle error | 90th percentile | Within 5° | Median eye distance / box width |
|---|---:|---:|---:|---:|
| Face Crop Studio SCRFD-80k raw landmarks | 5.63° | 20.29° | 46.4% | 0.112 |
| RetinaFace R34 (FaceCropJet weights) raw landmarks | 1.81° | 5.74° | 86.9% | 0.025 |
| Public SCRFD-500M raw landmarks | 2.02° | 6.74° | 83.2% | 0.028 |
| SCRFD-80k + eye refiner | 1.18° | 3.80° | 94.1% | 0.019 |
| RetinaFace + Face Crop Studio eye refiner (experiment) | 1.17° | 3.95° | 93.8% | 0.018 |

The eye refiner took a median 1.49 ms per face, including the crop. This is additional to detection time.

## Second domain: COCO 2017 — 21 October 2026

SCRFD-80k trained only on Open Images; RetinaFace and the public SCRFD trained on WIDER FACE. None of them trained on COCO, so on COCO none of them has a home-domain advantage. The thresholds are October's, unchanged, so they were fixed before COCO was scored.

`tools/dataset/detector_bench.py` produced every figure here and can reproduce them. Before it was used on COCO, it re-ran the October Open Images comparison above: every budget came within 0.4 points of October's figures, and RetinaFace matched exactly. The model files' SHA-256 hashes match those listed below.

### Conditions

- COCO 2017 train and val, limited to licences that allow commercial use (NonCommercial images left out). That gives 12,060 photographs with 23,883 people whose eyes are both labelled by hand, plus 1,687 randomly chosen photographs with no person in them.
- COCO has eye points but no face boxes. A face counts as found when a detection contains both of its eye points and is at most 8 times as wide as the distance between them. Matching is one-to-one, in score order. On 1,562 Open Images faces that have both a box and clicked eyes, this rule agrees with IoU ≥ 0.5 on 98% of faces for every model. It is about 1–1.5 points more generous than IoU, by much the same amount for all three, so it does not change the ranking.
- **FP/neg** is detections per no-person photograph. **FP/out** is detections per person photograph that fall outside every person box. A detection on a person but not on a labelled face (a profile, the back of a head, a hand) counts as neither.
- COCO images are at most 640 pixels on their longer side, so the eye distances below are close to their size at the detector's input.
- Preprocessing, NMS and the proposal floor are the same as in the October run.

### At the October thresholds

| Model | Threshold | All faces | Eyes ≥ 8 px apart | <8 px | 8–16 px | 16–32 px | ≥32 px | FP/neg | FP/out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Face Crop Studio SCRFD-80k | 0.4851 | 64.0% | **84.8%** | 30.3% | 76.6% | 92.7% | 97.0% | 0.030 | 0.017 |
| RetinaFace R34 | 0.9807 | 53.9% | 78.3% | 14.7% | 64.8% | 91.9% | 96.8% | 0.006 | 0.009 |
| Public SCRFD-500M | 0.6415 | 66.7% | 82.4% | 41.6% | 76.7% | 88.4% | 89.4% | 0.005 | 0.022 |

The face counts by eye distance are 9,151 (<8 px), 7,834, 4,570 and 2,328 (≥32 px).

### At equal false-positive rates

Recall on faces with eyes at least 8 px apart, at the threshold that allows the given number of detections per no-person photograph. These thresholds were chosen on COCO itself, so this table describes this set rather than predicting others.

| Model | 0.01 | 0.02 | 0.05 | 0.1 |
|---|---:|---:|---:|---:|
| Face Crop Studio SCRFD-80k | 78.8% | 83.1% | 87.0% | 88.9% |
| RetinaFace R34 | **88.1%** | **93.5%** | **95.0%** | **95.7%** |
| Public SCRFD-500M | 86.8% | 89.6% | 91.9% | 93.0% |

### What the false positives are

All of them were inspected by eye. At its October threshold, SCRFD-80k made 50 detections on the 1,687 no-person photographs. Most were animal faces (dogs, cats, sheep, a bear, cattle) or round objects: six clock faces, tomato slices, plates and bowls, wheels. RetinaFace made 10 and the public SCRFD 9. Most of RetinaFace's look human: faces on a TV screen, dolls, a poster. COCO's no-person photographs are not all face-free: the TV faces are real faces, and all three models find them.

SCRFD-80k was trained on 80,000 photographs that all contain faces, so it has seen few clocks or plates without a face nearby. Its training data has no photographs without faces. That gap is the most likely cause, and fixing it is a training-data change, not a threshold change.

### Eye alignment

On the 5,948 faces that all three detectors found at their October thresholds, with both eyes marked visible and at least 16 px apart. Distance is the median eye-point error divided by the labelled eye separation, because COCO has no face box to normalise by.

| Pipeline | Median angle error | 90th percentile | Within 5° | Eye distance / separation |
|---|---:|---:|---:|---:|
| SCRFD-80k raw landmarks | 6.10° | 20.10° | 43.0% | 0.311 |
| RetinaFace R34 raw landmarks | 2.38° | 6.88° | 80.2% | 0.083 |
| Public SCRFD-500M raw landmarks | 2.71° | 7.83° | 75.9% | 0.092 |
| SCRFD-80k + eye refiner | **2.03°** | **5.99°** | **85.4%** | **0.075** |
| RetinaFace + eye refiner | 2.04° | 5.93° | 85.4% | 0.074 |

On COCO the refiner is still the most accurate pipeline. But its median error is 2.03° here against 1.18° on Open Images, the domain it trained on. COCO's eye points come from body-keypoint labelling rather than careful clicks, so some of that rise is label noise. That noise affects every pipeline here, so it does not change the comparison.

## Limits and interpretation

Open Images is the SCRFD-80k training-data source, although these images are held out. That domain match can favour SCRFD-80k, and the COCO comparison above suggests it did. Image-ID checks confirmed zero overlap with both the SCRFD-80k 80,000-image detector training manifest and the 2,482-image eye-label training set; this does not rule out near-duplicate photographs.

Open Images does not label every real face. “Unmatched” is an annotation-based upper bound on false positives: it also includes genuine unlabelled faces and plausible boxes failing the 0.5 overlap test. Do not interpret it as a human-verified error count.

The [older training notes](../tools/dataset/SCRFD_80K.md) evaluated only images with usable clicked eyes and allowed one detection to cover multiple ground-truth boxes. This evaluation includes every test image and requires one-to-one matches, so the absolute figures are not directly interchangeable.

This does not reproduce FaceCropJet’s complete preprocessing, chosen input resolution, confidence settings, PDF/photo-boundary detection, crop logic or DirectML execution. It tests its detector weights under controlled settings.

COCO covers part of what a broader claim needs: a second domain, photographs with no faces, and thresholds fixed before scoring. It does not cover ordinary studio or school portraits, rotated images or scans, and most of its faces are small.

## Model identity and sources

The RetinaFace checkpoint extracted from FaceCropJet v2.0.2 was byte-identical to the
published [UniFace RetinaFace R34 weights](https://huggingface.co/yakhyo/uniface-weights/blob/main/retinaface_r34.onnx).
The [public SCRFD checkpoint](https://huggingface.co/yakhyo/uniface-weights/blob/main/scrfd_500m.onnx) came from the same mirror. The local SCRFD-80k and eye-refiner
exports were the models used by this project at the time of the run.

SHA-256 hashes:

- **SCRFD-80k:** `2437def9510f2af6431a5b2090ae9f357bc189cb0223be1b6c4d4ff085798e64`
- **RetinaFace R34:** `bd0263dc2a465d32859555cb1741f2d98991eb0053696e8ee33fec583d30e630`
- **Public SCRFD-500M:** `5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a`
- **Eye refiner:** `16b4ed6e42889c0eb3d6ce5068e399c19f12ad686bd121bcf959db34f8cef976`

Preprocessing and box decoding followed the
[InsightFace SCRFD reference](https://github.com/deepinsight/insightface/blob/master/detection/scrfd/tools/scrfd.py)
and [UniFace RetinaFace reference](https://github.com/yakhyo/uniface/blob/main/uniface/detection/retinaface.py).
The RetinaFace benchmark implementation was checked against the upstream methods on three
images; box matching, suppression and the mismatch-budget cutoff also passed self-checks.

The October timing figures come from a local run that is not in the repository. The detection and eye-alignment figures, on both corpora, can be reproduced with
[`detector_bench.py`](../tools/dataset/detector_bench.py). The repository's [dataset tools](../tools/dataset/) and
[training notes](../tools/dataset/SCRFD_80K.md) describe the held-out data and annotation format.
Timing results here use ONNX Runtime to compare weights under equal conditions; see the
[engine measurements](ENGINE_SPEED.md) for the separate historical Rust/WGSL runtime study.
