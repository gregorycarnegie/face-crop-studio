# Test Fixtures

Sample images, golden outputs, and reference metadata for the test suite.
Individual fixture assets should not be committed if they contain proprietary
or sensitive information; prefer synthetic or cleared data.

## Layout

- `fixtures/images/` — **committed.** 60 Open Images photographs (145 faces), all
  CC BY 2.0 and attributed in `fixtures/images/ATTRIBUTION.md`, downscaled to 640 px,
  with their face boxes in `ground_truth.json`, plus a synthetic `test_pattern.png`.
  `fcs-core/tests/detection_accuracy.rs` scores the detector against those boxes, and
  the CLI tests use them as ordinary inputs. They are a fixed sample of the held-out
  set behind `docs/MODEL_COMPARISON.md`; `tools/dataset/make_accuracy_fixtures.py`
  rebuilds them (see below). This folder used to hold local photographs with no
  licence to redistribute, which is why it was git-ignored and CI never ran the tests
  that needed it.
- `fixtures/opencv/` — **no longer used by any test.** These were OpenCV YuNet's
  detections, and `cli_detections_match_opencv_parity_samples` compared this
  project against them. That test went with YuNet: the scores are on a different
  scale and the current detector disagrees with these boxes by design. Kept on
  disk because they are local files, not repository content — delete them
  whenever you like.
- `fixtures/oracle/` — **committed**, and the only images in the repository. Eight
  Open Images photographs, all CC BY 2.0 and attributed in
  `fixtures/oracle/ATTRIBUTION.md`, plus `torch_epoch100.json`: what the Python
  implementation produces for them. `fcs-core/tests/python_parity.rs` compares
  this project against that, which is the only check here that does not compare
  the project against itself. Real faces, but licensed for redistribution —
  unlike everything else above, which is why these are the exception.
- `fixtures/golden/` — **committed** golden outputs. Synthetic, no image data:
  - `crop_regions.json` — expected `CropRegion` for the scenarios in
    `fcs-core/tests/golden_crop_regions.rs`.

## Rebuilding `fixtures/images/`

Only commit images you are licensed to redistribute. The set is a rule, not a
hand-picked selection: the held-out images ordered by the SHA-256 of their file
name, the eight in `fixtures/oracle/` skipped, the first 60 taken. To rebuild it
(for example after the held-out labels change), run this in the SCRFD venv from
the repository root, then re-measure the bounds in `detection_accuracy.rs`:

```bash
/home/grego/work/scrfd-venv/bin/python tools/dataset/make_accuracy_fixtures.py     /mnt/c/Users/grego/Downloads/face-data/openimages fixtures/images
```

Any other images you want to test with locally can go anywhere under `fixtures/`
outside the tracked folders; git ignores them.

## Generating OpenCV golden detections

The recipe that generated `fixtures/opencv/` is kept below for the record; nothing reads these
files any more. It ran OpenCV's `FaceDetectorYN` rather than Face Crop Studio, which is what made
the comparison independent — the thing the current test suite does not have, since every check
now compares this project against itself.

Each JSON file has this shape (use an empty `detections` array for `_n`):

```json
{
  "image": "../fixtures/images/portrait.jpg",
  "input_size": [640, 640],
  "score_threshold": 0.9,
  "nms_threshold": 0.3,
  "top_k": 5000,
  "detections": [
    {
      "score": 0.95,
      "bbox": [100.0, 80.0, 120.0, 160.0],
      "landmarks": [[130.0, 120.0], [180.0, 120.0], [155.0, 150.0], [135.0, 190.0], [175.0, 190.0]]
    }
  ]
}
```

Review the golden JSON and any annotated OpenCV output, then run:

```bash
cargo test -p fcs-core --test parity
```

## Regenerating golden crop regions

After an intentional change to the crop geometry, refresh and review the diff:

```powershell
$env:UPDATE_GOLDEN = "1"; cargo test -p fcs-core --test golden_crop_regions
```

```bash
UPDATE_GOLDEN=1 cargo test -p fcs-core --test golden_crop_regions
```

When adding fixtures that should be committed, re-include their path in
`.gitignore` (the `fixtures/*` rule ignores fixture contents by default).
