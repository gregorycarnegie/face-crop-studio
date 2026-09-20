# Test Fixtures

Sample images, golden outputs, and reference metadata for the test suite.
Individual fixture assets should not be committed if they contain proprietary
or sensitive information; prefer synthetic or cleared data.

## Layout

- `fixtures/images/` — raw input frames used by the CLI tests.
  **Local only** (git-ignored): these are real faces, so they are not committed.
  Tests that need them skip gracefully when the directory is absent (e.g. in CI).
- `fixtures/opencv/` — **no longer used by any test.** These were OpenCV YuNet's
  detections, and `cli_detections_match_opencv_parity_samples` compared this
  project against them. That test went with YuNet: the scores are on a different
  scale and the current detector disagrees with these boxes by design. Kept on
  disk because they are local files, not repository content — delete them
  whenever you like.
- `fixtures/golden/` — **committed** golden outputs. Synthetic, no image data:
  - `crop_regions.json` — expected `CropRegion` for the scenarios in
    `fcs-core/tests/golden_crop_regions.rs`.

## Curating your own image fixtures

Use images you own or have permission to process. Keep real faces local: both
`fixtures/images/` and `fixtures/opencv/` are git-ignored. Include varied face
sizes, poses, lighting, skin tones, image sizes, and supported file formats.

Copy each image into `fixtures/images/` and add one category suffix immediately
before its extension:

- `_n` — no face, for example `street_n.jpg`.
- `_o` — an obscured face, for example `mask_o.png`.
- `_g` — a group of faces, for example `team_g.webp`.
- No suffix — one unobscured face, for example `portrait.jpg`.

Use only one category suffix and inspect every image manually. The parity test
sorts filenames and checks at most the first three usable images in each
category. A negative golden must contain no detections; every other category
must contain at least one.

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
