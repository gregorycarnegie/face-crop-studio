# Test Fixtures

Sample images, golden outputs, and reference metadata for the test suite.
Individual fixture assets should not be committed if they contain proprietary
or sensitive information; prefer synthetic or cleared data.

## Layout

- `fixtures/images/` — raw input frames used by the OpenCV parity test.
  **Local only** (git-ignored): these are real faces, so they are not committed.
  Tests that need them skip gracefully when the directory is absent (e.g. in CI).
- `fixtures/opencv/` — OpenCV YuNet reference detections for parity validation.
  **Local only** (git-ignored), paired with `fixtures/images/`.
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

Generate the reference with OpenCV's `FaceDetectorYN`, not Face Crop Studio,
so the parity check remains independent. For each image:

1. Run `models/face_detection_yunet_2023mar_640.onnx` at `640x640` with score
   threshold `0.9`, NMS threshold `0.3`, and top-k `5000`.
2. Scale the returned bounding box and five landmark pairs from `640x640` back
   to the original image dimensions.
3. Save `fixtures/opencv/<image-stem>.json`, preserving the image's suffix.

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
