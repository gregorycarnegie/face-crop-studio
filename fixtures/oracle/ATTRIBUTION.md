# Oracle fixtures: attribution and provenance

These eight photographs are the only images committed to this repository. Everything else under
`fixtures/` is git-ignored, deliberately, because it is real faces that were never licensed for
redistribution (see `fixtures/README.md`). These eight are different: they come from the Open
Images dataset and every one is **CC BY 2.0**, the same licence as the 80,000 photographs the
shipped detector was trained on.

They exist so `fcs-core/tests/python_parity.rs` can run in CI. That test is the only check in
the workspace that compares this project against something it did not write, and it needs real
photographs — a synthetic gradient finds no faces, so there would be nothing to compare.

## Why these eight

Chosen for spread, not size: the point is to exercise both letterbox axes and all three strides,
because a wrong preprocessing convention shows up as a systematic box shift and those are the
paths it shifts. Aspect ratios run 0.57 to 2.60 and face heights 4% to 79%. 227 KB in total.

| Image | Why it is here |
|---|---|
| `64b5dea6c5e2af7f.jpg` | portrait 680x1024, one large face (45% of height) |
| `ab93efe2a60a5cea.jpg` | landscape 1024x639, one very large face (79%) |
| `1012be5da4a238ab.jpg` | square-ish 777x768, one mid-sized face (28%) |
| `261ef45576ed8302.jpg` | portrait 684x1024, five annotated faces; three found at 0.4 |
| `02018995730f71e9.jpg` | landscape 1024x682, two faces (21%) |
| `7e0251d3d846b35d.jpg` | landscape 1024x685, one small face (7%) -- stride 8 |
| `7fea60b7e0795984.jpg` | wide 1024x394 (2.6:1), one face (63%) |
| `b0e349f8a58ea04e.jpg` | tall 584x1024 (0.57:1), face too small to detect -- must find nothing |

The selection came from the val+test splits of every Open Images photograph carrying a
`Human face` box (12,416 images, all CC BY 2.0 — verified, not assumed: the licence column holds
exactly one distinct value across all 167,056 rows of those two splits). Candidates were limited
to files under 140 KB and then one was taken per category above, smallest first, which makes the
choice reproducible rather than a matter of taste.

## Regenerating the reference

`torch_epoch100.json` is the output of the Python implementation these images are compared
against. It is **not** produced by this codebase — that is the entire point. From WSL, with the
training environment described in `tools/dataset/TRAINING_SETUP.md`:

```bash
cd /home/grego/work/insightface/detection/scrfd
PYTHONPATH=$PWD /home/grego/work/scrfd-venv/bin/python     /mnt/c/.../tools/dataset/scrfd_detect.py     work_dirs/oi80k/epoch_100.pth     /mnt/c/.../fixtures/oracle/images     /mnt/c/.../fixtures/oracle/torch_epoch100.json --thresh 0.4
```

Then rewrite the absolute `/mnt/c/...` paths in the JSON to repo-relative
(`fixtures/oracle/images/<id>.jpg`), or the test will only work on the machine that generated it.

`epoch_100.pth` is the checkpoint `models/scrfd80k_500m_640.onnx` was exported from. The test
pins that: exporting from a different epoch moves every score far more than its tolerances allow,
so a mismatch here means either the shipped model or this reference is stale.

## Attribution

Unmodified originals, redistributed under [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/).

| Image | Author | Source |
|---|---|---|
| `64b5dea6c5e2af7f.jpg` | [Kamaljith K V](https://www.flickr.com/people/kamaljith/) | [Flickr](https://www.flickr.com/photos/kamaljith/9114858108) |
| `ab93efe2a60a5cea.jpg` | [filsinger](https://www.flickr.com/people/filsinger/) | [Flickr](https://www.flickr.com/photos/filsinger/409763398) |
| `1012be5da4a238ab.jpg` | [Kungsleden AB](https://www.flickr.com/people/60073775@N07/) | [Flickr](https://www.flickr.com/photos/60073775@N07/6095999381) |
| `261ef45576ed8302.jpg` | [Viva Vivanista](https://www.flickr.com/people/54585499@N04/) | [Flickr](https://www.flickr.com/photos/54585499@N04/5515638351) |
| `02018995730f71e9.jpg` | [Felix  Huth](https://www.flickr.com/people/felixhuth/) | [Flickr](https://www.flickr.com/photos/felixhuth/15754751554) |
| `7e0251d3d846b35d.jpg` | [M&amp;R Glasgow](https://www.flickr.com/people/glasgows/) | [Flickr](https://www.flickr.com/photos/glasgows/85752783) |
| `7fea60b7e0795984.jpg` | [Michael Aulia](https://www.flickr.com/people/michaelaulia/) | [Flickr](https://www.flickr.com/photos/michaelaulia/4708473251) |
| `b0e349f8a58ea04e.jpg` | [Phil Parker](https://www.flickr.com/people/45131642@N00/) | [Flickr](https://www.flickr.com/photos/45131642@N00/6931485438) |
