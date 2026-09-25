"""Stamp provenance and the CC BY 4.0 licence into a shipped model's ONNX metadata.

  stamp_model_metadata.py scrfd <scrfd80k_500m_640.onnx>
  stamp_model_metadata.py eye_refiner <eye_refiner.onnx>

The models are released as bare files, so credit has to travel inside them: whoever finds one
in an installer, a release asset or somebody else's project should be able to see where it came
from and on what terms without this repository. Netron shows these fields on opening the file,
ONNX Runtime returns them from `InferenceSession.get_modelmeta()`, and `onnx.load` exposes them
as `metadata_props`.

Metadata only: the graph and weights are untouched, so outputs are identical, but the file's
digest changes. `fcs-core`'s initializer reader declares the graph field alone and skips the
rest, and ONNX Runtime ignores metadata. Re-stamping replaces the keys rather than duplicating
them. Both exporters call `stamp` before they verify, so a fresh export is never unstamped.
"""

import argparse
from pathlib import Path

import onnx

REPO = "https://github.com/gregorycarnegie/face-crop-studio"
AUTHOR = "Face Crop Studio Contributors"
LICENSE = "CC-BY-4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
PRODUCER = "Face Crop Studio"

#: Per model: the version that matches its release tag, and what someone holding only the file
#: needs to know. Keep in step with `models/README.md`.
MODELS = {
    "scrfd": {
        "title": "SCRFD-80k face detector",
        "version": 2,  # of the weights, first released as models-scrfd-80k-v2
        "summary": (
            "SCRFD-500M face detector trained by Face Crop Studio on 80,000 CC BY 2.0 Open "
            "Images photographs (244,683 boxed faces), with eye landmarks only."
        ),
        "io": (
            "Input `input.1` [1, 3, 640, 640], RGB, (pixel - 127.5) / 128, the source "
            "letterboxed into the top-left. Outputs: nine tensors, three per stride 8/16/32 -- "
            "sigmoided scores [1, N, 1], box distances [1, N, 4] and keypoint distances "
            "[1, N, 10] in units of the stride. No NMS in the graph. Of the five keypoints only "
            "the two eyes were trained."
        ),
        "training_data": "Open Images V7, 80,000 photographs, CC BY 2.0",
        "produced_by": "tools/dataset/export_scrfd.py",
    },
    "eye_refiner": {
        "title": "Face Crop Studio eye refiner",
        "version": 1,  # of the weights, first released as models-eye-refiner-v1
        "summary": (
            "Refines a face detector's two eye points for eye-line alignment. Trained by Face "
            "Crop Studio from scratch on 1,988 hand-clicked eye pairs from CC BY 2.0 Open "
            "Images photographs."
        ),
        "io": (
            "Input `crop` [batch, 3, 112, 112], RGB, (pixel - 127.5) / 128: the face box "
            "expanded to max(w, h) * 1.25 about its centre. Output `eyes` [batch, 4]: two eye "
            "points as fractions of the crop side, the viewer's left eye first."
        ),
        "training_data": "Open Images V7, 1,988 hand-clicked eye pairs, CC BY 2.0",
        "produced_by": "tools/dataset/train_eye_refiner.py, exported by export_eye_refiner.py",
    },
}


def attribution(model: str) -> str:
    """The credit line CC BY asks reusers to carry."""
    return f"{MODELS[model]['title']} by {AUTHOR}, {REPO}, licensed under CC BY 4.0"


def stamp(path: str | Path, model: str) -> None:
    """Write the provenance fields for `model` into the ONNX file at `path`, in place."""
    info = MODELS[model]
    onnx_model = onnx.load(str(path))

    # Keep what actually produced the graph: the exporter's own name moves into the version.
    if onnx_model.producer_name != PRODUCER:
        exporter = f"{onnx_model.producer_name} {onnx_model.producer_version}".strip()
        onnx_model.producer_version = f"exported with {exporter}" if exporter else ""
        onnx_model.producer_name = PRODUCER
    onnx_model.model_version = info["version"]
    onnx_model.doc_string = (
        f"{info['title']}. {info['summary']}\n\n{info['io']}\n\n"
        f"Source: {REPO}\nLicence: CC BY 4.0 ({LICENSE_URL}). Credit: {attribution(model)}."
    )

    props = {
        "title": info["title"],
        "author": AUTHOR,
        "source": REPO,
        "license": LICENSE,
        "license_url": LICENSE_URL,
        "attribution": attribution(model),
        "training_data": info["training_data"],
        "produced_by": info["produced_by"],
    }
    kept = [p for p in onnx_model.metadata_props if p.key not in props]
    del onnx_model.metadata_props[:]
    onnx_model.metadata_props.extend(kept)
    for key, value in props.items():
        onnx_model.metadata_props.add(key=key, value=value)

    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, str(path))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("model", choices=sorted(MODELS))
    ap.add_argument("path")
    args = ap.parse_args()

    before = onnx.load(args.path).graph.SerializeToString()
    stamp(args.path, args.model)
    after = onnx.load(args.path)
    # Metadata only: a stamp that touched the graph would be a different model under old tests.
    if after.graph.SerializeToString() != before:
        raise SystemExit("the graph changed while stamping; not a metadata-only edit")
    print(
        f"stamped {args.path}: {after.producer_name}, model_version {after.model_version}"
    )
    for prop in after.metadata_props:
        print(f"  {prop.key}: {prop.value}")


if __name__ == "__main__":
    main()
