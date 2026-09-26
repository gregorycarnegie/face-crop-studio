"""Exercise a packaged CLI, its models, and eye-aligned crops without ONNX Runtime.

Usage: python tools/verify_package.py <package-root> <cli-path> <sample-image>
Paths are resolved before running from an empty temporary working directory, so
models or settings in the checkout cannot accidentally rescue a broken package.
"""

import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    package, cli, sample = (Path(arg).resolve() for arg in sys.argv[1:])
    for path in (cli, sample):
        if not path.is_file() or not path.is_relative_to(package):
            raise RuntimeError(f"packaged file missing or outside package: {path}")
    runtime_files = [p for p in package.rglob("*") if "onnxruntime" in p.name.lower()]
    if runtime_files:
        raise RuntimeError(f"ONNX Runtime must not be packaged: {runtime_files}")

    with tempfile.TemporaryDirectory(prefix="fcs-package-") as directory:
        work = Path(directory)
        env = dict(os.environ, RUST_LOG="info", ORT_DYLIB_PATH=str(work / "absent-runtime"))
        results = []
        for aligned in (False, True):
            output = work / f"detections-{aligned}.json"
            command = [str(cli), "--input", str(sample), "--json", str(output)]
            if aligned:
                command += ["--crop", "--eye-line-align", "--output-dir", str(work / "crops"),
                            "--output-format", "png", "--auto-detect-format=false"]
            run = subprocess.run(command, check=False, cwd=work, env=env, capture_output=True,
                                 text=True, encoding="utf-8", errors="replace", timeout=180)
            log = run.stdout + run.stderr
            print(log)
            run.check_returncode()
            if not re.search(r"Detector: .+ on (wgsl-gpu|cpu-graph)", log):
                raise RuntimeError("packaged detector did not select a built-in engine")
            if aligned and "eye refiner on the built-in CPU graph" not in log:
                raise RuntimeError("packaged eye-refiner model did not load")
            records = json.loads(output.read_text(encoding="utf-8"))
            if len(records) != 1 or not records[0]["detections"]:
                raise RuntimeError("sample must produce at least one detection")
            results.append(records[0]["detections"])

        before, after = results
        if len(before) != len(after):
            raise RuntimeError("eye alignment changed the detection count")
        changed = False
        for raw, refined in zip(before, after):
            for original, eye in zip(raw["landmarks"][:2], refined["landmarks"][:2]):
                if eye is None or not all(math.isfinite(v) for v in eye):
                    raise RuntimeError("refinement did not produce finite eye coordinates")
                changed |= original is None or any(abs(a - b) > 1e-4 for a, b in zip(original, eye))
        if not changed or not any((work / "crops").rglob("*.png")):
            raise RuntimeError("expected refined eyes and an exported PNG crop")
        print(f"PACKAGE OK: {len(after)} faces, refined eyes and crops; no ONNX Runtime")


if __name__ == "__main__":
    main()
