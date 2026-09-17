# Running SCRFD's training code on current hardware

SCRFD's code is MIT and still the right thing to train, but it was pinned to mmcv 1.2/1.3 and
PyTorch 1.x, and an RTX 4090 is Ada (sm_89), which needs CUDA 11.8 or newer. So the pinned
stack cannot run on this GPU at all, whatever else is true. This is the combination that does
run, and the eight fixes it needed, recorded because rediscovering them costs an afternoon.

Verified 2026-09-17, on a 4090 through WSL2: `mmcv.ops.nms` running as a compiled CUDA op, an
SCRFD-500M detector built from config, this project's labels parsed by SCRFD's own
`RetinaFaceDataset`, and 20 epochs of training with loss falling from 2.03 to 0.98 at about
0.11 s per iteration and 1.1 GB of VRAM (312 iterations per epoch over 2,482 images; the full
upstream schedule projects to roughly 7.5 hours).

That run used the non-keypoint config, so it trained boxes only -- see fix 11, which is the
trap most worth knowing about here.

## The stack

| | |
|---|---|
| Host | Windows 11, RTX 4090, driver 610.47 |
| WSL2 | Ubuntu 26.04, 30 GB RAM, CUDA visible via the Windows driver |
| Python | **3.11**, standalone via `uv` |
| PyTorch | 2.1.0+cu121, torchvision 0.16.0+cu121 |
| mmcv | **mmcv-full 1.7.2**, prebuilt wheel for cu121/torch2.1 |
| mmdet | 2.7.0, the copy vendored inside `detection/scrfd` |

Python 3.11 is not a preference, it is the intersection: PyTorch publishes CUDA wheels for
cp310-cp313, `mmcv_full` 1.7.2 only for cp38-cp311, and Ubuntu 26.04 and this Windows install
both ship Python 3.14, for which **no CUDA PyTorch wheels exist at all**.

## Setup

```sh
# uv, for a standalone 3.11 that the distro does not offer
curl -LsSf https://astral.sh/uv/install.sh | sh
~/.local/bin/uv python install 3.11
~/.local/bin/uv venv --python 3.11 ~/work/scrfd-venv

# SCRFD itself; sparse, because insightface is large and only one directory matters
git clone --filter=blob:none --sparse https://github.com/deepinsight/insightface.git
cd insightface && git sparse-checkout set detection/scrfd

# the stack, in this order
~/.local/bin/uv pip install --python ~/work/scrfd-venv/bin/python \
    torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
~/.local/bin/uv pip install --python ~/work/scrfd-venv/bin/python mmcv-full==1.7.2 \
    --find-links https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
~/.local/bin/uv pip install --python ~/work/scrfd-venv/bin/python \
    'numpy<2' 'setuptools<70' wheel scipy matplotlib six terminaltables tqdm pycocotools cython
```

## The fixes, and why each is needed

1. **`numpy<2`.** torch 2.1 and mmcv 1.7.2 are both compiled against numpy 1.x. With numpy 2.x
   torch still imports but prints `Failed to initialize NumPy: _ARRAY_API not found` and the
   numpy bridge is dead.
2. **`setuptools<70`.** torch 2.1's `cpp_extension` imports `pkg_resources`, which setuptools
   removed in v81. Without it, `import mmcv` fails outright.
3. **`wheel`.** Needed for any source build once `--no-build-isolation` is in play, since
   mmdet's `setup.py` imports torch and an isolated build environment would not have it.
4. **`scipy`.** mmdet's Hungarian assigner imports it at config-build time.
5. **`pycocotools` instead of `mmpycocotools`.** The open-mmlab fork is source-only and does
   not build here; the maintained package works. See fix 6 for the consequence.
6. **Patch the pycocotools guard** in `mmdet/datasets/coco.py` and `retinaface.py`. Both assert
   `pycocotools.__version__ >= '12.0.2'`, but modern pycocotools exposes **no `__version__` at
   all**, so the assert raises `AttributeError`, which their `except AssertionError` does not
   catch. Replace it with
   `assert getattr(pycocotools, "__version__", "12.0.2") >= "12.0.2"`, keeping the
   indentation: the line sits inside an `if`. Nothing here uses COCO-style evaluation anyway.
7. **Raise the mmcv ceiling** in `mmdet/__init__.py` from `1.4` to `1.8`. mmcv-full 1.7.2 is
   the only 1.x build with wheels for torch 2.x, and torch 2.x is the floor for this GPU. The
   assertion was the only obstacle; the APIs it guards still line up.
8. **Run from `detection/scrfd` *and* set `PYTHONPATH` to it.** The editable install fails
   (`autotorch` drags in `configspace<=0.4.11`, which will not build) and is not needed, but
   the working directory alone is not enough: running a *script* puts the script's directory
   on `sys.path`, not the cwd, so `python tools/train.py` cannot see `./mmdet` even though
   `python -c "import mmdet"` from the same place can.
9. **Replace the removed numpy aliases.** `np.int`, `np.float` and friends went in numpy 1.24,
   and this code uses them in 15 places. Word boundaries matter when fixing it, so that
   `np.int32`, `np.float32` and `np.bool_` (83 uses) are left alone:

   ```sh
   find mmdet -name '*.py' -print0 | xargs -0 sed -i -E \
     's/\bnp\.int\b/int/g; s/\bnp\.float\b/float/g; s/\bnp\.bool\b/bool/g'
   ```
10. **Patch mmcv's scatter for torch 2.x.** In
    `site-packages/mmcv/parallel/_functions.py`, mmcv passes a bare GPU *index* to torch's
    `_get_stream`, which in torch 2.x expects a `torch.device` and does `device.type`, so it
    raises `AttributeError: 'int' object has no attribute 'type'`. Wrap it:
    `_get_stream(torch.device("cuda", device) if isinstance(device, int) else device)`.
    Running under `torchrun --launcher pytorch` does **not** avoid this: mmcv 1.x's
    distributed wrapper uses the same scatter path, and the launcher only hides the real
    traceback behind its own. This is the one fix that edits an installed dependency rather
    than SCRFD's code, so it has to be reapplied whenever the venv is rebuilt.
11. **Derive from a `_bnkps` config.** Only those set `use_kps=True`. Against plain
    `scrfd_500m` the landmark head never trains and clicked eye points are silently
    discarded, while the run looks healthy because box loss still falls. Confirm `loss_kps`
    appears in the logged losses, not merely in the config dump.

## Checking it works

```sh
cd <scrfd>
~/work/scrfd-venv/bin/python -c "import torch; print(torch.cuda.get_device_name(0))"
~/work/scrfd-venv/bin/python -c "import torch; from mmcv.ops import nms; \
  b=torch.tensor([[0.,0.,10.,10.],[1.,1.,11.,11.]],device='cuda'); \
  s=torch.tensor([0.9,0.8],device='cuda'); print(nms(b,s,0.5)[1].tolist())"   # -> [0]
~/work/scrfd-venv/bin/python -c "from mmcv import Config; from mmdet.models import build_detector; \
  print(build_detector(Config.fromfile('configs/scrfd/scrfd_500m.py').model).__class__.__name__)"
```

## Training on this project's labels

```sh
python tools/dataset/to_labelv2.py <data_dir>          # COCO keypoints -> labelv2.txt
cp tools/dataset/scrfd_fcs_500m.py <scrfd>/configs/scrfd/
cd <scrfd>
env PYTHONPATH=. ~/work/scrfd-venv/bin/python tools/train.py \
    configs/scrfd/scrfd_fcs_500m.py --no-validate
```

Then check the log shows `loss_kps` alongside `loss_cls` and `loss_bbox`. If it does not, the
config is not a keypoint one and the eye labels are being ignored (fix 11).

A caution when driving WSL from a Windows shell: `$` does not survive the trip, escaped or
not, so WSL command payloads here use literal paths and no shell variables.
