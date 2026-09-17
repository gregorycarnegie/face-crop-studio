"""SCRFD-500M trained on Face Crop Studio's own labels.

Copy this into the SCRFD checkout's `configs/scrfd/` so the `_base_` include resolves:

    cp tools/dataset/scrfd_fcs_500m.py <scrfd>/configs/scrfd/
    cd <scrfd> && python tools/train.py configs/scrfd/scrfd_fcs_500m.py

Only the data paths differ from upstream `scrfd_500m.py`; the model, schedule and augmentation
are theirs. Faces here carry two eye points at most, and `labelv2.txt` marks the other three
landmarks ignored, so the landmark loss trains on eyes alone while every box still trains the
detector. See DATA_CARD.md for where the labels came from and TRAINING_SETUP.md for the
environment the upstream code needs on modern hardware.
"""

# The `_bnkps` variant, not plain `scrfd_500m`: only the bnkps configs set `use_kps=True`, and
# without it the landmark head never trains and every clicked eye point is ignored. A run
# against the plain config looks perfectly healthy -- loss falls, boxes learn -- while quietly
# throwing away the whole point of the labelling, so check `loss_kps` appears in the log.
_base_ = './scrfd_500m_bnkps.py'

# Edit for another machine: the directory holding images/ and the labelv2 files.
data_root = '/mnt/c/Users/grego/Downloads/face-data/openimages/'

data = {
    'samples_per_gpu': 8,
    'workers_per_gpu': 4,
    'train': {
        'ann_file': data_root + 'labelv2_train.txt',
        'img_prefix': data_root + 'images/',
    },
    'val': {
        'ann_file': data_root + 'labelv2_test.txt',
        'img_prefix': data_root + 'images/',
    },
    'test': {
        'ann_file': data_root + 'labelv2_test.txt',
        'img_prefix': data_root + 'images/',
    },
}
