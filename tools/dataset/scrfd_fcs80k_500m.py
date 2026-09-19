"""SCRFD-500M on 80,000 licence-clean Open Images photos: the WIDER-scale run.

Copy into the SCRFD checkout's `configs/scrfd/` beside `scrfd_500m_bnkps.py`, then:

    python tools/train.py configs/scrfd/scrfd_fcs80k_500m.py

The model is upstream's, unchanged. Three things differ from `scrfd_fcs_500m.py`, the config
behind the label-count curve, and each is there because that run got it wrong or could not ask:

1. **Every face is boxed.** `labelv2_train_all.txt` comes from `oi_to_labelv2.py`, which writes
   every Open Images face box. The curve's `labelv2_train.txt` held only the clicked faces --
   2,500 boxes where Open Images drew 12,070 -- so each curve run learned ~9,300 real faces as
   background.
2. **Ignore regions actually ignore.** Group-of boxes, depictions and YuNet detections Open
   Images left unboxed are written as ignore regions, and `RetinaFaceDataset` loads them into
   `gt_bboxes_ignore`. But `ATSSAssigner` defaults `ignore_iof_thr` to -1, which skips them
   entirely: the loader, the pipeline and the label file can all be right and the regions
   still train as background. It is set here, on the assigner inside `bbox_head.train_cfg`,
   which is the one the head actually calls.
3. **The schedule is a sample budget, not an epoch count.** Upstream trains 640 epochs over
   WIDER's 12,880 images, ~8.2 M image passes. Keeping the epoch count at 80,000 images would
   cost six times that; keeping the budget gives 100 epochs, with the learning-rate drops at the
   same fractions of the run (55/80 and 68/80).

**Batch 64, 12 loaders, and it is not about speed.** Measured on an RTX 4090 with 32 cores:
batch 16 ran 0.167 s an iteration (96 images/s), batch 64 ran 0.62 s (103 images/s). Throughput
is set by the data pipeline, not the card -- 2 GB of 24 used either way -- and 24 loaders were
slower still, starting no iteration in four minutes (and crashing outright on `received 0 items
of ancdata` until the open-file limit was raised). So the batch is chosen for optimisation:
64 gives 125,000 steps, near upstream's WIDER recipe of ~64,000 steps at batch 128 with this
same learning rate of 0.01, where batch 16 would have taken 500,000. About 23 hours.
"""

_base_ = './scrfd_500m_bnkps.py'

# On the WSL ext4 filesystem, not /mnt/c: 80,000 small files through the 9p bridge would starve
# the data workers. `rsync` the images across before training (TRAINING_SETUP.md).
data_root = '/home/grego/work/oi80k/'

epochs = 100
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=0.001,
    step=[69, 85])
total_epochs = epochs
checkpoint_config = dict(interval=10)

# Overlap (intersection over the anchor's own area) above which an anchor sitting on an
# ignore region is neither positive nor negative. 0.5 is mmdet's usual value where it is set.
IGNORE_IOF = 0.5

model = dict(
    bbox_head=dict(
        train_cfg=dict(assigner=dict(type='ATSSAssigner', topk=9,
                                     ignore_iof_thr=IGNORE_IOF))))
train_cfg = dict(assigner=dict(type='ATSSAssigner', topk=9, ignore_iof_thr=IGNORE_IOF))

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=12,
    train=dict(
        ann_file=data_root + 'labelv2_train_all.txt',
        img_prefix=data_root + 'images/',
    ),
    val=dict(
        ann_file=data_root + 'labelv2_test_all.txt',
        img_prefix=data_root + 'images/',
    ),
    test=dict(
        ann_file=data_root + 'labelv2_test_all.txt',
        img_prefix=data_root + 'images/',
    ),
)
