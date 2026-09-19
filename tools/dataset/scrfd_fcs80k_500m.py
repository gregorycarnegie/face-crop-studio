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

Batch 16 rather than 8: the curve run used 1.1 GB of a 24 GB card at 0.1 s an iteration, so
the GPU was mostly waiting. The learning rate stays at upstream's 0.01, which upstream pairs
with a much larger multi-GPU batch -- conservative rather than linearly scaled.
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
    samples_per_gpu=16,
    workers_per_gpu=8,
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
