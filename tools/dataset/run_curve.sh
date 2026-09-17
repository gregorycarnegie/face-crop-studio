#!/usr/bin/env bash
# Train SCRFD-500M-bnkps four times, on 400 / 800 / 1600 / 1988 labelled eye pairs, to find
# out how many the landmark head actually needs. Every run sees every image and every box;
# only the number of faces whose eye points are supervised changes (see subsample_labels.py).
#
# Roughly 7 hours per run, so about a day in total. Launch it detached and poll the logs:
#
#   chmod +x run_curve.sh
#   setsid nohup ./run_curve.sh > /home/grego/work/curve_all.log 2>&1 < /dev/null &
#   tail -f /home/grego/work/curve_1988.log
#
# Each size writes its own work_dir, so checkpoints never collide, and its own log.
set -u

SCRFD=/home/grego/work/insightface/detection/scrfd
PY=/home/grego/work/scrfd-venv/bin/python
DATA=/mnt/c/Users/grego/Downloads/face-data/openimages
CONFIG=configs/scrfd/scrfd_fcs_500m.py

cd "$SCRFD" || exit 1

for N in 400 800 1600 1988; do
    echo "=== size ${N}: starting $(date -Is)"
    env PYTHONPATH="$SCRFD" "$PY" tools/train.py "$CONFIG" \
        --no-validate \
        --work-dir "${SCRFD}/work_dirs/curve_${N}" \
        --cfg-options "data.train.ann_file=${DATA}/labelv2_train_${N}.txt" \
        > "/home/grego/work/curve_${N}.log" 2>&1
    status=$?
    last=$(grep -E 'Epoch \[[0-9]+\]\[[0-9]+/' "/home/grego/work/curve_${N}.log" | tail -1)
    echo "=== size ${N}: finished $(date -Is), exit ${status}"
    echo "    last iteration: ${last:-none logged}"
done

echo "=== all sizes done $(date -Is)"
