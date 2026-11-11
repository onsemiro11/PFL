#!/bin/bash
# Train PFL on the regenerated NGSIM .mat (independent of EPN/train.py).
set -euo pipefail
ROOT=/home/hyundo/motion_prediction
NAME=${NAME:-ngsim_pfl}
source /home/hyundo/miniconda3/etc/profile.d/conda.sh
conda activate epn
cd "$ROOT/PFL"
# B0 (no ego plan) is the control for the B0 vs B1 comparison: PLAN=0 drops the
# plan grid, the plan token and the relational feature, while the ego stays in the
# neighbour grid with its history. REL=0 keeps the plan grid but drops only the
# relational feature, isolating that contribution.
EXTRA=()
if [[ "${FREEZE_ENCODER:-0}" == "1" ]]; then
    EXTRA+=(--freeze_encoder)
fi
if [[ "${PLAN:-1}" == "0" ]]; then
    EXTRA+=(--use_planning)
fi
if [[ "${REL:-1}" == "0" ]]; then
    EXTRA+=(--use_relational)
fi
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python train.py \
    --name "$NAME" \
    --train_set "$ROOT/datasets/NGSIM/train.mat" \
    --val_set   "$ROOT/datasets/NGSIM/val.mat" \
    --batch_size ${BATCH_SIZE:-64} \
    --num_workers ${NUM_WORKERS:-4} \
    --train_epochs ${TRAIN_EPOCHS:-10} \
    "${EXTRA[@]}" "$@"
