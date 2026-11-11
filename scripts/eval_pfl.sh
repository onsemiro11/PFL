#!/bin/bash
# Evaluate a trained PFL checkpoint on the NGSIM test split.
set -euo pipefail
ROOT=/home/hyundo/motion_prediction
NAME=${NAME:-ngsim_pfl}
source /home/hyundo/miniconda3/etc/profile.d/conda.sh
conda activate epn
cd "$ROOT/PFL"
# Must mirror the flags the checkpoint was trained with, or the state dict will not load.
EXTRA=()
if [[ "${PLAN:-1}" == "0" ]]; then
    EXTRA+=(--use_planning)
fi
if [[ "${REL:-1}" == "0" ]]; then
    EXTRA+=(--use_relational)
fi
if [[ "${SINGLE_SAMPLE:-0}" == "1" ]]; then
    EXTRA+=(--single_sample)
fi
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python evaluate.py \
    --name "$NAME" \
    --test_set "$ROOT/datasets/NGSIM/test.mat" \
    --batch_size ${BATCH_SIZE:-64} \
    --num_workers ${NUM_WORKERS:-4} \
    --best_of_n ${BEST_OF_N:-6} \
    --mean_k ${MEAN_K:-20} \
    --cond_draws ${COND_DRAWS:-0} \
    --cond_beta "${COND_BETA:-}" \
    "${EXTRA[@]}" "$@"
