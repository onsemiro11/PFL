#!/bin/bash
# Build the highD train/val/test mats in the format PFL and EPN load.
#
# Two stages: a Python converter that makes highD look like the raw NGSIM txt
# files (direction split, 25 Hz -> 10 Hz, metres -> feet), and the MATLAB stage
# that is a copy of the NGSIM one bar the file list.
#
# Env:
#   HIGHD_RECS  process only the first N recordings (default all 60). The 60
#               recordings come to 15.9 M rows at 10 Hz, about 35% more than the
#               whole of NGSIM; MATLAB holds the merged table, the per-vehicle
#               tracks and the three splits at once, so the full set peaks well
#               above this machine's 30 GB. HIGHD_RECS=40 is roughly NGSIM-sized
#               and HIGHD_RECS=20 fits alongside a running training job.
#   SKIP_CONVERT=1  reuse the CSVs already in data/raw_highd/
set -euo pipefail
ROOT=/home/hyundo/motion_prediction
source /home/hyundo/miniconda3/etc/profile.d/conda.sh
conda activate epn

if [[ "${SKIP_CONVERT:-0}" != "1" ]]; then
    echo "== converting highD recordings to NGSIM format"
    python "$ROOT/preprocess/highd_to_ngsim.py" \
        ${HIGHD_RECS:+--recordings $(seq 1 "${HIGHD_RECS}")}
fi

echo "== MATLAB preprocessing"
cd "$ROOT/data"
HIGHD_RAW="$ROOT/data/raw_highd/" HIGHD_OUT="$ROOT/data/dataset/highd_processed/" \
HIGHD_RECS="${HIGHD_RECS:-}" \
    matlab -batch "addpath('$ROOT/preprocess'); run('$ROOT/preprocess/preprocess_highd_fast.m')"

echo "== linking datasets/highD"
mkdir -p "$ROOT/datasets/highD"
OUT="$ROOT/data/dataset/highd_processed/25x5_8Veh_nofL"
ln -sf "$OUT/gridTrainAround.mat" "$ROOT/datasets/highD/train.mat"
ln -sf "$OUT/gridValAround.mat"   "$ROOT/datasets/highD/val.mat"
ln -sf "$OUT/gridTestAround.mat"  "$ROOT/datasets/highD/test.mat"
ls -la "$ROOT/datasets/highD"
