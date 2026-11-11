#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/hyundo/motion_prediction
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec /home/hyundo/miniconda3/envs/epn/bin/python \
  "$ROOT/experiments/scripts/train_one.py" "$@"
