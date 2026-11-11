#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/hyundo/motion_prediction
PYTHON=/home/hyundo/miniconda3/envs/epn/bin/python
STATUS="$ROOT/experiments/run_status.json"
LOG="$ROOT/experiments/stage2_fair_reval.log"
CONFIG="$ROOT/experiments/configs/pfl_stage2_legacy_reval.yaml"
CHECKPOINT="$ROOT/PFL/trained_models/ngsim_pfl_stage2/ngsim_pfl_stage2-best.tar"

mark() {
  "$PYTHON" "$ROOT/experiments/scripts/update_run_status.py" --file "$STATUS" \
    --state "$1" --stage "$2" --dataset ngsim --model pfl_stage2_reval \
    --seed 0 --detail "$3"
}
on_error() {
  code=$?
  mark failed stage2_fair_reval "exit_code=$code; see $LOG"
  exit "$code"
}
trap on_error ERR

mark running stage2_fair_reval \
  "legacy checkpoint; current target set, nested K=6/20, full-horizon mask v2; flow samples only"
"$PYTHON" "$ROOT/experiments/scripts/evaluate_model.py" \
  --model pfl --config "$CONFIG" --dataset ngsim --seed 0 \
  --checkpoint "$CHECKPOINT" --allow-unverified-checkpoint |& tee -a "$LOG"

mark running stage2_reval_complete \
  "fair re-evaluation complete; resuming preregistered pipeline from saved checkpoint"
exec bash "$ROOT/experiments/scripts/run_all_experiments.sh"
