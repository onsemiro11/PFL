#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/hyundo/motion_prediction
PYTHON=/home/hyundo/miniconda3/envs/epn/bin/python
CONFIG="$ROOT/experiments/configs/pfl_relcross_mode_pilot.yaml"
NAME=fair_ngsim_pfl_relcross_mode_pilot_seed0
LOG="$ROOT/experiments/relcross_mode_pilot.log"
STATUS="$ROOT/experiments/run_status.json"

mark() {
  "$PYTHON" "$ROOT/experiments/scripts/update_run_status.py" \
    --file "$STATUS" --state "$1" --stage "$2" --dataset ngsim \
    --model pfl_relcross_mode --seed 0 --detail "$3"
}

on_error() {
  code=$?
  mark failed combined_pilot "exit_code=$code; see $LOG"
  exit "$code"
}
trap on_error ERR

mark running combined_pilot_training \
  "5-epoch fixed-validation pilot; epoch log: PFL/trained_models/$NAME/train.log"
"$PYTHON" "$ROOT/experiments/scripts/train_one.py" \
  --config "$CONFIG" --dataset ngsim --seed 0 --name "$NAME" |& tee -a "$LOG"

mark running combined_pilot_comparison "comparing against Full seed0 without test-set use"
"$PYTHON" "$ROOT/experiments/scripts/compare_relcross_mode_pilot.py" |& tee -a "$LOG"
decision=$("$PYTHON" -c \
  "import json; print(json.load(open('$ROOT/experiments/results/relcross_mode_pilot_decision.json'))['decision'])")
if [[ "$decision" == positive ]]; then
  mark running pilot_positive "candidate kept; baseline paper pipeline resumes while result is reviewed"
else
  mark running pilot_rejected "optional candidate disabled (rollback); baseline paper pipeline resumes"
fi

exec bash "$ROOT/experiments/scripts/run_all_experiments.sh"
