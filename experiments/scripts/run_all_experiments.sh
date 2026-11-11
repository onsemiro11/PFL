#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/hyundo/motion_prediction
PYTHON=/home/hyundo/miniconda3/envs/epn/bin/python
CONFIGS="$ROOT/experiments/configs"
LOG="$ROOT/experiments/run_all.log"
STATUS="$ROOT/experiments/run_status.json"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p "$ROOT/experiments/results" "$ROOT/experiments/figures"
mark() {
  extra=()
  if [[ -n "${6:-}" ]]; then
    extra=(--detail "$6")
  fi
  "$PYTHON" "$ROOT/experiments/scripts/update_run_status.py" \
    --file "$STATUS" --state "$1" --stage "$2" \
    --dataset "${3:-}" --model "${4:-}" --seed "${5:-}" \
    "${extra[@]}"
}
complete_result() {
  "$PYTHON" - "$1" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
raise SystemExit(0 if p.is_file() and json.loads(p.read_text()).get("complete_test") else 1)
PY
}
training_complete() {
  "$PYTHON" - "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
summary, checkpoint = map(Path, sys.argv[1:])
ok = summary.is_file() and checkpoint.is_file()
if ok:
    data = json.loads(summary.read_text())
    ok = int(data.get("actual_epochs", 0)) > 0
raise SystemExit(0 if ok else 1)
PY
}
on_error() {
  code=$?
  mark failed pipeline "${CURRENT_DATASET:-}" "${CURRENT_MODEL:-}" "${CURRENT_SEED:-}" "exit_code=$code; see $LOG"
  exit "$code"
}
trap on_error ERR

mark running data_audit
"$PYTHON" "$ROOT/experiments/scripts/audit_splits.py" | tee -a "$LOG"

for dataset in ngsim highd; do
  CURRENT_DATASET=$dataset CURRENT_MODEL=cv CURRENT_SEED=0
  cv_result="$ROOT/experiments/results/$dataset/per_seed/cv_cv_seed0.json"
  if ! complete_result "$cv_result"; then
    mark running evaluation "$dataset" cv 0
    "$PYTHON" "$ROOT/experiments/scripts/evaluate_model.py" \
      --model cv --config "$CONFIGS/pfl_full.yaml" --dataset "$dataset" --seed 0 | tee -a "$LOG"
  fi
  "$PYTHON" "$ROOT/experiments/scripts/update_run_status.py" --file "$STATUS" \
    --state running --stage evaluation_complete --dataset "$dataset" --model cv --seed 0 --completed

  for seed in 0 1 2; do
    for variant in plan0 rel0 full; do
      CURRENT_DATASET=$dataset CURRENT_MODEL="pfl_$variant" CURRENT_SEED=$seed
      config="$CONFIGS/pfl_${variant}.yaml"
      name="fair_${dataset}_pfl_${variant}_seed${seed}"
      result="$ROOT/experiments/results/$dataset/per_seed/pfl_${variant}_seed${seed}.json"
      checkpoint="$ROOT/PFL/trained_models/$name/${name}-best.tar"
      training_summary="$ROOT/PFL/trained_models/$name/training_summary.json"
      if ! complete_result "$result"; then
        if ! training_complete "$training_summary" "$checkpoint"; then
          mark running training "$dataset" "pfl_$variant" "$seed"
          "$PYTHON" "$ROOT/experiments/scripts/train_one.py" \
            --config "$config" --dataset "$dataset" --seed "$seed" --name "$name" | tee -a "$LOG"
        fi
        mark running evaluation "$dataset" "pfl_$variant" "$seed"
        "$PYTHON" "$ROOT/experiments/scripts/evaluate_model.py" \
          --model pfl --config "$config" --dataset "$dataset" --seed "$seed" \
          --checkpoint "$checkpoint" | tee -a "$LOG"
      fi
      "$PYTHON" "$ROOT/experiments/scripts/update_run_status.py" --file "$STATUS" \
        --state running --stage evaluation_complete --dataset "$dataset" \
        --model "pfl_$variant" --seed "$seed" --completed
    done

    CURRENT_DATASET=$dataset CURRENT_MODEL=epn CURRENT_SEED=$seed
    config="$CONFIGS/epn.yaml"
    name="fair_${dataset}_epn_epn_seed${seed}"
    result="$ROOT/experiments/results/$dataset/per_seed/epn_epn_seed${seed}.json"
    checkpoint="$ROOT/EPN/trained_models/$name/${name}-best.tar"
    training_summary="$ROOT/EPN/trained_models/$name/training_summary.json"
    if ! complete_result "$result"; then
      if ! training_complete "$training_summary" "$checkpoint"; then
        mark running training "$dataset" epn "$seed"
        "$PYTHON" "$ROOT/experiments/scripts/train_one.py" \
          --config "$config" --dataset "$dataset" --seed "$seed" --name "$name" | tee -a "$LOG"
      fi
      mark running evaluation "$dataset" epn "$seed"
      "$PYTHON" "$ROOT/experiments/scripts/evaluate_model.py" \
        --model epn --config "$config" --dataset "$dataset" --seed "$seed" \
        --checkpoint "$checkpoint" | tee -a "$LOG"
    fi
    "$PYTHON" "$ROOT/experiments/scripts/update_run_status.py" --file "$STATUS" \
      --state running --stage evaluation_complete --dataset "$dataset" \
      --model epn --seed "$seed" --completed
    mark running intermediate_summary "$dataset" all "$seed"
    "$PYTHON" "$ROOT/experiments/scripts/summarize_results.py" | tee -a "$LOG"
    "$PYTHON" "$ROOT/experiments/scripts/plot_results.py" | tee -a "$LOG"
    "$PYTHON" "$ROOT/experiments/scripts/progress_report.py" | tee -a "$LOG"
  done
done

CURRENT_DATASET= CURRENT_MODEL=summary CURRENT_SEED=
mark running summary
"$PYTHON" "$ROOT/experiments/scripts/summarize_results.py" | tee -a "$LOG"
"$PYTHON" "$ROOT/experiments/scripts/plot_results.py" | tee -a "$LOG"
mark complete complete "" summary "" "all official evaluations summarized"
"$PYTHON" "$ROOT/experiments/scripts/progress_report.py" | tee -a "$LOG"
