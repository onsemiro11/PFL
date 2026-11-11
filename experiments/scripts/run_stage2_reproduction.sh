#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/hyundo/motion_prediction
PYTHON=/home/hyundo/miniconda3/envs/epn/bin/python
PRETRAIN_CONFIG="$ROOT/experiments/configs/pfl_stage2_pretrain.yaml"
STAGE2_CONFIG="$ROOT/experiments/configs/pfl_stage2.yaml"
STATUS="$ROOT/experiments/stage2_run_status.json"
LOG="$ROOT/experiments/stage2_reproduction.log"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MALLOC_ARENA_MAX=2

mark() {
  "$PYTHON" - "$STATUS" "$1" "${2:-}" "${3:-}" "${4:-}" <<'PY'
import datetime, json, os, sys
path, stage, dataset, seed, detail = sys.argv[1:]
value = {
    "protocol": "stage2_reproduction_v1",
    "state": "complete" if stage == "complete" else "running",
    "stage": stage,
    "dataset": dataset,
    "seed": int(seed) if seed else None,
    "detail": detail,
    "updated_at": datetime.datetime.now().astimezone().isoformat(),
}
tmp = path + ".tmp"
with open(tmp, "w") as f:
    json.dump(value, f, indent=2)
os.replace(tmp, path)
PY
}

training_complete() {
  "$PYTHON" - "$1" "$2" "$3" <<'PY'
import json, sys
from pathlib import Path
summary, checkpoint, minimum = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
ok = summary.is_file() and checkpoint.is_file()
if ok:
    ok = int(json.loads(summary.read_text()).get("actual_epochs", 0)) >= minimum
raise SystemExit(0 if ok else 1)
PY
}

result_complete() {
  "$PYTHON" - "$1" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
raise SystemExit(0 if p.is_file() and json.loads(p.read_text()).get("complete_test") else 1)
PY
}

failed() {
  code=$?
  mark failed "${CURRENT_DATASET:-}" "${CURRENT_SEED:-}" "exit_code=$code; see $LOG"
  exit "$code"
}
trap failed ERR

cd "$ROOT"

# Select every checkpoint using validation before opening either test set.
for dataset in ngsim highd; do
  for seed in 0 1 2; do
    CURRENT_DATASET=$dataset CURRENT_SEED=$seed
    pre_name="stage2_${dataset}_pretrain_seed${seed}"
    pre_dir="$ROOT/PFL/trained_models/$pre_name"
    pre_final="$pre_dir/$pre_name.tar"
    if ! training_complete "$pre_dir/training_summary.json" "$pre_final" 3; then
      mark training_stage1 "$dataset" "$seed" "3-epoch deterministic mean warm-start"
      "$PYTHON" "$ROOT/experiments/scripts/train_one.py" \
        --config "$PRETRAIN_CONFIG" --dataset "$dataset" --seed "$seed" --name "$pre_name" \
        |& tee -a "$LOG"
    fi

    stage2_name="stage2_${dataset}_seed${seed}"
    stage2_dir="$ROOT/PFL/trained_models/$stage2_name"
    stage2_best="$stage2_dir/$stage2_name-best.tar"
    if ! training_complete "$stage2_dir/training_summary.json" "$stage2_best" 5; then
      mark training_stage2 "$dataset" "$seed" "flow matching from Stage1; fixed validation only"
      "$PYTHON" "$ROOT/experiments/scripts/train_one.py" \
        --config "$STAGE2_CONFIG" --dataset "$dataset" --seed "$seed" \
        --name "$stage2_name" --init-from "$pre_final" |& tee -a "$LOG"
    fi
  done
done

for dataset in ngsim highd; do
  for seed in 0 1 2; do
    CURRENT_DATASET=$dataset CURRENT_SEED=$seed
    stage2_name="stage2_${dataset}_seed${seed}"
    checkpoint="$ROOT/PFL/trained_models/$stage2_name/$stage2_name-best.tar"
    result="$ROOT/experiments/results/$dataset/per_seed/pfl_stage2_seed${seed}.json"
    if ! result_complete "$result"; then
      mark evaluating "$dataset" "$seed" "flow-only K=6/20 test evaluation"
      "$PYTHON" "$ROOT/experiments/scripts/evaluate_model.py" \
        --model pfl --config "$STAGE2_CONFIG" --dataset "$dataset" --seed "$seed" \
        --checkpoint "$checkpoint" |& tee -a "$LOG"
    fi
  done
done

mark summarizing "" "" "streaming paired bootstrap and paper artifacts"
"$PYTHON" "$ROOT/experiments/scripts/summarize_results.py" |& tee -a "$LOG"
"$PYTHON" "$ROOT/experiments/scripts/plot_results.py" |& tee -a "$LOG"
"$PYTHON" "$ROOT/experiments/scripts/progress_report.py" |& tee -a "$LOG"
mark complete "" "" "Stage2 training, evaluation, and summaries complete"
