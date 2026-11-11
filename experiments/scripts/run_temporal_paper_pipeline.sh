#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/hyundo/motion_prediction
PYTHON=/home/hyundo/miniconda3/envs/epn/bin/python
CONFIGS="$ROOT/experiments/configs"
LOG="$ROOT/experiments/temporal_paper_pipeline.log"
STATUS="$ROOT/experiments/temporal_study_status.json"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

status() {
  "$PYTHON" - "$STATUS" "$1" "${2:-}" "${3:-}" "${4:-}" <<'PY'
import json, os, sys
from datetime import datetime
from pathlib import Path
p=Path(sys.argv[1]); value=json.loads(p.read_text()) if p.is_file() else {
    "study": "temporal_relation", "state": "running", "completed": []}
stage, dataset, model, seed=sys.argv[2:]
value.update({"state": "running", "updated_at": datetime.now().astimezone().isoformat(),
              "current": {"stage": stage, "dataset": dataset, "model": model,
                          "seed": int(seed) if seed else None}})
tmp=p.with_suffix('.tmp'); tmp.write_text(json.dumps(value,indent=2)+'\n'); os.replace(tmp,p)
PY
}

finish_status() {
  "$PYTHON" - "$STATUS" "$1" <<'PY'
import json, os, sys
from datetime import datetime
from pathlib import Path
p=Path(sys.argv[1]); v=json.loads(p.read_text())
v.update({"state":sys.argv[2], "updated_at":datetime.now().astimezone().isoformat(),
          "current":{"stage":"complete" if sys.argv[2]=='complete' else 'pilot_rejected',
                     "dataset":"", "model":"", "seed":None}})
t=p.with_suffix('.tmp'); t.write_text(json.dumps(v,indent=2)+'\n'); os.replace(t,p)
PY
}

training_complete() {
  "$PYTHON" - "$1" "$2" <<'PY'
import json,sys
from pathlib import Path
s,c=map(Path,sys.argv[1:]); ok=s.is_file() and c.is_file()
if ok: ok=int(json.loads(s.read_text()).get('actual_epochs',0))>0
raise SystemExit(0 if ok else 1)
PY
}

evaluation_complete() {
  "$PYTHON" - "$1" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); raise SystemExit(0 if p.is_file() and json.loads(p.read_text()).get('complete_test') else 1)
PY
}

run_train() {
  local dataset=$1 config=$2 seed=$3 name=$4 model=$5
  local checkpoint="$ROOT/PFL/trained_models/$name/${name}-best.tar"
  local summary="$ROOT/PFL/trained_models/$name/training_summary.json"
  if ! training_complete "$summary" "$checkpoint"; then
    status training "$dataset" "$model" "$seed"
    "$PYTHON" -u "$ROOT/experiments/scripts/train_one.py" --config "$config" \
      --dataset "$dataset" --seed "$seed" --name "$name" 2>&1 | tee -a "$LOG"
  fi
}

run_eval() {
  local dataset=$1 config=$2 seed=$3 name=$4 variant_slug=$5 model=$6
  local result="$ROOT/experiments/results/$dataset/per_seed/pfl_${variant_slug}_seed${seed}.json"
  if ! evaluation_complete "$result"; then
    status evaluation "$dataset" "$model" "$seed"
    "$PYTHON" "$ROOT/experiments/scripts/evaluate_model.py" --model pfl \
      --config "$config" --dataset "$dataset" --seed "$seed" \
      --checkpoint "$ROOT/PFL/trained_models/$name/${name}-best.tar" 2>&1 | tee -a "$LOG"
  fi
}

wait_for_primary_pilot() {
  local summary="$ROOT/PFL/trained_models/fair_ngsim_pfl_temporal_rel_seed0/training_summary.json"
  while [[ ! -f "$summary" ]]; do
    if ! pgrep -f 'train_one.py --config experiments/configs/pfl_temporal_rel.yaml.*fair_ngsim_pfl_temporal_rel_seed0' >/dev/null; then
      echo "primary pilot stopped without training_summary.json" | tee -a "$LOG"
      exit 1
    fi
    sleep 30
  done
}

status waiting_pilot ngsim pfl_global_temporal 0
wait_for_primary_pilot
set +e
"$PYTHON" "$ROOT/experiments/scripts/compare_temporal_pilots.py" --gate | tee -a "$LOG"
gate=$?
set -e
adopt=1
[[ $gate -eq 2 ]] && adopt=0

# The primary pilot is within the preregistered 3% band or better. Run the two
# controls before adopting the architecture for expensive full runs.
if [[ $adopt -eq 1 ]]; then
  run_train ngsim "$CONFIGS/pfl_temporal_only_pilot.yaml" 0 \
    fair_ngsim_pfl_temporal_only_pilot_seed0 pfl_temporal_only_pilot
  run_train ngsim "$CONFIGS/pfl_temporal_zero_pilot.yaml" 0 \
    fair_ngsim_pfl_temporal_zero_pilot_seed0 pfl_temporal_zero_pilot

  set +e
  "$PYTHON" "$ROOT/experiments/scripts/compare_temporal_pilots.py" --gate | tee -a "$LOG"
  gate=$?
  set -e
  [[ $gate -ne 0 ]] && adopt=0
fi

# Full-budget proposed model on both datasets. NGSIM also receives the two
# mechanism controls; HighD is reserved for external generalisation.
if [[ $adopt -eq 1 ]]; then
  for dataset in ngsim highd; do
    for seed in 0 1 2; do
      name="fair_${dataset}_pfl_global_temporal_seed${seed}"
      run_train "$dataset" "$CONFIGS/pfl_global_temporal.yaml" "$seed" "$name" pfl_global_temporal
      run_eval "$dataset" "$CONFIGS/pfl_global_temporal.yaml" "$seed" "$name" \
        'global+temporal' pfl_global_temporal
    done
  done

  for variant in temporal_only temporal_zero; do
    config="$CONFIGS/pfl_${variant}.yaml"
    slug='temporal-only'; model=pfl_temporal_only
    [[ "$variant" == temporal_zero ]] && slug='temporal-zero-control' && model=pfl_temporal_zero
    for seed in 0 1 2; do
      name="fair_ngsim_${model}_seed${seed}"
      run_train ngsim "$config" "$seed" "$name" "$model"
      run_eval ngsim "$config" "$seed" "$name" "$slug" "$model"
    done
  done
fi

# Resume/complete all preregistered PFL/EPN/CV baselines with the corrected
# full-horizon evaluator, then build both the original and temporal summaries.
status baseline_matrix all all ''
bash "$ROOT/experiments/scripts/run_all_experiments.sh" 2>&1 | tee -a "$LOG"
"$PYTHON" "$ROOT/experiments/scripts/summarize_temporal_study.py" 2>&1 | tee -a "$LOG"
"$PYTHON" "$ROOT/experiments/scripts/plot_temporal_study.py" 2>&1 | tee -a "$LOG"
"$PYTHON" "$ROOT/experiments/scripts/plot_training_diagnostics.py" 2>&1 | tee -a "$LOG"
finish_status complete
