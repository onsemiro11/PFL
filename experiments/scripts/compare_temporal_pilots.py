#!/usr/bin/env python3
"""Compare fixed-validation seed-0 temporal-relation pilots without test peeking."""
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT / "experiments/results/temporal_pilot_decision.json"
OUT_MD = ROOT / "experiments/results/temporal_pilot_decision.md"

RUNS = {
    "Full": ROOT / "PFL/trained_models/fair_ngsim_pfl_full_seed0/training_summary.json",
    "Global+Temporal pilot": ROOT / "PFL/trained_models/fair_ngsim_pfl_temporal_rel_seed0/training_summary.json",
    "Temporal-only pilot": ROOT / "PFL/trained_models/fair_ngsim_pfl_temporal_only_pilot_seed0/training_summary.json",
    "Zero-relation control pilot": ROOT / "PFL/trained_models/fair_ngsim_pfl_temporal_zero_pilot_seed0/training_summary.json",
}


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def main():
    values = {name: read(path) for name, path in RUNS.items()}
    complete = {name: value is not None for name, value in values.items()}
    scores = {name: value.get("best_validation_rmse_m") for name, value in values.items()
              if value is not None}
    baseline = scores.get("Full")
    candidate = scores.get("Global+Temporal pilot")
    if baseline is None or candidate is None:
        decision, reason = "pending", "primary pilot or Full baseline is incomplete"
    elif candidate > baseline * 1.03:
        decision = "reject"
        reason = "Global+Temporal is more than 3% worse than Full on fixed validation"
    elif not all(complete.values()):
        decision = "continue_controls"
        reason = "primary pilot is not clearly worse; temporal-only and zero controls are required"
    elif candidate < baseline and candidate < scores["Zero-relation control pilot"]:
        decision = "adopt_for_full_runs"
        reason = "Global+Temporal beats Full and its capacity-matched zero-relation control"
    else:
        decision = "reject"
        reason = "Global+Temporal does not beat both Full and the zero-relation control"
    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "selection_data": "fixed validation only; official test is not used",
        "metric": "sample_mean_rmse_k20_m",
        "scores": scores, "complete": complete,
        "decision": decision, "reason": reason,
        "scope": "pilot decision only; not a paper result",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Temporal relation pilot decision", "",
             "Fixed validation K=20 sample-mean RMSE is used; test results are excluded.", "",
             "| Model | RMSE (m) |", "|---|---:|"]
    for name in RUNS:
        score = scores.get(name)
        lines.append(f"| {name} | {'pending' if score is None else f'{score:.4f}'} |")
    lines += ["", f"**Decision:** `{decision}`", "", reason, ""]
    OUT_MD.write_text("\n".join(lines))
    print(json.dumps(result, indent=2))
    if "--gate" in sys.argv:
        raise SystemExit(0 if decision == "adopt_for_full_runs" else
                         3 if decision == "continue_controls" else
                         2 if decision == "reject" else 1)


if __name__ == "__main__":
    main()
