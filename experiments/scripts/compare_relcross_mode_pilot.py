#!/usr/bin/env python3
"""Validation-only decision for the combined architecture pilot."""
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "PFL/trained_models/fair_ngsim_pfl_full_seed0/training_summary.json"
CANDIDATE = ROOT / "PFL/trained_models/fair_ngsim_pfl_relcross_mode_pilot_seed0/training_summary.json"
OUT = ROOT / "experiments/results/relcross_mode_pilot_decision.json"


def load(path):
    return json.loads(path.read_text())


def main():
    base = load(BASE)["best_validation_rmse_m"]
    candidate = load(CANDIDATE)["best_validation_rmse_m"]
    delta = candidate - base
    decision = "positive" if delta < 0 else "reject"
    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "selection_data": "fixed validation only; test set was not inspected",
        "metric": "sample_mean_rmse_k20_m",
        "full_seed0": base,
        "relcross_mode_seed0": candidate,
        "absolute_delta_m": delta,
        "relative_delta_percent": 100.0 * delta / base,
        "decision": decision,
        "scope": "combined pilot decision; attribution requires later ablation",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
