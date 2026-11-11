#!/usr/bin/env python3
"""Collect compact, machine-readable progress for the multi-day experiment."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/results/progress_report.json"


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def main():
    status = read(ROOT / "experiments/run_status.json") or {}
    runs = []
    for model_root in (ROOT / "PFL/trained_models", ROOT / "EPN/trained_models"):
        if not model_root.exists():
            continue
        for run_dir in sorted(model_root.glob("fair_*")):
            manifest = read(run_dir / "run_manifest.json")
            if not manifest:
                continue
            summary = read(run_dir / "training_summary.json")
            runs.append({
                "name": run_dir.name, "dataset": manifest.get("dataset"),
                "seed": manifest.get("seed"),
                "training_complete": summary is not None,
                "actual_epochs": None if summary is None else summary.get("actual_epochs"),
                "best_validation_rmse_m": None if summary is None else summary.get("best_validation_rmse_m"),
                "training_seconds": None if summary is None else summary.get("training_seconds"),
            })
    official = {}
    for dataset in ("ngsim", "highd"):
        directory = ROOT / "experiments/results" / dataset / "per_seed"
        official[dataset] = sorted(
            p.stem for p in directory.glob("*.json")
            if (read(p) or {}).get("complete_test")
        ) if directory.exists() else []
    value = {"status": status, "training_runs": runs,
             "official_evaluations": official,
             "note": "smoke outputs are excluded"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
