#!/usr/bin/env python3
"""Paper-facing statistics for temporal relational conditioning."""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.common import atomic_json, write_csv
from experiments.scripts.summarize_results import bootstrap_comparison, complete_stems, read_rows

VARIANTS = ("REL=0", "Full", "Global+Temporal", "Temporal-only",
            "Temporal-zero-control", "EPN")
GROUPS = ("ALL", "G1_direct_follower", "G2_adjacent_near", "G3_other")


def vehicle_values(rows, variant, protocol, rule, k, group="ALL"):
    values = defaultdict(list)
    for row in rows:
        if (row["variant"] == variant and row["protocol"] == protocol
                and row["selection_rule"] == rule and int(row["K"]) == k
                and row["group"] == group and row["RMSE"] != ""):
            values[(int(row["dataset_segment"]), int(row["vehicle_id"]))].append(
                float(row["RMSE"]))
    return {key: float(np.mean(value)) for key, value in values.items()}


def equivalence(rows, left_name, right_name, k=6, margin=0.03, repetitions=2000):
    left = vehicle_values(rows, left_name, "oracle", "gt_endpoint_nearest_trajectory", k)
    right = vehicle_values(rows, right_name, "oracle", "gt_endpoint_nearest_trajectory", k)
    keys = sorted(set(left) & set(right))
    if not keys:
        return {"status": "pending", "paired_vehicles": 0, "margin": margin}
    relative = np.array([(left[x] - right[x]) / max(right[x], 1e-9) for x in keys])
    rng = np.random.default_rng(20260914 + k)
    draws = np.array([rng.choice(relative, len(relative), replace=True).mean()
                      for _ in range(repetitions)])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    status = ("practically_equivalent" if lo >= -margin and hi <= margin else
              f"{right_name}_better" if lo > margin else
              f"{left_name}_better" if hi < -margin else "inconclusive")
    return {"left": left_name, "right": right_name, "K": k,
            "relative_difference": float(relative.mean()),
            "vehicle_bootstrap_95ci": [float(lo), float(hi)],
            "paired_vehicles": len(keys), "margin": margin, "status": status}


def main():
    all_results = {}
    table_rows = []
    for dataset in ("ngsim", "highd"):
        base = ROOT / "experiments/results" / dataset
        stems = complete_stems(base) if base.exists() else set()
        seed_rows, vehicle_rows = [], []
        for stem in stems:
            sp, vp = base / "per_seed" / f"{stem}.csv", base / "per_vehicle" / f"{stem}.csv"
            if sp.is_file(): seed_rows += read_rows([sp])
            if vp.is_file(): vehicle_rows += read_rows([vp])
        relevant = [r for r in seed_rows if r["variant"] in VARIANTS]
        grouped = defaultdict(list)
        for r in relevant:
            if r["group"] == "ALL" and r["RMSE"] != "":
                key = (r["variant"], r["protocol"], r["selection_rule"],
                       int(r["K"]), float(r["horizon"]))
                grouped[key].append(float(r["RMSE"]))
        for key, vals in sorted(grouped.items()):
            variant, protocol, rule, k, horizon = key
            table_rows.append({"dataset": dataset, "variant": variant,
                               "protocol": protocol, "selection_rule": rule,
                               "K": k, "horizon": horizon, "RMSE_mean_m": np.mean(vals),
                               "RMSE_std_m": np.std(vals, ddof=1) if len(vals)>1 else 0.0,
                               "num_seeds": len(vals)})
        comparisons = {}
        for group in GROUPS:
            comparisons[f"global_temporal_vs_full/{group}"] = bootstrap_comparison(
                vehicle_rows, "Full", "Global+Temporal", "gt_free", "sample_medoid",
                6, 2000, group=group)
            comparisons[f"global_temporal_vs_zero/{group}"] = bootstrap_comparison(
                vehicle_rows, "Temporal-zero-control", "Global+Temporal", "gt_free",
                "sample_medoid", 6, 2000, group=group)
            comparisons[f"global_temporal_vs_epn/{group}"] = bootstrap_comparison(
                vehicle_rows, "EPN", "Global+Temporal", "gt_free", "sample_medoid",
                6, 2000, group=group)
        all_results[dataset] = {
            "complete_run_files": sorted(stems), "comparisons": comparisons,
            "oracle_equivalence_to_epn_k6": equivalence(
                vehicle_rows, "Global+Temporal", "EPN", 6),
            "oracle_equivalence_to_epn_k20": equivalence(
                vehicle_rows, "Global+Temporal", "EPN", 20),
            "interpretation": "pending until three seeds and corrected full-test files exist",
        }
    fields = ["dataset", "variant", "protocol", "selection_rule", "K", "horizon",
              "RMSE_mean_m", "RMSE_std_m", "num_seeds"]
    write_csv(ROOT / "experiments/results/temporal_study_table.csv", table_rows, fields)
    atomic_json(ROOT / "experiments/results/temporal_study_summary.json", all_results)
    print(ROOT / "experiments/results/temporal_study_summary.json")


if __name__ == "__main__":
    main()
