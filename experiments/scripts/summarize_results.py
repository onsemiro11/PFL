#!/usr/bin/env python3
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.common import RESULT_COLUMNS, atomic_json, load_config, write_csv


def read_rows(pattern):
    rows = []
    for path in sorted(pattern):
        with open(path) as f:
            rows.extend(csv.DictReader(f))
    return rows


def complete_stems(base):
    stems = set()
    for path in (base / "per_seed").glob("*.json"):
        value = json.loads(path.read_text())
        if value.get("complete_test"):
            stems.add(path.stem)
    return stems


def seed_summary(rows):
    grouped = defaultdict(list)
    identity = ("dataset", "model", "variant", "protocol", "selection_rule",
                "K", "NFE", "horizon", "group")
    for row in rows:
        key = tuple(row[k] for k in identity)
        grouped[key].append(row)
    output = []
    for key, members in grouped.items():
        out = dict(zip(identity, key))
        for metric in ("RMSE", "ADE", "FDE", "minADE", "minFDE"):
            values = np.array([float(x[metric]) for x in members if x[metric] != ""])
            out[metric] = float(values.mean()) if len(values) else ""
            out[metric + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0 if len(values) else ""
        out.update({
            "seed": "mean", "num_seeds": len({x["seed"] for x in members}),
            "num_windows": int(round(np.mean([int(x["num_windows"]) for x in members]))),
            "num_vehicles": int(round(np.mean([int(x["num_vehicles"]) for x in members]))),
            "checkpoint": "per-seed-best-validation", "recording_id": "",
            "dataset_segment": "", "vehicle_id": "",
        })
        output.append(out)
    return output


PRIMARY_SPECS = {
    ("PLAN=0", "gt_free", "sample_medoid", 6),
    ("REL=0", "gt_free", "sample_medoid", 6),
    ("Full", "gt_free", "sample_medoid", 6),
    ("Full", "oracle", "gt_endpoint_nearest_trajectory", 6),
    ("Stage2", "gt_free", "sample_medoid", 6),
    ("Stage2", "oracle", "gt_endpoint_nearest_trajectory", 6),
    ("Stage2_REL0", "gt_free", "sample_medoid", 6),
    ("Stage2_REL0", "oracle", "gt_endpoint_nearest_trajectory", 6),
    ("EPN", "gt_free", "sample_medoid", 6),
    ("EPN", "oracle", "gt_endpoint_nearest_trajectory", 6),
}


def stream_primary_values(paths):
    """Read multi-gigabyte vehicle CSVs with bounded memory.

    Each input file is one model/seed. We first average its horizon rows for a
    vehicle, then merge those per-seed values. This is equivalent to the old
    list-based implementation but avoids retaining every CSV row as a dict.
    """
    across_seeds = defaultdict(lambda: [0.0, 0])
    recordings = {}
    for path in sorted(paths):
        within_seed = defaultdict(lambda: [0.0, 0])
        with open(path) as f:
            for row in csv.DictReader(f):
                if not row["RMSE"]:
                    continue
                spec = (row["variant"], row["protocol"], row["selection_rule"],
                        int(row["K"]))
                if spec not in PRIMARY_SPECS:
                    continue
                vehicle = (int(row["dataset_segment"]), int(row["vehicle_id"]))
                key = spec + (row["group"],) + vehicle
                within_seed[key][0] += float(row["RMSE"])
                within_seed[key][1] += 1
                recordings[vehicle] = int(row["recording_id"])
        for key, (total, count) in within_seed.items():
            across_seeds[key][0] += total / count
            across_seeds[key][1] += 1

    values = defaultdict(dict)
    for key, (total, count) in across_seeds.items():
        spec_group, vehicle = key[:5], key[5:]
        values[spec_group][vehicle] = total / count
    return values, recordings


def primary_vehicle_values(index, variant, protocol, rule, k, group="ALL"):
    values, recordings = index
    return values.get((variant, protocol, rule, k, group), {}), recordings


def bootstrap_comparison(index, baseline, candidate, protocol, rule, k, repetitions,
                         group="ALL", seed=20260910):
    left, rec_l = primary_vehicle_values(index, baseline, protocol, rule, k, group)
    right, _ = primary_vehicle_values(index, candidate, protocol, rule, k, group)
    keys = sorted(set(left) & set(right))
    if not keys:
        return {"status": "pending", "paired_vehicles": 0}
    diff = np.array([left[x] - right[x] for x in keys])
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions)
    for i in range(repetitions):
        draws[i] = rng.choice(diff, size=len(diff), replace=True).mean()
    recordings = defaultdict(list)
    for key, value in zip(keys, diff):
        recordings[rec_l[key]].append(value)
    rec_values = np.array([np.mean(v) for v in recordings.values()])
    rec_draws = np.empty(repetitions)
    for i in range(repetitions):
        rec_draws[i] = rng.choice(rec_values, size=len(rec_values), replace=True).mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    rlo, rhi = np.percentile(rec_draws, [2.5, 97.5])
    status = "candidate_significantly_better" if lo > 0 else \
        "baseline_significantly_better" if hi < 0 else "inconclusive"
    return {
        "baseline": baseline, "candidate": candidate, "protocol": protocol,
        "selection_rule": rule, "K": k, "group": group,
        "mean_gain_m": float(diff.mean()),
        "vehicle_bootstrap_95ci_m": [float(lo), float(hi)],
        "recording_bootstrap_95ci_m": [float(rlo), float(rhi)],
        "paired_vehicles": len(keys), "recordings": len(rec_values), "status": status,
    }


def equivalence(index, margin, repetitions, candidate="Full"):
    full, _ = primary_vehicle_values(index, candidate, "oracle",
                                     "gt_endpoint_nearest_trajectory", 6)
    epn, _ = primary_vehicle_values(index, "EPN", "oracle",
                                    "gt_endpoint_nearest_trajectory", 6)
    keys = sorted(set(full) & set(epn))
    if not keys:
        return {"status": "pending", "paired_vehicles": 0, "margin": margin}
    relative = np.array([(full[x] - epn[x]) / max(epn[x], 1e-9) for x in keys])
    rng = np.random.default_rng(20260911)
    draws = np.empty(repetitions)
    for i in range(repetitions):
        draws[i] = rng.choice(relative, len(relative), replace=True).mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    if lo >= -margin and hi <= margin:
        status = "practically_equivalent"
    elif lo > margin:
        status = "EPN_significantly_better"
    elif hi < -margin:
        status = "PFL_significantly_better"
    else:
        status = "inconclusive"
    return {"relative_difference_full_minus_epn": float(relative.mean()),
            "vehicle_bootstrap_95ci": [float(lo), float(hi)], "margin": margin,
            "paired_vehicles": len(keys), "status": status}


def classify(relation, oracle_eq, gtfree, matrix_complete):
    if not matrix_complete:
        return {"case": "pending", "reason": "three-seed model matrix is incomplete"}
    if "pending" in (relation.get("status"), oracle_eq.get("status"), gtfree.get("status")):
        return {"case": "pending", "reason": "full test matrix is incomplete"}
    relation_good = relation["status"] == "candidate_significantly_better"
    gtfree_good = gtfree["status"] == "candidate_significantly_better"
    oracle_same = oracle_eq["status"] == "practically_equivalent"
    if relation_good and oracle_same and gtfree_good:
        case = "A"
    elif relation_good and oracle_eq["status"] == "EPN_significantly_better":
        case = "B"
    elif gtfree_good and not relation_good:
        case = "C"
    else:
        case = "D"
    return {"case": case,
            "epn_gt_free_selection": "sample_medoid_fallback_no_candidate_confidence",
            "warning": "EPN top-confidence is unavailable; any GT-free conclusion must name the fallback rule."}


def main():
    config = load_config(ROOT / "experiments/configs/pfl_full.yaml")
    repetitions = config["statistics"]["bootstrap_repetitions"]
    margin = config["statistics"]["equivalence_margin"]
    all_claims = {}
    for dataset in ("ngsim", "highd"):
        base = ROOT / "experiments/results" / dataset
        stems = complete_stems(base) if base.exists() else set()
        required = {"cv_cv_seed0"}
        required.update("pfl_{}_seed{}".format(variant, seed)
                        for variant in ("plan0", "rel0", "full") for seed in (0, 1, 2))
        required.update("epn_epn_seed{}".format(seed) for seed in (0, 1, 2))
        missing = sorted(required - stems)
        stage2_required = {"pfl_stage2_seed{}".format(seed) for seed in (0, 1, 2)}
        stage2_missing = sorted(stage2_required - stems)
        stage2_rel0_required = {"pfl_stage2_rel0_seed{}".format(seed) for seed in (0, 1, 2)}
        stage2_rel0_missing = sorted(stage2_rel0_required - stems)
        seed_rows = []
        vehicle_files = []
        for stem in stems:
            seed_file = base / "per_seed" / (stem + ".csv")
            vehicle_file = base / "per_vehicle" / (stem + ".csv")
            if seed_file.is_file():
                seed_rows += read_rows([seed_file])
            if vehicle_file.is_file():
                vehicle_files.append(vehicle_file)
        vehicle_index = stream_primary_values(vehicle_files)
        summary = seed_summary(seed_rows)
        fields = RESULT_COLUMNS + ["num_seeds", "RMSE_std", "ADE_std", "FDE_std",
                                   "minADE_std", "minFDE_std"]
        write_csv(base / "summary.csv", summary, fields)
        groups = ("ALL", "G1_direct_follower", "G2_adjacent_near", "G3_other")
        comparisons = {}
        specs = {
            "planning_gain": ("PLAN=0", "REL=0", "gt_free", "sample_medoid"),
            "relational_gain": ("REL=0", "Full", "gt_free", "sample_medoid"),
            "full_vs_epn_gtfree": ("EPN", "Full", "gt_free", "sample_medoid"),
            "full_vs_epn_oracle": ("EPN", "Full", "oracle",
                                   "gt_endpoint_nearest_trajectory"),
            "stage2_vs_epn_gtfree": ("EPN", "Stage2", "gt_free", "sample_medoid"),
            "stage2_vs_epn_oracle": ("EPN", "Stage2", "oracle",
                                     "gt_endpoint_nearest_trajectory"),
            "stage2_relational_gain": ("Stage2_REL0", "Stage2", "gt_free",
                                        "sample_medoid"),
        }
        for name, spec in specs.items():
            comparisons[name] = {
                group: bootstrap_comparison(vehicle_index, *spec, 6, repetitions,
                                            group=group)
                for group in groups
            }
        eq = equivalence(vehicle_index, margin, repetitions)
        stage2_eq = equivalence(vehicle_index, margin, repetitions, candidate="Stage2")
        claim = classify(comparisons["relational_gain"]["ALL"], eq,
                         comparisons["full_vs_epn_gtfree"]["ALL"], not missing)
        value = {
            "dataset": dataset, "complete_run_files": sorted(stems),
            "matrix_complete": not missing, "missing_run_files": missing,
            "stage2_matrix_complete": not stage2_missing,
            "stage2_missing_run_files": stage2_missing,
            "stage2_rel0_matrix_complete": not stage2_rel0_missing,
            "stage2_rel0_missing_run_files": stage2_rel0_missing,
            "pre_registered_equivalence_margin": margin,
            "comparisons": comparisons, "oracle_equivalence": eq,
            "stage2_oracle_equivalence": stage2_eq,
            "claim_classification": claim,
            "window_independence": "statistics use paired dataset-scoped target vehicles",
        }
        atomic_json(base / "summary.json", value)
        all_claims[dataset] = claim
    atomic_json(ROOT / "experiments/results/claim_summary.json", all_claims)


if __name__ == "__main__":
    main()
