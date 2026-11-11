#!/usr/bin/env python3
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "experiments/results/temporal_study_table.csv"
FIG = ROOT / "experiments/figures"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(TABLE.open())) if TABLE.is_file() else []
    rows = [r for r in rows if r["protocol"] == "gt_free"
            and r["selection_rule"] == "sample_medoid" and int(r["K"]) == 6]
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["dataset"], r["variant"])].append(r)
    plt.figure(figsize=(9, 5.5))
    for key, values in sorted(grouped.items()):
        values.sort(key=lambda r: float(r["horizon"]))
        plt.errorbar([float(r["horizon"]) for r in values],
                     [float(r["RMSE_mean_m"]) for r in values],
                     yerr=[float(r["RMSE_std_m"]) for r in values], marker="o",
                     capsize=2, label="/".join(key))
    if not grouped:
        plt.text(.5,.5,"Pending complete temporal study",ha="center",va="center",
                 transform=plt.gca().transAxes)
    plt.xlabel("Prediction horizon (s)"); plt.ylabel("RMSE (m)")
    plt.title("Temporal relational conditioning: GT-free medoid")
    plt.grid(alpha=.25)
    if grouped: plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / "temporal_relation_rmse.png", dpi=200); plt.close()
    print(FIG / "temporal_relation_rmse.png")


if __name__ == "__main__":
    main()
