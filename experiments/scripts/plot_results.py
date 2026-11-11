#!/usr/bin/env python3
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "experiments/figures"
FIG.mkdir(parents=True, exist_ok=True)


def rows():
    out = []
    for dataset in ("ngsim", "highd"):
        path = ROOT / "experiments/results" / dataset / "summary.csv"
        if path.is_file():
            with open(path) as f:
                out += list(csv.DictReader(f))
    return out


def select(data, protocol, rule, k=6, group="ALL"):
    return [r for r in data if r["protocol"] == protocol and r["selection_rule"] == rule
            and int(r["K"]) == k and r["group"] == group and r["RMSE"] != ""]


def line_plot(data, filename, title):
    plt.figure(figsize=(8, 5))
    grouped = defaultdict(list)
    for row in data:
        grouped[(row["dataset"], row["variant"], row["protocol"],
                 row["selection_rule"])].append(row)
    for (dataset, variant, protocol, rule), values in sorted(grouped.items()):
        values.sort(key=lambda x: float(x["horizon"]))
        plt.plot([float(x["horizon"]) for x in values],
                 [float(x["RMSE"]) for x in values], marker="o",
                 label=" ".join((dataset, variant, protocol, rule)))
    if not grouped:
        plt.text(.5, .5, "Pending full experiment", ha="center", va="center",
                 transform=plt.gca().transAxes)
    plt.xlabel("Horizon (s)")
    plt.ylabel("RMSE (m)")
    plt.title(title)
    plt.grid(alpha=.25)
    if grouped:
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / filename, dpi=180)
    plt.close()


def bar_plot(data, filename, title):
    values = defaultdict(list)
    for row in data:
        values[(row["dataset"], row["variant"], row["group"])].append(float(row["RMSE"]))
    labels = ["/".join(k) for k in sorted(values)]
    heights = [sum(values[k]) / len(values[k]) for k in sorted(values)]
    plt.figure(figsize=(max(8, len(labels) * .7), 5))
    plt.bar(range(len(labels)), heights)
    if not labels:
        plt.text(.5, .5, "Pending full experiment", ha="center", va="center",
                 transform=plt.gca().transAxes)
    plt.xticks(range(len(labels)), labels, rotation=40, ha="right")
    plt.ylabel("Mean horizon RMSE (m)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIG / filename, dpi=180)
    plt.close()


def oracle_gain_plot(data):
    values = defaultdict(dict)
    for row in data:
        if (row["group"] != "ALL" or int(row["K"]) != 6
                or row["RMSE"] == ""):
            continue
        key = (row["dataset"], row["variant"], row["horizon"])
        if row["protocol"] == "gt_free" and row["selection_rule"] == "sample_medoid":
            values[key]["gt_free"] = float(row["RMSE"])
        if (row["protocol"] == "oracle"
                and row["selection_rule"] == "gt_endpoint_nearest_trajectory"):
            values[key]["oracle"] = float(row["RMSE"])
    grouped = defaultdict(list)
    for (dataset, variant, horizon), pair in values.items():
        if pair.keys() >= {"gt_free", "oracle"}:
            grouped[(dataset, variant)].append(pair["gt_free"] - pair["oracle"])
    labels = ["/".join(key) for key in sorted(grouped)]
    heights = [sum(grouped[key]) / len(grouped[key]) for key in sorted(grouped)]
    plt.figure(figsize=(max(8, len(labels) * .8), 5))
    plt.bar(range(len(labels)), heights)
    if not labels:
        plt.text(.5, .5, "Pending full experiment", ha="center", va="center",
                 transform=plt.gca().transAxes)
    plt.axhline(0, color="black", linewidth=.8)
    plt.xticks(range(len(labels)), labels, rotation=40, ha="right")
    plt.ylabel("GT-free RMSE - oracle RMSE (m)")
    plt.title("Oracle selection gain")
    plt.tight_layout()
    plt.savefig(FIG / "oracle_gain.png", dpi=180)
    plt.close()


def main():
    data = rows()
    line_plot(select(data, "gt_free", "sample_mean", 20),
              "rmse_by_horizon.png", "GT-free sample-mean RMSE")
    oracle_vs_gtfree = (select(data, "oracle", "gt_endpoint_nearest_trajectory", 6)
                        + select(data, "gt_free", "sample_medoid", 6))
    line_plot(oracle_vs_gtfree, "oracle_vs_gtfree.png",
              "Endpoint oracle and GT-free medoid RMSE")
    oracle_gain_plot(data)
    bar_plot([r for r in select(data, "gt_free", "sample_medoid", 6)
              if r["variant"] in ("PLAN=0", "REL=0", "Full")],
             "relation_ablation.png", "Planning and relational ablation")
    grouped = [r for r in data if r["protocol"] == "gt_free"
               and r["selection_rule"] == "sample_medoid" and int(r["K"]) == 6
               and r["group"] != "ALL" and r["RMSE"] != ""]
    bar_plot(grouped, "interaction_group_results.png", "Interaction-group RMSE")


if __name__ == "__main__":
    main()
