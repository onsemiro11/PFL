#!/usr/bin/env python3
"""Create paper-ready matched trajectory comparisons from qualitative caches."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

M_PER_FT = 0.3048


def load(path):
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files if not k.endswith("_json")}
    out["metadata"] = json.loads(str(z["metadata_json"]))
    out["protocol"] = json.loads(str(z["protocol_json"]))
    return out


def keys(metadata):
    return [(m["dataset_segment"], m["ego_vehicle_id"], m["vehicle_id"], m["frame_id"])
            for m in metadata]


def ade(pred, gt, mask):
    valid = mask[:, 0] > 0
    if not valid.any():
        return np.nan
    return np.linalg.norm(pred[valid] - gt[valid], axis=-1).mean() * M_PER_FT


def category(i, data):
    lat = int(np.argmax(data["lat"][i])) + 1
    lon = int(np.argmax(data["lon"][i])) + 1
    group = data["metadata"][i]["group"]
    valid = data["future_mask"][i, :, 0] > 0
    future_lateral_m = np.abs(data["future"][i, valid, 0]).max() * M_PER_FT if valid.any() else 0.0
    values = []
    # The preprocessing label spans +/-4 seconds around a lane-ID transition, so
    # it can describe a change already completed before t=0. Require visible
    # future lateral motion for a useful prediction figure.
    if lat in (2, 3) and future_lateral_m >= 1.5: values.append("lane_change")
    if lon == 2: values.append("braking")
    if group == "G1_direct_follower": values.append("direct_follower")
    if group == "G2_adjacent_near": values.append("adjacent_near")
    return values


def representative(indices, gains):
    wins = [i for i in indices if gains[i] > 0]
    pool = wins or indices
    if not pool:
        return None
    ordered = sorted(pool, key=lambda i: gains[i])
    return ordered[len(ordered) // 2]


def vehicle(ax, center, size, heading, face, edge, label=None, alpha=.85):
    length, width = size
    norm = float(np.linalg.norm(heading))
    # A stopped NGSIM vehicle has no finite-difference heading. Preprocessing
    # normalizes both carriageways to +longitudinal, so use that road direction.
    h = heading / norm if norm > 1e-6 else np.array([0.0, 1.0])
    p = np.array([-h[1], h[0]])
    corners = np.array([center + h*length/2 + p*width/2,
                        center + h*length/2 - p*width/2,
                        center - h*length/2 - p*width/2,
                        center - h*length/2 + p*width/2])
    ax.add_patch(Polygon(corners, closed=True, facecolor=face, edgecolor=edge,
                         lw=1.4, alpha=alpha, label=label, zorder=7))
    ax.arrow(center[0], center[1], h[0]*length*.35, h[1]*length*.35,
             width=0, head_width=max(width*.25, .15), color=edge,
             length_includes_head=True, zorder=8)


def draw_scene(ax, i, pfl, title="Current interaction (true scale)"):
    h = pfl["target_hist"][i] * M_PER_FT
    ego_h = pfl["ego_hist"][i] * M_PER_FT
    gt = pfl["future"][i] * M_PER_FT
    ego = pfl["ego_plan"][i] * M_PER_FT
    ego_current = pfl["ego_current"][i] * M_PER_FT
    ax.plot(h[-6:, 0], h[-6:, 1], color="#555", lw=1.7)
    ax.plot(gt[:5, 0], gt[:5, 1], color="black", lw=2.2)
    ax.plot(ego_h[-6:, 0], ego_h[-6:, 1], color="#d97706", lw=1.4)
    ax.plot(np.r_[ego_current[0], ego[:5, 0]], np.r_[ego_current[1], ego[:5, 1]],
            color="#f59e0b", lw=2, ls="--")
    vehicle(ax, np.zeros(2), pfl["target_size"][i] * M_PER_FT,
            pfl["target_heading"][i], "#93c5fd", "#0756b8", "Target vehicle")
    vehicle(ax, ego_current, pfl["ego_size"][i] * M_PER_FT,
            pfl["ego_heading"][i], "#fcd34d", "#b45309", "Ego vehicle")
    ax.annotate("TARGET", np.zeros(2), xytext=(5, 4), textcoords="offset points",
                fontsize=8, weight="bold", color="#0756b8")
    ax.annotate("EGO", ego_current, xytext=(5, 4), textcoords="offset points",
                fontsize=8, weight="bold", color="#b45309")
    points = np.vstack([np.zeros((1, 2)), ego_current[None], h[-6:], gt[:5], ego_h[-6:], ego[:5]])
    lo, hi = points.min(0), points.max(0)
    center = (lo + hi) / 2
    span = max(float((hi - lo).max()), 12.0) * .65
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Lateral (m)"); ax.set_ylabel("Longitudinal (m)")
    ax.grid(alpha=.2)


def draw_spatial(ax, i, pfl, epn, title):
    # All positions are target-current-relative and converted from feet to metres.
    for sample in pfl["samples"][i]:
        ax.plot(sample[:, 0] * M_PER_FT, sample[:, 1] * M_PER_FT,
                color="#3b82f6", alpha=.10, lw=.7)
    for sample in epn["samples"][i]:
        ax.plot(sample[:, 0] * M_PER_FT, sample[:, 1] * M_PER_FT,
                color="#ef4444", alpha=.10, lw=.7)
    h = pfl["target_hist"][i] * M_PER_FT
    ego_h = pfl["ego_hist"][i] * M_PER_FT
    gt = pfl["future"][i] * M_PER_FT
    valid = pfl["future_mask"][i, :, 0] > 0
    ego = pfl["ego_plan"][i] * M_PER_FT
    ego_current = pfl["ego_current"][i] * M_PER_FT
    ego_path = np.vstack([ego_current, ego])
    ax.plot(h[:, 0], h[:, 1], color="#555555", lw=1.8, label="Target history")
    ax.plot(ego_h[:, 0], ego_h[:, 1], color="#d97706", lw=1.3, alpha=.65, label="Ego history")
    ax.plot(ego_path[:, 0], ego_path[:, 1], color="#f59e0b", lw=2.0, ls="--", label="Ego plan")
    ax.plot(gt[valid, 0], gt[valid, 1], color="black", lw=2.8, label="Ground truth")
    pm = pfl["medoid"][i] * M_PER_FT
    em = epn["medoid"][i] * M_PER_FT
    ax.plot(pm[:, 0], pm[:, 1], color="#0756b8", lw=2.5, label="PFL medoid")
    ax.plot(em[:, 0], em[:, 1], color="#c51f1a", lw=2.5, label="EPN medoid")
    vehicle(ax, np.zeros(2), pfl["target_size"][i] * M_PER_FT,
            pfl["target_heading"][i], "#93c5fd", "#0756b8", "Target vehicle")
    vehicle(ax, ego_current, pfl["ego_size"][i] * M_PER_FT,
            pfl["ego_heading"][i], "#fcd34d", "#b45309", "Ego vehicle")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Lateral position (m)")
    ax.set_ylabel("Longitudinal position (m)")
    ax.grid(alpha=.2)
    ax.set_aspect("equal", adjustable="box")


def draw_time(ax, i, pfl, epn, axis, title):
    hist = pfl["target_hist"][i, :, axis] * M_PER_FT
    gt = pfl["future"][i, :, axis] * M_PER_FT
    valid = pfl["future_mask"][i, :, 0] > 0
    th = np.linspace(-3.0, 0.0, len(hist))
    tf = np.arange(1, len(gt) + 1) * .2
    for sample in pfl["samples"][i]:
        ax.plot(tf, sample[:, axis] * M_PER_FT, color="#3b82f6", alpha=.09, lw=.7)
    for sample in epn["samples"][i]:
        ax.plot(tf, sample[:, axis] * M_PER_FT, color="#ef4444", alpha=.09, lw=.7)
    ax.plot(th, hist, color="#555555", lw=1.8, label="Target history")
    ax.plot(tf[valid], gt[valid], color="black", lw=2.6, label="Ground truth")
    ax.plot(tf, pfl["medoid"][i, :, axis] * M_PER_FT, color="#0756b8", lw=2.2, label="PFL medoid")
    ax.plot(tf, epn["medoid"][i, :, axis] * M_PER_FT, color="#c51f1a", lw=2.2, label="EPN medoid")
    ax.axvline(0, color="#888", lw=.8, ls=":")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Time from prediction (s)")
    ax.set_ylabel(("Lateral" if axis == 0 else "Longitudinal") + " position (m)")
    ax.grid(alpha=.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pfl", required=True)
    ap.add_argument("--epn", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    pfl, epn = load(args.pfl), load(args.epn)
    if keys(pfl["metadata"]) != keys(epn["metadata"]):
        raise RuntimeError("PFL and EPN caches do not contain identical targets in identical order")
    if pfl["protocol"]["dataset"] != epn["protocol"]["dataset"]:
        raise RuntimeError("dataset mismatch")

    n = len(pfl["metadata"])
    pfl_ade = np.array([ade(pfl["medoid"][i], pfl["future"][i], pfl["future_mask"][i]) for i in range(n)])
    epn_ade = np.array([ade(epn["medoid"][i], epn["future"][i], epn["future_mask"][i]) for i in range(n)])
    gains = epn_ade - pfl_ade
    valid_targets = np.isfinite(gains)
    categories = ("lane_change", "braking", "direct_follower", "adjacent_near")
    selected, used = [], set()
    for name in categories:
        candidates = [i for i in range(n) if valid_targets[i] and name in category(i, pfl) and i not in used]
        idx = representative(candidates, gains)
        if idx is not None:
            selected.append((name, idx))
            used.add(idx)
    # Always include one representative failure case for a balanced appendix panel.
    losses = np.flatnonzero(valid_targets & (gains < 0))
    failure = int(losses[np.argsort(gains[losses])[len(losses) // 2]]) if len(losses) else None

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(selected), 3, figsize=(15, 4.1 * len(selected)), squeeze=False)
    for row, (name, i) in enumerate(selected):
        m = pfl["metadata"][i]
        label = f"{name.replace('_', ' ').title()} | PFL {pfl_ade[i]:.2f} m, EPN {epn_ade[i]:.2f} m"
        draw_scene(axes[row, 0], i, pfl, label + " | current scene (true scale)")
        draw_time(axes[row, 1], i, pfl, epn, 0, "Full-horizon lateral motion")
        draw_time(axes[row, 2], i, pfl, epn, 1, "Full-horizon longitudinal motion")
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[0, 1].get_legend_handles_labels()
    handles, labels = h1 + h2, l1 + l2
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.01))
    fig.subplots_adjust(left=.06, right=.99, top=.98, bottom=.07, hspace=.35, wspace=.24)
    dataset = pfl["protocol"]["dataset"]
    figure_path = out / f"{dataset}_qualitative_comparison.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    figure_pdf = out / f"{dataset}_qualitative_comparison.pdf"
    fig.savefig(figure_pdf, bbox_inches="tight")
    plt.close(fig)

    failure_path = ""
    if failure is not None:
        fig, axes_f = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
        draw_scene(axes_f[0], failure, pfl,
             f"Representative failure | PFL {pfl_ade[failure]:.2f} m, EPN {epn_ade[failure]:.2f} m")
        draw_time(axes_f[1], failure, pfl, epn, 0, "Full-horizon lateral motion")
        draw_time(axes_f[2], failure, pfl, epn, 1, "Full-horizon longitudinal motion")
        axes_f[1].legend(fontsize=8, ncol=2)
        failure_path = out / f"{dataset}_qualitative_failure.png"
        fig.savefig(failure_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    rows = []
    for label, i in selected + ([('failure', failure)] if failure is not None else []):
        m = dict(pfl["metadata"][i])
        m.update({"case": label, "pfl_ade_m": pfl_ade[i], "epn_ade_m": epn_ade[i],
                  "epn_minus_pfl_ade_m": gains[i], "categories": ";".join(category(i, pfl))})
        rows.append(m)
    with open(out / f"{dataset}_qualitative_cases.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "dataset": dataset, "targets_screened": n,
        "targets_with_future": int(valid_targets.sum()),
        "selection_protocol": "median PFL ADE gain among PFL wins in each pre-defined category",
        "failure_protocol": "median PFL ADE loss among PFL losses",
        "metric": "GT-free K=20 sample-medoid ADE",
        "pfl_win_rate": float((gains[valid_targets] > 0).mean()),
        "mean_epn_minus_pfl_ade_m": float(gains[valid_targets].mean()),
        "category_metrics": {
            name: {
                "targets": len(idx),
                "pfl_win_rate": float((gains[idx] > 0).mean()) if len(idx) else None,
                "mean_epn_minus_pfl_ade_m": float(gains[idx].mean()) if len(idx) else None,
                "median_epn_minus_pfl_ade_m": float(np.median(gains[idx])) if len(idx) else None,
            }
            for name in categories
            for idx in [[i for i in range(n) if valid_targets[i] and name in category(i, pfl)]]
        },
        "figure": str(figure_path), "figure_pdf": str(figure_pdf),
        "failure_figure": str(failure_path),
        "cases_csv": str(out / f"{dataset}_qualitative_cases.csv"),
        "pfl_protocol": pfl["protocol"], "epn_protocol": epn["protocol"],
    }
    (out / f"{dataset}_qualitative_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
