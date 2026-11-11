#!/usr/bin/env python3
"""Plot validation and FM-loss diagnostics from preserved per-run logs."""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "experiments/figures"
PROGRESS = re.compile(r"Epoch (\d+) progress ([0-9.]+)% loss [0-9.e+-]+ FM ([0-9.e+-]+)")
VAL = re.compile(r"\[epoch-ema\] RMSE ([0-9.e+-]+) m")


def label(path):
    return path.parent.name.replace("fair_ngsim_pfl_", "").replace("_seed", "/s")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    logs = sorted((ROOT / "PFL/trained_models").glob("fair_ngsim_pfl_*/train.log"))
    selected = [p for p in logs if any(x in p.parent.name for x in
                ("_full_", "temporal_rel", "temporal_only", "temporal_zero"))]
    plt.figure(figsize=(9, 5.5))
    for path in selected:
        xs, ys = [], []
        for line in path.read_text(errors="replace").splitlines():
            m = PROGRESS.search(line)
            if m:
                xs.append(int(m.group(1)) - 1 + float(m.group(2))/100)
                ys.append(float(m.group(3)))
        if xs: plt.plot(xs, ys, linewidth=1, alpha=.8, label=label(path))
    if selected: plt.legend(fontsize=7)
    plt.xlabel("Epoch"); plt.ylabel("Flow-matching loss")
    plt.title("NGSIM training diagnostics"); plt.grid(alpha=.25); plt.tight_layout()
    plt.savefig(FIG / "temporal_training_fm_loss.png", dpi=200); plt.close()

    plt.figure(figsize=(8, 5))
    for path in selected:
        vals = [float(m.group(1)) for line in path.read_text(errors="replace").splitlines()
                if (m := VAL.search(line))]
        if vals: plt.plot(range(1, len(vals)+1), vals, marker="o", label=label(path))
    if selected: plt.legend(fontsize=7)
    plt.xlabel("Epoch validation"); plt.ylabel("K=20 sample-mean RMSE (m)")
    plt.title("Fixed-validation checkpoint metric"); plt.grid(alpha=.25); plt.tight_layout()
    plt.savefig(FIG / "temporal_validation_rmse.png", dpi=200); plt.close()
    print(FIG / "temporal_training_fm_loss.png")
    print(FIG / "temporal_validation_rmse.png")


if __name__ == "__main__":
    main()
