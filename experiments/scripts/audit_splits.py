#!/usr/bin/env python3
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.common import atomic_json, dataset_paths, write_csv


def scan(path, dataset, chunk=200000):
    targets = set()
    egos = set()
    segments = set()
    recordings = set()
    target_ranges = {}
    with h5py.File(path, "r") as f:
        data = f["traj"]
        windows = data.shape[1]
        for start in range(0, windows, chunk):
            block = np.asarray(data[:, start:min(start + chunk, windows)]).T
            ds = block[:, 0].astype(np.int64)
            ego = block[:, 1].astype(np.int64)
            frame = block[:, 2].astype(np.int64)
            segments.update(map(int, np.unique(ds)))
            recordings.update(map(int, np.unique((ds + 1) // 2 if dataset == "highd" else ds)))
            egos.update(zip(ds.tolist(), ego.tolist()))
            grid = block[:, 13:138].astype(np.int64)
            rows, cols = np.nonzero(grid)
            ids = grid[rows, cols]
            for d, vehicle, t in zip(ds[rows], ids, frame[rows]):
                key = (int(d), int(vehicle))
                targets.add(key)
                if key in target_ranges:
                    lo, hi = target_ranges[key]
                    target_ranges[key] = (min(lo, int(t)), max(hi, int(t)))
                else:
                    target_ranges[key] = (int(t), int(t))
    return {
        "windows": windows, "egos": egos, "targets": targets,
        "segments": segments, "recordings": recordings,
        "target_ranges": target_ranges,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--near-frames", type=int, default=50,
                    help="flag shared target ranges within this many 10 Hz frames")
    args = ap.parse_args()
    report = {"protocol": "existing_vehicle_id_split", "near_frames": args.near_frames,
              "datasets": {}}
    rows = []
    for dataset in ("ngsim", "highd"):
        info = {split: scan(path, dataset)
                for split, path in dataset_paths(dataset).items()}
        ds_report = {"splits": {}, "overlap": {}}
        for split, value in info.items():
            plain = {
                "windows": value["windows"], "unique_ego_vehicles": len(value["egos"]),
                "unique_target_vehicles": len(value["targets"]),
                "segments": len(value["segments"]), "recordings": len(value["recordings"]),
            }
            ds_report["splits"][split] = plain
            rows.append({"dataset": dataset, "split": split, **plain})
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            shared = info[left]["targets"] & info[right]["targets"]
            near = 0
            for key in shared:
                a0, a1 = info[left]["target_ranges"][key]
                b0, b1 = info[right]["target_ranges"][key]
                if a0 <= b1 + args.near_frames and b0 <= a1 + args.near_frames:
                    near += 1
            ds_report["overlap"][left + "_" + right] = {
                "ego_vehicle_overlap": len(info[left]["egos"] & info[right]["egos"]),
                "target_vehicle_overlap": len(shared),
                "target_near_time_overlap": near,
                "segment_overlap": len(info[left]["segments"] & info[right]["segments"]),
                "recording_overlap": len(info[left]["recordings"] & info[right]["recordings"]),
            }
        report["datasets"][dataset] = ds_report
    out = ROOT / "experiments" / "results" / "data_audit.json"
    atomic_json(out, report)
    write_csv(out.with_suffix(".csv"), rows)
    print(out)


if __name__ == "__main__":
    main()
