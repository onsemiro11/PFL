#!/usr/bin/env python3
"""Collect reproducible, matched PFL/EPN trajectory samples for paper figures."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.common import dataset_paths, gt_free_medoid, load_config, seed_everything, target_metadata
from experiments.scripts.evaluate_model import build_pfl, epn_args, load_verified, move, unpack


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("pfl", "epn"), required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", choices=("ngsim", "highd"), required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scene-count", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    seed_everything(args.seed, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "pfl":
        model, model_args, transform = build_pfl(cfg, device)
        load_verified(model, args.checkpoint, cfg_path, args.dataset, args.seed, transform)
        sys.path.insert(0, str(ROOT / "PFL"))
        from data import highwayTrajDataset
    else:
        sys.path.insert(0, str(ROOT / "EPN"))
        from data import highwayTrajDataset
        from model import EPNet
        model_args = epn_args(cfg)
        model = EPNet(model_args).to(device)
        load_verified(model, args.checkpoint, cfg_path, args.dataset, args.seed)
        model.train_output_flag = False
    model.eval()

    dataset = highwayTrajDataset(
        str(dataset_paths(args.dataset)["test"]), targ_enc_size=112,
        grid_size=[25, 5], fit_plan_traj=True,
        fit_plan_further_ds=cfg["evaluation"]["plan_info_ds"],
    )
    count = min(args.scene_count, len(dataset))
    # A fixed, endpoint-inclusive grid covers the complete ordered test set and is
    # shared by both models. This avoids selecting scenes after viewing predictions.
    scene_indices = np.unique(np.linspace(0, len(dataset) - 1, count, dtype=np.int64))
    loader = DataLoader(
        Subset(dataset, scene_indices.tolist()), batch_size=args.batch_size,
        shuffle=False, num_workers=4, collate_fn=dataset.collate_fn,
        persistent_workers=True,
    )

    fields = {k: [] for k in (
        "samples", "medoid", "target_hist", "future", "future_mask",
        "ego_plan", "ego_hist", "ego_current", "target_size", "ego_size",
        "target_heading", "ego_heading", "lat", "lon",
    )}
    metadata_all = []
    with torch.no_grad():
        for batch_i, raw in enumerate(loader):
            batch = move(raw, device)
            enc, target_hist, future, mask, idxs = unpack(batch)
            metadata = target_metadata(dataset, idxs, args.dataset, 30.0)
            if args.model == "pfl":
                context = model.encode(*enc, None)
                torch.manual_seed(args.seed * 1000003 + batch_i)
                samples = model.sample(context, target_hist, n_steps=10, n_samples=20, chunk=20)
            else:
                context = model.soc_encode(*enc, None)
                torch.manual_seed(args.seed * 1000003 + batch_i)
                generated, _ = model.generate_trajectories(context, 20)
                samples = generated[..., :2]
            medoid = gt_free_medoid(samples)
            # getPlanFuture stores time in reverse order for the encoder.
            ego_plan = torch.flip(enc[2], dims=[0])
            lat, lon = enc[6], enc[7]
            geometry = {k: [] for k in ("ego_hist", "ego_current", "target_size",
                                         "ego_size", "target_heading", "ego_heading")}
            for meta in metadata:
                ds, target, ego, t = (int(meta[k]) for k in
                                      ("dataset_segment", "vehicle_id", "ego_vehicle_id", "frame_id"))
                target_col = dataset.getTracksCol(ds, target, t)
                ego_col = dataset.getTracksCol(ds, ego, t)
                target_track = dataset.Tracks[ds - 1][target - 1]
                ego_track = dataset.Tracks[ds - 1][ego - 1]
                target_xy = target_track[1:3, target_col]
                ego_xy = ego_track[1:3, ego_col]
                target_prev = target_track[1:3, max(0, target_col - dataset.d_s)]
                ego_prev = ego_track[1:3, max(0, ego_col - dataset.d_s)]
                th = target_xy - target_prev
                eh = ego_xy - ego_prev
                th = th / max(float(np.linalg.norm(th)), 1e-6)
                eh = eh / max(float(np.linalg.norm(eh)), 1e-6)
                geometry["ego_hist"].append(dataset.getHistory(ds, ego, target, t))
                geometry["ego_current"].append(ego_xy - target_xy)
                geometry["target_size"].append(dataset.getSize(ds, target))
                geometry["ego_size"].append(dataset.getSize(ds, ego))
                geometry["target_heading"].append(th)
                geometry["ego_heading"].append(eh)
            arrays = {
                "samples": samples.permute(1, 0, 2, 3),
                "medoid": medoid,
                "target_hist": target_hist.permute(1, 0, 2),
                "future": future.permute(1, 0, 2),
                "future_mask": mask.permute(1, 0, 2),
                "ego_plan": ego_plan.permute(1, 0, 2),
                "lat": lat,
                "lon": lon,
            }
            for key, value in arrays.items():
                fields[key].append(value.detach().cpu().numpy())
            for key, value in geometry.items():
                fields[key].append(np.asarray(value, dtype=np.float32))
            metadata_all.extend(metadata)
            print(f"{args.model} batch {batch_i + 1}/{len(loader)} targets={len(metadata_all)}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: np.concatenate(value, axis=0) for key, value in fields.items()}
    payload["metadata_json"] = np.asarray(json.dumps(metadata_all))
    payload["scene_indices"] = scene_indices
    payload["protocol_json"] = np.asarray(json.dumps({
        "dataset": args.dataset, "model": args.model, "seed": args.seed,
        "K": 20, "NFE": 10 if args.model == "pfl" else 1,
        "selection": "sample_medoid_gt_free",
        "scene_sampling": "linspace_over_ordered_test_set",
        "scene_count": int(len(scene_indices)),
        "checkpoint": str(Path(args.checkpoint).resolve()),
    }))
    np.savez_compressed(output, **payload)
    print(output)


if __name__ == "__main__":
    main()
