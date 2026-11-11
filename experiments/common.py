import csv
import json
import os
import platform
import random
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
METRES_PER_FOOT = 0.3048
HORIZONS = (5, 10, 15, 20, 25)


def load_config(path):
    """Load JSON syntax stored in a .yaml file (JSON is valid YAML 1.2)."""
    with open(path) as f:
        cfg = json.load(f)
    required = {"model", "variant", "model_args", "train", "evaluation", "statistics"}
    missing = required - set(cfg)
    if missing:
        raise ValueError("missing config keys: " + ", ".join(sorted(missing)))
    return cfg


def dataset_paths(dataset):
    if dataset not in ("ngsim", "highd"):
        raise ValueError("dataset must be ngsim or highd")
    folder = ROOT / "datasets" / ("NGSIM" if dataset == "ngsim" else "highD")
    return {s: folder / (s + ".mat") for s in ("train", "val", "test")}


def seed_everything(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
    os.replace(tmp, path)


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parameter_counts(model):
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def system_metadata(device):
    value = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "device": str(device),
    }
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        value.update({
            "gpu": torch.cuda.get_device_name(device),
            "cuda": torch.version.cuda,
        })
    return value


def gt_free_mean(samples):
    """GT-free point estimate. Deliberately accepts no ground-truth argument."""
    return samples.mean(dim=0)


def gt_free_medoid(samples):
    """GT-free candidate selection. Deliberately accepts no ground-truth argument."""
    flat = samples.flatten(2).permute(1, 0, 2)
    idx = torch.cdist(flat, flat).sum(-1).argmin(-1)
    batch = torch.arange(samples.shape[1], device=samples.device)
    return samples[idx, batch]


def oracle_endpoint(samples, gt_endpoint):
    """Oracle selector; GT is explicit and isolated from GT-free functions."""
    if gt_endpoint.shape != (samples.shape[1], 2):
        raise ValueError("gt_endpoint must have shape [B, 2]")
    idx = (samples[:, :, -1] - gt_endpoint).square().sum(-1).argmin(0)
    batch = torch.arange(samples.shape[1], device=samples.device)
    return samples[idx, batch]


def full_horizon_subset(samples, future, mask, metadata):
    """Keep only targets with an observed 5-second endpoint.

    Endpoint-oracle metrics must never select against a padded future value.
    """
    valid = mask[-1, :, 0].bool()
    keep = valid.detach().cpu().tolist()
    return (samples[:, valid], future[:, valid], mask[:, valid],
            [meta for meta, include in zip(metadata, keep) if include])


def direct_min_metrics(samples, future, mask):
    """Direct minADE/minFDE per target; selectors may differ by metric."""
    gt = future.permute(1, 0, 2)
    valid = mask.permute(1, 0, 2)[..., 0]
    dist = torch.linalg.vector_norm(samples - gt.unsqueeze(0), dim=-1)
    ade = (dist * valid.unsqueeze(0)).sum(-1) / valid.sum(-1).clamp_min(1).unsqueeze(0)
    last = valid.sum(-1).long().clamp_min(1) - 1
    b = torch.arange(gt.shape[0], device=gt.device)
    fde = dist[:, b, last]
    return ade.min(0).values, fde.min(0).values


def target_metadata(dataset, idxs, dataset_name, threshold_m):
    """Metadata in collate target order, derived only from the current frame."""
    result = []
    threshold_ft = threshold_m / METRES_PER_FOOT
    for idx in idxs:
        ds = int(dataset.itsDsId(idx))
        ego = int(dataset.itsPlanVehId(idx))
        t = dataset.itsTime(idx)
        ego_col = dataset.getTracksCol(ds, ego, t)
        ego_track = dataset.Tracks[ds - 1][ego - 1]
        ego_xy = ego_track[1:3, ego_col]
        ego_lane = int(ego_track[3, ego_col])
        recording = (ds + 1) // 2 if dataset_name == "highd" else ds
        for target in dataset.itsTargVehsId(idx):
            target = int(target)
            col = dataset.getTracksCol(ds, target, t)
            track = dataset.Tracks[ds - 1][target - 1]
            xy = track[1:3, col]
            lane = int(track[3, col])
            dy = float(xy[1] - ego_xy[1])
            if lane == ego_lane and dy < 0:
                group = "G1_direct_follower"
            elif abs(lane - ego_lane) == 1 and abs(dy) <= threshold_ft:
                group = "G2_adjacent_near"
            else:
                group = "G3_other"
            result.append({
                "recording_id": recording, "dataset_segment": ds,
                "vehicle_id": target, "ego_vehicle_id": ego,
                "frame_id": int(t), "group": group,
            })
    return result


class VehicleAccumulator:
    """Accumulate overlapping windows, then emit vehicle-level independent units."""

    def __init__(self, dataset, model, variant, seed, checkpoint, nfe):
        self.fixed = dict(dataset=dataset, model=model, variant=variant, seed=seed,
                          checkpoint=str(checkpoint or ""), NFE=nfe)
        self.pred = defaultdict(lambda: {
            "sq": np.zeros(25), "dist": np.zeros(25), "count": np.zeros(25),
            "windows": 0,
        })
        self.scalar = defaultdict(lambda: {"sum": 0.0, "count": 0, "windows": 0})

    @staticmethod
    def _keys(meta, protocol, rule, k):
        base = (meta["recording_id"], meta["dataset_segment"], meta["vehicle_id"],
                protocol, rule, int(k))
        return [("ALL",) + base, (meta["group"],) + base]

    def add_prediction(self, pred, future, mask, metadata, protocol, rule, k):
        gt = future.permute(1, 0, 2)
        valid = mask.permute(1, 0, 2)[..., 0]
        sq = ((pred - gt).square().sum(-1) * valid).detach().cpu().numpy()
        dist = (torch.linalg.vector_norm(pred - gt, dim=-1) * valid).detach().cpu().numpy()
        cnt = valid.detach().cpu().numpy()
        for i, meta in enumerate(metadata):
            if cnt[i].sum() == 0:
                continue
            for key in self._keys(meta, protocol, rule, k):
                rec = self.pred[key]
                rec["sq"] += sq[i]
                rec["dist"] += dist[i]
                rec["count"] += cnt[i]
                rec["windows"] += 1

    def add_scalar(self, values, metadata, protocol, rule, k):
        values = values.detach().cpu().numpy() * METRES_PER_FOOT
        for value, meta in zip(values, metadata):
            for key in self._keys(meta, protocol, rule, k):
                rec = self.scalar[key]
                rec["sum"] += float(value)
                rec["count"] += 1
                rec["windows"] += 1

    def rows(self):
        rows = []
        for key, rec in self.pred.items():
            group, recording, segment, vehicle, protocol, rule, k = key
            count = np.maximum(rec["count"], 1)
            for h in HORIZONS:
                # A target contributes to a horizon only when that endpoint was
                # observed. This prevents padded zeros from becoming perfect
                # long-horizon predictions.
                if rec["count"][h - 1] <= 0:
                    continue
                ade_denom = max(float(rec["count"][:h].sum()), 1.0)
                row = dict(self.fixed, group=group, recording_id=recording,
                           dataset_segment=segment, vehicle_id=vehicle,
                           protocol=protocol, selection_rule=rule, K=k,
                           horizon=h / 5.0,
                           RMSE=float(np.sqrt(rec["sq"][h-1] / count[h-1]) * METRES_PER_FOOT),
                           ADE=float(rec["dist"][:h].sum() / ade_denom * METRES_PER_FOOT),
                           FDE=float(rec["dist"][h-1] / count[h-1] * METRES_PER_FOOT),
                           minADE="", minFDE="", num_windows=rec["windows"], num_vehicles=1)
                rows.append(row)
        for key, rec in self.scalar.items():
            group, recording, segment, vehicle, protocol, rule, k = key
            value = rec["sum"] / max(rec["count"], 1)
            rows.append(dict(self.fixed, group=group, recording_id=recording,
                             dataset_segment=segment, vehicle_id=vehicle,
                             protocol=protocol, selection_rule=rule, K=k, horizon=5.0,
                             RMSE="", ADE="", FDE="",
                             minADE=value if rule == "direct_minADE" else "",
                             minFDE=value if rule == "direct_minFDE" else "",
                             num_windows=rec["windows"], num_vehicles=1))
        return rows


RESULT_COLUMNS = [
    "dataset", "model", "variant", "seed", "protocol", "selection_rule",
    "K", "NFE", "horizon", "RMSE", "ADE", "FDE", "minADE", "minFDE",
    "num_windows", "num_vehicles", "checkpoint", "recording_id",
    "dataset_segment", "vehicle_id", "group",
]
