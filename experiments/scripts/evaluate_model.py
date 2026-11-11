#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys
import time
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.common import (
    HORIZONS, METRES_PER_FOOT, RESULT_COLUMNS, VehicleAccumulator, atomic_json,
    dataset_paths, direct_min_metrics, gt_free_mean, gt_free_medoid, load_config,
    full_horizon_subset, oracle_endpoint, parameter_counts, seed_everything, system_metadata,
    target_metadata, write_csv,
)


def build_pfl(cfg, device):
    sys.path.insert(0, str(ROOT / "PFL"))
    from config import add_encoder_args
    from model import PFLNet, strip_derived
    parser = ArgumentParser(add_help=False)
    add_encoder_args(parser)
    args = parser.parse_args([])
    m = cfg["model_args"]
    args.use_planning = m["use_planning"]
    args.use_relational = m["use_relational"]
    args.use_attention = m["use_attention"]
    args.order = m["order"]
    args.relation_control = m.get("relation_control", "none")
    # Legacy checkpoints may contain a mean head even when the fair metric below
    # deliberately uses flow samples only. Constructing it permits strict loading;
    # no prediction path in this evaluator reads it.
    args.use_mean_head = bool(m.get("mean_head", False))
    args.mean_weight = 0.0
    args.vf_dropout = m["vf_dropout"]
    args.temporal_relational = m.get("temporal_relational", False)
    args.temporal_only = m.get("temporal_only", False)
    args.temporal_relation_zero = m.get("temporal_relation_zero", False)
    args.relation_cross_attention = m.get("relation_cross_attention", False)
    args.relation_attention_heads = m.get("relation_attention_heads", 4)
    args.mode_conditioned = m.get("mode_conditioned", False)
    model = PFLNet(args)
    return model.to(device), args, strip_derived


def epn_args(cfg):
    m = cfg["model_args"]
    return Namespace(
        use_cuda=torch.cuda.is_available(), use_planning=m["use_planning"],
        use_fusion=m["use_fusion"], use_attention=m["use_attention"],
        train_output_flag=False, grid_size=[25, 5], in_length=16, out_length=25,
        num_lat_classes=3, num_lon_classes=2, temporal_embedding_size=32,
        encoder_size=64, decoder_size=128, soc_conv_depth=64, soc_conv2_depth=16,
        dynamics_encoding_size=32, social_context_size=80, fuse_enc_size=112,
        dest_dec_size=[128, 64, 32], dest_latent_size=[8, 50],
        dest_enc_size=[8, 16], zdim=16, fdim=16, sigma=m["sigma"],
        order=m["order"], best_of_n=20,
    )


def load_verified(model, checkpoint, cfg_path, dataset, seed, transform=lambda x: x,
                  allow_unverified=False):
    checkpoint = Path(checkpoint).resolve()
    manifest_path = checkpoint.parent / "run_manifest.json"
    if not allow_unverified:
        if not manifest_path.is_file():
            raise RuntimeError("checkpoint has no fair-run manifest: " + str(checkpoint))
        manifest = json.loads(manifest_path.read_text())
        digest = hashlib.sha256(Path(cfg_path).read_bytes()).hexdigest()
        checks = {
            "config_sha256": digest, "dataset": dataset, "seed": seed,
            "eligible_for_fair_summary": True,
        }
        for key, expected in checks.items():
            if manifest.get(key) != expected:
                raise RuntimeError("checkpoint manifest mismatch for %s" % key)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "ema" in state:
        state = state["ema"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(transform(state), strict=True)
    return checkpoint


def move(batch, device):
    return [x.to(device) if torch.is_tensor(x) else x for x in batch]


def unpack(batch):
    (nbs_hist, nbs_mask, plan_fut, plan_mask, targ_hist, targ_enc_mask,
     future, future_mask, lat, lon, idxs, nbs_vel, nbs_acc, nbs_coll,
     vel_hist, acc_hist, coll_hist) = batch
    enc = (nbs_hist, nbs_mask, plan_fut, plan_mask, targ_hist, targ_enc_mask,
           lat, lon, nbs_vel, nbs_acc, nbs_coll, vel_hist, acc_hist, coll_hist)
    return enc, targ_hist, future, future_mask, idxs


class DistributionStats:
    def __init__(self, ks, radii_m):
        self.ks = ks
        self.radii_m = radii_m
        self.values = {k: defaultdict(float) for k in ks}

    def add(self, samples, future, mask, k):
        s = samples[:k]
        batch, steps = s.shape[1:3]
        std = s.std(0, unbiased=k > 1)
        v = self.values[k]
        v["lateral_spread_sum"] += std[..., 0].sum().item() * METRES_PER_FOOT
        v["longitudinal_spread_sum"] += std[..., 1].sum().item() * METRES_PER_FOOT
        v["spread_points"] += batch * steps
        endpoint = s[:, :, -1]
        end_mean = endpoint.mean(0)
        end_spread = torch.linalg.vector_norm(endpoint - end_mean, dim=-1).mean(0)
        v["endpoint_spread_sum"] += end_spread.sum().item() * METRES_PER_FOOT
        pair_sum, pairs = 0.0, 0
        for i in range(k):
            for j in range(i + 1, k):
                pair_sum += torch.linalg.vector_norm(s[i] - s[j], dim=-1).mean().item()
                pairs += 1
        v["pairwise_sum"] += pair_sum * batch * METRES_PER_FOOT
        v["pairwise_batches"] += pairs * batch
        gt = future.permute(1, 0, 2)
        valid = mask.permute(1, 0, 2)[..., 0]
        mean_error = (((s.mean(0) - gt).square().sum(-1) * valid).sum(-1)
                      / valid.sum(-1).clamp_min(1)).sqrt()
        spread_rms = std.square().sum(-1).mean(-1).sqrt()
        v["spread_error_ratio_sum"] += (spread_rms / mean_error.clamp_min(1e-6)).sum().item()
        v["targets"] += batch
        endpoint_error_m = torch.linalg.vector_norm(endpoint - gt[:, -1].unsqueeze(0), dim=-1).min(0).values * METRES_PER_FOOT
        for radius in self.radii_m:
            v["coverage_" + str(radius)] += (endpoint_error_m <= radius).sum().item()

    def summary(self):
        out = {}
        for k, v in self.values.items():
            points = max(v["spread_points"], 1)
            targets = max(v["targets"], 1)
            pairs = max(v["pairwise_batches"], 1)
            out[str(k)] = {
                "lateral_sample_spread_m": v["lateral_spread_sum"] / points,
                "longitudinal_sample_spread_m": v["longitudinal_spread_sum"] / points,
                "average_pairwise_trajectory_distance_m": v["pairwise_sum"] / pairs,
                "endpoint_spread_m": v["endpoint_spread_sum"] / targets,
                "spread_error_ratio": v["spread_error_ratio_sum"] / targets,
                "empirical_endpoint_coverage": {
                    str(r): v["coverage_" + str(r)] / targets for r in self.radii_m
                },
            }
        return out


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in (
            "dataset", "model", "variant", "seed", "protocol", "selection_rule",
            "K", "NFE", "horizon", "checkpoint", "group"))
        groups[key].append(row)
    result = []
    for key, members in groups.items():
        base = dict(zip(("dataset", "model", "variant", "seed", "protocol",
                         "selection_rule", "K", "NFE", "horizon", "checkpoint", "group"), key))
        def vals(field):
            return np.array([float(x[field]) for x in members if x[field] != ""])
        rmse, ade, fde = vals("RMSE"), vals("ADE"), vals("FDE")
        mina, minf = vals("minADE"), vals("minFDE")
        base.update({
            "RMSE": float(np.sqrt(np.mean(rmse ** 2))) if len(rmse) else "",
            "ADE": float(np.mean(ade)) if len(ade) else "",
            "FDE": float(np.mean(fde)) if len(fde) else "",
            "minADE": float(np.mean(mina)) if len(mina) else "",
            "minFDE": float(np.mean(minf)) if len(minf) else "",
            "num_windows": sum(int(x["num_windows"]) for x in members),
            "num_vehicles": len({(x["dataset_segment"], x["vehicle_id"]) for x in members}),
            "recording_id": "", "dataset_segment": "", "vehicle_id": "",
        })
        result.append(base)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("pfl", "epn", "cv"), required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", choices=("ngsim", "highd"), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--test-set", default="", help="override only for smoke testing")
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--allow-unverified-checkpoint", action="store_true")
    ap.add_argument("--deterministic-algorithms", action="store_true",
                    help="force deterministic CUDA kernels (supported but very slow for PFL Conv1d)")
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    if args.model != "cv" and cfg["model"].lower() != args.model:
        raise ValueError("config/model mismatch")
    if args.seed not in cfg["statistics"]["seeds"]:
        raise ValueError("seed outside preregistered list")
    seed_everything(args.seed, deterministic=args.deterministic_algorithms)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "pfl":
        model, model_args, transform = build_pfl(cfg, device)
        checkpoint = load_verified(model, args.checkpoint, cfg_path, args.dataset,
                                   args.seed, transform, args.allow_unverified_checkpoint)
        from data import highwayTrajDataset
    elif args.model == "epn":
        sys.path.insert(0, str(ROOT / "EPN"))
        from data import highwayTrajDataset
        from model import EPNet
        model_args = epn_args(cfg)
        model = EPNet(model_args).to(device)
        checkpoint = load_verified(model, args.checkpoint, cfg_path, args.dataset,
                                   args.seed, allow_unverified=args.allow_unverified_checkpoint)
        model.train_output_flag = False
    else:
        sys.path.insert(0, str(ROOT / "PFL"))
        from data import highwayTrajDataset
        from encoder import cv_baseline
        model, model_args, checkpoint = None, None, ""

    test_path = Path(args.test_set).resolve() if args.test_set else dataset_paths(args.dataset)["test"]
    test_set = highwayTrajDataset(str(test_path), targ_enc_size=112, grid_size=[25, 5],
                                  fit_plan_traj=True,
                                  fit_plan_further_ds=cfg["evaluation"]["plan_info_ds"])
    loader = DataLoader(test_set, batch_size=cfg["train"]["batch_size"], shuffle=False,
                        num_workers=4, collate_fn=test_set.collate_fn,
                        persistent_workers=True)
    if model is not None:
        model.eval()
    model_name = "CV" if args.model == "cv" else cfg["model"]
    variant = "CV" if args.model == "cv" else cfg["variant"]
    nfe = 0 if args.model == "cv" else cfg["evaluation"]["NFE"]
    acc = VehicleAccumulator(args.dataset, model_name, variant, args.seed,
                             checkpoint, nfe)
    dist_stats = DistributionStats(cfg["evaluation"]["K"], [0.5, 1.0, 2.0, 3.0, 5.0])
    coverage = {}
    elapsed_encode = elapsed_sample = 0.0
    k6_seconds = k6_targets = k6_scenes = 0.0
    scenes = targets = full_horizon_targets = processed_batches = 0

    with torch.no_grad():
        for batch_i, raw in enumerate(loader):
            if args.max_batches and batch_i >= args.max_batches:
                break
            batch = move(raw, device)
            enc_inputs, target_hist, future, mask, idxs = unpack(batch)
            metadata = target_metadata(test_set, idxs, args.dataset,
                                       cfg["evaluation"]["interaction_threshold_m"])
            if len(metadata) != target_hist.shape[1]:
                raise RuntimeError("target metadata order/count mismatch")
            if device.type == "cuda":
                torch.cuda.synchronize()
            begin = time.perf_counter()
            if args.model == "pfl":
                context = model.encode(*enc_inputs, None)
            elif args.model == "epn":
                context = model.soc_encode(*enc_inputs, None)
            else:
                context = None
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_encode += time.perf_counter() - begin

            if args.model == "cv":
                gate = target_hist.new_tensor([0.0, 1.0])
                pred = cv_baseline(target_hist, 25, gate)
                acc.add_prediction(pred, future, mask, metadata, "gt_free", "constant_velocity", 0)
            else:
                torch.manual_seed(args.seed * 1000003 + batch_i)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                begin = time.perf_counter()
                if args.model == "pfl":
                    samples = model.sample(context, target_hist,
                                           n_steps=cfg["evaluation"]["NFE"], n_samples=20,
                                           chunk=cfg["evaluation"].get("sample_chunk", 20))
                else:
                    generated, _ = model.generate_trajectories(context, 20)
                    samples = generated[..., :2]
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed_sample += time.perf_counter() - begin

                # Measure K=6 directly on a fixed ten-batch prefix. It is a runtime
                # benchmark only; metric candidates remain the first six of the K=20
                # draw so K is nested and comparable.
                if batch_i < 10:
                    torch.manual_seed(args.seed * 1000003 + batch_i + 50000000)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    k6_begin = time.perf_counter()
                    if args.model == "pfl":
                        model.sample(context, target_hist,
                                     n_steps=cfg["evaluation"]["NFE"], n_samples=6,
                                     chunk=min(6, cfg["evaluation"].get("sample_chunk", 20)))
                    else:
                        model.generate_trajectories(context, 6)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    k6_seconds += time.perf_counter() - k6_begin
                    k6_targets += len(metadata)
                    k6_scenes += len(idxs)

                for k in cfg["evaluation"]["K"]:
                    selected = samples[:k]
                    acc.add_prediction(gt_free_mean(selected), future, mask, metadata,
                                       "gt_free", "sample_mean", k)
                    acc.add_prediction(gt_free_medoid(selected), future, mask, metadata,
                                       "gt_free", "sample_medoid", k)
                    full_samples, full_future, full_mask, full_metadata = full_horizon_subset(
                        selected, future, mask, metadata)
                    if full_metadata:
                        oracle = oracle_endpoint(full_samples, full_future[-1])
                        acc.add_prediction(oracle, full_future, full_mask, full_metadata,
                                           "oracle", "gt_endpoint_nearest_trajectory", k)
                        min_ade, min_fde = direct_min_metrics(
                            full_samples, full_future, full_mask)
                        acc.add_scalar(min_ade, full_metadata, "oracle_coverage",
                                       "direct_minADE", k)
                        acc.add_scalar(min_fde, full_metadata, "oracle_coverage",
                                       "direct_minFDE", k)
                        if args.model == "pfl":
                            dist_stats.add(full_samples, full_future, full_mask, k)
                    if k == cfg["evaluation"]["K"][0]:
                        full_horizon_targets += len(full_metadata)
            scenes += len(idxs)
            targets += len(metadata)
            processed_batches += 1
            if processed_batches % 500 == 0:
                elapsed = elapsed_encode + elapsed_sample
                print("evaluation progress batches={} scenes={} targets={} model_seconds={:.1f}".format(
                    processed_batches, scenes, targets, elapsed), flush=True)

    rows = acc.rows()
    summary = aggregate(rows)
    stem = "{}_{}_seed{}".format(args.model, variant.lower().replace("=", ""), args.seed)
    is_smoke = bool(args.test_set or args.max_batches)
    base = (ROOT / "experiments" / "results" / "smoke" / args.dataset
            if is_smoke else ROOT / "experiments" / "results" / args.dataset)
    per_vehicle_path = base / "per_vehicle" / (stem + ".csv")
    per_seed_path = base / "per_seed" / (stem + ".csv")
    metadata_path = base / "per_seed" / (stem + ".json")
    write_csv(per_vehicle_path, rows, RESULT_COLUMNS)
    write_csv(per_seed_path, summary, RESULT_COLUMNS)
    latency = {
        "encoding_seconds": elapsed_encode, "sampling_seconds": elapsed_sample,
        "scene_latency_ms": 1000 * (elapsed_encode + elapsed_sample) / max(scenes, 1),
        "target_latency_ms": 1000 * (elapsed_encode + elapsed_sample) / max(targets, 1),
        "scenes": scenes, "targets": targets, "batches": processed_batches,
        "full_horizon_targets": full_horizon_targets,
        "K_generated": 0 if args.model == "cv" else 20, "NFE": nfe,
        "generation_by_K": ({
            "6": {"seconds": k6_seconds,
                  "scene_latency_ms": 1000 * k6_seconds / max(k6_scenes, 1),
                  "target_latency_ms": 1000 * k6_seconds / max(k6_targets, 1),
                  "benchmark_batches": min(processed_batches, 10)},
            "20": {"seconds": elapsed_sample,
                   "scene_latency_ms": 1000 * elapsed_sample / max(scenes, 1),
                   "target_latency_ms": 1000 * elapsed_sample / max(targets, 1),
                   "benchmark_batches": processed_batches},
        } if args.model != "cv" else {}),
    }
    metadata_out = {
        "protocol_version": "fair_comparison_v1", "dataset": args.dataset,
        "metric_revision": "full_horizon_mask_v2",
        "model": model_name, "variant": variant, "seed": args.seed,
        "checkpoint": str(checkpoint), "complete_test": not is_smoke,
        "result_scope": "smoke" if is_smoke else "official_test",
        "deterministic_algorithms": args.deterministic_algorithms,
        "parameter_count": {"total": 0, "trainable": 0} if model is None else parameter_counts(model),
        "runtime": latency, "system": system_metadata(device),
        "selection_definitions": {
            "sample_mean": "arithmetic mean of K trajectories; not a sampled trajectory; GT-free",
            "sample_medoid": "real candidate minimizing summed pairwise trajectory distance; GT-free",
            "gt_endpoint_nearest_trajectory": "candidate selected by the observed GT 5-second endpoint, then full trajectory scored; oracle; targets without a valid 5-second endpoint are excluded",
            "direct_minADE": "minimum ADE over candidates with a valid full 5-second future; may select a different candidate",
            "direct_minFDE": "minimum FDE over candidates with a valid full 5-second future; may select a different candidate",
        },
        "epn_top_confidence": ("unavailable_no_candidate_specific_confidence"
                               if args.model == "epn" else "not_applicable"),
        "distribution": dist_stats.summary() if args.model == "pfl" else {},
        "ego_plan_source": "recorded_ego_future_proxy_with_test_time_quintic_fit",
        "per_vehicle_csv": str(per_vehicle_path), "summary_csv": str(per_seed_path),
    }
    atomic_json(metadata_path, metadata_out)
    print(json.dumps(metadata_out, indent=2))


if __name__ == "__main__":
    main()
