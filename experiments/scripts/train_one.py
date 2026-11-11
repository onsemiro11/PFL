#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.common import atomic_json, dataset_paths, load_config


def slug(value):
    return value.lower().replace("=", "").replace(" ", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", choices=("ngsim", "highd"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--init-from", default="",
                    help="PFL checkpoint used only to initialise model weights")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    if args.seed not in cfg["statistics"]["seeds"]:
        raise ValueError("seed is outside the preregistered seed list")
    paths = dataset_paths(args.dataset)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    name = args.name or "fair_{}_{}_{}_seed{}".format(
        args.dataset, cfg["model"].lower(), slug(cfg["variant"]), args.seed)
    t = cfg["train"]

    if cfg["model"] == "PFL":
        cmd = [
            "/home/hyundo/miniconda3/envs/epn/bin/python", "-u", "train.py",
            "--name", name, "--train_set", str(paths["train"]),
            "--val_set", str(paths["val"]), "--seed", str(args.seed),
            "--batch_size", str(t["batch_size"]), "--num_workers", "4",
            "--log_every", str(t.get("log_every", 500)),
            "--learning_rate", str(t["learning_rate"]),
            "--train_epochs", str(t["max_epochs"]), "--min_epochs", str(t["min_epochs"]),
            "--early_stopping_patience", str(t["early_stopping_patience"]),
            "--val_batches", str(t["validation_batches"]),
            "--val_samples", str(t["validation_samples"]),
            "--sample_steps", str(cfg["evaluation"]["NFE"]),
            "--flow_weight", str(t.get("flow_weight", 1)),
            "--mean_weight", str(t.get("mean_weight", 0)),
            "--aux_weight", str(cfg["model_args"]["aux_weight"]),
            "--vf_dropout", str(cfg["model_args"]["vf_dropout"]),
            "--weight_decay", str(cfg["model_args"]["weight_decay"]),
            "--val_every", "0", "--epoch_validation", "--fixed_validation",
        ]
        if t.get("deterministic_algorithms", False):
            cmd.append("--deterministic")
        if cfg["model_args"].get("mean_head", False):
            cmd.append("--use_mean_head")
        if args.init_from:
            init_path = Path(args.init_from).resolve()
            if not init_path.is_file():
                raise FileNotFoundError(init_path)
            cmd.extend(["--init_from", str(init_path)])
        if not cfg["model_args"]["use_planning"]:
            cmd.append("--use_planning")
        if not cfg["model_args"]["use_relational"]:
            cmd.append("--use_relational")
        if not cfg["model_args"]["use_attention"]:
            cmd.append("--use_attention")
        if cfg["model_args"].get("temporal_relational", False):
            cmd.append("--temporal_relational")
        if cfg["model_args"].get("temporal_only", False):
            cmd.append("--temporal_only")
        if cfg["model_args"].get("temporal_relation_zero", False):
            cmd.append("--temporal_relation_zero")
        if cfg["model_args"].get("relation_cross_attention", False):
            cmd.append("--relation_cross_attention")
        if cfg["model_args"].get("mode_conditioned", False):
            cmd.append("--mode_conditioned")
        relation_control = cfg["model_args"].get("relation_control", "none")
        if relation_control != "none":
            cmd.extend(["--relation_control", relation_control])
        cwd = ROOT / "PFL"
        run_dir = cwd / "trained_models" / name
    elif cfg["model"] == "EPN":
        cmd = [
            "/home/hyundo/miniconda3/envs/epn/bin/python", "-u",
            str(ROOT / "experiments/scripts/train_epn_fair.py"),
            "--config", str(config_path), "--dataset", args.dataset,
            "--seed", str(args.seed), "--name", name,
        ]
        cwd = ROOT
        run_dir = ROOT / "EPN" / "trained_models" / name
    else:
        raise ValueError("unsupported model " + cfg["model"])

    init_path = Path(args.init_from).resolve() if args.init_from else None
    manifest = {
        "protocol": "fair_comparison_v1", "config": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "dataset": args.dataset, "seed": args.seed, "name": name,
        "train_set": str(paths["train"]), "val_set": str(paths["val"]),
        "command": cmd,
        "eligible_for_fair_summary": bool(cfg.get("eligible_for_fair_summary", True)),
        "init_from": str(init_path) if init_path else "",
        "init_from_sha256": (hashlib.sha256(init_path.read_bytes()).hexdigest()
                             if init_path else ""),
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        identity = ("protocol", "config_sha256", "dataset", "seed", "name")
        if init_path:
            identity += ("init_from_sha256",)
        mismatch = [key for key in identity if previous.get(key) != manifest.get(key)]
        if mismatch:
            raise RuntimeError("refusing to resume a run with mismatched provenance: "
                               + ", ".join(mismatch))
    atomic_json(manifest_path, manifest)
    env = os.environ.copy()
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


if __name__ == "__main__":
    main()
