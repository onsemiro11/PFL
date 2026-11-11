#!/usr/bin/env python3
"""Write preregistered model/ablation parameter counts without loading data."""
import json
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.common import atomic_json, parameter_counts


def pfl_counts():
    sys.path.insert(0, str(ROOT / "PFL"))
    from config import add_encoder_args
    from model import PFLNet
    variants = {
        "PLAN=0": ["--use_planning"],
        "REL=0": ["--use_relational"],
        "REL=0_capacity_control": ["--use_relational", "--relation_control", "target_cv"],
        "Full": [],
        "Global+Temporal": ["--temporal_relational"],
        "Temporal-only": ["--temporal_relational", "--temporal_only"],
        "Temporal-zero-control": ["--temporal_relational", "--temporal_relation_zero"],
    }
    output = {}
    for name, flags in variants.items():
        parser = ArgumentParser(add_help=False)
        add_encoder_args(parser)
        args = parser.parse_args(flags)
        args.mean_weight = 0.0
        model = PFLNet(args)
        output[name] = {
            **parameter_counts(model),
            "encoder": sum(p.numel() for p in model.encoder.parameters()),
            "context_dimension": model.ctx_dim,
            "mean_head": model.use_mean_head,
        }
    return output


def epn_counts():
    # PFL and EPN use top-level module names such as `model` and `encoder`.
    # Clear them before importing the second repository implementation.
    for name in ("config", "model", "encoder", "vf", "utils", "data"):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(ROOT / "EPN"))
    from model import EPNet
    args = Namespace(
        use_cuda=False, use_planning=True, use_fusion=False, use_attention=True,
        train_output_flag=False, grid_size=[25, 5], in_length=16, out_length=25,
        num_lat_classes=3, num_lon_classes=2, temporal_embedding_size=32,
        encoder_size=64, decoder_size=128, soc_conv_depth=64, soc_conv2_depth=16,
        dynamics_encoding_size=32, social_context_size=80, fuse_enc_size=112,
        dest_dec_size=[128, 64, 32], dest_latent_size=[8, 50],
        dest_enc_size=[8, 16], zdim=16, fdim=16, sigma=1.3, order=3,
        best_of_n=20,
    )
    return parameter_counts(EPNet(args))


def main():
    counts = {"PFL": pfl_counts(), "EPN": epn_counts()}
    full = counts["PFL"]["Full"]["total"]
    rel0 = counts["PFL"]["REL=0"]["total"]
    counts["relational_parameter_difference"] = {
        "parameters": full - rel0,
        "fraction_of_full": (full - rel0) / full,
        "main_matrix_uses_capacity_control": False,
        "reason": "the difference is below one percent; the exact-size optional control remains available",
    }
    path = ROOT / "experiments/results/parameter_counts.json"
    atomic_json(path, counts)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
