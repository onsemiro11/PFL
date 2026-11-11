import inspect
import sys
import unittest
from argparse import ArgumentParser
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "PFL"))

from experiments.common import (
    VehicleAccumulator, full_horizon_subset, gt_free_mean, gt_free_medoid,
    load_config, oracle_endpoint,
)
from config import add_encoder_args
from model import PFLNet
from vf import VectorField


class ProtocolTest(unittest.TestCase):
    def args(self, extra=()):
        parser = ArgumentParser()
        add_encoder_args(parser)
        value = parser.parse_args(list(extra))
        value.mean_weight = 0.0
        return value

    def test_configs_preregister_no_mean_head(self):
        for name in ("pfl_plan0.yaml", "pfl_rel0.yaml", "pfl_full.yaml"):
            cfg = load_config(ROOT / "experiments/configs" / name)
            self.assertFalse(cfg["model_args"]["mean_head"])
            self.assertEqual(cfg["statistics"]["seeds"], [0, 1, 2])
            self.assertEqual(cfg["statistics"]["equivalence_margin"], 0.03)

    def test_pfl_variants_keep_context_width_and_no_mean_head(self):
        variants = {
            "PLAN=0": ("--use_planning",),
            "REL=0": ("--use_relational",),
            "Full": (),
        }
        counts = {}
        for name, flags in variants.items():
            model = PFLNet(self.args(flags))
            self.assertEqual(model.ctx_dim, 112)
            self.assertFalse(model.use_mean_head)
            self.assertFalse(hasattr(model, "mean_head"))
            counts[name] = sum(p.numel() for p in model.parameters())
        self.assertGreater(counts["Full"], counts["REL=0"])
        self.assertGreater(counts["REL=0"], counts["PLAN=0"])

    def test_capacity_control_matches_relation_encoder_parameter_count(self):
        full = PFLNet(self.args())
        control = PFLNet(self.args(("--use_relational", "--relation_control", "target_cv")))
        full_rel = sum(p.numel() for p in full.encoder.rel_ego.parameters())
        control_rel = sum(p.numel() for p in control.encoder.rel_control.parameters())
        self.assertEqual(full_rel, control_rel)

    def test_temporal_relation_is_injected_at_matching_horizons(self):
        vf = VectorField(ctx_dim=112, channels=[16, 32, 64], relation_dim=5)
        xt = torch.randn(2, 25, 2)
        t = torch.rand(2)
        ctx = torch.randn(2, 112)
        relation = torch.randn(2, 25, 5)
        out = vf(xt, t, ctx, relation)
        self.assertEqual(out.shape, xt.shape)
        with self.assertRaisesRegex(ValueError, "relation must have shape"):
            vf(xt, t, ctx, relation[:, :-1])

    def test_temporal_controls_have_identical_capacity(self):
        values = []
        for flags in (("--temporal_relational",),
                      ("--temporal_relational", "--temporal_only"),
                      ("--temporal_relational", "--temporal_relation_zero")):
            values.append(sum(p.numel() for p in PFLNet(self.args(flags)).parameters()))
        self.assertEqual(len(set(values)), 1)

    def test_gt_free_selectors_have_no_gt_argument(self):
        self.assertEqual(list(inspect.signature(gt_free_mean).parameters), ["samples"])
        self.assertEqual(list(inspect.signature(gt_free_medoid).parameters), ["samples"])
        samples = torch.tensor([
            [[[-1.0, 0.0], [-1.0, 0.0]]],
            [[[0.0, 0.0], [0.0, 0.0]]],
            [[[10.0, 0.0], [10.0, 0.0]]],
        ])
        self.assertTrue(torch.equal(gt_free_medoid(samples), samples[1]))

    def test_oracle_is_explicit_and_endpoint_based(self):
        samples = torch.tensor([
            [[[-1.0, 0.0], [-1.0, 0.0]]],
            [[[2.0, 0.0], [2.0, 0.0]]],
        ])
        selected = oracle_endpoint(samples, torch.tensor([[1.8, 0.0]]))
        self.assertTrue(torch.equal(selected, samples[1]))

    def test_endpoint_oracle_excludes_padded_futures(self):
        samples = torch.zeros(2, 2, 25, 2)
        future = torch.zeros(25, 2, 2)
        mask = torch.ones(25, 2, 2)
        mask[-1, 1] = 0
        metadata = [{"vehicle_id": 1}, {"vehicle_id": 2}]
        selected, gt, valid, kept = full_horizon_subset(
            samples, future, mask, metadata)
        self.assertEqual(selected.shape[1], 1)
        self.assertEqual(gt.shape[1], 1)
        self.assertEqual(valid.shape[1], 1)
        self.assertEqual(kept, [metadata[0]])

    def test_accumulator_does_not_emit_unobserved_horizon(self):
        acc = VehicleAccumulator("ngsim", "PFL", "Full", 0, "x", 5)
        pred = torch.zeros(1, 25, 2)
        future = torch.zeros(25, 1, 2)
        mask = torch.ones(25, 1, 2)
        mask[20:] = 0
        meta = [{"recording_id": 1, "dataset_segment": 1,
                 "vehicle_id": 1, "group": "G3_other"}]
        acc.add_prediction(pred, future, mask, meta, "gt_free", "sample_mean", 6)
        horizons = {row["horizon"] for row in acc.rows()}
        self.assertEqual(horizons, {1.0, 2.0, 3.0, 4.0})


if __name__ == "__main__":
    unittest.main()
