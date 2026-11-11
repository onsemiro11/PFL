"""Random-tensor smoke test for PFL (no .mat required)."""
import argparse

import torch

from config import add_encoder_args
from model import PFLNet
from vf import VectorField


def _dummy_batch(args, device, b=4, n_nbr=6):
    t, tf, d = args.in_length, args.out_length, args.encoder_size
    nbs_hist = torch.randn(t, n_nbr, 2, device=device)
    nbs_mask = torch.zeros(b, 5, 25, d, dtype=torch.bool, device=device)
    nbs_mask[0, 2, 10, :] = True
    nbs_mask[0, 2, 14, :] = True
    nbs_mask[1, 1, 12, :] = True
    nbs_mask[2, 3, 8, :] = True
    nbs_mask[2, 3, 16, :] = True
    nbs_mask[3, 2, 12, :] = True
    plan_fut = torch.randn(tf, b, 2, device=device)
    plan_mask = torch.zeros_like(nbs_mask)
    plan_mask[0, 2, 12, :] = True
    plan_mask[1, 2, 12, :] = True
    plan_mask[2, 2, 12, :] = True
    plan_mask[3, 2, 12, :] = True
    targs = torch.randn(t, b, 2, device=device)
    targs_mask = torch.zeros_like(nbs_mask)
    fut = torch.randn(tf, b, 2, device=device)
    fut_mask = torch.ones(tf, b, 2, device=device)
    lat = torch.zeros(b, 3, device=device)
    lat[:, 0] = 1.0
    lon = torch.zeros(b, 2, device=device)
    lon[:, 0] = 1.0
    nbs_vel = torch.randn(t, n_nbr, 1, device=device)
    nbs_acc = torch.randn(t, n_nbr, 1, device=device)
    nbs_coll = torch.zeros(t, n_nbr, 1, device=device)
    vel = torch.randn(t, b, 1, device=device)
    acc = torch.randn(t, b, 1, device=device)
    coll = torch.zeros(t, b, 1, device=device)
    return (nbs_hist, nbs_mask, plan_fut, plan_mask, targs, targs_mask, fut, fut_mask,
            lat, lon, nbs_vel, nbs_acc, nbs_coll, vel, acc, coll)


def main():
    parser = argparse.ArgumentParser()
    add_encoder_args(parser)
    args = parser.parse_args([])
    args.use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.use_cuda else "cpu")

    vf = VectorField().to(device)
    xt = torch.randn(4, 25, 2, device=device)
    t = torch.rand(4, device=device)
    ctx = torch.randn(4, 112, device=device)
    v = vf(xt, t, ctx)
    assert v.shape == xt.shape, v.shape

    net = PFLNet(args).to(device)
    batch = _dummy_batch(args, device)
    (nbs_hist, nbs_mask, plan_fut, plan_mask, targs, targs_mask, fut, fut_mask,
     lat, lon, nbs_vel, nbs_acc, nbs_coll, vel, acc, coll) = batch
    enc = net.encode(nbs_hist, nbs_mask, plan_fut, plan_mask, targs, targs_mask,
                     lat, lon, nbs_vel, nbs_acc, nbs_coll, vel, acc, coll, None)
    assert enc.shape == (4, 112), enc.shape
    loss = net.flow_loss(enc, fut, fut_mask, targs) + net.maneuver_loss(enc, lat, lon)
    loss.backward()
    # The relational branch is zero-initialised, so it must reach a gradient without
    # yet moving the encoder output.
    assert net.encoder.rel_ego.mlp[-1].weight.grad is not None, "relational branch got no gradient"
    samples = net.sample(enc, targs, n_steps=4, n_samples=3)
    mean = net.mean_prediction(samples)
    best = net.pick_best(samples, fut)
    assert samples.shape == (3, 4, 25, 2), samples.shape
    assert mean.shape == (4, 25, 2), mean.shape
    assert best.shape == (4, 25, 2), best.shape

    # Round-tripping the normalisation must be exact.
    net.res_mean.normal_()
    net.res_std.uniform_(0.5, 2.0)
    rt = net.denormalize(net.normalize(fut, targs), targs) - fut.permute(1, 0, 2)
    assert rt.abs().max() < 1e-3, rt.abs().max().item()

    # Combined relation-cross-attention + maneuver-conditioned trajectory flow.
    args.temporal_relational = True
    args.relation_cross_attention = True
    args.mode_conditioned = True
    combined = PFLNet(args).to(device)
    condition = combined.encode(nbs_hist, nbs_mask, plan_fut, plan_mask, targs,
                                targs_mask, lat, lon, nbs_vel, nbs_acc, nbs_coll,
                                vel, acc, coll, None)
    combined_loss = combined.flow_loss(condition, fut, fut_mask, targs, lat, lon)
    combined_loss.backward()
    adapters = [m for m in combined.modules() if m.__class__.__name__ == "RelationCrossAttention"]
    assert len(adapters) == 12, len(adapters)
    assert all(a.out_proj.weight.grad is not None for a in adapters)
    assert combined.vf.ce.mode_proj.weight.grad is not None
    combined_samples = combined.sample(condition, targs, n_steps=2, n_samples=3)
    assert combined_samples.shape == (3, 4, 25, 2)

    print("smoke ok  loss={:.4f}  enc={}  device={}".format(loss.item(), tuple(enc.shape), device))


if __name__ == "__main__":
    main()
