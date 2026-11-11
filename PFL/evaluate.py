"""Agent-based RMSE / ADE / FDE for PFL, same reporting as EPN evaluate.py.

Two estimators are reported side by side, because a flow model and EPN's Gaussian
head are not the same kind of predictor:

  * RMSE / ADE / FDE  -- from the mean of `mean_k` samples, a Monte-Carlo estimate
    of E[Y | c]. This is the same quantity EPN's mu head reports, so the numbers
    are comparable. A single sample would carry E||Y^ - Y||^2 = 2 tr(Sigma) against
    tr(Sigma), a structural sqrt(2) penalty; --single_sample shows exactly that.
  * minADE / minFDE   -- from oracle endpoint selection over `best_of_n` samples,
    matching EPN's best-of-n destination protocol. This is where the distributional
    quality of the flow actually shows up.
"""
import argparse
import logging
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import add_eval_args
from data import highwayTrajDataset
from model import PFLNet, strip_derived
from utils import initLogging, maskedTest


def _load_state(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "ema" in sd:
        sd = sd["ema"]
    elif isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    return strip_derived(sd)


class AgentStats:
    """Per-vehicle accumulation, exactly as EPN does it."""

    def __init__(self, n_ds, n_veh, out_length, device):
        shape = (n_ds, n_veh, out_length)
        self.sq = torch.zeros(shape, device=device)
        self.dist = torch.zeros(shape, device=device)
        self.count = torch.zeros(shape, device=device)

    def add(self, pred, fut, mask, ds_ids, targ_ids):
        ll, ll_r, cc = maskedTest(pred, fut, mask)
        for j, targ in enumerate(targ_ids):
            ds = ds_ids[j]
            self.sq[ds, targ, :] += ll[:, j].detach()
            self.dist[ds, targ, :] += ll_r[:, j].detach()
            self.count[ds, targ, :] += cc[:, j].detach()

    def summary(self, out_length):
        sq = self.sq.cpu().numpy()
        dist = self.dist.cpu().numpy()
        counts = self.count.cpu().numpy()
        ds_ids, veh_ids = counts[:, :, 0].nonzero()
        n = len(veh_ids)
        sq_avg = np.zeros((out_length, n))
        d_avg = np.zeros((out_length, n))
        c_avg = np.zeros((out_length, n))
        for i in range(n):
            c_avg[:, i] = counts[ds_ids[i], veh_ids[i], :].astype(bool)
            denom = counts[ds_ids[i], veh_ids[i], :] + 1e-9
            sq_avg[:, i] = sq[ds_ids[i], veh_ids[i], :] * c_avg[:, i] / denom
            d_avg[:, i] = dist[ds_ids[i], veh_ids[i], :] * c_avg[:, i] / denom
        sq_sum = np.sum(sq_avg, axis=1)
        d_sum = np.sum(d_avg, axis=1)
        c_sum = np.sum(c_avg, axis=1)
        rmse = np.power(sq_sum / c_sum, 0.5) * 0.3048
        fde = (d_sum / c_sum) * 0.3048
        ade = np.array([(np.sum(d_sum[0:h]) / np.sum(c_sum[0:h])) * 0.3048
                        for h in (5, 10, 15, 20, 25)])
        return rmse[4::5], fde[4::5], ade


def _log(tag, stats, out_length, min_metrics=False):
    rmse, fde, ade = stats.summary(out_length)
    p = "min" if min_metrics else ""
    logging.info("[%s] RMSE (m)\t=> %s, Mean=%.3f", tag, np.round(rmse, 4), rmse.mean())
    logging.info("[%s] %sFDE (m)\t=> %s, Mean=%.3f", tag, p, np.round(fde, 4), fde.mean())
    logging.info("[%s] %sADE (m)\t=> %s, Mean=%.3f", tag, p, np.round(ade, 4), ade.mean())


def evaluate():
    parser = argparse.ArgumentParser(description="PFL evaluation")
    add_eval_args(parser)
    args = parser.parse_args()

    log_dir = "./trained_models/{}/".format(args.name.split("-")[0])
    os.makedirs(log_dir, exist_ok=True)
    initLogging(log_file=log_dir + "evaluation.log")

    ckpt = "./trained_models/{}/{}.tar".format(args.name.split("-")[0], args.name)
    if not os.path.isfile(ckpt):
        ckpt = log_dir + "latest.pt"
    logging.info("Loading %s", ckpt)

    net = PFLNet(args)
    # Not strict: a checkpoint written before the mean head existed carries no weights for
    # it, and the flow columns do not use it. Anything else going missing is reported, so a
    # genuinely mismatched checkpoint is still visible rather than silently half-loaded.
    missing, unexpected = net.load_state_dict(_load_state(ckpt), strict=False)
    heads = sorted({k.split(".")[0] for k in missing})
    if heads:
        logging.warning("checkpoint has no weights for: %s (left at initialisation)",
                        ", ".join(heads))
    if unexpected:
        logging.warning("checkpoint carries unused tensors: %s",
                        ", ".join(sorted({k.split(".")[0] for k in unexpected})))
    if args.use_cuda:
        net = net.cuda()
    net.eval()
    if not bool(net.stats_ready):
        logging.warning("checkpoint carries no fitted residual statistics; "
                        "predictions will be de-normalised")

    ts_set = highwayTrajDataset(path=args.test_set,
                                targ_enc_size=args.social_context_size + args.dynamics_encoding_size,
                                grid_size=args.grid_size, fit_plan_traj=True,
                                fit_plan_further_ds=args.plan_info_ds)
    loader = DataLoader(ts_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=ts_set.collate_fn)
    k = 1 if args.single_sample else max(args.mean_k, args.best_of_n, args.cond_draws)
    logging.info("test %s | mean_k=%s best_of_n=%s draws=%s plan=%s rel=%s",
                 len(ts_set), args.mean_k, args.best_of_n, k,
                 args.use_planning, args.use_relational)

    dev = "cuda" if args.use_cuda else "cpu"
    n_ds = np.max(ts_set.Data[:, 0]).astype(int) + 1
    n_veh = np.max(ts_set.Data[:, 13:(13 + ts_set.grid_cells)]).astype(int) + 1
    mean_stats = AgentStats(n_ds, n_veh, args.out_length, dev)
    best_stats = AgentStats(n_ds, n_veh, args.out_length, dev)
    # Endpoint-conditioned mean: the estimator EPN gets for free from a deterministic
    # decoder, swept over kernel bandwidths because the right one is not known a priori.
    # each entry is "beta" or "beta:gamma", gamma being the T/t bandwidth exponent
    betas = []
    for spec in (args.cond_beta.split(",") if args.cond_beta else []):
        b, _, g = spec.partition(":")
        betas.append((float(b), float(g) if g else 0.0))
    cond_stats = {b: AgentStats(n_ds, n_veh, args.out_length, dev) for b in betas}

    avg_t = 0.0
    with torch.no_grad():
        for i, data in enumerate(loader):
            if args.max_batches and i >= args.max_batches:
                break
            st = time.time()
            (nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask, targsFut,
             targsFutMask, lat_enc, lon_enc, idxs, nbsVel, nbsAcc, nbsColl,
             vel_hist, acc_hist, coll_hist) = data
            if args.use_cuda:
                nbsHist, nbsMask, planFut, planMask = nbsHist.cuda(), nbsMask.cuda(), planFut.cuda(), planMask.cuda()
                targsHist, targsEncMask = targsHist.cuda(), targsEncMask.cuda()
                targsFut, targsFutMask = targsFut.cuda(), targsFutMask.cuda()
                lat_enc, lon_enc = lat_enc.cuda(), lon_enc.cuda()
                nbsVel, nbsAcc, nbsColl = nbsVel.cuda(), nbsAcc.cuda(), nbsColl.cuda()
                vel_hist, acc_hist, coll_hist = vel_hist.cuda(), acc_hist.cuda(), coll_hist.cuda()

            ctx = net.encode(nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask,
                             lat_enc, lon_enc, nbsVel, nbsAcc, nbsColl,
                             vel_hist, acc_hist, coll_hist, None)
            samples = net.sample(ctx, targsHist, n_steps=args.sample_steps, n_samples=k)

            ds_ids, targs_ids = ts_set.batchTargetVehsInfo(idxs)
            mean_pred = net.mean_prediction(samples[:args.mean_k]).permute(1, 0, 2)
            mean_stats.add(mean_pred, targsFut, targsFutMask, ds_ids, targs_ids)
            if not args.single_sample:
                sel = samples[:args.best_of_n]
                best = net.pick_best(sel, targsFut).permute(1, 0, 2)
                best_stats.add(best, targsFut, targsFutMask, ds_ids, targs_ids)
                if betas:
                    # Destination chosen from the same n draws EPN would use; the
                    # conditional mean is then taken over every draw available.
                    d_star = net.pick_best_dest(sel, targsFut)
                    for b in betas:
                        cm = net.cond_mean(samples, d_star, *b).permute(1, 0, 2)
                        cond_stats[b].add(cm, targsFut, targsFutMask, ds_ids, targs_ids)

            avg_t += time.time() - st
            if i % 100 == 99:
                total = args.max_batches or (len(ts_set) / args.batch_size)
                eta = avg_t / 100 * (total - i)
                logging.info("Evaluation progress(%%):%.2f | ETA(s):%s",
                             i / total * 100, int(eta))
                avg_t = 0.0
                # Every column, not just the mean one: a long sweep that dies partway
                # should still leave the conditional numbers on disk.
                _log("mean-%d" % args.mean_k, mean_stats, args.out_length)
                if not args.single_sample:
                    _log("best-of-%d" % args.best_of_n, best_stats, args.out_length,
                         min_metrics=True)
                    for b in betas:
                        _log("cond-mean-%d/%d beta=%g gamma=%g" % (args.best_of_n, k, b[0], b[1]),
                             cond_stats[b], args.out_length, min_metrics=True)

    tag = "single-sample" if args.single_sample else "mean-%d" % args.mean_k
    _log(tag, mean_stats, args.out_length)
    if not args.single_sample:
        _log("best-of-%d" % args.best_of_n, best_stats, args.out_length, min_metrics=True)
        for b in betas:
            _log("cond-mean-%d/%d beta=%g gamma=%g" % (args.best_of_n, k, b[0], b[1]),
                 cond_stats[b], args.out_length, min_metrics=True)


if __name__ == "__main__":
    evaluate()
