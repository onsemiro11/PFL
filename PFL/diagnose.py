"""Collapse / NFE diagnostics for a trained PFL checkpoint.

Two questions, one script:

  * has the flow collapsed onto E[Y|c]?  The spread across k draws is compared
    both to the marginal residual std and to the error the mean of those draws
    actually makes (see train.log_diag).
  * is the ODE doing any work?  RMSE and spread are swept over the number of
    Euler steps. A rectified flow that still bends needs its steps; one that has
    degenerated into a straight shot from noise to the conditional mean gives the
    same answer at 2 steps as at 50, and that invariance is the signature.
"""
import argparse
import logging
import os

import torch
from torch.utils.data import DataLoader

from config import add_eval_args
from data import highwayTrajDataset
from model import PFLNet
from train import validate, log_diag
from utils import initLogging


def _load_state(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "ema" in sd:
        return sd["ema"]
    if isinstance(sd, dict) and "model" in sd:
        return sd["model"]
    return sd


def _clean(sd):
    from model import strip_derived
    return strip_derived(sd)


def main():
    parser = argparse.ArgumentParser(description="PFL collapse / NFE diagnostics")
    add_eval_args(parser)
    parser.add_argument("--nfe", type=int, nargs="*", default=[2, 5, 10, 20, 50])
    parser.add_argument("--diag_k", type=int, default=20,
                        help="draws used for the spread estimate")
    parser.add_argument("--val_batches", type=int, default=20)
    parser.add_argument("--ckpt", type=str, default="")
    args = parser.parse_args()

    log_dir = "./trained_models/{}/".format(args.name.split("-")[0])
    os.makedirs(log_dir, exist_ok=True)
    initLogging(log_file=log_dir + "diagnose.log")

    ckpt = args.ckpt or "./trained_models/{}/{}.tar".format(args.name.split("-")[0], args.name)
    if not os.path.isfile(ckpt):
        ckpt = log_dir + "latest.pt"
    logging.info("Loading %s", ckpt)

    net = PFLNet(args)
    missing, _ = net.load_state_dict(_clean(_load_state(ckpt)), strict=False)
    if missing:
        logging.warning("checkpoint has no weights for: %s",
                        ", ".join(sorted({k.split(".")[0] for k in missing})))
    if args.use_cuda:
        net = net.cuda()
    net.eval()
    if not bool(net.stats_ready):
        logging.warning("checkpoint carries no fitted residual statistics")

    ds = highwayTrajDataset(path=args.test_set,
                            targ_enc_size=args.social_context_size + args.dynamics_encoding_size,
                            grid_size=args.grid_size, fit_plan_traj=True,
                            fit_plan_further_ds=args.plan_info_ds)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=ds.collate_fn)

    # The same batches feed every NFE so the sweep is not confounded by which
    # scenes each setting happened to see; shuffle is off for the same reason.
    logging.info("marginal residual std (ft) @5s  lat %.3f  lon %.3f",
                 net.res_std[-1, 0].item(), net.res_std[-1, 1].item())
    logging.info("NFE sweep on %s batches, k=%s draws", args.val_batches, args.diag_k)
    # RMSE drift alone under-detects a collapse: the mean of k draws flattens the
    # distribution anyway, so even a healthy flow can be nearly NFE-invariant in
    # RMSE. The spread is the sharper signal -- a coarser integration transports
    # less of the variance, so a live ODE shows the spread moving with NFE while a
    # degenerate one that just pushes noise onto the conditional mean does not.
    base = None
    for n in args.nfe:
        res = validate(net, loader, args, n_steps=n, n_samples=args.diag_k)
        rmse = log_diag("nfe=%d" % n, net, res, args.diag_k)
        sd = res[1]
        if base is None:
            base = (rmse, sd)
        d = (sd / base[1].clamp(min=1e-9) - 1.0).mean(0) * 100.0
        logging.info("           vs NFE=%d: RMSE %+.4f m (%+.2f%%) | "
                     "spread lat %+.2f%% lon %+.2f%%",
                     args.nfe[0], rmse - base[0],
                     100.0 * (rmse - base[0]) / max(base[0], 1e-9),
                     d[0].item(), d[1].item())


if __name__ == "__main__":
    main()
