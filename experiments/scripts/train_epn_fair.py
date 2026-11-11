#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
import time
from argparse import Namespace
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "EPN"))

from experiments.common import atomic_json, dataset_paths, load_config, parameter_counts, seed_everything
from data import highwayTrajDataset
from model import EPNet
from utils import KL, maskedDestMSE, maskedMSE, maskedNLL


def model_args(cfg):
    m = cfg["model_args"]
    return Namespace(
        use_cuda=torch.cuda.is_available(), use_planning=m["use_planning"],
        use_fusion=m["use_fusion"], use_attention=m["use_attention"],
        train_output_flag=True, grid_size=[25, 5], in_length=16, out_length=25,
        num_lat_classes=3, num_lon_classes=2, temporal_embedding_size=32,
        encoder_size=64, decoder_size=128, soc_conv_depth=64, soc_conv2_depth=16,
        dynamics_encoding_size=32, social_context_size=80, fuse_enc_size=112,
        dest_dec_size=[128, 64, 32], dest_latent_size=[8, 50],
        dest_enc_size=[8, 16], zdim=16, fdim=16, sigma=m["sigma"],
        order=m["order"], best_of_n=20,
    )


def move(batch, device):
    return [x.to(device) if torch.is_tensor(x) else x for x in batch]


def unpack(batch):
    (nbs_hist, nbs_mask, plan_fut, plan_mask, targ_hist, targ_enc_mask,
     targ_fut, targ_mask, lat, lon, _, nbs_vel, nbs_acc, nbs_coll,
     vel_hist, acc_hist, coll_hist) = batch
    inputs = (nbs_hist, nbs_mask, plan_fut, plan_mask, targ_hist, targ_enc_mask,
              lat, lon, nbs_vel, nbs_acc, nbs_coll, vel_hist, acc_hist,
              coll_hist)
    return inputs, targ_fut, targ_mask


@torch.no_grad()
def validation_rmse(model, loader, device, batches, k, seed):
    model.eval()
    model.train_output_flag = False
    sq_sum = 0.0
    count = 0.0
    # torch.device("cuda") has index=None even though it resolves to the active
    # GPU. fork_rng requires a concrete integer device index.
    fork_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed + 100003)
        for i, batch in enumerate(loader):
            if i >= batches:
                break
            inputs, future, mask = unpack(move(batch, device))
            enc = model.soc_encode(*inputs, None)
            candidates, _ = model.generate_trajectories(enc, k)
            pred = candidates[..., :2].mean(0).permute(1, 0, 2)
            err = (pred - future).square().sum(-1) * mask[..., 0]
            sq_sum += err.sum().item()
            count += mask[..., 0].sum().item()
    return (sq_sum / max(count, 1.0)) ** 0.5 * 0.3048


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", choices=("ngsim", "highd"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--name", required=True)
    cli = ap.parse_args()
    cfg = load_config(cli.config)
    tcfg = cfg["train"]
    seed_everything(cli.seed, deterministic=tcfg.get("deterministic_algorithms", False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = model_args(cfg)
    model = EPNet(args).to(device)
    counts = parameter_counts(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["learning_rate"])
    paths = dataset_paths(cli.dataset)
    train_set = highwayTrajDataset(str(paths["train"]), targ_enc_size=112,
                                   grid_size=[25, 5], fit_plan_traj=False)
    val_set = highwayTrajDataset(str(paths["val"]), targ_enc_size=112,
                                 grid_size=[25, 5], fit_plan_traj=True)
    train_loader = DataLoader(
        train_set, batch_size=tcfg["batch_size"], shuffle=True, num_workers=4,
        collate_fn=train_set.collate_fn, persistent_workers=True,
        generator=torch.Generator().manual_seed(cli.seed))
    val_loader = DataLoader(
        val_set, batch_size=tcfg["batch_size"], shuffle=False, num_workers=4,
        collate_fn=val_set.collate_fn, persistent_workers=True)

    run = ROOT / "EPN" / "trained_models" / cli.name
    run.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        handlers=[logging.FileHandler(run / "train.log"), logging.StreamHandler()],
                        format="[%(levelname)s %(asctime)s] %(message)s")
    atomic_json(run / "args.json", {
        "config": str(Path(cli.config).resolve()), "dataset": cli.dataset,
        "seed": cli.seed, "name": cli.name, "model_args": vars(args),
        "train": tcfg, "parameter_count": counts,
        "deterministic_algorithms": tcfg.get("deterministic_algorithms", False),
    })
    logging.info("EPN fair run %s dataset=%s seed=%d parameters=%s",
                 cli.name, cli.dataset, cli.seed, counts)

    latest = run / "latest.pt"
    best_path = run / (cli.name + "-best.tar")
    start_epoch, best, bad_epochs = 0, float("inf"), 0
    if latest.is_file():
        state = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"]
        best = state["best_validation_rmse_m"]
        bad_epochs = state["bad_epochs"]
        logging.info("resumed epoch=%d best=%.4f", start_epoch, best)

    started = time.time()
    actual_epochs = start_epoch
    for epoch in range(start_epoch, tcfg["max_epochs"]):
        model.train()
        model.train_output_flag = True
        running, used, epoch_started = 0.0, 0, time.time()
        for i, batch in enumerate(train_loader):
            inputs, future, mask = unpack(move(batch, device))
            dest = future[-1]
            pred, _, _, generated, mu, logvar = model(*inputs, dest)
            trajectory = maskedMSE(pred, future, mask) if epoch < tcfg["pretrain_epochs"] \
                else maskedNLL(pred, future, mask)
            loss = trajectory + 0.5 * maskedDestMSE(generated, dest, mask) + 0.5 * KL(mu, logvar)
            if not torch.isfinite(loss):
                logging.warning("epoch=%d batch=%d non-finite loss skipped", epoch + 1, i)
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                logging.warning("epoch=%d batch=%d non-finite gradient skipped", epoch + 1, i)
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()
            running += loss.item()
            used += 1
            log_every = tcfg.get("log_every", 500)
            if log_every and (i + 1) % log_every == 0:
                logging.info("epoch=%d progress=%.2f%% loss=%.4f ETA=%ds", epoch + 1,
                             100 * (i + 1) / len(train_loader), running / max(used, 1),
                             int((time.time() - epoch_started) / (i + 1) * (len(train_loader) - i - 1)))

        logging.info("epoch=%d train summary loss=%.4f batches=%d seconds=%.1f",
                     epoch + 1, running / max(used, 1), used,
                     time.time() - epoch_started)

        score = validation_rmse(model, val_loader, device,
                                tcfg["validation_batches"], tcfg["validation_samples"], cli.seed)
        actual_epochs = epoch + 1
        if score < best:
            best, bad_epochs = score, 0
            torch.save(model.state_dict(), best_path)
            logging.info("epoch=%d new best fixed validation RMSE=%.4f m", epoch + 1, score)
        else:
            bad_epochs += 1
            logging.info("epoch=%d validation RMSE=%.4f m no improvement %d/%d",
                         epoch + 1, score, bad_epochs, tcfg["early_stopping_patience"])
        tmp = latest.with_suffix(".pt.tmp")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1, "best_validation_rmse_m": best,
                    "bad_epochs": bad_epochs}, tmp)
        os.replace(tmp, latest)
        torch.save(model.state_dict(), run / (cli.name + "-ep%d.tar" % (epoch + 1)))
        if (epoch + 1 >= tcfg["min_epochs"]
                and bad_epochs >= tcfg["early_stopping_patience"]):
            logging.info("early stopping at epoch %d", epoch + 1)
            break

    torch.save(model.state_dict(), run / (cli.name + ".tar"))
    atomic_json(run / "training_summary.json", {
        "model": "EPN", "dataset": cli.dataset, "seed": cli.seed,
        "actual_epochs": actual_epochs, "max_epochs": tcfg["max_epochs"],
        "best_validation_rmse_m": best,
        "validation_metric": tcfg["validation_metric"],
        "early_stopping_patience": tcfg["early_stopping_patience"],
        "training_seconds": time.time() - started, "parameter_count": counts,
    })


if __name__ == "__main__":
    main()
