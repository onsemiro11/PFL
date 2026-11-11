"""Train PFL on the NGSIM .mat pipeline (encoder + flow decoder)."""
import argparse
import copy
import logging
import math
import os
import time
import json
import random
from functools import wraps

# Required by deterministic CUDA GEMM.  It must be present before torch first
# creates a CUDA context, so set it at module import rather than in train().
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

import torch
from torch.utils.data import DataLoader

from config import add_train_args
from data import highwayTrajDataset
from model import PFLNet, strip_derived
from utils import initLogging, maskedTest


def _cuda(batch, use_cuda):
    if not use_cuda:
        return batch
    return [t.cuda() if torch.is_tensor(t) else t for t in batch]


def _unpack(data):
    (nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask, targsFut,
     targsFutMask, lat_enc, lon_enc, _, nbsVel, nbsAcc, nbsColl,
     vel_hist, acc_hist, coll_hist) = data
    enc_args = (nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask,
                lat_enc, lon_enc, nbsVel, nbsAcc, nbsColl, vel_hist, acc_hist,
                coll_hist, None)
    return enc_args, targsHist, targsFut, targsFutMask, lat_enc, lon_enc


class EMA:
    """Exponential moving average of the weights.

    Flow and diffusion models sample noticeably worse from the raw optimiser
    iterate than from an averaged one; the evaluated checkpoint is the EMA copy.
    """

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        shadow_p = dict(self.shadow.named_parameters())
        for k, p in model.named_parameters():
            shadow_p[k].mul_(d).add_(p.detach(), alpha=1.0 - d)
        # Buffers are copied, not averaged: res_mean / res_std are fitted statistics,
        # and decaying them towards their fitted value would leave the shadow
        # de-normalised for the first few thousand steps.
        shadow_b = dict(self.shadow.named_buffers())
        for k, b in model.named_buffers():
            shadow_b[k].copy_(b)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        self.shadow.load_state_dict(sd)


def fit_stats(net, loader, n_batches, use_cuda, out_length):
    """Per-step, per-axis mean and std of the constant-velocity residual."""
    dev = next(net.parameters()).device
    s = torch.zeros(out_length, 2, device=dev, dtype=torch.float64)
    ss = torch.zeros_like(s)
    n = torch.zeros_like(s)
    with torch.no_grad():
        for i, data in enumerate(loader):
            if i >= n_batches:
                break
            data = _cuda(data, use_cuda)
            _, targsHist, targsFut, targsFutMask, _, _ = _unpack(data)
            r = (targsFut.permute(1, 0, 2) - net.cv_baseline(targsHist)).double()
            m = targsFutMask.permute(1, 0, 2).double()
            s += (r * m).sum(0)
            ss += (r * r * m).sum(0)
            n += m.sum(0)
    net.fit_norm_stats(s.float(), ss.float(), n.float())
    logging.info("residual std (ft) lat %s", net.res_std[[4, 9, 14, 19, 24], 0].tolist())
    logging.info("residual std (ft) lon %s", net.res_std[[4, 9, 14, 19, 24], 1].tolist())


def fixed_validation(fn):
    @wraps(fn)
    def wrapped(net, loader, args, *a, **kw):
        if not getattr(args, "fixed_validation", False):
            return fn(net, loader, args, *a, **kw)
        devices = [next(net.parameters()).device.index] if next(net.parameters()).is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(args.seed + 1)
            return fn(net, loader, args, *a, **kw)
    return wrapped


@fixed_validation
@torch.no_grad()
def validate(net, loader, args, n_steps=None, n_samples=None):
    """RMSE under the same convention as evaluate.py / maskedTest, plus the
    statistics needed to tell a live flow from one that collapsed onto E[Y|c].

    maskedTest sums (dx^2 + dy^2) over points; dividing by mask.sum() instead
    would divide by twice the number of points and report RMSE/sqrt(2).

    RMSE alone cannot see a collapse -- a model whose k samples are all identical
    minimises exactly this number. So the per-step, per-axis spread across the k
    draws is accumulated alongside it, together with the squared error the mean of
    those draws actually makes. Returns (rmse_m, spread_std, mean_err_std) with the
    two std arrays [out_length, 2] in feet.
    """
    was_training = net.training
    net.eval()
    steps = args.sample_steps if n_steps is None else n_steps
    k = args.best_of_n if n_samples is None else n_samples
    dev = next(net.parameters()).device
    err, cnt = 0.0, 0.0
    var_s = torch.zeros(args.out_length, 2, device=dev, dtype=torch.float64)
    var_e = torch.zeros_like(var_s)
    n_pt = torch.zeros_like(var_s)
    for i, data in enumerate(loader):
        if i >= args.val_batches:
            break
        data = _cuda(data, args.use_cuda)
        enc_args, targsHist, targsFut, targsFutMask, _, _ = _unpack(data)
        ctx = net.encode(*enc_args)
        samples = net.sample(ctx, targsHist, n_steps=steps, n_samples=k)
        mean = net.mean_prediction(samples)
        ll, _, cc = maskedTest(mean.permute(1, 0, 2), targsFut, targsFutMask)
        err += ll.sum().item()
        cnt += cc.sum().item()
        m = targsFutMask.permute(1, 0, 2).double()
        if samples.shape[0] > 1:
            var_s += (samples.double().var(dim=0, unbiased=True) * m).sum(0)
        var_e += ((mean.double() - targsFut.permute(1, 0, 2).double()) ** 2 * m).sum(0)
        n_pt += m.sum(0)
    net.train(was_training)
    n = n_pt.clamp(min=1.0)
    rmse = (err / max(cnt, 1.0)) ** 0.5 * 0.3048
    return rmse, (var_s / n).sqrt(), (var_e / n).sqrt()


@fixed_validation
@torch.no_grad()
def validate_mean(net, loader, args):
    """RMSE of the deterministic mean head, under maskedTest's convention."""
    was_training = net.training
    net.eval()
    err, cnt = 0.0, 0.0
    for i, data in enumerate(loader):
        if i >= args.val_batches:
            break
        data = _cuda(data, args.use_cuda)
        enc_args, targsHist, targsFut, targsFutMask, _, _ = _unpack(data)
        pred = net.mean_predict(net.encode_mean(*enc_args), targsHist).permute(1, 0, 2)
        ll, _, cc = maskedTest(pred, targsFut, targsFutMask)
        err += ll.sum().item()
        cnt += cc.sum().item()
    net.train(was_training)
    return (err / max(cnt, 1.0)) ** 0.5 * 0.3048


def log_diag(tag, net, res, k):
    """Collapse diagnostic for one validate() result.

    spread/marg is the sample spread against the marginal residual std that
    res_std holds; it goes to zero exactly when the flow has collapsed onto the
    conditional mean, and a live model sits at the fraction of the marginal
    variance the context does not explain.

    calib is sharper: it compares the spread the model believes in against the
    error the mean of k samples really makes. For a calibrated conditional model
    the mean of k draws carries (1 + 1/k) sigma^2 while the spread measures
    sigma^2, so the ratio sits near k/(k+1). Collapse drives it to 0, an
    over-dispersed model above 1.
    """
    M = 0.3048
    rmse, sd, ed = res
    ref = net.res_std.double()
    marg = (sd / ref.clamp(min=1e-9)).mean(0)
    calib = (sd.pow(2).sum(0) / ed.pow(2).sum(0).clamp(min=1e-12))
    logging.info("  [%s] RMSE %.3f m | spread@5s lat %.3f lon %.3f m | "
                 "spread/marg lat %.3f lon %.3f | calib lat %.3f lon %.3f (target %.2f)",
                 tag, rmse, sd[-1, 0].item() * M, sd[-1, 1].item() * M,
                 marg[0].item(), marg[1].item(), calib[0].item(), calib[1].item(),
                 k / (k + 1.0))
    return rmse


def train():
    parser = argparse.ArgumentParser(description="PFL: planning-informed flow matching")
    add_train_args(parser)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    training_started = time.time()

    log_path = "./trained_models/{}/".format(args.name)
    os.makedirs(log_path, exist_ok=True)
    initLogging(log_file=log_path + "train.log")
    with open(log_path + "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    logging.info("intent_condition=%s relational_gate=%s seed=%s fixed_validation=%s",
                 args.intent_condition, args.relational_gate, args.seed, args.fixed_validation)
    logging.info("PFL  name=%s  plan=%s  attn=%s  rel=%s  aux=%s  ema=%s",
                 args.name, args.use_planning, args.use_attention,
                 args.use_relational, args.aux_weight, args.ema_decay)

    net = PFLNet(args)
    if args.freeze_encoder:
        for p in net.encoder.parameters():
            p.requires_grad = False
        logging.info("Encoder frozen")
    if args.freeze_vf:
        for p in net.vf.parameters():
            p.requires_grad = False
        logging.info("Vector field frozen; the flow decoder is left exactly as loaded")
    if args.use_cuda:
        net = net.cuda()

    n_params = sum(p.numel() for p in net.parameters())
    n_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logging.info("parameters total=%d trainable=%d mean_head=%s",
                 n_params, n_trainable, net.use_mean_head)

    params = [p for p in net.parameters() if p.requires_grad]
    # Decoupled weight decay, applied to conv and linear weights only. Decaying the
    # GroupNorm / LayerNorm scales and the biases is the usual way a small model picks
    # up an offset it cannot correct, and it would pull directly against the
    # zero-initialised relational head and the learned positional / pooling queries.
    decay, no_decay = [], []
    for pname, p in net.named_parameters():
        if not p.requires_grad:
            continue
        leaf = pname.split(".")[-1]
        (no_decay if p.ndim <= 1 or leaf in ("pos", "pool_q") else decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.learning_rate)

    # Gradient clipping is what would re-couple the two paths even after the encoders are
    # split: |g_mean| was measured around 500 against |g_flow| around 3.5, so a single
    # clip_grad_norm_ over everything rescales by ~1/500 and leaves the flow branch with a
    # step two orders of magnitude too small. Each branch is therefore clipped separately,
    # which gives both a unit-norm step regardless of the other's magnitude.
    if net.separate_encoders and args.mean_weight:
        mean_params = [p for p in list(net.encoder_mean.parameters())
                       + list(net.mean_head.parameters()) if p.requires_grad]
        mean_ids = {id(p) for p in mean_params}
        flow_params = [p for p in params if id(p) not in mean_ids]
        logging.info("split clipping: flow branch %d tensors, mean branch %d tensors",
                     len(flow_params), len(mean_params))
    else:
        mean_params, flow_params = [], params
    logging.info("AdamW  wd=%s on %s tensors, 0.0 on %s  |  vf_dropout=%s",
                 args.weight_decay, len(decay), len(no_decay), args.vf_dropout)
    latest_path = log_path + "latest.pt"

    tr_set = highwayTrajDataset(path=args.train_set,
                                targ_enc_size=args.social_context_size + args.dynamics_encoding_size,
                                grid_size=args.grid_size, fit_plan_traj=False)
    val_set = highwayTrajDataset(path=args.val_set,
                                 targ_enc_size=args.social_context_size + args.dynamics_encoding_size,
                                 grid_size=args.grid_size, fit_plan_traj=True)
    tr_loader = DataLoader(tr_set, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers, collate_fn=tr_set.collate_fn,
                           generator=torch.Generator().manual_seed(args.seed),
                           persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=not args.fixed_validation,
                            num_workers=args.num_workers, collate_fn=val_set.collate_fn,
                            generator=torch.Generator().manual_seed(args.seed + 1),
                            persistent_workers=args.num_workers > 0)
    logging.info("train %s  val %s", len(tr_set), len(val_set))
    epoch_batches = len(tr_loader)
    total_steps = max(1, epoch_batches * args.train_epochs)

    def lr_at(step):
        if step < args.warmup_steps:
            return (step + 1) / max(1, args.warmup_steps)
        p = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    ema = EMA(net, args.ema_decay)

    def save_ckpt(epoch, batch):
        state = net.state_dict()
        bad = [k for k, v in state.items() if torch.is_floating_point(v) and not torch.isfinite(v).all()]
        if bad:
            raise RuntimeError("non-finite weights: " + bad[0])
        tmp = latest_path + ".tmp"
        torch.save({"model": state, "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch, "batch": batch, "best_rmse": best[0],
                    "bad_epochs": bad_epochs[0]}, tmp)
        os.replace(tmp, latest_path)

    # Single-element list so save_ckpt closes over a mutable cell.
    best = [float("inf")]
    bad_epochs = [0]
    best_path = log_path + "{}-best.tar".format(args.name)

    if args.init_from:
        sd = torch.load(args.init_from, map_location="cuda" if args.use_cuda else "cpu",
                        weights_only=False)
        sd = sd.get("ema", sd.get("model", sd)) if isinstance(sd, dict) else sd
        missing, unexpected = net.load_state_dict(net.initial_state(sd), strict=False)
        if args.intent_condition:
            allowed = ("encoder.rel_gate.", "encoder_mean.")
            if unexpected or any(not k.startswith(allowed) for k in missing):
                raise RuntimeError(f"Unexpected warm-start mismatch: {missing}, {unexpected}")
        if net.separate_encoders and any(k.startswith("encoder_mean.") for k in missing):
            net.encoder_mean.load_state_dict(net.encoder.state_dict())
            logging.info("  encoder_mean seeded from the loaded encoder (both paths start equal)")
        net._set_loss_w()
        logging.info("Initialised from %s (%d missing, %d unexpected)",
                     args.init_from, len(missing), len(unexpected))
        if missing:
            logging.info("  newly initialised: %s", ", ".join(sorted({k.split(".")[0] for k in missing})))
        # EMA must start at the loaded checkpoint, not the earlier random model.
        ema.load_state_dict(net.state_dict())

    start_epoch, done = 0, 0
    resume_path = latest_path if args.resume == "auto" else args.resume
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location="cuda" if args.use_cuda else "cpu",
                          weights_only=False)
        net.load_state_dict(strip_derived(ckpt["model"]))
        optimizer.load_state_dict(ckpt["optimizer"])
        if "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch, done = ckpt["epoch"], ckpt["batch"] + 1
        best[0] = ckpt.get("best_rmse", float("inf"))
        bad_epochs[0] = ckpt.get("bad_epochs", 0)
        net._set_loss_w()          # loss_w is derived, and "std" mode needs the loaded res_std
        logging.info("Resumed %s epoch %s batch %s (best val RMSE %.3f)",
                     resume_path, start_epoch + 1, done, best[0])

    if not bool(net.stats_ready):
        logging.info("Fitting residual normalisation on %s batches...", args.stats_batches)
        fit_stats(net, tr_loader, args.stats_batches, args.use_cuda, args.out_length)
        ema.update(net)

    completed_epochs = start_epoch
    for epoch in range(start_epoch, args.train_epochs):
        net.train()
        epoch_started = time.time()
        offset, done = done, 0
        avg_loss, avg_fm, avg_aux, avg_mh, avg_t, n_log = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        epoch_loss, epoch_fm, epoch_aux, epoch_mh, epoch_used = 0.0, 0.0, 0.0, 0.0, 0
        for batch_i, data in enumerate(tr_loader):
            i = batch_i + offset
            if i >= epoch_batches:
                break
            st = time.time()
            data = _cuda(data, args.use_cuda)
            enc_args, targsHist, targsFut, targsFutMask, lat_enc, lon_enc = _unpack(data)
            ctx = net.encode(*enc_args)
            # flow_weight 0 skips the vector field entirely, which is what makes the
            # pure-regression ceiling measurable: the encoder then sees only the MSE
            # gradient, never the flow-matching one.
            if args.flow_weight:
                fm = net.flow_loss(ctx, targsFut, targsFutMask, targsHist,
                                   lat_enc, lon_enc)
            else:
                ctx_tensor, _ = net._split_condition(ctx)
                fm = torch.zeros((), device=ctx_tensor.device)
            aux = net.maneuver_loss(ctx, lat_enc, lon_enc)
            loss = args.flow_weight * fm + args.aux_weight * aux
            mh = torch.zeros((), device=fm.device)
            if args.mean_weight:
                ctx_m = net.encode_mean(*enc_args) if net.separate_encoders else ctx
                mh = net.mean_loss(ctx_m, targsFut, targsFutMask, targsHist)
                loss = loss + args.mean_weight * mh
            if not torch.isfinite(loss):
                logging.warning("epoch %s batch %s non-finite loss, skip", epoch + 1, i)
                optimizer.zero_grad()
                continue
            optimizer.zero_grad()
            loss.backward()
            if mean_params:
                torch.nn.utils.clip_grad_norm_(flow_params, 1.0)
                torch.nn.utils.clip_grad_norm_(mean_params, 1.0)
            else:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            ema.update(net)
            avg_loss += loss.item()
            avg_fm += fm.item()
            avg_aux += aux.item()
            avg_mh += mh.item()
            avg_t += time.time() - st
            n_log += 1
            epoch_loss += loss.item()
            epoch_fm += fm.item()
            epoch_aux += aux.item()
            epoch_mh += mh.item()
            epoch_used += 1
            if args.ckpt_every and i % args.ckpt_every == args.ckpt_every - 1:
                save_ckpt(epoch, i)
            if (args.log_every and (i + 1) % args.log_every == 0
                    and not (args.val_every > 0 and i % args.val_every == args.val_every - 1)):
                eta = (avg_t / max(n_log, 1)) * (epoch_batches - i - 1)
                logging.info("Epoch %s progress %.2f%% loss %.4f FM %.4f man %.4f mean %.4f lr %.2e ETA %ss",
                             epoch + 1, 100.0 * (i + 1) / epoch_batches,
                             avg_loss / max(n_log, 1), avg_fm / max(n_log, 1),
                             avg_aux / max(n_log, 1), avg_mh / max(n_log, 1),
                             scheduler.get_last_lr()[0], int(eta))
                avg_loss, avg_fm, avg_aux, avg_mh, avg_t, n_log = 0.0, 0.0, 0.0, 0.0, 0.0, 0
            if args.val_every > 0 and i % args.val_every == args.val_every - 1:
                eta = (avg_t / max(n_log, 1)) * (epoch_batches - i)
                logging.info("Epoch %s  progress %.2f%%  FM %.4f  man %.4f  mean %.4f  lr %.2e  ETA %ss",
                             epoch + 1, 100.0 * i / epoch_batches,
                             avg_fm / max(n_log, 1), avg_aux / max(n_log, 1),
                             avg_mh / max(n_log, 1), scheduler.get_last_lr()[0], int(eta))
                if not args.flow_weight:
                    if not net.use_mean_head:
                        raise RuntimeError("flow_weight=0 requires --use_mean_head or mean_weight>0")
                    raw_rmse = validate_mean(net, val_loader, args)
                    ema_rmse = validate_mean(ema.shadow, val_loader, args)
                    logging.info("  [raw] mean-head val RMSE %.3f m | [ema] %.3f m",
                                 raw_rmse, ema_rmse)
                    rmse_ema = ema_rmse
                else:
                    log_diag("raw", net,
                             validate(net, val_loader, args, n_samples=args.val_samples),
                             args.val_samples)
                    rmse_ema = log_diag("ema", ema.shadow,
                                        validate(ema.shadow, val_loader, args,
                                                 n_samples=args.val_samples),
                                        args.val_samples)
                # The score is a val_batches-sized estimate and therefore noisy, so the
                # saved point is "near-optimal", not exactly optimal. It exists so the
                # optimum is not lost when the run degrades past it; re-measure the kept
                # checkpoint with evaluate.py before quoting a number from it.
                if rmse_ema < best[0]:
                    best[0] = rmse_ema
                    torch.save(ema.state_dict(), best_path)
                    logging.info("  new best EMA val RMSE %.3f m -> %s", best[0], best_path)
                avg_loss, avg_fm, avg_aux, avg_mh, avg_t, n_log = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        logging.info("Epoch %s train summary loss %.4f FM %.4f man %.4f mean %.4f batches %d seconds %.1f",
                     epoch + 1, epoch_loss / max(epoch_used, 1),
                     epoch_fm / max(epoch_used, 1), epoch_aux / max(epoch_used, 1),
                     epoch_mh / max(epoch_used, 1), epoch_used, time.time() - epoch_started)
        save_ckpt(epoch + 1, -1)
        torch.save(ema.state_dict(), log_path + "{}-ep{}.tar".format(args.name, epoch + 1))
        completed_epochs = epoch + 1

        if args.epoch_validation:
            if not args.flow_weight:
                rmse_ema = validate_mean(ema.shadow, val_loader, args)
                logging.info("Epoch %s fixed validation mean-head RMSE %.4f m",
                             epoch + 1, rmse_ema)
            else:
                result = validate(ema.shadow, val_loader, args,
                                  n_samples=args.val_samples)
                rmse_ema = log_diag("epoch-ema", ema.shadow, result, args.val_samples)
            if rmse_ema < best[0]:
                best[0] = rmse_ema
                bad_epochs[0] = 0
                torch.save(ema.state_dict(), best_path)
                logging.info("Epoch %s new best fixed val RMSE %.4f m -> %s",
                             epoch + 1, best[0], best_path)
            else:
                bad_epochs[0] += 1
                logging.info("Epoch %s no improvement (%d/%d)", epoch + 1,
                             bad_epochs[0], args.early_stopping_patience)
            save_ckpt(epoch + 1, -1)
            if (args.early_stopping_patience > 0 and epoch + 1 >= args.min_epochs
                    and bad_epochs[0] >= args.early_stopping_patience):
                logging.info("Early stopping at epoch %s", epoch + 1)
                break

    # The evaluated checkpoint is the EMA copy.
    torch.save(ema.state_dict(), log_path + "{}.tar".format(args.name))
    logging.info("saved %s%s.tar (EMA)", log_path, args.name)
    summary = {
        "model": "PFL", "name": args.name, "seed": args.seed,
        "actual_epochs": completed_epochs, "max_epochs": args.train_epochs,
        "best_validation_rmse_m": best[0],
        "validation_metric": "sample_mean_rmse_k%d" % args.val_samples,
        "early_stopping_patience": args.early_stopping_patience,
        "parameter_count": n_params, "trainable_parameter_count": n_trainable,
        "mean_head": net.use_mean_head,
        "training_seconds": time.time() - training_started,
    }
    with open(log_path + "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    train()
