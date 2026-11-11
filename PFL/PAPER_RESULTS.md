# PFL — Measured Results

Companion to `PAPER_HANDOFF.md` (code structure). This file contains measurements only.

## Measurement conditions

All numbers below come from the **same 1500 test batches** of
`datasets/NGSIM/test.mat` (`batch_size=64`, `shuffle=False`, so every model sees an
identical subset in an identical order), with the **same per-vehicle aggregation**
(average within a vehicle, then sum across vehicles), converted to metres and reported
at 1–5 s. Metric code is shared: `utils.maskedTest`.

| model | checkpoint | training |
|---|---|---|
| PFL | `ngsim_pfl_e15-ep1.tar` (EMA) | **1 epoch** |
| EPN | `ngsim_epn.tar` | **15 epochs** (5 MSE + 10 NLL) |

**The training budgets are not matched.** PFL numbers come from a 1-epoch checkpoint.
See §5 for why longer PFL training was not used.

## 1. Protocol definitions

Two protocols were measured. They differ in whether ground truth is used at test time.

**Oracle (best-of-N).** Both models draw N candidates and use the ground-truth endpoint
to choose among them. The two implementations are **not equivalent**:

- **PFL** draws N full trajectories, selects the one whose endpoint is nearest the
  ground-truth endpoint, and reports that trajectory. Ground truth is used as a *filter*.
  (`model.py: pick_best`)
- **EPN** draws N destinations, selects the one nearest the ground-truth endpoint, and
  then **feeds that destination into the trajectory decoder**
  (`predict(enc, best_guess_dest)` → `dest_enc(generated_dest)` concatenated to the
  context). Ground truth is used as a *conditioning input*.
  (`EPN/model.py:363-375`, `EPN/evaluate.py:143-153`)

**Deployable (mean-of-K).** No ground truth. K samples are drawn and averaged into an
estimate of `E[Y|c]`.

- **PFL**: mean of K sampled trajectories (`model.py: mean_prediction`).
- **EPN**: each of K sampled destinations is decoded to a trajectory and the K
  trajectories are averaged (`meantraj`). Two weaker variants were also measured:
  `mean` (average the destinations, decode once) and `random` (a single draw).

## 2. Main result

RMSE (m), mean over the 1–5 s horizons.

| protocol | K | PFL | EPN | difference |
|---|---|---|---|---|
| **oracle** | 6 | 1.543 | **1.432** | EPN 7.2 % better |
| **oracle** | 20 | 1.294 | **1.048** | EPN 19.0 % better |
| **deployable** | 6 | **2.174** | 2.278 | PFL 4.6 % better |
| **deployable** | 20 | **2.114** | 2.124 | PFL 0.5 % better |

Full metric set:

| | RMSE | FDE / minFDE | ADE / minADE |
|---|---|---|---|
| PFL oracle-6 | 1.543 | 1.030 | 0.597 |
| PFL oracle-20 | 1.294 | 0.823 | 0.550 |
| PFL deployable-6 | 2.174 | 1.595 | 0.719 |
| PFL deployable-20 | 2.114 | 1.544 | 0.696 |
| EPN oracle-6 | 1.432 | 0.925 | 0.497 |
| EPN oracle-20 | 1.048 | 0.651 | 0.422 |
| EPN deployable-6 (`meantraj`) | 2.278 | 1.691 | 0.730 |
| EPN deployable-20 (`meantraj`) | 2.124 | 1.565 | 0.687 |

**Which model wins depends on the protocol, and the two gaps move in opposite
directions as K grows.**

## 3. Size of the oracle effect

The reduction in RMSE going from the deployable estimator to the oracle estimator, at
matched K, same weights:

| K | PFL | EPN | difference |
|---|---|---|---|
| 6 | 29.0 % | **37.1 %** | 8.1 pp |
| 20 | 38.8 % | **50.7 %** | 11.9 pp |

At K = 20, **half of EPN's reported RMSE performance comes from the oracle protocol.**
Both models gain from oracle selection; EPN gains more, and its lead grows with K.

### EPN under all four selection rules

Same weights, same batches. Shows how much the choice of non-oracle estimator matters.

| rule | K=6 RMSE | K=20 RMSE |
|---|---|---|
| `oracle` (published protocol) | 1.432 | 1.048 |
| `meantraj` (decode each, average trajectories) | 2.278 | 2.124 |
| `mean` (average destinations, decode once) | 2.925 | 2.530 |
| `random` (single draw) | 3.132 | 3.137 |

`meantraj` is the estimator that matches what PFL's mean-of-K computes, and is the one
used in §2. The published `oracle` number reproduced here as 1.432 (K=6) against the
1.355 reported on the full test set; the difference is the 1500-batch subset.

## 4. Reference floors

Same protocol and aggregation, no learned model.

| predictor | RMSE | FDE | ADE |
|---|---|---|---|
| stay put (zero velocity) | 27.244 | 24.557 | 13.141 |
| **constant velocity, longitudinal only** (the model's baseline) | **4.343** | 3.273 | 1.429 |
| constant velocity, both axes | 4.384 | 3.345 | 1.460 |

The longitudinal-only gate that `cv_gate` uses by default is confirmed better than
extrapolating both axes (4.343 vs 4.384) on the test set.

Error reduction from the CV baseline, deployable protocol at K=20: PFL 51.3 %,
EPN 51.1 %.

## 5. PFL training behaviour

A 15-epoch run was attempted. **Validation RMSE is best at epoch 1 and does not improve
afterwards.** EMA validation RMSE, mean per epoch (10-batch estimates, `shuffle=True`):

| run | vf params | ep1 | ep2 | ep3 | ep4 | ep5 | ep6 |
|---|---|---|---|---|---|---|---|
| baseline, no regularisation | 2.96 M | **1.924** | 1.910 | 1.990 | 2.010 | 2.015 | 2.029 |
| + weight decay 0.01, dropout 0.1 | 2.96 M | 2.115 | 2.161 | — | — | — | — |
| U-Net channels `[32,64,128]` | 0.95 M | 2.009 | 2.006 | — | — | — | — |

Validation flow-matching loss, measured on held-out data with the noise and `t` draws
pinned so checkpoints are compared on identical paths:

| checkpoint | train FM | val FM |
|---|---|---|
| baseline ep1 | 0.319 | **0.398** |
| baseline ep5 | 0.243 | **0.423** |
| + wd/dropout ep1 | 0.328 | 0.443 |
| reduced channels ep1 | 0.334 | 0.424 |

Train FM falls 24 % while val FM rises 6.5 %. Neither weight decay + dropout nor a 68 %
reduction in decoder capacity reduced the val loss or narrowed the train/val gap; both
raised val FM.

Test-set best-of-6 RMSE is flat across every checkpoint tried: 1.543 (ep1), 1.600 (ep2),
1.602 (reduced-capacity best), 1.539 (a separate 2-epoch run).

**Dataset note relevant to this section**: the training split holds 5,777,839 windows
drawn from **8,844 distinct (dataset, vehicle) pairs** — 653 windows per vehicle at a
0.2 s stride. The independent-sample count is therefore far below the window count.

## 6. NFE sweep

Forward Euler; NFE = number of steps. Measured on two independent checkpoints (epoch 1
and epoch 2 of the earlier 2-epoch run) with nearly identical curves. `calib` is defined
in §7; its target here is `K/(K+1) = 0.95` at K = 20.

| NFE | RMSE (m) | sample-spread drift vs NFE=2 (lat / lon) | calib (lat / lon) |
|---|---|---|---|
| 2 | 1.893 | — | 0.061 / 0.123 |
| 5 | 1.903 | +105 % / +91 % | 0.257 / 0.441 |
| 10 | 1.921 | +173 % / +131 % | 0.462 / 0.621 |
| 20 | 1.924 | +214 % / +153 % | 0.619 / 0.739 |
| 50 | 1.927 | +247 % / +171 % | 0.762 / 0.824 |
| 100 | 1.931 | +259 % / +174 % | 0.810 / 0.850 |

RMSE varies by 2 % across a 50× range of NFE while the sample spread varies by 2.6×.

Downstream effect on the test set (2-epoch checkpoint, 1500 batches):

| | NFE=10 | NFE=50 |
|---|---|---|
| best-of-6 RMSE | **1.539** | 1.547 |
| minFDE | **1.035** | 1.052 |
| minADE | **0.596** | 0.618 |
| mean-20 RMSE | **2.118** | 2.126 |

Raising NFE improved calibration (calib 0.46/0.62 → 0.76/0.82) and made every downstream
metric slightly worse. All main-table numbers use NFE = 10.

## 7. Collapse diagnostic

Two ratios logged during training (`train.py: validate`, `log_diag`):

- `spread/marg` — sample spread ÷ marginal CV-residual std.
- `calib` = Var(model) / Var(error of the mean of K). For a calibrated conditional model
  the mean of K draws carries `(1 + 1/K)σ²` while the spread measures `σ²`, so
  **`calib → K/(K+1)`**. Collapse drives it to 0; over-dispersion exceeds 1.

Validated on synthetic data at K = 6 (target 0.857): calibrated model **0.853**, fully
collapsed **2.5e-17**, 2× over-dispersed **2.389**.

On PFL: `spread/marg` settled at ~0.40 (longitudinal) / ~0.49 (lateral) and stayed flat
for 1.5 epochs — no collapse. `calib` reaches 0.81 / 0.85 at NFE = 100 against a 0.95
target.

## 8. Open items

1. **Training budgets are not matched** (PFL 1 epoch vs EPN 15). PFL does not improve
   past epoch 1 under any configuration tried (§5), but no equal-budget PFL checkpoint
   exists.
2. **Oracle numbers use a 1500-batch subset**, not the full test set. All models use the
   same subset, so the comparison is internally consistent; the absolute values differ
   slightly from full-test-set numbers (EPN oracle-6: 1.432 here vs 1.355 published).
3. **The encoder / decoder split has not been measured.** Freezing the PFL encoder and
   training an MSE regression head on it would show whether the encoder or the flow
   decoder limits the deployable numbers. Not run.
4. `spread/marg` and `calib` were logged with K = 6 during training and K = 20 in the
   sweep; the `calib` target differs accordingly (0.857 vs 0.95).
