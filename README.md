# PFL

**Planning-informed flow-matching trajectory prediction for highway driving.**

PFL predicts the future trajectories of the vehicles around an ego vehicle,
conditioned on the ego vehicle's own planned trajectory. It combines

- a **relational ego-conditioned encoder**: temporal self-attention over vehicle
  histories, social pooling on a 25 × 5 grid, and a GRU over the per-step relation
  between the ego plan and each target's constant-velocity rollout, and
- a **conditional rectified-flow decoder**: a 1D U-Net that generates the full
  5-second future trajectory of each target in one pass, solved with an Euler ODE.

The model is trained in two stages. A deterministic regression head first
warm-starts the encoder. The flow decoder is then trained on top of it.

Supported datasets: **NGSIM** (US-101, I-80) and **highD**.

---

## Repository layout

```text
.
├── preprocess/      Raw NGSIM / highD  →  .mat training files
├── PFL/             Model, training, and evaluation code
├── experiments/     Preregistered comparison harness: configs, runners, statistics, figures
├── scripts/         Convenience shell wrappers and the paper figure script
└── datasets/        (not tracked) .mat files go here
```

| Path | Purpose |
|---|---|
| `preprocess/preprocess_ngsim_fast.m` | NGSIM raw txt → `.mat` (maneuver labels, 25 × 5 neighbour grid, splits) |
| `preprocess/highd_to_ngsim.py` | highD → NGSIM-format CSV (direction split, 25 → 10 Hz, metres → feet) |
| `preprocess/preprocess_highd_fast.m` | highD CSV → `.mat`, same format as NGSIM |
| `preprocess/nbhCheckerFast.m`, `targSpecFast.m` | Edge filtering and target selection, called by the two `.m` scripts |
| `preprocess/compare_mats.py` | Compare two `.mat` outputs |
| `PFL/data.py` | `.mat` dataset loader and per-target batching (`highwayTrajDataset`) |
| `PFL/encoder.py` | Constant-velocity baseline, temporal attention, social pooling, relational ego encoder |
| `PFL/vf.py` | Conditional 1D U-Net vector field |
| `PFL/model.py` | `PFLNet`: residual normalisation, losses, sampling |
| `PFL/train.py`, `PFL/evaluate.py` | Training and test-set evaluation |
| `PFL/diagnose.py`, `PFL/smoke.py` | Flow-collapse / NFE diagnostics; a smoke test that needs no data |
| `experiments/configs/*.yaml` | Fixed configurations of every variant (`pfl_stage2*.yaml` = final model) |
| `experiments/scripts/` | Train / evaluate runners, paired bootstrap summary, plots |
| `experiments/results/` | Per-seed and summary metrics of the reported runs |

---

## Requirements

Tested with:

| Package | Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.5 (CUDA 12.4) |
| NumPy | 1.26 |
| SciPy | 1.12 |
| h5py | 3.12 |
| pandas | needed by `highd_to_ngsim.py` |

MATLAB (R2019b or later, for `-batch`) is required for preprocessing.

```bash
conda create -n pfl python=3.10 -y
conda activate pfl
pip install torch numpy scipy h5py pandas matplotlib
```

Check the installation without any data:

```bash
cd PFL && python smoke.py
# smoke ok  loss=...  enc=(4, 112)  device=cuda
```

> **Note.** Some shell wrappers in `scripts/` and `experiments/scripts/` hard-code
> `ROOT=/home/hyundo/motion_prediction` and a conda path. Before using them, edit
> those lines to point at your clone. The Python entry points work from any location.

---

## Data preparation

Raw and processed data are **not** part of this repository. The steps below build
them locally.

### 1. Download the raw datasets

- **NGSIM**: US-101 and I-80 trajectories from the
  [U.S. DOT data portal](https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj).
  The pipeline expects the six classic per-period text files in one folder:

  ```text
  us101-0750am-0805am.txt  us101-0805am-0820am.txt  us101-0820am-0835am.txt
  i80-0400-0415.txt        i80-0500-0515.txt        i80-0515-0530.txt
  ```

- **highD**: request access at [levelxdata.com](https://levelxdata.com/highd-dataset/).
  You need the `NN_tracks.csv`, `NN_tracksMeta.csv` and `NN_recordingMeta.csv` files.

### 2. Preprocess

Each MATLAB script writes `gridTrainAround.mat`, `gridValAround.mat` and
`gridTestAround.mat` to `<OUT>/25x5_8Veh_nofL/`.

**NGSIM**

```bash
NGSIM_RAW=/path/to/raw_ngsim/ NGSIM_OUT=/path/to/processed/ngsim/ \
matlab -batch "addpath('preprocess'); run('preprocess/preprocess_ngsim_fast.m')"
```

**highD**

```bash
# (a) highD → NGSIM-format CSV, one file per recording and direction
python preprocess/highd_to_ngsim.py --src /path/to/highD/data --dst /path/to/raw_highd

# (b) CSV → .mat
HIGHD_RAW=/path/to/raw_highd/ HIGHD_OUT=/path/to/processed/highd/ \
matlab -batch "addpath('preprocess'); run('preprocess/preprocess_highd_fast.m')"
```

All 60 highD recordings need more than 30 GB of RAM in the MATLAB stage.
Set `HIGHD_RECS=N` to use only the first `N` recordings. `HIGHD_RECS=40` is
roughly the size of NGSIM. `scripts/preprocess_highd.sh` runs both steps.

Optional environment variables: `NGSIM_MAX_ROWS` (quick test on a subset) and
`NGSIM_SAVE_RAW=1` / `HIGHD_SAVE_RAW=1` (also save the pre-filter `.mat` files).

### 3. Place the files

`experiments/` expects exactly this layout. `PFL/` accepts any path.
Symbolic links work:

```bash
mkdir -p datasets/NGSIM datasets/highD
for s in Train Val Test; do
  l=$(echo $s | tr A-Z a-z)
  ln -s /path/to/processed/ngsim/25x5_8Veh_nofL/grid${s}Around.mat datasets/NGSIM/$l.mat
  ln -s /path/to/processed/highd/25x5_8Veh_nofL/grid${s}Around.mat datasets/highD/$l.mat
done
```

```text
datasets/
├── NGSIM/{train,val,test}.mat
└── highD/{train,val,test}.mat
```

### Data format and tensor contract

Every file is a MATLAB v7.3 (HDF5) `.mat` with two variables:

- `traj`: one row per scene (ego vehicle × frame). Columns are dataset ID, ego
  vehicle ID, frame, Local X/Y, lane, maneuver labels, size, class, velocity,
  acceleration, and the vehicle IDs on the 25 × 5 grid.
- `tracks`: per-vehicle time series, `{dataset_id, vehicle_id}`.

The full column list is in the header of `PFL/data.py`. Units are feet at 10 Hz.
`Local X` is lateral and `Local Y` is longitudinal.

With the default settings `t_h=30, t_f=50, d_s=2`, the loader produces:

| Tensor | Shape | Notes |
|---|---|---|
| Target history | `[16, N, 5]` | t = −3.0 … 0.0 s at 0.2 s steps, current frame included |
| Neighbour history | `[16, M, 5]` | Positions relative to the target at each step |
| Ego plan | `[25, N, 2]` | Relative to each target's current position |
| Target future (prediction) | `[25, N, 2]` | t = 0.2 … 5.0 s |

Each history step has five channels: `[lateral, longitudinal, velocity,
acceleration, TTC]`. `N` is the number of target vehicles in the batch, pooled
across scenes.

---

## Training

Run the commands from `PFL/`. Each run writes to `PFL/trained_models/<name>/`.

### Stage 1: regression warm-start (3 epochs)

```bash
cd PFL
python train.py \
    --name stage1_ngsim_seed0 \
    --train_set ../datasets/NGSIM/train.mat \
    --val_set   ../datasets/NGSIM/val.mat \
    --seed 0 --batch_size 64 --learning_rate 1e-4 \
    --train_epochs 3 --min_epochs 3 --early_stopping_patience 0 \
    --flow_weight 0.0 --mean_weight 1.0 --aux_weight 0.3 \
    --vf_dropout 0.0 --weight_decay 0.0 \
    --val_every 0 --val_batches 100 --val_samples 20 --sample_steps 10 \
    --epoch_validation --fixed_validation --use_mean_head
```

### Stage 2: flow matching, initialised from the Stage 1 EMA weights

```bash
python train.py \
    --name stage2_ngsim_seed0 \
    --train_set ../datasets/NGSIM/train.mat \
    --val_set   ../datasets/NGSIM/val.mat \
    --init_from trained_models/stage1_ngsim_seed0/stage1_ngsim_seed0.tar \
    --seed 0 --batch_size 64 --learning_rate 1e-4 \
    --train_epochs 15 --min_epochs 5 --early_stopping_patience 3 \
    --flow_weight 1.0 --mean_weight 0.01 --aux_weight 0.3 \
    --vf_dropout 0.0 --weight_decay 0.0 \
    --val_every 0 --val_batches 100 --val_samples 20 --sample_steps 10 \
    --epoch_validation --fixed_validation --use_mean_head
```

For highD, replace the dataset paths and run names. The reported results use
seeds 0, 1 and 2.

### Training outputs

```text
PFL/trained_models/<name>/
├── args.json               # full configuration of the run
├── train.log
├── latest.pt               # resumable state (model, optimiser, EMA, scheduler)
├── <name>-ep<k>.tar        # EMA weights after epoch k
├── <name>-best.tar         # best EMA weights on the fixed validation set
├── <name>.tar              # final EMA weights (used by --init_from)
└── training_summary.json
```

An interrupted run resumes automatically from `latest.pt` (`--resume auto`).

---

## Evaluation

```bash
cd PFL
python evaluate.py \
    --name stage2_ngsim_seed0-best \
    --test_set ../datasets/NGSIM/test.mat \
    --use_mean_head \
    --mean_k 20 --best_of_n 6 --sample_steps 10
```

`--name` uses the form `<run>-<tag>`. It resolves to
`trained_models/<run>/<run>-<tag>.tar`, and the log is written to
`trained_models/<run>/evaluation.log`. Because of this, run names must not
contain `-`; use `_` instead. If the file does not exist, the script falls back
to `trained_models/<run>/latest.pt`.

Model flags must match those used for training. The defaults already match the
configuration above.

Reported metrics, in metres per horizon:

- **RMSE / ADE / FDE**: from the mean of `--mean_k` flow samples. This needs no ground truth.
- **minADE / minFDE**: from the best of `--best_of_n` samples (oracle selection).

| Option | Effect |
|---|---|
| `--single_sample` | RMSE from a single raw sample |
| `--max_batches N` | Quick check on the first N batches |
| `--plan_info_ds K` | Fit the ego plan to every K-th waypoint only |

`diagnose.py` sweeps the number of Euler steps and reports sample spread.
It checks whether the flow has collapsed onto the conditional mean:

```bash
python diagnose.py --name stage2_ngsim_seed0-best \
    --test_set ../datasets/NGSIM/val.mat --nfe 2 5 10 20 50
```

---

## Reproducing the reported experiments

`experiments/` runs every model under one fixed protocol: same splits,
validation rule, K ∈ {6, 20}, NFE = 10, and seeds 0–2. It then computes
vehicle-level paired bootstrap statistics.

```bash
# one Stage 1 + Stage 2 run
python experiments/scripts/train_one.py --config experiments/configs/pfl_stage2_pretrain.yaml \
    --dataset ngsim --seed 0 --name stage2_ngsim_pretrain_seed0
python experiments/scripts/train_one.py --config experiments/configs/pfl_stage2.yaml \
    --dataset ngsim --seed 0 --name stage2_ngsim_seed0 \
    --init-from PFL/trained_models/stage2_ngsim_pretrain_seed0/stage2_ngsim_pretrain_seed0.tar

# protocol evaluation, written to experiments/results/<dataset>/
python experiments/scripts/evaluate_model.py --model pfl --config experiments/configs/pfl_stage2.yaml \
    --dataset ngsim --seed 0 \
    --checkpoint PFL/trained_models/stage2_ngsim_seed0/stage2_ngsim_seed0-best.tar

# statistics and figures
python experiments/scripts/summarize_results.py
python experiments/scripts/plot_results.py
```

`experiments/scripts/run_stage2_reproduction.sh` runs the complete Stage 2
matrix (2 datasets × 3 seeds). See `experiments/README.md` for the full protocol.

The repository contains the per-seed and summary results of the reported runs
in `experiments/results/`. It does not contain the per-vehicle CSVs that the
bootstrap needs, because of their size; `evaluate_model.py` regenerates them.

---

## Ablations

| Variant | `PFL/train.py` flags | Config |
|---|---|---|
| Full (final model) | none | `pfl_stage2*.yaml` |
| REL = 0: no relational ego encoder | `--use_relational` | `pfl_stage2_rel0*.yaml` |
| PLAN = 0: no ego plan | `--use_planning` | `pfl_plan0.yaml` |

These flags are on by default, so passing a flag turns the component **off**.
Pass the same flag to `evaluate.py` when evaluating an ablated model.

---

## Notes

- **Ego plan is a proxy.** The ego plan input is the **recorded** future of the ego vehicle,
  not the output of an online planner. During validation and testing, each axis
  is smoothed by a least-squares quintic polynomial fit with the current position fixed.
  Training uses the raw recorded future.
- **Units.** Coordinates are in feet inside the model. All reported metrics are converted to metres.
- **Not in this repository.** Checkpoints, raw data and processed data are not included.
  The EPN baseline code is also not included; `experiments/` imports it from `EPN/`
  only when evaluating EPN.
