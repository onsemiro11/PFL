# PFL–EPN fair-comparison repository audit

This audit was fixed before implementing or running the new experiment matrix.
The existing `ngsim_*` and interrupted `highd_pfl_*` checkpoints are historical
artifacts and are not eligible for the new summaries.

## Implemented components

- PFL model: `PFL/model.py::PFLNet`
- PFL temporal self-attention and attention pooling:
  `PFL/encoder.py::TemporalAttentionEncoder`
- PFL grid social pooling and social cross-attention:
  `PFL/encoder.py::PFLEncoder.soc_encode`
- PFL relational ego conditioning:
  `PFL/encoder.py::RelationalEgoEncoder`
- Rectified-flow vector field and Euler sampling:
  `PFL/vf.py::VectorField` and `PFL/model.py::PFLNet.sample`
- EPN endpoint CVAE, endpoint-conditioned decoder, and sampling:
  `EPN/model.py::EPNet.forward` and `EPNet.predict`
- Shared dataset implementation: `PFL/data.py` and `EPN/data.py` are byte-identical.
- NGSIM and highD preprocessing: `preprocess/preprocess_ngsim_fast.m` and
  `preprocess/preprocess_highd_fast.m`.

## Data representation

- Both processed datasets store lateral `x` and longitudinal `y` in feet.
- Metrics multiply distances by `0.3048` and are reported in metres.
- NGSIM is sampled at 10 Hz. highD is converted from 25 Hz to 10 Hz before the
  shared MATLAB preprocessing.
- The loader uses `t_h=30`, `t_f=50`, `d_s=2`: 16 history states including the
  current state at 5 Hz, and 25 future states covering 5 seconds at 5 Hz.
- The ego row is the planning vehicle. Only non-ego targets found in its grid are
  predicted.
- The supplied ego-plan tensor is built from the recorded ego future. Validation
  and test optionally fit a quintic spline to those points. There is no separate
  logged planner-output field. It must therefore be described as an executed-future
  plan proxy, even when candidate selection is GT-free.
- PFL models a standardised residual from a constant-velocity baseline. The
  current default estimates velocity from the last three 0.2 s steps and applies
  it only longitudinally; residual mean and standard deviation are fitted on the
  training set. EPN models raw relative future coordinates in feet and uses a
  Gaussian trajectory head conditioned on a sampled endpoint.

## Existing variant structure

- `PLAN=0`: `use_planning=False` also forces relational conditioning off.
- `REL=0`: `use_planning=True`, `use_relational=False`; the grid plan path remains.
- `Full`: both flags are true.
- All variants return a 112-dimensional context. With the mean head absent, total
  parameter counts are 3,138,711 (PLAN=0), 3,178,727 (REL=0), and 3,210,439
  (Full). The relation encoder is 31,712 parameters, 0.99% of Full. An optional
  `target_cv` capacity control exactly matches Full at 3,210,439 parameters, but
  the preregistered main matrix keeps the clean REL=0 removal because the gap is
  below one percent.
- The relational branch uses relative longitudinal/lateral displacement, relative
  first differences, and a wrapped difference between headings derived from the
  ego-plan and target-CV position sequences. It does not consume or invent a stored
  yaw field.
- The final relational MLP is already zero-initialised and its output is added to
  the target dynamics embedding.

## Current evaluation and checkpoint limitations

- PFL currently reports sample-mean RMSE and endpoint-selected oracle samples, but
  does not report GT-free medoids, direct minADE/minFDE, per-vehicle CSVs, groups,
  runtime, or paired uncertainty in one reproducible artifact.
- EPN test evaluation samples endpoints and selects the endpoint closest to the GT.
  Its comment about selecting the highest maneuver probability does not match the
  executed code.
- EPN has no candidate-specific endpoint confidence. Its inference `forward`
  returns the input maneuver tensors rather than predicted candidate scores.
  A top-confidence EPN result is therefore unavailable without changing the model.
  The evaluator will record this as unavailable and may report explicitly labelled
  sample-mean/medoid fallbacks; it will not invent a confidence rule.
- PFL has seeded validation support, but the old runs used small periodically drawn
  validation subsets. EPN has no seed setup, deterministic validation, best-model
  checkpoint, scheduler metadata, or early stopping.
- Existing EPN checkpoints lack full configuration and seed provenance. Existing
  PFL checkpoints use different objectives, including a mean head in recent runs.
  None will be mixed with the preregistered experiment matrix.

## Split leakage audit of the existing official-style files

The preprocessing splits ego vehicles by vehicle ID within every dataset segment.
It does not split recordings. A vehicle that is an ego in one split does not occur
as an ego in another split, but target vehicles can cross split boundaries.

| Dataset | Split | Windows | Ego vehicles | Target vehicles | Segments |
|---|---|---:|---:|---:|---:|
| NGSIM | train | 5,777,839 | 8,844 | 9,165 | 6 |
| NGSIM | val | 761,734 | 1,010 | 1,457 | 6 |
| NGSIM | test | 1,320,103 | 1,925 | 2,934 | 6 |
| highD | train | 3,796,329 | 67,583 | 67,565 | 120 |
| highD | val | 498,337 | 9,648 | 9,842 | 120 |
| highD | test | 1,010,583 | 18,756 | 18,827 | 120 |

Target-vehicle overlaps are NGSIM train/val 487, train/test 826, val/test 557;
highD train/val 209, train/test 0, val/test 254. Every segment appears in every
split. This is a material limitation for generalisation claims. The initial
experiment implementation will audit and report the supplied split without silently
changing it. A recording-disjoint split requires a separately approved dataset
version because it changes the benchmark and, for NGSIM, leaves only six segments.

## Preregistered implementation choices

- Seeds: `[0, 1, 2]`.
- PFL mean head: disabled and absent from the new model instances.
- PFL variants share decoder, 112-D context, optimiser, cosine schedule, batch size,
  validation rule, and sampling setup.
- Validation/checkpoint criterion: fixed validation subset, sample-mean trajectory
  RMSE with `K=20`; lower is better. Maximum 15 epochs, minimum 5 epochs, patience 3.
- Test K values: 6 and 20. The first six draws are a prefix of the same 20-draw set.
- NFE: 10 for PFL unless a config explicitly preregisters another value.
- Interaction group threshold: 30 metres, stored in config before testing.
- Oracle equivalence relative margin: 3%, stored in config before testing.
- Bootstrap unit: dataset-scoped target vehicle; recording bootstrap is also
  reported when there are enough recording units. Sliding windows are never used
  as independent bootstrap samples.
- Hyperparameter search budget: zero for this confirmatory matrix. Test outputs do
  not alter configuration or thresholds.
- Python, NumPy, Torch, CUDA sampling, DataLoader order, and fixed validation draws
  are seeded. Forced deterministic CUDA kernels are supported by a flag but are
  preregistered off: on the available RTX 4070 SUPER they make the PFL Conv1d
  sampler hundreds of times slower. This runtime decision was made from a random
  one-batch smoke benchmark before any test performance was inspected.

## Implemented fair pipeline

1. Mean-head construction is opt-in; fair configs do not instantiate it.
2. GT-free mean/medoid APIs and the explicit endpoint-oracle API are separate.
3. EPN candidate generation is device-safe and does not invent confidence scores.
4. Configs and launchers cover both datasets, three seeds, three PFL variants,
   EPN, and CV.
5. Unified evaluation writes vehicle/group metrics, distribution/runtime metadata,
   paired bootstrap summaries, figures, gains, and A–D claim classification.
6. Protocol tests and one-batch PFL/EPN evaluation smoke tests cover the executable
   paths. Full training is resumable and remains an explicit long-running command.

## TODOs that cannot be resolved from this repository

- A true online planner trajectory is not present; only recorded ego future is
  available as a plan proxy.
- EPN candidate confidence is not implemented, so `top_confidence` cannot be
  reported honestly.
- PiP source exists but no checkpoint exists. It is left outside the minimum matrix
  rather than retrained under an unverified setup.
- NGSIM recording-disjoint allocation is not uniquely defined for only six temporal
  segments. The supplied split is retained and flagged.
