# Temporal relational conditioning study

## Scope

The paper direction remains planning-informed full-trajectory flow matching. No endpoint
module, deterministic mean head, causal mask, or learned base-trajectory predictor is added.
The candidate tests whether the already defined target-specific relation is easier to use
when its 25 physical-horizon positions are also passed to the matching U-Net positions.

## Relation and model variants

For each target, the existing encoder constructs the normalized sequence

```text
R_i[τ] = [longitudinal gap, lateral gap,
          relative longitudinal step velocity, relative lateral step velocity,
          heading difference], τ=1..25.
```

`CV_i(H_i)` is a history-based kinematic reference. It is not interpreted as a
counterfactual trajectory without ego influence. The relation is a reparameterization and
inductive bias, not identification of an independent causal effect.

| Variant | Existing pooled relational embedding | Horizon-aligned U-Net input |
|---|---:|---:|
| `REL=0` | no | no |
| `Full` | yes | no |
| `Temporal-only` | no | yes |
| `Global+Temporal` | yes | yes |
| `Temporal-zero-control` | modules retained, zero relation input | modules retained, zero relation input |

All variants retain the same history inputs, grid-plan path, plan token path, CV residual,
full 25-step trajectory flow-matching objective, optimizer, sampling settings, and evaluation
rules. The three temporal variants have identical parameter counts.

## Decision protocol

1. Run the seed-0 `Global+Temporal` pilot for five epochs and select its checkpoint only by
   fixed validation K=20 sample-mean RMSE.
2. Reject before further training if it is more than 3% worse than the existing seed-0 Full
   checkpoint.
3. Otherwise run seed-0 `Temporal-only` and `Temporal-zero-control` pilots.
4. Adopt for full runs only if `Global+Temporal` beats both Full and the zero-relation control
   on fixed validation. This gate is a resource decision and is not reported as a test result.
5. If adopted, train the proposed model with the original 15-epoch maximum, patience 3, and
   three seeds on NGSIM and highD. Run the two mechanism controls with three seeds on NGSIM.
6. Complete the preregistered CV/PFL/EPN baseline matrix and evaluate every retained model
   using the corrected full-horizon mask protocol.

Official test data do not participate in model adoption or checkpoint selection.

## Paper outputs

- `experiments/results/temporal_pilot_decision.{json,md}`: pilot resource gate
- `experiments/results/temporal_study_table.csv`: seed mean and standard deviation table
- `experiments/results/temporal_study_summary.json`: vehicle-bootstrap comparisons and oracle
  equivalence to EPN
- `experiments/figures/temporal_relation_rmse.png`: horizon-wise GT-free performance
- `experiments/figures/temporal_training_fm_loss.png`: flow-matching optimization curves
- `experiments/figures/temporal_validation_rmse.png`: fixed-validation checkpoint curves
- `experiments/results/{dataset}/per_vehicle/`: paired physical-vehicle units
- `experiments/temporal_paper_pipeline.log`: sequential experiment log
- `experiments/temporal_study_status.json`: current stage

Pilot results are diagnostic and are not paper headline results.
