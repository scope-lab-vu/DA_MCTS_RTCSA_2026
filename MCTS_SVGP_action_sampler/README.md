# Learned Gaussian Action Prior

This directory trains and stores the **learned Gaussian action prior** used by
DA-MCTS (paper §IV.B). The prior maps the current driving context to a
distribution over actions, so the planner can focus its continuous-action
sampling instead of sampling uniformly:

```
(velocity, max_curvature ahead, cornering_stiffness C_alpha)
        ->  mean & std for (acceleration, steering_velocity)
```

## Model used by DA-MCTS

DA-MCTS loads the prior from `action_prior_model_path` in
`params/config_da_mcts.yaml`. By default this is the **original (MPPI-collected)
prior**:

- **`minimal_feature_1st_train/`** — the active prior (trained on the original
  dataset).

A **Stanley-collected** SVGP prior is also included and can be trained and used
instead — just point `action_prior_model_path` at it:

- `stanley_dt0.02_svgp_no_lateral/` (dt = 0.02)
- `stanley_dt0.04_sim_dt_0.04_svgp_no_lateral/` (dt = 0.04)

## Files

| File | Purpose |
|------|---------|
| `train.py` | Train the multi-output **SVGP** prior. Also provides `load_model()`, which the DA-MCTS controller imports at runtime — **do not delete.** |
| `train_krasserm.py` | Alternative **SGPR** (Krasserm / VFE) trainer; also provides `load_model()` used by the controller — **do not delete.** |
| `hp_search*.py`, `run_hp_search*` | Optuna hyperparameter search over the trainers. |
| `training_data/` | Demonstration datasets used for training. |

## (Re)train the prior — optional

You do **not** need to retrain to run the experiments; the default prior above is
already included.

1. Hyperparameter search (Optuna):

```bash
python MCTS_SVGP_action_sampler/hp_search.py \
    --data-dir MCTS_SVGP_action_sampler/training_data --n-trials 100
```

2. Train the SVGP prior:

```bash
python MCTS_SVGP_action_sampler/train.py \
    --data-dir MCTS_SVGP_action_sampler/training_data --epochs 150 --plot
```

Then set `action_prior_model_path` in `params/config_da_mcts.yaml` to the new
model directory.

## Related

The SVGP **dynamics** models (`svgp_omega_model_path` / `svgp_beta_model_path`)
are separate from the action prior. The default MPPI-trained dynamics models live
in `models/svgp/`; Stanley-trained variants live in `MCTS_SVGP_dynaimcs_model/`.
