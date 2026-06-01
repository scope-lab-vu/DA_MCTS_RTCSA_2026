# SVGP Dynamics Models

This directory trains and stores the **SVGP residual dynamics models** used by
DA-MCTS (paper §III.B / §IV.A). DA-MCTS predicts the next state with an extended
kinematic (E-kin) model plus two sparse-GP corrections learned here:

- **omega** (`VAR5`) — yaw-rate residual correction.
- **beta** (`VAR6`) — sideslip-angle residual correction.

Each is augmented with the estimated cornering stiffness `C_alpha`, which lets the
planner generalise to unseen friction.

## Models used by DA-MCTS

The dynamics models DA-MCTS loads are set by `svgp_omega_model_path` /
`svgp_beta_model_path` in `params/config_da_mcts.yaml`, which point into this
directory's `data/models/`. **By default they use the original (MPPI-trained)
models:**

- `data/models/krasser_svgp_omega_…MPPI…M_11.npz`
- `data/models/krasser_svgp_beta_…MPPI…M_12.npz`

The **Stanley-collected** variants are also here and can be swapped in via the
config:

- `data/models/krasser_svgp_omega_…stanley_dt002…M_15.npz`
- `data/models/krasser_svgp_beta_…stanley_dt002…M_15.npz`

Newly trained models are written to `data/models/` as well, so enabling one is
just a matter of pointing the config at its `.npz`.

## Layout

| Path | Contents |
|------|----------|
| `data/raw/` | Raw datasets (MPPI trajectory CSVs + Stanley datasets). |
| `data/splits/` | Per-task train/val/test `splits.npz` (+ scalers). |
| `data/models/` | Trained SVGP dynamics models (the Stanley `M_15` omega/beta). |
| `src/data_preprocess/` | Turn raw logs into `data/splits/`. |
| `src/training_scripts/` | Training notebooks (`svgp_train_varidx_5` = omega, `svgp_train_varidx_6` = beta), the `svgp_train_cli.py` / `hp_search_svgp_ard_cli.py` CLIs, and the best-HP `*.json`. |
| `src/verify/` | Sanity-check scripts for trained models. |
| `train_old_models_DA_MCTS_RTCSA_paper.txt` | Exact recipe used for the paper's (MPPI) models. |

## (Re)train — optional

You do **not** need to retrain to run the experiments; the models above are
already included. The notebooks `svgp_train_varidx_5.ipynb` (omega) and
`svgp_train_varidx_6.ipynb` (beta) reproduce the paper's models — see
`train_old_models_DA_MCTS_RTCSA_paper.txt` for the exact datasets and settings.

For command-line training, run from `src/training_scripts/`:

```bash
cd MCTS_SVGP_dynaimcs_model/src/training_scripts

# Hyperparameter search (Optuna) — varidx 5 = omega, 6 = beta
python hp_search_svgp_ard_cli.py \
    --varidx 5 --feature-mode c_alpha_smooth \
    --task-dir ../../data/splits/YasMarina_VAR5_Ts0.020_stanley_mu_1.0489 \
    --n-trials 100 --out-name krasser_svgp_omega_stanley

# Train a single model with chosen inducing points (M)
python svgp_train_cli.py \
    --varidx 5 --feature-mode c_alpha_smooth \
    --task-dir ../../data/splits/YasMarina_VAR5_Ts0.020_stanley_mu_1.0489 \
    --num-inducing 15 --sigma-y 1e-3 --out-name krasser_svgp_omega_stanley
```

Trained models are written to `data/models/`. To use one, point
`svgp_omega_model_path` / `svgp_beta_model_path` in `params/config_da_mcts.yaml`
at the new `.npz`.
