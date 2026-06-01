# Deadline-Aware Action-Probability MLP

This directory trains and stores the **MLP that predicts the action-sampling
weights** for DA-MCTS (paper §IV.C). During planning, the MLP maps the current
runtime features to a probability distribution over the three action-sampling
categories, recalibrated every `N_r` iterations:

```
(n_iter, t_remain_s, v, kappa, max_depth)
        ->  softmax(p_agg, p_slow, p_feas)
```

where `p_agg` = aggressive, `p_slow` = cautious, `p_feas` = GP-prior weight.
The network is 2 hidden layers × 32 units with a softmax output.

## Model used by DA-MCTS

The DA-MCTS controller loads the MLP automatically from this directory
(`MCTS_action_prob_MLP/`) at runtime — **do not move or delete** these files:

- `model_best.pt` — trained network weights.
- `scaler.json` — input feature normalisation (mean / std).
- `best_hp.json` — architecture/hyperparameters used to rebuild the network.

## Files

| File | Purpose |
|------|---------|
| `prepare_dataset.py` | Bin-and-deduplicate the raw weight logs into `data/dataset_reduced.csv`. |
| `hp_search.py` | Optuna hyperparameter search → writes `best_hp.json`. |
| `train_mlp.py` | Train the MLP with the best HPs → writes `model_best.pt` + `scaler.json`. |
| `evaluate.py` | Evaluate a trained model. |
| `data/` | Training data: `dataset_reduced.csv` (processed set used to train) and the per-budget `mcts_weight_log_*.csv` raw logs it is built from. |

## (Re)train the MLP — optional

You do **not** need to retrain to run the experiments; the trained model above is
already included. The full pipeline:

```bash
# 1. (optional) Rebuild the reduced dataset from the raw weight logs
python MCTS_action_prob_MLP/prepare_dataset.py

# 2. Hyperparameter search (Optuna)
python MCTS_action_prob_MLP/hp_search.py --trials 100

# 3. Train the MLP (reads data/dataset_reduced.csv + best_hp.json)
python MCTS_action_prob_MLP/train_mlp.py
```

The new `model_best.pt` / `scaler.json` are written back into this directory and
picked up automatically by the controller.
